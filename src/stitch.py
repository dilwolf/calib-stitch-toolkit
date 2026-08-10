"""Feature-based panorama stitching built from OpenCV primitives.

detect -> ratio-test match -> RANSAC homography -> common canvas -> seam -> blend
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def load(paths, max_w=1200):
    imgs = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            raise SystemExit(f"cannot read {p}")
        if im.shape[1] > max_w:
            s = max_w / im.shape[1]
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        imgs.append(im)
    return imgs


def cylindrical_warp(img, f):
    """Reproject onto a cylinder of focal length f.

    A single homography maps onto a plane, and a plane cannot hold a wide field
    of view -- the outer images stretch without bound as the total angle grows.
    Cylindrical coordinates keep that bounded, at the cost of straight lines
    through the scene no longer being straight.
    """
    h, w = img.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    theta = (x - w / 2) / f
    height = (y - h / 2) / f

    X, Y, Z = np.sin(theta), height, np.cos(theta)
    mx = (f * X / Z + w / 2).astype(np.float32)
    my = (f * Y / Z + h / 2).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _solve_focal(d1, d2, v1, v2):
    """Pick between the two focal-square candidates, preferring the one whose
    denominator is better conditioned. Returns None if neither is positive."""
    if v1 < v2:
        v1, v2 = v2, v1
    if v1 > 0 and v2 > 0:
        return float(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
    if v1 > 0:
        return float(np.sqrt(v1))
    return None


def focal_from_homography(H, size):
    """Focal length estimates implied by a homography between two views.

    For a camera that only rotates, H = K R K^-1. With square pixels and the
    principal point at the image centre that leaves f as the single unknown,
    and it drops out in closed form (Szeliski; the same relations OpenCV uses
    in autocalib). Two estimates come back, one per view.

    cv2.detail.focalsFromHomography exists but writes its results through
    reference parameters, which the Python binding cannot express -- it returns
    None -- so the relations are written out here.
    """
    w, h_img = size
    to_pixels = np.array([[1.0, 0.0, w / 2], [0.0, 1.0, h_img / 2], [0.0, 0.0, 1.0]])
    h = (np.linalg.inv(to_pixels) @ H @ to_pixels).ravel()

    out = []
    eps = 1e-12

    d1, d2 = h[6] * h[7], (h[7] - h[6]) * (h[7] + h[6])
    if abs(d1) > eps and abs(d2) > eps:
        out.append(_solve_focal(d1, d2,
                                -(h[0] * h[1] + h[3] * h[4]) / d1,
                                (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2))

    d1, d2 = h[0] * h[3] + h[1] * h[4], h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
    if abs(d1) > eps and abs(d2) > eps:
        out.append(_solve_focal(d1, d2,
                                -h[2] * h[5] / d1,
                                (h[5] ** 2 - h[2] ** 2) / d2))

    return [f for f in out if f]


def estimate_focal(pairwise, size):
    """Median focal over every pairwise homography.

    Median rather than mean: a single badly conditioned pair produces a wild
    estimate and would drag an average with it.
    """
    fs = [f for H in pairwise for f in focal_from_homography(H, size)]
    if not fs:
        raise RuntimeError("no usable focal estimate from the homographies")
    return float(np.median(fs)), fs


def make_detector(name, n_features):
    if name == "sift":
        return cv2.SIFT_create(nfeatures=n_features), cv2.NORM_L2
    if name == "orb":
        return cv2.ORB_create(nfeatures=n_features), cv2.NORM_HAMMING
    raise SystemExit(f"unknown detector {name}")


def match_pair(img1, img2, detector, n_features=4000, ratio=0.75):
    det, norm = make_detector(detector, n_features)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return k1, k2, []

    pairs = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    # Lowe's ratio test. A descriptor that matches two places about equally well
    # is ambiguous, and ambiguous matches are what wreck the homography.
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    return k1, k2, good


def estimate_homography(k1, k2, matches, thresh=4.0):
    if len(matches) < 4:
        raise RuntimeError(f"{len(matches)} matches, need at least 4")
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # RANSAC, not least squares: least squares minimises over every match
    # including the wrong ones, so a handful of bad pairs drags the fit.
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, thresh, maxIters=5000)
    if H is None:
        raise RuntimeError("homography did not converge")
    return H, int(mask.sum())


def chain_to_reference(pairwise, ref):
    """Compose neighbour homographies into a common frame.

    Each hop multiplies in its own error, so the reference is the middle image
    by default: it halves the longest chain compared with anchoring on an end.
    """
    n = len(pairwise) + 1
    Hs = [None] * n
    Hs[ref] = np.eye(3)
    for i in range(ref - 1, -1, -1):
        Hs[i] = Hs[i + 1] @ np.linalg.inv(pairwise[i])
    for i in range(ref + 1, n):
        Hs[i] = Hs[i - 1] @ pairwise[i - 1]
    return Hs


def match_graph(imgs, detector="sift", min_inliers=30, quiet=False):
    """Match every pair, keeping the homographies that survive RANSAC.

    ponytail: O(n^2) matching. Fine to a couple of dozen images; past that the
    usual fix is to only test pairs whose global descriptors already look alike.
    """
    n = len(imgs)
    weights = np.zeros((n, n))
    goods = np.zeros((n, n))
    Hs = {}
    for i in range(n):
        for j in range(i + 1, n):
            k1, k2, good = match_pair(imgs[i], imgs[j], detector)
            try:
                H, inliers = estimate_homography(k1, k2, good)
            except RuntimeError:
                continue
            if inliers < min_inliers:
                continue
            weights[i, j] = weights[j, i] = inliers
            goods[i, j] = goods[j, i] = len(good)
            Hs[(i, j)] = H
            Hs[(j, i)] = np.linalg.inv(H)
    if not quiet:
        print("  pairwise inliers")
        print("      " + "".join(f"{j:>6}" for j in range(n)))
        for i in range(n):
            cells = "".join("     -" if i == j else f"{int(weights[i, j]):>6}"
                            for j in range(n))
            print(f"  {i:>3} {cells}")
    return weights, goods, Hs


def spanning_order(weights, ref):
    """Prim's algorithm on the inlier graph, heaviest edge first.

    Chaining in file order assumes the inputs are an ordered sweep. When they
    are not, the chain crosses a pair that barely matches and every image after
    it is placed by a homography fitted to noise.
    """
    n = len(weights)
    seen = {ref}
    order = []
    while len(seen) < n:
        best, best_w = None, 0
        for u in seen:
            for v in range(n):
                if v in seen:
                    continue
                if weights[u, v] > best_w:
                    best, best_w = (v, u), weights[u, v]
        if best is None:
            break  # remaining images share no usable overlap
        order.append(best)
        seen.add(best[0])
    return order, sorted(set(range(n)) - seen)


def chain_via_graph(Hs, order, ref, n):
    out = [None] * n
    out[ref] = np.eye(3)
    for node, parent in order:
        out[node] = out[parent] @ Hs[(parent, node)]
    return out


def warp_all(imgs, Hs):
    corners = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, H))
    allc = np.concatenate(corners)

    xmin, ymin = np.int32(allc.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(allc.max(axis=0).ravel() + 0.5)
    shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
    size = (int(xmax - xmin), int(ymax - ymin))

    warped, masks = [], []
    for im, H in zip(imgs, Hs):
        warped.append(cv2.warpPerspective(im, shift @ H, size))
        solid = np.full(im.shape[:2], 255, np.uint8)
        masks.append(cv2.warpPerspective(solid, shift @ H, size))
    return warped, masks, size


def find_seams(warped, masks):
    """Cut each overlap along a low-contrast path instead of a straight edge."""
    corners = [(0, 0)] * len(warped)
    src = [im.astype(np.float32) for im in warped]
    seam_masks = [m.copy() for m in masks]
    finder = cv2.detail_DpSeamFinder("COLOR")
    seam_masks = finder.find(src, corners, seam_masks)
    out = []
    for sm, m in zip(seam_masks, masks):
        sm = sm.get() if hasattr(sm, "get") else sm
        # the seam finder returns labels only inside the overlap; outside it the
        # original mask still decides what is image and what is empty canvas
        out.append(cv2.bitwise_and(sm, m))
    return out


def blend_multiband(warped, masks, bands=5):
    """Blend low frequencies over a wide band and high frequencies over a narrow
    one, so exposure differences disappear without smearing detail."""
    h, w = warped[0].shape[:2]
    blender = cv2.detail_MultiBandBlender(try_gpu=0, num_bands=bands)
    blender.prepare((0, 0, w, h))
    for im, m in zip(warped, masks):
        blender.feed(im.astype(np.int16), m, (0, 0))
    res, _ = blender.blend(None, None)
    return np.clip(res, 0, 255).astype(np.uint8)


def blend_average(warped, masks):
    """Naive mean over the overlap, kept for the comparison in the README."""
    acc = np.zeros(warped[0].shape, np.float32)
    count = np.zeros(warped[0].shape[:2], np.float32)
    for im, m in zip(warped, masks):
        on = m > 0
        acc += im.astype(np.float32) * on[..., None]
        count += on
    return (acc / np.maximum(count, 1)[..., None]).astype(np.uint8)


def pairwise_homographies(imgs, detector="sift", quiet=False):
    pairwise, stats = [], []
    for i in range(len(imgs) - 1):
        t0 = time.perf_counter()
        k1, k2, good = match_pair(imgs[i], imgs[i + 1], detector)
        t_match = time.perf_counter() - t0

        t0 = time.perf_counter()
        H, inliers = estimate_homography(k1, k2, good)
        t_h = time.perf_counter() - t0

        pairwise.append(H)
        stats.append({"pair": f"{i}->{i+1}", "kp1": len(k1), "kp2": len(k2),
                      "good": len(good), "inliers": inliers,
                      "inlier_pct": 100 * inliers / len(good),
                      "match_s": t_match, "homography_s": t_h})
        if not quiet:
            s = stats[-1]
            print(f"  {s['pair']}  kp {s['kp1']:5d}/{s['kp2']:5d}  "
                  f"good {s['good']:5d}  inliers {s['inliers']:5d} "
                  f"({s['inlier_pct']:5.1f}%)  {t_match:.2f}s")
    return pairwise, stats


def stitch(imgs, detector="sift", ref=None, bands=5, seams=True, quiet=False,
           graph=False):
    ref = len(imgs) // 2 if ref is None else ref

    if graph:
        weights, goods, pair_H = match_graph(imgs, detector, quiet=quiet)
        order, orphans = spanning_order(weights, ref)
        if orphans and not quiet:
            print(f"  no usable overlap, dropped: {orphans}")
        all_Hs = chain_via_graph(pair_H, order, ref, len(imgs))
        keep = [i for i, H in enumerate(all_Hs) if H is not None]
        imgs = [imgs[i] for i in keep]
        Hs = [all_Hs[i] for i in keep]
        stats = [{"pair": f"{p}->{n}", "kp1": 0, "kp2": 0,
                  "good": int(goods[p, n]), "inliers": int(weights[p, n]),
                  "inlier_pct": 100 * weights[p, n] / max(goods[p, n], 1),
                  "match_s": 0.0, "homography_s": 0.0}
                 for n, p in order]
        if not quiet:
            print("  spanning tree: " + "  ".join(
                f"{p}->{n}({int(weights[p, n])})" for n, p in order))
    else:
        pairwise, stats = pairwise_homographies(imgs, detector, quiet)
        Hs = chain_to_reference(pairwise, ref)

    warped, masks, size = warp_all(imgs, Hs)
    blend_masks = find_seams(warped, masks) if seams else masks
    return warped, masks, blend_masks, size, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/pano")
    ap.add_argument("--detector", default="sift", choices=["sift", "orb"])
    ap.add_argument("--max-width", type=int, default=1200)
    ap.add_argument("--ref", type=int, default=None,
                    help="index of the reference image (default: middle)")
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--no-seams", action="store_true")
    ap.add_argument("--graph", action="store_true",
                    help="match all pairs and order them by a maximum spanning "
                         "tree, instead of assuming the files are a sweep")
    ap.add_argument("--cylindrical", metavar="F", default=None,
                    help="pre-warp onto a cylinder; F in px, or 'auto' to "
                         "estimate it from the pairwise homographies")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) < 2:
        raise SystemExit(f"need at least 2 images in {args.images}")

    imgs = load(paths, args.max_width)
    print(f"{len(imgs)} images at {imgs[0].shape[1]}x{imgs[0].shape[0]}, "
          f"detector={args.detector}")
    tag = f"{Path(args.images).name}_{args.detector}"
    if args.cylindrical:
        img_size = (imgs[0].shape[1], imgs[0].shape[0])
        if args.cylindrical == "auto":
            # the focal has to come from somewhere before we can warp, so match
            # once on the flat images purely to estimate it
            print("estimating focal length from pairwise homographies")
            pre, _ = pairwise_homographies(imgs, args.detector, quiet=True)
            f, all_f = estimate_focal(pre, img_size)
            fov = np.degrees(2 * np.arctan(img_size[0] / (2 * f)))
            print(f"  {len(all_f)} estimates, median f = {f:.1f} px  "
                  f"(spread {min(all_f):.0f}-{max(all_f):.0f}), "
                  f"horizontal FOV {fov:.1f} deg")
        else:
            f = float(args.cylindrical)
        imgs = [cylindrical_warp(im, f) for im in imgs]
        tag += "_cyl"

    t0 = time.perf_counter()
    warped, masks, blend_masks, size, stats = stitch(
        imgs, args.detector, args.ref, args.bands, not args.no_seams,
        graph=args.graph)
    total = time.perf_counter() - t0
    if args.graph:
        tag += "_graph"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(outdir / f"pano_{tag}_average.jpg"),
                blend_average(warped, masks))
    cv2.imwrite(str(outdir / f"pano_{tag}_multiband.jpg"),
                blend_multiband(warped, blend_masks, args.bands))

    mean_inlier = np.mean([s["inlier_pct"] for s in stats])
    print(f"\ncanvas {size[0]}x{size[1]}  mean inliers {mean_inlier:.1f}%  "
          f"{total:.2f}s total")
    print(f"wrote {outdir}/pano_{tag}_*.jpg")


if __name__ == "__main__":
    main()
