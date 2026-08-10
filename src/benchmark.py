"""Compare detectors and blenders against OpenCV's own stitcher.

Runtime is easy to measure; seam quality is not, so it gets an explicit metric
rather than an adjective. See seam_visibility.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from stitch import (blend_average, blend_multiband, cylindrical_warp,
                    estimate_focal, load, match_pair, pairwise_homographies,
                    stitch)


def seam_visibility(blended, blend_masks, overlap, band=4):
    """Gradient energy on the seam relative to the rest of the panorama.

    A hard cut leaves an edge that is not in the scene, so gradient magnitude
    spikes along it. Normalising by the whole image keeps the number
    comparable across scenes, which are not equally textured. 1.0 means the
    seam is indistinguishable from ordinary scene detail.
    """
    gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))

    kernel = np.ones((3, 3), np.uint8)
    seam_band = np.zeros(gray.shape, np.uint8)
    for m in blend_masks:
        edge = cv2.dilate(m, kernel, iterations=band) - cv2.erode(m, kernel, iterations=band)
        seam_band |= edge
    # only the interior counts; the outer border of the canvas is not a seam
    seam_band = (seam_band > 0) & overlap

    valid = np.zeros(gray.shape, bool)
    for m in blend_masks:
        valid |= m > 0
    if seam_band.sum() < 50:
        return None
    return float(grad[seam_band].mean() / max(grad[valid].mean(), 1e-6))


def time_matching(imgs, detector):
    """Detect and match one representative pair, so the detector cost is
    isolated from everything downstream of it."""
    t0 = time.perf_counter()
    match_pair(imgs[0], imgs[1], detector)
    return time.perf_counter() - t0


def run_detector(imgs, detector, bands, graph=False):
    t_match = time_matching(imgs, detector)

    t0 = time.perf_counter()
    warped, masks, blend_masks, size, stats = stitch(
        imgs, detector, None, bands, seams=True, quiet=True, graph=graph)
    t_align = time.perf_counter() - t0

    counts = np.zeros(warped[0].shape[:2], np.uint8)
    for m in masks:
        counts += (m > 0).astype(np.uint8)
    overlap = counts >= 2

    t0 = time.perf_counter()
    mb = blend_multiband(warped, blend_masks, bands)
    t_mb = time.perf_counter() - t0
    avg = blend_average(warped, masks)

    return {
        "detector": detector,
        "good_matches": int(np.mean([s["good"] for s in stats])),
        "inliers": int(np.mean([s["inliers"] for s in stats])),
        "inlier_pct": round(float(np.mean([s["inlier_pct"] for s in stats])), 1),
        "match_pair_s": round(t_match, 3),
        "align_s": round(t_align, 2),
        "blend_s": round(t_mb, 2),
        "total_s": round(t_align + t_mb, 2),
        "canvas": list(size),
        "seam_multiband": seam_visibility(mb, blend_masks, overlap),
        "seam_average": seam_visibility(avg, masks, overlap),
    }, mb, avg


def run_opencv_stitcher(paths, max_w):
    imgs = load(paths, max_w)
    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    t0 = time.perf_counter()
    status, pano = st.stitch(imgs)
    dt = time.perf_counter() - t0
    if status != cv2.Stitcher_OK:
        return {"detector": "cv2.Stitcher", "total_s": round(dt, 2),
                "status": int(status), "canvas": None}, None
    return {"detector": "cv2.Stitcher", "total_s": round(dt, 2), "status": 0,
            "canvas": [pano.shape[1], pano.shape[0]]}, pano


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/pano")
    ap.add_argument("--max-width", type=int, default=1200)
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--cylindrical", action="store_true",
                    help="pre-warp with a focal estimated from the homographies")
    ap.add_argument("--graph", action="store_true",
                    help="order images by a maximum spanning tree over all pairs")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = load(paths, args.max_width)
    scene = Path(args.images).name
    print(f"{scene}: {len(imgs)} images at {imgs[0].shape[1]}x{imgs[0].shape[0]}")

    focal = None
    if args.cylindrical:
        size = (imgs[0].shape[1], imgs[0].shape[0])
        pre, _ = pairwise_homographies(imgs, "sift", quiet=True)
        focal, _ = estimate_focal(pre, size)
        imgs = [cylindrical_warp(im, focal) for im in imgs]
        print(f"cylindrical pre-warp at f={focal:.1f} px")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for detector in ("sift", "orb"):
        row, mb, avg = run_detector(imgs, detector, args.bands, args.graph)
        rows.append(row)
        suffix = ("_cyl" if args.cylindrical else "") + ("_graph" if args.graph else "")
        cv2.imwrite(str(outdir / f"pano_{detector}{suffix}_multiband.jpg"), mb)
        cv2.imwrite(str(outdir / f"pano_{detector}{suffix}_average.jpg"), avg)

    base, pano = run_opencv_stitcher(paths, args.max_width)
    rows.append(base)
    if pano is not None:
        cv2.imwrite(str(outdir / "pano_opencv_stitcher.jpg"), pano)

    hdr = f"{'method':<14} {'good':>6} {'inl':>6} {'inl%':>6} " \
          f"{'match/pair':>11} {'align':>7} {'blend':>7} {'total':>7}  canvas"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["detector"] == "cv2.Stitcher":
            canvas = "x".join(map(str, r["canvas"])) if r["canvas"] else f"FAILED({r['status']})"
            print(f"{r['detector']:<14} {'-':>6} {'-':>6} {'-':>6} "
                  f"{'-':>11} {'-':>7} {'-':>7} {r['total_s']:>7.2f}  {canvas}")
        else:
            print(f"{r['detector']:<14} {r['good_matches']:>6} "
                  f"{r['inliers']:>6} {r['inlier_pct']:>6.1f} {r['match_pair_s']:>11.3f} "
                  f"{r['align_s']:>7.2f} {r['blend_s']:>7.2f} {r['total_s']:>7.2f}  "
                  f"{r['canvas'][0]}x{r['canvas'][1]}")

    print(f"\nseam visibility (gradient on the seam / gradient overall, 1.0 = invisible)")
    for r in rows:
        if r["detector"] == "cv2.Stitcher":
            continue
        print(f"  {r['detector']:<6} multiband {r['seam_multiband']:.3f}   "
              f"average {r['seam_average']:.3f}")

    out = outdir / f"benchmark_{scene}.json"
    with open(out, "w") as f:
        json.dump({"scene": scene, "images": len(imgs),
                   "size": [imgs[0].shape[1], imgs[0].shape[0]],
                   "cylindrical_focal_px": focal, "results": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
