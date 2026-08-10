"""Pinhole (Brown-Conrady) calibration from chessboard views."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SUBPIX_TERM = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def board_points(pattern, square):
    """Board corners in board frame. Planar, so Z=0 for every point."""
    n_cols, n_rows = pattern
    pts = np.zeros((n_cols * n_rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)
    return pts * square


def detect(paths, pattern):
    detections, missed = [], []
    size = None
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            missed.append(p.name)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[1], gray.shape[0]
        ok, corners = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS)
        if not ok:
            missed.append(p.name)
            continue
        # findChessboardCorners is integer-accurate; the fit needs sub-pixel
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_TERM)
        detections.append((p.name, corners))
    return detections, missed, size


def per_view_rms(objp, corners, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    d = corners.reshape(-1, 2) - proj.reshape(-1, 2)
    # RMS is sqrt(mean of squared point distances). Dividing the residual norm
    # by N instead of sqrt(N) is the common slip and will not pool back to
    # calibrateCamera's reported RMS.
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


def calibrate(detections, pattern, square, size):
    objp = board_points(pattern, square)
    objpoints = [objp] * len(detections)
    imgpoints = [c for _, c in detections]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None)

    errors = [per_view_rms(objp, c, rvecs[i], tvecs[i], K, dist)
              for i, (_, c) in enumerate(detections)]
    return rms, K, dist, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/calib")
    ap.add_argument("--pattern", default="9x6", help="inner corners, colsxrows")
    ap.add_argument("--square", type=float, default=1.0,
                    help="square size in mm; only scales tvecs, not K or dist")
    ap.add_argument("--max-err", type=float, default=None,
                    help="drop views above this reprojection RMS and refit")
    ap.add_argument("--out", default="outputs/calib_pinhole.npz")
    args = ap.parse_args()

    pattern = tuple(int(v) for v in args.pattern.split("x"))
    paths = sorted(Path(args.images).glob("*.jpg")) + \
            sorted(Path(args.images).glob("*.png"))
    if not paths:
        raise SystemExit(f"no images in {args.images}")

    detections, missed, size = detect(paths, pattern)
    print(f"{len(detections)}/{len(paths)} views with a {args.pattern} board  {size[0]}x{size[1]}")
    for name in missed:
        print(f"  no board: {name}")
    if len(detections) < 5:
        raise SystemExit("too few views to calibrate")

    rms, K, dist, errors = calibrate(detections, pattern, args.square, size)
    print(f"\nRMS {rms:.4f} px over {len(detections)} views")

    if args.max_err is not None:
        keep = [d for d, e in zip(detections, errors) if e <= args.max_err]
        dropped = [(d[0], e) for d, e in zip(detections, errors) if e > args.max_err]
        for name, e in dropped:
            print(f"  drop {name}  {e:.3f} px")
        if dropped:
            detections = keep
            rms, K, dist, errors = calibrate(detections, pattern, args.square, size)
            print(f"RMS {rms:.4f} px over {len(detections)} views after refit")

    print("\nper-view RMS")
    for (name, _), e in zip(detections, errors):
        print(f"  {name}  {e:.4f}")

    # per-view errors must pool back to the global RMS, or the fit is being
    # read wrong somewhere
    pooled = np.sqrt(np.mean(np.square(errors)))
    assert abs(pooled - rms) < 1e-6, f"pooled {pooled:.6f} != reported {rms:.6f}"

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"\nfx {fx:.2f}  fy {fy:.2f}")
    print(f"cx {cx:.2f}  cy {cy:.2f}   (image centre {size[0]/2:.1f}, {size[1]/2:.1f})")
    print("k1 k2 p1 p2 k3 =", np.array2string(dist.ravel(), precision=5))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K=K, dist=dist, rms=rms, size=size)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({
            "views_used": len(detections),
            "views_total": len(paths),
            "rms_px": round(float(rms), 4),
            "image_size": list(size),
            "fx": round(float(fx), 3), "fy": round(float(fy), 3),
            "cx": round(float(cx), 3), "cy": round(float(cy), 3),
            "dist": [round(float(v), 6) for v in dist.ravel()],
            "per_view_rms": {n: round(float(e), 4)
                             for (n, _), e in zip(detections, errors)},
        }, f, indent=2)
    print(f"\nwrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
