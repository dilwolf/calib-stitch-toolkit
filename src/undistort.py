"""Apply a calibration: build remap tables once, then remap per frame.

Also measures what that split costs, and how straight the straight lines
actually get, so the correction is a number rather than a claim.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from calibrate import FIND_FLAGS, SUBPIX_TERM


def row_straightness(img, pattern):
    """RMS deviation of each board row from its own best-fit line, in px.

    A pinhole camera images a straight line as a straight line; whatever is
    left here is lens distortion the model has not removed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_TERM)
    pts = corners.reshape(pattern[1], pattern[0], 2)

    devs = []
    for row in pts:
        centred = row - row.mean(axis=0)
        # direction of the row is the leading singular vector; the residual
        # against it is the perpendicular deviation
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        normal = vt[1]
        devs.append(centred @ normal)
    return float(np.sqrt(np.mean(np.concatenate(devs) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", default="outputs/calib_pinhole.npz")
    ap.add_argument("--image", default="data/calib/left01.jpg")
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="0 crops every invalid pixel, 1 keeps the full frame")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="outputs/undistort_before_after.png")
    args = ap.parse_args()

    pattern = tuple(int(v) for v in args.pattern.split("x"))
    d = np.load(args.calib)
    K, dist, size = d["K"], d["dist"], tuple(int(v) for v in d["size"])

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")

    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, args.alpha)

    t0 = time.perf_counter()
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, newK, size, cv2.CV_16SC2)
    build_ms = (time.perf_counter() - t0) * 1e3

    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    t0 = time.perf_counter()
    for _ in range(args.reps):
        cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    remap_ms = (time.perf_counter() - t0) * 1e3 / args.reps

    print(f"{size[0]}x{size[1]}  alpha={args.alpha}")
    print(f"map build   {build_ms:8.2f} ms   once")
    print(f"remap       {remap_ms:8.3f} ms   per frame  ({1e3/remap_ms:.0f} fps, 1 core)")

    before = row_straightness(img, pattern)
    after = row_straightness(und, pattern)
    if before is None or after is None:
        print("board not found in one of the images, skipping straightness")
    else:
        print(f"\nrow straightness (RMS deviation from a fitted line)")
        print(f"  distorted   {before:.3f} px")
        print(f"  undistorted {after:.3f} px   ({100 * (1 - after / before):.0f}% lower)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([img, und]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
