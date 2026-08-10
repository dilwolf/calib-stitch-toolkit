"""Recover known Kannala-Brandt fisheye distortion from synthetic observations.

No fisheye lens was available, so ground truth is generated instead. That is a
weaker claim about hardware but a stronger one about the estimator: the true
parameters are known exactly, so recovery error can be measured rather than
eyeballed.

Pinhole cannot stand in here. It projects r = f*tan(theta), which runs to
infinity as theta approaches 90 deg, so it cannot represent a hemispherical
lens at all. Kannala-Brandt uses r = f*theta and stays finite, with four radial
terms k1..k4 and no tangential ones.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

K_TRUE = np.array([[420.0, 0.0, 634.0],
                   [0.0, 420.0, 364.0],
                   [0.0, 0.0, 1.0]])
D_TRUE = np.array([[-0.032], [0.0041], [-0.0009], [0.00012]])

PATTERN = (9, 6)
SQUARE = 25.0
SIZE = (1280, 720)

# OpenCV 5 moved the fisheye calibration flags out of cv2.fisheye to the top level
_FLAG_NS = cv2 if hasattr(cv2, "CALIB_RECOMPUTE_EXTRINSIC") else cv2.fisheye
FISHEYE_FLAGS = _FLAG_NS.CALIB_RECOMPUTE_EXTRINSIC | _FLAG_NS.CALIB_FIX_SKEW


def board_points():
    pts = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float64)
    pts[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    pts *= SQUARE
    return pts - pts.mean(axis=0)


def incidence_deg(objp, rvec, tvec):
    """Angle of each board point off the optical axis, in degrees.

    This is the quantity the fisheye model is about. If the boards only ever
    sit near the axis the higher radial terms are unconstrained and any
    'recovery' of k3/k4 is meaningless.
    """
    R, _ = cv2.Rodrigues(rvec)
    cam = (R @ objp.T + tvec).T
    return np.degrees(np.arctan2(np.linalg.norm(cam[:, :2], axis=1), cam[:, 2]))


def make_views(n_target, rng, min_hull_area=2000.0):
    """Noise-free projections. Noise is added later so every run in the sweep
    sees the same geometry and only the noise changes."""
    objp = board_points()
    objpoints, clean, angles = [], [], []
    attempts = 0

    while len(objpoints) < n_target and attempts < n_target * 60:
        attempts += 1
        rvec = rng.uniform(-0.7, 0.7, 3).reshape(3, 1)
        # pushed well off-axis on purpose, to drive the board out to the
        # periphery where the radial terms actually bite
        tvec = np.array([[rng.uniform(-620, 620)],
                         [rng.uniform(-380, 380)],
                         [rng.uniform(330, 750)]])

        proj, _ = cv2.fisheye.projectPoints(
            objp.reshape(-1, 1, 3), rvec, tvec, K_TRUE, D_TRUE)
        proj = proj.reshape(-1, 2)

        if proj[:, 0].min() < 0 or proj[:, 0].max() > SIZE[0]:
            continue
        if proj[:, 1].min() < 0 or proj[:, 1].max() > SIZE[1]:
            continue

        # a near edge-on board carries almost no information and collapses the
        # homography that seeds the extrinsics (InitExtrinsics asserts on it)
        if cv2.contourArea(cv2.convexHull(proj.astype(np.float32))) < min_hull_area:
            continue

        angles.append(incidence_deg(objp, rvec, tvec))
        # fisheye.calibrate wants 1xNx3 / 1xNx2 per view; Nx1x3 raises a
        # size-mismatch from deep inside arithm_op
        objpoints.append(objp.reshape(1, -1, 3))
        clean.append(proj)

    return objpoints, clean, np.concatenate(angles) if angles else None


def recover(objpoints, imgpoints, seed=True):
    """Fit K and D back out of the observations.

    seed=False reproduces OpenCV's own initialisation, which does not survive
    realistic corner noise -- see the --cold-start note in the README.
    """
    flags = FISHEYE_FLAGS
    if seed:
        # f = max(w,h)/pi is the focal length a hemispherical Kannala-Brandt
        # lens would have, and is what OpenCV documents as its own default
        f = max(SIZE) / np.pi
        K = np.array([[f, 0.0, SIZE[0] / 2],
                      [0.0, f, SIZE[1] / 2],
                      [0.0, 0.0, 1.0]])
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    else:
        K = np.zeros((3, 3))

    rms, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, SIZE, K, np.zeros((4, 1)), flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-10))
    return rms, K, D


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", type=int, default=40)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5],
                    help="corner localisation noise sigma in px, one run each")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold-start", action="store_true",
                    help="skip the intrinsic guess and let OpenCV initialise itself")
    ap.add_argument("--out", default="outputs/fisheye_recovery.json")
    args = ap.parse_args()

    objpoints, clean, angles = make_views(
        args.views, np.random.default_rng(args.seed))
    print(f"{len(objpoints)} views, {len(objpoints) * PATTERN[0] * PATTERN[1]} points")
    print(f"incidence angle off the optical axis: median {np.median(angles):.1f} deg, "
          f"p95 {np.percentile(angles, 95):.1f} deg, max {angles.max():.1f} deg\n")

    rows = []
    for noise in args.noise:
        nrng = np.random.default_rng(args.seed + 1)
        imgpoints = [(p + nrng.normal(0, noise, p.shape)).reshape(1, -1, 2)
                     for p in clean]
        try:
            rms, K, D = recover(objpoints, imgpoints, seed=not args.cold_start)
        except cv2.error as e:
            rows.append({"noise_px": noise, "failed": e.err.strip().splitlines()[-1]})
            continue

        rows.append({
            "noise_px": noise,
            "rms_px": round(float(rms), 5),
            "fx": round(float(K[0, 0]), 3),
            "cx": round(float(K[0, 2]), 3),
            "cy": round(float(K[1, 2]), 3),
            "d": [float(v) for v in D.ravel()],
        })

    print(f"{'noise':>6} {'rms':>9} {'rms/noise':>10} {'fx err':>9} "
          f"{'cx err':>9} {'cy err':>9} {'max |dk|':>10}")
    for r in rows:
        if "failed" in r:
            print(f"{r['noise_px']:>6.2f}   did not converge: {r['failed']}")
            continue
        dk = max(abs(a - b) for a, b in zip(r["d"], D_TRUE.ravel()))
        ratio = "-" if r["noise_px"] == 0 else f"{r['rms_px'] / r['noise_px']:.3f}"
        print(f"{r['noise_px']:>6.2f} {r['rms_px']:>9.5f} {ratio:>10} "
              f"{abs(r['fx'] - K_TRUE[0, 0]):>9.4f} "
              f"{abs(r['cx'] - K_TRUE[0, 2]):>9.4f} "
              f"{abs(r['cy'] - K_TRUE[1, 2]):>9.4f} {dk:>10.2e}")

    converged = [r for r in rows if "failed" not in r]
    if converged:
        base = converged[-1]
        print(f"\ncoefficients at noise={base['noise_px']} px")
        print(f"{'':>4} {'true':>12} {'estimated':>12} {'abs err':>11}")
        for i, (t, e) in enumerate(zip(D_TRUE.ravel(), base["d"]), start=1):
            print(f"  k{i} {t:>12.6f} {e:>12.6f} {abs(t - e):>11.2e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "truth": {"fx": K_TRUE[0, 0], "cx": K_TRUE[0, 2], "cy": K_TRUE[1, 2],
                      "d": list(D_TRUE.ravel())},
            "views": len(objpoints),
            "incidence_deg": {"median": round(float(np.median(angles)), 2),
                              "p95": round(float(np.percentile(angles, 95)), 2),
                              "max": round(float(angles.max()), 2)},
            "runs": rows,
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
