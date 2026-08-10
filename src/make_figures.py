"""Build the figures the README embeds, from whatever is already in outputs/."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def fit_width(img, width):
    s = width / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stack_same_width(images, width):
    resized = [fit_width(im, width) for im in images]
    return np.vstack(resized)


def worst_seam_crop(a, b, win=(520, 380)):
    """Crop where the two blends disagree most -- that is where the choice of
    blender actually shows."""
    diff = cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_BGR2GRAY).astype(np.float32)
    w, h = win
    score = cv2.boxFilter(diff, -1, (w, h), normalize=True)
    _, _, _, (cx, cy) = cv2.minMaxLoc(score)
    x = int(np.clip(cx - w // 2, 0, a.shape[1] - w))
    y = int(np.clip(cy - h // 2, 0, a.shape[0] - h))
    return a[y:y + h, x:x + w], b[y:y + h, x:x + w]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--dest", default="docs/img")
    ap.add_argument("--width", type=int, default=1100)
    args = ap.parse_args()

    src, dest = Path(args.outputs), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    def read(name):
        img = cv2.imread(str(src / name))
        if img is None:
            raise SystemExit(f"missing {src / name}; run the pipeline first")
        return img

    cv2.imwrite(str(dest / "undistort.jpg"),
                fit_width(read("undistort_alpha1.png"), args.width))

    planar = label(read("pano_pano_sift_graph_multiband.jpg"), "planar homography")
    cyl = label(read("pano_pano_sift_cyl_graph_multiband.jpg"),
                "cylindrical pre-warp, f estimated from the homographies")
    cv2.imwrite(str(dest / "projection.jpg"),
                stack_same_width([planar, cyl], args.width))

    cv2.imwrite(str(dest / "panorama.jpg"),
                fit_width(read("pano_pano_sift_cyl_graph_multiband.jpg"), args.width))
    cv2.imwrite(str(dest / "panorama_budapest.jpg"),
                fit_width(read("pano_pano_budapest_sift_graph_multiband.jpg"), args.width))

    avg = read("pano_pano_sift_cyl_graph_average.jpg")
    mb = read("pano_pano_sift_cyl_graph_multiband.jpg")
    ca, cb = worst_seam_crop(avg, mb)
    cv2.imwrite(str(dest / "blending.jpg"),
                np.hstack([label(ca, "average"), label(cb, "multi-band")]))

    for p in sorted(dest.glob("*.jpg")):
        im = cv2.imread(str(p))
        print(f"{p}  {im.shape[1]}x{im.shape[0]}")


if __name__ == "__main__":
    main()
