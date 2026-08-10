"""Download the public image sets used by the calibration and stitching demos."""
import argparse
import urllib.request
from pathlib import Path

OPENCV = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data"
EXTRA = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching"

# left10 is absent upstream; the gap in the numbering is theirs, not ours.
SETS = {
    "data/calib": [f"{OPENCV}/left{i:02d}.jpg"
                   for i in list(range(1, 10)) + list(range(11, 15))],
    "data/pano_budapest": [f"{EXTRA}/budapest{i}.jpg" for i in range(1, 7)],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    for folder, urls in SETS.items():
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        for url in urls:
            dest = out / url.rsplit("/", 1)[-1]
            if dest.exists() and not args.force:
                continue
            urllib.request.urlretrieve(url, dest)
            print("fetched", dest)
        print(f"{folder}: {len(list(out.glob('*.jpg')))} images")


if __name__ == "__main__":
    main()
