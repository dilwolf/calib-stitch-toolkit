import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stitch import (chain_to_reference, estimate_homography,
                    focal_from_homography, spanning_order)


def test_ransac_recovers_transform_despite_outliers():
    H_true = np.array([[1.02, 0.03, 12.0],
                       [-0.02, 0.99, -7.0],
                       [1e-5, 2e-5, 1.0]])
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 500, (60, 1, 2)).astype(np.float32)
    dst = cv2.perspectiveTransform(src, H_true)
    dst[:8] += rng.uniform(80, 160, (8, 1, 2))  # 8 of 60 matches are wrong

    class M:
        def __init__(self, i):
            self.queryIdx = self.trainIdx = i

    k_src = [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 1) for p in dst]
    k_dst = [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 1) for p in src]
    H, inliers = estimate_homography(k_src, k_dst, [M(i) for i in range(60)])

    H = H / H[2, 2]
    assert np.allclose(H, H_true / H_true[2, 2], atol=0.05)
    assert inliers >= 45  # the injected outliers are rejected, not fitted


def test_estimate_homography_rejects_too_few_matches():
    with pytest.raises(RuntimeError):
        estimate_homography([], [], [])


@pytest.mark.parametrize("f_true", [400.0, 1051.0, 2500.0])
def test_focal_recovered_from_rotational_homography(f_true):
    """A camera that only rotates gives H = K R K^-1, so f is recoverable."""
    size = (1200, 1600)
    K = np.array([[f_true, 0, size[0] / 2],
                  [0, f_true, size[1] / 2],
                  [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(np.array([0.02, 0.35, -0.01]))
    H = K @ R @ np.linalg.inv(K)

    estimates = focal_from_homography(H, size)
    assert estimates, "no focal estimate produced"
    assert np.median(estimates) == pytest.approx(f_true, rel=0.01)


def test_pure_translation_yields_no_focal():
    """A translation carries no perspective, so f is not observable from it."""
    H = np.array([[1.0, 0, 40.0], [0, 1.0, -13.0], [0, 0, 1.0]])
    assert focal_from_homography(H, (1200, 1600)) == []


def test_chain_to_reference_composes_and_anchors():
    rng = np.random.default_rng(1)
    pairwise = [np.eye(3) + rng.normal(0, 0.01, (3, 3)) for _ in range(3)]
    Hs = chain_to_reference(pairwise, ref=1)

    assert np.allclose(Hs[1], np.eye(3))
    assert np.allclose(Hs[2], pairwise[1])
    assert np.allclose(Hs[3], pairwise[1] @ pairwise[2])
    assert np.allclose(Hs[0], np.linalg.inv(pairwise[0]))


def test_spanning_order_routes_around_a_weak_pair():
    """The budapest failure in miniature: 2 and 3 are adjacent by filename but
    share almost nothing, so ordering by file index walks through a homography
    fitted to five points."""
    w = np.zeros((4, 4))
    for (i, j), n in {(0, 1): 470, (1, 2): 889, (2, 3): 4, (0, 3): 976}.items():
        w[i, j] = w[j, i] = n

    order, orphans = spanning_order(w, ref=0)
    assert orphans == []
    parents = {node: parent for node, parent in order}
    assert parents[3] == 0        # via the strong 0-3 edge, not through 2
    assert (2, 3) not in order and (3, 2) not in order


def test_spanning_order_reports_disconnected_images():
    w = np.zeros((3, 3))
    w[0, 1] = w[1, 0] = 500       # image 2 overlaps nothing
    order, orphans = spanning_order(w, ref=0)
    assert orphans == [2]
    assert len(order) == 1
