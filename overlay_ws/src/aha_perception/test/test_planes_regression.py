"""Regression guards for the optimised plane detection code.

The node detector on a fixed noisy scene must keep its exact output, and the
fast helpers must match their straightforward versions. A change that alters
the expected values is a behaviour change, not an optimisation.
"""

import numpy as np
import pytest
from scipy.ndimage import binary_closing
from scipy.ndimage import label as connected_label
from scipy.spatial import ConvexHull

from aha_perception.planes import (
    DETECTORS,
    DetectionOptions,
    detect_planes,
    params_for_input_stride,
)
from aha_perception.planes.geometry import convex_hull_2d
from aha_perception.planes.normals import smallest_eigh3
from aha_perception.planes.preprocess import Frame
from aha_perception.planes.refine import EIGHT, fill_holes, keep_seeded_components

# (label, inlier count) of ransac_gravity_refined with merge, per input stride.
EXPECTED = {
    1: [
        ("floor", 158275),
        ("wall", 95794),
        ("table", 43240),
        ("table", 5830),
        ("other", 3612),
    ],
    2: [("floor", 39397), ("wall", 25002), ("table", 10904), ("table", 1486)],
}
EXPECTED_HEIGHTS = {1: [0.0, 0.646, 0.72, 0.4, 1.06], 2: [0.0, 0.665, 0.72, 0.4]}


@pytest.mark.parametrize("stride", [1, 2])
def test_node_detector_output_is_unchanged(render_cloud, camera_pose, stride):
    scene = render_cloud(
        horizontal=[
            (0.0, -5, 5, -5, 5),
            (0.72, 1.0, 1.8, -0.6, 0.6),
            (0.4, 0.9, 1.3, 0.8, 1.4),
        ],
        vertical=[(2.6, -3, 3, 0.0, 2.5)],
        noise_scale=1.0,
        seed=7,
    )
    options = DetectionOptions(camera_to_base=camera_pose())
    params = DETECTORS["ransac_gravity_refined"].params_type(merge=True)
    cloud = scene[::stride, ::stride]
    planes = detect_planes(cloud, "ransac_gravity_refined", options, params)
    assert [(p.label.value, len(p.inliers)) for p in planes] == EXPECTED[stride]
    heights = [float(p.centroid[2]) for p in planes]
    assert heights == pytest.approx(EXPECTED_HEIGHTS[stride], abs=1e-3)


def test_params_for_input_stride():
    params = DETECTORS["ransac_gravity_refined"].params_type(merge=True)
    assert params_for_input_stride(params, 1) is params
    scaled = params_for_input_stride(params, 2)
    assert scaled.stride == max(1, params.stride // 2)
    assert scaled.merge and scaled.min_component_pixels == params.min_component_pixels
    assert params_for_input_stride(params, 4).stride == 1
    cape = DETECTORS["cape"].params_type()
    assert params_for_input_stride(cape, 2) is cape  # no working stride


def test_convex_hull_prefilter_matches_qhull():
    rng = np.random.default_rng(1)
    for trial in range(30):
        n = int(rng.integers(64, 3000))
        points = rng.normal(size=(n, 2)) * rng.uniform(0.1, 2.0, 2)
        if trial % 3 == 0:
            points = np.round(points, 2)  # duplicates and collinear points
        hull, area = convex_hull_2d(points)
        reference = ConvexHull(points)
        assert np.array_equal(hull, points[reference.vertices])
        assert area == reference.volume


def test_smallest_eigh3_matches_eigh():
    rng = np.random.default_rng(2)
    m = rng.normal(size=(500, 3, 3)) * [1.0, 0.5, 0.01]
    cov = m @ m.transpose(0, 2, 1)
    values, vectors = np.linalg.eigh(cov)
    rows, cols = np.triu_indices(3)  # xx, xy, xz, yy, yz, zz
    smallest, vec, trace = smallest_eigh3(*(cov[:, r, c] for r, c in zip(rows, cols)))
    assert smallest == pytest.approx(values[:, 0], rel=1e-6, abs=1e-12)
    assert trace == pytest.approx(values.sum(axis=1))
    assert np.abs(np.einsum("ni,ni->n", vec, vectors[:, :, 0])) == pytest.approx(1.0)


def _random_labels(rng, shape=(60, 80), count=4):
    labels = rng.integers(-1, count, size=shape).astype(np.int32)
    # Blocky regions so that components have various sizes.
    return np.kron(labels[::4, ::4], np.ones((4, 4), dtype=np.int32))[: shape[0]]


def test_keep_seeded_components_matches_full_image_version():
    rng = np.random.default_rng(3)
    for _ in range(10):
        labels = _random_labels(rng)
        count = int(labels.max()) + 1
        seeds = [rng.choice(labels.size, 20, replace=False) for _ in range(count)]
        expected = labels.copy()
        for index in range(count):
            mask = labels == index
            components, n = connected_label(mask, structure=EIGHT)
            keep = np.bincount(components.reshape(-1), minlength=n + 1) >= 12
            keep[0] = False
            seeded = np.zeros(n + 1, dtype=bool)
            seeded[components.reshape(-1)[seeds[index]]] = True
            expected[~(keep & seeded)[components] & mask] = -1
        assert np.array_equal(keep_seeded_components(labels, seeds, 12), expected)


def test_fill_holes_matches_binary_closing():
    rng = np.random.default_rng(4)
    h, w = 60, 80
    points = np.zeros((h, w, 3))
    points[..., 2] = rng.normal(scale=0.002, size=(h, w))
    depth = np.full((h, w), 1.0)
    valid = rng.random((h, w)) > 0.1
    frame = Frame(points, depth, valid, 1, points, depth, valid, np.zeros(3))
    planes = [(np.array([0.0, 0.0, 1.0]), 0.0)] * 4
    options = DetectionOptions()
    for iterations in (1, 2, 3):
        labels = _random_labels(rng)
        labels[rng.random((h, w)) < 0.2] = -1
        expected = labels.copy()
        for index, (normal, offset) in enumerate(planes):
            mask = labels == index
            closed = binary_closing(mask, EIGHT, iterations=iterations, border_value=0)
            add = closed & ~mask & (expected < 0) & valid
            limit = np.maximum(4.0 * options.noise.sigma(depth), 0.005)
            add &= np.abs(points @ normal + offset) < limit
            expected[add] = index
        out = fill_holes(frame, labels, planes, iterations, 4.0, options)
        assert np.array_equal(out, expected)
