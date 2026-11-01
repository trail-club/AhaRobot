import numpy as np
import pytest

from aha_perception.planes import DetectionOptions
from aha_perception.planes.normals import (
    box_sum,
    depth_discontinuities,
    knn_normals,
    organized_normals,
)
from aha_perception.planes.preprocess import prepare_frame


def test_box_sum_matches_direct_sum():
    values = np.random.default_rng(0).random((7, 9))
    out = box_sum(values, 1)
    assert np.isclose(out[3, 4], values[2:5, 3:6].sum())
    assert np.isclose(out[0, 0], values[:2, :2].sum())


def _frame(render_cloud, camera_pose, horizontal, stride=2, noise_scale=0.5):
    options = DetectionOptions(camera_to_base=camera_pose())
    cloud = render_cloud(horizontal=horizontal, noise_scale=noise_scale)
    return prepare_frame(cloud, options, stride), options


@pytest.mark.parametrize("noise_scale, max_median_deg", [(0.0, 0.1), (0.5, 3.0)])
def test_organized_normals_on_floor_point_up(
    render_cloud, camera_pose, noise_scale, max_median_deg
):
    frame, options = _frame(
        render_cloud, camera_pose, [(0.0, -5, 5, -5, 5)], noise_scale=noise_scale
    )
    normals, mask = organized_normals(
        frame.points,
        frame.depth,
        options.noise.inverse_sigma,
        viewpoint=frame.camera_origin,
    )
    assert mask.sum() > 0.8 * frame.valid.sum()
    near = mask & (frame.depth < 2.5)
    angles = np.degrees(np.arccos(np.clip(normals[near] @ [0, 0, 1], -1, 1)))
    assert np.median(angles) < max_median_deg


def test_discontinuities_follow_table_edge(render_cloud, camera_pose):
    frame, options = _frame(
        render_cloud, camera_pose, [(0.0, -5, 5, -5, 5), (0.72, 1.0, 1.8, -0.6, 0.6)]
    )
    disc = depth_discontinuities(frame.depth, options.noise.inverse_sigma)
    heights = frame.points[..., 2]
    table = frame.valid & (np.abs(heights - 0.72) < 0.03)
    floor = frame.valid & (np.abs(heights) < 0.03)
    border = table & ~(np.roll(table, 1, 0) & np.roll(table, -1, 0))
    assert disc[border].mean() > 0.5
    interior = floor.copy()
    interior[:20], interior[-20:], interior[:, :20], interior[:, -20:] = 0, 0, 0, 0
    assert disc[interior & ~np.isnan(frame.depth)].mean() < 0.15


def test_knn_normals_on_noisy_plane():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, size=(2000, 2))
    points = np.column_stack([xy, 0.002 * rng.standard_normal(2000)])
    normals, curvature = knn_normals(points, k=16, viewpoint=[0, 0, 2])
    assert np.median(normals @ [0, 0, 1]) > 0.98
    assert (normals @ [0, 0, 1] > 0).all()
    assert np.median(curvature) < 0.01
