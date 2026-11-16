"""aha_perception.cloud_map: voxel map and view test of cloud_accumulator.py."""

import math

import numpy as np
import pytest

from aha_perception.cloud_map import (
    VoxelMap,
    in_box,
    key_centers,
    min_filter_depth,
    pack_keys,
    pose_change,
    transform_points,
    unpack_keys,
    voxelize,
)
from aha_perception.planes import DepthNoiseModel
from aha_perception.surfaces import ABSENT, PRESENT, UNKNOWN, CameraView, TrackerOptions

SIZE = 0.01
RED, BLUE = (255, 0, 0), (0, 0, 255)


def test_keys_round_trip_with_negative_indices():
    indices = np.array([[0, 0, 0], [-1, 2, -3], [1000, -1000, 5]])
    np.testing.assert_array_equal(unpack_keys(pack_keys(indices)), indices)


def test_key_out_of_range_raises():
    with pytest.raises(ValueError):
        pack_keys(np.array([[1 << 21, 0, 0]]))


def test_voxelize_averages_points_and_colors_per_voxel():
    points = np.array([[0.001, 0.001, 0.001], [0.009, 0.003, 0.003], [0.015, 0, 0]])
    keys, means, rgb = voxelize(points, np.array([RED, BLUE, BLUE]), SIZE)
    assert len(keys) == 2
    first = np.argmin(key_centers(keys, SIZE)[:, 0])
    np.testing.assert_allclose(means[first], [0.005, 0.002, 0.002])
    assert tuple(rgb[first]) == (128, 0, 128)
    np.testing.assert_allclose(means[1 - first], [0.015, 0, 0])


def frame(points, color=RED):
    points = np.atleast_2d(points)
    return voxelize(points, np.tile(color, (len(points), 1)), SIZE)


def by_key(voxel_map):
    return {k: i for i, k in enumerate(voxel_map.keys)}


def test_insert_counts_hits_and_replaces_color_and_stamp():
    voxel_map = VoxelMap(SIZE)
    voxel_map.insert(*frame([[0.0, 0, 0], [0.5, 0, 0]]), 1.0)
    b_keys, b_points, b_rgb = frame([[0.5, 0, 0]], BLUE)
    voxel_map.insert(b_keys, b_points, b_rgb, 2.0)
    a, b = frame([[0.0, 0, 0], [0.5, 0, 0]])[0]
    index = by_key(voxel_map)
    assert len(voxel_map) == 2
    assert voxel_map.hits[index[a]] == 1 and voxel_map.hits[index[b]] == 2
    assert tuple(voxel_map.rgb[index[b]]) == BLUE
    assert voxel_map.last_seen[index[a]] == 1.0
    assert voxel_map.last_seen[index[b]] == 2.0
    with pytest.raises(ValueError):
        voxel_map.insert(np.array([a, a]), np.zeros((2, 3)), np.zeros((2, 3)), 3.0)


def test_insert_averages_the_frames_then_moving_average():
    voxel_map = VoxelMap(SIZE)
    voxel_map.insert(*frame([0.002, 0.0, 0.0]), 1.0)
    voxel_map.insert(*frame([0.006, 0.0, 0.0]), 2.0)
    np.testing.assert_allclose(voxel_map.points[0], [0.004, 0, 0])
    for i in range(20):
        voxel_map.insert(*frame([0.004, 0.0, 0.0]), 3.0 + i)
    voxel_map.insert(*frame([0.009, 0.0, 0.0]), 30.0)
    # hits 23 > 10: weight MIN_WEIGHT 0.1.
    np.testing.assert_allclose(voxel_map.points[0], [0.0045, 0, 0])


def test_output_min_hits_and_forget():
    voxel_map = VoxelMap(SIZE)
    keys, points, rgb = frame([[0.0, 0, 0], [0.5, 0, 0]])
    voxel_map.insert(keys, points, rgb, 1.0)
    voxel_map.insert(keys[:1], points[:1], rgb[:1], 5.0)
    xyz, _ = voxel_map.output(min_hits=2)
    np.testing.assert_allclose(xyz, [[0, 0, 0]])
    assert voxel_map.forget_older(5.0, 0.0) == 0
    assert voxel_map.forget_older(5.0, 2.0) == 1
    assert len(voxel_map) == 1 and voxel_map.last_seen[0] == 5.0
    assert voxel_map.points.shape == (1, 3)
    voxel_map.clear()
    assert len(voxel_map) == 0


def test_pose_change():
    a = np.eye(4)
    b = np.eye(4)
    angle = math.radians(10.0)
    b[:3, :3] = [
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ]
    b[:3, 3] = (0.03, 0.04, 0.0)
    translation, rotation = pose_change(a, b)
    assert translation == pytest.approx(0.05)
    assert rotation == pytest.approx(angle)


def test_transform_points_and_in_box():
    matrix = np.eye(4)
    matrix[:3, 3] = (1.0, 0.0, 0.0)
    points = transform_points(matrix, np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    np.testing.assert_allclose(points, [[1, 0, 0], [2, 0, 0]])
    np.testing.assert_array_equal(in_box(points, (0, -1, -1), (1.5, 1, 1)), [1, 0])


def test_min_filter_depth_ignores_invalid():
    depth = np.array(
        [[2.0, 2.0, np.nan], [2.0, 1.0, 0.0], [np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )
    filtered = min_filter_depth(depth)
    assert filtered[0, 0] == 1.0
    assert filtered[2, 2] == 1.0
    assert np.isnan(min_filter_depth(np.full((3, 3), np.nan))).all()


def view(depth, stride=1):
    """Camera at the origin looking along +z (optical frame = map frame)."""
    rows, cols = depth.shape
    return CameraView(
        camera_to_tracking=np.eye(4),
        fx=100.0,
        fy=100.0,
        cx=(cols * stride - 1) / 2.0,
        cy=(rows * stride - 1) / 2.0,
        width=cols * stride,
        height=rows * stride,
        min_depth=0.3,
        max_depth=float("inf"),
        depth=min_filter_depth(depth),
        stride=stride,
    )


def states(depth, points):
    options = TrackerOptions(occlusion_tol=0.03)
    return view(depth).states(np.array(points), DepthNoiseModel(focal=100.0), options)


def test_view_test_clears_only_voxels_seen_through():
    far = np.full((40, 40), 2.0, dtype=np.float32)
    assert states(far, [[0, 0, 1.0]])[0] == ABSENT
    same = np.full((40, 40), 1.0, dtype=np.float32)
    assert states(same, [[0, 0, 1.0]])[0] == PRESENT
    near = np.full((40, 40), 0.6, dtype=np.float32)
    assert states(near, [[0, 0, 1.0]])[0] == UNKNOWN
    invalid = np.full((40, 40), np.nan, dtype=np.float32)
    assert states(invalid, [[0, 0, 1.0]])[0] == UNKNOWN
    # Out of the image, and nearer than min_depth.
    assert states(far, [[5.0, 0, 1.0], [0, 0, 0.2]]).tolist() == [UNKNOWN, UNKNOWN]


def test_view_test_keeps_an_edge_voxel_next_to_a_nearer_pixel():
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    depth[:, 21:] = 1.0  # object edge one pixel right of the voxel's pixel
    point = [0.0, 0.0, 1.0]  # projects to column 19.5 -> 20
    assert states(depth, [point])[0] == PRESENT
