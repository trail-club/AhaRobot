import math

import numpy as np
import pytest
from conftest import CX, CY, FX, FY, render

from aha_perception.plane_detection_io import (
    LABEL_COLORS,
    POINT_DTYPE,
    depth_from_buffer,
    depth_mm_to_m,
    depth_to_buffer,
    depth_to_cloud,
    label_color,
    pack_rgb,
    plane_palette,
    plane_points,
    plane_text,
    quaternion_from_matrix,
    quaternion_from_normal_yaw,
    quaternion_from_z_axis,
    seconds_to_stamp,
    stamp_to_seconds,
    transform_point,
    transform_to_matrix,
    xyzrgb_records,
)
from aha_perception.planes import DetectionOptions, Plane, PlaneLabel, detect_planes


def make_plane(label, normal, centroid, inliers, area=1.0):
    return Plane(
        normal=np.asarray(normal, float),
        offset=-float(np.dot(normal, centroid)),
        inliers=np.asarray(inliers),
        centroid=np.asarray(centroid, float),
        axes=np.eye(3)[:2],
        hull=np.zeros((3, 2)),
        area=area,
        rms=0.0,
        label=label,
    )


def test_depth_to_cloud_backprojects_and_masks_invalid():
    depth = np.full((4, 6), 2.0, np.float32)
    depth[0, 0] = 0.0
    depth[1, 1] = np.nan
    depth[2, 2] = np.inf
    cloud = depth_to_cloud(depth, fx=2.0, fy=4.0, cx=1.0, cy=2.0)
    assert cloud.shape == (4, 6, 3) and cloud.dtype == np.float32
    assert np.isnan(cloud[0, 0]).all()
    assert np.isnan(cloud[1, 1]).all()
    assert np.isnan(cloud[2, 2]).all()
    np.testing.assert_allclose(cloud[3, 5], [(5 - 1) / 2 * 2, (3 - 2) / 4 * 2, 2.0])


def test_depth_to_cloud_stride_keeps_full_resolution_pixels():
    depth = np.full((480, 640), 1.5, np.float32)
    full = depth_to_cloud(depth, FX, FY, CX, CY)
    strided = depth_to_cloud(depth, FX, FY, CX, CY, stride=2)
    assert strided.shape == (240, 320, 3)
    np.testing.assert_allclose(strided, full[::2, ::2])
    with pytest.raises(ValueError):
        depth_to_cloud(depth, FX, FY, CX, CY, stride=0)


def test_depth_from_buffer_32f_and_16u_with_padding():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    padded = np.zeros((3, 5), np.float32)
    padded[:, :4] = image
    out = depth_from_buffer(padded.tobytes(), 3, 4, 20, "32FC1")
    np.testing.assert_array_equal(out, image)

    mm = np.array([[0, 1000], [1500, 65535]], dtype=">u2")
    out = depth_from_buffer(mm.tobytes(), 2, 2, 4, "16UC1", is_bigendian=True)
    np.testing.assert_allclose(out, [[0.0, 1.0], [1.5, 65.535]], rtol=1e-6)
    np.testing.assert_allclose(depth_mm_to_m(np.array([2000], np.uint16)), [2.0])

    with pytest.raises(ValueError):
        depth_from_buffer(b"", 1, 1, 4, "rgb8")
    with pytest.raises(ValueError):
        depth_from_buffer(image.tobytes(), 3, 4, 8, "32FC1")


def test_depth_to_buffer_round_trip_and_invalid():
    depth = np.array([[1.25, np.nan, np.inf], [0.0, 2.0004, 70.0]], np.float32)
    data, step = depth_to_buffer(depth, "32FC1")
    assert step == 12
    out = depth_from_buffer(data, 2, 3, step, "32FC1")
    np.testing.assert_array_equal(np.isnan(out), [[0, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(out[0, 0], depth[0, 0])

    data, step = depth_to_buffer(depth, "16UC1")
    assert step == 6
    mm = np.frombuffer(data, "<u2").reshape(2, 3)
    np.testing.assert_array_equal(mm, [[1250, 0, 0], [0, 2000, 65535]])
    with pytest.raises(ValueError):
        depth_to_buffer(depth, "rgb8")


def test_transform_to_matrix():
    np.testing.assert_allclose(
        transform_to_matrix((1, 2, 3), (0, 0, 0, 1))[:3, 3], [1, 2, 3]
    )
    s = np.sqrt(0.5)
    matrix = transform_to_matrix((0, 0, 0), (0, 0, s, s))  # yaw +90 deg
    np.testing.assert_allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # Unnormalized input is normalized.
    np.testing.assert_allclose(
        transform_to_matrix((0, 0, 0), (0, 0, 2, 2)), matrix, atol=1e-12
    )
    with pytest.raises(ValueError):
        transform_to_matrix((0, 0, 0), (0, 0, 0, 0))


def test_transform_point():
    s = np.sqrt(0.5)
    matrix = transform_to_matrix((1, 0, 0), (0, 0, s, s))
    np.testing.assert_allclose(
        transform_point(matrix, (1, 0, 0)), [1, 1, 0], atol=1e-12
    )


@pytest.mark.parametrize(
    "direction", [(0, 0, 1), (0, 0, -1), (0.1, -0.05, 1.0), (1, 0, 0), (0, 1, 0.2)]
)
def test_quaternion_from_z_axis(direction):
    q = quaternion_from_z_axis(direction)
    assert np.isclose(np.linalg.norm(q), 1.0)
    rotated = transform_to_matrix((0, 0, 0), q)[:3, :3] @ [0, 0, 1]
    expected = np.asarray(direction, float) / np.linalg.norm(direction)
    np.testing.assert_allclose(rotated, expected, atol=1e-9)
    # No rotation about z for an upright direction.
    assert quaternion_from_z_axis((0, 0, 2)) == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("normal", [(0, 0, 1), (0.1, -0.05, 1.0), (-0.15, 0.1, 1.0)])
@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.5])
def test_quaternion_from_normal_yaw(normal, yaw):
    q = quaternion_from_normal_yaw(normal, yaw)
    rotation = transform_to_matrix((0, 0, 0), q)[:3, :3]
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    np.testing.assert_allclose(rotation[:, 2], n, atol=1e-9)
    # x lies in the plane and heads the yaw.
    assert rotation[:, 0] @ n == pytest.approx(0.0, abs=1e-9)
    heading = math.atan2(rotation[1, 0], rotation[0, 0])
    assert math.remainder(heading - yaw, 2 * math.pi) == pytest.approx(0.0, abs=1e-9)


def test_quaternion_from_normal_yaw_rejects_a_horizontal_normal():
    with pytest.raises(ValueError):
        quaternion_from_normal_yaw((1.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "q",
    [(0, 0, 0, 1), (0, 0, 0.6, 0.8), (1, 0, 0, 0), (0, 1, 0, 0), (0.1, -0.7, 0.2, 0.3)],
)
def test_quaternion_from_matrix(q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    rotation = transform_to_matrix((0, 0, 0), q)[:3, :3]
    result = np.asarray(quaternion_from_matrix(rotation))
    assert min(np.abs(result - q).max(), np.abs(result + q).max()) < 1e-9


def test_stamp_seconds_round_trip():
    class Stamp:
        sec, nanosec = 1791105388, 607856500

    seconds = stamp_to_seconds(Stamp)
    assert seconds == pytest.approx(1791105388.6078565)
    sec, nanosec = seconds_to_stamp(seconds)
    assert sec == Stamp.sec
    assert abs(nanosec - Stamp.nanosec) < 1000
    assert seconds_to_stamp(2.9999999999) == (3, 0)


def test_colors():
    assert set(LABEL_COLORS) == set(PlaneLabel)
    assert label_color("table") == label_color(PlaneLabel.TABLE)
    palette = [plane_palette(i) for i in range(8)]
    assert all(0.0 <= c <= 1.0 for rgb in palette for c in rgb)
    assert len(set(palette)) == len(palette)


def test_plane_text():
    table = make_plane(PlaneLabel.TABLE, (0, 0, 1), (1.0, 0.0, 0.724), [0])
    wall = make_plane(PlaneLabel.WALL, (-1, 0, 0), (2.0, 0.0, 1.0), [0], area=2.345)
    assert plane_text(table) == "table 0.72 m"
    assert plane_text(wall) == "wall 2.3 m2"


def test_plane_points_transform_colors_and_subsampling():
    cloud = np.arange(30, dtype=np.float32).reshape(2, 5, 3)
    cloud[1, 4] = np.nan
    transform = transform_to_matrix((0, 0, 1), (0, 0, 0, 1))
    planes = [
        make_plane(PlaneLabel.FLOOR, (0, 0, 1), (0, 0, 0), [0, 1, 2, 3, 4]),
        make_plane(PlaneLabel.TABLE, (0, 0, 1), (0, 0, 0.7), [5, 9]),
    ]
    xyz, rgb = plane_points(planes, cloud, transform, [(1, 0, 0), (0, 0.5, 1)])
    assert xyz.dtype == np.float32 and rgb.dtype == np.uint8
    assert len(xyz) == 6  # the NaN inlier is skipped
    np.testing.assert_allclose(xyz[0], [0, 1, 3])
    np.testing.assert_array_equal(rgb[0], [255, 0, 0])
    np.testing.assert_array_equal(rgb[-1], [0, 128, 255])

    xyz, rgb = plane_points(
        planes, cloud, transform, [(1, 0, 0), (0, 1, 0)], max_points_per_plane=2
    )
    assert len(xyz) == 3
    assert len(np.unique(xyz[:2], axis=0)) == 2

    xyz, rgb = plane_points([], cloud, transform, [])
    assert xyz.shape == (0, 3) and rgb.shape == (0, 3)
    with pytest.raises(ValueError):
        plane_points(planes, cloud, transform, [(1, 0, 0)])


def test_xyzrgb_records_layout():
    xyz = np.array([[1.0, 2.0, 3.0]], np.float32)
    rgb = np.array([[0x12, 0x34, 0x56]], np.uint8)
    records = xyzrgb_records(xyz, rgb)
    assert POINT_DTYPE.itemsize == 16
    assert [POINT_DTYPE.fields[n][1] for n in ("x", "y", "z", "rgb")] == [0, 4, 8, 12]
    data = records.tobytes()
    assert np.frombuffer(data[12:16], np.uint32)[0] == 0x123456
    assert pack_rgb(rgb).dtype == np.float32
    assert xyzrgb_records(np.empty((0, 3)), np.empty((0, 3), np.uint8)).size == 0


def test_detection_on_backprojected_depth(camera_pose):
    """End to end: depth image -> cloud -> planes -> colored points."""
    transform = camera_pose()
    rendered = render(
        horizontal=[(0.0, 0.0, 5.0, -3.0, 3.0), (0.72, 1.2, 2.0, -0.5, 0.5)],
        transform=transform,
    )
    cloud = depth_to_cloud(rendered[..., 2], FX, FY, CX, CY)
    np.testing.assert_allclose(cloud, rendered, rtol=1e-5, atol=1e-5, equal_nan=True)

    options = DetectionOptions(camera_to_base=transform, max_depth=3.0)
    planes = detect_planes(cloud, "ransac_gravity_refined", options)
    labels = sorted(PlaneLabel(p.label).value for p in planes)
    assert labels == ["floor", "table"]
    table = next(p for p in planes if p.label == PlaneLabel.TABLE)
    assert plane_text(table) == "table 0.72 m"

    colors = [plane_palette(i) for i in range(len(planes))]
    xyz, _ = plane_points(planes, cloud, transform, colors, max_points_per_plane=500)
    assert len(xyz) == 1000
    floor_z = xyz[:500, 2] if planes[0].label == PlaneLabel.FLOOR else xyz[500:, 2]
    assert np.abs(floor_z).max() < 0.05
