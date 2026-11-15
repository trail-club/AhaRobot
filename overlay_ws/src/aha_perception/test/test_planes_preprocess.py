import numpy as np

from aha_perception.planes import DetectionOptions
from aha_perception.planes.preprocess import (
    crop_depth,
    decimate,
    prepare_frame,
    transform_points,
    upsample_labels,
    voxel_downsample,
)


def test_crop_depth_masks_out_of_range_and_nan():
    cloud = np.zeros((2, 3, 3))
    cloud[..., 2] = [[0.1, 1.0, 5.0], [np.nan, 2.0, 3.9]]
    out = crop_depth(cloud, 0.3, 4.0)
    valid = np.isfinite(out).all(axis=-1)
    assert valid.tolist() == [[False, True, False], [False, True, True]]
    assert np.isfinite(cloud[0, 0]).all()


def test_decimate_keeps_every_stride_pixel():
    cloud = np.arange(480 * 640 * 3, dtype=float).reshape(480, 640, 3)
    out = decimate(cloud, 2)
    assert out.shape == (240, 320, 3)
    assert np.array_equal(out[1, 1], cloud[2, 2])


def test_voxel_downsample_averages_and_maps_back():
    points = np.array([[0.01, 0.01, 0.01], [0.03, 0.01, 0.01], [1.01, 1.0, 1.0]])
    values = np.array([1.0, 3.0, 5.0])
    centroids, inverse, means = voxel_downsample(points, 0.1, values)
    assert len(centroids) == 2
    assert inverse[0] == inverse[1] != inverse[2]
    assert np.allclose(centroids[inverse[0]], [0.02, 0.01, 0.01])
    assert means[inverse[0]] == 2.0 and means[inverse[2]] == 5.0


def test_transform_points_applies_rotation_and_translation():
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [1, 2, 3]
    assert np.allclose(transform_points(np.array([1.0, 0, 0]), transform), [1, 3, 3])


def test_prepare_frame_and_label_upsampling(render_cloud, camera_pose):
    cloud = render_cloud(horizontal=[(0.0, -5, 5, -5, 5)])
    options = DetectionOptions(camera_to_base=camera_pose())
    frame = prepare_frame(cloud, options, stride=2)
    assert frame.points.shape == (240, 320, 3)
    assert frame.full_shape == (480, 640)
    assert np.allclose(frame.camera_origin, [0, 0, 1.43])
    heights = frame.full_points[frame.full_valid][:, 2]
    assert abs(np.median(heights)) < 0.005
    assert np.nanmax(frame.full_depth) <= options.max_depth

    labels = np.arange(6).reshape(2, 3)
    up = upsample_labels(labels, 2, (3, 5))
    assert up.shape == (3, 5)
    assert up[2, 4] == labels[1, 2]
