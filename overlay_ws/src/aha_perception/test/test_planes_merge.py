import numpy as np
import pytest

from aha_perception.planes import DETECTORS, PlaneLabel, detect_planes
from aha_perception.planes.merge import merge_coplanar
from aha_perception.planes.preprocess import prepare_frame
from aha_perception.planes.segmentation import planes_from_labels


def _fragments(cloud, options, split):
    """Planes built from a label image given by ``split(points, rows, cols)``."""
    frame = prepare_frame(cloud, options, 1)
    rows, cols = np.indices(frame.valid.shape)
    labels = np.where(frame.valid, split(frame.points, rows, cols), -1)
    return frame, planes_from_labels(frame, labels.astype(np.int32), options)


def _floor_halves(options, render_cloud):
    cloud = render_cloud(horizontal=[(0.0, 0.5, 3.0, -1.0, 1.0)])
    return _fragments(cloud, options, lambda p, r, c: (c >= 320).astype(int))


def test_adjacent_coplanar_fragments_merge(options, render_cloud):
    frame, planes = _floor_halves(options, render_cloud)
    assert len(planes) == 2
    merged = merge_coplanar(frame, planes, options)
    assert len(merged) == 1
    plane = merged[0]
    assert len(plane.inliers) >= 0.99 * sum(len(p.inliers) for p in planes)
    assert abs(plane.centroid[2]) < 0.01
    assert plane.label == PlaneLabel.FLOOR


def test_merged_plane_fields_consistent(options, render_cloud):
    frame, planes = _floor_halves(options, render_cloud)
    plane = merge_coplanar(frame, planes, options)[0]
    points = frame.full_points.reshape(-1, 3)[plane.inliers]
    assert np.linalg.norm(plane.normal) == pytest.approx(1.0)
    assert plane.offset == pytest.approx(-float(plane.normal @ plane.centroid))
    assert np.allclose(plane.centroid, points.mean(axis=0), atol=1e-9)
    residual = points @ plane.normal + plane.offset
    assert plane.rms == pytest.approx(float(np.sqrt(np.mean(residual**2))))
    assert np.abs(plane.hull_points() @ plane.normal + plane.offset).max() < 1e-9
    assert plane.area == pytest.approx(2.5 * 2.0, rel=0.3)
    assert len(np.unique(plane.inliers)) == len(plane.inliers)
    assert frame.full_valid.reshape(-1)[plane.inliers].all()


def test_merge_is_idempotent(options, render_cloud):
    frame, planes = _floor_halves(options, render_cloud)
    once = merge_coplanar(frame, planes, options)
    twice = merge_coplanar(frame, once, options)
    assert len(twice) == len(once)
    for a, b in zip(once, twice):
        assert np.array_equal(a.inliers, b.inliers)
        assert a.offset == b.offset


def test_parallel_planes_5cm_apart_stay_separate(options, render_cloud):
    cloud = render_cloud(
        horizontal=[(0.70, 1.0, 1.6, -0.5, 0.5), (0.75, 1.5, 2.2, -0.5, 0.5)]
    )
    frame, planes = _fragments(
        cloud, options, lambda p, r, c: (p[..., 2] > 0.725).astype(int)
    )
    assert len(planes) == 2
    merged = merge_coplanar(frame, planes, options)
    assert merged is planes
    heights = sorted(p.centroid[2] for p in merged)
    assert heights == pytest.approx([0.70, 0.75], abs=0.01)


def test_separate_coplanar_tables_stay_separate(options, render_cloud):
    cloud = render_cloud(
        horizontal=[(0.72, 1.0, 1.8, -0.9, -0.3), (0.72, 1.0, 1.8, 0.3, 0.9)]
    )
    frame, planes = _fragments(
        cloud, options, lambda p, r, c: (p[..., 1] > 0.0).astype(int)
    )
    assert len(planes) == 2
    assert len(merge_coplanar(frame, planes, options)) == 2
    assert len(merge_coplanar(frame, planes, options, hull_gap=0.1)) == 2
    assert len(merge_coplanar(frame, planes, options, hull_gap=1.0)) == 1


@pytest.mark.parametrize("method", ["ransac_gravity", "ransac_gravity_refined"])
def test_detector_merge_option(method, options, render_cloud):
    cloud = render_cloud(horizontal=[(0.0, -5, 5, -5, 5), (0.72, 1.0, 1.8, -0.6, 0.6)])
    params = DETECTORS[method].params_type(merge=True)
    planes = detect_planes(cloud, method, options, params)
    labels = sorted(p.label.value for p in planes)
    assert labels == ["floor", "table"]
