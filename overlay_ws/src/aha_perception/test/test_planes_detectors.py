import numpy as np
import pytest

from aha_perception.planes import DETECTORS, PlaneLabel, detect_planes

METHODS = sorted(DETECTORS)
UP = np.array([0.0, 0.0, 1.0])


def _height(plane):
    return float(plane.centroid @ UP)


def _tilt_deg(plane):
    return float(np.degrees(np.arccos(min(1.0, abs(plane.normal @ UP)))))


@pytest.mark.parametrize("method", METHODS)
def test_single_clean_plane(method, options, render_cloud):
    cloud = render_cloud(horizontal=[(0.0, -5, 5, -5, 5)])
    planes = detect_planes(cloud, method, options)
    assert len(planes) == 1
    plane = planes[0]
    assert _tilt_deg(plane) < 1.0
    assert abs(_height(plane)) < 0.01
    assert plane.label == PlaneLabel.FLOOR
    assert len(plane.inliers) > 0.8 * np.isfinite(cloud[..., 2]).sum()


@pytest.mark.parametrize("method", METHODS)
def test_two_parallel_planes(method, options, render_cloud):
    cloud = render_cloud(
        horizontal=[(0.3, 1.0, 2.2, -0.9, -0.1), (0.8, 1.0, 2.2, 0.1, 0.9)]
    )
    planes = detect_planes(cloud, method, options)
    assert len(planes) == 2
    heights = sorted(_height(p) for p in planes)
    assert heights == pytest.approx([0.3, 0.8], abs=0.015)
    assert all(_tilt_deg(p) < 2.0 for p in planes)


@pytest.mark.parametrize("method", METHODS)
def test_table_and_floor(method, options, render_cloud):
    cloud = render_cloud(horizontal=[(0.0, -5, 5, -5, 5), (0.72, 1.0, 1.8, -0.6, 0.6)])
    planes = detect_planes(cloud, method, options)
    by_label = {}
    for plane in planes:
        by_label.setdefault(plane.label, []).append(plane)
    assert len(by_label.get(PlaneLabel.TABLE, [])) == 1
    assert by_label.get(PlaneLabel.FLOOR)
    table = by_label[PlaneLabel.TABLE][0]
    assert _height(table) == pytest.approx(0.72, abs=0.015)
    assert table.area == pytest.approx(0.8 * 1.2, rel=0.25)
    assert all(abs(_height(p)) < 0.02 for p in by_label[PlaneLabel.FLOOR])


@pytest.mark.parametrize("method", METHODS)
def test_horizontal_only_drops_walls(method, options, render_cloud):
    cloud = render_cloud(
        horizontal=[(0.0, -5, 2.5, -5, 5)], vertical=[(2.5, -3, 3, 0.0, 2.5)]
    )
    walls_on = detect_planes(cloud, method, options)
    assert any(p.label == PlaneLabel.WALL for p in walls_on)
    options.detect_vertical = False
    walls_off = detect_planes(cloud, method, options)
    assert walls_off
    assert all(_tilt_deg(p) < options.horizontal_tol_deg for p in walls_off)


def test_ransac_gravity_organized_normals(options, render_cloud):
    cloud = render_cloud(horizontal=[(0.0, -5, 5, -5, 5), (0.72, 1.0, 1.8, -0.6, 0.6)])
    params = DETECTORS["ransac_gravity"].params_type(normals="organized")
    planes = detect_planes(cloud, "ransac_gravity", options, params)
    labels = sorted(p.label for p in planes)
    assert labels == sorted([PlaneLabel.FLOOR, PlaneLabel.TABLE])


def test_registry_signature(options, render_cloud):
    cloud = render_cloud(horizontal=[(0.0, -5, 5, -5, 5)])
    for name, detector in DETECTORS.items():
        assert detector.name == name
        planes = detector.detect(cloud, options, detector.params_type())
        assert all(p.inliers.dtype.kind == "i" for p in planes)
        assert all(p.inliers.max() < cloud.shape[0] * cloud.shape[1] for p in planes)
