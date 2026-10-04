import numpy as np

from aha_perception.planes import DetectionOptions, PlaneLabel, select_output_planes
from aha_perception.planes.classify import classify_planes
from aha_perception.planes.segmentation import make_plane

UP = np.array([0.0, 0.0, 1.0])
CAMERA = np.array([0.0, 0.0, 1.43])


def _horizontal(z, x0, x1, y0, y1, n=400):
    x, y = np.meshgrid(np.linspace(x0, x1, 20), np.linspace(y0, y1, n // 20))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])
    return make_plane(points, np.arange(len(points)), CAMERA, UP)


def _wall(x, y0, y1, z0, z1):
    y, z = np.meshgrid(np.linspace(y0, y1, 20), np.linspace(z0, z1, 20))
    points = np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])
    return make_plane(points, np.arange(len(points)), CAMERA, UP)


def _labels(planes):
    classify_planes(planes, DetectionOptions())
    return [p.label for p in planes]


def test_floor_table_wall_other():
    planes = [
        _horizontal(0.0, 0.5, 3.0, -1.5, 1.5),
        _horizontal(0.74, 1.0, 1.8, -0.4, 0.4),
        _wall(3.0, -1.5, 1.5, 0.0, 1.2),
        _horizontal(1.3, 1.0, 1.3, 0.8, 1.0),
        _wall(2.0, 0.0, 0.3, 0.2, 0.4),
    ]
    assert _labels(planes) == [
        PlaneLabel.FLOOR,
        PlaneLabel.TABLE,
        PlaneLabel.WALL,
        PlaneLabel.OTHER,
        PlaneLabel.OTHER,
    ]
    assert planes[1].area == np.float64(planes[1].area)
    assert abs(planes[1].area - 0.64) < 1e-6


def test_stacked_boards_are_shelf():
    planes = [
        _horizontal(0.0, 0.3, 3.0, -1.5, 1.5),
        _horizontal(0.40, 1.2, 1.5, -0.4, 0.4),
        _horizontal(0.75, 1.2, 1.45, -0.4, 0.4),
        _horizontal(1.10, 1.2, 1.4, -0.4, 0.4),
    ]
    assert _labels(planes) == [
        PlaneLabel.FLOOR,
        PlaneLabel.SHELF,
        PlaneLabel.SHELF,
        PlaneLabel.SHELF,
    ]


def test_object_top_does_not_turn_table_into_shelf():
    planes = [
        _horizontal(0.74, 1.0, 1.8, -0.4, 0.4),
        _horizontal(0.90, 1.3, 1.42, -0.05, 0.07),
    ]
    assert _labels(planes)[0] == PlaneLabel.TABLE


def test_tilted_plane_is_other():
    plane = _horizontal(0.74, 1.0, 1.8, -0.4, 0.4)
    tilt = np.radians(25.0)
    plane.normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    assert _labels([plane]) == [PlaneLabel.OTHER]


def _oblique(tilt_deg):
    """1 m x 1 m plane tilted about the y axis, centered at x=1.5, z=0.8."""
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-0.5, 0.5, 20))
    tilt = np.radians(tilt_deg)
    points = np.column_stack(
        [1.5 + u.ravel() * np.cos(tilt), v.ravel(), 0.8 + u.ravel() * np.sin(tilt)]
    )
    return make_plane(points, np.arange(len(points)), CAMERA, UP)


def _selection_scene():
    return [
        _wall(3.0, -1.5, 1.5, 0.0, 1.2),
        _oblique(45.0),
        _horizontal(0.0, 0.5, 3.0, -1.5, 1.5),
        _wall(2.0, 0.0, 0.3, 0.2, 0.4),
        _horizontal(1.3, 1.0, 1.3, 0.8, 1.0),
        _horizontal(0.74, 1.0, 1.8, -0.4, 0.4),
    ]


def test_select_output_planes_keeps_horizontal_and_walls_in_order():
    planes = _selection_scene()
    options = DetectionOptions()
    classify_planes(planes, options)
    assert planes[1].label == PlaneLabel.OTHER
    assert planes[3].label == PlaneLabel.OTHER
    selected = select_output_planes(planes, options)
    assert [id(p) for p in selected] == [id(planes[i]) for i in (0, 2, 4, 5)]
    assert [p.label for p in selected] == [
        PlaneLabel.WALL,
        PlaneLabel.FLOOR,
        PlaneLabel.OTHER,
        PlaneLabel.TABLE,
    ]


def test_select_output_planes_drops_walls_without_detect_vertical():
    planes = _selection_scene()
    options = DetectionOptions(detect_vertical=False)
    classify_planes(planes, options)
    selected = select_output_planes(planes, options)
    assert [id(p) for p in selected] == [id(planes[i]) for i in (2, 4, 5)]
