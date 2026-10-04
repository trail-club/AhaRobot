import numpy as np

from aha_perception.planes import DetectionOptions, PlaneLabel
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
