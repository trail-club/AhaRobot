import numpy as np
import pytest

from aha_perception.place import (
    PlaceOptions,
    base_cloud,
    find_place_candidates,
    footprint_radius,
)
from aha_perception.planes import PlaneLabel
from aha_perception.planes.segmentation import make_plane

UP = np.array([0.0, 0.0, 1.0])
CAMERA = np.array([0.0, 0.0, 1.43])
STEP = 0.005
TABLE = (1.0, 1.8, -0.3, 0.3, 0.75)  # x0, x1, y0, y1, z: 0.8 x 0.6 m
CENTER = np.array([1.4, 0.0, 0.75])


def grid(x0, x1, y0, y1, z, keep=None):
    x, y = np.meshgrid(np.arange(x0, x1 + 1e-9, STEP), np.arange(y0, y1 + 1e-9, STEP))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])
    return points if keep is None else points[keep(points)]


def box_points(cx, cy, z0, sx, sy, sz):
    """Top and side faces of an upright box."""
    x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    faces = [grid(x0, x1, y0, y1, z0 + sz)]
    for z in np.arange(z0, z0 + sz, STEP):
        faces += [grid(x0, x0, y0, y1, z), grid(x1, x1, y0, y1, z)]
        faces += [grid(x0, x1, y0, y0, z), grid(x0, x1, y1, y1, z)]
    return np.concatenate(faces)


class Scene:
    """Organized (1, N, 3) cloud from parts; planes index into it."""

    def __init__(self):
        self.parts = []
        self.planes = []

    def add(self, points, label=None):
        start = sum(len(p) for p in self.parts)
        self.parts.append(np.asarray(points, dtype=np.float64))
        if label is not None:
            inliers = np.arange(start, start + len(points))
            plane = make_plane(self.parts[-1], inliers, CAMERA, UP)
            plane.label = label
            self.planes.append(plane)
        return self

    def run(self, size, **kwargs):
        points = np.concatenate(self.parts)[None]
        depth = np.ones(points.shape[:2])
        return find_place_candidates(self.planes, points, depth, size, **kwargs)


def table_scene(keep=None):
    return Scene().add(grid(*TABLE, keep=keep), PlaneLabel.TABLE)


def test_empty_table_center_candidate():
    size = (0.06, 0.06, 0.12)
    result = table_scene().run(size)
    assert result.candidates
    top = result.candidates[0]
    assert np.linalg.norm(top.position - CENTER) < 0.02
    assert top.label == "table"
    assert top.surface_height == pytest.approx(0.75, abs=1e-6)
    assert top.clearance == pytest.approx(0.3, abs=0.01)
    np.testing.assert_allclose(top.normal, UP, atol=1e-9)
    # Placeable centers keep radius + edge margin to the table edges.
    options = PlaceOptions()
    grid_ = result.grids[0]
    centers = grid_.cell_points(np.argwhere(grid_.placeable))
    reach = footprint_radius(size) + options.edge_margin
    assert centers[:, 0].min() >= TABLE[0] + reach - 0.01
    assert centers[:, 0].max() <= TABLE[1] - reach + 0.01
    assert np.abs(centers[:, 1]).max() <= 0.3 - reach + 0.01
    assert np.abs(centers[:, 1]).max() >= 0.3 - reach - 0.02
    clearances = [c.clearance for c in result.candidates]
    assert clearances == sorted(clearances, reverse=True)


def test_box_in_the_middle_is_avoided():
    size = (0.08, 0.08, 0.15)
    half = 0.1

    def outside_box(p):
        return (np.abs(p[:, 0] - CENTER[0]) > half) | (np.abs(p[:, 1]) > half)

    scene = table_scene(outside_box)
    scene.add(box_points(CENTER[0], 0.0, 0.75, 2 * half, 2 * half, 0.12))
    result = scene.run(size)
    assert result.candidates
    options = PlaceOptions()
    reach = footprint_radius(size) + options.obstacle_margin
    for c in result.candidates:
        d = np.maximum(np.abs(c.position[:2] - CENTER[:2]) - half, 0.0)
        assert np.linalg.norm(d) >= reach - 0.01
    assert result.grids[0].obstacle.sum() > 0


def test_object_too_large_has_no_candidate():
    result = table_scene().run((0.6, 0.5, 0.1))
    assert result.candidates == []
    assert not result.grids[0].placeable.any()


def test_board_above_too_low_blocks_the_shelf():
    board = (1.0, 1.35, -0.4, 0.4)
    scene = Scene()
    scene.add(grid(*board, 0.40), PlaneLabel.SHELF)
    scene.add(grid(*board, 0.62), PlaneLabel.SHELF)
    tall = scene.run((0.08, 0.08, 0.25))
    assert tall.candidates
    assert {c.plane_index for c in tall.candidates} == {1}
    assert tall.grids[0].overhead.any()
    short = scene.run((0.08, 0.08, 0.10))
    assert {c.plane_index for c in short.candidates} == {0, 1}
    assert not short.grids[0].overhead.any()


def test_unobserved_region_is_not_used():
    hole = (1.0, 1.4, -0.3, 0.3)  # front half of the table is unobserved

    def observed(p):
        return ~(
            (p[:, 0] >= hole[0] + 0.05)
            & (p[:, 0] <= hole[1])
            & (p[:, 1] >= hole[2] + 0.05)
            & (p[:, 1] <= hole[3] - 0.05)
        )

    size = (0.06, 0.06, 0.12)
    result = table_scene(observed).run(size)
    grid_ = result.grids[0]
    assert not grid_.support.all()
    assert result.candidates
    reach = footprint_radius(size)
    for c in result.candidates:
        assert c.position[0] >= hole[1] + reach
    centers = grid_.cell_points(np.argwhere(grid_.placeable))
    assert centers[:, 0].min() >= hole[1] + reach


def test_near_orders_by_distance():
    near = np.array([1.05, 0.25, 0.75])
    result = table_scene().run((0.06, 0.06, 0.12), near=near, max_candidates=5)
    assert len(result.candidates) == 5
    distances = [c.distance for c in result.candidates]
    assert distances == sorted(distances)
    reach = footprint_radius((0.06, 0.06, 0.12)) + PlaceOptions().edge_margin
    corner = np.array([TABLE[0] + reach, TABLE[3] - reach, 0.75])
    assert np.linalg.norm(result.candidates[0].position - corner) < 0.02
    assert result.candidates[0].score == pytest.approx(-distances[0])


def test_labels_select_supports():
    scene = Scene()
    scene.add(grid(0.5, 1.5, -0.5, 0.5, 0.0), PlaneLabel.FLOOR)
    size = (0.06, 0.06, 0.12)
    assert scene.run(size).candidates == []
    assert scene.run(size, labels=["floor"]).candidates
    with pytest.raises(ValueError):
        scene.run(size, labels=["wall"])
    with pytest.raises(ValueError):
        scene.run((0.1, 0.0, 0.1))


def test_plane_indices_restrict_supports():
    scene = table_scene()
    scene.add(grid(2.5, 3.1, -0.3, 0.3, 0.75), PlaneLabel.OTHER)
    size = (0.06, 0.06, 0.12)
    assert {c.plane_index for c in scene.run(size).candidates} == {0}
    # With plane indices every support label is allowed unless labels are given.
    only_other = scene.run(size, plane_indices=[1])
    assert only_other.supports == [1]
    assert {c.plane_index for c in only_other.candidates} == {1}
    assert scene.run(size, plane_indices=[1], labels=["table"]).supports == []
    assert scene.run(size, plane_indices=[]).candidates == []


def test_low_obstacle_points_are_ignored():
    scene = table_scene()
    scene.add(grid(1.3, 1.5, -0.1, 0.1, 0.76))  # 1 cm above the table
    result = scene.run((0.06, 0.06, 0.12))
    assert not result.grids[0].obstacle.any()


def test_options_for_input_stride():
    options = PlaceOptions()
    assert options.for_input_stride(1) is options
    scaled = options.for_input_stride(2)
    assert scaled.obstacle_min_points == 1
    assert scaled.support_closing == options.support_closing + 1
    assert scaled.resolution == options.resolution
    dense = PlaceOptions(obstacle_min_points=9).for_input_stride(2)
    assert dense.obstacle_min_points == 3


def test_base_cloud():
    cloud = np.array([[[0.0, 0.0, 2.0], [np.nan, np.nan, np.nan]]])
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 0.0, 0.5)
    points, depth = base_cloud(cloud, transform)
    np.testing.assert_allclose(points[0, 0], [1.0, 0.0, 2.5])
    assert depth[0, 0] == 2.0
    assert np.isnan(points[0, 1]).all()
