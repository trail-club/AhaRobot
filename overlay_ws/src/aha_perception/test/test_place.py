import math

import numpy as np
import pytest

from aha_perception.place import (
    PlaceOptions,
    PlaceStatus,
    base_cloud,
    find_place_candidates,
    find_place_obstacles,
    footprint_radius,
    min_area_rectangle,
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


def box_points(cx, cy, z0, sx, sy, sz, yaw=0.0):
    """Top and side faces of an upright box rotated by ``yaw`` about its center."""
    x0, x1, y0, y1 = -sx / 2, sx / 2, -sy / 2, sy / 2
    faces = [grid(x0, x1, y0, y1, z0 + sz)]
    for z in np.arange(z0, z0 + sz, STEP):
        faces += [grid(x0, x0, y0, y1, z), grid(x1, x1, y0, y1, z)]
        faces += [grid(x0, x1, y0, y0, z), grid(x0, x1, y1, y1, z)]
    local = np.concatenate(faces)
    c, s = math.cos(yaw), math.sin(yaw)
    xy = local[:, :2] @ np.array([[c, s], [-s, c]]) + [cx, cy]
    return np.column_stack([xy, local[:, 2]])


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

    def cloud(self):
        points = np.concatenate(self.parts)[None]
        return points, np.ones(points.shape[:2])

    def run(self, size, **kwargs):
        points, depth = self.cloud()
        return find_place_candidates(self.planes, points, depth, size, **kwargs)

    def obstacles(self, **kwargs):
        points, depth = self.cloud()
        return find_place_obstacles(self.planes, points, depth, **kwargs)


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
    result = scene.run(size)
    assert result.candidates == []
    assert result.status == PlaceStatus.NO_SURFACE
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
    assert scaled.box_min_points == 3
    assert scaled.support_closing == options.support_closing + 1
    assert scaled.resolution == options.resolution
    dense = PlaceOptions(obstacle_min_points=9).for_input_stride(2)
    assert dense.obstacle_min_points == 3


BOARD = (1.0, 1.35, -0.4, 0.4)  # x0, x1, y0, y1 of two stacked shelf boards


def shelf_scene():
    scene = Scene()
    scene.add(grid(*BOARD, 0.40), PlaneLabel.SHELF)
    scene.add(grid(*BOARD, 0.62), PlaneLabel.SHELF)
    return scene


def test_height_restriction_picks_the_board():
    scene = shelf_scene()
    size = (0.08, 0.08, 0.10)
    for height, index in ((0.40, 0), (0.62, 1), (0.42, 0)):
        result = scene.run(size, surface_height=height)
        assert result.status == PlaceStatus.OK
        assert result.supports == [index]
        assert result.candidates
        assert {c.plane_index for c in result.candidates} == {index}
    assert scene.run(size).status == PlaceStatus.OK
    assert {c.plane_index for c in scene.run(size).candidates} == {0, 1}


def test_height_restriction_without_a_match_is_no_surface():
    scene = shelf_scene()
    size = (0.08, 0.08, 0.10)
    for kwargs in (
        {"surface_height": 0.50},
        {"surface_height": 0.42, "height_tolerance": 0.01},
        {"surface_height": 0.40, "labels": ["table"]},
    ):
        result = scene.run(size, **kwargs)
        assert result.status == PlaceStatus.NO_SURFACE
        assert result.candidates == [] and result.supports == []
    with pytest.raises(ValueError):
        scene.run(size, surface_height=float("nan"))


def test_height_restriction_on_a_full_board_is_no_space():
    # The board above covers the whole lower board for a 0.25 m tall object.
    result = shelf_scene().run((0.08, 0.08, 0.25), surface_height=0.40)
    assert result.status == PlaceStatus.NO_SPACE
    assert result.supports == [0]
    assert result.candidates == []


def yaw_error(a, b):
    """Difference of two rectangle axes, modulo pi."""
    return abs(math.remainder(a - b, math.pi))


def test_min_area_rectangle_of_rotated_points():
    yaw = math.radians(25)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[x, y] for x in (-0.3, 0.3) for y in (-0.1, 0.1)])
    inner = np.random.default_rng(0).uniform(-0.09, 0.09, size=(30, 2))
    points = np.vstack([local, inner]) @ np.array([[c, s], [-s, c]]) + [1.0, -0.5]
    center, angle, (length, width) = min_area_rectangle(points)
    np.testing.assert_allclose(center, [1.0, -0.5], atol=1e-9)
    assert sorted([length, width]) == pytest.approx([0.2, 0.6])
    long_axis = angle if length > width else angle + math.pi / 2
    assert yaw_error(long_axis, yaw) == pytest.approx(0.0, abs=1e-9)


def test_box_on_a_table_becomes_one_oriented_box():
    yaw = math.radians(30)
    scene = table_scene()
    scene.add(box_points(1.4, 0.05, 0.75, 0.10, 0.06, 0.12, yaw))
    (box,) = scene.obstacles()
    assert box.plane_index == 0
    np.testing.assert_allclose(box.center[:2], [1.4, 0.05], atol=0.01)
    assert box.center[2] - box.size[2] / 2 == pytest.approx(0.75, abs=1e-6)
    assert box.size[2] == pytest.approx(0.12, abs=0.01)
    length, width = box.size[:2]
    assert sorted([length, width]) == pytest.approx([0.06, 0.10], abs=0.02)
    long_axis = box.yaw if length > width else box.yaw + math.pi / 2
    assert yaw_error(long_axis, yaw) < math.radians(3)
    np.testing.assert_allclose(box.rotation[:, 2], UP, atol=1e-9)
    np.testing.assert_allclose(box.rotation.T @ box.rotation, np.eye(3), atol=1e-9)


def test_two_objects_become_two_boxes():
    scene = table_scene()
    scene.add(box_points(1.2, -0.1, 0.75, 0.05, 0.05, 0.10))
    scene.add(box_points(1.5, 0.15, 0.75, 0.08, 0.06, 0.15))
    boxes = sorted(scene.obstacles(), key=lambda b: b.center[0])
    assert len(boxes) == 2
    np.testing.assert_allclose(boxes[0].center[:2], [1.2, -0.1], atol=0.01)
    np.testing.assert_allclose(boxes[1].center[:2], [1.5, 0.15], atol=0.01)
    assert [b.size[2] for b in boxes] == pytest.approx([0.10, 0.15], abs=0.01)


def test_object_on_the_lower_board_and_the_cap_of_the_board_above():
    scene = shelf_scene()
    # Front face of the upper board, seen from the camera: not an obstacle.
    for z in np.arange(0.60, 0.62, STEP):
        scene.add(grid(BOARD[0], BOARD[0], BOARD[2], BOARD[3], z))
    scene.add(box_points(1.2, 0.1, 0.40, 0.06, 0.06, 0.10))
    (box,) = scene.obstacles()
    assert box.plane_index == 0
    np.testing.assert_allclose(box.center[:2], [1.2, 0.1], atol=0.01)
    assert box.size[2] == pytest.approx(0.10, abs=0.01)

    # A tall object (synthetic, through the board above) is cut at
    # cap_margin below that board on the lower one.
    tall = shelf_scene().add(box_points(1.2, 0.1, 0.40, 0.06, 0.06, 0.30))
    options = PlaceOptions()

    def lower(boxes):
        (box,) = [b for b in boxes if b.plane_index == 0]
        return box

    cap = 0.62 - 0.40 - options.cap_margin
    assert cap - 0.01 <= lower(tall.obstacles(options=options)).size[2] <= cap + 1e-9
    uncapped = lower(tall.obstacles(options=PlaceOptions(cap_min_overlap=2.0)))
    assert uncapped.size[2] == pytest.approx(0.30, abs=0.01)


def test_l_shaped_structure_is_split_into_tiles():
    # Back and side panels of a shelf: a thin L must not become a box over the
    # free board between them.
    scene = table_scene()
    scene.add(box_points(1.79, 0.0, 0.75, 0.02, 0.6, 0.2))
    scene.add(box_points(1.4, 0.29, 0.75, 0.8, 0.02, 0.2))
    boxes = scene.obstacles()
    assert len(boxes) > 2
    corner = np.array([1.6, 0.1, 0.80])
    for box in boxes:
        local = box.rotation.T @ (corner - box.center)
        assert np.any(np.abs(local) > box.size / 2)
    assert max(b.size[2] for b in boxes) == pytest.approx(0.2, abs=0.01)


def test_wall_noise_is_not_a_box():
    # A wall at the back edge of the table; its noise (1 cm in front, not
    # inliers) is left out, an object 10 cm in front of it is kept.
    scene = table_scene()
    y, z = np.meshgrid(np.arange(-0.4, 0.4, STEP), np.arange(0.5, 1.5, STEP))
    wall = np.column_stack([np.full(y.size, 1.85), y.ravel(), z.ravel()])
    scene.add(wall, PlaneLabel.WALL)
    scene.add(box_points(1.7, 0.0, 0.75, 0.06, 0.06, 0.1))
    noisy = Scene()
    noisy.parts, noisy.planes = list(scene.parts), list(scene.planes)
    scene.add(wall[::7] - [0.01, 0.0, 0.0])
    (box,) = scene.obstacles()
    np.testing.assert_allclose(box.center[:2], [1.7, 0.0], atol=0.01)
    # Beyond max(obstacle_min_height, wall_k * sigma) the same points are boxes.
    assert len(noisy.add(wall[::7] - [0.05, 0.0, 0.0]).obstacles()) > 1


def test_speckle_and_points_off_the_plane_are_not_boxes():
    scene = table_scene()
    scene.add(grid(1.3, 1.31, 0.0, 0.005, 0.80))  # 2 cells of noise
    scene.add(np.array([[1.6, -0.2, 0.85], [1.1, 0.2, 0.9]]))
    # A hand in front of the table edge (outside the hull), above its height.
    scene.add(box_points(0.97, 0.0, 0.85, 0.03, 0.06, 0.08))
    assert scene.obstacles() == []
    # Floor planes get no boxes by default.
    floor = Scene().add(grid(0.5, 1.5, -0.5, 0.5, 0.0), PlaneLabel.FLOOR)
    floor.add(box_points(1.0, 0.0, 0.0, 0.1, 0.1, 0.2))
    assert floor.obstacles() == []


def test_base_cloud():
    cloud = np.array([[[0.0, 0.0, 2.0], [np.nan, np.nan, np.nan]]])
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 0.0, 0.5)
    points, depth = base_cloud(cloud, transform)
    np.testing.assert_allclose(points[0, 0], [1.0, 0.0, 2.5])
    assert depth[0, 0] == 2.0
    assert np.isnan(points[0, 1]).all()
