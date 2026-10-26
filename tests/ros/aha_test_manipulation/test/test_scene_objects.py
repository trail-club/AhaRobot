"""Planning scene objects from planes, obstacles and the held object
(aha_test_manipulation.scene_objects)."""

import math

import numpy as np
import pytest
from aha_msgs.msg import PlaceObstacle, Plane
from geometry_msgs.msg import Point32
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation import scene_objects as scene


def rectangle(center, yaw, length, width, z):
    corners = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        local = np.array([sx * length / 2, sy * width / 2])
        xy = geo.rot_z(yaw)[:2, :2] @ local + center
        corners.append((xy[0], xy[1], z))
    return corners


def plane(normal, offset, hull, label="table"):
    msg = Plane()
    msg.label = label
    msg.normal.x, msg.normal.y, msg.normal.z = normal
    msg.offset = offset
    msg.hull.points = [Point32(x=float(x), y=float(y), z=float(z)) for x, y, z in hull]
    c = np.mean(np.array(hull), axis=0)
    msg.centroid.x, msg.centroid.y, msg.centroid.z = (float(v) for v in c)
    msg.height = float(c[2])
    return msg


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def test_min_area_rectangle_of_rotated_points():
    rng = np.random.default_rng(0)
    yaw = math.radians(25)
    corners = np.array(rectangle(np.array([1.0, -0.5]), yaw, 1.2, 0.6, 0))[:, :2]
    inner = rng.uniform(-0.3, 0.3, size=(20, 2)) @ geo.rot_z(yaw)[:2, :2].T + [
        1.0,
        -0.5,
    ]
    center, angle, (a, b) = scene.min_area_rectangle(np.vstack([corners, inner]))
    np.testing.assert_allclose(center, [1.0, -0.5], atol=1e-9)
    assert sorted([a, b]) == pytest.approx([0.6, 1.2])
    assert math.sin(2 * (angle - yaw)) == pytest.approx(0.0, abs=1e-9)


def test_horizontal_plane_becomes_a_box_under_the_surface():
    yaw = math.radians(30)
    hull = rectangle(np.array([0.8, 0.1]), yaw, 1.0, 0.5, 0.8)
    params = scene.SceneParams(support_thickness=0.03, surface_clearance=0.005)
    box = scene.plane_box(plane((0, 0, 1), -0.8, hull), params)
    assert box.kind == scene.SURFACE
    assert box.pose[2, 3] + box.size[2] / 2 == pytest.approx(0.795)
    np.testing.assert_allclose(box.pose[:2, 3], [0.8, 0.1], atol=1e-9)
    assert sorted(box.size[:2]) == pytest.approx([0.5, 1.0])
    assert math.sin(2 * (yaw_of(box.pose) - yaw)) == pytest.approx(0.0, abs=1e-9)


def test_support_padding_grows_the_box_sides_only():
    hull = rectangle(np.array([0.8, 0.1]), 0.0, 1.0, 0.5, 0.8)
    params = scene.SceneParams(
        support_thickness=0.03, surface_clearance=0.005, support_padding=0.03
    )
    box = scene.plane_box(plane((0, 0, 1), -0.8, hull), params)
    assert sorted(box.size[:2]) == pytest.approx([0.56, 1.06])
    assert box.size[2] == pytest.approx(0.03)
    assert box.pose[2, 3] + box.size[2] / 2 == pytest.approx(0.795)
    np.testing.assert_allclose(box.pose[:2, 3], [0.8, 0.1], atol=1e-9)


def test_object_at_needs_a_box_standing_on_the_support_near_the_spot():
    standing = obstacle((0.60, -0.25, 0.65), 0.0, (0.08, 0.08, 0.08))  # bottom 0.61
    assert scene.object_at([standing], (0.60, -0.25, 0.62), 0.61, 0.03, 0.04)
    # 0.025 m beyond the box side (0.64) still counts, 0.05 m does not.
    assert scene.object_at([standing], (0.665, -0.25, 0.62), 0.61, 0.03, 0.04)
    assert not scene.object_at([standing], (0.69, -0.25, 0.62), 0.61, 0.03, 0.04)
    # Another support (fallen to the floor, or the level above).
    assert not scene.object_at([standing], (0.60, -0.25, 0.92), 0.91, 0.03, 0.04)
    fallen = obstacle((0.60, -0.25, 0.04), 0.0, (0.08, 0.08, 0.08))
    assert not scene.object_at([fallen], (0.60, -0.25, 0.62), 0.61, 0.03, 0.04)
    assert not scene.object_at([], (0.60, -0.25, 0.62), 0.61, 0.03, 0.04)


def test_floor_slanted_and_low_planes_are_skipped():
    hull = rectangle(np.array([1.0, 0.0]), 0.0, 1.0, 1.0, 0.0)
    assert scene.plane_box(plane((0, 0, 1), 0.0, hull, "floor")) is None
    assert scene.plane_box(plane((0, 0, 1), -0.02, hull, "other")) is None
    n = np.array([0.0, 0.7, 0.7]) / np.linalg.norm([0.0, 0.7, 0.7])
    assert scene.plane_box(plane(tuple(n), 0.0, hull, "other")) is None
    assert scene.plane_box(plane((0, 0, 1), -0.8, hull[:2])) is None


def test_wall_box_is_centered_on_the_plane():
    # wall x = 2.0 facing -x, spanning y -0.5..1.5 and z 0..2
    hull = [(2.0, -0.5, 0.0), (2.0, 1.5, 0.0), (2.0, 1.5, 2.0), (2.0, -0.5, 2.0)]
    box = scene.plane_box(
        plane((-1, 0, 0), 2.0, hull, "wall"), scene.SceneParams(wall_thickness=0.02)
    )
    assert box.kind == scene.WALL
    np.testing.assert_allclose(box.pose[:3, 3], [2.0, 0.5, 1.0], atol=1e-9)
    assert box.size == pytest.approx((0.02, 2.0, 2.0))
    np.testing.assert_allclose(np.abs(box.pose[:3, 0]), [1, 0, 0], atol=1e-9)


def test_plane_objects_ids_and_frame():
    table = plane((0, 0, 1), -0.8, rectangle(np.array([0.8, 0]), 0, 1, 1, 0.8))
    board = plane(
        (0, 0, 1), -0.6, rectangle(np.array([0.6, 0]), 0, 0.4, 0.6, 0.6), "shelf"
    )
    wall = plane((-1, 0, 0), 2.0, [(2, 0, 0), (2, 1, 0), (2, 1, 2), (2, 0, 2)], "wall")
    floor = plane((0, 0, 1), 0.0, rectangle(np.array([1, 0]), 0, 2, 2, 0.0), "floor")
    objects = scene.plane_objects([table, wall, floor, board], "base_footprint")
    assert [o.id for o in objects] == [
        "aha_place_surface_0",
        "aha_place_wall_0",
        "aha_place_surface_1",
    ]
    for obj in objects:
        assert obj.header.frame_id == "base_footprint"
        assert obj.operation == CollisionObject.ADD
        assert obj.primitives[0].type == SolidPrimitive.BOX
        assert obj.id.startswith(scene.PERCEIVED_PREFIXES)


def obstacle(center, yaw, size, plane_id=0):
    msg = PlaceObstacle()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = center
    msg.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.orientation.w = math.cos(yaw / 2)
    msg.size.x, msg.size.y, msg.size.z = size
    msg.plane_id = plane_id
    return msg


def test_obstacles_become_padded_boxes_on_the_surface():
    yaw = math.radians(20)
    msg = obstacle((0.6, -0.2, 0.86), yaw, (0.10, 0.06, 0.12))
    box = scene.obstacle_box(msg, padding=0.01)
    assert box.kind == scene.OBSTACLE
    assert box.size == pytest.approx((0.12, 0.08, 0.13))
    np.testing.assert_allclose(box.pose[:2, 3], [0.6, -0.2], atol=1e-9)
    # Bottom stays on the surface (0.80), the top grows by the padding.
    assert box.pose[2, 3] - box.size[2] / 2 == pytest.approx(0.80)
    assert box.pose[2, 3] + box.size[2] / 2 == pytest.approx(0.93)
    assert yaw_of(box.pose) == pytest.approx(yaw)
    objects = scene.obstacle_objects([box, box], "base_footprint")
    assert [o.id for o in objects] == ["aha_place_obstacle_0", "aha_place_obstacle_1"]
    assert all(o.id.startswith(scene.PERCEIVED_PREFIXES) for o in objects)
    assert list(objects[0].primitives[0].dimensions) == pytest.approx(list(box.size))
    p = objects[0].pose.position
    np.testing.assert_allclose([p.x, p.y, p.z], box.pose[:3, 3])


def test_box_distance():
    box = scene.obstacle_box(obstacle((1.0, 0.0, 0.85), math.pi / 2, (0.2, 0.1, 0.1)))
    # Rotated by 90 deg: 0.1 along x, 0.2 along y.
    assert scene.box_distance(box, (1.0, 0.0, 0.85)) == 0.0
    assert scene.box_distance(box, (1.15, 0.0, 0.85)) == pytest.approx(0.1)
    assert scene.box_distance(box, (1.0, 0.2, 0.85)) == pytest.approx(0.1)
    assert scene.box_distance(box, (1.0, 0.0, 1.0)) == pytest.approx(0.1)
    assert scene.box_distance(box, (1.15, 0.2, 0.85)) == pytest.approx(math.sqrt(0.02))


def test_place_pose_clears_padded_obstacles():
    # A candidate keeps r + obstacle_margin (perception) from obstacle cells,
    # r the circumscribed radius of the padded footprint (object_padding).
    size, padding = (0.08, 0.08, 0.15), 0.01
    object_padding, obstacle_margin = 0.03, 0.02
    r = 0.5 * math.hypot(size[0] + object_padding, size[1] + object_padding)
    box = scene.obstacle_box(obstacle((0.5, 0.0, 0.85), 0.0, (0.1, 0.1, 0.1)), padding)
    held = 0.5 * math.hypot(size[0], size[1])
    center = (0.5 + 0.05 + r + obstacle_margin, 0.0, 0.80 + size[2] / 2)
    assert scene.box_distance(box, center) > held


def test_held_object_is_attached_at_the_grasp():
    size = (0.06, 0.08, 0.12)
    T = geo.object_in_tcp(geo.TOP, size[2], 0.05)
    attached = scene.held_object(
        "aha_held_object_right", "link_ree", ["link_r6", "link_r7r"], scene.BOX, size, T
    )
    assert attached.link_name == "link_ree"
    assert attached.touch_links == ["link_r6", "link_r7r"]
    obj = attached.object
    assert obj.header.frame_id == "link_ree"
    assert obj.operation == CollisionObject.ADD
    assert list(obj.primitives[0].dimensions) == pytest.approx(list(size))
    p = obj.pose.position
    np.testing.assert_allclose([p.x, p.y, p.z], T[:3, 3])
    q = obj.pose.orientation
    np.testing.assert_allclose(
        geo.matrix_from_quaternion(q.x, q.y, q.z, q.w), T[:3, :3], atol=1e-9
    )


def test_boxes_overlap():
    a = scene.Box(scene.OBSTACLE, geo.transform(None, [0.0, 0.0, 0.0]), (0.2, 0.2, 0.2))
    near = scene.Box(scene.OBSTACLE, geo.transform(None, [0.19, 0, 0]), (0.2, 0.2, 0.2))
    far = scene.Box(scene.OBSTACLE, geo.transform(None, [0.21, 0, 0]), (0.2, 0.2, 0.2))
    assert scene.boxes_overlap(a, near)
    assert not scene.boxes_overlap(a, far)
    # Rotated by 45 deg: the corner reaches 0.1 * sqrt(2) along x.
    turned = scene.Box(
        scene.OBSTACLE,
        geo.transform(geo.rot_z(math.pi / 4), [0.23, 0, 0]),
        (0.2, 0.2, 0.2),
    )
    assert scene.boxes_overlap(a, turned)
    turned.pose[0, 3] = 0.25
    assert not scene.boxes_overlap(a, turned)
    above = scene.Box(
        scene.OBSTACLE, geo.transform(None, [0, 0, 0.25]), (0.1, 0.1, 0.1)
    )
    assert not scene.boxes_overlap(a, above)


def corners(center, size):
    box = scene.Box(scene.OBSTACLE, geo.transform(None, center), size)
    return scene.box_corners(box)


def test_clear_box_lowers_the_top_under_the_held_object():
    # Failure 1 of the stress trials: the box of a 0.08 m object merged with the
    # held object above it (top 0.22 m over the 0.75 m table).
    box = scene.obstacle_box(obstacle((0.2, -0.3, 0.86), 0.0, (0.18, 0.09, 0.22)))
    held_bottom = 0.75 + 0.08 + 0.015
    held = corners((0.2, -0.3, held_bottom + 0.06), (0.06, 0.06, 0.12))
    cut = scene.clear_box(box, held, 0.01, 0.02)
    bottom = cut.pose[2, 3] - cut.size[2] / 2
    assert bottom == pytest.approx(0.75)
    assert cut.size[2] == pytest.approx(0.085)
    assert cut.size[:2] == pytest.approx(box.size[:2])
    np.testing.assert_allclose(cut.pose[:2, 3], box.pose[:2, 3])
    assert not scene.boxes_overlap(
        cut, scene.Box(scene.OBSTACLE, geo.transform(None, held.mean(0)), (0.06,) * 3)
    )


def test_clear_box_moves_a_side_back_for_a_small_overlap():
    # Failure 1 (edge 3): a box 5 mm into the held object beside it keeps its
    # height; its side facing the object moves back.
    yaw = math.radians(90)
    box = scene.obstacle_box(
        obstacle((0.311, -0.248, 0.813), yaw, (0.088, 0.089, 0.126))
    )
    held = corners((0.236, -0.25, 0.86), (0.072, 0.06, 0.12))  # x up to 0.272
    cut = scene.clear_box(box, held, 0.01, 0.02)
    assert cut.size[2] == pytest.approx(0.126)
    lower_x = min(scene.box_corners(cut)[:, 0])
    assert lower_x == pytest.approx(0.282)
    # Turned by 90 deg: the 0.089 side lies along base x.
    assert max(scene.box_corners(cut)[:, 0]) == pytest.approx(0.311 + 0.0445)


def test_clear_box_unchanged_or_impossible():
    box = scene.obstacle_box(obstacle((0.5, 0.0, 0.80), 0.0, (0.06, 0.06, 0.10)))
    assert (
        scene.clear_box(box, corners((0.5, 0.0, 1.0), (0.1, 0.1, 0.1)), 0.01, 0.02)
        is box
    )
    # A link reaching down to the surface all around: nothing useful is left.
    around = corners((0.5, 0.0, 0.80), (0.2, 0.2, 0.2))
    assert scene.clear_box(box, around, 0.01, 0.02) is None


def test_allowed_collision_matrix_edits():
    acm = scene.AllowedCollisionMatrix()
    acm.entry_names = ["link_r6", "link_r7r"]
    acm.entry_values = [
        scene.AllowedCollisionEntry(enabled=[False, True]),
        scene.AllowedCollisionEntry(enabled=[True, False]),
    ]
    acm.default_entry_names = ["aha_place_obstacle_9"]
    acm.default_entry_values = [True]
    allowed = scene.acm_allow(acm, "aha_place_obstacle_0", ["link_r6", "link_ree"])
    names = list(allowed.entry_names)
    assert names == ["link_r6", "link_r7r", "aha_place_obstacle_0", "link_ree"]
    matrix = np.array([list(e.enabled) for e in allowed.entry_values])
    assert matrix.shape == (4, 4)
    assert (matrix == matrix.T).all()
    assert matrix[2, 0] and matrix[2, 3] and matrix[0, 1]
    assert not matrix[2, 1] and not matrix[1, 3]
    # The input is unchanged.
    assert list(acm.entry_names) == ["link_r6", "link_r7r"]
    forgotten = scene.acm_forget(
        allowed, lambda name: name.startswith(scene.OBSTACLE_PREFIX)
    )
    assert list(forgotten.entry_names) == ["link_r6", "link_r7r", "link_ree"]
    matrix = np.array([list(e.enabled) for e in forgotten.entry_values])
    assert matrix.tolist() == [
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]
    assert list(forgotten.default_entry_names) == []


def test_obstacle_msg_round_trip():
    box = scene.obstacle_box(obstacle((0.6, 0.1, 0.80), 0.3, (0.1, 0.05, 0.1)), 0.0)
    msg = scene.obstacle_msg(box)
    back = scene.obstacle_box(msg)
    np.testing.assert_allclose(back.pose, box.pose, atol=1e-9)
    assert back.size == pytest.approx(box.size)


def test_cylinder_and_removal_messages():
    prim = scene.primitive(scene.CYLINDER, (0.07, 0.06, 0.12))
    assert prim.type == SolidPrimitive.CYLINDER
    assert list(prim.dimensions) == pytest.approx([0.12, 0.035])
    with pytest.raises(ValueError):
        scene.primitive("sphere", (0.1, 0.1, 0.1))
    assert (
        scene.remove_object("x", "base_footprint").operation == CollisionObject.REMOVE
    )
    detach = scene.detach_object("aha_held_object_left", "link_lee")
    assert detach.link_name == "link_lee"
    assert detach.object.operation == CollisionObject.REMOVE
