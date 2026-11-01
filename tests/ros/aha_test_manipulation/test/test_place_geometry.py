"""Object-in-hand convention and place poses (aha_test_manipulation.place_geometry)."""

import math

import numpy as np
import pytest

from aha_test_manipulation import place_geometry as geo

SIZE = (0.06, 0.08, 0.12)


def test_side_tcp_is_horizontal_with_z_up():
    R = geo.tcp_rotation(geo.SIDE, math.radians(30))
    np.testing.assert_allclose(
        R[:, 0], [math.cos(math.radians(30)), math.sin(math.radians(30)), 0]
    )
    np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-12)


def test_top_tcp_points_down_tilted_toward_yaw():
    tilt = 0.05
    R = geo.tcp_rotation(geo.TOP, math.pi / 2, tilt)
    # x down, leaning toward +y (the yaw direction) by the tilt
    np.testing.assert_allclose(
        R[:, 0], [0, math.sin(tilt), -math.cos(tilt)], atol=1e-12
    )
    # fingers keep closing horizontally
    assert abs(R[2, 1]) < 1e-12


def test_side_object_center_is_the_pad_center_and_upright():
    T = geo.object_in_tcp(geo.SIDE, SIZE[2])
    np.testing.assert_allclose(T[:3, 3], geo.GRASP_POINT)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)


@pytest.mark.parametrize("tilt", [0.0, 0.05, 0.2])
def test_top_object_is_upright_below_the_pads(tilt):
    yaw = 0.7
    tcp = geo.transform(geo.tcp_rotation(geo.TOP, yaw, tilt), [0.5, -0.2, 1.0])
    obj = tcp @ geo.object_in_tcp(geo.TOP, SIZE[2], tilt)
    assert geo.tilt_from_vertical(obj[:3, :3]) < 1e-9
    pad = tcp[:3, :3] @ geo.GRASP_POINT + tcp[:3, 3]
    top = obj[:3, 3] + obj[:3, 2] * SIZE[2] / 2
    assert top[2] == pytest.approx(pad[2] + geo.TOP_GRASP_DEPTH)
    np.testing.assert_allclose(top[:2], pad[:2], atol=1e-12)
    # object x stays along the approach direction
    assert math.atan2(obj[1, 0], obj[0, 0]) == pytest.approx(yaw)


@pytest.mark.parametrize("grasp", geo.GRASPS)
def test_tcp_for_object_inverts_object_in_tcp(grasp):
    obj = geo.object_pose([0.4, 0.1, 0.8], 0.3, SIZE[2])
    hand = geo.object_in_tcp(grasp, SIZE[2], 0.05)
    tcp = geo.tcp_for_object(obj, hand)
    np.testing.assert_allclose(tcp @ hand, obj, atol=1e-12)


def test_object_lower_in_the_hand_raises_the_place_pose():
    # Slid 1.5 cm down and turned 10 deg in the side grasp.
    nominal = geo.object_in_tcp(geo.SIDE, SIZE[2])
    slid = nominal @ geo.transform(geo.rot_z(math.radians(10)), [0, 0, 0])
    slid[2, 3] -= 0.015
    yaw = math.radians(-20)
    a = geo.place_motion([0.5, -0.2, 0.8], SIZE, yaw, geo.SIDE, geo.SIDE)
    b = geo.place_motion([0.5, -0.2, 0.8], SIZE, yaw, geo.SIDE, geo.SIDE, hand=slid)
    # The object ends at the same pose, the hand 1.5 cm higher.
    np.testing.assert_allclose(b.tcp_place @ slid, a.object_pose, atol=1e-12)
    assert b.tcp_place[2, 3] == pytest.approx(a.tcp_place[2, 3] + 0.015)
    # The hand enters along its own x (turned against the object).
    entry = b.tcp_place[:3, 3] - b.approach_waypoints[0][:3, 3]
    back = b.approach_waypoints[0][:3, 3] - b.pre_place[:3, 3]
    assert entry[:2] == pytest.approx([0.0, 0.0]) and entry[2] < 0
    heading = math.atan2(back[1], back[0])
    assert heading == pytest.approx(yaw - math.radians(10))


def test_side_place_motion():
    params = geo.MotionParams(
        approach_distance=0.1,
        lift_margin=0.03,
        retreat_distance=0.08,
        retreat_lift=0.05,
    )
    yaw = math.radians(-20)
    motion = geo.place_motion(
        [0.5, -0.2, 0.8], SIZE, yaw, geo.SIDE, geo.SIDE, 0.01, 0.0, params
    )
    direction = np.array([math.cos(yaw), math.sin(yaw), 0])
    np.testing.assert_allclose(motion.place_pose[:3, 3], [0.5, -0.2, 0.81])
    np.testing.assert_allclose(
        motion.object_pose[:3, 3], [0.5, -0.2, 0.81 + SIZE[2] / 2]
    )
    held = motion.tcp_place @ geo.object_in_tcp(geo.SIDE, SIZE[2])
    np.testing.assert_allclose(held, motion.object_pose, atol=1e-12)
    p = motion.tcp_place[:3, 3]
    np.testing.assert_allclose(
        motion.pre_place[:3, 3], p + [0, 0, 0.03] - direction * 0.1
    )
    np.testing.assert_allclose(motion.approach_waypoints[0][:3, 3], p + [0, 0, 0.03])
    assert motion.approach_waypoints[-1] is motion.tcp_place
    np.testing.assert_allclose(motion.retreat_waypoints[0][:3, 3], p - direction * 0.08)
    np.testing.assert_allclose(
        motion.retreat_waypoints[1][:3, 3], p - direction * 0.08 + [0, 0, 0.05]
    )
    for T in (motion.pre_place, *motion.approach_waypoints, *motion.retreat_waypoints):
        np.testing.assert_allclose(T[:3, :3], motion.tcp_place[:3, :3])


def test_top_place_motion_descends_and_lifts():
    params = geo.MotionParams(approach_height=0.12, retreat_distance=0.09)
    motion = geo.place_motion(
        [0.5, 0.0, 0.75], SIZE, 0.0, geo.TOP, geo.TOP, 0.02, 0.05, params
    )
    p = motion.tcp_place[:3, 3]
    np.testing.assert_allclose(motion.pre_place[:3, 3], p + [0, 0, 0.12])
    assert len(motion.approach_waypoints) == 1
    # Back up the descent: max(retreat_distance, approach_height).
    (retreat,) = motion.retreat_waypoints
    np.testing.assert_allclose(retreat[:3, 3], p + [0, 0, 0.12])
    bottom = motion.object_pose[:3, 3] - [0, 0, SIZE[2] / 2]
    np.testing.assert_allclose(bottom, [0.5, 0.0, 0.77])
    # fingertips (TCP) stay above the surface
    assert p[2] > 0.75 + 0.02


def test_side_grasp_with_top_approach_retreats_straight_up():
    params = geo.MotionParams(approach_height=0.10, retreat_distance=0.08)
    motion = geo.place_motion(
        [0.5, 0.0, 0.75], SIZE, 0.3, geo.SIDE, geo.TOP, params=params
    )
    assert motion.approach == geo.TOP
    (retreat,) = motion.retreat_waypoints
    np.testing.assert_allclose(retreat[:3, 3], motion.tcp_place[:3, 3] + [0, 0, 0.10])
    np.testing.assert_allclose(retreat[:3, :3], motion.tcp_place[:3, :3])
    assert motion.retreat_fallback == []
    # After a side approach the same rise is the fallback of the side retreat.
    side = geo.place_motion(
        [0.5, 0.0, 0.75], SIZE, 0.3, geo.SIDE, geo.SIDE, params=params
    )
    assert len(side.retreat_waypoints) == 2
    (up,) = side.retreat_fallback
    np.testing.assert_allclose(up[:3, 3], side.tcp_place[:3, 3] + [0, 0, 0.10])


def test_top_grasp_with_side_approach_lifts():
    params = geo.MotionParams(retreat_distance=0.07)
    motion = geo.place_motion(
        [0.5, 0.0, 0.75], SIZE, 0.0, geo.TOP, geo.SIDE, 0.01, 0.05, params
    )
    (retreat,) = motion.retreat_waypoints
    np.testing.assert_allclose(retreat[:3, 3], motion.tcp_place[:3, 3] + [0, 0, 0.07])


def test_object_fit_against_the_gripper():
    # 0.06 m travel per side, 5 mm clearance: at most 0.11 m between the fingers.
    assert geo.max_object_width() == pytest.approx(0.11)
    assert geo.object_fit_error(geo.SIDE, (0.06, 0.11, 0.12)) is None
    assert "wider" in geo.object_fit_error(geo.SIDE, (0.08, 0.14, 0.10))
    assert "wider" in geo.object_fit_error(geo.TOP, (0.08, 0.14, 0.10))
    # Side grasp: the back end at pad center - x / 2 stays 5 mm before the palm.
    limit = 2 * (geo.GRASP_POINT[0] - geo.PALM_X - 0.005)
    assert limit == pytest.approx(0.104)
    assert geo.object_fit_error(geo.SIDE, (0.10, 0.10, 0.25)) is None
    assert "palm" in geo.object_fit_error(geo.SIDE, (0.20, 0.10, 0.10))
    # The top grasp keeps the palm above the object.
    assert geo.object_fit_error(geo.TOP, (0.20, 0.10, 0.10)) is None
    assert geo.max_object_length(geo.TOP) == math.inf
    assert geo.object_fit_error(geo.SIDE, (0.06, 0.10, 0.12), 0.05, 0.0) is None
    assert geo.object_fit_error(geo.SIDE, (0.06, 0.10, 0.12), 0.05, 0.01)


def test_pose_change():
    T0 = geo.transform(geo.rot_z(0.2), [1.0, 2.0, 0.0])
    T1 = geo.transform(geo.rot_z(0.2 + math.radians(1.5)), [1.006, 2.008, 0.0])
    distance, angle = geo.pose_change(T0, T1)
    assert distance == pytest.approx(0.01)
    assert angle == pytest.approx(math.radians(1.5))
    assert geo.pose_change(T0, T0) == pytest.approx((0.0, 0.0), abs=1e-7)
    # Base check of the place server: 1 cm / 1 deg.
    turn = math.radians(1.0)
    assert geo.moved_beyond(T0, T1, 0.01, turn)  # 1.5 deg
    assert not geo.moved_beyond(T0, T1, 0.02, math.radians(2.0))
    shifted = geo.transform(geo.rot_z(0.2), [1.011, 2.0, 0.0])
    assert geo.moved_beyond(T0, shifted, 0.01, turn)
    assert not geo.moved_beyond(
        T0, geo.transform(geo.rot_z(0.2), [1.009, 2, 0]), 0.01, turn
    )


@pytest.mark.parametrize("approach", [geo.SIDE, geo.TOP])
def test_descent_column(approach):
    params = geo.MotionParams(approach_height=0.10, lift_margin=0.03)
    motion = geo.place_motion(
        [0.5, -0.2, 0.75], SIZE, 0.4, geo.SIDE, approach, 0.01, params=params
    )
    T, size = geo.descent_column(motion, SIZE, 0.01, params)
    rise = 0.10 if approach == geo.TOP else 0.03
    assert size == pytest.approx(
        (SIZE[0] + 0.02, SIZE[1] + 0.02 + 2 * geo.FINGER_WIDTH, SIZE[2] + rise)
    )
    np.testing.assert_allclose(T[:3, :3], geo.rot_z(0.4), atol=1e-12)
    bottom = T[:3, 3] - [0, 0, size[2] / 2]
    np.testing.assert_allclose(bottom, motion.place_pose[:3, 3], atol=1e-12)


def test_invalid_grasp_and_approach():
    with pytest.raises(ValueError):
        geo.tcp_rotation("front", 0.0)
    with pytest.raises(ValueError):
        geo.place_motion([0, 0, 0], SIZE, 0.0, geo.SIDE, "auto")


def test_finger_position_is_clamped():
    assert geo.finger_position(0.06, 0.002) == pytest.approx(0.032)
    assert geo.finger_position(0.5) == geo.FINGER_MAX_OPENING
    assert geo.finger_position(-1.0) == 0.0


def test_approach_yaw():
    assert geo.approach_yaw([1.0, 1.0], [0.0, 0.0]) == pytest.approx(math.pi / 4)


@pytest.mark.parametrize("seed", range(5))
def test_quaternion_round_trip(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    R = geo.matrix_from_quaternion(*q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    q2 = np.array(geo.quaternion_from_matrix(R))
    assert abs(abs(np.dot(q, q2)) - 1.0) < 1e-9
