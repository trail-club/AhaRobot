import math

import numpy as np
import pytest
from conftest import SHELF

from aha_test_orchestration.viewpoint import (
    ViewOptions,
    front_face,
    level_angles,
    observe_pose,
    required_distance,
    view_points,
)

CAMERA = 1.426  # m, head camera height of the simulated robot
FORWARD = 0.103  # m, camera ahead of base_footprint


def test_required_distance_under_the_board_above():
    heights = (0.02, 0.31, 0.61, 0.91, 1.20)
    # Level 3 (0.61) under the board bottom 0.89, depth 0.4.
    need = required_distance(heights, 0.02, 0.4, CAMERA, [3])
    assert math.isclose(need, (CAMERA - 0.89) * 0.4 / (0.89 - 0.61))
    # The lower level 2 needs more; the top level and level 4 need little.
    assert required_distance(heights, 0.02, 0.4, CAMERA, [2]) > need
    assert required_distance(heights, 0.02, 0.4, CAMERA, [5]) == 0.0
    assert required_distance(heights, 0.02, 0.4, CAMERA, [4]) < 0.5
    # A camera below the board above sees under it from anywhere.
    assert required_distance(heights, 0.02, 0.4, 0.8, [3]) == 0.0


def test_front_face_is_the_side_the_robot_is_in_front_of():
    # The shelf (yaw 0) opens towards -x, where the robot stands.
    normal, half = front_face(SHELF, (SHELF.x - 1.0, SHELF.y))
    assert np.allclose(normal, (-1.0, 0.0)) and half == SHELF.depth / 2
    normal, half = front_face(SHELF, (SHELF.x, SHELF.y + 1.0))
    assert np.allclose(normal, (0.0, 1.0)) and half == SHELF.width / 2


def test_observe_pose_backs_straight_off_and_views_cover_the_levels():
    options = ViewOptions()
    # The robot facing the open side, its right arm at the shelf center.
    start = (SHELF.x - SHELF.depth / 2 - 0.5, SHELF.y + 0.2, 0.0)
    levels = [2, 3, 4, 5]
    pose, camera, depth = observe_pose(SHELF, start, FORWARD, CAMERA, levels, options)
    need = required_distance(SHELF.heights, SHELF.board_thickness, depth, CAMERA, [2])
    assert math.isclose(camera, min(need + options.margin, options.max_distance))
    # Same lateral offset and heading, further back by the camera distance.
    assert math.isclose(pose[1], start[1], abs_tol=1e-6)
    assert math.isclose(pose[2], start[2])
    front = SHELF.x - SHELF.depth / 2
    assert math.isclose(front - pose[0], camera + FORWARD, abs_tol=1e-6)
    looks = view_points(SHELF, camera, depth, CAMERA, levels, options)
    assert sorted(n for _, numbers in looks for n in numbers) == levels
    assert len(looks) < len(levels)
    span = options.vertical_fov - 2 * options.fov_margin
    for point, numbers in looks:
        angles = [
            level_angles(SHELF.heights[n - 1], depth, camera, CAMERA, 0.2)
            for n in numbers
        ]
        low = min(a[0] for a in angles)
        high = max(a[1] for a in angles)
        assert high - low <= span + 1e-9
        center = camera + depth / 2
        aim = math.atan2(CAMERA - point[2], center)
        assert low <= aim <= high
        assert point[:2] == pytest.approx((SHELF.x, SHELF.y))


def test_observe_distance_is_bounded():
    options = ViewOptions(max_distance=0.6)
    start = (SHELF.x - 0.7, SHELF.y, 0.0)
    _, camera, _ = observe_pose(SHELF, start, FORWARD, CAMERA, [2, 3], options)
    assert camera == 0.6
