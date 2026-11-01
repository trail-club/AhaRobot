"""Candidate selection and status of the place server
(aha_test_manipulation.place_selection)."""

from aha_msgs.msg import PlaceCandidate, Plane
from aha_msgs.srv import GetPlaceCandidates
from aha_test_msgs.action import PlaceObject
from moveit_msgs.msg import MoveItErrorCodes

from aha_test_manipulation import place_selection as selection

Result = PlaceObject.Result
Response = GetPlaceCandidates.Response


def candidate(x, y, height):
    msg = PlaceCandidate()
    msg.pose.header.frame_id = "base_footprint"
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.pose.pose.position.z = height
    msg.surface_height = height
    return msg


def response(candidates=(), status=Response.STATUS_OK, success=True, heights=()):
    msg = Response()
    msg.success = success
    msg.status = status
    msg.candidates = list(candidates)
    msg.planes.planes = [Plane(height=h) for h in heights]
    return msg


def test_status_constants_match_perception():
    from_perception = {
        Response.STATUS_NO_SURFACE: Result.STATUS_NO_SURFACE,
        Response.STATUS_NO_SPACE: Result.STATUS_NO_SPACE,
    }
    for status, expected in from_perception.items():
        assert selection.no_candidate_status(response(status=status)) == expected
    failed = response(status=Response.STATUS_ERROR, success=False)
    assert selection.no_candidate_status(failed) == Result.STATUS_ERROR
    assert selection.select_candidates(failed) == ([], Result.STATUS_ERROR)


def test_unrestricted_selection_prefers_the_near_point_candidates():
    table = [candidate(0.8, 0.0, 0.80), candidate(0.5, -0.2, 0.80)]
    near = [candidate(0.45, -0.25, 0.80)]
    assert selection.select_candidates(response(table)) == (table, Result.STATUS_OK)
    assert selection.select_candidates(response(table), near) == (
        near,
        Result.STATUS_OK,
    )
    # Empty near-point results fall back to the response's candidates.
    assert selection.select_candidates(response(table), [])[0] == table


def test_restricted_selection_never_returns_another_height():
    mixed = [candidate(0.5, -0.2, 0.91), candidate(0.5, -0.2, 0.615)]
    chosen, status = selection.select_candidates(response(mixed), None, 0.61, 0.0)
    assert status == Result.STATUS_OK
    assert [c.surface_height for c in chosen] == [0.615]
    # Only the other board has room: no candidate, not success on 0.91.
    other = [candidate(0.5, -0.2, 0.91)]
    chosen, status = selection.select_candidates(response(other), other, 0.61, 0.03)
    assert chosen == [] and status == Result.STATUS_NO_SPACE
    chosen, status = selection.select_candidates(
        response(status=Response.STATUS_NO_SURFACE), None, 0.45, 0.03
    )
    assert chosen == [] and status == Result.STATUS_NO_SURFACE
    # The tolerance (<= 0: 0.03) plus HEIGHT_SLACK bounds the heights kept.
    edge = [candidate(0.5, 0.0, 0.61 + 0.035)]
    assert selection.select_candidates(response(edge), None, 0.61, 0.0)[0] == edge
    assert selection.select_candidates(response(edge), None, 0.61, 0.02)[0] == []


def test_near_point_candidates_keep_their_height_and_sort_by_distance():
    first = response([candidate(0.7, -0.25, 0.80), candidate(0.5, -0.3, 0.81)])
    second = response([candidate(0.46, -0.25, 0.61), candidate(0.6, -0.25, 0.91)])
    failed = response([candidate(0.45, -0.25, 0.40)], success=False)
    chosen = selection.near_point_candidates(
        [(0.80, first), (0.61, second), (0.40, failed)], (0.45, -0.25)
    )
    assert [(c.pose.pose.position.x, c.surface_height) for c in chosen] == [
        (0.46, 0.61),
        (0.5, 0.81),
        (0.7, 0.80),
    ]
    planes = response(heights=(0.912, 0.61, 0.608)).planes.planes
    assert selection.support_heights(planes) == [0.61, 0.91]


def test_reachable_candidates_keep_the_near_order():
    # Perception's order (distance to near 0.95 m): the first ones are out of reach.
    ordered = [candidate(x, -0.25, 0.75) for x in (0.95, 0.85, 0.75, 0.6, 0.5, 0.4)]
    shoulder = (0.05, -0.2)
    checked = []

    def reachable(c):
        checked.append(c.pose.pose.position.x)
        return selection.within_reach(
            (c.pose.pose.position.x, c.pose.pose.position.y), shoulder, 0.75
        )

    chosen = selection.first_reachable(ordered, reachable, 2)
    assert [c.pose.pose.position.x for c in chosen] == [0.75, 0.6]
    assert checked == [0.95, 0.85, 0.75, 0.6]  # lazy: stops at the limit
    assert selection.first_reachable(ordered, lambda _: False, 3) == []


def test_drop_near_blocked_spots():
    spots = [candidate(0.5, -0.2, 0.75), candidate(0.55, -0.2, 0.75)]
    spots.append(candidate(0.5, 0.1, 0.75))
    kept = selection.drop_near(spots, [(0.5, -0.2, 0.75)], 0.085)
    assert [c.pose.pose.position.y for c in kept] == [0.1]
    assert selection.drop_near(spots, [], 0.1) == spots


def test_group_order_keeps_away_from_other_categories():
    # Lemon (near) at y -0.38, a food object (avoid) at y -0.11, same depth.
    near, avoid = (0.68, -0.38), [(0.68, -0.11)]
    between = candidate(0.59, -0.25, 0.91)  # 0.157 from near, 0.166 from avoid
    beside = candidate(0.68, -0.29, 0.91)  # 0.09 from near
    front = candidate(0.59, -0.33, 0.91)  # 0.103 from near
    far_side = candidate(0.59, -0.15, 0.91)  # nearer to the food
    spots = [far_side, between, front, beside]
    assert selection.in_group(between, near, avoid)
    assert not selection.in_group(between, near, avoid, 0.15)
    assert not selection.in_group(far_side, near, avoid)
    ordered = selection.order_for_group(spots, near, avoid)
    assert ordered == [beside, front, between, far_side]
    assert selection.order_for_group(spots, near, avoid, 0.15) == [beside, front]
    # Without avoid points only the distance counts.
    assert selection.order_for_group(spots, near, [])[0] is beside
    # Clearly in the group at ratio 0.6: 'between' (0.157 vs 0.166 to the
    # food) is not, 'beside' (0.09 vs 0.18) and 'front' (0.103 vs 0.238) are.
    assert not selection.in_group(between, near, avoid, 0.0, 0.6)
    assert selection.in_group(beside, near, avoid, 0.0, 0.6)
    assert selection.in_group(front, near, avoid, 0.0, 0.6)
    assert selection.order_for_group(spots, near, avoid, 0.15, 0.0, 0.6) == [
        beside,
        front,
    ]
    # The near point's row first: beside it before in front of it.
    assert selection.in_row(beside, near, 0.04)
    assert not selection.in_row(front, near, 0.04)
    near_front = candidate(0.60, -0.38, 0.91)  # 0.08 in front: nearer than beside
    ordered = selection.order_for_group([near_front, beside], near, avoid, 0.15, 0.04)
    assert ordered == [beside, near_front]
    by_distance = selection.order_for_group([near_front, beside], near, avoid, 0.15)
    assert by_distance == [near_front, beside]
    assert selection.order_for_group([front, between], near, avoid, 0.15, 0.04) == [
        front
    ]


def test_distances_to_the_group_use_its_nearest_member():
    # Green tea (near) at the right end, lemonade beside it, cola goes left.
    tea, lemonade = (0.60, -0.46), (0.60, -0.39)
    spot = candidate(0.60, -0.30, 0.61)  # 0.16 from tea, 0.09 from lemonade
    assert not selection.in_group(spot, tea, [], 0.15)
    assert selection.in_group(spot, tea, [], 0.15, 1.0, [lemonade])
    d, member = selection.nearest_member(spot, tea, [lemonade])
    assert abs(d - 0.09) < 1e-9 and member == lemonade
    assert selection.order_for_group([spot], tea, [], 0.15, 0.04, 0.6, [lemonade]) == [
        spot
    ]
    # In the row of its nearest member.
    assert selection.in_row(spot, member, 0.04)


def test_base_for_spot():
    # Spot beside the lemon at (0.68, -0.29); the right arm places best at
    # (0.59, -0.25) in the base frame: move 0.09 forward and 0.04 right.
    x, y = selection.base_for_spot((0.68, -0.29), (0.59, -0.25))
    assert abs(x - 0.09) < 1e-9 and abs(y + 0.04) < 1e-9


def test_new_status_values_are_distinct():
    values = [
        getattr(Result, name) for name in dir(Result) if name.startswith("STATUS_")
    ]
    assert len(values) == len(set(values))
    assert Result.STATUS_BASE_MOVED == 7
    assert Result.STATUS_INVALID_OBJECT == 8
    assert Result.STATUS_NOT_AT_SPOT == 9
    assert Result().returned is False
    assert Result().suggested_base_pose.header.frame_id == ""


def test_failure_status():
    assert selection.failure_status(False) == Result.STATUS_UNREACHABLE
    assert selection.failure_status(True) == Result.STATUS_EXECUTION_FAILED
    assert MoveItErrorCodes.CONTROL_FAILED in selection.EXECUTION_ERRORS
    assert MoveItErrorCodes.PLANNING_FAILED not in selection.EXECUTION_ERRORS
    assert MoveItErrorCodes.NO_IK_SOLUTION not in selection.EXECUTION_ERRORS
