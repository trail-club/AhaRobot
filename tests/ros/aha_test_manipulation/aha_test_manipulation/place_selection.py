"""Candidate selection and result status of the place server.

Works on aha_msgs/GetPlaceCandidates responses and PlaceCandidate messages
(attribute access only) and returns aha_test_msgs/action/PlaceObject STATUS_*
values; PlaceError carries one.
"""

from __future__ import annotations

import math

from aha_msgs.srv import GetPlaceCandidates
from aha_test_msgs.action import PlaceObject
from moveit_msgs.msg import MoveItErrorCodes

DEFAULT_HEIGHT_TOLERANCE = 0.03  # same default as perception
# Perception filters supports by the plane centroid height; candidate heights
# (points on the plane) may differ by this much more.
HEIGHT_SLACK = 0.01
SAME_HEIGHT = 0.03  # m, a re-query near a support height keeps its candidates

Result = PlaceObject.Result
Response = GetPlaceCandidates.Response


class PlaceError(Exception):
    """A place goal failure with its PlaceObject status."""

    def __init__(self, message, status=Result.STATUS_ERROR):
        super().__init__(message)
        self.status = status


# MoveIt error codes after the arm started moving.
EXECUTION_ERRORS = frozenset(
    {
        MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE,
        MoveItErrorCodes.CONTROL_FAILED,
        MoveItErrorCodes.TIMED_OUT,
        MoveItErrorCodes.PREEMPTED,
    }
)


def height_tolerance(value: float) -> float:
    return float(value) if value > 0.0 else DEFAULT_HEIGHT_TOLERANCE


def support_heights(planes) -> list[float]:
    """Sorted support heights (cm resolution) of aha_msgs/Plane messages."""
    return sorted({round(plane.height, 2) for plane in planes})


def near_point_candidates(queries, point_xy) -> list:
    """Candidates of per-height re-queries, sorted by horizontal distance to ``point_xy``.

    ``queries`` are (support height, response of a request with ``near`` at
    that height); only candidates at that height are kept.
    """
    candidates = []
    for height, response in queries:
        if response.success:
            candidates += [
                c
                for c in response.candidates
                if abs(c.surface_height - height) < SAME_HEIGHT
            ]
    candidates.sort(
        key=lambda c: math.hypot(
            c.pose.pose.position.x - point_xy[0], c.pose.pose.position.y - point_xy[1]
        )
    )
    return candidates


def no_candidate_status(response) -> int:
    """PlaceObject status when a GetPlaceCandidates response leaves nothing to try."""
    if not response.success:
        return Result.STATUS_ERROR
    if response.status == Response.STATUS_NO_SURFACE:
        return Result.STATUS_NO_SURFACE
    return Result.STATUS_NO_SPACE


def select_candidates(
    response, near_candidates=None, surface_height=None, tolerance=0.0
):
    """(candidates to try, status). The status is STATUS_OK with candidates.

    ``near_candidates`` (from ``near_point_candidates``) replace the response's
    candidates when not empty. With ``surface_height`` only candidates within
    the tolerance (+ HEIGHT_SLACK) of it remain, whatever perception returned.
    """
    if not response.success:
        return [], Result.STATUS_ERROR
    candidates = list(near_candidates or response.candidates)
    if surface_height is not None:
        tol = height_tolerance(tolerance) + HEIGHT_SLACK
        candidates = [
            c for c in candidates if abs(c.surface_height - surface_height) <= tol
        ]
    if candidates:
        return candidates, Result.STATUS_OK
    return [], no_candidate_status(response)


def within_reach(point, shoulder_xy, reach: float) -> bool:
    """Whether ``point`` is within ``reach`` (horizontal) of the shoulder."""
    return math.hypot(point[0] - shoulder_xy[0], point[1] - shoulder_xy[1]) <= reach


def first_reachable(candidates, reachable, limit: int) -> list:
    """The first ``limit`` candidates, in their order, for which
    ``reachable(candidate)`` is true (checked lazily, in order)."""
    out = []
    for candidate in candidates:
        if len(out) >= limit:
            break
        if reachable(candidate):
            out.append(candidate)
    return out


def horizontal_distance(candidate, point) -> float:
    p = candidate.pose.pose.position
    return math.hypot(p.x - point[0], p.y - point[1])


def drop_near(candidates, points, distance: float) -> list:
    """Candidates farther than ``distance`` (horizontal) from all ``points``."""
    return [
        c
        for c in candidates
        if all(horizontal_distance(c, p) > distance for p in points)
    ]


def nearest_member(candidate, near_xy, group_xy=()):
    """(distance, point) of the nearest of ``near_xy`` and ``group_xy``."""
    return min(
        ((horizontal_distance(candidate, p), p) for p in (near_xy, *group_xy)),
        key=lambda item: item[0],
    )


def in_group(
    candidate,
    near_xy,
    avoid_xy,
    max_distance: float = 0.0,
    ratio: float = 1.0,
    group_xy=(),
) -> bool:
    """Whether the spot is clearly with the group of ``near_xy`` (and the
    other members ``group_xy``): its distance to the nearest member below
    ``ratio`` times the distance to every avoid point (a spot about halfway
    between two groups belongs to neither for a referee), and with
    ``max_distance`` > 0 within it of that member."""
    d, _ = nearest_member(candidate, near_xy, group_xy)
    if max_distance > 0.0 and d > max_distance:
        return False
    return all(d < ratio * horizontal_distance(candidate, a) for a in avoid_xy)


def in_row(candidate, near_xy, row_tolerance: float) -> bool:
    """Whether the spot is beside ``near_xy`` (within ``row_tolerance`` of it
    along x, the base's forward axis) rather than in front of or behind it."""
    return abs(candidate.pose.pose.position.x - near_xy[0]) <= row_tolerance


def order_for_group(
    candidates,
    near_xy,
    avoid_xy,
    max_distance: float = 0.0,
    row_tolerance: float = 0.0,
    ratio: float = 1.0,
    group_xy=(),
) -> list:
    """Spots of the near point's group (``in_group`` with ``ratio`` and the
    members ``group_xy``) first, with ``row_tolerance`` > 0 those in the row
    of their nearest member (``in_row``: a row of a group keeps the objects
    behind visible and leaves room in front for more) before the others,
    each part nearest first; with ``max_distance`` > 0 only the group's
    spots, else the others follow (nearest to near first)."""

    def key(c):
        return horizontal_distance(c, near_xy)

    def group_key(c):
        d, member = nearest_member(c, near_xy, group_xy)
        beside = row_tolerance > 0.0 and in_row(c, member, row_tolerance)
        return (not beside, d)

    inside = [
        c
        for c in candidates
        if in_group(c, near_xy, avoid_xy, max_distance, ratio, group_xy)
    ]
    inside.sort(key=group_key)
    if max_distance > 0.0:
        return inside
    taken = {id(c) for c in inside}
    rest = sorted((c for c in candidates if id(c) not in taken), key=key)
    return inside + rest


def base_for_spot(spot_xy, spot_offset_xy) -> tuple[float, float]:
    """Base position (same frame and heading as now) that puts the spot at
    ``spot_offset_xy`` in the base frame."""
    return (
        float(spot_xy[0] - spot_offset_xy[0]),
        float(spot_xy[1] - spot_offset_xy[1]),
    )


def failure_status(execution_failed: bool) -> int:
    """Status when every candidate failed before the release."""
    if execution_failed:
        return Result.STATUS_EXECUTION_FAILED
    return Result.STATUS_UNREACHABLE
