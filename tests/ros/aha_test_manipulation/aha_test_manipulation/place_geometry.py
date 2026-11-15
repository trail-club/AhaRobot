"""Object-in-hand convention and place motions for AhaRobot's grippers (ROS free).

Frames (4x4 homogeneous matrices, numpy):
  base    planning frame, z up (base_footprint)
  TCP     link_ree / link_lee: +x approach axis, fingers close along y
  object  origin at the object's center, z up when placed, x along the
          approach direction

The object in the hand is defined once, by the TCP orientation of each grasp
(``tcp_rotation`` with yaw 0) and the object standing upright at the finger
pads in that orientation:
  side  TCP x horizontal, z up; the object center is at the pad center
  top   TCP x down, tilted by ``tilt`` toward +x (KDL / joint_[r|l]5 cannot
        reach exactly pi/2); the object top is ``TOP_GRASP_DEPTH`` above the
        pad center
Placing with the TCP in the same orientation puts the object upright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SIDE = "side"
TOP = "top"
GRASPS = (SIDE, TOP)
APPROACHES = (SIDE, TOP)

# Finger pad center in the TCP frame: the pads span x -0.04..0.01 and
# z -0.025..-0.005 (astra_description finger meshes at the closed position).
GRASP_POINT = np.array([-0.02, 0.0, -0.015])
# Top grasp: object top above the pad center (the palm is at TCP x -0.077).
TOP_GRASP_DEPTH = 0.02
# Top grasp tilt (rad): joint_[r|l]5 = pi/2 - tilt, below its 1.57 limit and
# away from the wrist singularity at pi/2.
DEFAULT_TOP_TILT = 0.05
# Finger travel per side and the inner face position at a joint value:
# joint_[r|l]7r = +w/2 and joint_[r|l]7l = -w/2 leave a gap of w.
FINGER_MAX_OPENING = 0.06
# Palm (link_[r|l]6 mesh) front face along the TCP x axis.
PALM_X = -0.077
# Finger thickness beside the object (y), for footprint checks.
FINGER_WIDTH = 0.015

UP = np.array([0.0, 0.0, 1.0])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def transform(rotation=None, translation=None) -> np.ndarray:
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation
    return T


def inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return transform(R.T, -R.T @ T[:3, 3])


def check_grasp(grasp: str) -> None:
    if grasp not in GRASPS:
        raise ValueError(f"grasp must be one of {GRASPS}, got {grasp!r}")


def tcp_rotation(grasp: str, yaw: float, tilt: float = 0.0) -> np.ndarray:
    """TCP orientation in base for a grasp, with the approach direction ``yaw``."""
    check_grasp(grasp)
    if grasp == SIDE:
        return rot_z(yaw)
    return rot_z(yaw) @ rot_y(math.pi / 2 - tilt)


def object_in_tcp(grasp: str, height: float, tilt: float = 0.0) -> np.ndarray:
    """Nominal object center pose in the TCP frame (T_tcp_object) of a grasp:
    upright, centered on the finger pads (side) or its top TOP_GRASP_DEPTH
    above the pad center (top)."""
    R = tcp_rotation(grasp, 0.0, tilt)
    center = R @ GRASP_POINT
    if grasp == TOP:
        center = center - UP * (height / 2 - TOP_GRASP_DEPTH)
    return inverse(transform(R)) @ transform(None, center)


def object_pose(bottom_center, yaw: float, height: float) -> np.ndarray:
    """Upright object center pose in base from its bottom center."""
    return transform(
        rot_z(yaw), np.asarray(bottom_center, dtype=float) + UP * height / 2
    )


def tcp_for_object(T_base_object: np.ndarray, T_tcp_object: np.ndarray) -> np.ndarray:
    """TCP pose that puts the held object (at T_tcp_object in the hand) at
    T_base_object."""
    return T_base_object @ inverse(T_tcp_object)


def hand_direction(T_base_tcp: np.ndarray, yaw: float) -> np.ndarray:
    """Horizontal unit direction the hand points to (TCP x projected), or
    the direction ``yaw`` when the TCP x is near vertical (top grasp)."""
    x = np.array([T_base_tcp[0, 0], T_base_tcp[1, 0], 0.0])
    norm = float(np.linalg.norm(x))
    if norm < 0.5:
        return np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return x / norm


def finger_position(width: float, margin: float = 0.0) -> float:
    """Finger joint value (per side) leaving ``margin`` to an object of ``width``."""
    return min(max(width / 2 + margin, 0.0), FINGER_MAX_OPENING)


def approach_yaw(target_xy, origin_xy) -> float:
    """Horizontal direction from ``origin_xy`` (e.g. the shoulder) to the target."""
    return math.atan2(target_xy[1] - origin_xy[1], target_xy[0] - origin_xy[0])


def max_object_width(max_opening: float = FINGER_MAX_OPENING, clearance=0.005):
    """Largest y size (between the fingers) the gripper can open around with
    ``clearance`` per side; ``max_opening`` is the finger travel per side."""
    return 2 * (max_opening - clearance)


def max_object_length(grasp: str, clearance: float = 0.005) -> float:
    """Largest x size (approach axis) the hand holds without the palm: the side
    grasp centers the object on the pads, so its back end must stay in front
    of the palm. In the top grasp the palm is above the object's top (no
    limit)."""
    check_grasp(grasp)
    if grasp == TOP:
        return math.inf
    return 2 * (GRASP_POINT[0] - PALM_X - clearance)


def object_fit_error(
    grasp: str,
    size,
    max_opening: float = FINGER_MAX_OPENING,
    clearance: float = 0.005,
) -> str | None:
    """Why an object of ``size`` (x, y, z) does not fit the gripper, or None."""
    width = max_object_width(max_opening, clearance)
    if size[1] > width + 1e-9:
        return (
            f"object y {size[1]:.3f} m is wider than the gripper opens "
            f"({width:.3f} m with {clearance:g} m per side)"
        )
    length = max_object_length(grasp, clearance)
    if size[0] > length + 1e-9:
        return (
            f"object x {size[0]:.3f} m reaches the palm in a {grasp} grasp "
            f"(at most {length:.3f} m)"
        )
    return None


def pose_change(T0: np.ndarray, T1: np.ndarray) -> tuple[float, float]:
    """(translation distance, rotation angle in rad) from T0 to T1."""
    distance = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    R = T0[:3, :3].T @ T1[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))
    return distance, angle


def moved_beyond(T0, T1, max_shift: float, max_turn: float) -> bool:
    """Whether T1 is more than ``max_shift`` (m) or ``max_turn`` (rad) from T0."""
    distance, angle = pose_change(T0, T1)
    return distance > max_shift or angle > max_turn


@dataclass
class MotionParams:
    approach_distance: float = 0.10  # side: horizontal entry length
    lift_margin: float = 0.03  # side: entry height above the place pose
    approach_height: float = 0.10  # top: descent length
    retreat_distance: float = 0.10  # side: back along -x (side grasp) or up
    retreat_lift: float = 0.05  # side grasp, side approach: up after moving back


@dataclass
class PlaceMotion:
    grasp: str
    approach: str
    yaw: float
    object_pose: np.ndarray  # object center at release
    place_pose: np.ndarray  # object bottom center at release (z up, x = yaw)
    tcp_place: np.ndarray
    pre_place: np.ndarray  # joint-space goal before the Cartesian approach
    approach_waypoints: list = field(default_factory=list)  # ends at tcp_place
    retreat_waypoints: list = field(default_factory=list)
    # Straight up, when the retreat above is not: tried if it fails.
    retreat_fallback: list = field(default_factory=list)


def _moved(T: np.ndarray, offset) -> np.ndarray:
    out = T.copy()
    out[:3, 3] = T[:3, 3] + offset
    return out


def place_motion(
    surface_point,
    size,
    yaw: float,
    grasp: str,
    approach: str,
    release_height: float = 0.01,
    tilt: float = 0.0,
    params: MotionParams | None = None,
    hand: np.ndarray | None = None,
) -> PlaceMotion:
    """TCP poses to put an object of ``size`` (x, y, z) down at ``surface_point``
    upright with its x along ``yaw``. ``hand`` is the object's pose in the TCP
    frame (T_tcp_object; None: the grasp's nominal ``object_in_tcp``).

    side approach: pre-place ``approach_distance`` back along the direction
    the hand points (``hand_direction``; the approach direction for the
    nominal grasp) and ``lift_margin`` up, move horizontally, then descend.
    top approach: pre-place ``approach_height`` straight above, then descend.
    Retreat: after a top approach straight up by max(retreat_distance,
    approach_height), back through the corridor of the descent (pulling back
    horizontally hits what stands in front of a spot reached from above);
    after a side approach the side grasp moves back along -x then up (with
    the straight rise as ``retreat_fallback``), the top grasp moves up.
    """
    check_grasp(grasp)
    if approach not in APPROACHES:
        raise ValueError(f"approach must be one of {APPROACHES}, got {approach!r}")
    params = params or MotionParams()
    height = float(size[2])
    bottom = np.asarray(surface_point, dtype=float) + UP * release_height
    T_object = object_pose(bottom, yaw, height)
    if hand is None:
        hand = object_in_tcp(grasp, height, tilt)
    tcp = tcp_for_object(T_object, hand)
    direction = hand_direction(tcp, yaw)

    if approach == SIDE:
        above = _moved(tcp, UP * params.lift_margin)
        pre = _moved(above, -direction * params.approach_distance)
        waypoints = [above, tcp]
    else:
        pre = _moved(tcp, UP * params.approach_height)
        waypoints = [tcp]

    up = [_moved(tcp, UP * max(params.retreat_distance, params.approach_height))]
    fallback = []
    if approach == TOP:
        retreat = up
    elif grasp == SIDE:
        back = _moved(tcp, -direction * params.retreat_distance)
        retreat = [back, _moved(back, UP * params.retreat_lift)]
        fallback = up
    else:
        retreat = [_moved(tcp, UP * params.retreat_distance)]

    return PlaceMotion(
        grasp=grasp,
        approach=approach,
        yaw=yaw,
        object_pose=T_object,
        place_pose=transform(rot_z(yaw), bottom),
        tcp_place=tcp,
        pre_place=pre,
        approach_waypoints=waypoints,
        retreat_waypoints=retreat,
        retreat_fallback=fallback,
    )


def release_for_tcp_height(
    tcp_height: float, surface_z: float, height: float, hand: np.ndarray
) -> float:
    """Release height (object bottom above the surface at ``surface_z``) that
    puts the TCP at ``tcp_height`` (base z) when the object of ``height``,
    held at ``hand`` (T_tcp_object), is placed upright: the TCP is then
    inverse(hand)[2, 3] above the object center at any yaw."""
    return tcp_height - (inverse(hand)[2, 3] + height / 2) - surface_z


def fixed_tcp_release(
    tcp_height: float,
    surface_z: float,
    height: float,
    hand: np.ndarray,
    min_release: float,
    max_release: float,
) -> tuple[float, str | None]:
    """(release height, None) for a TCP fixed at ``tcp_height``
    (``release_for_tcp_height``), or (release height, why) when it is below
    ``min_release`` (the object would be pushed into the surface) or above
    ``max_release`` (dropped from too high)."""
    release = release_for_tcp_height(tcp_height, surface_z, height, hand)
    if min_release - 1e-9 <= release <= max_release + 1e-9:
        return release, None
    return release, (
        f"the TCP at {tcp_height:.3f} m releases {release * 100:.1f} cm above "
        f"the {surface_z:.3f} m surface (allowed {min_release * 100:.1f} to "
        f"{max_release * 100:.1f} cm)"
    )


def descent_column(
    motion: PlaceMotion, size, margin: float, params: MotionParams | None = None
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """(center pose, size) of the box an obstacle must not enter for the
    place: the object footprint grown by ``margin`` (and the fingers in y),
    from the object bottom at release up to its top at the start of the
    descent (``approach_height`` for a top approach, ``lift_margin`` for a
    side approach; the side entry is left to the collision-checked path)."""
    params = params or MotionParams()
    rise = params.approach_height if motion.approach == TOP else params.lift_margin
    height = float(size[2]) + rise
    T = transform(rot_z(motion.yaw), motion.place_pose[:3, 3] + UP * height / 2)
    extent = (
        float(size[0]) + 2 * margin,
        float(size[1]) + 2 * (margin + FINGER_WIDTH),
        height,
    )
    return T, extent


def quaternion_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    m = np.asarray(R, dtype=float)[:3, :3]
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def tilt_from_vertical(R: np.ndarray) -> float:
    """Angle (rad) between the frame's z axis and the world z axis."""
    return math.acos(max(-1.0, min(1.0, float(np.asarray(R)[2, 2]))))
