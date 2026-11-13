"""Right-arm joint values that point the gripper at a target, without ROS.

The right arm (upstream astra_description) is a lift (joint_r1) carrying three
joints about vertical axes (joint_r2 shoulder, joint_r3 elbow, joint_r4 wrist
yaw), then the wrist pitch (joint_r5) and roll (joint_r6). The pointing
direction is the x axis of link_ree:

    yaw   = joint_r2 + joint_r3 - joint_r4   (from base_footprint +x, + = left)
    pitch = -joint_r5                        (+ = up)

The joint_r4 and joint_r5 axes pass through the wrist (the link_r4 origin), so
the wrist position depends on joint_r1..joint_r3 only. link_ree sits WRIST_REACH
ahead of the wrist along the pointing direction and EE_DROP below that line
(with joint_r6 = 0), so the ray along link_ree x passes EE_DROP from the wrist.
The constants come from the URDF; test_pointing.py checks them against it.

All positions are in base_footprint (m), angles in rad.
"""

from dataclasses import dataclass
import math

ARM_JOINTS = ("joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6")
LIFT_JOINT = "joint_r1"

SHOULDER_XY = (-0.0405, -0.133)  # joint_r2 axis
ARM_PLANE_Z = 0.182  # wrist height with joint_r1 = 0 (base_link 0.042 above floor)
UPPER_ARM = 0.29812  # joint_r2 -> joint_r3
FOREARM = 0.265  # joint_r3 -> wrist
UPPER_ARM_YAW = -1.38525  # upper arm yaw at joint_r2 = 0 (-pi/2 + 0.18555)
ELBOW_YAW = 1.3852  # forearm yaw relative to the upper arm at joint_r3 = 0
EE_DROP = 0.063  # link_ree below the wrist line, perpendicular to it
WRIST_REACH = 0.195  # link_ree ahead of the wrist along the pointing direction

LIMITS = {
    "joint_r1": (0.0, 1.2),
    "joint_r2": (-1.57, 1.57),
    "joint_r3": (-1.57, 1.57),
    "joint_r4": (-3.14, 3.14),
    "joint_r5": (-1.57, 1.57),
    "joint_r6": (-1.57, 1.57),
}
MARGIN = 0.05  # rad kept from the joint limits
# Bearings (from the shoulder, + = left) the arm points at without crossing
# the body; turn the base for others.
MIN_BEARING = math.radians(-100.0)
MAX_BEARING = math.radians(60.0)


@dataclass(frozen=True)
class Pointing:
    lift: float  # joint_r1
    arm: tuple  # ARM_JOINTS values
    wrist: tuple  # (x, y, z) of the wrist


def lift_for_height(height):
    """joint_r1 that puts the wrist `height` m above the floor, within limits."""
    low, high = LIMITS[LIFT_JOINT]
    return min(max(height - ARM_PLANE_Z, low), high)


def shoulder_bearing(target):
    return math.atan2(target[1] - SHOULDER_XY[1], target[0] - SHOULDER_XY[0])


def elbow_position(r2):
    upper = UPPER_ARM_YAW + r2
    return (
        SHOULDER_XY[0] + UPPER_ARM * math.cos(upper),
        SHOULDER_XY[1] + UPPER_ARM * math.sin(upper),
    )


def wrist_position(lift, r2, r3):
    fore = UPPER_ARM_YAW + r2 + ELBOW_YAW + r3
    x, y = elbow_position(r2)
    return (
        x + FOREARM * math.cos(fore),
        y + FOREARM * math.sin(fore),
        ARM_PLANE_Z + lift,
    )


def within(name, value, margin=0.0):
    low, high = LIMITS[name]
    return low + margin <= value <= high - margin


def point_at(target, lift):
    """Joint values pointing link_ree's x axis at `target`; ValueError if not.

    The upper arm and forearm stretch towards the target while it is in front
    or to the right; for targets to the left the upper arm stays forward and
    the forearm turns. The wrist then aims exactly, including EE_DROP.
    """
    bearing = shoulder_bearing(target)
    if not MIN_BEARING <= bearing <= MAX_BEARING:
        raise ValueError(
            f"target bearing {math.degrees(bearing):+.0f} deg is outside "
            f"{math.degrees(MIN_BEARING):+.0f}..{math.degrees(MAX_BEARING):+.0f} deg"
        )
    if not within(LIFT_JOINT, lift):
        raise ValueError(f"lift {lift:.3f} m is outside {LIMITS[LIFT_JOINT]}")
    # Straight arm: forearm in line with the upper arm, both along the bearing.
    r2 = bearing - UPPER_ARM_YAW
    r3 = -ELBOW_YAW
    high = LIMITS["joint_r2"][1] - MARGIN
    if r2 > high:
        r2 = high
        elbow = elbow_position(r2)
        fore = math.atan2(target[1] - elbow[1], target[0] - elbow[0])
        r3 = fore - (UPPER_ARM_YAW + r2) - ELBOW_YAW
    wrist = wrist_position(lift, r2, r3)
    dx, dy, dz = (t - w for t, w in zip(target, wrist))
    horizontal = math.hypot(dx, dy)
    distance = math.hypot(horizontal, dz)
    if distance <= EE_DROP + WRIST_REACH:
        raise ValueError(f"target is {distance:.2f} m from the wrist, too close")
    yaw = math.atan2(dy, dx)
    # The ray runs EE_DROP below the wrist line: (target - wrist) . n = EE_DROP
    # with n = (sin p, -cos p) in the (horizontal, up) plane, p the pitch.
    pitch = math.atan2(dz, horizontal) + math.asin(EE_DROP / distance)
    r4 = math.remainder(r2 + r3 - yaw, math.tau)
    joints = {"joint_r2": r2, "joint_r3": r3, "joint_r4": r4, "joint_r5": -pitch}
    joints["joint_r6"] = 0.0
    for name, value in joints.items():
        if not within(name, value):
            raise ValueError(f"{name} {value:+.3f} is outside {LIMITS[name]}")
    return Pointing(
        lift=lift, arm=tuple(joints[name] for name in ARM_JOINTS), wrist=wrist
    )
