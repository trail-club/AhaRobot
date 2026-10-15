"""Robot model given to Gazebo physics only; ROS keeps the original URDF.

Gazebo's DART engine applies only the limit constraint while a joint is
outside its position limit by any amount: the velocity command that
gz_ros2_control sends for a position command is then ignored, and the joint
stays at the limit for good. The model keeps joints inside:

- Physics limits LIMIT_MARGIN outside the URDF range. MoveIt, the
  controllers and the panels command at most the URDF limit, where a joint
  settles about 1e-14 outside it (the lift at 0 m, a finger at 0 m). A joint
  overshooting a fast move can reach the physics limit (joint_r4 at 0.005
  plus 9e-12), so MoveIt's state input accepts up to LIMIT_TOLERANCE.
- A large holding force on the lifts. When the robot lands after the spawn
  (z 0.10 m in the RCJO arena), stopping the arm on a lift takes about 3 kN.
  With the URDF effort (400 N) the lifts slid about 5 mm below 0 m and stayed
  there until lift_controller started.
"""

import xml.etree.ElementTree as ET

LIMIT_MARGIN = 0.005  # m for prismatic, rad for revolute joints
LIMIT_TOLERANCE = LIMIT_MARGIN + 1e-6  # physics limit with its roundoff
LIFT_JOINTS = ("joint_l1", "joint_r1")
LIFT_EFFORT = 10000.0  # N


def gazebo_model(urdf):
    """URDF string for Gazebo with widened limits and holding lifts."""
    root = ET.fromstring(urdf)
    # Top-level joints only; ros2_control <joint> entries have no <limit>.
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") not in ("revolute", "prismatic") or limit is None:
            continue
        limit.set("lower", repr(float(limit.get("lower", 0.0)) - LIMIT_MARGIN))
        limit.set("upper", repr(float(limit.get("upper", 0.0)) + LIMIT_MARGIN))
        if joint.get("name") in LIFT_JOINTS:
            limit.set("effort", repr(LIFT_EFFORT))
    return ET.tostring(root, encoding="unicode")
