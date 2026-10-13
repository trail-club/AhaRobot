"""Robot model given to Gazebo physics only; ROS keeps the original URDF.

Gazebo's DART engine applies only the limit constraint while a joint is
outside its position limit by any amount: the velocity command that
gz_ros2_control sends for a position command is then ignored, and the joint
stays at the limit for good. Two adjustments keep joints inside:

- Physics limits PHYSICS_LIMIT_MARGIN outside the URDF range. MoveIt, the
  controllers and the panels command at most the URDF limit (the lift at
  0 m, a closed finger at 0 m), where a joint settles about 1e-14 outside it.
- Coulomb friction on the lifts. The controllers start a few seconds after
  the spawn; until then the unactuated lifts would fall onto the physics
  limit under the arm's weight (about 22 N each).
"""

import xml.etree.ElementTree as ET

PHYSICS_LIMIT_MARGIN = 0.005  # m for prismatic, rad for revolute joints
LIFT_JOINTS = ("joint_l1", "joint_r1")
# Above the arm's weight on each lift, well below the lift effort limit (400 N).
LIFT_FRICTION = 60.0  # N


def gazebo_model(urdf, margin=PHYSICS_LIMIT_MARGIN, lift_friction=LIFT_FRICTION):
    """URDF string for Gazebo with widened limits and friction on the lifts."""
    root = ET.fromstring(urdf)
    # Top-level joints only; ros2_control <joint> entries have no <limit>.
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") not in ("revolute", "prismatic") or limit is None:
            continue
        limit.set("lower", repr(float(limit.get("lower", 0.0)) - margin))
        limit.set("upper", repr(float(limit.get("upper", 0.0)) + margin))
        if joint.get("name") in LIFT_JOINTS:
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("friction", repr(float(lift_friction)))
    return ET.tostring(root, encoding="unicode")
