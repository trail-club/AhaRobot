"""MoveIt request builders shared by place_server.py and the sim helpers."""

import xml.etree.ElementTree as ET

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint


def named_states(srdf_xml: str) -> dict:
    """{(group, state name): {joint: value}} from the SRDF's group_state elements."""
    states = {}
    for state in ET.fromstring(srdf_xml).findall("group_state"):
        joints = {j.get("name"): float(j.get("value")) for j in state.findall("joint")}
        states[(state.get("group"), state.get("name"))] = joints
    return states


def joint_goal(
    group: str,
    joints: dict,
    planning_time: float,
    scaling: float,
    tolerance: float = 1e-3,
) -> MoveGroup.Goal:
    """MoveGroup goal that plans from the current state to ``joints`` and executes."""
    goal = MoveGroup.Goal()
    request = goal.request
    request.group_name = group
    request.num_planning_attempts = 1
    request.allowed_planning_time = float(planning_time)
    request.max_velocity_scaling_factor = float(scaling)
    request.max_acceleration_scaling_factor = float(scaling)
    request.start_state.is_diff = True
    constraints = Constraints()
    for name, value in joints.items():
        constraint = JointConstraint()
        constraint.joint_name = name
        constraint.position = float(value)
        constraint.tolerance_above = tolerance
        constraint.tolerance_below = tolerance
        constraint.weight = 1.0
        constraints.joint_constraints.append(constraint)
    request.goal_constraints = [constraints]
    goal.planning_options.plan_only = False
    goal.planning_options.planning_scene_diff.is_diff = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    return goal
