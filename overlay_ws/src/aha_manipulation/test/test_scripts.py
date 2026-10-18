"""eef_goal_monitor.py and joint_state_bounds.py helpers (no ROS transport)."""

import importlib.util
import math
from pathlib import Path

import yaml

from geometry_msgs.msg import Pose
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

SHARE = Path(__file__).resolve().parents[1]
SCRIPTS = SHARE / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monitor = load("eef_goal_monitor")
bounds = load("joint_state_bounds")


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    p.orientation.z = math.sin(yaw / 2)
    p.orientation.w = math.cos(yaw / 2)
    return p


def test_final_state_overrides_start_with_last_waypoint():
    display = DisplayTrajectory()
    display.trajectory_start.joint_state.name = ["joint_r1", "joint_r2", "joint_l1"]
    display.trajectory_start.joint_state.position = [0.0, 0.0, 0.3]
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = ["joint_r1", "joint_r2"]
    trajectory.joint_trajectory.points = [
        JointTrajectoryPoint(positions=[0.1, 0.2]),
        JointTrajectoryPoint(positions=[0.5, -1.0]),
    ]
    display.trajectory = [RobotTrajectory(), trajectory]
    state = monitor.final_state(display)
    assert dict(zip(state.name, state.position)) == {
        "joint_r1": 0.5,
        "joint_r2": -1.0,
        "joint_l1": 0.3,
    }


def test_final_state_keeps_joints_of_earlier_segments():
    display = DisplayTrajectory()
    display.trajectory_start.joint_state.name = ["joint_r1", "joint_r2"]
    display.trajectory_start.joint_state.position = [0.0, 0.0]
    lift, arm = RobotTrajectory(), RobotTrajectory()
    lift.joint_trajectory.joint_names = ["joint_r1"]
    lift.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.4])]
    arm.joint_trajectory.joint_names = ["joint_r2"]
    arm.joint_trajectory.points = [JointTrajectoryPoint(positions=[-0.7])]
    display.trajectory = [lift, arm]
    state = monitor.final_state(display)
    assert dict(zip(state.name, state.position)) == {"joint_r1": 0.4, "joint_r2": -0.7}


def test_final_state_of_empty_plan_is_none():
    assert monitor.final_state(DisplayTrajectory()) is None
    display = DisplayTrajectory(trajectory=[RobotTrajectory()])
    assert monitor.final_state(display) is None


def test_pose_error_distance_and_angle():
    distance, angle = monitor.pose_error(pose(0.1, 0.2, 0.3), pose(0.1, 0.2, 0.3))
    assert distance == 0.0 and angle == 0.0
    distance, angle = monitor.pose_error(pose(z=0.2), pose(x=0.3, z=0.6, yaw=0.5))
    assert math.isclose(distance, 0.5)
    assert math.isclose(angle, 0.5)
    # q and -q are the same rotation.
    flipped = pose(yaw=0.5)
    for axis in ("x", "y", "z", "w"):
        setattr(flipped.orientation, axis, -getattr(flipped.orientation, axis))
    assert math.isclose(
        monitor.pose_error(pose(yaw=0.5), flipped)[1], 0.0, abs_tol=1e-6
    )


def test_error_text_has_no_spaces():
    text = monitor.describe_error(0.0123, math.radians(4.56))
    assert text == "err=12.3mm/4.6deg"
    assert " " not in text


def test_markers_place_axes_at_goal_and_line_from_current():
    view = monitor.EefGoalMonitor.__new__(monitor.EefGoalMonitor)
    view.frame, view.link = "base_footprint", "link_ree"
    goal = pose(0.4, -0.2, 0.6, yaw=0.5)
    axes, line, label = view.markers(pose(0.4, -0.2, 0.5), goal).markers
    assert axes.pose == goal
    assert [(p.x, p.y, p.z) for p in axes.points[:2]] == [
        (0.0, 0.0, 0.0),
        (monitor.AXIS_LENGTH, 0.0, 0.0),
    ]
    assert [(p.x, p.y, p.z) for p in line.points] == [
        (0.4, -0.2, 0.5),
        (0.4, -0.2, 0.6),
    ]
    assert label.text == "err=100.0mm/28.6deg"
    # Without TF only the axes and the text remain; the line expires.
    assert len(view.markers(None, goal).markers) == 2


class _Logger:
    def warning(self, _message):
        pass


class _FK:
    def service_is_ready(self):
        return False


class _Future:
    def result(self):
        return None


def _monitor_with_goal():
    view = monitor.EefGoalMonitor.__new__(monitor.EefGoalMonitor)
    view.node = type("Node", (), {"get_logger": lambda self: _Logger()})()
    view.fk = _FK()
    view.goal = pose(0.4, -0.2, 0.6)
    view._pending = None
    return view


def test_plan_without_fk_drops_the_previous_goal():
    view = _monitor_with_goal()
    display = DisplayTrajectory()
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = ["joint_r1"]
    trajectory.joint_trajectory.points = [JointTrajectoryPoint(positions=[0.3])]
    display.trajectory = [trajectory]
    view._on_plan(display)
    assert view.goal is None


def test_failed_fk_drops_the_previous_goal():
    view = _monitor_with_goal()
    view._pending = future = _Future()
    view._on_fk(future)
    assert view.goal is None and view._pending is None


def test_rviz_shows_the_monitor_markers():
    rviz = yaml.safe_load((SHARE / "rviz/manip.rviz").read_text())
    topics = {
        display["Name"]: display.get("Topic", {}).get("Value")
        for display in rviz["Visualization Manager"]["Displays"]
    }
    assert topics["EefGoalError"] == monitor.MARKER_TOPIC


def test_position_limits_from_urdf():
    urdf = """<robot name="r">
      <joint name="lift" type="prismatic"><limit lower="0" upper="1.2"/></joint>
      <joint name="arm" type="revolute"><limit lower="-1.57" upper="1.57"/></joint>
      <joint name="wheel" type="continuous"/>
      <joint name="mount" type="fixed"/>
    </robot>"""
    assert bounds.position_limits(urdf) == {"lift": (0.0, 1.2), "arm": (-1.57, 1.57)}


def test_clamp_near_limit_only_within_the_tolerance():
    assert bounds.clamp_near_limit(-1.3e-14, 0.0, 1.2, bounds.ROUNDOFF) == 0.0
    assert bounds.clamp_near_limit(1.2 + 1e-7, 0.0, 1.2, bounds.ROUNDOFF) == 1.2
    assert bounds.clamp_near_limit(-0.001, 0.0, 1.2, bounds.ROUNDOFF) == -0.001
    # A simulator margin widens the range; larger excursions still pass.
    assert bounds.clamp_near_limit(-0.005, 0.0, 1.2, 0.005 + 1e-6) == 0.0
    assert bounds.clamp_near_limit(-7e-3, 0.0, 1.2, 0.005 + 1e-6) == -7e-3
    assert bounds.clamp_near_limit(0.5, 0.0, 1.2, 0.005 + 1e-6) == 0.5


def test_clamp_positions_uses_each_joint_limit():
    limits = {"lift": (0.0, 1.2), "arm": (-1.57, 1.57)}
    msg = JointState(name=["lift", "arm", "wheel"], position=[-1e-14, 1.5700001, 9.0])
    clamped = bounds.clamp_positions(msg, limits, bounds.ROUNDOFF).position
    assert list(clamped) == [0.0, 1.57, 9.0]
    # Velocity-only messages are relayed unchanged.
    msg = JointState(name=["lift"], velocity=[0.1])
    assert list(bounds.clamp_positions(msg, limits, bounds.ROUNDOFF).position) == []


def test_launch_default_tolerance_is_roundoff_only():
    spec = importlib.util.spec_from_file_location(
        "manip_launch", SHARE / "launch/manip.launch.py"
    )
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    (argument,) = [
        entity
        for entity in launch.generate_launch_description().entities
        if getattr(entity, "name", None) == "limit_tolerance"
    ]
    assert float(argument.default_value[0].text) == bounds.ROUNDOFF
