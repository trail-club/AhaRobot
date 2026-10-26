"""MoveIt 2 move_group (and optional RViz) for AhaRobot.

Simulation (default): expects robot_state_publisher, /joint_states and the
arm / lift controllers from aha_bringup/sim.launch.py. robot_description is
generated from the xacro and args in .setup_assistant (aha_test_description's
test bed xacro, sim:=true), the same as the simulation's
robot_state_publisher.

Real robot on its frame (frame_config:=<aha_description frame YAML>): next to
aha_bringup/real.launch.py with the same frame_config. robot_description is
the frame model (aha_description.frame.frame_description, lifts fixed), the
SRDF, joint limits and controllers come from aha_test_manipulation.frame_moveit:
trajectories go to the arm trajectory bridges of ``arms`` only.

Arguments:
  use_sim_time (true)   use /clock
  rviz         (true)   RViz with the MotionPlanning display
  frame_config ("")     frame YAML; empty = the simulation's test bed robot
  arms         (right)  with frame_config: arms whose bridges execute
                        trajectories, comma separated (right,left)
"""

from pathlib import Path

import yaml
from aha_description.frame import frame_description
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from aha_test_manipulation import frame_moveit


def moveit_config():
    return (
        MoveItConfigsBuilder("aha_robot", package_name="aha_test_manipulation")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .planning_scene_monitor(publish_robot_description_semantic=True)
        .to_moveit_configs()
    )


def frame_moveit_config(frame_config, arms):
    """moveit_config() for the real robot on the frame at ``frame_config``,
    executing through the bridges of ``arms``."""
    config = moveit_config()
    urdf = frame_description(frame_config)
    bridges = yaml.safe_load(
        (
            Path(get_package_share_directory("aha_manipulation"))
            / "config"
            / "arm.yaml"
        ).read_text()
    )
    srdf = config.robot_description_semantic["robot_description_semantic"]
    config.robot_description = {"robot_description": urdf}
    config.robot_description_semantic = {
        "robot_description_semantic": frame_moveit.frame_srdf(srdf, urdf)
    }
    planning = config.joint_limits["robot_description_planning"]
    config.joint_limits = {
        "robot_description_planning": frame_moveit.frame_joint_limits(
            planning, bridges, urdf
        )
    }
    config.trajectory_execution = {
        "moveit_manage_controllers": True,
        **frame_moveit.frame_controllers(bridges, arms),
    }
    return config


def _nodes(context):
    frame_config = LaunchConfiguration("frame_config").perform(context)
    if frame_config:
        arms = [
            a.strip()
            for a in LaunchConfiguration("arms").perform(context).split(",")
            if a.strip()
        ]
        config = frame_moveit_config(frame_config, arms)
    else:
        config = moveit_config()
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    rviz_config = str(
        Path(get_package_share_directory("aha_test_manipulation"))
        / "rviz"
        / "manip.rviz"
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        parameters=[config.to_dict(), use_sim_time],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="moveit_rviz",
        arguments=["-d", rviz_config],
        parameters=[
            config.robot_description,
            config.robot_description_semantic,
            config.robot_description_kinematics,
            config.planning_pipelines,
            config.joint_limits,
            use_sim_time,
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="log",
    )
    return [move_group, rviz]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz with the MotionPlanning display",
            ),
            DeclareLaunchArgument(
                "frame_config",
                default_value="",
                description="Frame YAML of the real robot (aha_description/frame.py); "
                "empty: the simulation's test bed robot",
            ),
            DeclareLaunchArgument(
                "arms",
                default_value="right",
                description="With frame_config: arms whose trajectory bridges "
                "execute trajectories (comma separated)",
            ),
            OpaqueFunction(function=_nodes),
        ]
    )
