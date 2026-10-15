"""MoveIt 2 right-arm planning, KDL IK and ros2_control trajectory execution.

move_group reads joint states through joint_state_bounds.py, which relays
joint_states_topic to /aha/manipulation/joint_states with positions up to
limit_tolerance outside the URDF limits moved onto them.
rviz:=true adds the RViz manipulation view (manip_view.launch.py).
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import xacro
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _move_group(context):
    share = Path(get_package_share_directory("aha_manipulation"))
    description = Path(get_package_share_directory("aha_description"))

    def load(filename):
        return yaml.safe_load((share / "config" / filename).read_text())

    parameters = {
        "robot_description": xacro.process_file(
            str(description / "urdf/aha_robot.urdf.xacro"),
            mappings={
                "sim": LaunchConfiguration("use_sim_time").perform(context),
                "cameras": LaunchConfiguration("cameras").perform(context),
            },
        ).toxml(),
        "robot_description_semantic": (share / "srdf/aha_robot.srdf").read_text(),
        "robot_description_kinematics": load("kinematics.yaml"),
        "robot_description_planning": load("joint_limits.yaml"),
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": load("ompl.yaml"),
        "use_sim_time": LaunchConfiguration("use_sim_time").perform(context) == "true",
        "allow_trajectory_execution": True,
        "publish_robot_description_semantic": True,
        **load("controllers.yaml"),
    }
    environment = {}
    controller_share = Path(
        get_package_share_directory("moveit_simple_controller_manager")
    )
    if ET.parse(controller_share / "package.xml").findtext("version") == "2.12.4":
        # Keep the controller plugin mapped while rclcpp destroys its callbacks.
        # Jazzy 2.12.4 otherwise crashes during TrajectoryExecutionManager teardown.
        plugin = (
            Path(get_package_prefix("moveit_simple_controller_manager"))
            / "lib/libmoveit_simple_controller_manager.so"
        )
        environment["LD_PRELOAD"] = " ".join(
            filter(None, [str(plugin), os.environ.get("LD_PRELOAD", "")])
        )
    return [
        Node(
            package="aha_manipulation",
            executable="joint_state_bounds.py",
            output="screen",
            parameters=[
                {
                    "robot_description": parameters["robot_description"],
                    "use_sim_time": parameters["use_sim_time"],
                    "limit_tolerance": float(
                        LaunchConfiguration("limit_tolerance").perform(context)
                    ),
                }
            ],
            remappings=[("joint_states", LaunchConfiguration("joint_states_topic"))],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[parameters],
            additional_env=environment,
            remappings=[("joint_states", "/aha/manipulation/joint_states")],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument(
                "limit_tolerance",
                default_value="1e-6",
                description="m / rad outside a URDF limit still moved onto it for MoveIt",
            ),
            DeclareLaunchArgument(
                "cameras", default_value="true", choices=["true", "false"]
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                choices=["true", "false"],
                description="Start the RViz manipulation view (manip_view.launch.py)",
            ),
            DeclareLaunchArgument(
                "camera_bridge",
                default_value="false",
                choices=["true", "false"],
                description="With rviz:=true, bridge the Gazebo cameras",
            ),
            OpaqueFunction(function=_move_group),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("aha_manipulation"),
                            "launch",
                            "manip_view.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "camera_bridge": LaunchConfiguration("camera_bridge"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
