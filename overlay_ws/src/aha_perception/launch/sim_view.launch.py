"""Integrated simulation view: RViz (map + robot + head point cloud), the
control panel (base joystick + head pan/tilt) and the head look-at node.

Included by perception.launch.py (sim.launch.py use_perception:=true, unless
rviz:=false) and by real_head_camera.launch.py. Runnable on its own against an
already-running simulation, so the view can be restarted without restarting
Gazebo:

  ros2 launch aha_perception sim_view.launch.py   # fixed frame odom

Nodes:
  rviz2                          rviz/sim_view.rviz (fixed frame overridden by -f)
  sim_control_panel.py           /diff_drive_controller/cmd_vel (TwistStamped),
                                 head trajectories; wall clock
  head_look_at.py                /aha/perception/look_at; RViz's Publish Point
                                 tool (/clicked_point) is remapped there;
                                 wall clock (uses the latest TF, cleared
                                 when the sim restarts or resets)
Head limits, speed and presets: config/head.yaml.

The two rclpy nodes get aha_perception's Python dir prepended to PYTHONPATH,
so they also start from a shell sourced before that module existed.
"""

import os
import sys

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def python_env():
    """PYTHONPATH with aha_perception's module dir first (ament_cmake_python layout)."""
    pkg_python = os.path.join(
        get_package_prefix("aha_perception"),
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    return {
        "PYTHONPATH": os.pathsep.join(
            filter(None, [pkg_python, os.environ.get("PYTHONPATH")])
        )
    }


def generate_launch_description():
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    share = FindPackageShare("aha_perception")
    head_params = PathJoinSubstitution([share, "config", "head.yaml"])
    env = python_env()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="RViz uses /clock from Gazebo",
            ),
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="odom",
                description="RViz fixed frame",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution([share, "rviz", "sim_view.rviz"]),
                description="RViz configuration for the simulation view",
            ),
            DeclareLaunchArgument(
                "control_panel",
                default_value="true",
                description="Start the control panel (base joystick + head)",
            ),
            # No name override: it would rename every node in the process,
            # including the slam_toolbox panel's client node.
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=[
                    "-d",
                    LaunchConfiguration("rviz_config"),
                    "-f",
                    LaunchConfiguration("fixed_frame"),
                ],
                parameters=[{"use_sim_time": use_sim_time}],
                # Publish Point tool drives head_look_at.py.
                remappings=[("/clicked_point", "/aha/perception/look_at")],
                output="screen",
            ),
            # The two rclpy nodes run on the wall clock on purpose: with
            # use_sim_time each handles /clock at 1 kHz (head_look_at measured
            # 69% vs 20% CPU) and neither needs it (latest-TF lookups,
            # unstamped trajectories, wall-stamped cmd_vel is accepted).
            Node(
                package="aha_perception",
                executable="head_look_at.py",
                parameters=[head_params],
                additional_env=env,
                output="screen",
            ),
            Node(
                package="aha_perception",
                executable="sim_control_panel.py",
                parameters=[head_params],
                additional_env=env,
                output="screen",
                condition=IfCondition(LaunchConfiguration("control_panel")),
            ),
        ]
    )
