"""Simulation perception pipeline, included by aha_bringup/sim.launch.py.

  ros2 launch aha_bringup sim.launch.py use_perception:=true

sim.launch.py passes use_sim_time, and control_panel:=false with
manip_rviz:=true (the manipulation view has its own panel); the other
arguments are read from the command line. Starts:
  - camera_bridge.launch.py for Gazebo head and wrist cameras:
    /camera/color/image_raw, /camera/color/camera_info,
    /camera/depth_registered/image_rect (frame camera_color_optical_frame)
  - depth_noise:=<preset>  the bridge publishes the depth on
    /camera/depth_registered/image_rect_clean instead and sim_depth_noise.py
    publishes it with D435i noise on /camera/depth_registered/image_rect
  - pointcloud.launch.py: /camera/depth/points
  - rviz:=true (default)  sim_view.launch.py (RViz, control panel, head look-at)
  - planes:=true          plane_detection.launch.py: /aha/perception/planes/*
    (furniture_config:=<yaml> is passed to it)

The real camera does not use this file: camera_view.launch.py (Mac/Linux
RealSense stream through rosbridge) and real_head_camera.launch.py (real head
servos + camera).
"""

import os
import sys

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    IfElseSubstitution,
    LaunchConfiguration,
    NotEqualsSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

DEPTH_TOPIC = "/camera/depth_registered/image_rect"
CLEAN_DEPTH_TOPIC = "/camera/depth_registered/image_rect_clean"
# Presets of aha_perception.depth_noise fitted to the real D435i.
DEPTH_NOISE_CHOICES = ["none", "d435i_filtered", "d435i_raw"]


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
    share = FindPackageShare("aha_perception")
    use_sim_time = LaunchConfiguration("use_sim_time")
    depth_noise = LaunchConfiguration("depth_noise")
    noisy = NotEqualsSubstitution(depth_noise, "none")

    camera_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "camera_bridge.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "depth_topic": IfElseSubstitution(noisy, CLEAN_DEPTH_TOPIC, DEPTH_TOPIC),
        }.items(),
    )

    # Wall clock: the node only copies the image stamps.
    depth_noise_node = Node(
        package="aha_perception",
        executable="sim_depth_noise.py",
        parameters=[
            {
                "preset": depth_noise,
                "input_topic": CLEAN_DEPTH_TOPIC,
                "output_topic": DEPTH_TOPIC,
            }
        ],
        additional_env=python_env(),
        condition=IfCondition(noisy),
        output="screen",
    )

    pointcloud = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    share,
                    "launch",
                    "pointcloud.launch.py",
                ]
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    sim_view = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    share,
                    "launch",
                    "sim_view.launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "fixed_frame": LaunchConfiguration("fixed_frame"),
            "control_panel": LaunchConfiguration("control_panel"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    planes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "plane_detection.launch.py"])
        ),
        launch_arguments={
            "furniture_config": LaunchConfiguration("furniture_config"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("planes")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use /clock from Gazebo",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz, the control panel and the head look-at node",
            ),
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="odom",
                description="RViz fixed frame",
            ),
            DeclareLaunchArgument(
                "control_panel",
                default_value="true",
                description="Start the control panel (base joystick + head)",
            ),
            DeclareLaunchArgument(
                "planes",
                default_value="false",
                description="Start plane detection on the head depth camera",
            ),
            DeclareLaunchArgument(
                "furniture_config",
                default_value="",
                description="Furniture YAML for plane detection (empty = none)",
            ),
            DeclareLaunchArgument(
                "depth_noise",
                default_value="none",
                choices=DEPTH_NOISE_CHOICES,
                description="D435i depth noise preset added by sim_depth_noise.py",
            ),
            camera_bridge,
            depth_noise_node,
            pointcloud,
            sim_view,
            planes,
        ]
    )
