"""Accumulate the head camera cloud in a voxel map.

  ros2 launch aha_perception cloud_accumulator.launch.py

Included by perception.launch.py with merged_points:=true. Starts
cloud_accumulator.py:
  in   /camera/depth_registered/image_rect, /camera/color/image_raw,
       /camera/color/camera_info, /robot_description, TF
  out  /aha/perception/merged_points (PointCloud2) in fixed_frame
  srv  /aha/perception/merged_points/clear (std_srvs/Trigger)

The node gets aha_perception's Python dir prepended to PYTHONPATH, so it also
starts from a shell sourced before that module existed.
"""

import os
import sys

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="odom",
                description="Map frame (base_footprint is used until it has TF)",
            ),
            DeclareLaunchArgument(
                "max_rate",
                default_value="5.0",
                description="Maximum frames per second (0: as fast as possible)",
            ),
            # Wall clock on purpose, like plane_detection.launch.py: the node
            # only uses image stamps (TF lookups, ages, output stamps) and
            # wall-clock rate limits.
            Node(
                package="aha_perception",
                executable="cloud_accumulator.py",
                parameters=[
                    {
                        "fixed_frame": LaunchConfiguration("fixed_frame"),
                        "max_rate": ParameterValue(
                            LaunchConfiguration("max_rate"), value_type=float
                        ),
                    }
                ],
                additional_env=python_env(),
                output="screen",
            ),
        ]
    )
