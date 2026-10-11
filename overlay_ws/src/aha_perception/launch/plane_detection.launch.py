"""Plane detection on the head depth camera (simulation and real robot).

  ros2 launch aha_perception plane_detection.launch.py
  ros2 launch aha_perception plane_detection.launch.py method:=ransac_gravity

Included by perception.launch.py and real_head_camera.launch.py with
planes:=true. Starts plane_detection_node.py:
  in   /camera/depth_registered/image_rect, /camera/color/camera_info, TF
  out  /aha/perception/planes (aha_msgs/PlaneArray),
       /aha/perception/planes/cloud (PointCloud2),
       /aha/perception/planes/markers (MarkerArray), in target_frame
       /aha/perception/surfaces (aha_msgs/SurfaceArray), in tracking_frame
  srv  /aha/perception/get_place_candidates (aha_msgs/GetPlaceCandidates),
       candidates shown on /aha/perception/place/markers (MarkerArray)

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
                "method",
                default_value="ransac_gravity_refined",
                description="Detector name in aha_perception.planes.DETECTORS",
            ),
            DeclareLaunchArgument(
                "target_frame",
                default_value="base_footprint",
                description="Output frame: origin on the floor, z up",
            ),
            DeclareLaunchArgument(
                "max_rate",
                default_value="5.0",
                description="Maximum detections per second (0: as fast as possible)",
            ),
            # Wall clock on purpose, like head_look_at.py: the node only uses
            # image stamps (TF lookups, output stamps) and a wall-clock rate
            # limit, so handling /clock at 1 kHz would only cost CPU.
            Node(
                package="aha_perception",
                executable="plane_detection_node.py",
                parameters=[
                    {
                        "method": LaunchConfiguration("method"),
                        "target_frame": LaunchConfiguration("target_frame"),
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
