"""Compatibility entry used by the existing simulation bringup.

The Mac camera path is intentionally not selected from the simulation launch.
Use ``camera_view.launch.py`` for the camera-only RViz display instead.
"""
from launch import LaunchDescription
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        LogInfo(
            msg=(
                "[aha_perception] simulation perception is not part of the "
                "Mac RealSense path; use camera_view.launch.py for camera RViz"
            )
        ),
    ])
