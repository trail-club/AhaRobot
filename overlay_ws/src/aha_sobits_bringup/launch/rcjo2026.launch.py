"""Compatibility entrypoint for the default SOBITS Japan Open simulation."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("aha_bringup"),
                            "launch",
                            "sim.launch.py",
                        ]
                    )
                ),
                launch_arguments={"world": "rcjo2026"}.items(),
            ),
        ]
    )
