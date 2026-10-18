"""slam_toolbox (online async) for the simulation, included by nav.launch.py.

Consumes /scan (sim-only LiDAR) and odom -> base_footprint; publishes /map and
map -> odom.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution(
        [FindPackageShare("aha_navigation"), "config", "slam_toolbox.yaml"]
    )
    online_async_launch = PathJoinSubstitution(
        [FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("slam_params_file", default_value=default_params),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(online_async_launch),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "slam_params_file": LaunchConfiguration("slam_params_file"),
                    "autostart": "true",
                    "use_lifecycle_manager": "false",
                }.items(),
            ),
        ]
    )
