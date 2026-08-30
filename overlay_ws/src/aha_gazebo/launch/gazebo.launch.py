"""Launch: Gazebo Harmonic with a chosen world (default: empty).

Args:
  world     — filename under aha_gazebo/worlds/  (default: empty.sdf)
  headless  — true/false. When true, appends `-s --headless-rendering`
              to gz_args, giving a server-only run suitable for CI or
              slow hosts (macOS).
  gz_args   — extra args appended verbatim after world path + '-r'.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration("world")
    headless = LaunchConfiguration("headless")
    extra = LaunchConfiguration("gz_args")

    world_path = PathJoinSubstitution([
        FindPackageShare("aha_gazebo"), "worlds", world,
    ])
    gz_launch = PathJoinSubstitution([
        FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="empty.sdf"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("gz_args", default_value=""),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            condition=IfCondition(headless),
            launch_arguments={
                "gz_args": [world_path, " -r -s --headless-rendering ", extra],
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            condition=UnlessCondition(headless),
            launch_arguments={
                "gz_args": [world_path, " -r ", extra],
            }.items(),
        ),
    ])
