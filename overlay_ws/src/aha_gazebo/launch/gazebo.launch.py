"""Launch: Gazebo Harmonic with a chosen world (default: empty)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration("world")

    world_path = PathJoinSubstitution([
        FindPackageShare("aha_gazebo"), "worlds", world,
    ])

    gz_launch = PathJoinSubstitution([
        FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="empty.sdf"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={"gz_args": [world_path, " -r"]}.items(),
        ),
    ])
