"""Launch the camera-only RGB-D to PointCloud2 view.

The Mac RealSense process and rosbridge_server are deliberately started
outside this launch.  This launch consumes the three standard camera topics,
includes pointcloud.launch.py (depth_image_proc, shared with the simulation)
for PointCloud2 generation, and starts RViz.

The simulation uses perception.launch.py instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("aha_perception"), "rviz", "camera.rviz"]
                ),
                description="RViz configuration for the camera-only view",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("aha_perception"),
                            "launch",
                            "pointcloud.launch.py",
                        ]
                    )
                ),
                launch_arguments={"use_sim_time": "false"}.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="camera_view",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": False}],
                output="screen",
            ),
        ]
    )
