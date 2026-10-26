"""Launch the camera-only RGB-D to PointCloud2 view.

camera_source:=rosbridge (default): the Mac RealSense process and
rosbridge_server are deliberately started outside this launch.
camera_source:=realsense: realsense.launch.py (RealSense ROS driver, Linux)
is started here.

This launch consumes the three standard camera topics, includes
pointcloud.launch.py (depth_image_proc, shared with the simulation) for
PointCloud2 generation, and starts RViz.

The simulation uses perception.launch.py instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EqualsSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def perception_launch(name):
    return PythonLaunchDescriptionSource(
        PathJoinSubstitution([FindPackageShare("aha_perception"), "launch", name])
    )


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
            DeclareLaunchArgument(
                "camera_source",
                default_value="rosbridge",
                choices=["rosbridge", "realsense"],
                description="rosbridge: topics from a separately started "
                "rosbridge stream; realsense: start realsense.launch.py",
            ),
            DeclareLaunchArgument(
                "color_profile",
                default_value="640,480,15",
                description="camera_source:=realsense color width,height,fps",
            ),
            DeclareLaunchArgument(
                "depth_profile",
                default_value="640,480,15",
                description="camera_source:=realsense depth width,height,fps",
            ),
            IncludeLaunchDescription(
                perception_launch("realsense.launch.py"),
                launch_arguments={
                    "color_profile": LaunchConfiguration("color_profile"),
                    "depth_profile": LaunchConfiguration("depth_profile"),
                    "imu": "false",
                }.items(),
                condition=IfCondition(
                    EqualsSubstitution(
                        LaunchConfiguration("camera_source"), "realsense"
                    )
                ),
            ),
            IncludeLaunchDescription(
                perception_launch("pointcloud.launch.py"),
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
