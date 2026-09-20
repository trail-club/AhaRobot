"""Launch the camera-only RGB-D to PointCloud2 view.

The Mac RealSense process and rosbridge_server are deliberately started
outside this launch.  This launch consumes the three standard camera topics,
uses depth_image_proc for PointCloud2 generation, and starts RViz.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
            Node(
                package="depth_image_proc",
                executable="point_cloud_xyzrgb_node",
                name="point_cloud_xyzrgb",
                remappings=[
                    ("rgb/image_rect_color", "/camera/color/image_raw"),
                    ("rgb/camera_info", "/camera/color/camera_info"),
                    (
                        "depth_registered/image_rect",
                        "/camera/depth_registered/image_rect",
                    ),
                    ("points", "/camera/depth/points"),
                ],
                output="screen",
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
