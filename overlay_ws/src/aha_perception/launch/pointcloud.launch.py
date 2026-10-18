"""Build /camera/depth/points from the head RGB-D camera topics.

Shared by the real camera (camera_view.launch.py, real_head_camera.launch.py)
and the simulation (perception.launch.py) so all use the same depth_image_proc
node.

Inputs (depth aligned to color, frame camera_color_optical_frame):
  /camera/color/image_raw, /camera/color/camera_info,
  /camera/depth_registered/image_rect
Output:
  /camera/depth/points (sensor_msgs/PointCloud2, XYZRGB)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use /clock (true in simulation)",
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
                parameters=[
                    {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)}
                ],
                output="screen",
            ),
        ]
    )
