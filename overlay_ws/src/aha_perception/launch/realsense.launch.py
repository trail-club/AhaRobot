"""Real head D435i through the RealSense ROS driver (Linux).

  ros2 launch aha_perception realsense.launch.py
  ros2 launch aha_perception realsense.launch.py color_profile:=1280,720,15 depth_profile:=848,480,15

Starts realsense2_camera_node (node /camera) and publishes the head camera
contract of docs/interfaces.md:
  /camera/color/image_raw (rgb8), /camera/color/camera_info,
  /camera/depth_registered/image_rect (16UC1 mm, aligned to color),
  /camera/depth_registered/camera_info, /camera/imu (imu:=true)
All images use camera_color_optical_frame; the IMU uses
camera_imu_optical_frame (not in the URDF, read by imu_head.py only).
publish_tf is off: robot_state_publisher owns the camera TF.

Depth is post-processed like tools/perception/macos/stream_realsense.py
(disparity -> spatial -> temporal -> depth, SDK defaults; filters:=false
turns them off) before it is aligned to color. Infrared streams are off.

Profiles are "width,height,fps". The defaults run at USB 2.0; USB 3 also
offers 848x480 color and 848x480 depth at 15/30 fps.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    filters = ParameterValue(LaunchConfiguration("filters"), value_type=bool)
    imu = ParameterValue(LaunchConfiguration("imu"), value_type=bool)
    driver = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        # The driver's default namespace is /camera; topics are /camera/<stream>.
        ros_arguments=["-r", "__ns:=/"],
        parameters=[
            {
                "camera_name": "camera",
                "serial_no": ParameterValue(
                    LaunchConfiguration("serial_no"), value_type=str
                ),
                "publish_tf": False,
                "enable_color": True,
                "enable_depth": True,
                "enable_infra1": False,
                "enable_infra2": False,
                "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
                "rgb_camera.color_format": "RGB8",
                "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
                "align_depth.enable": True,
                "disparity_filter.enable": filters,
                "spatial_filter.enable": filters,
                "temporal_filter.enable": filters,
                "disparity_to_depth.enable": filters,
                "enable_accel": imu,
                "enable_gyro": imu,
                # 2: linear interpolation of accel onto the gyro samples
                "unite_imu_method": 2,
            }
        ],
        remappings=[
            (
                "/camera/aligned_depth_to_color/image_raw",
                "/camera/depth_registered/image_rect",
            ),
            (
                "/camera/aligned_depth_to_color/camera_info",
                "/camera/depth_registered/camera_info",
            ),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "color_profile",
                default_value="640,480,15",
                description="Color width,height,fps (aligned depth has this size)",
            ),
            DeclareLaunchArgument(
                "depth_profile",
                default_value="640,480,15",
                description="Depth width,height,fps before alignment",
            ),
            DeclareLaunchArgument(
                "filters",
                default_value="true",
                description="Depth post-processing (disparity, spatial, temporal)",
            ),
            DeclareLaunchArgument(
                "imu",
                default_value="true",
                description="Accel + gyro on /camera/imu",
            ),
            DeclareLaunchArgument(
                "serial_no",
                default_value="",
                description="Device serial number (empty: first device)",
            ),
            driver,
        ]
    )
