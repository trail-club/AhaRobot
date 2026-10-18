"""Simulation perception pipeline, included by aha_bringup/sim.launch.py.

  ros2 launch aha_bringup sim.launch.py use_perception:=true

sim.launch.py passes no arguments; the ones below are read from the command
line. Starts:
  - camera_bridge.launch.py for Gazebo head and wrist cameras:
    /camera/color/image_raw, /camera/color/camera_info,
    /camera/depth_registered/image_rect (frame camera_color_optical_frame)
  - pointcloud.launch.py: /camera/depth/points
  - rviz:=true (default)  sim_view.launch.py (RViz, control panel, head look-at)

The real camera does not use this file: camera_view.launch.py (Mac/Linux
RealSense stream through rosbridge) and real_head_camera.launch.py (real head
servos + camera).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("aha_perception")
    use_sim_time = LaunchConfiguration("use_sim_time")

    camera_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "camera_bridge.launch.py"])
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    pointcloud = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    share,
                    "launch",
                    "pointcloud.launch.py",
                ]
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    sim_view = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    share,
                    "launch",
                    "sim_view.launch.py",
                ]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "fixed_frame": LaunchConfiguration("fixed_frame"),
            "control_panel": LaunchConfiguration("control_panel"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use /clock from Gazebo",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz, the control panel and the head look-at node",
            ),
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="odom",
                description="RViz fixed frame",
            ),
            DeclareLaunchArgument(
                "control_panel",
                default_value="true",
                description="Start the control panel (base joystick + head)",
            ),
            camera_bridge,
            pointcloud,
            sim_view,
        ]
    )
