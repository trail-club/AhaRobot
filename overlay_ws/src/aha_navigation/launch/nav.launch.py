"""Navigation launch, included by aha_bringup/sim.launch.py with use_nav:=true.

Starts:
  - ros_gz_bridge for /scan from the sim-only LiDAR
    (aha_description/urdf/lidar.xacro)
  - slam.launch.py (slam_toolbox online async: /map and map -> odom)
Nav2 is not started.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_launch = PathJoinSubstitution(
        [FindPackageShare("aha_navigation"), "launch", "slam.launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="aha_nav_scan_bridge",
                arguments=["/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"],
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slam_launch),
                launch_arguments={"use_sim_time": use_sim_time}.items(),
            ),
        ]
    )
