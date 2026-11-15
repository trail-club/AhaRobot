"""Bridge simulated head RGB-D and wrist RGB cameras without a GUI or point cloud.

depth_topic moves the head depth image off /camera/depth_registered/image_rect
(perception.launch.py depth_noise:=<preset> republishes it there with noise).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

DEPTH_TOPIC = "/camera/depth_registered/image_rect"


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "depth_topic",
                default_value=DEPTH_TOPIC,
                description="ROS topic of the bridged head depth image",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="camera_bridge",
                parameters=[
                    {
                        "config_file": PathJoinSubstitution(
                            [
                                FindPackageShare("aha_perception"),
                                "config",
                                "sim_camera_bridge.yaml",
                            ]
                        ),
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                    }
                ],
                remappings=[(DEPTH_TOPIC, LaunchConfiguration("depth_topic"))],
                output="screen",
            ),
        ]
    )
