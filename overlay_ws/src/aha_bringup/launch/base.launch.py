"""実機の台車（ODrive v3.6、USB）を aha_base の base_node で動かす。

シミュレーションの diff_drive_controller と同じ `/diff_drive_controller/cmd_vel`
（TwistStamped）を受け、`/diff_drive_controller/odom` と odom -> base_footprint の TF を出す。
キーボード操作は別の対話シェルで `ros2 run aha_bringup teleop_base.sh`。

パラメータは `params_file`（既定 aha_base/config/base.yaml）。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("aha_base"), "config", "base.yaml"]
                ),
            ),
            Node(
                package="aha_base",
                executable="base_node.py",
                name="aha_base",
                parameters=[LaunchConfiguration("params_file")],
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
