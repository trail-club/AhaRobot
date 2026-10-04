"""実機の左右 arm_node を上げる。

キーボード教示はここに含めない。`ros2 launch` はキーを渡さないので、
別の対話シェルで `ros2 run aha_arm_teleop keyboard_teleop` する。
手順は docs/arm-keyboard-teleop.md。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _arm(side: str, joints: list[str], grippers: list[str]) -> Node:
    return Node(
        package="astra_controller",
        executable="arm_node",
        namespace=f"{side}/arm",
        parameters=[{
            "device": LaunchConfiguration(f"{side}_device"),
            "joint_names": joints,
            "gripper_joint_names": grippers,
        }],
        remappings=[
            ("joint_states", "/joint_states"),
            ("gripper_joint_states", "/joint_states"),
        ],
        condition=IfCondition(LaunchConfiguration(f"use_{side}")),
        output="screen",
        emulate_tty=True,
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("left_device", default_value="/dev/tty_puppet_left"),
        DeclareLaunchArgument("right_device", default_value="/dev/tty_puppet_right"),
        DeclareLaunchArgument("use_left", default_value="true"),
        DeclareLaunchArgument("use_right", default_value="true"),
        _arm("left", ["joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6"], ["joint_l7l", "joint_l7r"]),
        _arm("right", ["joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6"], ["joint_r7l", "joint_r7r"]),
    ])
