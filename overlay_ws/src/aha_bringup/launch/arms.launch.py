"""実機の左右 arm_node と、腕ごとの軌道ブリッジを上げる。

ブリッジ（aha_manipulation の arm_trajectory_bridge.py、`trajectory_bridge:=false` で止める）は
`/<side>_arm_controller` の軌道を受けたときだけ arm_node へ指令を送る。

arm_node の腕とグリッパの関節状態は `/<side>/arm/joint_states` に出す。`/joint_states` へは
joint_state_publisher の `source_list` で合流させる（aha_perception の real_head_camera.launch.py）。

キーボード教示はここに含めない。`ros2 launch` はキーを渡さないので、
別の対話シェルで `ros2 run aha_arm_teleop keyboard_teleop` する。
手順は docs/arm-keyboard-teleop.md。

既定で全トピックを rosbag に記録する（`record:=false` で止める）。
保存先は `bag_dir`（既定 /app/overlay_ws/bags）の下の `arms_<日時>/`。
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (
    AndSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

DEFAULT_BAG_DIR = os.environ.get("AHA_BAG_DIR", "/app/overlay_ws/bags")
JST = timezone(timedelta(hours=9), "JST")
# side -> (arm_node joint_names, gripper_joint_names). The bridge's `joints`
# in aha_manipulation/config/arm.yaml use the same order.
ARMS = {
    "left": (
        ["joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6"],
        ["joint_l7l", "joint_l7r"],
    ),
    "right": (
        ["joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6"],
        ["joint_r7l", "joint_r7r"],
    ),
}


def state_topic(side: str) -> str:
    """arm_node の腕とグリッパの関節状態。ブリッジの `state_topic`（arm.yaml）と同じ。"""
    return f"/{side}/arm/joint_states"


def _arm(side: str, joints: list[str], grippers: list[str]) -> Node:
    return Node(
        package="astra_controller",
        executable="arm_node",
        namespace=f"{side}/arm",
        parameters=[
            {
                "device": LaunchConfiguration(f"{side}_device"),
                "joint_names": joints,
                "gripper_joint_names": grippers,
            }
        ],
        remappings=[
            ("joint_states", state_topic(side)),
            ("gripper_joint_states", state_topic(side)),
        ],
        condition=IfCondition(LaunchConfiguration(f"use_{side}")),
        output="screen",
        emulate_tty=True,
    )


def _bridge(side: str) -> Node:
    return Node(
        package="aha_manipulation",
        executable="arm_trajectory_bridge.py",
        name=f"aha_{side}_arm_bridge",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("aha_manipulation"), "config", "arm.yaml"]
            )
        ],
        condition=IfCondition(
            AndSubstitution(
                LaunchConfiguration(f"use_{side}"),
                LaunchConfiguration("trajectory_bridge"),
            )
        ),
        output="screen",
        emulate_tty=True,
    )


def _record(context):
    if LaunchConfiguration("record").perform(context).lower() not in ("1", "true"):
        return []
    root = Path(LaunchConfiguration("bag_dir").perform(context))
    root.mkdir(parents=True, exist_ok=True)
    # bags/ の中を colcon がパッケージとして探しに行かないように。
    (root / "COLCON_IGNORE").touch(exist_ok=True)
    # out = root / datetime.now().strftime("arms_%Y%m%d_%H%M%S")
    out = root / datetime.now(JST).strftime("arms_%Y%m%d_%H%M%S")
    return [
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "-a", "-s", "mcap", "-o", str(out)],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("left_device", default_value="/dev/tty_puppet_left"),
            DeclareLaunchArgument(
                "right_device", default_value="/dev/tty_puppet_right"
            ),
            DeclareLaunchArgument("use_left", default_value="true"),
            DeclareLaunchArgument("use_right", default_value="true"),
            DeclareLaunchArgument("trajectory_bridge", default_value="true"),
            DeclareLaunchArgument("record", default_value="true"),
            DeclareLaunchArgument("bag_dir", default_value=DEFAULT_BAG_DIR),
            *(_arm(side, *ARMS[side]) for side in ARMS),
            *(_bridge(side) for side in ARMS),
            OpaqueFunction(function=_record),
        ]
    )
