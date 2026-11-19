"""実機の左右 arm_node と lift_node、腕ごとと昇降の軌道ブリッジを上げる。

ブリッジ（aha_manipulation の arm_trajectory_bridge.py、`trajectory_bridge:=false` で止める）は
`/<side>_arm_controller` と `/lift_controller` の軌道を受けたときだけ arm_node / lift_node へ指令を送る。
昇降の lift_node は `use_<side>_lift:=true` で左右ごとに起動する（既定は起動しない）。
デバイス名の既定は upstream AstraFirmwares の udev 規則（99-astra-fixed-port.rules）の名前。

arm_node の腕とグリッパの関節状態は `/<side>/arm/joint_states`、lift_node の昇降は
`/<side>/lift/joint_states` に出す。`/joint_states` へは joint_state_publisher の `source_list` で
合流させる（aha_perception の real_head_camera.launch.py）。

lift_node は起動時に基板から位置が届くまで待つ。基板は電源投入時に下端の
リミットスイッチまで下がって原点を取る。

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
    OrSubstitution,
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


# side -> lift_node の joint_names。aha_manipulation/config/lift.yaml の `joints` と同じ。
LIFTS = {"left": "joint_l1", "right": "joint_r1"}


def state_topic(side: str) -> str:
    """arm_node の腕とグリッパの関節状態。ブリッジの `state_topic`（arm.yaml）と同じ。"""
    return f"/{side}/arm/joint_states"


def lift_state_topic(side: str) -> str:
    """lift_node の関節状態。lift.yaml の `<joint>.state_topic` と同じ。"""
    return f"/{side}/lift/joint_states"


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


def _lift(side: str) -> Node:
    return Node(
        package="astra_controller",
        executable="lift_node",
        namespace=f"{side}/lift",
        parameters=[
            {
                "device": LaunchConfiguration(f"{side}_lift_device"),
                "joint_names": [LIFTS[side]],
            }
        ],
        remappings=[("joint_states", lift_state_topic(side))],
        condition=IfCondition(LaunchConfiguration(f"use_{side}_lift")),
        output="screen",
        emulate_tty=True,
    )


def _lift_bridge() -> Node:
    return Node(
        package="aha_manipulation",
        executable="arm_trajectory_bridge.py",
        name="aha_lift_bridge",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("aha_manipulation"), "config", "lift.yaml"]
            )
        ],
        condition=IfCondition(
            AndSubstitution(
                OrSubstitution(
                    LaunchConfiguration("use_left_lift"),
                    LaunchConfiguration("use_right_lift"),
                ),
                LaunchConfiguration("trajectory_bridge"),
            )
        ),
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
            DeclareLaunchArgument(
                "left_lift_device", default_value="/dev/tty_puppet_lift_left"
            ),
            DeclareLaunchArgument(
                "right_lift_device", default_value="/dev/tty_puppet_lift_right"
            ),
            DeclareLaunchArgument("use_left", default_value="true"),
            DeclareLaunchArgument("use_right", default_value="true"),
            DeclareLaunchArgument("use_left_lift", default_value="false"),
            DeclareLaunchArgument("use_right_lift", default_value="false"),
            DeclareLaunchArgument("trajectory_bridge", default_value="true"),
            DeclareLaunchArgument("record", default_value="true"),
            DeclareLaunchArgument("bag_dir", default_value=DEFAULT_BAG_DIR),
            *(_arm(side, *ARMS[side]) for side in ARMS),
            *(_lift(side) for side in LIFTS),
            *(_bridge(side) for side in ARMS),
            _lift_bridge(),
            OpaqueFunction(function=_record),
        ]
    )
