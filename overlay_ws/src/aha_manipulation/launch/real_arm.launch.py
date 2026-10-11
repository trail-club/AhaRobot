"""Bring up the real Astra arms, each through the aha_manipulation bridge.

For each arm this launches:
  * the upstream ``astra_controller`` ``arm_node`` (in ``<side>/arm``), driven
    over its astra ``joint_command`` topic, and
  * an ``aha_arm_bridge`` that paces goals from the ``/<side>_arm_controller``
    interface onto that ``joint_command`` topic (see config/arm.yaml).

Usage:
    ros2 launch aha_manipulation real_arm.launch.py            # both arms
    ros2 launch aha_manipulation real_arm.launch.py sides:=left
    ros2 launch aha_manipulation real_arm.launch.py \\
        device_left:=/dev/tty_puppet_left device_right:=/dev/tty_puppet_right

The lift (joint_l1) and gripper are separate arm_node topics and are out of
scope for the 5-DOF arm bridge.
"""

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

_LEFT_JOINTS = ["joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6"]
_LEFT_GRIPPER = ["joint_l7l", "joint_l7r"]
_RIGHT_JOINTS = ["joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6"]
_RIGHT_GRIPPER = ["joint_r7l", "joint_r7r"]

_SIDES = {
    "left": {"device_arg": "device_left", "joints": _LEFT_JOINTS, "gripper": _LEFT_GRIPPER},
    "right": {"device_arg": "device_right", "joints": _RIGHT_JOINTS, "gripper": _RIGHT_GRIPPER},
}


def _arm_node(side: str, device: str, cfg: dict) -> Node:
    return Node(
        package="astra_controller",
        executable="arm_node",
        namespace=f"{side}/arm",
        parameters=[
            {
                "device": device,
                "joint_names": cfg["joints"],
                "gripper_joint_names": cfg["gripper"],
            }
        ],
        remappings=[
            ("joint_states", "/joint_states"),
            ("gripper_joint_states", "/joint_states"),
        ],
        output="screen",
        emulate_tty=True,
    )


def _bridge(side: str, params_file: str, use_bridge: str) -> Node:
    return Node(
        package="aha_manipulation",
        executable="arm_trajectory_bridge",
        name=f"aha_{side}_arm_bridge",
        parameters=[params_file],
        condition=IfCondition(use_bridge),
        output="screen",
    )


def _setup(context, *args, **kwargs):
    sides = context.launch_configurations.get("sides", "left right").split()
    use_bridge = context.launch_configurations.get("use_bridge", "true")
    params_file = f"{get_package_share_directory('aha_manipulation')}/config/arm.yaml"

    actions = []
    for side in sides:
        if side not in _SIDES:
            LogInfo(msg=[f"unknown side '{side}', skipping (expected: left, right)"])
            continue
        cfg = _SIDES[side]
        device = context.launch_configurations.get(cfg["device_arg"], "")
        if not device:
            LogInfo(msg=[f"{side} arm: no device (device_{side}:=... required); arm_node will not drive hardware"])
        actions.append(_arm_node(side, device, cfg))
        actions.append(_bridge(side, params_file, use_bridge))
    return actions


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()
    ld.add_action(
        DeclareLaunchArgument(
            "sides",
            default_value="left right",
            description="Space-separated arm sides to bring up: 'left' and/or 'right'.",
        )
    )
    ld.add_action(DeclareLaunchArgument("device_left", default_value=""))
    ld.add_action(DeclareLaunchArgument("device_right", default_value=""))
    ld.add_action(
        DeclareLaunchArgument(
            "use_bridge", default_value="true", description="Launch the aha_arm_bridge for each arm."
        )
    )
    ld.add_action(
        OpaqueFunction(
            function=_setup,
            # forward the launch configs the setup reads
            launch_configuration={"sides": LaunchConfiguration("sides")},
        )
    )
    return ld

