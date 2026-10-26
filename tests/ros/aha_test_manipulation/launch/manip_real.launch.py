"""Test bed manipulation on the real robot on its frame, next to
aha_bringup/real.launch.py with the same frame_config (README.md).

Starts:
  - moveit.launch.py with frame_config and arms:=<arm>: move_group on the
    frame model (lifts fixed), trajectories through the arm's trajectory
    bridge; RViz unless rviz:=false
  - place_server:=true (default)  place_server.py for <arm>: no release
    topic, the gripper through arm_node's gripper_joint_command, the side
    approach moving horizontally only, the side-grasp TCP fixed at the
    height computed from the frame model, the head aimed before perceiving
    unless aim_head:=false
"""

from pathlib import Path

from aha_description.frame import frame_description
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from aha_test_manipulation.frame_moveit import side_grasp_tcp_height

# arm_node's gripper range is -0.002 .. 0.055 m per finger (astra_controller
# ARM_PROFILES); it drops a command outside it.
GRIPPER_OPEN = 0.055
# arm_node sends each gripper command as the target at full speed; closing at
# once on an object reset the arm board (docs/context/motor.md, 2026-10-25).
# The place server ramps its gripper commands (the release opens) over this
# time (s).
GRIPPER_RAMP_TIME = 1.0
# Without the lift only joint_<l|r>5 tilts the hand, so a TCP with z up (side
# grasp) stays at one height (tcp_height, from the frame model): place_server
# matches it with each spot's release height (RELEASE_HEIGHT to
# MAX_RELEASE_HEIGHT), a 2 cm band of surface heights per object height near
# the robot's own table (0.38 m above the floor); the top grasp reaches none.
# The side approach and retreat therefore move horizontally only, and so does
# no escape (escape_lift 0). The approach moves at the release height: a real
# TCP lower than the frame model's would drag the object over the surface, so
# the release height starts at 2 cm.
LOOK_HEIGHT = 0.38
RELEASE_HEIGHT = 0.02
# The arm trajectory bridge rejected joint goals planned at velocity_scaling
# 0.5 for 2.2 rad/s^2 between the first points (limit 2.0).
VELOCITY_SCALING = 0.3
# A cup standing at the spot was missing from the first detection after the
# return and seen in a later one.
VERIFY_DETECTIONS = 3
# The return brushed a placed cup detected 3-4 cm off the place pose: its box
# in the planning scene grows this much per side.
PLACED_PADDING = 0.03
MAX_RELEASE_HEIGHT = 0.04


def _place_server(context):
    arm = LaunchConfiguration("arm").perform(context)
    urdf = frame_description(LaunchConfiguration("frame_config").perform(context))
    return [
        Node(
            package="aha_test_manipulation",
            executable="place_server.py",
            parameters=[
                {
                    "use_sim_time": False,
                    "default_arm": arm,
                    "gripper_interface": "arm_node",
                    "gripper_open": GRIPPER_OPEN,
                    "gripper_ramp_time": GRIPPER_RAMP_TIME,
                    "release_topic": "",
                    "aim_head": ParameterValue(
                        LaunchConfiguration("aim_head"), value_type=bool
                    ),
                    "look_height": LOOK_HEIGHT,
                    "default_approach": "side",
                    "lift_margin": 0.0,
                    "retreat_lift": 0.0,
                    "escape_lift": 0.0,
                    "release_height": RELEASE_HEIGHT,
                    "velocity_scaling": VELOCITY_SCALING,
                    "verify_detections": VERIFY_DETECTIONS,
                    "placed_padding": PLACED_PADDING,
                    "tcp_height": side_grasp_tcp_height(urdf, arm),
                    "max_release_height": MAX_RELEASE_HEIGHT,
                }
            ],
            condition=IfCondition(LaunchConfiguration("place_server")),
            output="screen",
        )
    ]


def generate_launch_description():
    arm = LaunchConfiguration("arm")
    moveit_launch = PathJoinSubstitution(
        [FindPackageShare("aha_test_manipulation"), "launch", "moveit.launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "frame_config",
                default_value=str(
                    Path(get_package_share_directory("aha_description"))
                    / "config"
                    / "frame_table_202610.yaml"
                ),
                description="Frame model, the same as real.launch.py's",
            ),
            DeclareLaunchArgument(
                "arm",
                default_value="right",
                choices=["right", "left"],
                description="Arm whose trajectory bridge executes trajectories "
                "and the place server's default arm",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz with the MotionPlanning display",
            ),
            DeclareLaunchArgument(
                "place_server",
                default_value="true",
                description="Start the PlaceObject action server",
            ),
            DeclareLaunchArgument(
                "aim_head",
                default_value="true",
                description="Aim the head (/aha/perception/look_at) before "
                "perceiving; false without a head",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(moveit_launch),
                launch_arguments={
                    "use_sim_time": "false",
                    "rviz": LaunchConfiguration("rviz"),
                    "frame_config": LaunchConfiguration("frame_config"),
                    "arms": arm,
                }.items(),
            ),
            OpaqueFunction(function=_place_server),
        ]
    )
