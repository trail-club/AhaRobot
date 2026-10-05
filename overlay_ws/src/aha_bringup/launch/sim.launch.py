"""End-to-end simulation bringup for AhaRobot.

Starts:
  - Gazebo Harmonic with the requested world
  - robot_state_publisher (from xacro, sim:=true)
  - spawn AhaRobot into Gazebo
  - ros_gz_bridge for /clock
  - controller_manager spawners:
        joint_state_broadcaster, diff_drive_controller,
        left/right_arm_controller, left/right_gripper_controller,
        lift_controller, head_controller
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    AndSubstitution,
    EqualsSubstitution,
    IfElseSubstitution,
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from aha_gazebo.world import SPAWN_COORDINATES, load_preset, validate_spawn


CONTROLLERS_AFTER_JSB = [
    "diff_drive_controller",
    "left_arm_controller",
    "right_arm_controller",
    "left_gripper_controller",
    "right_gripper_controller",
    "lift_controller",
    "head_controller",
]


def _spawner(name, switch_timeout: int = 30, condition=None):
    # Default spawner's controller-switch timeout is 5 s, which is too short
    # on slow hosts (macOS docker software rendering) and makes JSB fail to
    # activate on first try. Bump so /joint_states reliably comes up.
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            name,
            "--controller-manager",
            "/controller_manager",
            "--switch-timeout",
            str(switch_timeout),
        ],
        output="screen",
        condition=condition,
    )


def _spawn_arguments():
    legacy = {"x": 0.0, "y": 0.0, "z": 0.05, "yaw": 0.0}
    preset = load_preset(
        Path(get_package_share_directory("aha_gazebo")) / "config" / "rcjo2026.json"
    )
    use_arena_pose = AndSubstitution(
        EqualsSubstitution(LaunchConfiguration("world"), "rcjo2026"),
        EqualsSubstitution(LaunchConfiguration("world_path"), ""),
    )
    return [
        DeclareLaunchArgument(
            "spawn_" + key,
            default_value=IfElseSubstitution(
                use_arena_pose, str(preset["spawn"][key]), str(legacy[key])
            ),
            description="Initial world-frame " + key + " (metres, yaw in radians)",
        )
        for key in SPAWN_COORDINATES
    ]


def _validate_spawn(context):
    validate_spawn(
        {
            key: LaunchConfiguration("spawn_" + key).perform(context)
            for key in SPAWN_COORDINATES
        }
    )
    return []


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    xacro_path = PathJoinSubstitution(
        [
            FindPackageShare("aha_description"),
            "urdf",
            "aha_robot.urdf.xacro",
        ]
    )
    gazebo_launch = PathJoinSubstitution(
        [
            FindPackageShare("aha_gazebo"),
            "launch",
            "gazebo.launch.py",
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", xacro_path, " sim:=true"]),
            value_type=str,
        ),
    }

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={
            "world": LaunchConfiguration("world"),
            "world_path": LaunchConfiguration("world_path"),
            "headless": LaunchConfiguration("headless"),
        }.items(),
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "aha_robot",
            "-topic",
            "robot_description",
            "-x",
            LaunchConfiguration("spawn_x"),
            "-y",
            LaunchConfiguration("spawn_y"),
            "-z",
            LaunchConfiguration("spawn_z"),
            "-Y",
            LaunchConfiguration("spawn_yaw"),
        ],
        output="screen",
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        condition=IfCondition(LaunchConfiguration("bridge_clock")),
        output="screen",
    )

    jsb = _spawner("joint_state_broadcaster")

    # Load remaining controllers after joint_state_broadcaster is active.
    load_rest = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jsb,
            on_exit=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=[
                        *CONTROLLERS_AFTER_JSB,
                        "--controller-manager",
                        "/controller_manager",
                        "--switch-timeout",
                        "30",
                        "--activate-as-group",
                    ],
                    condition=IfCondition(
                        LaunchConfiguration("activate_controllers_as_group")
                    ),
                    output="screen",
                ),
                *[
                    _spawner(
                        c,
                        condition=UnlessCondition(
                            LaunchConfiguration("activate_controllers_as_group")
                        ),
                    )
                    for c in CONTROLLERS_AFTER_JSB
                ],
            ],
        )
    )

    # Let Gazebo resolve `model://astra_description/...` URIs from installed shares.
    # We add the parent of each pkg's share dir; Gazebo searches these roots
    # for a subdirectory matching the URI host part.
    share_roots = os.pathsep.join(
        os.path.join(get_package_prefix(p), "share")
        for p in ("astra_description", "aha_description")
    )
    # Read the launch context so resource roots set by a parent launch survive.
    resource_path = [
        share_roots,
        os.pathsep,
        EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
    ]

    # Optional per-squad subsystems. Default off — bringup is minimal.
    subsystems = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare(package),
                        "launch",
                        filename,
                    ]
                )
            ),
            condition=IfCondition(LaunchConfiguration(argument)),
        )
        for package, filename, argument in (
            ("aha_navigation", "nav.launch.py", "use_nav"),
            ("aha_manipulation", "manip.launch.py", "use_manip"),
            ("aha_perception", "perception.launch.py", "use_perception"),
        )
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value="rcjo2026",
                description="rcjo2026 (SOBITS Japan Open), or a bundled SDF filename",
            ),
            DeclareLaunchArgument(
                "world_path",
                default_value="",
                description="Optional absolute SDF path; overrides world",
            ),
            *_spawn_arguments(),
            OpaqueFunction(function=_validate_spawn),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument(
                "bridge_clock",
                default_value="true",
                description="Bridge Gazebo's clock; disable when a parent launch supplies /clock",
            ),
            DeclareLaunchArgument("use_nav", default_value="false"),
            DeclareLaunchArgument(
                "activate_controllers_as_group",
                default_value="false",
                description="Load command controllers and activate them in one switch",
            ),
            DeclareLaunchArgument("use_manip", default_value="false"),
            DeclareLaunchArgument("use_perception", default_value="false"),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
            gz,
            rsp,
            clock_bridge,
            spawn,
            jsb,
            load_rest,
            *subsystems,
        ]
    )
