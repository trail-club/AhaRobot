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

from ament_index_python.packages import get_package_prefix

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


CONTROLLERS_AFTER_JSB = [
    "diff_drive_controller",
    "left_arm_controller",
    "right_arm_controller",
    "left_gripper_controller",
    "right_gripper_controller",
    "lift_controller",
    "head_controller",
]


def _spawner(name):
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=[name, "--controller-manager", "/controller_manager"],
        output="screen",
    )


def generate_launch_description():
    world = LaunchConfiguration("world")
    use_sim_time = LaunchConfiguration("use_sim_time")

    xacro_path = PathJoinSubstitution([
        FindPackageShare("aha_description"), "urdf", "aha_robot.urdf.xacro",
    ])
    gazebo_launch = PathJoinSubstitution([
        FindPackageShare("aha_gazebo"), "launch", "gazebo.launch.py",
    ])

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
            "world": world,
            "headless": LaunchConfiguration("headless"),
        }.items(),
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "aha_robot",
            "-topic", "robot_description",
            "-z", "0.05",
        ],
        output="screen",
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    jsb = _spawner("joint_state_broadcaster")

    # Load remaining controllers after joint_state_broadcaster is active.
    load_rest = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jsb,
            on_exit=[_spawner(c) for c in CONTROLLERS_AFTER_JSB],
        )
    )

    # Let Gazebo resolve `model://astra_description/...` URIs from installed shares.
    # We add the parent of each pkg's share dir; Gazebo searches these roots
    # for a subdirectory matching the URI host part.
    share_roots = os.pathsep.join(
        os.path.dirname(os.path.join(get_package_prefix(p), "share", p))
        for p in ("astra_description", "aha_description")
    )
    existing = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_path = os.pathsep.join(x for x in (share_roots, existing) if x)

    # Optional per-squad subsystems. Default off — bringup is minimal.
    nav_launch = PathJoinSubstitution([
        FindPackageShare("aha_navigation"), "launch", "nav.launch.py",
    ])
    manip_launch = PathJoinSubstitution([
        FindPackageShare("aha_manipulation"), "launch", "manip.launch.py",
    ])
    perception_launch = PathJoinSubstitution([
        FindPackageShare("aha_perception"), "launch", "perception.launch.py",
    ])

    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav_launch),
        condition=IfCondition(LaunchConfiguration("use_nav")),
    )
    manip = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(manip_launch),
        condition=IfCondition(LaunchConfiguration("use_manip")),
    )
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(perception_launch),
        condition=IfCondition(LaunchConfiguration("use_perception")),
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="empty.sdf"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("use_nav", default_value="false"),
        DeclareLaunchArgument("use_manip", default_value="false"),
        DeclareLaunchArgument("use_perception", default_value="false"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
        gz,
        rsp,
        clock_bridge,
        spawn,
        jsb,
        load_rest,
        nav,
        manip,
        perception,
    ])
