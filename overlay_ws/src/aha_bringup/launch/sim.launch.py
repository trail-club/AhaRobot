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
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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
        "robot_description": Command(["xacro ", xacro_path, " sim:=true"]),
    }

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={"world": world}.items(),
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

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="empty.sdf"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        gz,
        rsp,
        clock_bridge,
        spawn,
        jsb,
        load_rest,
    ])
