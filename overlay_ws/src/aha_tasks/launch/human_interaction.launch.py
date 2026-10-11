"""RCJO2026 navigation-only Human Robot Interaction demo, including simulation."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def task_exited(event, context):
    return [
        LogInfo(msg=f"Human interaction runner exited with code {event.returncode}"),
        EmitEvent(
            event=Shutdown(reason="Navigation flow ended; stopping simulation and Nav2")
        ),
    ]


def generate_launch_description():
    def include(package, filename, arguments=None):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare(package), "launch", filename])
            ),
            launch_arguments=(arguments or {}).items(),
        )

    runner = Node(
        package="aha_tasks",
        executable="human_interaction",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "--config",
            LaunchConfiguration("task_config"),
            "--startup-timeout",
            LaunchConfiguration("startup_timeout"),
            "--goal-timeout",
            LaunchConfiguration("goal_timeout"),
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("startup_timeout", default_value="300.0"),
            DeclareLaunchArgument("goal_timeout", default_value="600.0"),
            DeclareLaunchArgument(
                "task_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("aha_tasks"), "config", "human_interaction.json"]
                ),
            ),
            include(
                "aha_bringup",
                "sim.launch.py",
                {
                    "world": "rcjo2026",
                    "world_path": "",
                    "use_nav": "false",
                    "spawn_x": "-2.0",
                    "spawn_y": "1.5",
                    "spawn_z": "0.10",
                    "spawn_yaw": "0.0",
                    "use_sim_time": "true",
                    "cameras": "false",
                    "use_perception": "false",
                    "headless": LaunchConfiguration("headless"),
                },
            ),
            include("aha_navigation", "localized_nav.launch.py"),
            RegisterEventHandler(
                OnProcessExit(target_action=runner, on_exit=task_exited)
            ),
            runner,
        ]
    )
