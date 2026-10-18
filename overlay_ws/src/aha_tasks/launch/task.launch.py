"""Subgoal task on the RCJO2026 simulation with fixed-map Nav2 and the web GUI."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aha_tasks.core import load_spec
from aha_tasks.skills import LOCATION, resolve_pose

# Launch argument -> task_runner option.
RUNNER_OPTIONS = {
    "subgoals": "",
    "on_failure": "pause",
    "max_retries": "1",
    "startup_timeout": "300.0",
    "goal_timeout": "600.0",
}


def task_config(task):
    if task.endswith(".json"):
        return Path(task)
    return Path(get_package_share_directory("aha_tasks"), "config", f"{task}.json")


def include(package, filename, arguments):
    path = Path(get_package_share_directory(package), "launch", filename)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(path)),
        launch_arguments=arguments.items(),
    )


def task_exited(event, context):
    return [
        LogInfo(msg=f"Task runner exited with code {event.returncode}"),
        EmitEvent(event=Shutdown(reason="Task runner ended; stopping simulation")),
    ]


def launch_task(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    config = task_config(value("task"))
    spec = load_spec(config)
    if LOCATION not in spec.initial_state:
        raise ValueError(f"{config}: initial_state needs {LOCATION} as the spawn pose")
    x, y, yaw = (
        str(float(v)) for v in resolve_pose(spec, spec.initial_state[LOCATION])
    )
    runner = Node(
        package="aha_tasks",
        executable="task_runner",
        parameters=[{"use_sim_time": True}],
        arguments=[f"--config={config}"]
        + [f"--{name.replace('_', '-')}={value(name)}" for name in RUNNER_OPTIONS],
        output="screen",
    )
    return [
        include(
            "aha_bringup",
            "sim.launch.py",
            {
                "world": "rcjo2026",
                "cameras": "false",
                "headless": value("headless"),
                "spawn_x": x,
                "spawn_y": y,
                "spawn_yaw": yaw,
            },
        ),
        include(
            "aha_navigation",
            "localized_nav.launch.py",
            {"initial_pose": f"{x},{y},{yaw}"},
        ),
        RegisterEventHandler(OnProcessExit(target_action=runner, on_exit=task_exited)),
        runner,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "task", description="Task name in config/, or a JSON path"
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("gui_port", default_value="8765"),
            *(
                DeclareLaunchArgument(k, default_value=v)
                for k, v in RUNNER_OPTIONS.items()
            ),
            OpaqueFunction(function=launch_task),
            Node(
                package="aha_tasks",
                executable="task_monitor",
                arguments=["--port", LaunchConfiguration("gui_port")],
                output="screen",
            ),
        ]
    )
