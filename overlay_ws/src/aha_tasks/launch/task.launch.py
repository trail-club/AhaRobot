"""Subgoal task on the RCJO2026 simulation with fixed-map Nav2 and the web GUI.

`people:=true` adds the head camera bridge, person_perception (DGX Spark) and
head_look_at.py, which turns the head towards the people the skills look at.
"""

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


# Launch argument -> people.launch.py argument (DGX connection).
PEOPLE_OPTIONS = {"ssh_host": "10.99.0.1", "ssh_port": "2222", "ssh_user": ""}


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
    people = value("people") == "true"
    actions = [
        include(
            "aha_bringup",
            "sim.launch.py",
            {
                "world": "rcjo2026",
                "cameras": "true" if people else "false",
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
    if people:
        actions += [
            include("aha_perception", "camera_bridge.launch.py", {}),
            include(
                "aha_perception",
                "people.launch.py",
                {
                    "use_sim_time": "true",
                    **{name: value(name) for name in PEOPLE_OPTIONS},
                },
            ),
            # Wall clock as in sim_view.launch.py; it uses the latest TF.
            Node(
                package="aha_perception",
                executable="head_look_at.py",
                parameters=[
                    str(
                        Path(
                            get_package_share_directory("aha_perception"),
                            "config",
                            "head.yaml",
                        )
                    )
                ],
                output="screen",
            ),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "task", description="Task name in config/, or a JSON path"
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("gui_port", default_value="8765"),
            DeclareLaunchArgument(
                "people",
                default_value="false",
                description="Head cameras and person_perception on the DGX",
            ),
            *(
                DeclareLaunchArgument(k, default_value=v)
                for k, v in PEOPLE_OPTIONS.items()
            ),
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
