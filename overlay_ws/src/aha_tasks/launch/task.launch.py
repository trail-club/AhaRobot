"""Subgoal task on the RCJO2026 simulation with fixed-map Nav2 and the web GUI.

`people:=true` adds the head camera bridge, person_perception (DGX Spark) and
head_look_at.py, which turns the head towards the people the skills look at.
`guests:=true` adds sim_guests, which moves the task's simulated guests.

`e2e:=final` (default) runs everything for real: Nav2 drives the robot and
person_perception asks the models on the DGX. `e2e:=debug` swaps in the
simulation stand-ins below and keeps the simulation up when the runner exits;
the runner (and the people node) restart when they exit, so that a changed
node can be rerun without restarting Gazebo.

- navigation (debug: teleport): aha_tasks sim_teleport moves the robot in
  Gazebo and resets AMCL instead of the Nav2 planner, controller and BT.
- perception (debug: truth): aha_perception sim_person_detector.py publishes
  the people from Gazebo's poses instead of person_perception.py.
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
from aha_tasks.guests import load_config as load_scene
from aha_tasks.skills import LOCATION, resolve_pose

# Launch argument -> task_runner option.
RUNNER_OPTIONS = {
    "subgoals": "",
    "on_failure": "pause",
    "max_retries": "1",
    "startup_timeout": "300.0",
    "goal_timeout": "600.0",
}


# e2e mode -> default navigation and perception; final allows only the real ones.
E2E_DEFAULTS = {
    "final": {"navigation": "nav2", "perception": "model"},
    "debug": {"navigation": "teleport", "perception": "truth"},
}
CHOICES = {"navigation": ("nav2", "teleport"), "perception": ("model", "truth")}

# Launch argument -> people.launch.py argument (DGX connection).
PEOPLE_OPTIONS = {
    "ssh_host": "10.99.0.1",
    "ssh_port": "2222",
    "ssh_user": "",
    "worker_command": "",
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


def e2e_choices(mode, navigation="", perception=""):
    """(navigation, perception) of an e2e mode; ValueError for invalid choices."""
    if mode not in E2E_DEFAULTS:
        raise ValueError(f"e2e must be one of {sorted(E2E_DEFAULTS)}: {mode!r}")
    chosen = {
        "navigation": navigation or E2E_DEFAULTS[mode]["navigation"],
        "perception": perception or E2E_DEFAULTS[mode]["perception"],
    }
    for name, value in chosen.items():
        if value not in CHOICES[name]:
            raise ValueError(f"{name} must be one of {CHOICES[name]}: {value!r}")
        if mode == "final" and value != E2E_DEFAULTS["final"][name]:
            raise ValueError(
                f"e2e:=final runs the real {name} ({E2E_DEFAULTS['final'][name]}); "
                f"use e2e:=debug for {name}:={value}"
            )
    return chosen["navigation"], chosen["perception"]


def task_exited(event, context):
    return [
        LogInfo(msg=f"Task runner exited with code {event.returncode}"),
        EmitEvent(event=Shutdown(reason="Task runner ended; stopping simulation")),
    ]


def launch_task(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    debug = value("e2e") == "debug"
    navigation, perception = e2e_choices(
        value("e2e"), value("navigation"), value("perception")
    )
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
        # Debug: a stopped runner comes back with the changed code; the
        # simulation stays up.
        respawn=debug,
        respawn_delay=2.0,
    )
    people = value("people") == "true"
    scene = task_config(f"sim/{spec.task}")
    guests = value("guests") == "true"
    if guests and not scene.is_file():
        raise ValueError(f"guests:=true: task {spec.task} has no {scene}")
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
            {
                "initial_pose": f"{x},{y},{yaw}",
                "navigation": "true" if navigation == "nav2" else "false",
            },
        ),
    ]
    if not debug:
        actions.append(
            RegisterEventHandler(
                OnProcessExit(target_action=runner, on_exit=task_exited)
            )
        )
    actions.append(runner)
    guest_scene = load_scene(scene) if guests else None
    if navigation == "teleport":
        teleport = {"use_sim_time": True, "speed": 0.0}
        if guest_scene is not None:
            # Not faster than the guests walk, who would otherwise still be on
            # their way where the robot looks for them.
            teleport.update(
                speed=float(guest_scene["walk_speed"]), world=guest_scene["world"]
            )
        actions.append(
            Node(
                package="aha_tasks",
                executable="sim_teleport",
                parameters=[teleport],
                output="screen",
                # It exits when no Gazebo world shows up in time.
                respawn=True,
                respawn_delay=2.0,
            )
        )
    if people and perception == "truth":
        # Wall clock like sim_object_detector.py; it uses the image stamps.
        people_node = Node(
            package="aha_perception",
            executable="sim_person_detector.py",
            parameters=[
                {
                    "models": list(guest_scene["guests"])
                    if guest_scene is not None
                    else ["guest*"]
                }
            ],
            output="screen",
            respawn=True,
            respawn_delay=2.0,
        )
    elif people:
        people_node = include(
            "aha_perception",
            "people.launch.py",
            {
                "use_sim_time": "true",
                "respawn": "true" if debug else "false",
                **{name: value(name) for name in PEOPLE_OPTIONS},
            },
        )
    if people:
        actions += [
            include("aha_perception", "camera_bridge.launch.py", {}),
            people_node,
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
    if guests:
        actions.append(
            Node(
                package="aha_tasks",
                executable="sim_guests",
                parameters=[{"use_sim_time": True}],
                arguments=[f"--config={scene}"],
                output="screen",
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "task", description="Task name in config/, or a JSON path"
            ),
            DeclareLaunchArgument(
                "e2e",
                default_value="final",
                choices=sorted(E2E_DEFAULTS),
                description="final: Nav2 and the models; debug: simulation stand-ins",
            ),
            DeclareLaunchArgument(
                "navigation",
                default_value="",
                description="nav2 | teleport (debug only); empty: by e2e",
            ),
            DeclareLaunchArgument(
                "perception",
                default_value="",
                description="model | truth (debug only), with people:=true; empty: by e2e",
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("gui_port", default_value="8765"),
            DeclareLaunchArgument(
                "people",
                default_value="false",
                description="Head cameras and person_perception on the DGX",
            ),
            DeclareLaunchArgument(
                "guests",
                default_value="false",
                description="Simulated guests from config/sim/<task>.json",
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
