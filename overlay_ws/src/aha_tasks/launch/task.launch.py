"""Subgoal task on the RCJO2026 simulation with fixed-map Nav2 and the web GUI.

head_look_at.py turns the head where the skills look (aha_tasks.gaze).
`people:=true` adds the head camera bridge and person_perception (DGX Spark).
`objects:=true` adds the head camera bridge and an object detector on
/aha/perception/objects (frame map) for the object skills (aha_tasks.objects),
and the identity TF world -> map: the fixed map is drawn in Gazebo's world
coordinates, the frame of aha_perception's furniture config of the arena.
`guests:=true` adds sim_guests, which moves the task's simulated guests, and
sim_sounds, which makes the sounds they make at the door.
`people_scene:=<name>` adds sim_people, which puts the posed people of
config/sim/people/<name>.json (or a JSON path) into Gazebo.
`object_scene:=<name>` adds sim_objects, which puts the objects of
config/sim/objects/<name>.json (or a JSON path) onto the furniture.

`e2e:=final` (default) runs everything for real: Nav2 drives the robot and
person_perception asks the models on the DGX. `e2e:=debug` swaps in the
simulation stand-ins below and keeps the simulation up when the runner exits;
the runner (and the people node) restart when they exit, so that a changed
node can be rerun without restarting Gazebo.

- navigation (debug: teleport): aha_tasks sim_teleport moves the robot in
  Gazebo and resets AMCL instead of the Nav2 planner, controller and BT.
- perception (debug: truth): aha_perception sim_person_detector.py publishes
  the people from Gazebo's poses instead of person_perception.py,
  sim_object_detector.py the objects instead of objects.launch.py, and sim_sounds
  publishes the guests' doorbell and knocks as detections instead of
  aha_audio_doorbell (no audio container).
- manipulation (debug: sim): aha_tasks sim_manipulation serves pick, place
  and handover by moving the objects in Gazebo. `none` (final) starts no
  server: the manipulation squad's servers are started on their own.
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
# e2e mode -> default pick, place and handover servers; final allows only none.
MANIPULATION = {"final": "none", "debug": "sim"}

# Launch arguments of the DGX workers. people.launch.py and objects.launch.py
# share the connection; each takes its own worker_command (the person and the
# object workers speak different protocols).
DGX_OPTIONS = {
    "ssh_host": "10.99.0.1",
    "ssh_port": "2222",
    "ssh_user": "",
    "worker_command": "",
    "object_worker_command": "",
}
DGX_CONNECTION = ("ssh_host", "ssh_port", "ssh_user")


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


def manipulation_choice(mode, manipulation=""):
    """The pick, place and handover servers of an e2e mode: sim or none."""
    chosen = manipulation or MANIPULATION[mode]
    if chosen not in MANIPULATION.values():
        raise ValueError(
            f"manipulation must be one of {sorted(MANIPULATION.values())}: {chosen!r}"
        )
    if mode == "final" and chosen != MANIPULATION["final"]:
        raise ValueError("e2e:=final starts no stand-in; use e2e:=debug for it")
    return chosen


def object_detector(perception, debug, value):
    """The object detector of objects:=true, publishing in map."""
    if perception == "truth":
        # Wall clock like sim_person_detector.py; it uses the image stamps.
        return Node(
            package="aha_perception",
            executable="sim_object_detector.py",
            parameters=[{"output_frame": "map"}],
            output="screen",
            respawn=True,
            respawn_delay=2.0,
        )
    return include(
        "aha_perception",
        "objects.launch.py",
        {
            "target_frame": "map",
            "use_sim_time": "true",
            "respawn": "true" if debug else "false",
            **{name: value(name) for name in DGX_CONNECTION},
            "worker_command": value("object_worker_command"),
        },
    )


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
    manipulation = manipulation_choice(
        value("e2e"), context.launch_configurations.get("manipulation", "")
    )
    people = value("people") == "true"
    objects = value("objects") == "true"
    if people and perception == "truth" and value("people_scene"):
        # sim_person_detector.py has no ClassifyPerson and only knows the guests.
        raise ValueError(
            "people_scene:= with people:=true needs perception:=model "
            "(the debug stand-in cannot judge posed people)"
        )
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
                "cameras": "true" if people or objects else "false",
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
    if manipulation == "sim":
        actions.append(
            Node(
                package="aha_tasks",
                executable="sim_manipulation",
                parameters=[{"use_sim_time": True}],
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
                **{name: value(name) for name in (*DGX_CONNECTION, "worker_command")},
            },
        )
    if people or objects:
        actions.append(include("aha_perception", "camera_bridge.launch.py", {}))
    if people:
        actions.append(people_node)
    if objects:
        actions += [
            object_detector(perception, debug, value),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="world_to_map",
                arguments=["--frame-id", "world", "--child-frame-id", "map"],
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
        # The microphone stream with the guests' doorbell and knocks, for
        # aha_audio_doorbell input:=topic (wall clock, like a microphone); with
        # perception truth, also the detections themselves.
        actions.append(
            Node(
                package="aha_tasks",
                executable="sim_sounds",
                parameters=[{"truth": perception == "truth"}],
                output="screen",
            )
        )
    people_scene = value("people_scene")
    if people_scene:
        path = (
            Path(people_scene)
            if people_scene.endswith(".json")
            else task_config(f"sim/people/{people_scene}")
        )
        if not path.is_file():
            raise ValueError(f"people_scene:={people_scene}: no {path}")
        actions.append(
            Node(
                package="aha_tasks",
                executable="sim_people",
                parameters=[{"use_sim_time": True}],
                arguments=[f"--config={path}"],
                output="screen",
            )
        )
    object_scene = value("object_scene")
    if object_scene:
        path = (
            Path(object_scene)
            if object_scene.endswith(".json")
            else task_config(f"sim/objects/{object_scene}")
        )
        if not path.is_file():
            raise ValueError(f"object_scene:={object_scene}: no {path}")
        # Spawns (or puts back) the objects once and exits.
        actions.append(
            Node(
                package="aha_tasks",
                executable="sim_objects",
                arguments=[f"--config={path}"],
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
                description="model | truth (debug only), with people:=true or "
                "objects:=true; empty: by e2e",
            ),
            DeclareLaunchArgument(
                "manipulation",
                default_value="",
                description="sim (debug only) | none: pick, place and handover "
                "servers; empty: by e2e",
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("gui_port", default_value="8765"),
            DeclareLaunchArgument(
                "people",
                default_value="false",
                description="Head cameras and person_perception on the DGX",
            ),
            DeclareLaunchArgument(
                "objects",
                default_value="false",
                description="Head cameras and the object detector (sim truth or model)",
            ),
            DeclareLaunchArgument(
                "guests",
                default_value="false",
                description="Simulated guests from config/sim/<task>.json",
            ),
            DeclareLaunchArgument(
                "people_scene",
                default_value="",
                description="Posed people from config/sim/people/<name>.json",
            ),
            DeclareLaunchArgument(
                "object_scene",
                default_value="",
                description="Objects from config/sim/objects/<name>.json",
            ),
            *(
                DeclareLaunchArgument(k, default_value=v)
                for k, v in DGX_OPTIONS.items()
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
