"""Launch the evaluation scene and robot, without a policy or scorer."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from aha_sim_tasks.scene import expand_world, load_scene


def _launch_scene(context):
    share = Path(get_package_share_directory("aha_sim_tasks"))
    configuration = share / "config/scene.json"
    scene = load_scene(configuration)
    temporary = TemporaryDirectory(prefix="aha-task-scene-")
    world = Path(temporary.name) / "task_tables.sdf"
    try:
        world.write_text(
            expand_world(share / "worlds/task_tables.sdf.xacro", configuration)
        )
    except Exception:
        temporary.cleanup()
        raise

    def cleanup(_context):
        temporary.cleanup()
        return []

    return [
        RegisterEventHandler(
            OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("aha_bringup"), "launch", "sim.launch.py"]
                )
            ),
            launch_arguments={
                "world_path": str(world),
                "headless": LaunchConfiguration("headless"),
                "bridge_clock": "false",
                "activate_controllers_as_group": "true",
                "robot_name": scene["entities"]["robot_model"],
                **{
                    "spawn_" + key: str(value)
                    for key, value in scene["robot_spawn"].items()
                },
            }.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="true"),
            SetEnvironmentVariable(
                "GZ_SIM_SYSTEM_PLUGIN_PATH",
                [
                    os.path.join(get_package_prefix("aha_sim_tasks"), "lib"),
                    os.pathsep,
                    EnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", default_value=""),
                ],
            ),
            OpaqueFunction(function=_launch_scene),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/evaluation/state@std_msgs/msg/String[gz.msgs.StringMsg",
                    "/evaluation/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                ],
                remappings=[("/evaluation/clock", "/clock")],
                output="screen",
            ),
        ]
    )
