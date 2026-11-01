"""Launch: Gazebo Harmonic with a chosen world (default: SOBITS RCJO2026).

Args:
  world     — rcjo2026 (default), or a filename under aha_gazebo/worlds/
  world_path — optional absolute SDF path, overrides world
  headless  — true/false. When true, appends `-s --headless-rendering`
              to gz_args, giving a server-only run suitable for CI or
              slow hosts (macOS).
  gz_args   — extra args appended verbatim after world path + '-r'.
"""

from pathlib import Path
import shlex

from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    LogInfo,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from aha_gazebo.world import load_preset, prepare_world, resource_path


def _resolve_world(context):
    explicit_path = LaunchConfiguration("world_path").perform(context)
    if explicit_path:
        world_path = Path(explicit_path)
        if not world_path.is_absolute():
            raise ValueError("world_path must be an absolute path to an expanded SDF")
    else:
        world_path = (
            Path(get_package_share_directory("aha_gazebo"))
            / "worlds"
            / LaunchConfiguration("world").perform(context)
        )
    if not world_path.is_file():
        raise FileNotFoundError(f"Gazebo world does not exist: {world_path}")
    return world_path


def _prepare_rcjo2026(context):
    preset = load_preset(
        Path(get_package_share_directory("aha_gazebo")) / "config" / "rcjo2026.json"
    )
    try:
        sobits = Path(get_package_share_directory("sobits_gazebo_worlds"))
        tmc = Path(get_package_share_directory("tmc_wrs_gz_worlds"))
    except PackageNotFoundError as error:
        raise RuntimeError(
            "SOBITS Japan Open dependencies are missing. Initialize the submodules with "
            "git submodule update --init --recursive, then rebuild the overlay workspace; or use world:=empty.sdf."
        ) from error
    roots = [sobits / "models", tmc / "models"]
    temporary, world = prepare_world(sobits, roots, preset)

    def cleanup(_context):
        temporary.cleanup()
        return []

    return world, [
        RegisterEventHandler(
            OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])
        ),
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            resource_path(roots, context.environment.get("GZ_SIM_RESOURCE_PATH", "")),
        ),
        LogInfo(msg="SOBITS Japan Open 2026 world expanded: " + str(world)),
    ]


def _launch_gazebo(context):
    headless = LaunchConfiguration("headless").perform(context).lower()
    if headless not in ("true", "false", "1", "0"):
        raise ValueError("headless must be true or false")
    setup = []
    if (
        not LaunchConfiguration("world_path").perform(context)
        and LaunchConfiguration("world").perform(context) == "rcjo2026"
    ):
        world_path, setup = _prepare_rcjo2026(context)
    else:
        world_path = _resolve_world(context)
    arguments = [str(world_path), "-r"]
    if headless in ("true", "1"):
        arguments.extend(["-s", "--headless-rendering"])
    command = shlex.join(arguments)
    extra = LaunchConfiguration("gz_args").perform(context)
    if extra:
        command += " " + extra
    gz_launch = PathJoinSubstitution(
        [
            FindPackageShare("ros_gz_sim"),
            "launch",
            "gz_sim.launch.py",
        ]
    )
    return [
        *setup,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={
                "gz_args": command,
            }.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value="rcjo2026"),
            DeclareLaunchArgument("world_path", default_value=""),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("gz_args", default_value=""),
            OpaqueFunction(function=_launch_gazebo),
        ]
    )
