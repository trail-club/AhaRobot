"""Launch construction and resource regression tests, without starting processes."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest

pytest.importorskip("launch")
from launch import LaunchContext  # noqa: E402
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable  # noqa: E402
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions  # noqa: E402

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "aha_gazebo"))


def load_launch(package, name):
    path = SOURCE / package / "launch" / name
    spec = importlib.util.spec_from_file_location(package + "_test_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "world,world_path,pose",
    [
        (None, "", ("-2.0", "1.5", "0.1", "0.0")),
        ("empty.sdf", "", ("0.0", "0.0", "0.05", "0.0")),
        ("home.sdf", "", ("0.0", "0.0", "0.05", "0.0")),
        (None, "/custom/world's file.sdf", ("0.0", "0.0", "0.05", "0.0")),
    ],
)
def test_standard_sim_defaults_and_inherited_resources(
    monkeypatch, world, world_path, pose
):
    module = load_launch("aha_bringup", "sim.launch.py")
    monkeypatch.setattr(
        module, "get_package_prefix", lambda package: "/install/" + package
    )
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda package: str(SOURCE / package)
    )
    context = LaunchContext()
    if world is not None:
        context.launch_configurations["world"] = world
    context.launch_configurations["world_path"] = world_path
    context.environment["GZ_SIM_RESOURCE_PATH"] = "/sobits/models:/tmc/models:/custom"
    description = module.generate_launch_description()
    for action in description.entities:
        if isinstance(action, (DeclareLaunchArgument, SetEnvironmentVariable)):
            action.execute(context)
    assert context.launch_configurations["world"] == (world or "rcjo2026")
    assert context.launch_configurations["world_path"] == world_path
    assert (
        tuple(
            context.launch_configurations["spawn_" + key]
            for key in ("x", "y", "z", "yaw")
        )
        == pose
    )
    roots = context.environment["GZ_SIM_RESOURCE_PATH"].split(os.pathsep)
    assert roots == [
        "/install/astra_description/share",
        "/install/aha_description/share",
        "/sobits/models",
        "/tmc/models",
        "/custom",
    ]


@pytest.mark.parametrize("headless", ["true", "false"])
def test_gazebo_accepts_expanded_absolute_path(tmp_path, headless):
    module = load_launch("aha_gazebo", "gazebo.launch.py")
    world = tmp_path / "world with spaces.sdf"
    world.write_text('<sdf version="1.8"><world name="fixture"/></sdf>')
    context = LaunchContext()
    context.launch_configurations.update(
        world_path=str(world), headless=headless, gz_args=""
    )
    actions = module._launch_gazebo(context)
    arguments = dict(actions[0].launch_arguments)
    command = perform_substitutions(
        context, normalize_to_list_of_substitutions(arguments["gz_args"])
    )
    assert "'" + str(world) + "'" in command
    assert ("--headless-rendering" in command) == (headless == "true")


def test_gazebo_legacy_world_path_still_resolves(tmp_path, monkeypatch):
    module = load_launch("aha_gazebo", "gazebo.launch.py")
    (tmp_path / "worlds").mkdir()
    world = tmp_path / "worlds" / "empty.sdf"
    world.write_text('<sdf version="1.8"><world name="empty"/></sdf>')
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda package: str(tmp_path)
    )
    context = LaunchContext()
    context.launch_configurations.update(
        world_path="", world="empty.sdf", headless="false", gz_args=""
    )
    action = module._launch_gazebo(context)[0]
    assert str(world) in perform_substitutions(
        context,
        normalize_to_list_of_substitutions(dict(action.launch_arguments)["gz_args"]),
    )


def test_gazebo_rejects_relative_override():
    module = load_launch("aha_gazebo", "gazebo.launch.py")
    context = LaunchContext()
    context.launch_configurations.update(
        world_path="not-absolute.sdf", headless="false"
    )
    with pytest.raises(ValueError, match="absolute"):
        module._launch_gazebo(context)


def test_rcjo2026_compatibility_launch_uses_standard_sim():
    module = load_launch("aha_sobits_bringup", "rcjo2026.launch.py")
    action = module.generate_launch_description().entities[0]
    assert dict(action.launch_arguments) == {"world": "rcjo2026"}


def test_standard_sim_rejects_nonfinite_cli_pose():
    module = load_launch("aha_bringup", "sim.launch.py")
    context = LaunchContext()
    context.launch_configurations.update(
        spawn_x="inf", spawn_y="1.5", spawn_z="0.1", spawn_yaw="0.0"
    )
    with pytest.raises(ValueError, match="spawn_x must be finite"):
        module._validate_spawn(context)


@pytest.mark.parametrize("headless", ["1", "0", "invalid"])
def test_gazebo_headless_values(tmp_path, headless):
    module = load_launch("aha_gazebo", "gazebo.launch.py")
    world = tmp_path / "world.sdf"
    world.write_text('<sdf version="1.8"><world name="fixture"/></sdf>')
    context = LaunchContext()
    context.launch_configurations.update(
        world_path=str(world), headless=headless, gz_args="--verbose"
    )
    if headless == "invalid":
        with pytest.raises(ValueError, match="headless"):
            module._launch_gazebo(context)
        return
    action = module._launch_gazebo(context)[0]
    command = perform_substitutions(
        context,
        normalize_to_list_of_substitutions(dict(action.launch_arguments)["gz_args"]),
    )
    assert ("--headless-rendering" in command) == (headless == "1")
    assert command.endswith(" --verbose")


def test_gazebo_missing_world_fails_before_launch(tmp_path):
    module = load_launch("aha_gazebo", "gazebo.launch.py")
    context = LaunchContext()
    context.launch_configurations.update(
        world_path=str(tmp_path / "missing.sdf"), headless="false"
    )
    with pytest.raises(FileNotFoundError, match="Gazebo world does not exist"):
        module._launch_gazebo(context)


def test_prepared_world_survives_until_cleanup(tmp_path):
    import aha_gazebo.world as module

    (tmp_path / "worlds").mkdir()
    source = tmp_path / "worlds" / "fixture.world.xacro"
    source.write_text('<sdf version="1.8"><world name="fixture"/></sdf>')
    import hashlib

    preset = {
        "world_source": source.name,
        "world_name": "fixture",
        "world_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    temporary, world = module.prepare_world(tmp_path, [], preset)
    try:
        assert world.is_file()
        assert world.read_text().find('name="fixture"') != -1
    finally:
        temporary.cleanup()
    assert not world.parent.exists()


def test_world_preparation_failure_removes_temporary_directory(tmp_path, monkeypatch):
    import aha_gazebo.world as module

    destinations = []

    def fail_expansion(source, destination, *args, **kwargs):
        destinations.append(destination)
        raise ValueError("invalid world")

    monkeypatch.setattr(module, "expand_world", fail_expansion)
    with pytest.raises(ValueError, match="invalid world"):
        module.prepare_world(
            tmp_path,
            [],
            {
                "world_source": "fixture.world.xacro",
                "world_name": "fixture",
                "world_sha256": "0" * 64,
            },
        )
    assert len(destinations) == 1
    assert not destinations[0].parent.exists()


def test_standard_sim_preserves_explicit_pose(monkeypatch):
    module = load_launch("aha_bringup", "sim.launch.py")
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda package: str(SOURCE / package)
    )
    monkeypatch.setattr(
        module, "get_package_prefix", lambda package: "/install/" + package
    )
    context = LaunchContext()
    context.launch_configurations.update(spawn_x="3.0", spawn_y="-1.0")
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    assert context.launch_configurations["spawn_x"] == "3.0"
    assert context.launch_configurations["spawn_y"] == "-1.0"
    assert context.launch_configurations["spawn_z"] == "0.1"


def test_standard_clock_bridge_can_be_supplied_by_parent(monkeypatch):
    from launch_ros.actions import Node

    module = load_launch("aha_bringup", "sim.launch.py")
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda package: str(SOURCE / package)
    )
    monkeypatch.setattr(
        module, "get_package_prefix", lambda package: "/install/" + package
    )
    context = LaunchContext()
    description = module.generate_launch_description()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    clock_bridge = next(
        action
        for action in description.entities
        if isinstance(action, Node)
        and perform_substitutions(
            context, normalize_to_list_of_substitutions(action.node_executable)
        )
        == "parameter_bridge"
    )
    assert context.launch_configurations["bridge_clock"] == "true"
    assert clock_bridge.condition.evaluate(context)
    context.launch_configurations["bridge_clock"] = "false"
    assert not clock_bridge.condition.evaluate(context)


def test_default_gazebo_expands_arena_and_preserves_resources(tmp_path, monkeypatch):
    import hashlib
    from launch.actions import IncludeLaunchDescription
    from launch.events import Shutdown

    module = load_launch("aha_gazebo", "gazebo.launch.py")
    sobits = tmp_path / "sobits_gazebo_worlds"
    (sobits / "worlds").mkdir(parents=True)
    (sobits / "models").mkdir()
    tmc = tmp_path / "tmc_wrs_gz_worlds"
    (tmc / "models").mkdir(parents=True)
    world = sobits / "worlds/rcjo2026_arena.world.xacro"
    world.write_text('<sdf version="1.8"><world name="rcjo2026_arena"/></sdf>')
    preset = {
        "world_source": world.name,
        "world_name": "rcjo2026_arena",
        "world_sha256": hashlib.sha256(world.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(module, "load_preset", lambda path: preset)
    shares = {
        "aha_gazebo": SOURCE / "aha_gazebo",
        "sobits_gazebo_worlds": sobits,
        "tmc_wrs_gz_worlds": tmc,
    }
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda package: str(shares[package])
    )
    context = LaunchContext()
    context.environment["GZ_SIM_RESOURCE_PATH"] = "/robot/share:/custom"
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    actions = module._launch_gazebo(context)
    assert isinstance(actions[-1], IncludeLaunchDescription)
    command = perform_substitutions(
        context,
        normalize_to_list_of_substitutions(
            dict(actions[-1].launch_arguments)["gz_args"]
        ),
    )
    import shlex

    expanded = Path(shlex.split(command)[0])
    assert expanded.is_file()
    try:
        actions[1].execute(context)
        assert context.environment["GZ_SIM_RESOURCE_PATH"].split(os.pathsep) == [
            str(sobits / "models"),
            str(tmc / "models"),
            "/robot/share",
            "/custom",
        ]
    finally:
        for cleanup in actions[0].event_handler.handle(
            Shutdown(reason="test"), context
        ):
            cleanup.execute(context)
    assert not expanded.exists()


def test_missing_default_arena_has_actionable_error(monkeypatch):
    from ament_index_python.packages import PackageNotFoundError

    module = load_launch("aha_gazebo", "gazebo.launch.py")

    def share(package):
        if package == "aha_gazebo":
            return str(SOURCE / package)
        raise PackageNotFoundError(package)

    monkeypatch.setattr(module, "get_package_share_directory", share)
    context = LaunchContext()
    context.launch_configurations.update(
        world="rcjo2026", world_path="", headless="false", gz_args=""
    )
    with pytest.raises(RuntimeError, match="git submodule update"):
        module._launch_gazebo(context)
