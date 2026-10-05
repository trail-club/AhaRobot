"""Scene changes must move world geometry, evaluation targets, and policy together."""

from copy import deepcopy
from pathlib import Path
import json
import math
import sys
import subprocess
import xml.etree.ElementTree as ET

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from aha_sim_tasks.evaluation import load_tasks  # noqa: E402
from aha_sim_tasks.policies import ScriptedPolicy  # noqa: E402
from aha_sim_tasks.scene import expand_world, load_scene, scripted_distances  # noqa: E402


def test_scene_coordinate_changes_reach_world_targets_and_policy(tmp_path, monkeypatch):
    scene = load_scene(PACKAGE / "config/scene.json")
    scene["source_position"][0] += 0.2
    scene["destination_position"][0] += 0.3
    scene["robot_spawn"]["x"] += 0.1
    scene["entities"]["object_model"] = "renamed_fruit"
    scene["entities"]["object_link"] = "fruit_body"
    scene["entities"]["object_collision"] = "skin"
    configuration = tmp_path / "scene.json"
    configuration.write_text(json.dumps(scene))
    world = ET.fromstring(
        expand_world(PACKAGE / "worlds/task_tables.sdf.xacro", configuration)
    )
    apple = world.find("world/model[@name='renamed_fruit']")
    assert apple is not None
    assert [float(v) for v in apple.findtext("pose").split()[:3]] == scene[
        "source_position"
    ]
    assert apple.find("link[@name='fruit_body']/collision[@name='skin']") is not None
    plugin = world.find("world/plugin[@name='aha_sim_tasks::EvaluationSystem']")
    assert {key: plugin.findtext(key) for key in scene["entities"]} == scene["entities"]
    destination = world.findtext("world/model[@name='destination_table']/pose").split()
    assert [float(v) for v in destination[:2]] == scene["destination_position"][:2]
    tasks = load_tasks(PACKAGE / "config/tasks.json", scene)
    assert (
        tasks["place_apple"].parameters["object_target"]
        == scene["destination_position"]
    )
    assert tasks["approach_apple"].parameters["robot_target"][:2] == pytest.approx(
        [0.69, 0]
    )
    monkeypatch.setattr("aha_sim_tasks.policies.load_scene", lambda: scene)
    policy = ScriptedPolicy()
    policy.reset("place_apple", "Place.")
    assert (policy.approach_distance, policy.transfer_distance) == pytest.approx(
        [0.59, 1.09]
    )


def test_scripted_scene_rejects_unreachable_straight_line_target():
    scene = load_scene(PACKAGE / "config/scene.json")
    scene["source_position"][1] += 0.1
    with pytest.raises(ValueError, match="spawn heading"):
        scripted_distances(scene)


def test_scripted_distances_respect_rotated_spawn_heading():
    scene = deepcopy(load_scene(PACKAGE / "config/scene.json"))
    scene["robot_spawn"]["yaw"] = math.pi / 2
    for key in ("source_position", "destination_position"):
        x, y, z = scene[key]
        scene[key] = [-y, x, z]
    assert scripted_distances(scene) == pytest.approx([0.49, 0.89])


def test_real_robot_sdf_retains_both_configured_finger_collision_links(tmp_path):
    from ament_index_python.packages import get_package_share_directory

    robot = (
        Path(get_package_share_directory("aha_description"))
        / "urdf/aha_robot.urdf.xacro"
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        subprocess.check_output(["xacro", str(robot), "sim:=true"], text=True)
    )
    model = ET.fromstring(
        subprocess.check_output(["gz", "sdf", "-p", str(urdf)], text=True)
    ).find("model")
    scene = load_scene(PACKAGE / "config/scene.json")
    for key in ("right_finger_link", "left_finger_link"):
        name = scene["entities"][key]
        link = model.find(f"link[@name='{name}']")
        assert link is not None, f"Missing finger link after URDF conversion: {name}"
        assert link.findall("collision"), f"No collision geometry on {name}"
        joints = [
            joint for joint in model.findall("joint") if joint.findtext("child") == name
        ]
        assert len(joints) == 1 and joints[0].get("type") == "prismatic"
