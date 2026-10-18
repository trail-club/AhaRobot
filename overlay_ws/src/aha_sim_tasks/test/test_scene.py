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
from aha_sim_tasks.scene import load_catalog, sample_scene  # noqa: E402


def test_scene_coordinate_changes_reach_world_targets_and_policy(tmp_path, monkeypatch):
    scene = load_scene(PACKAGE / "config/scene.json")
    scene["source_position"][0] += 0.2
    scene["destination_position"][0] += 0.3
    scene["table_size"][:2] = [0.6, 0.8]
    scene["source_table_position"][0] += 0.2
    scene["destination_table_position"][0] += 0.3
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
    assert [float(v) for v in destination[:2]] == [
        *scene["destination_table_position"][:2],
        scene["destination_position"][2],
    ][:2]
    size = world.findtext(
        "world/model[@name='destination_table']/link/collision/geometry/box/size"
    )
    assert [float(v) for v in size.split()] == scene["table_size"]
    tasks = load_tasks(PACKAGE / "config/tasks.json", scene)
    assert tasks["place_apple"].parameters["object_target"] == [
        *scene["destination_table_position"][:2],
        scene["destination_position"][2],
    ]
    assert tasks["place_apple"].parameters["xy_tolerance"] == [0.3, 0.4]
    assert tasks["approach_apple"].parameters["robot_target"][:2] == pytest.approx(
        [0.69, 0]
    )
    monkeypatch.setattr("aha_sim_tasks.policies.load_scene", lambda: scene)
    policy = ScriptedPolicy()
    policy.reset("place_apple", "Place.")
    assert [target[0] for target in policy.targets] == pytest.approx([0.69, 1.59])


def test_scripted_scene_rejects_unreachable_straight_line_target():
    scene = load_scene(PACKAGE / "config/scene.json")
    scene["source_position"][1] += 0.1
    with pytest.raises(ValueError, match="spawn heading"):
        scripted_distances(scene)


@pytest.mark.parametrize("ground_truth", [True, False])
def test_world_truth_publication_is_opt_in(ground_truth):
    world = ET.fromstring(
        expand_world(
            PACKAGE / "worlds/task_tables.sdf.xacro",
            PACKAGE / "config/scene.json",
            cameras=False,
            ground_truth=ground_truth,
        )
    )
    assert (
        world.findtext(
            "world/plugin[@name='aha_sim_tasks::EvaluationSystem']/ground_truth"
        )
        == str(ground_truth).lower()
    )


def test_scripted_distances_respect_rotated_spawn_heading():
    scene = deepcopy(load_scene(PACKAGE / "config/scene.json"))
    scene["robot_spawn"]["yaw"] = math.pi / 2
    scene["approach_yaw"] = math.pi / 2
    for key in ("source_position", "destination_position"):
        x, y, z = scene[key]
        scene[key] = [-y, x, z]
    assert scripted_distances(scene) == pytest.approx([0.49, 1.29])


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


@pytest.mark.parametrize("cameras", [True, False])
def test_rendering_and_robot_sensors_can_be_disabled_together(tmp_path, cameras):
    from ament_index_python.packages import get_package_share_directory

    world = ET.fromstring(
        expand_world(
            PACKAGE / "worlds/task_tables.sdf.xacro",
            PACKAGE / "config/scene.json",
            cameras,
        )
    )
    sensors = world.find("world/plugin[@name='gz::sim::systems::Sensors']")
    assert (sensors is not None) == cameras
    robot = (
        Path(get_package_share_directory("aha_description"))
        / "urdf/aha_robot.urdf.xacro"
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        subprocess.check_output(
            ["xacro", str(robot), "sim:=true", "cameras:=" + str(cameras).lower()],
            text=True,
        )
    )
    model = ET.fromstring(
        subprocess.check_output(["gz", "sdf", "-p", str(urdf)], text=True)
    )
    names = {
        sensor.attrib["name"]
        for sensor in model.findall(".//sensor")
        if sensor.attrib["type"] in ("camera", "rgbd_camera")
    }
    assert names == (
        {"head_camera", "left_wrist_camera", "right_wrist_camera"} if cameras else set()
    )


def test_seeded_scene_is_repeatable_and_keeps_clutter_separated():
    base = load_scene(PACKAGE / "config/scene.json")
    for name in load_catalog():
        for seed in range(30):
            scene = sample_scene(base, name, seed=seed, randomize=True, distractors=3)
            assert scene == sample_scene(
                base, name, seed=seed, randomize=True, distractors=3
            )
            assert scene["source_table_position"] == base["source_table_position"]
            assert (
                scene["destination_table_position"]
                == base["destination_table_position"]
            )
            items = [(scene["source_position"], scene["object"]["radius"])]
            items += [
                (item["position"], item["object"]["radius"])
                for item in scene["distractors"]
            ]
            center = scene["source_table_position"]
            for i, (position, radius) in enumerate(items):
                assert all(
                    abs(position[axis] - center[axis]) + radius
                    <= scene["table_size"][axis] / 2
                    for axis in range(2)
                )
                for other, other_radius in items[i + 1 :]:
                    assert (
                        math.dist(position[:2], other[:2])
                        > radius + other_radius + 0.02
                    )
            assert base["source_position"] != scene["source_position"]


@pytest.mark.parametrize("name", ["apple", "orange", "soup_can", "gelatin_box"])
def test_ycb_geometry_preserves_scale_mass_and_inertial_offset(tmp_path, name):
    scene = sample_scene(load_scene(PACKAGE / "config/scene.json"), name)
    config = tmp_path / "scene.json"
    config.write_text(json.dumps(scene))
    world = ET.fromstring(
        expand_world(PACKAGE / "worlds/task_tables.sdf.xacro", config, False)
    )
    model = world.find("world/model[@name='apple']")
    link = model.find("link")
    assert float(link.findtext("inertial/mass")) == scene["object"]["mass"]
    assert link.findtext("visual/geometry/mesh/uri").startswith(
        "model://" + scene["object"]["model"]
    )
    assert link.findtext("collision/geometry/mesh/uri").endswith("nontextured.stl")
    assert link.find("collision/geometry/mesh/scale") is None
    assert link.find("collision/geometry/sphere") is None
    assert model.findall("joint") == []
    assert (
        float(link.findtext("collision/surface/friction/ode/mu"))
        == scene["physics"]["object_friction"]
    )
    table = world.find("world/model[@name='source_table']")
    assert len(table.findall("link")) == 5
    assert scene["table_height"] == 0.74


def test_task_lift_initialization_does_not_expand_robot_joint_limits():
    from ament_index_python.packages import get_package_share_directory

    robot = (
        Path(get_package_share_directory("aha_description"))
        / "urdf/aha_robot.urdf.xacro"
    )
    urdf = ET.fromstring(
        subprocess.check_output(
            ["xacro", str(robot), "sim:=true", "initial_lift:=0.02"], text=True
        )
    )
    for name in ("joint_l1", "joint_r1"):
        assert (
            float(
                urdf.find(
                    f"ros2_control/joint[@name='{name}']/state_interface[@name='position']/param"
                ).text
            )
            == 0.02
        )
        assert urdf.find(f"joint[@name='{name}']/limit").get("lower") == "0"
