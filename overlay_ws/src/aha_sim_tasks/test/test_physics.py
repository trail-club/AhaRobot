"""Actual YCB meshes must settle, respond to a push, and fall when unsupported."""

import json
import math
import os
import subprocess
import uuid
import xml.etree.ElementTree as ET

import pytest

from aha_sim_tasks.scene import expand_world, load_scene, package_share, sample_scene


@pytest.mark.parametrize("object_id", ["apple", "orange", "soup_can", "gelatin_box"])
def test_ycb_free_body_physics(tmp_path, object_id):
    probe = os.environ.get("YCB_PHYSICS_PROBE")
    if not probe:
        pytest.skip("Run with colcon test to build the Gazebo physics probe")
    share = package_share()
    scene = sample_scene(load_scene(), object_id)
    configuration = tmp_path / "scene.json"
    configuration.write_text(json.dumps(scene))
    root = ET.fromstring(
        expand_world(share / "worlds/task_tables.sdf.xacro", configuration, False)
    )
    world = root.find("world")
    world.remove(world.find("plugin[@name='aha_sim_tasks::EvaluationSystem']"))
    world_path = tmp_path / "physics.sdf"
    ET.ElementTree(root).write(world_path, encoding="unicode")
    environment = {
        **os.environ,
        "GZ_PARTITION": "aha-ycb-physics-" + uuid.uuid4().hex,
        "GZ_SIM_RESOURCE_PATH": str(package_share("tmc_wrs_gz_worlds") / "models"),
    }
    completed = subprocess.run(
        [probe, str(world_path)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=40,
        check=True,
    )
    measurements = json.loads(
        completed.stdout.split("YCB_PHYSICS=", 1)[1].splitlines()[0]
    )
    expected_height = scene["table_height"] + scene["object"]["size"][2] / 2
    assert abs(measurements["rest"][2] - expected_height) < 0.015
    assert math.dist(measurements["impulse"][:2], measurements["rest"][:2]) > 0.00001
    assert math.dist(measurements["pushed"][:2], measurements["rest"][:2]) < 0.15
    assert abs(measurements["pushed"][2] - expected_height) < 0.02
    assert 0 < measurements["dropped"][2] < 0.12
