"""Preparation tests that never start Gazebo or ROS transport."""

import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_gazebo.world import expand_world, load_preset, resource_path


PRESET = Path(__file__).resolve().parents[1] / "config" / "rcjo2026.json"


def test_default_pose_is_inside_bedroom_not_origin():
    preset = load_preset(PRESET)
    assert preset["spawn"] == {"x": -2.0, "y": 1.5, "z": 0.10, "yaw": 0.0}


@pytest.mark.parametrize("key", ["x", "y", "z", "yaw"])
def test_reject_nonfinite_preset(tmp_path, key):
    preset = load_preset(PRESET)
    preset["spawn"][key] = float("inf")
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(preset))
    with pytest.raises(ValueError, match=f"spawn_{key} must be finite"):
        load_preset(path)


def test_resource_path_preserves_and_deduplicates():
    assert resource_path(
        ["/sobits/models", "/tmc/models"],
        os.pathsep.join(["/existing", "/sobits/models"]),
    ) == os.pathsep.join(["/sobits/models", "/tmc/models", "/existing"])
    assert resource_path(["/sobits/models"], "") == "/sobits/models"


@pytest.fixture
def source(tmp_path):
    pytest.importorskip("xacro")
    source = tmp_path / "arena.world.xacro"
    source.write_text("""<sdf version="1.8" xmlns:xacro="http://www.ros.org/wiki/xacro">
      <xacro:property name="name" value="fixture"/>
      <world name="rcjo2026_arena"><include><uri>model://${name}</uri></include></world>
    </sdf>""")
    model = tmp_path / "models" / "fixture"
    model.mkdir(parents=True)
    (model / "model.config").write_text("<model/>")
    return source


def test_expands_xacro_before_sdf_is_returned(source, tmp_path):
    result = expand_world(
        source, tmp_path / "expanded.sdf", "rcjo2026_arena", [tmp_path / "models"]
    )
    assert (
        ET.parse(result).getroot().find("world/include/uri").text == "model://fixture"
    )
    assert "${" not in result.read_text()
    assert "<xacro:" not in result.read_text()


def test_missing_model_fails_before_writing_world(source, tmp_path):
    output = tmp_path / "expanded.sdf"
    with pytest.raises(FileNotFoundError, match="Unresolved world model"):
        expand_world(source, output, "rcjo2026_arena", [])
    assert not output.exists()


def test_wrong_world_fails_before_writing_world(source, tmp_path):
    with pytest.raises(ValueError, match="Expected world"):
        expand_world(source, tmp_path / "expanded.sdf", "wrong", [tmp_path / "models"])


def test_missing_world_fails_clearly(tmp_path):
    pytest.importorskip("xacro")
    with pytest.raises(FileNotFoundError, match="Missing pinned SOBITS world"):
        expand_world(tmp_path / "missing", tmp_path / "out", "rcjo2026_arena", [])


def test_pinned_world_hash_is_required_when_supplied(source, tmp_path):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expand_world(
        source,
        tmp_path / "valid.sdf",
        "rcjo2026_arena",
        [tmp_path / "models"],
        expected_sha256=digest,
    )
    source.write_text(source.read_text() + "\n<!-- changed -->")
    with pytest.raises(ValueError, match="checksum differs"):
        expand_world(
            source,
            tmp_path / "invalid.sdf",
            "rcjo2026_arena",
            [tmp_path / "models"],
            expected_sha256=digest,
        )
    assert not (tmp_path / "invalid.sdf").exists()
