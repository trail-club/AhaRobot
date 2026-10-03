"""Pure preparation helpers; no Gazebo server or ROS nodes are started here."""

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


XACRO_NAMESPACE = "http://www.ros.org/wiki/xacro"
SPAWN_COORDINATES = ("x", "y", "z", "yaw")


def validate_spawn(spawn):
    """Reject invalid positions from both the preset and launch overrides."""
    for key in SPAWN_COORDINATES:
        if not math.isfinite(float(spawn[key])):
            raise ValueError(f"spawn_{key} must be finite")


def load_preset(path):
    preset = json.loads(Path(path).read_text(encoding="utf-8"))
    if preset["world_source"] != "rcjo2026_arena.world.xacro":
        raise ValueError("Only the pinned RCJO2026 world is supported by this preset")
    if preset["world_name"] != "rcjo2026_arena":
        raise ValueError("Unexpected world name in the RCJO2026 preset")
    digest = preset["world_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid world SHA256 in preset")
    validate_spawn(preset["spawn"])
    return preset


def resource_path(roots, existing=""):
    """Prepend model directories without dropping caller or AhaRobot resources."""
    return os.pathsep.join(
        dict.fromkeys(
            str(item) for item in [*roots, *existing.split(os.pathsep)] if item
        )
    )


def expand_world(source, destination, expected_name, model_roots, expected_sha256=None):
    """Expand xacro and fail before startup on an invalid world or missing model."""
    import xacro

    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Missing pinned SOBITS world: {source}")
    if expected_sha256 is not None:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                "SOBITS world checksum differs from the pinned RCJO2026 preset; "
                "restore the pinned dependency before using its default spawn"
            )
    xml = xacro.process_file(str(source)).toxml()
    _validate_sdf(xml, expected_name, model_roots)
    destination.write_text(xml, encoding="utf-8")
    return destination


def _validate_sdf(xml, expected_name, model_roots):
    """Check the expanded world and resolve its top-level model includes."""
    root = ET.fromstring(xml)
    if root.tag != "sdf" or len(root.findall("world")) != 1:
        raise ValueError("Expected an SDF document containing exactly one world")
    world = root.find("world")
    if world.get("name") != expected_name:
        raise ValueError(f"Expected world {expected_name}")
    for element in root.iter():
        if element.tag.startswith("{" + XACRO_NAMESPACE + "}"):
            raise ValueError("Unexpanded xacro remains in world")
    model_roots = tuple(Path(directory) for directory in model_roots)
    for uri in root.findall(".//include/uri"):
        value = (uri.text or "").strip()
        if not value.startswith("model://"):
            raise ValueError(f"Unexpected non-local world model: {value}")
        model = value[len("model://") :]
        if not any(
            (directory / model / "model.config").is_file() for directory in model_roots
        ):
            raise FileNotFoundError(f"Unresolved world model: {value}")


def prepare_world(sobits, roots, preset):
    """Keep the expanded world alive until launch shutdown; clean up on failure."""
    temporary = TemporaryDirectory(prefix="aha-rcjo2026-")
    try:
        world = expand_world(
            Path(sobits) / "worlds" / preset["world_source"],
            Path(temporary.name) / "rcjo2026_arena.sdf",
            preset["world_name"],
            roots,
            expected_sha256=preset["world_sha256"],
        )
    except Exception:
        temporary.cleanup()
        raise
    return temporary, world
