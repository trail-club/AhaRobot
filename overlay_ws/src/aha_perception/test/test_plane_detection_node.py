"""Message conversions of plane_detection_node.py (needs the ROS messages)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("aha_msgs")

from std_msgs.msg import Header  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from aha_perception.furniture import Match, parse_config  # noqa: E402
from aha_perception.surfaces import Surface  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plane_detection_node.py"


@pytest.fixture(scope="module")
def node_module():
    spec = importlib.util.spec_from_file_location("plane_detection_node", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_surface(sid, height):
    hull = np.array([[0.0, 0.0, height], [1.0, 0.0, height], [0.0, 1.0, height]])
    return Surface(
        id=sid,
        label="shelf",
        level=1,
        normal=np.array([0.0, 0.0, 1.0]),
        offset=-height,
        centroid=hull.mean(axis=0),
        hull=hull,
        area=0.5,
        observations=3,
        first_seen=1.0,
        last_seen=2.5,
        visible=True,
        vertical=False,
    )


def test_surface_array_carries_the_furniture_match(node_module):
    header = Header(frame_id="odom")
    surfaces = [make_surface(3, 0.42), make_surface(4, 0.9)]
    matches = {3: Match("cabinet", 2, 0.01, 0.1)}
    msg = node_module.make_surface_array(header, surfaces, matches)
    assert [(s.furniture, s.furniture_level) for s in msg.surfaces] == [
        ("cabinet", 2),
        ("", 0),
    ]
    assert msg.surfaces[0].plane.height == pytest.approx(0.42)
    plain = node_module.make_surface_array(header, surfaces)
    assert {(s.furniture, s.furniture_level) for s in plain.surfaces} == {("", 0)}


def test_furniture_markers(node_module):
    config = parse_config(
        {
            "frame": "map",
            "furniture": [
                {
                    "name": "cabinet",
                    "pose": {"x": 1.0, "y": 0.0},
                    "size": {"x": 0.4, "y": 0.8},
                    "levels": [0.4, 0.8],
                }
            ],
        }
    )
    msg = node_module.make_furniture_markers(config)
    assert msg.markers[0].action == Marker.DELETEALL
    outlines = [m for m in msg.markers if m.ns == "level"]
    labels = [m for m in msg.markers if m.ns == "label"]
    assert len(outlines) == len(labels) == 2
    assert all(m.header.frame_id == "map" for m in outlines + labels)
    assert len(outlines[1].points) == 5
    assert all(p.z == pytest.approx(0.8) for p in outlines[1].points)
    assert [m.text for m in labels] == ["cabinet 1", "cabinet 2"]
