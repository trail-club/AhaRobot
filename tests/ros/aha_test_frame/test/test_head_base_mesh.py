"""The head base mesh cut at build time (scripts/crop_head_base.py) against
the frame model's collision box, the frame model's mesh URIs (needs the
installed aha_test_frame and aha_description: colcon test), and the cut on
synthetic triangles."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

from aha_test_frame.frame import HEAD_BASE_BOX, HEAD_BASE_MESH, frame_description

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crop_head_base.py"
SHARE = Path(get_package_share_directory("aha_test_frame"))
FRAME_CONFIG = SHARE / "config" / "frame_table_202610.yaml"


@pytest.fixture(scope="module")
def crop():
    spec = importlib.util.spec_from_file_location("crop_head_base", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_file(uri: str) -> Path:
    package, _, relative = uri.removeprefix("package://").partition("/")
    return Path(get_package_share_directory(package)) / relative


def test_installed_mesh_fills_the_collision_box(crop):
    mesh = package_file(HEAD_BASE_MESH)
    assert mesh == SHARE / "meshes" / "head_base.stl"
    points = np.array(crop.read_stl(mesh)).reshape(-1, 3)
    center, size = (np.array(v) for v in HEAD_BASE_BOX)
    np.testing.assert_allclose(points.min(axis=0), center - size / 2, atol=1e-3)
    np.testing.assert_allclose(points.max(axis=0), center + size / 2, atol=1e-3)


def test_frame_model_meshes_exist():
    uris = set(
        re.findall(r'filename="(package://[^"]+)"', frame_description(FRAME_CONFIG))
    )
    assert HEAD_BASE_MESH in uris
    missing = [uri for uri in sorted(uris) if not package_file(uri).is_file()]
    assert missing == []


ORIGIN = (0.05, 0.0, 1.429)  # joint_head_pan
KEPT = ((0.01, 0.0, 1.47), (0.03, 0.0, 1.48), (0.02, 0.01, 1.50))
BELOW = ((0.01, 0.0, 1.45), (0.03, 0.0, 1.48), (0.02, 0.01, 1.50))  # one vertex
BEHIND = ((-0.03, 0.0, 1.47), (-0.01, 0.0, 1.48), (0.0, 0.01, 1.50))  # center x < 0


def test_crop_keeps_the_triangles_above_and_in_front(crop):
    (kept,) = crop.crop([KEPT, BELOW, BEHIND], ORIGIN)
    expected = np.subtract(KEPT, ORIGIN)
    np.testing.assert_allclose(kept, expected, atol=1e-12)


def test_crop_main_round_trip(crop, tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="r"><joint name="joint_head_pan" type="revolute">'
        '<origin xyz="0.05 0 1.429"/></joint></robot>'
    )
    body = tmp_path / "body.stl"
    crop.write_stl(body, [KEPT, BELOW, BEHIND])
    assert len(crop.read_stl(body)) == 3
    out = tmp_path / "out" / "head_base.stl"
    assert crop.main(urdf, body, out) == 0
    (triangle,) = crop.read_stl(out)
    np.testing.assert_allclose(triangle, np.subtract(KEPT, ORIGIN), atol=1e-6)
    # Nothing above the threshold: no file.
    crop.write_stl(body, [BELOW, BEHIND])
    empty = tmp_path / "empty.stl"
    assert crop.main(urdf, body, empty) == 1
    assert not empty.exists()


def test_crop_input_errors(crop, tmp_path):
    ascii_text = "solid t\nfacet normal 0 0 1\nouter loop\n"
    ascii_text += (
        "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n"
    )
    (tmp_path / "ascii.stl").write_text(ascii_text)
    with pytest.raises(ValueError):
        crop.read_stl(tmp_path / "ascii.stl")
    (tmp_path / "robot.urdf").write_text('<robot name="r"/>')
    with pytest.raises(ValueError):
        crop.pan_origin(tmp_path / "robot.urdf")
