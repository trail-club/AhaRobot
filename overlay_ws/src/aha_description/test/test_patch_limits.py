"""scripts/patch_limits.py on the upstream astra_description URDF."""

import importlib.util
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_limits.py"
MESH_DIR = "package://aha_description/meshes/"
MESHES = {
    "link_l6": MESH_DIR + "wrist_cam_mount_d435i_link6.stl",
    "link_r6": MESH_DIR + "wrist_cam_mount_d435i_link6.stl",
    "link_head_tilt": MESH_DIR + "head_cam_mount_d435i.stl",
}
D435I_MASS = 0.072


def load_script():
    spec = importlib.util.spec_from_file_location("patch_limits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch_limits = load_script()


@pytest.fixture(scope="module")
def upstream_urdf():
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        pytest.skip("ament_index_python not available")
    try:
        share = get_package_share_directory("astra_description")
    except PackageNotFoundError:
        pytest.skip("astra_description not installed")
    path = os.path.join(share, "urdf", "astra_description.urdf")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    return path


@pytest.fixture(scope="module")
def patched(upstream_urdf, tmp_path_factory):
    dst = tmp_path_factory.mktemp("patched") / "patched.urdf"
    assert patch_limits.main(upstream_urdf, str(dst)) == 0
    return ET.parse(dst).getroot()


def mass(root, link_name):
    return float(root.find(f"link[@name='{link_name}']/inertial/mass").get("value"))


def test_no_zero_effort(patched):
    for elem in patched.findall("joint"):
        if elem.get("type") not in ("revolute", "prismatic"):
            continue
        limit = elem.find("limit")
        assert float(limit.get("effort")) > 0, elem.get("name")
        assert float(limit.get("velocity")) > 0, elem.get("name")


def test_camera_mount_meshes(patched):
    for link_name, filename in MESHES.items():
        link = patched.find(f"link[@name='{link_name}']")
        for tag in ("visual", "collision"):
            meshes = link.findall(f"{tag}/geometry/mesh")
            assert meshes, f"{link_name} {tag}"
            for mesh in meshes:
                assert mesh.get("filename") == filename
                assert mesh.get("scale") == "0.001 0.001 0.001"


def test_wrist_inertial_adds_the_camera(upstream_urdf, patched):
    upstream = ET.parse(upstream_urdf).getroot()
    for link_name in ("link_l6", "link_r6"):
        assert mass(patched, link_name) >= mass(upstream, link_name) + D435I_MASS


def test_head_tilt_inertial_unchanged(patched):
    inertial = patched.find("link[@name='link_head_tilt']/inertial")
    assert inertial.find("origin").get("xyz") == "0.04297 0 -0.00285"
    assert inertial.find("mass").get("value") == "0.10050"
    assert inertial.find("inertia").get("ixz") == "-1.1244e-05"


def test_missing_link_fails(upstream_urdf, tmp_path):
    tree = ET.parse(upstream_urdf)
    link = tree.getroot().find("link[@name='link_r6']")
    link.set("name", "link_r6_renamed")
    src = tmp_path / "upstream_renamed.urdf"
    tree.write(src, xml_declaration=True, encoding="utf-8")
    dst = tmp_path / "patched.urdf"
    assert patch_limits.main(str(src), str(dst)) != 0
    assert not dst.exists()
