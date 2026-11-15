"""urdf/sensors.xacro through aha_robot.urdf.xacro: camera frames and Gazebo sensors."""

import math
import subprocess
import xml.etree.ElementTree as ET

import pytest

WRIST_FRAMES = [
    f"{side}_wrist_camera_{suffix}"
    for side in ("left", "right")
    for suffix in ("link", "optical_frame")
]
# D435i color HFOV at 640x480 (sensors.xacro camera_hfov).
CAMERA_HFOV = 0.9759


def robot(*args):
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        pytest.skip("ament_index_python not available")
    try:
        share = get_package_share_directory("aha_description")
    except PackageNotFoundError:
        pytest.skip("aha_description not installed")
    xacro = f"{share}/urdf/aha_robot.urdf.xacro"
    return ET.fromstring(subprocess.check_output(["xacro", xacro, *args], text=True))


def sensors(root):
    return {s.get("name"): s for s in root.iter("sensor")}


def test_real_robot_has_wrist_frames_without_sensors():
    root = robot("sim:=false")
    links = {link.get("name") for link in root.findall("link")}
    assert set(WRIST_FRAMES) <= links
    assert sensors(root) == {}


def test_sim_cameras_switch():
    names = {"head_camera", "left_wrist_camera", "right_wrist_camera"}
    assert set(sensors(robot("sim:=true"))) >= names
    assert not names & set(sensors(robot("sim:=true", "cameras:=false")))


@pytest.mark.parametrize("side,parent", [("left", "link_l6"), ("right", "link_r6")])
def test_wrist_camera_on_the_wrist_part(side, parent):
    root = robot("sim:=true")
    joint = root.find(f"joint[@name='{side}_wrist_camera_joint']")
    assert joint.find("parent").get("link") == parent
    xyz = [float(v) for v in joint.find("origin").get("xyz").split()]
    rpy = [float(v) for v in joint.find("origin").get("rpy").split()]
    # Body center of the D435i on the wrist part, tilted 30 deg toward the fingers.
    assert xyz == pytest.approx([0.005637, 0.0, 0.087903], abs=1e-6)
    assert rpy == pytest.approx([0.0, -math.pi / 3, 0.0], abs=1e-9)
    optical = root.find(f"joint[@name='{side}_wrist_camera_optical_joint']")
    head_optical = root.find("joint[@name='camera_optical_joint']")
    assert optical.find("origin").attrib == head_optical.find("origin").attrib


def test_wrist_sensor_matches_the_head_camera_optics():
    found = sensors(robot("sim:=true"))
    head = found["head_camera"]
    for side in ("left", "right"):
        wrist = found[f"{side}_wrist_camera"]
        assert float(wrist.findtext("camera/horizontal_fov")) == pytest.approx(
            CAMERA_HFOV
        )
        assert wrist.findtext("pose") == head.findtext("pose")
        assert wrist.findtext("gz_frame_id") == f"{side}_wrist_camera_optical_frame"
