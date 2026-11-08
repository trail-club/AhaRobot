"""What `ros2 launch aha_test_manipulation manip_real.launch.py` starts, with
moveit.launch.py's include evaluated in launch order (aha_test_frame's
launch_walk; needs the installed aha_test_manipulation, aha_test_frame and
aha_manipulation: colcon test), and moveit.launch.py's config of the real
robot on its frame. Nothing is started."""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from aha_test_frame.frame import frame_description
from ament_index_python.packages import get_package_share_directory

from aha_test_manipulation.frame_moveit import GOAL_MARGIN, side_grasp_tcp_height

PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "manip_real.launch.py"
# The helper lives with aha_test_frame's launch tests.
WALK = PACKAGE.parent / "aha_test_frame" / "test" / "launch_walk.py"
FRAME_CONFIG = (
    Path(get_package_share_directory("aha_test_frame"))
    / "config"
    / "frame_table_202610.yaml"
)
BRIDGE_CONFIG = (
    Path(get_package_share_directory("aha_manipulation")) / "config" / "arm.yaml"
)
MOVE_GROUP = "moveit_ros_move_group/move_group"
PLACE_SERVER = "aha_test_manipulation/place_server.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def walk():
    return load("launch_walk", WALK)


@pytest.fixture(scope="module")
def moveit_launch():
    return load("moveit_launch", PACKAGE / "launch" / "moveit.launch.py")


@pytest.fixture(scope="module")
def urdf_xml():
    return frame_description(FRAME_CONFIG)


@pytest.fixture(scope="module")
def goal_time():
    return yaml.safe_load(BRIDGE_CONFIG.read_text())["/**"]["ros__parameters"][
        "goal_time"
    ]


def by_label(processes):
    out = {}
    for p in processes:
        assert p.label not in out, p.label
        out[p.label] = p
    return out


@pytest.mark.parametrize("arm", ["right", "left"])
def test_move_group_and_place_server_for_the_arm(walk, urdf_xml, tmp_path, arm):
    processes = walk.launched(LAUNCH, tmp_path, rviz="false", arm=arm)
    assert [p.label for p in processes] == [MOVE_GROUP, PLACE_SERVER]
    found = by_label(processes)

    move_group = found[MOVE_GROUP].parameters
    assert move_group["moveit_simple_controller_manager.controller_names"] == [
        f"{arm}_arm_controller"
    ]
    robot = ET.fromstring(move_group["robot_description"])
    assert robot.find("link[@name='frame_column']") is not None

    place = found[PLACE_SERVER].parameters
    assert place["default_arm"] == arm
    assert place["gripper_interface"] == "arm_node"
    assert place["release_topic"] == ""
    assert place["tcp_height"] == pytest.approx(side_grasp_tcp_height(urdf_xml, arm))
    assert place["velocity_scaling"] == pytest.approx(0.3)
    assert place["verify_detections"] == 3
    assert place["placed_padding"] == pytest.approx(0.03)
    assert place["frozen_window"] == pytest.approx(0.5)
    assert place["aim_head"] is True
    assert place["use_sim_time"] is False


def test_launch_switches(walk, tmp_path):
    labels = [p.label for p in walk.launched(LAUNCH, tmp_path)]
    assert labels == [MOVE_GROUP, "rviz2/rviz2", PLACE_SERVER]
    labels = [
        p.label
        for p in walk.launched(LAUNCH, tmp_path, rviz="false", place_server="false")
    ]
    assert labels == [MOVE_GROUP]
    processes = walk.launched(LAUNCH, tmp_path, rviz="false", aim_head="false")
    assert by_label(processes)[PLACE_SERVER].parameters["aim_head"] is False


def test_frame_config_for_one_arm(moveit_launch, goal_time):
    params = moveit_launch.frame_moveit_config(FRAME_CONFIG, ["right"]).to_dict()
    robot = ET.fromstring(params["robot_description"])
    assert robot.find("link[@name='frame_column']") is not None
    assert robot.find("joint[@name='joint_r1']").get("type") == "fixed"
    manager = params["moveit_simple_controller_manager"]
    assert manager["controller_names"] == ["right_arm_controller"]
    assert manager["right_arm_controller"]["action_ns"] == "follow_joint_trajectory"
    assert params["moveit_manage_controllers"] is True
    assert params["trajectory_execution"]["allowed_goal_duration_margin"] == (
        pytest.approx(goal_time + GOAL_MARGIN)
    )
    assert GOAL_MARGIN == pytest.approx(0.5)


def test_frame_config_for_both_arms(moveit_launch):
    params = moveit_launch.frame_moveit_config(FRAME_CONFIG, ["right", "left"])
    manager = params.to_dict()["moveit_simple_controller_manager"]
    assert manager["controller_names"] == [
        "right_arm_controller",
        "left_arm_controller",
    ]
    assert manager["left_arm_controller"]["joints"] == [
        f"joint_l{i}" for i in range(2, 7)
    ]
