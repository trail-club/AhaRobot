"""What `ros2 launch aha_test_frame real.launch.py` starts, with the includes,
groups and OpaqueFunctions evaluated in launch order (launch_walk; needs the
installed aha_test_frame, aha_bringup, aha_manipulation, aha_servo and
aha_perception: colcon test). Nothing is started."""

from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_prefix

from launch_walk import launched

LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "real.launch.py"
ARM_NODE = "astra_controller/arm_node"
BRIDGE = "arm_trajectory_bridge.py"


def arm_namespaces(processes):
    return sorted(p.namespace for p in processes if p.label == ARM_NODE)


def bridges(processes):
    """{node name: command line} of the arm trajectory bridges run by python3."""
    out = {}
    for p in processes:
        if any(Path(part).name == BRIDGE for part in p.cmd):
            (rule,) = [part for part in p.cmd if part.startswith("__node:=")]
            out[rule.split(":=", 1)[1]] = p.cmd
    return out


def node_bridges(processes):
    """The bridges started as Nodes (arms.launch.py's own)."""
    return [p for p in processes if p.executable == BRIDGE]


def test_defaults_start_both_arms_and_both_bridges(tmp_path):
    processes = launched(LAUNCH, tmp_path)
    assert arm_namespaces(processes) == ["/left/arm", "/right/arm"]
    found = bridges(processes)
    assert sorted(found) == ["aha_left_arm_bridge", "aha_right_arm_bridge"]
    script = Path(get_package_prefix("aha_manipulation")) / "lib" / "aha_manipulation"
    for name, cmd in found.items():
        assert cmd[0] == "python3"
        assert Path(cmd[1]) == script / BRIDGE and Path(cmd[1]).is_file()
        params = Path(cmd[cmd.index("--params-file") + 1])
        assert params.name == "arm.yaml"
        # The node name selects its section of the parameter file.
        assert name in yaml.safe_load(params.read_text())
    # Started once: not by arms.launch.py as well.
    assert node_bridges(processes) == []


def test_arm_arguments_reach_the_arm_nodes(tmp_path):
    """real.launch.py's devices and record go through the scoped include
    (arms.launch.py's own defaults differ)."""
    processes = launched(LAUNCH, tmp_path, right_device="/dev/ttyUSB7")
    devices = {
        p.namespace: p.parameters["device"] for p in processes if p.label == ARM_NODE
    }
    assert devices == {"/left/arm": "/dev/ttyUSB0", "/right/arm": "/dev/ttyUSB7"}
    assert not [p for p in processes if p.cmd[:3] == ["ros2", "bag", "record"]]


def test_arm_states_are_joint_state_sources(tmp_path):
    processes = launched(LAUNCH, tmp_path)
    (jsp,) = [p for p in processes if p.executable == "joint_state_publisher"]
    for p in processes:
        if p.label == ARM_NODE:
            assert p.remappings["joint_states"] in jsp.parameters["source_list"]
            assert p.remappings["gripper_joint_states"] == p.remappings["joint_states"]


@pytest.mark.parametrize(
    "arguments, expected_bridges, expected_arms",
    [
        ({"use_left": "false"}, ["aha_right_arm_bridge"], ["/right/arm"]),
        ({"use_right": "false"}, ["aha_left_arm_bridge"], ["/left/arm"]),
        ({"trajectory_bridge": "false"}, [], ["/left/arm", "/right/arm"]),
        ({"arms": "false"}, [], []),
    ],
)
def test_arm_and_bridge_arguments(tmp_path, arguments, expected_bridges, expected_arms):
    processes = launched(LAUNCH, tmp_path, **arguments)
    assert sorted(bridges(processes)) == expected_bridges
    assert arm_namespaces(processes) == expected_arms
    assert node_bridges(processes) == []


def test_without_hardware(tmp_path):
    processes = launched(
        LAUNCH,
        tmp_path,
        arms="false",
        head="manual",
        camera="false",
        planes="false",
        rviz="false",
    )
    assert sorted(p.label for p in processes) == [
        "aha_perception/manual_head.py",
        "joint_state_publisher/joint_state_publisher",
        "robot_state_publisher/robot_state_publisher",
        "tf2_ros/static_transform_publisher",
    ]
