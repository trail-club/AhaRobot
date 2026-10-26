"""Topic, joint and controller names the real robot's nodes share:
aha_bringup arms.launch.py (arm_node, evaluated with aha_test_frame's
launch_walk), aha_manipulation config/arm.yaml (arm trajectory bridges),
place_scene.ARMS and place_server.py's frozen-reading watch, and
gripper_handover.py (needs the installed aha_bringup and aha_manipulation:
colcon test)."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory

from aha_test_manipulation.frame_moveit import frame_controllers
from aha_test_manipulation.gripper_ramp import finger_joint
from aha_test_manipulation.place_scene import ARMS

PACKAGE = Path(__file__).resolve().parents[1]
# The helper lives with aha_test_frame's launch tests.
WALK = PACKAGE.parent / "aha_test_frame" / "test" / "launch_walk.py"
ARMS_LAUNCH = (
    Path(get_package_share_directory("aha_bringup")) / "launch" / "arms.launch.py"
)
BRIDGE_CONFIG = (
    Path(get_package_share_directory("aha_manipulation")) / "config" / "arm.yaml"
)
SIDES = ["right", "left"]
# gripper_handover.py's node only creates its topics, on a domain of its own
# (never the robot's) and on this host only.
TEST_DOMAIN = 75


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arms_launch():
    return load("arms_launch", ARMS_LAUNCH)


@pytest.fixture(scope="module")
def processes(tmp_path_factory):
    walk = load("launch_walk", WALK)
    return walk.launched(ARMS_LAUNCH, tmp_path_factory.mktemp("launch"), record="false")


@pytest.fixture(scope="module")
def bridge_config():
    return yaml.safe_load(BRIDGE_CONFIG.read_text())


def bridge_params(config, side):
    return {
        **config["/**"]["ros__parameters"],
        **config[f"aha_{side}_arm_bridge"]["ros__parameters"],
    }


def arm_node(processes, side):
    (node,) = [
        p
        for p in processes
        if p.label == "astra_controller/arm_node" and p.namespace == f"/{side}/arm"
    ]
    return node


def frozen_watch_topic(side):
    """place_server.py's feedback topic of the arm ``side`` (the one f-string
    naming an action's feedback, evaluated for ``arm``)."""
    tree = ast.parse((PACKAGE / "scripts" / "place_server.py").read_text())
    (topic,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        and any(
            isinstance(part, ast.Constant) and "_action/feedback" in part.value
            for part in node.values
        )
    ]
    return eval(
        compile(ast.Expression(topic), "place_server.py", "eval"), {"arm": side}
    )


@pytest.mark.parametrize("side", SIDES)
def test_arm_node_bridge_and_place_scene_agree(
    arms_launch, processes, bridge_config, side
):
    joints, grippers = arms_launch.ARMS[side]
    node = arm_node(processes, side)
    assert node.parameters["joint_names"] == joints
    assert node.parameters["gripper_joint_names"] == grippers
    state = arms_launch.state_topic(side)
    assert node.remappings["joint_states"] == state
    assert node.remappings["gripper_joint_states"] == state

    params = bridge_params(bridge_config, side)
    assert params["joints"] == joints
    assert params["command_topic"] == f"{node.namespace}/joint_command"
    assert params["state_topic"] == state

    cfg = ARMS[side]
    assert cfg["joints"][1:] == joints
    assert cfg["gripper_joint_topic"] == f"{node.namespace}/gripper_joint_command"
    # arm_node takes a gripper command for its second gripper joint only.
    assert cfg["gripper_joint"] == grippers[1] == finger_joint(side)
    assert cfg["gripper_state_topic"] == state


@pytest.mark.parametrize("side", SIDES)
def test_bridge_nodes_are_named_as_their_config(processes, bridge_config, side):
    (bridge,) = [
        p
        for p in processes
        if p.executable == "arm_trajectory_bridge.py"
        and p.name == f"aha_{side}_arm_bridge"
    ]
    assert bridge.name in bridge_config
    assert [Path(f).resolve() for f in bridge.parameter_files] == [
        BRIDGE_CONFIG.resolve()
    ]


@pytest.mark.parametrize("side", SIDES)
def test_frozen_watch_is_on_the_bridge_feedback(bridge_config, side):
    """The topic place_server.py watches is the feedback of the action MoveIt
    sends the arm's trajectories to."""
    manager = frame_controllers(bridge_config, [side])[
        "moveit_simple_controller_manager"
    ]
    (controller,) = manager["controller_names"]
    assert controller == bridge_params(bridge_config, side)["controller_name"]
    action = f"/{controller}/{manager[controller]['action_ns']}"
    assert frozen_watch_topic(side) == f"{action}/_action/feedback"


@pytest.fixture
def ros(monkeypatch):
    import rclpy

    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    rclpy.init(domain_id=TEST_DOMAIN)
    yield
    rclpy.try_shutdown()


@pytest.mark.parametrize("side", SIDES)
def test_gripper_handover_topics(ros, arms_launch, bridge_config, side):
    handover = load("gripper_handover", PACKAGE / "scripts" / "gripper_handover.py")
    joints, _ = arms_launch.ARMS[side]
    params = bridge_params(bridge_config, side)
    node = handover.GripperNode(side)
    try:
        # Commanded in the arm_node's joint_names order.
        assert node.arm_joints == joints
        assert node.joint == ARMS[side]["gripper_joint"]
        published = {p.topic_name for p in node.publishers}
        assert {ARMS[side]["gripper_joint_topic"], params["command_topic"]} <= published
        subscribed = {s.topic_name for s in node.subscriptions}
        assert params["state_topic"] in subscribed
    finally:
        node.destroy_node()
