"""MoveIt boundary failures and true-state grasp phase regressions."""

from dataclasses import replace
from pathlib import Path
import math
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_gazebo.model import LIMIT_MARGIN  # noqa: E402
from aha_sim_tasks.api import GroundTruth, JOINT_LIMITS, Observation  # noqa: E402
from aha_sim_tasks import ground_truth_ik  # noqa: E402
from aha_sim_tasks.moveit_client import (  # noqa: E402
    ARM_JOINTS,
    MoveItClient,
    relative_pose,
    world_pose,
    robot_state,
)
from aha_sim_tasks.policy_protocol import decode_observation, observation_message  # noqa: E402
from action_msgs.msg import GoalStatus  # noqa: E402
from moveit_msgs.msg import MoveItErrorCodes  # noqa: E402


class FakeMoveIt:
    def __init__(self):
        self.goals = []
        self.scenes = []

    def forward(self, joints):
        return (0.4745, -0.426, 0.119 + joints["joint_r1"], 0, 0, 0, 1)

    def inverse(self, target, joints):
        self.target = target
        return {name: 0.1 for name in ARM_JOINTS}

    def update_scene(self, root, scene, apple):
        self.scenes.append((root, apple))

    def start(self, target, joints):
        self.goals.append(target)

    def attach_apple(self, root, joints, apple, radius):
        self.attached = apple

    def completed(self):
        return False


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(ground_truth_ik, "MoveItClient", FakeMoveIt)
    policy = ground_truth_ik.GroundTruthIKPolicy()
    policy.reset("pick_apple", "Pick.")
    return policy


def observation(time=0, contacts=0):
    return Observation(
        time,
        "pick_apple",
        "Pick.",
        {name: 0.0 for name in JOINT_LIMITS},
        {},
        (100, 100, 3),  # Bad odometry must have no effect.
        ground_truth=GroundTruth(
            (0.4, 0, 0, 0, 0, 0, 1), (0.962, -0.426, 0.16), contacts
        ),
    )


def test_full_root_transform_round_trip_includes_roll_and_pitch():
    qx = (math.sin(0.1), 0, 0, math.cos(0.1))
    root = (0.4, -0.3, -0.009, *qx)
    local = (0.5, -0.4, 0.3, 0, math.sin(0.2), 0, math.cos(0.2))
    assert relative_pose(root, world_pose(root, local)) == pytest.approx(local)


def test_truth_ipc_is_opt_in():
    value = observation()
    assert decode_observation(observation_message(value)["observation"]) == value
    assert (
        "ground_truth"
        not in observation_message(replace(value, ground_truth=None))["observation"]
    )


def test_pick_waits_for_two_contacts_and_routes_arm_to_moveit(policy):
    with pytest.raises(ValueError, match="requires Gazebo"):
        policy.act(replace(observation(), ground_truth=None))
    action = policy.act(observation())
    assert set(action.joint_positions) == {"joint_r7r", "joint_r7l"}
    policy.phase = "close"
    policy.target = (0.962, -0.426, 0.16, 0, 0, 0, 1)
    policy.act(observation(2, contacts=1))
    assert len(policy.client.goals) == 1
    action = policy.act(observation(2.1, contacts=2))
    assert policy.phase == "lift"
    assert len(policy.client.goals) == 2
    assert policy.client.attached == observation().ground_truth.object_position
    assert policy.client.target[:3] == pytest.approx((0.562, -0.426, 0.29))
    assert action.joint_positions["joint_r7r"] == policy.settings["grasp_position"]
    assert not set(ARM_JOINTS) & action.joint_positions.keys()


def test_grasp_has_bounded_deadline(policy):
    policy.act(observation())
    policy.phase = "close"
    with pytest.raises(RuntimeError, match="close timed out"):
        policy.act(observation(31))


def test_approach_returns_stop_before_ik_and_uses_fresh_pose_after_settling(policy):
    policy.act(observation())
    policy.phase = "approach"
    policy.phase_time = 0
    near = replace(
        observation(2),
        ground_truth=replace(
            observation().ground_truth,
            robot_world_pose=(0.403, 0, 0, 0, 0, 0, 1),
        ),
    )
    moving = policy.act(observation(1.1))
    assert moving.linear_velocity > 0
    stopped = policy.act(near)
    assert policy.phase == "settle"
    assert stopped.linear_velocity == stopped.angular_velocity == 0
    assert len(policy.client.goals) == 1
    policy.act(replace(near, sim_time=2.4))
    assert len(policy.client.goals) == 1
    settled = replace(
        near,
        sim_time=2.6,
        ground_truth=replace(
            near.ground_truth, robot_world_pose=(0.404, 0, 0, 0, 0, 0, 1)
        ),
    )
    action = policy.act(settled)
    assert policy.phase == "pregrasp"
    assert len(policy.client.goals) == 2
    assert policy.client.target[0] == pytest.approx(0.962 - 0.404)
    assert action.linear_velocity == action.angular_velocity == 0


def test_reaching_a_pose_does_not_advance_until_moveit_succeeds(policy):
    policy.act(observation())
    policy.phase = "grasp"
    policy.target = world_pose(
        observation().ground_truth.robot_world_pose,
        policy.client.forward(observation().joint_positions),
    )
    policy.motion_done = False
    policy.act(observation(3))
    assert policy.phase == "grasp"
    policy.client.completed = lambda: True
    policy.act(observation(3.1))
    assert policy.phase == "close"


def test_close_releases_ros_resources_when_goal_cancellation_fails():
    client = MoveItClient.__new__(MoveItClient)
    cleaned = []
    client.pending_goal = None

    def fail():
        raise RuntimeError("cancel failed")

    client.goal = SimpleNamespace(cancel_goal_async=fail)
    client.move = SimpleNamespace(destroy=lambda: cleaned.append("action"))
    client.executor = SimpleNamespace(shutdown=lambda: cleaned.append("executor"))
    client.node = SimpleNamespace(destroy_node=lambda: cleaned.append("node"))
    client.context = SimpleNamespace(shutdown=lambda: cleaned.append("context"))
    with pytest.raises(RuntimeError, match="cancel failed"):
        client.close()
    assert cleaned == ["action", "executor", "node", "context"]


@pytest.mark.parametrize(
    "status,code",
    [
        (GoalStatus.STATUS_ABORTED, MoveItErrorCodes.CONTROL_FAILED),
        (GoalStatus.STATUS_SUCCEEDED, MoveItErrorCodes.PLANNING_FAILED),
    ],
)
def test_execution_errors_never_advance_grasp(monkeypatch, status, code):
    client = MoveItClient.__new__(MoveItClient)
    client.executor = SimpleNamespace(spin_once=lambda **_: None)
    client.pending_goal = None
    client.pending_result = SimpleNamespace(
        done=lambda: True,
        result=lambda: SimpleNamespace(
            status=status, result=SimpleNamespace(error_code=SimpleNamespace(val=code))
        ),
    )
    with pytest.raises(RuntimeError, match="execution failed"):
        client.completed()


def test_ik_failure_does_not_send_a_controller_goal():
    client = MoveItClient.__new__(MoveItClient)
    client.ik = object()
    client.call = lambda *_: SimpleNamespace(
        error_code=SimpleNamespace(val=MoveItErrorCodes.NO_IK_SOLUTION)
    )
    with pytest.raises(RuntimeError, match="KDL IK failed"):
        client.inverse((100, 0, 0, 0, 0, 0, 1), observation().joint_positions)


def test_full_start_state_clamps_up_to_the_physics_limits():
    joints = observation().joint_positions
    state = robot_state(
        {**joints, "joint_r1": -LIMIT_MARGIN - 1e-11, "joint_l1": -LIMIT_MARGIN - 1e-3}
    )
    values = dict(zip(state.joint_state.name, state.joint_state.position))
    assert state.is_diff is False
    assert values["joint_r1"] == 0
    assert values["joint_l1"] == -LIMIT_MARGIN - 1e-3
    assert values.keys() == JOINT_LIMITS.keys()


def test_planning_scene_retains_self_collision_matrix_and_limits_apple_contacts():
    from moveit_msgs.msg import (
        AllowedCollisionEntry,
        AllowedCollisionMatrix,
        PlanningScene,
    )
    from aha_sim_tasks.scene import load_scene

    names = ["base_link", "link_r1", "link_r7r", "link_r7l"]
    matrix = AllowedCollisionMatrix(
        entry_names=names,
        entry_values=[
            AllowedCollisionEntry(
                enabled=[i == j or {i, j} in ({0, 1}, {2, 3}) for j in range(4)]
            )
            for i in range(4)
        ],
    )
    client = MoveItClient.__new__(MoveItClient)
    client.attached = False
    client.scene, client.get_scene = object(), object()
    requests = []

    def call(service, request):
        if service is client.get_scene:
            return SimpleNamespace(scene=PlanningScene(allowed_collision_matrix=matrix))
        requests.append(request.scene)
        return SimpleNamespace(success=True)

    client.call = call
    client.update_scene(
        (0.4, 0, -0.009, 0, 0, 0, 1), load_scene(), (0.962, -0.426, 0.16)
    )
    result = requests[0].allowed_collision_matrix
    apple = result.entry_names.index("apple")
    assert result.entry_values[0].enabled[1]  # SRDF adjacent pair survives.
    assert result.entry_values[2].enabled[3]  # Existing finger exclusion survives.
    assert not result.entry_values[apple].enabled[0]
    assert not result.entry_values[apple].enabled[1]
    assert result.entry_values[apple].enabled[2:4] == [True, True]
    objects = {obj.id: obj for obj in requests[0].world.collision_objects}
    assert objects.keys() == {"source_table", "destination_table", "apple"}
    assert objects["apple"].primitive_poses[0].position.x == pytest.approx(0.562)
    assert objects["apple"].primitive_poses[0].position.z == pytest.approx(0.169)


def test_attached_geometry_is_local_and_preserves_scene_state():
    client = MoveItClient.__new__(MoveItClient)
    client.scene = object()
    client.forward = lambda _: (0.5, -0.4, 0.15, 0, 0, 0, 1)
    requests = []
    client.call = lambda _, request: requests.append(request) or SimpleNamespace(
        success=True
    )
    client.attach_apple((0.4, 0, -0.009, 0, 0, 0, 1), {}, (0.9, -0.4, 0.16), 0.04)
    scene = requests[0].scene
    assert scene.is_diff and scene.robot_state.is_diff
    assert (
        scene.world.collision_objects == []
    )  # MoveIt removes the world apple automatically.
    attached = scene.robot_state.attached_collision_objects[0]
    assert attached.link_name == attached.object.header.frame_id == "link_ree"
    assert attached.touch_links == ["link_r7r", "link_r7l"]
    assert attached.object.primitive_poses[0].position.z == pytest.approx(0.019)
    assert client.attached
