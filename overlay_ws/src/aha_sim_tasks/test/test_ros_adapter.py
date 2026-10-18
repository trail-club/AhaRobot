"""Controller transport regressions without launching ROS processes."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_gazebo.model import LIMIT_MARGIN  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from aha_sim_tasks.api import Action, CONTROLLER_JOINTS, GRIPPER_JOINTS, JOINT_LIMITS  # noqa: E402
from aha_sim_tasks.ros_environment import RosEnvironment  # noqa: E402


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def environment():
    return SimpleNamespace(
        positions={name: 0.0 for name in JOINT_LIMITS},
        last_targets={},
        last_trajectory_times={},
        base=Publisher(),
        trajectories={name: Publisher() for name in CONTROLLER_JOINTS},
        grippers={name: Publisher() for name in GRIPPER_JOINTS},
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10))
        ),
    )


def test_partial_lift_command_clamps_numerical_drift_in_held_joint():
    node = environment()
    node.positions["joint_l1"] = -1e-14
    RosEnvironment.apply(node, Action(joint_positions={"joint_r1": 0.18}))
    trajectory = node.trajectories["lift_controller"].messages[-1]
    assert trajectory.joint_names == ["joint_l1", "joint_r1"]
    assert list(trajectory.points[0].positions) == [0.0, 0.18]
    assert trajectory.points[0].time_from_start.sec == 2


def test_omitted_joint_preserves_previous_command():
    node = environment()
    RosEnvironment.apply(
        node, Action(joint_positions={"joint_l1": 0.1, "joint_r1": 0.2})
    )
    node.positions["joint_l1"] = 0.05
    RosEnvironment.apply(node, Action(joint_positions={"joint_r1": 0.3}))
    assert list(
        node.trajectories["lift_controller"].messages[-1].points[0].positions
    ) == [
        0.1,
        0.3,
    ]


def test_invalid_action_publishes_nothing():
    node = environment()
    with pytest.raises(ValueError):
        RosEnvironment.apply(node, Action(joint_positions={"joint_r1": float("nan")}))
    assert node.base.messages == []
    assert all(not publisher.messages for publisher in node.trajectories.values())


def test_unreached_target_is_reissued_after_trajectory_duration():
    node = environment()
    action = Action(joint_positions={"joint_r1": 0.18})
    RosEnvironment.apply(node, action)
    RosEnvironment.apply(node, action)
    assert len(node.trajectories["lift_controller"].messages) == 1
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=14))
    )
    RosEnvironment.apply(node, action)
    assert len(node.trajectories["lift_controller"].messages) == 2
    node.positions["joint_r1"] = 0.18
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=18))
    )
    RosEnvironment.apply(node, action)
    assert len(node.trajectories["lift_controller"].messages) == 2


def scoring_environment():
    received = []
    return SimpleNamespace(
        world=None,
        previous_object=None,
        previous_robot=None,
        mark=received.append,
        received=received,
    )


def sample(t, contacts, x):
    from std_msgs.msg import String

    return String(
        data=json.dumps(
            {
                "schema_version": 1,
                "frame_id": "world",
                "stamp": {"sec": int(t), "nanosec": round((t - int(t)) * 1e9)},
                "contact_window_start": {
                    "sec": int(t - 0.05),
                    "nanosec": round(((t - 0.05) % 1) * 1e9),
                },
                "robot_pose": [x, 0, 0],
                "object_position": [x, 0, 0.3],
                "finger_contacts": contacts,
            }
        )
    )


def test_scoring_state_keeps_contacts_and_poses_from_same_sample():
    node = scoring_environment()
    RosEnvironment.on_state(node, sample(1, 2, 0.5))
    RosEnvironment.on_state(node, sample(1.05, 0, 0.55))
    assert node.world.sim_time == pytest.approx(1.05)
    assert node.world.finger_contacts == 0
    assert node.world.object_position == (0.55, 0, 0.3)
    assert node.world.object_speed == pytest.approx(1)
    assert node.world.robot_speed == pytest.approx(1)
    RosEnvironment.on_state(node, sample(1.02, 2, 100))
    RosEnvironment.on_state(node, sample(1.05, 1, 100))
    assert node.world.finger_contacts == 0
    assert len(node.received) == 2
    RosEnvironment.on_state(node, sample(1.1, 1, 0.6))
    assert node.world.finger_contacts == 1
    assert node.world.object_speed == pytest.approx(1)


@pytest.mark.parametrize(
    "change",
    [
        {"finger_contacts": 3},
        {"finger_contacts": True},
        {"object_position": [0, 0, float("nan")]},
        {"robot_pose": [0, 0]},
        {"schema_version": 2},
        {"frame_id": "odom"},
    ],
)
def test_bad_scoring_sample_does_not_change_last_state(change):
    from std_msgs.msg import String

    node = scoring_environment()
    RosEnvironment.on_state(node, sample(1, 2, 0.5))
    previous = node.world
    data = json.loads(sample(1.05, 0, 0.55).data)
    data.update(change)
    with pytest.raises(ValueError):
        RosEnvironment.on_state(node, String(data=json.dumps(data)))
    assert node.world is previous
    assert node.previous_object == (1, (0.5, 0, 0.3))


def camera_environment():
    from geometry_msgs.msg import TransformStamped

    def transform(parent, child, stamp):
        result = TransformStamped(child_frame_id=child)
        result.header.frame_id = parent
        result.header.stamp = stamp.to_msg()
        return result

    node = SimpleNamespace(
        cameras_enabled=True,
        camera_samples={},
        received={},
        mark=lambda source: None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)
        ),
        tf_buffer=SimpleNamespace(lookup_transform=transform),
    )
    return node


def truth_sample(t=1, **changes):
    from std_msgs.msg import String

    data = json.loads(sample(t, 2, 0.5).data)
    data["ground_truth"] = {
        "robot_world_pose": [0.5, 0, -0.009, 0, 0, 0, 1],
        "joint_positions": {name: 0.1 for name in JOINT_LIMITS},
        **changes,
    }
    return String(data=json.dumps(data))


def test_truth_mode_uses_atomic_physics_positions_and_state_time():
    node = scoring_environment()
    node.truth_enabled = True
    node.truth_joints = Publisher()
    node.task = SimpleNamespace(task_id="pick_apple", instruction="Pick.")
    node.camera_observations = lambda: {}
    node.image = None
    RosEnvironment.on_state(node, truth_sample())
    observation = RosEnvironment.observation(node)
    assert node.truth_joints.messages[-1].header.stamp.sec == 1
    assert node.truth_joints.messages[-1].name == list(node.positions)
    assert observation.sim_time == 1
    assert observation.joint_positions["joint_r2"] == 0.1
    assert observation.joint_velocities == {}
    assert observation.ground_truth.robot_world_pose[2] == -0.009
    assert observation.ground_truth.object_position == node.world.object_position
    assert observation.ground_truth.finger_contacts == 2
    RosEnvironment.on_state(node, truth_sample(0.9, joint_positions={}))
    assert node.positions["joint_r2"] == 0.1
    assert len(node.received) == 1


def test_normal_policy_ignores_truth_even_if_present_in_state():
    node = scoring_environment()
    node.truth_enabled = False
    node.ground_truth = None
    node.positions = {"joint_r2": 0.7}
    RosEnvironment.on_state(node, truth_sample())
    assert node.positions == {"joint_r2": 0.7}
    assert node.ground_truth is None


@pytest.mark.parametrize(
    "pose", [[0, 0], [0, 0, 0, 0, 0, 0, 0], [float("nan"), 0, 0, 0, 0, 0, 1]]
)
def test_invalid_truth_does_not_advance_scoring_or_freshness(pose):
    node = scoring_environment()
    node.truth_enabled = True
    node.truth_joints = Publisher()
    with pytest.raises(ValueError, match="Invalid Gazebo"):
        RosEnvironment.on_state(node, truth_sample(robot_world_pose=pose))
    assert node.world is None and node.received == []


def test_unfilled_physics_joints_do_not_mark_truth_ready():
    node = scoring_environment()
    node.truth_enabled = True
    node.truth_joints = Publisher()
    RosEnvironment.on_state(node, truth_sample(joint_positions={"joint_r1": 0.1}))
    assert node.world is None and node.received == []


def test_truth_adapter_subscribes_to_no_joint_odometry_or_camera_sensors():
    import rclpy

    rclpy.init()
    node = RosEnvironment(SimpleNamespace(), ground_truth=True)
    try:
        topics = {subscription.topic_name for subscription in node.subscriptions}
        assert node.required == {"state"}
        assert not topics & {"/joint_states", "/diff_drive_controller/odom"}
        assert not any(topic.startswith("/camera/") for topic in topics)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def camera_sample(name, nanosec=0):
    from sensor_msgs.msg import CameraInfo, Image

    image = Image(height=1, width=1, encoding="rgb8", step=3, data=b"abc")
    image.header.frame_id = name + "_optical"
    image.header.stamp.sec = 10
    image.header.stamp.nanosec = nanosec
    return image, CameraInfo(header=image.header, height=1, width=1)


def test_camera_bundle_requires_all_views_and_timestamped_transforms():
    node = camera_environment()
    for name in ("head", "left_wrist"):
        RosEnvironment.on_camera(node, name, *camera_sample(name))
    assert RosEnvironment.camera_observations(node) is None
    RosEnvironment.on_camera(node, "right_wrist", *camera_sample("right_wrist"))
    bundle = RosEnvironment.camera_observations(node)
    assert set(bundle) == {"head", "left_wrist", "right_wrist"}
    for camera in bundle.values():
        assert camera.base_transform.header.frame_id == "base_link"
        assert camera.base_transform.header.stamp == camera.image.header.stamp
        assert camera.base_transform.child_frame_id == camera.image.header.frame_id


def test_camera_callback_rejects_mismatched_frames_and_older_samples():
    from copy import deepcopy

    node = camera_environment()
    image, info = camera_sample("head")
    info.header = deepcopy(info.header)
    info.header.frame_id = "wrong_optical"
    RosEnvironment.on_camera(node, "head", image, info)
    assert node.camera_samples == {}
    RosEnvironment.on_camera(node, "head", *camera_sample("head", 50_000_000))
    RosEnvironment.on_camera(node, "head", *camera_sample("head"))
    assert node.camera_samples["head"][-1][0].header.stamp.nanosec == 50_000_000
    assert len(node.camera_samples["head"]) == 1


def test_camera_bundle_rejects_skew_stale_images_and_missing_tf():
    from tf2_ros import TransformException

    node = camera_environment()
    for name in ("head", "left_wrist", "right_wrist"):
        RosEnvironment.on_camera(node, name, *camera_sample(name))
    node.camera_samples["right_wrist"][-1][0].header.stamp.sec = 9
    assert RosEnvironment.camera_observations(node) is None
    node.camera_samples["right_wrist"][-1][0].header.stamp.sec = 10
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=11_000_000_000)
    )
    assert RosEnvironment.camera_observations(node) is None
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)
    )

    def missing(*args):
        raise TransformException("Missing TF")

    node.tf_buffer.lookup_transform = missing
    assert RosEnvironment.camera_observations(node) is None


def test_camera_history_handles_tf_lag_longer_than_a_frame_period():
    from copy import deepcopy
    from tf2_ros import TransformException

    node = camera_environment()
    transform = node.tf_buffer.lookup_transform
    clock = 10_000_000_000
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=clock)
    )

    def delayed_transform(parent, child, stamp):
        if stamp.nanoseconds > clock - 133_000_000:
            raise TransformException("TF has not arrived yet")
        return transform(parent, child, stamp)

    node.tf_buffer.lookup_transform = delayed_transform
    for frame in range(9):
        stamp = round(frame * 1e9 / 15)
        clock = 10_000_000_000 + stamp
        for name in ("head", "left_wrist", "right_wrist"):
            image, info = camera_sample(name, stamp)
            depth = deepcopy(image) if name == "head" else None
            RosEnvironment.on_camera(node, name, image, info, depth)
        bundle = RosEnvironment.camera_observations(node)
        if frame < 2:
            assert bundle is None
            continue
        assert bundle is not None
        expected_stamp = round((frame - 2) * 1e9 / 15)
        for camera in bundle.values():
            assert camera.image.header.stamp.nanosec == expected_stamp
            assert camera.base_transform.header.stamp == camera.image.header.stamp
            assert camera.camera_info.header == camera.image.header
        assert bundle["head"].depth_image.header == bundle["head"].image.header

    # Once TF catches up, the most recent retained images are selected.
    node.tf_buffer.lookup_transform = transform
    bundle = RosEnvironment.camera_observations(node)
    assert all(camera.image.header.stamp.nanosec == stamp for camera in bundle.values())


def test_camera_history_falls_back_to_newest_bundle_with_compatible_skew():
    node = camera_environment()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_400_000_000)
    )
    for name in ("head", "left_wrist", "right_wrist"):
        RosEnvironment.on_camera(node, name, *camera_sample(name, 50_000_000))
        newest = 200_000_000 if name == "head" else 400_000_000
        RosEnvironment.on_camera(node, name, *camera_sample(name, newest))
    bundle = RosEnvironment.camera_observations(node)
    assert bundle is not None
    assert all(
        camera.image.header.stamp.nanosec == 50_000_000 for camera in bundle.values()
    )


def test_camera_history_is_bounded_and_expired_frames_cannot_reach_policy():
    node = camera_environment()
    for frame in range(30):
        for name in ("head", "left_wrist", "right_wrist"):
            RosEnvironment.on_camera(
                node, name, *camera_sample(name, frame * 1_000_000)
            )
    assert all(len(samples) == 10 for samples in node.camera_samples.values())
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=11_000_000_000)
    )
    assert RosEnvironment.camera_observations(node) is None


def test_camera_liveness_tolerates_dropped_frames_without_weakening_state_checks(
    monkeypatch,
):
    monkeypatch.setattr("aha_sim_tasks.ros_environment.time.monotonic", lambda: 100)
    node = SimpleNamespace(
        required={"joints", "camera:head"},
        received={"joints": 99, "camera:head": 96},
        camera_timeout=10,
    )
    assert RosEnvironment.stale_sources(node) == {}
    node.received["joints"] = 97
    assert RosEnvironment.stale_sources(node) == {"joints": 3}
    node.received["camera:head"] = 89
    assert RosEnvironment.stale_sources(node) == {"joints": 3, "camera:head": 11}


def test_physics_limits_are_bounded_for_moveit_but_not_observation():
    node = scoring_environment()
    node.truth_enabled = True
    node.truth_joints = Publisher()
    positions = {name: 0.0 for name in JOINT_LIMITS}
    positions["joint_r1"] = -LIMIT_MARGIN - 1e-11  # At the physics limit.
    positions["joint_r2"] = 1.57 + 1e-14
    # Excursions past the physics limit must remain visible.
    positions["joint_l1"] = -LIMIT_MARGIN - 0.001
    RosEnvironment.on_state(node, truth_sample(joint_positions=positions))
    message = node.truth_joints.messages[-1]
    monitored = dict(zip(message.name, message.position))
    assert monitored["joint_r1"] == 0
    assert monitored["joint_r2"] == 1.57
    assert node.positions["joint_r1"] == -LIMIT_MARGIN - 1e-11
    assert monitored["joint_l1"] == node.positions["joint_l1"] == -LIMIT_MARGIN - 0.001
