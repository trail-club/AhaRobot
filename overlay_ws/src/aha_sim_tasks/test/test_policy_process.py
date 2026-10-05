"""Exercise the actual worker boundary, failures, and ROS discovery separation."""

import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType, SimpleNamespace
import uuid

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DOMAIN = 91
POLICY_DOMAIN = 92
sys.path.insert(0, str(PACKAGE_ROOT))
from aha_sim_tasks.api import Action, Observation  # noqa: E402
from aha_sim_tasks.policy_process import PolicyProcess  # noqa: E402
from aha_sim_tasks.policy_protocol import (  # noqa: E402
    decode_observation,
    observation_message,
)
from aha_sim_tasks.processes import partition_processes  # noqa: E402


@pytest.fixture
def worker_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(EVALUATOR_DOMAIN))
    monkeypatch.setenv("GZ_PARTITION", "aha-worker-test-" + uuid.uuid4().hex)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [str(tmp_path), str(PACKAGE_ROOT), os.environ.get("PYTHONPATH", "")]
        ),
    )
    monkeypatch.setenv("AHA_POLICY_TEST_RECORD", str(tmp_path / "record.json"))
    return tmp_path


def worker(directory, policy="noop", **kwargs):
    return PolicyProcess(
        policy,
        "pick_apple",
        "Pick the apple.",
        POLICY_DOMAIN,
        directory,
        os.environ["GZ_PARTITION"],
        **kwargs,
    )


def ready(process):
    deadline = time.monotonic() + 10
    while not process.ready:
        process.poll()
        assert time.monotonic() < deadline
        time.sleep(0.01)


def response(process):
    deadline = time.monotonic() + 10
    while True:
        action = process.poll()
        if action is not None:
            return action
        assert time.monotonic() < deadline
        time.sleep(0.01)


def observation(image=None):
    return Observation(
        1.5,
        "pick_apple",
        "Pick the apple.",
        MappingProxyType({"joint_r1": 0.1}),
        MappingProxyType({"joint_r1": 0.0}),
        (0.2, 0.0, 0.0),
        image,
    )


def test_custom_factory_and_frames_stay_outside_evaluator(worker_environment):
    directory = worker_environment
    (directory / "boundary_probe.py").write_text("""
import inspect, json, os, sys
from pathlib import Path
from aha_sim_tasks.api import Action
record = Path(os.environ["AHA_POLICY_TEST_RECORD"])
record.write_text(json.dumps({"pid": os.getpid(), "domain": os.environ["ROS_DOMAIN_ID"],
                             "partition": os.environ["GZ_PARTITION"]}))
class Probe:
    phase = "probe"
    def reset(self, task_id, instruction):
        self.task_id, self.instruction = task_id, instruction
    def act(self, observation):
        data = json.loads(record.read_text())
        data["evaluator_imported"] = "aha_sim_tasks.ros_environment" in sys.modules
        data["parent_frame"] = any("privileged_marker" in f.frame.f_locals
                                   for f in inspect.stack())
        data["fields"] = sorted(vars(observation))
        data["reset"] = [self.task_id, self.instruction]
        try:
            observation.joint_positions["joint_r1"] = 0.5
        except TypeError:
            data["read_only"] = True
        record.write_text(json.dumps(data))
        print("custom policy output")
        return Action(linear_velocity=0.07)
def create():
    return Probe()
""")
    privileged_marker = object()
    assert privileged_marker is not None
    process = worker(directory, "boundary_probe:create")
    try:
        ready(process)
        process.act(observation())
        assert response(process) == Action(linear_velocity=0.07)
        assert process.phase == "probe"
        data = json.loads((directory / "record.json").read_text())
        assert "boundary_probe" not in sys.modules
        assert data["pid"] != os.getpid()
        assert data["domain"] == str(POLICY_DOMAIN)
        assert data["partition"] != os.environ["GZ_PARTITION"]
        assert not data["evaluator_imported"] and not data["parent_frame"]
        assert data["read_only"]
        assert data["reset"] == ["pick_apple", "Pick the apple."]
        assert data["fields"] == sorted(Observation.__dataclass_fields__)
    finally:
        process.close()
    assert process.process.returncode == 0
    assert "custom policy output" in (directory / "policy.log").read_text()


def test_observation_serializer_excludes_extra_attributes():
    source = SimpleNamespace(**vars(observation()), world="privileged", score=1)
    message = observation_message(source)
    assert set(message["observation"]) == set(Observation.__dataclass_fields__)
    decoded = decode_observation(message["observation"])
    assert decoded == observation()


@pytest.mark.parametrize("large", [False, True])
def test_optional_ros_image_survives_worker_ipc(worker_environment, large):
    pytest.importorskip("rclpy")
    from sensor_msgs.msg import Image

    (worker_environment / "image_probe.py").write_text("""
from aha_sim_tasks.api import Action
class Probe:
    def reset(self, task_id, instruction): pass
    def act(self, observation):
        image = observation.head_image
        assert image.header.frame_id == "camera_color_optical_frame"
        assert image.header.stamp.sec == 12 and image.header.stamp.nanosec == 34
        assert image.step == image.width * 3
        expected = bytes(range(6)) if image.width == 2 else bytes(range(256)) * 3600
        assert image.encoding == "rgb8" and bytes(image.data) == expected
        return Action()
def create(): return Probe()
""")
    image = Image(
        height=480 if large else 1,
        width=640 if large else 2,
        encoding="rgb8",
        step=1920 if large else 6,
        data=bytes(range(256)) * 3600 if large else bytes(range(6)),
    )
    image.header.frame_id = "camera_color_optical_frame"
    image.header.stamp.sec, image.header.stamp.nanosec = 12, 34
    process = worker(worker_environment, "image_probe:create")
    try:
        ready(process)
        process.act(observation(image))
        assert response(process) == Action()
    finally:
        process.close()


@pytest.mark.parametrize("failure", ["invalid", "exception", "crash", "hang"])
def test_worker_failures_are_bounded_and_cleaned(worker_environment, failure):
    code = {
        "invalid": "return Action(linear_velocity=0.9)",
        "exception": 'raise ValueError("inference failed")',
        "crash": "os._exit(3)",
        "hang": "time.sleep(60); return Action()",
    }[failure]
    (worker_environment / "failure_probe.py").write_text(f"""
import os, time
from aha_sim_tasks.api import Action
class Probe:
    def reset(self, task_id, instruction): pass
    def act(self, observation):
        {code}
def create(): return Probe()
""")
    process = worker(worker_environment, "failure_probe:create", action_timeout=0.2)
    try:
        ready(process)
        started = time.monotonic()
        process.act(observation())
        error = {"invalid": ValueError, "hang": TimeoutError}.get(failure, RuntimeError)
        with pytest.raises(error):
            response(process)
        assert time.monotonic() - started < 3
    finally:
        process.close()
    assert process.process.poll() is not None
    assert not partition_processes(process.partition)


@pytest.mark.parametrize("failure", ["import", "reset", "hang"])
def test_worker_initialization_errors(worker_environment, failure):
    (worker_environment / "init_probe.py").write_text("""
import time
from aha_sim_tasks.api import Action
class Probe:
    def reset(self, task_id, instruction): raise ValueError("reset failed")
    def act(self, observation): return Action()
def create(): return Probe()
def hang(): time.sleep(60); return Probe()
""")
    name = {
        "import": "missing_policy_module:create",
        "reset": "init_probe:create",
        "hang": "init_probe:hang",
    }[failure]
    process = worker(worker_environment, name, startup_timeout=0.5)
    try:
        with pytest.raises(TimeoutError if failure == "hang" else RuntimeError):
            ready(process)
    finally:
        process.close()
    assert not partition_processes(process.partition)


def test_policy_domain_must_differ_from_evaluator(worker_environment):
    with pytest.raises(ValueError, match="domains must differ"):
        PolicyProcess(
            "noop",
            "pick_apple",
            "Pick.",
            EVALUATOR_DOMAIN,
            worker_environment,
            os.environ["GZ_PARTITION"],
        )


def test_worker_cannot_discover_evaluation_topics(worker_environment):
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String

    (worker_environment / "discovery_probe.py").write_text("""
import json, os, time
from pathlib import Path
import rclpy
from std_msgs.msg import UInt32, String
from aha_sim_tasks.api import Action
class Probe:
    def reset(self, task_id, instruction):
        rclpy.init()
        self.node = rclpy.create_node("policy_discovery_probe")
        self.received = []
        self.local = []
        self.node.create_subscription(String, "/evaluation/state",
                                      lambda msg: self.received.append("state"), 10)
        self.publisher = self.node.create_publisher(UInt32, "/policy_probe", 10)
        self.node.create_subscription(UInt32, "/policy_probe",
                                      lambda msg: self.local.append(msg.data), 10)
    def act(self, observation):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self.publisher.publish(UInt32(data=42))
            rclpy.spin_once(self.node, timeout_sec=0.05)
        data = {"received": self.received, "local": self.local,
                "nodes": self.node.get_node_names()}
        Path(os.environ["AHA_POLICY_TEST_RECORD"]).write_text(json.dumps(data))
        return Action()
def create(): return Probe()
""")
    context = Context()
    rclpy.init(context=context, domain_id=EVALUATOR_DOMAIN)
    node = rclpy.create_node("evaluation_privileged_probe", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    state = node.create_publisher(String, "/evaluation/state", 10)
    process = worker(worker_environment, "discovery_probe:create")
    try:
        ready(process)
        process.act(observation())
        deadline = time.monotonic() + 10
        while process.pending is not None:
            state.publish(String(data='{"privileged": true}'))
            executor.spin_once(timeout_sec=0.01)
            process.poll()
            assert time.monotonic() < deadline
        data = json.loads((worker_environment / "record.json").read_text())
        assert data["local"], "Worker ROS transport must be functioning"
        assert data["received"] == []
        assert "evaluation_privileged_probe" not in data["nodes"]
    finally:
        process.close()
        executor.shutdown()
        node.destroy_node()
        context.shutdown()
