"""End to end: arm_trajectory_bridge.py against fake_arm on a test domain.

Runs the bridge (and a fake arm stand-in) as subprocesses on ROS_DOMAIN_ID 97
(``AHA_TEST_DOMAIN_ID`` to change) so they do not see a running robot, then
drives them from an in-process test node.
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(PKG, "scripts", "arm_trajectory_bridge.py")
FAKE_ARM = os.path.join(PKG, "test", "fake_arm.py")
DOMAIN_ID = int(os.environ.get("AHA_TEST_DOMAIN_ID", "97"))

JOINTS = ["joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6"]
LIMITS = {
    "joint_l2": (-1.57, 1.57),
    "joint_l3": (-1.57, 1.57),
    "joint_l4": (-3.14, 3.14),
    "joint_l5": (-1.57, 1.57),
    "joint_l6": (-1.57, 1.57),
}
TOL = 0.05


def _launch(script, params, tmp):
    path = os.path.join(tmp, f"params_{os.urandom(3).hex()}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"/**": {"ros__parameters": params}}, f)
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [PKG, env.get("PYTHONPATH")]))
    return subprocess.Popen(
        [sys.executable, script, "--ros-args", "--params-file", path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _bridge_params():
    params = {
        "controller_name": "test_arm_controller",
        "command_topic": "arm/joint_command",
        "state_topic": "arm/joint_states",
        "joint_names": JOINTS,
        "command_joint_names": JOINTS,
        "rate_hz": 50.0,
        "max_speed": 5.0,
        "goal_tolerance": TOL,
        "goal_timeout": 5.0,
        "state_timeout": 0.5,
        "closed_loop": False,
        "hold_interval": 1.0,
        "command_hold": 0.0,
    }
    for joint, (low, high) in LIMITS.items():
        params[f"{joint}__min"] = low
        params[f"{joint}__max"] = high
    return params


def _fake_arm_params():
    return {
        "joint_names": JOINTS,
        "command_topic": "arm/joint_command",
        "state_topic": "arm/joint_states",
        "publish_rate": 50.0,
    }


def _stop(proc, sig=signal.SIGINT, timeout=10.0):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            proc.send_signal(sig)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        out, _ = proc.communicate()
    return proc.returncode, out


def trajectory(points):
    msg = JointTrajectory(joint_names=list(JOINTS))
    for positions, t in points:
        sec = int(t)
        msg.points.append(
            JointTrajectoryPoint(
                positions=list(positions),
                time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9)),
            )
        )


class ArmBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fake_proc = _launch(FAKE_ARM, _fake_arm_params(), cls.tmp.name)
        cls.bridge_proc = _launch(BRIDGE, _bridge_params(), cls.tmp.name)

        cls.context = rclpy.Context()
        rclpy.init(context=cls.context, domain_id=DOMAIN_ID)
        cls.node = rclpy.create_node("arm_bridge_test", context=cls.context)
        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.node)

        cls.trajectory_pub = cls.node.create_publisher(
            JointTrajectory, "test_arm_controller/joint_trajectory", 10
        )
        cls.arm_state = {}
        cls.bridge_state = None
        cls.node.create_subscription(
            JointState, "arm/joint_states", lambda m: cls._on_arm_state(m)
        )
        cls.node.create_subscription(
            JointState, "test_arm_controller/joint_states", lambda m: cls._on_bridge_state(m)
        )
        cls.action_client = ActionClient(
            cls.node, FollowJointTrajectory, "test_arm_controller/follow_joint_trajectory"
        )
        cls._spin = threading.Thread(target=cls.executor.spin, daemon=True)
        cls._spin.start()

    @classmethod
    def _on_arm_state(cls, msg):
        cls.arm_state = dict(zip(msg.name, msg.position))

    @classmethod
    def _on_bridge_state(cls, msg):
        cls.bridge_state = dict(zip(msg.name, msg.position))

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown()
        cls._spin.join(timeout=5.0)
        _stop(cls.bridge_proc)
        _stop(cls.fake_proc)
        cls.tmp.cleanup()
        if rclpy.ok(context=cls.context):
            rclpy.shutdown(context=cls.context)

    def _until(self, predicate, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
            if predicate():
                return True
        return False

    def _l2(self):
        return self.arm_state.get("joint_l2")

    def test_1_bridge_publishes_state(self):
        self.assertTrue(
            self._until(lambda: self.bridge_state is not None),
            "bridge never published /test_arm_controller/joint_states",
        )
        for joint in JOINTS:
            self.assertIn(joint, self.bridge_state)

    def test_2_trajectory_reaches_target(self):
        self.trajectory_pub.publish(
            trajectory([((0.0, 0.0, 0.0, 0.0, 0.0), 0.0), ((1.0, 0.0, 0.0, 0.0, 0.0), 0.5)])
        )
        self.assertTrue(
            self._until(lambda: self._l2() is not None and abs(self._l2() - 1.0) <= TOL),
            f"arm did not reach target: {self.arm_state}",
        )
        self.assertAlmostEqual(self._l2(), 1.0, delta=TOL)
        self.assertAlmostEqual(self.arm_state.get("joint_l3", 0.0), 0.0, delta=TOL)

    def test_3_goal_succeeds(self):
        self.assertTrue(self.action_client.wait_for_server(timeout_sec=10.0))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory([((0.6, 0.0, 0.0, 0.0, 0.0), 0.5)])
        handle = self.action_client.send_goal_async(goal)
        self.assertTrue(self._until(lambda: handle.done(), timeout=15.0), "goal did not finish")
        result = handle.result()
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED, f"status {result.status}")


if __name__ == "__main__":
    unittest.main()
