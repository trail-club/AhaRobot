"""End to end: arm_trajectory_bridge.py against fake_arm.py.

Both run from the source tree as processes on ROS_DOMAIN_ID 97
(AHA_TEST_DOMAIN_ID to change), so they do not see a running robot or sim.
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
from astra_controller_interfaces.msg import JointCommand
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, UInt32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(PKG, "scripts", "arm_trajectory_bridge.py")
FAKE_ARM = os.path.join(PKG, "test", "fake_arm.py")
DOMAIN_ID = int(os.environ.get("AHA_TEST_DOMAIN_ID", "97"))

JOINTS = ["joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6"]
INITIAL = [0.3, -0.2, 0.1, 0.0, 0.2]
CONTROLLER = "/test_arm_controller"
COMMAND_TOPIC = "/test/arm/joint_command"
STATE_TOPIC = "/test/joint_states"
RATE_HZ = 50.0
Result = FollowJointTrajectory.Result


def bridge_params():
    params = {
        "controller_name": CONTROLLER[1:],
        "command_topic": COMMAND_TOPIC,
        "state_topic": STATE_TOPIC,
        "rate_hz": RATE_HZ,
        "path_tolerance": 0.3,
        "goal_tolerance": 0.03,
        "goal_time": 2.0,
        "state_timeout": 0.3,
        "joints": JOINTS,
    }
    for name in JOINTS:
        params[name] = {
            "min": -1.5,
            "max": 1.5,
            "max_velocity": 1.0,
            "max_acceleration": 2.0,
        }
    return params


def fake_params():
    return {
        "joint_names": JOINTS,
        "initial": INITIAL,
        "max_rate": 2.0,
        "command_topic": COMMAND_TOPIC,
        "state_topic": STATE_TOPIC,
    }


def start(script, params, tmpdir):
    path = os.path.join(tmpdir, os.path.basename(script) + ".yaml")
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


def stop_process(proc, sig=signal.SIGINT, timeout=10.0):
    """Signal the process group; returns (returncode, output)."""
    if proc.poll() is None:
        os.killpg(proc.pid, sig)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    return proc.returncode, out


def trajectory(names, points):
    """points: [(positions, t) or (positions, velocities, t)]."""
    msg = JointTrajectory(joint_names=list(names))
    for point in points:
        positions, *velocities, t = point
        sec = int(t)
        msg.points.append(
            JointTrajectoryPoint(
                positions=[float(p) for p in positions],
                velocities=[float(v) for v in velocities[0]] if velocities else [],
                time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9)),
            )
        )
    return msg


class BridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ROS_DOMAIN_ID"] = str(DOMAIN_ID)
        cls.tmpdir = tempfile.mkdtemp()
        cls.fake = start(FAKE_ARM, fake_params(), cls.tmpdir)
        cls.bridge = start(BRIDGE, bridge_params(), cls.tmpdir)
        cls.stopped = None  # (returncode, output) once a test stopped the bridge
        rclpy.init()
        cls.node = rclpy.create_node("arm_bridge_test")
        cls.lock = threading.Lock()
        cls.commands = []  # (monotonic time, names, positions)
        cls.state = {}
        cls.rejected = 0
        cls.node.create_subscription(
            JointCommand, COMMAND_TOPIC, cls._on_command, qos_profile_sensor_data
        )
        cls.node.create_subscription(JointState, STATE_TOPIC, cls._on_state, 10)
        cls.node.create_subscription(UInt32, "/fake/rejected", cls._on_rejected, 10)
        cls.trajectory_pub = cls.node.create_publisher(
            JointTrajectory, f"{CONTROLLER}/joint_trajectory", 10
        )
        cls.stall_pub = cls.node.create_publisher(Bool, "/fake/stall", 10)
        cls.mute_pub = cls.node.create_publisher(Bool, "/fake/mute", 10)
        cls.client = ActionClient(
            cls.node, FollowJointTrajectory, f"{CONTROLLER}/follow_joint_trajectory"
        )
        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.node)
        cls.spinning = True
        cls.spin_thread = threading.Thread(target=cls._spin, daemon=True)
        cls.spin_thread.start()
        if not cls.client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("bridge action server did not come up")

    @classmethod
    def tearDownClass(cls):
        code, out = cls.stopped or stop_process(cls.bridge)
        stop_process(cls.fake)
        cls.spinning = False
        cls.spin_thread.join(timeout=5.0)
        cls.node.destroy_node()
        rclpy.try_shutdown()
        print(out)
        if code != 0 or "stopped" not in out:
            raise AssertionError(f"bridge did not stop cleanly (exit {code})")

    @classmethod
    def _spin(cls):
        while cls.spinning:
            cls.executor.spin_once(timeout_sec=0.05)

    @classmethod
    def _on_command(cls, msg):
        with cls.lock:
            cls.commands.append(
                (time.monotonic(), list(msg.name), list(msg.position_cmd))
            )

    @classmethod
    def _on_state(cls, msg):
        with cls.lock:
            cls.state.update(zip(msg.name, msg.position))

    @classmethod
    def _on_rejected(cls, msg):
        cls.rejected = msg.data

    # Helpers --------------------------------------------------------------

    def wait_for(self, predicate, timeout, message="condition"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {message}")

    def current(self):
        self.wait_for(lambda: all(n in self.state for n in JOINTS), 10.0, "arm state")
        time.sleep(0.1)
        with self.lock:
            return {n: self.state[n] for n in JOINTS}

    def commands_since(self, t0):
        with self.lock:
            return [c for c in self.commands if c[0] >= t0]

    def settle(self):
        """Wait until the bridge has stopped sending."""
        self.wait_for(
            lambda: not self.commands_since(time.monotonic() - 0.3), 10.0, "no commands"
        )

    def send_goal(self, msg):
        goal = FollowJointTrajectory.Goal(trajectory=msg)
        feedback = []
        future = self.client.send_goal_async(goal, feedback_callback=feedback.append)
        self.wait_for(future.done, 10.0, "goal response")
        return future.result(), feedback

    def wait_result(self, handle, timeout=15.0):
        future = handle.get_result_async()
        self.wait_for(future.done, timeout, "goal result")
        return future.result()

    def set_fake(self, pub, value):
        pub.publish(Bool(data=value))
        time.sleep(0.2)

    # Tests (run in name order) --------------------------------------------

    def test_1_nothing_is_sent_before_a_trajectory(self):
        start_pose = self.current()
        self.assertEqual(start_pose, dict(zip(JOINTS, INITIAL)))
        time.sleep(1.0)
        self.assertEqual(self.commands_since(0.0), [])

    def test_2_topic_trajectory_starts_at_the_measured_pose(self):
        self.settle()
        start_pose = self.current()
        target = start_pose["joint_l2"] + 0.4
        t0 = time.monotonic()
        # 0.4 rad/s, 0.8 rad/s^2 from rest.
        self.trajectory_pub.publish(trajectory(["joint_l2"], [([target], 1.0)]))
        self.wait_for(
            lambda: abs(self.current()["joint_l2"] - target) < 0.02, 5.0, "target"
        )
        sent = self.commands_since(t0)
        first = dict(zip(sent[0][1], sent[0][2]))
        for name in JOINTS:
            self.assertAlmostEqual(first[name], start_pose[name], delta=0.02)
        moving = [c for c in sent if c[0] <= sent[0][0] + 1.0]
        self.assertGreaterEqual(len(moving), 0.7 * RATE_HZ)
        # 0.008 rad per sample; the best-effort subscription here may miss some.
        steps = [abs(b[2][0] - a[2][0]) for a, b in zip(sent, sent[1:])]
        self.assertLessEqual(max(steps), 0.05)
        for name in JOINTS[1:]:
            held = [dict(zip(c[1], c[2]))[name] for c in sent]
            self.assertEqual(set(held), {start_pose[name]})

    def test_3_goal_like_moveit_succeeds_with_feedback(self):
        self.settle()
        start_pose = self.current()
        positions = [start_pose[n] for n in JOINTS]
        mid, end = list(positions), list(positions)
        mid[1] += 0.15
        end[1] += 0.3
        zero = [0.0] * 5
        middle = [0.0, 0.3, 0.0, 0.0, 0.0]
        handle, feedback = self.send_goal(
            trajectory(
                JOINTS,
                [(positions, zero, 0.0), (mid, middle, 0.75), (end, zero, 1.5)],
            )
        )
        self.assertTrue(handle.accepted)
        result = self.wait_result(handle)
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED)
        self.assertEqual(result.result.error_code, Result.SUCCESSFUL)
        self.assertGreater(len(feedback), 5)
        self.assertAlmostEqual(self.current()["joint_l3"], end[1], delta=0.03)

    def test_4_invalid_goals_are_rejected(self):
        self.settle()
        start_pose = self.current()
        l2 = start_pose["joint_l2"]
        for msg in (
            trajectory(["joint_l2"], [([l2 + 0.8], 0.5)]),  # 1.6 rad/s
            trajectory(["joint_l2"], [([1.6], 5.0)]),  # outside the limits
            trajectory(["joint_l2"], [([l2 + 0.3], 0.0), ([l2], 1.0)]),  # not the start
            trajectory(["joint_x"], [([0.0], 1.0)]),  # unknown joint
        ):
            handle, _ = self.send_goal(msg)
            self.assertFalse(handle.accepted)
        t0 = time.monotonic()
        self.trajectory_pub.publish(trajectory(["joint_l2"], [([l2 + 0.8], 0.5)]))
        time.sleep(0.5)
        self.assertEqual(self.commands_since(t0), [])

    def test_5_cancel_holds_the_measured_position(self):
        self.settle()
        start_pose = self.current()
        target = start_pose["joint_l4"] + 0.8
        handle, _ = self.send_goal(trajectory(["joint_l4"], [([target], 2.0)]))
        self.assertTrue(handle.accepted)
        time.sleep(0.7)
        cancel = handle.cancel_goal_async()
        self.wait_for(cancel.done, 5.0, "cancel response")
        result = self.wait_result(handle)
        self.assertEqual(result.status, GoalStatus.STATUS_CANCELED)
        self.settle()
        last = dict(zip(*self.commands_since(0.0)[-1][1:]))
        held = self.current()["joint_l4"]
        self.assertAlmostEqual(last["joint_l4"], held, delta=0.05)
        self.assertGreater(target - held, 0.2)

    def test_6_stalled_arm_aborts_off_the_path(self):
        self.settle()
        start_pose = self.current()
        self.set_fake(self.stall_pub, True)
        try:
            handle, _ = self.send_goal(
                trajectory(["joint_l2"], [([start_pose["joint_l2"] - 0.5], 1.0)])
            )
            self.assertTrue(handle.accepted)
            result = self.wait_result(handle)
            self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
            self.assertEqual(result.result.error_code, Result.PATH_TOLERANCE_VIOLATED)
            self.assertIn("off the path", result.result.error_string)
        finally:
            self.set_fake(self.stall_pub, False)
        self.settle()
        last = dict(zip(*self.commands_since(0.0)[-1][1:]))
        self.assertAlmostEqual(last["joint_l2"], start_pose["joint_l2"], delta=0.05)

    def test_7_missing_state_aborts_and_rejects(self):
        self.settle()
        start_pose = self.current()
        handle, _ = self.send_goal(
            trajectory(["joint_l5"], [([start_pose["joint_l5"] + 0.6], 2.0)])
        )
        self.assertTrue(handle.accepted)
        time.sleep(0.4)
        self.set_fake(self.mute_pub, True)
        try:
            result = self.wait_result(handle)
            self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
            self.assertIn("no state", result.result.error_string)
            self.settle()
            handle, _ = self.send_goal(
                trajectory(["joint_l5"], [([start_pose["joint_l5"]], 2.0)])
            )
            self.assertFalse(handle.accepted)
        finally:
            self.set_fake(self.mute_pub, False)

    def test_8_newer_goal_preempts(self):
        self.settle()
        start_pose = self.current()
        first, _ = self.send_goal(
            trajectory(["joint_l6"], [([start_pose["joint_l6"] + 0.6], 3.0)])
        )
        time.sleep(0.5)
        second, _ = self.send_goal(
            trajectory(["joint_l6"], [([start_pose["joint_l6"]], 2.0)])
        )
        self.assertTrue(first.accepted and second.accepted)
        result = self.wait_result(first)
        self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
        self.assertIn("preempted", result.result.error_string)
        self.assertEqual(self.wait_result(second).status, GoalStatus.STATUS_SUCCEEDED)

    def test_9_every_command_names_the_arm_joints(self):
        sent = self.commands_since(0.0)
        self.assertGreater(len(sent), 0)
        self.assertTrue(all(names == JOINTS for _, names, _ in sent))
        time.sleep(0.2)
        self.assertEqual(self.rejected, 0)

    def test_99_sigint_aborts_the_running_goal_with_a_result(self):
        # MoveIt waits for the result of a goal it sent; a bridge stopped
        # under it must still answer. Stops the bridge: runs last.
        self.settle()
        start_pose = self.current()
        handle, _ = self.send_goal(
            trajectory(["joint_l2"], [([start_pose["joint_l2"] - 0.6], 3.0)])
        )
        self.assertTrue(handle.accepted)
        result = handle.get_result_async()  # requested right away, like MoveIt
        time.sleep(0.5)
        os.killpg(self.bridge.pid, signal.SIGINT)
        self.wait_for(result.done, 5.0, "goal result after SIGINT")
        self.assertEqual(result.result().status, GoalStatus.STATUS_ABORTED)
        self.assertIn("shutting down", result.result().result.error_string)
        type(self).stopped = stop_process(self.bridge)


if __name__ == "__main__":
    unittest.main()
