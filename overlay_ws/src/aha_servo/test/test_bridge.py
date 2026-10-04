"""End to end: servo_trajectory_bridge.py against the fake bus on a pty.

Runs the bridge from the source tree on ROS_DOMAIN_ID 97 (AHA_TEST_DOMAIN_ID
to change), so it does not see a running sim.
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

from aha_servo.joint_map import parse_config
from fake_bus import FakeStsBus

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(PKG, "scripts", "servo_trajectory_bridge.py")
DOMAIN_ID = int(os.environ.get("AHA_TEST_DOMAIN_ID", "97"))
PAN, TILT = "joint_head_pan", "joint_head_tilt"
START = {12: 2100, 13: 2000}
TOL = 0.03


def bus_params(port, max_speed=3.0):
    return {
        "controller_name": "head_controller",
        "port": port,
        "baud": 921600,
        "rate_hz": 50.0,
        "max_speed": max_speed,
        "acc": 0,
        "goal_tolerance": TOL,
        "goal_timeout": 2.0,
        "joints": [PAN, TILT],
        PAN: {"id": 12, "zero": 2048, "sign": -1, "min": -1.57, "max": 1.57},
        TILT: {"id": 13, "zero": 2048, "sign": -1, "min": -0.524, "max": 1.047},
    }


def start_bridge(params, tmpdir):
    path = os.path.join(tmpdir, "bus.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"/**": {"ros__parameters": params}}, f)
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [PKG, env.get("PYTHONPATH")]))
    return subprocess.Popen(
        [sys.executable, BRIDGE, "--ros-args", "--params-file", path],
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


def trajectory(points, joints=(PAN, TILT)):
    msg = JointTrajectory(joint_names=list(joints))
    for positions, t in points:
        sec = int(t)
        msg.points.append(
            JointTrajectoryPoint(
                positions=list(positions),
                time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9)),
            )
        )
    return msg


class BridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeStsBus(START)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = parse_config(bus_params(cls.fake.port))
        cls.proc = start_bridge(bus_params(cls.fake.port), cls.tmp.name)

        cls.context = rclpy.Context()
        rclpy.init(context=cls.context, domain_id=DOMAIN_ID)
        cls.node = rclpy.create_node("aha_servo_bridge_test", context=cls.context)
        cls.states = {}
        cls.state_count = 0

        def on_state(msg):
            # Merge: a cycle with a failed read carries only the other joint.
            cls.states = {**cls.states, **dict(zip(msg.name, msg.position))}
            cls.state_count += 1

        cls.node.create_subscription(
            JointState, "/head_controller/joint_states", on_state, 10
        )
        cls.pub = cls.node.create_publisher(
            JointTrajectory, "/head_controller/joint_trajectory", 10
        )
        cls.action = ActionClient(
            cls.node, FollowJointTrajectory, "/head_controller/follow_joint_trajectory"
        )
        cls.executor = SingleThreadedExecutor(context=cls.context)
        cls.executor.add_node(cls.node)
        cls.spinner = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.spinner.start()

        deadline = time.monotonic() + 20.0
        while cls.state_count == 0 or cls.pub.get_subscription_count() == 0:
            if cls.proc.poll() is not None or time.monotonic() > deadline:
                _, out = stop_process(cls.proc)
                cls.tearDownClass()
                raise AssertionError(f"bridge did not come up:\n{out}")
            time.sleep(0.1)
        if not cls.action.wait_for_server(timeout_sec=10.0):
            raise AssertionError("action server not found")

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            stop_process(cls.proc, signal.SIGKILL)
        cls.executor.shutdown()
        cls.node.destroy_node()
        rclpy.try_shutdown(context=cls.context)
        cls.fake.close()
        cls.tmp.cleanup()

    # helpers -----------------------------------------------------------

    def wait_for(self, predicate, timeout, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail(f"timeout waiting for {what}; states={self.states}")

    def wait_reached(self, pan, tilt, timeout=5.0):
        self.wait_for(
            lambda: abs(self.states[PAN] - pan) < TOL
            and abs(self.states[TILT] - tilt) < TOL,
            timeout,
            f"pan {pan:+.3f} tilt {tilt:+.3f}",
        )

    def wait_future(self, future, timeout=10.0):
        self.wait_for(future.done, timeout, "future")
        return future.result()

    def send_goal(self, points, joints=(PAN, TILT), time_tolerance=0.0):
        goal = FollowJointTrajectory.Goal(trajectory=trajectory(points, joints))
        sec = int(time_tolerance)
        goal.goal_time_tolerance = Duration(
            sec=sec, nanosec=int((time_tolerance - sec) * 1e9)
        )
        feedback = []
        handle = self.wait_future(self.action.send_goal_async(goal, feedback.append))
        return handle, feedback

    def steps(self, joint, rad):
        return self.cfg.joint(joint).to_steps(rad)

    # tests (run in name order) -----------------------------------------

    def test_1_start_holds_present_position(self):
        pan, tilt = self.cfg.joint(PAN), self.cfg.joint(TILT)
        self.assertAlmostEqual(self.states[PAN], pan.to_rad(START[12]), places=3)
        self.assertAlmostEqual(self.states[TILT], tilt.to_rad(START[13]), places=3)
        for sid, pos in START.items():
            servo = self.fake.servo(sid)
            self.assertEqual(servo.torque, 1)
            self.assertEqual(servo.goal, pos)
            self.assertAlmostEqual(servo.pos, pos, delta=1)

    def test_2_topic_moves_to_final_point(self):
        self.pub.publish(trajectory([((0.1, 0.1), 0.2), ((0.4, 0.2), 0.5)]))
        self.wait_reached(0.4, 0.2)
        self.assertEqual(self.fake.servo(12).goal, self.steps(PAN, 0.4))
        self.assertEqual(self.fake.servo(13).goal, self.steps(TILT, 0.2))
        # Only the final point is used on the topic.
        self.assertNotIn(self.steps(PAN, 0.1), self.fake.servo(12).goals)

    def test_3_topic_clamps_to_limits(self):
        self.pub.publish(trajectory([((-3.0, 2.0), 0.5)]))
        self.wait_reached(-1.57, 1.047)
        self.assertEqual(self.fake.servo(12).goal, self.steps(PAN, -1.57))
        self.assertEqual(self.fake.servo(13).goal, self.steps(TILT, 1.047))

    def test_4_topic_partial_joints(self):
        self.pub.publish(trajectory([((0.0,), 0.3)], joints=(PAN,)))
        self.wait_reached(0.0, 1.047)

    def test_5_action_succeeds(self):
        handle, feedback = self.send_goal([((0.2, 0.1), 0.3), ((-0.3, 0.3), 0.8)])
        self.assertTrue(handle.accepted)
        result = self.wait_future(handle.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED)
        self.assertEqual(
            result.result.error_code, FollowJointTrajectory.Result.SUCCESSFUL
        )
        self.assertTrue(feedback)
        self.wait_reached(-0.3, 0.3, timeout=1.0)
        # Points are executed in order.
        goals = self.fake.servo(12).goals
        self.assertIn(self.steps(PAN, 0.2), goals)
        self.assertLess(goals.index(self.steps(PAN, 0.2)), len(goals) - 1)

    def test_6_action_cancel_holds(self):
        handle, _ = self.send_goal([((1.2, 0.0), 3.0)])
        time.sleep(0.6)
        self.wait_future(handle.cancel_goal_async())
        result = self.wait_future(handle.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_CANCELED)
        time.sleep(0.3)
        held = dict(self.states)
        time.sleep(0.5)
        self.assertLess(held[PAN], 1.0)
        self.assertAlmostEqual(self.states[PAN], held[PAN], delta=0.02)

    def test_7_action_rejects_unknown_joint(self):
        handle, _ = self.send_goal([((0.1,), 0.5)], joints=("joint_l2",))
        self.assertFalse(handle.accepted)

    def test_8_topic_preempts_action(self):
        handle, _ = self.send_goal([((-1.0, 0.0), 3.0)])
        time.sleep(0.3)
        self.pub.publish(trajectory([((0.5, 0.5), 0.5)]))
        result = self.wait_future(handle.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
        self.wait_reached(0.5, 0.5)

    def test_8a_timeout_abort_holds(self):
        # A slow servo misses the deadline; the abort must stop it there.
        self.fake.set_max_speed(12, 150)
        try:
            handle, _ = self.send_goal([((-0.5, 0.5), 0.5)], time_tolerance=0.5)
            result = self.wait_future(handle.get_result_async())
            self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
            self.assertEqual(
                result.result.error_code,
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
            )
            time.sleep(0.2)
            held = dict(self.states)
            time.sleep(1.0)
            self.assertGreater(held[PAN], 0.0)
            self.assertAlmostEqual(self.states[PAN], held[PAN], delta=0.02)
        finally:
            self.fake.set_max_speed(12)

    def test_8b_stale_state_never_succeeds(self):
        # The last read already equals the target, but the bus is silent.
        target = (self.states[PAN], self.states[TILT])
        self.fake.set_silent(True)
        try:
            time.sleep(0.5)
            count = self.state_count
            handle, _ = self.send_goal([(target, 0.2)], time_tolerance=0.5)
            result = self.wait_future(handle.get_result_async())
            self.assertEqual(result.status, GoalStatus.STATUS_ABORTED)
            self.assertEqual(
                result.result.error_code,
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
            )
            self.assertIn("no state", result.result.error_string)
            # No joint_states without fresh reads.
            self.assertEqual(self.state_count, count)
        finally:
            self.fake.set_silent(False)
        count = self.state_count
        self.wait_for(lambda: self.state_count > count, 3.0, "joint_states again")

    def test_8c_hold_without_fresh_read_sends_no_goal(self):
        # Replies are lost but writes still reach the servos, which keep moving:
        # holding the last read position would send them back there.
        self.pub.publish(trajectory([((0.0, 0.0), 0.3)]))
        self.wait_reached(0.0, 0.0)
        self.pub.publish(trajectory([((1.0, 0.0), 1.5)]))
        time.sleep(0.4)
        self.fake.set_mute(True)
        try:
            time.sleep(0.5)
            goals = {sid: len(self.fake.servo(sid).goals) for sid in START}
            self.pub.publish(trajectory([]))
            # Covers the hold's failed reads (3 retries per joint).
            time.sleep(1.5)
            for sid, n in goals.items():
                self.assertEqual(self.fake.servo(sid).goals[n:], [], f"ID{sid}")
        finally:
            self.fake.set_mute(False)
        # The servo kept its last goal.
        self.wait_reached(1.0, 0.0)

    def test_8d_stale_joint_gets_no_segment(self):
        self.fake.set_mute(True)
        try:
            time.sleep(0.6)
            goals = {sid: len(self.fake.servo(sid).goals) for sid in START}
            self.pub.publish(trajectory([((-0.5, 0.2), 0.5)]))
            time.sleep(1.0)
            for sid, n in goals.items():
                self.assertEqual(self.fake.servo(sid).goals[n:], [], f"ID{sid}")
        finally:
            self.fake.set_mute(False)
        self.pub.publish(trajectory([((-0.5, 0.2), 0.5)]))
        self.wait_reached(-0.5, 0.2)

    def test_8e_cancel_after_recovery_holds_present_position(self):
        pan = self.fake.servo(12)
        self.pub.publish(trajectory([((0.0, 0.0), 0.3)]))
        self.wait_reached(0.0, 0.0)
        handle, _ = self.send_goal([((1.2, 0.0), 2.0)])
        time.sleep(0.4)
        self.fake.set_silent(True)
        stale = self.steps(PAN, self.states[PAN])
        # The servo keeps moving while the bridge cannot read it.
        time.sleep(0.6)
        self.fake.set_silent(False)
        self.wait_future(handle.cancel_goal_async())
        result = self.wait_future(handle.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_CANCELED)
        time.sleep(0.3)
        self.assertGreater(abs(pan.goal - stale), 150, (pan.goal, stale))
        self.assertAlmostEqual(pan.pos, pan.goal, delta=5)

    def test_9_shutdown_disables_torque(self):
        # Ctrl-C under ros2 launch: SIGINT from the terminal, then from launch.
        os.killpg(self.proc.pid, signal.SIGINT)
        time.sleep(0.05)
        code, out = stop_process(self.proc, signal.SIGINT)
        self.assertEqual(code, 0, out)
        self.assertIn("torque off", out)
        self.assertNotIn("Traceback", out)
        self.assertNotIn("Failed to publish", out)
        for sid in START:
            self.assertEqual(self.fake.servo(sid).torque, 0)


class SigtermTest(unittest.TestCase):
    def test_sigterm_disables_torque(self):
        fake = FakeStsBus(START)
        with tempfile.TemporaryDirectory() as tmp:
            proc = start_bridge(bus_params(fake.port), tmp)
            try:
                deadline = time.monotonic() + 20.0
                while not all(fake.servo(sid).torque for sid in START):
                    self.assertIsNone(proc.poll(), "bridge exited early")
                    self.assertLess(time.monotonic(), deadline, "torque never came on")
                    time.sleep(0.1)
                code, out = stop_process(proc, signal.SIGTERM)
            finally:
                fake.close()
        self.assertEqual(code, 0, out)
        self.assertEqual([fake.servo(sid).torque for sid in START], [0, 0], out)


class MissingServoTest(unittest.TestCase):
    def test_exits_when_a_servo_does_not_answer(self):
        fake = FakeStsBus({12: 2048})
        with tempfile.TemporaryDirectory() as tmp:
            proc = start_bridge(bus_params(fake.port), tmp)
            try:
                out, _ = proc.communicate(timeout=20.0)
            except subprocess.TimeoutExpired:
                _, out = stop_process(proc, signal.SIGKILL)
                self.fail(f"bridge kept running:\n{out}")
            finally:
                fake.close()
        self.assertEqual(proc.returncode, 1, out)
        self.assertIn("joint_head_tilt (ID13)", out)
        self.assertEqual(fake.servo(12).torque, 0)


if __name__ == "__main__":
    unittest.main()
