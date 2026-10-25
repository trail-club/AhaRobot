"""End to end: arm_trajectory_bridge.py with config/lift.yaml against fake lifts.

fake_arm.py stands in for one lift_node per side. The bridge and the fakes run
from the source tree on ROS_DOMAIN_ID 96 (AHA_TEST_LIFT_DOMAIN_ID to change).
Only the right lift runs at first, as with `use_left_lift:=false`.
"""

import os
import tempfile
import threading
import time
import unittest

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from astra_controller_interfaces.msg import JointCommand
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

import test_arm_bridge as arm

DOMAIN_ID = int(os.environ.get("AHA_TEST_LIFT_DOMAIN_ID", "96"))
CONFIG = os.path.join(arm.PKG, "config", "lift.yaml")
PARAMS = yaml.safe_load(open(CONFIG, encoding="utf-8"))["aha_lift_bridge"][
    "ros__parameters"
]
CONTROLLER = f"/{PARAMS['controller_name']}"
LIFTS = {"left": "joint_l1", "right": "joint_r1"}
INITIAL = {"joint_l1": 0.4, "joint_r1": 0.2}


def fake_params(side):
    joint = LIFTS[side]
    return {
        "joint_names": [joint],
        "initial": [INITIAL[joint]],
        "max_rate": 0.3,
        "command_topic": PARAMS[joint]["command_topic"],
        "state_topic": PARAMS[joint]["state_topic"],
    }


def start(script, params, tmpdir, name):
    os.environ["ROS_DOMAIN_ID"] = str(DOMAIN_ID)
    sub = os.path.join(tmpdir, name)
    os.makedirs(sub, exist_ok=True)
    return arm.start(script, params, sub)


class LiftBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arm.DOMAIN_ID = DOMAIN_ID  # arm.start reads it for the child env
        cls.tmpdir = tempfile.mkdtemp()
        cls.fakes = {
            "right": start(arm.FAKE_ARM, fake_params("right"), cls.tmpdir, "r")
        }
        cls.bridge = start(arm.BRIDGE, PARAMS, cls.tmpdir, "bridge")
        rclpy.init()
        cls.node = rclpy.create_node("lift_bridge_test")
        cls.lock = threading.Lock()
        cls.commands = []  # (monotonic time, topic, names, positions)
        cls.state = {}
        for joint in LIFTS.values():
            topic = PARAMS[joint]["command_topic"]
            cls.node.create_subscription(
                JointCommand,
                topic,
                lambda msg, topic=topic: cls._on_command(topic, msg),
                qos_profile_sensor_data,
            )
            cls.node.create_subscription(
                JointState, PARAMS[joint]["state_topic"], cls._on_state, 10
            )
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
        code, out = arm.stop_process(cls.bridge)
        for fake in cls.fakes.values():
            arm.stop_process(fake)
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
    def _on_command(cls, topic, msg):
        with cls.lock:
            cls.commands.append(
                (time.monotonic(), topic, list(msg.name), list(msg.position_cmd))
            )

    @classmethod
    def _on_state(cls, msg):
        with cls.lock:
            cls.state.update(zip(msg.name, msg.position))

    wait_for = arm.BridgeTest.wait_for

    def position(self, joint):
        self.wait_for(lambda: joint in self.state, 10.0, f"{joint} state")
        time.sleep(0.1)
        with self.lock:
            return self.state[joint]

    def commands_since(self, t0):
        with self.lock:
            return [c for c in self.commands if c[0] >= t0]

    def settle(self):
        self.wait_for(
            lambda: not self.commands_since(time.monotonic() - 0.3), 10.0, "no commands"
        )

    def send_goal(self, msg):
        future = self.client.send_goal_async(FollowJointTrajectory.Goal(trajectory=msg))
        self.wait_for(future.done, 10.0, "goal response")
        return future.result()

    def wait_result(self, handle, timeout=15.0):
        future = handle.get_result_async()
        self.wait_for(future.done, timeout, "goal result")
        return future.result()

    # Tests (run in name order) --------------------------------------------

    def test_1_one_lift_moves_while_the_other_is_off(self):
        start_r = self.position("joint_r1")
        self.assertAlmostEqual(start_r, INITIAL["joint_r1"])
        t0 = time.monotonic()
        # 0.1 m in 2 s from rest: 0.05 m/s, 0.05 m/s^2 at the knots.
        handle = self.send_goal(arm.trajectory(["joint_r1"], [([start_r + 0.1], 2.0)]))
        self.assertTrue(handle.accepted)
        result = self.wait_result(handle)
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED)
        self.assertAlmostEqual(self.position("joint_r1"), start_r + 0.1, delta=0.003)
        sent = self.commands_since(t0)
        self.assertGreater(len(sent), 0)
        right = PARAMS["joint_r1"]["command_topic"]
        self.assertTrue(all(c[1] == right and c[2] == ["joint_r1"] for c in sent))

    def test_2_goals_for_a_lift_without_state_are_rejected(self):
        self.settle()
        r1 = self.position("joint_r1")
        for names, positions in (
            (["joint_l1"], [0.4]),
            (["joint_l1", "joint_r1"], [0.4, r1]),
        ):
            handle = self.send_goal(arm.trajectory(names, [(positions, 3.0)]))
            self.assertFalse(handle.accepted)

    def test_3_too_fast_lift_motion_is_rejected(self):
        self.settle()
        r1 = self.position("joint_r1")
        # 0.2 m in 1 s from rest needs 0.4 m/s^2, over 0.1.
        handle = self.send_goal(arm.trajectory(["joint_r1"], [([r1 - 0.2], 1.0)]))
        self.assertFalse(handle.accepted)
        # Outside [0, 1.19] m.
        handle = self.send_goal(arm.trajectory(["joint_r1"], [([1.2], 30.0)]))
        self.assertFalse(handle.accepted)

    def test_4_both_lifts_each_on_their_own_topic(self):
        self.settle()
        self.fakes["left"] = start(arm.FAKE_ARM, fake_params("left"), self.tmpdir, "l")
        l1 = self.position("joint_l1")
        r1 = self.position("joint_r1")
        time.sleep(0.5)  # the bridge's first state of the left lift
        t0 = time.monotonic()
        handle = self.send_goal(
            arm.trajectory(["joint_r1", "joint_l1"], [([r1 - 0.1, l1 + 0.1], 2.0)])
        )
        self.assertTrue(handle.accepted)
        self.assertEqual(self.wait_result(handle).status, GoalStatus.STATUS_SUCCEEDED)
        self.assertAlmostEqual(self.position("joint_l1"), l1 + 0.1, delta=0.003)
        self.assertAlmostEqual(self.position("joint_r1"), r1 - 0.1, delta=0.003)
        by_topic = {}
        for _, topic, names, _ in self.commands_since(t0):
            by_topic.setdefault(topic, set()).add(tuple(names))
        self.assertEqual(
            by_topic,
            {PARAMS[j]["command_topic"]: {(j,)} for j in ("joint_l1", "joint_r1")},
        )

    def test_5_cancel_holds_only_the_goal_lift(self):
        self.settle()
        r1 = self.position("joint_r1")
        t0 = time.monotonic()
        handle = self.send_goal(arm.trajectory(["joint_r1"], [([r1 + 0.2], 4.0)]))
        self.assertTrue(handle.accepted)
        time.sleep(1.0)
        cancel = handle.cancel_goal_async()
        self.wait_for(cancel.done, 5.0, "cancel response")
        self.assertEqual(self.wait_result(handle).status, GoalStatus.STATUS_CANCELED)
        self.settle()
        topics = {c[1] for c in self.commands_since(t0)}
        self.assertEqual(topics, {PARAMS["joint_r1"]["command_topic"]})
        last = [c for c in self.commands_since(t0)][-1]
        self.assertAlmostEqual(last[3][0], self.position("joint_r1"), delta=0.01)


if __name__ == "__main__":
    unittest.main()
