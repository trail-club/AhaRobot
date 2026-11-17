"""End to end: base_node.py with the fake odrive package (no hardware).

Runs the node from the source tree on ROS_DOMAIN_ID 97 (AHA_TEST_DOMAIN_ID to
change), so it does not see a running sim.
"""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from tf2_msgs.msg import TFMessage

TEST = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(TEST)
NODE = os.path.join(PKG, "scripts", "base_node.py")
DOMAIN_ID = int(os.environ.get("AHA_TEST_DOMAIN_ID", "97"))


def start_node(tmpdir, mode="ok", params=None, extra_env=None):
    log = os.path.join(tmpdir, f"odrive_{mode}.jsonl")
    env = dict(
        os.environ,
        ROS_DOMAIN_ID=str(DOMAIN_ID),
        PYTHONUNBUFFERED="1",
        FAKE_ODRIVE_MODE=mode,
        FAKE_ODRIVE_LOG=log,
        **(extra_env or {}),
    )
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [os.path.join(TEST, "fake_odrive"), PKG, env.get("PYTHONPATH")])
    )
    args = [sys.executable, NODE, "--ros-args"]
    for k, v in (params or {}).items():
        args += ["-p", f"{k}:={v}"]
    proc = subprocess.Popen(
        args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return proc, log


def read_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def last(events, path):
    vals = [e["value"] for e in events if e["path"] == path]
    return vals[-1] if vals else None


def values(events, path):
    return [e["value"] for e in events if e["path"] == path]


@pytest.fixture(scope="module")
def ros():
    rclpy.init(domain_id=DOMAIN_ID)
    node = rclpy.create_node("aha_base_test")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    yield node, ex
    ex.shutdown()
    node.destroy_node()
    rclpy.shutdown()


def spin_for(ex, seconds, every=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if every:
            every()
        ex.spin_once(timeout_sec=0.02)


def test_drive_odom_and_stop(ros):
    node, ex = ros
    odoms, tfs = [], []
    sub = node.create_subscription(
        Odometry, "/diff_drive_controller/odom", odoms.append, 10
    )
    tsub = node.create_subscription(TFMessage, "/tf", tfs.append, 10)
    pub = node.create_publisher(TwistStamped, "/diff_drive_controller/cmd_vel", 10)

    def send(linear, angular):
        def f():
            msg = TwistStamped()
            msg.twist.linear.x = linear
            msg.twist.angular.z = angular
            pub.publish(msg)

        return f

    with tempfile.TemporaryDirectory() as tmp:
        proc, log = start_node(tmp, extra_env={"FAKE_ODRIVE_START": "5.0,-3.0"})
        try:
            deadline = time.monotonic() + 30
            while not odoms and time.monotonic() < deadline:
                assert proc.poll() is None, proc.stdout.read()
                ex.spin_once(timeout_sec=0.1)
            assert odoms, "no odom"
            first = read_log(log)
            # setpoint = present position before closed loop: no jump
            assert values(first, "axis0.controller.input_pos")[0] == 5.0
            assert values(first, "axis1.controller.input_pos")[0] == -3.0

            # forward 0.2 m/s for 2 s (0.46 wheel turn/s)
            spin_for(ex, 2.0, send(0.2, 0.0))
            o = odoms[-1]
            assert 0.2 < o.pose.pose.position.x < 0.45
            assert abs(o.pose.pose.position.y) < 0.01
            assert o.twist.twist.linear.x == pytest.approx(0.2, abs=0.05)
            assert o.header.frame_id == "odom"
            assert o.child_frame_id == "base_footprint"
            ev = read_log(log)
            # left motor turns backwards for forward (left_sign -1)
            assert last(ev, "axis0.controller.input_pos") < 5.0 - 0.5
            assert last(ev, "axis1.controller.input_pos") > -3.0 + 0.5

            # no command: stops after cmd_timeout + trajectory catch-up
            spin_for(ex, 1.5)
            x_stop = odoms[-1].pose.pose.position.x
            spin_for(ex, 1.0)
            assert odoms[-1].pose.pose.position.x == pytest.approx(x_stop, abs=1e-6)
            assert odoms[-1].twist.twist.linear.x == pytest.approx(0.0, abs=1e-6)

            # turn left in place
            spin_for(ex, 2.0, send(0.0, 0.5))
            yaw = 2 * math.atan2(
                odoms[-1].pose.pose.orientation.z, odoms[-1].pose.pose.orientation.w
            )
            assert 0.5 < yaw < 1.1
            assert odoms[-1].pose.pose.position.x == pytest.approx(x_stop, abs=0.02)
            assert any(
                t.child_frame_id == "base_footprint" and t.header.frame_id == "odom"
                for m in tfs
                for t in m.transforms
            )
        finally:
            proc.send_signal(signal.SIGINT)
            out = proc.communicate(timeout=15)[0]
        assert proc.returncode == 0, out
        ev = read_log(log)
        for ax in ("axis0", "axis1"):
            assert last(ev, f"{ax}.requested_state") == 1
            assert last(ev, f"{ax}.config.enable_watchdog") is False
            assert last(ev, f"{ax}.controller.config.input_mode") == 1
            assert last(ev, f"{ax}.trap_traj.config.vel_limit") == 2.0
            assert values(ev, f"{ax}.watchdog_expired") == []
            # gains were not written (device values kept)
            assert values(ev, f"{ax}.controller.config.pos_gain") == []
    node.destroy_subscription(sub)
    node.destroy_subscription(tsub)
    node.destroy_publisher(pub)


def test_speed_limited_by_traj_vel_limit(ros):
    node, ex = ros
    odoms = []
    sub = node.create_subscription(
        Odometry, "/diff_drive_controller/odom", odoms.append, 10
    )
    pub = node.create_publisher(TwistStamped, "/diff_drive_controller/cmd_vel", 10)

    def fast():
        msg = TwistStamped()
        msg.twist.linear.x = 5.0
        pub.publish(msg)

    with tempfile.TemporaryDirectory() as tmp:
        proc, _ = start_node(tmp, params={"traj_vel_limit": 0.5})
        try:
            deadline = time.monotonic() + 30
            while not odoms and time.monotonic() < deadline:
                assert proc.poll() is None, proc.stdout.read()
                ex.spin_once(timeout_sec=0.1)
            spin_for(ex, 2.0, fast)
            limit = 0.5 * math.pi * 0.139
            assert odoms[-1].twist.twist.linear.x == pytest.approx(limit, rel=0.1)
        finally:
            proc.send_signal(signal.SIGINT)
            out = proc.communicate(timeout=15)[0]
        assert proc.returncode == 0, out
    node.destroy_subscription(sub)
    node.destroy_publisher(pub)


@pytest.mark.parametrize(
    "mode,params,message",
    [
        ("missing", {"connect_timeout": 1.0}, "no ODrive on USB"),
        ("uncalibrated", {}, "axis1 motor is not calibrated"),
        ("fault", {}, "left closed-loop control"),
    ],
)
def test_failures_exit_1_with_axes_idle(mode, params, message):
    with tempfile.TemporaryDirectory() as tmp:
        proc, log = start_node(tmp, mode=mode, params=params)
        try:
            out = proc.communicate(timeout=30)[0]
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 1, out
        assert message in out
        if mode == "missing":
            return
        ev = read_log(log)
        for ax in ("axis0", "axis1"):
            assert last(ev, f"{ax}.requested_state") in (None, 1)
            assert last(ev, f"{ax}.config.enable_watchdog") in (None, False)
        if mode == "uncalibrated":
            assert 8 not in values(ev, "axis0.requested_state")


def test_bad_parameter_exits_2():
    with tempfile.TemporaryDirectory() as tmp:
        proc, _ = start_node(tmp, params={"left_axis": 1})
        out = proc.communicate(timeout=30)[0]
    assert proc.returncode == 2, out
    assert "left_axis" in out
