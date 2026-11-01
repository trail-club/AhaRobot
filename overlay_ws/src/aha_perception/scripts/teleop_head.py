#!/usr/bin/env python3
"""Keyboard teleop for the head (pan/tilt).

Prereq: a head_controller is running, either the simulation
(sim.launch.py) or the real head (real_head_camera.launch.py). Run in an
interactive shell on the same ROS_DOMAIN_ID:
  ros2 run aha_perception teleop_head.py

Keys: arrows = 5 deg steps (left/right = pan, up/down = tilt),
      space = center, q = quit.
Limits and move speed come from aha_perception/config/head.yaml.
"""

import math
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

try:
    import aha_perception  # noqa: F401
except ModuleNotFoundError:
    # Stale PYTHONPATH (shell sourced before aha_perception had a Python
    # module): find it via the ament index.
    from ament_index_python.packages import get_package_prefix

    sys.path.append(
        os.path.join(
            get_package_prefix("aha_perception"),
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )
    )

from aha_perception.head_config import (
    TRAJECTORY_TOPIC,
    declare_head_config,
    default_params_file,
    describe,
    head_position,
)

STEP = math.radians(5.0)
# Callbacks run per loop at most (joint_states at 30+ Hz, loop at 10 Hz).
DRAIN_MAX = 20
# (pan, tilt) deltas; +pan = right, +tilt = down
KEYS = {
    "\x1b[D": (-STEP, 0.0),  # left
    "\x1b[C": (STEP, 0.0),  # right
    "\x1b[A": (0.0, -STEP),  # up
    "\x1b[B": (0.0, STEP),  # down
}

HELP = """\
頭部テレオペ (head_controller)
  ← / →  : パン 左 / 右 (5°)
  ↑ / ↓  : チルト 上 / 下 (5°)
  space  : 正面
  q      : 終了
"""


def read_keys(fd, timeout):
    """Return the keys typed within timeout; arrow escape sequences are one key.

    Reads the raw fd (not sys.stdin) so buffered bytes never hide from select.
    """
    if not select.select([fd], [], [], timeout)[0]:
        return []
    data = os.read(fd, 64).decode(errors="ignore")
    keys = []
    while data:
        n = 3 if data.startswith("\x1b[") else 1
        keys.append(data[:n])
        data = data[n:]
    return keys


def main():
    rclpy.init(args=["--ros-args", "--params-file", default_params_file()])
    node = rclpy.create_node("aha_teleop_head")
    head = declare_head_config(node)
    pub = node.create_publisher(JointTrajectory, TRAJECTORY_TOPIC, 10)
    state = {"current": None, "received": 0}

    def on_joint_states(msg):
        state["received"] += 1
        pos = head_position(msg)
        if pos is not None:
            state["current"] = pos

    node.create_subscription(
        JointState, "/joint_states", on_joint_states, qos_profile_sensor_data
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def drain():
        """Take every queued joint_states, not just one per loop."""
        for _ in range(DRAIN_MAX):
            received = state["received"]
            executor.spin_once(timeout_sec=0.0)
            if state["received"] == received:
                break

    deadline = time.monotonic() + 3.0
    while state["current"] is None and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
    if state["current"] is None:
        print("警告: /joint_states が来ないので正面 (0, 0) から始めます")
    target = head.clamp(*(state["current"] or (0.0, 0.0)))[:2]

    print(HELP)
    print(f"目標: {describe(*target)}")
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        running = True
        while running and rclpy.ok():
            keys = read_keys(fd, 0.1)
            drain()
            new, clamped, center = target, False, False
            for key in keys:
                if key == "q":
                    running = False
                    break
                if key == " ":
                    new, center = (0.0, 0.0), True
                elif key in KEYS:
                    pan, tilt, hit = head.clamp(
                        new[0] + KEYS[key][0], new[1] + KEYS[key][1]
                    )
                    new, clamped = (pan, tilt), clamped or hit
            if not running or (new == target and not (clamped or center)):
                continue
            target = new
            pub.publish(
                head.trajectory(*target, head.duration(state["current"], target))
            )
            note = " (限界)" if clamped else ""
            print(
                f"\r目標: {describe(*target)}  "
                f"[pan {target[0]:+.3f}, tilt {target[1]:+.3f} rad]{note}\033[K"
            )
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
