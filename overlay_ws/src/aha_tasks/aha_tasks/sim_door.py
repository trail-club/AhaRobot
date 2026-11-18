"""Simulation only: the referee who opens the entrance door (task.launch.py start:=outside).

The door stays closed until the trigger subgoal (start.wait_door) runs its
skill, and opens `delay` s (simulation clock) later, inwards as an arena's
entrance door does (rulebook appendix). It stays open after the subgoal ends,
also when the subgoal is not selected or mocked, and while the runner brings
the robot back out for it (restoring robot.location). A new run or a rewind
to the trigger closes it again. Uses gz-transport set_pose like sim_guests.
"""

import argparse
import json
from pathlib import Path

CONFIG_KEYS = {"world", "door", "trigger", "location"}
DOOR_KEYS = {"model", "closed", "open"}
PENDING, RUNNING = "pending", "running"
PERIOD = 0.1  # s (simulation clock)


def load_config(path):
    """The entrance scene; ValueError if it is incomplete."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != CONFIG_KEYS or set(config["door"]) != DOOR_KEYS:
        raise ValueError(
            f"{path}: needs exactly {sorted(CONFIG_KEYS)}, door {sorted(DOOR_KEYS)}"
        )
    return config


class EntranceDoor:
    """Whether the door is open, from the runner's status over time."""

    def __init__(self, trigger, delay):
        self.trigger, self.delay = trigger, delay
        self.since = None  # when the trigger's skill started (sim seconds)

    def update(self, status, now):
        """True if the door is open now; status is a /aha/tasks/status dict or None."""
        state, phase = PENDING, None
        if status is not None:
            statuses = {s["id"]: s["status"] for s in status["subgoals"]}
            state = statuses.get(self.trigger, PENDING)
            if status.get("current") == self.trigger:
                phase = status.get("phase")
        if state != RUNNING:
            self.since = None
            return state != PENDING  # done, failed, skipped or mocked: opened
        if phase != "executing":
            self.since = None
            return phase == "restoring"  # the robot goes back out through the door
        if self.since is None:
            self.since = now
        return now - self.since >= self.delay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--delay", type=float, default=5.0, help="s after the trigger starts"
    )
    args, ros_args = parser.parse_known_args(argv)
    config = load_config(args.config)

    from gz.transport13 import Node as GzNode
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    from aha_tasks.guest_node import set_pose
    from aha_tasks.runner import STATUS_TOPIC

    gz = GzNode()
    rclpy.init(args=ros_args)
    node = Node("sim_door")
    log = node.get_logger()
    state = {"status": None}

    def on_status(message):
        state["status"] = json.loads(message.data)

    node.create_subscription(
        String,
        STATUS_TOPIC,
        on_status,
        QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        ),
    )
    door = EntranceDoor(config["trigger"], args.delay)
    model = config["door"]["model"]
    shown = None

    def tick():
        nonlocal shown
        now = node.get_clock().now().nanoseconds * 1e-9
        is_open = door.update(state["status"], now)
        if is_open != shown:
            x, y, z, yaw = config["door"]["open" if is_open else "closed"]
            if set_pose(gz, config["world"], model, x, y, z, yaw):
                shown = is_open
                log.info(f"Door {'open' if is_open else 'closed'}")

    node.create_timer(PERIOD, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
