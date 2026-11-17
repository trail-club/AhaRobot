"""The runner's own task status (/aha/tasks/status) for skills that must know
how earlier subgoals ended: a failed subgoal that was skipped leaves its mock
state in memory, which a bypassed (not selected) one also does."""

import json
import time

from aha_tasks.core.skill import Skill
from aha_tasks.runner import STATUS_TOPIC

STATUS_WAIT = 2.0  # s (wall clock) for the status of the running subgoal


class TaskStatusSkill(Skill):
    """The runner's own task status (/aha/tasks/status), for skills that look
    at how other subgoals ended."""

    def _subscribe_status(self):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self.status = None
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.runtime.node.create_subscription(
            String, STATUS_TOPIC, self._on_status, qos
        )

    def _on_status(self, message):
        self.status = message.data

    def _parsed(self):
        try:
            return json.loads(self.status) if getattr(self, "status", None) else None
        except ValueError:
            return None

    def latest_status(self, subgoal):
        """The task status published once `subgoal` (this skill's) started, or
        the last one after STATUS_WAIT (None without any).

        The runner publishes and subscribes on one node through DDS: the cached
        status may still be one from before the previous subgoal ended. The
        one with this subgoal as current has every earlier record final.
        """
        if getattr(self.runtime, "node", None) is not None:
            deadline = time.monotonic() + STATUS_WAIT
            while time.monotonic() < deadline:
                status = self._parsed()
                if status and status.get("current") == subgoal:
                    return status
                time.sleep(0.05)
        return self._parsed()


def unheard(keys, spec, status):
    """Memory keys among `keys` whose listen_command subgoal was skipped.

    A skipped (failed, then continued) subgoal leaves its mock command text in
    memory: that text was not heard and must not be planned. A bypassed one
    (not selected) keeps it, so that a run can start from the mock commands.
    """
    if not status:
        return set()
    outcome = {
        item.get("id"): item.get("status") for item in status.get("subgoals", [])
    }
    return {
        f"cmd{subgoal.params.get('command')}.text"
        for subgoal in spec.subgoals
        if subgoal.skill == "listen_command"
        and outcome.get(subgoal.id) == "skipped"
        and f"cmd{subgoal.params.get('command')}.text" in keys
    }


def planning_skipped(spec, status):
    """Whether the plan_commands subgoal failed and was skipped (its mock plan
    is in memory then). Not when it was not selected (bypassed)."""
    if not status:
        return False
    outcome = {
        item.get("id"): item.get("status") for item in status.get("subgoals", [])
    }
    return any(
        outcome.get(subgoal.id) == "skipped"
        for subgoal in spec.subgoals
        if subgoal.skill == "plan_commands"
    )
