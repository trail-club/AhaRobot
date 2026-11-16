"""Simulated people for the GPSR follow and guide steps, without ROS or Gazebo.

A walker reacts to the first step of the plan with its `action`, by the status
of the plan slot (run_plan_step subgoal) that runs it:

- follow_person (the robot follows them): while the slot runs, walks its
  route and waits while the robot is more than `wait_gap` away, like a person
  who looks back; stands facing the robot at the end.
- guide_person (they follow the robot): while the slot runs, walks the
  robot's path `follow_gap` behind it, as the HIC guests do; stands facing the
  robot otherwise.

Before that step a walker stands at its start; when the step is rewound it
goes back there. A follow walker whose step is passed without running (not
selected, skipped) moves to the end of its route without walking.
"""

import json
import math
from pathlib import Path

from aha_tasks.guests import COLOR_PARTS, Walker

ACTIONS = ("follow_person", "guide_person")
SCENE_KEYS = {
    "world",
    "robot_model",
    "walk_speed",
    "follow_gap",
    "wait_gap",
    "walkers",
}
WALKER_KEYS = {"action", "start", "yaw", "route", "colors"}
WAIT, RUN, PASSED = "wait", "run", "passed"
PENDING, RUNNING = "pending", "running"
# Plan slots that ended without running the step.
NOT_RUN = {"bypassed", "skipped", "unsupported"}
RUN_PLAN_STEP = "run_plan_step"


def load_config(path):
    """The walkers of a scene file; ValueError if it is incomplete."""
    scene = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(scene) != SCENE_KEYS:
        raise ValueError(f"{path}: needs exactly {sorted(SCENE_KEYS)}")
    for key in ("walk_speed", "follow_gap", "wait_gap"):
        if not scene[key] > 0:
            raise ValueError(f"{path}: {key} must be positive")
    for name, walker in scene["walkers"].items():
        if set(walker) != WALKER_KEYS:
            raise ValueError(f"{path}: {name} needs exactly {sorted(WALKER_KEYS)}")
        if walker["action"] not in ACTIONS:
            raise ValueError(f"{path}: {name} action must be one of {ACTIONS}")
        if walker["action"] == "follow_person" and not walker["route"]:
            raise ValueError(f"{path}: {name} needs a route to walk")
        unknown = set(walker["colors"]) - set(COLOR_PARTS)
        if unknown:
            raise ValueError(f"{path}: {name} has unknown colors {sorted(unknown)}")
    return scene


def step_slot(status, action):
    """The plan slot that runs the first step with `action`, or None.

    Slots are the run_plan_step subgoals in order: the i-th runs plan[i].
    """
    plan = status.get("memory", {}).get("plan")
    if not isinstance(plan, list):
        return None
    slots = [s for s in status["subgoals"] if s.get("skill") == RUN_PLAN_STEP]
    for slot, step in zip(slots, plan):
        if isinstance(step, dict) and step.get("action") == action:
            return slot
    return None


def step_status(status, action):
    """Status of the plan slot that runs the first step with `action`, or None."""
    slot = step_slot(status, action)
    return None if slot is None else slot["status"]


def step_attempts(status, action):
    """Attempts of that slot (counts retries), or None."""
    slot = step_slot(status, action)
    return None if slot is None else slot.get("attempts")


def stage_of(status, previous):
    """WAIT before the step, RUN once it ran, PASSED if it ended without running."""
    if status is None or status == PENDING:
        return WAIT
    if previous == RUN or status not in NOT_RUN:
        return RUN
    return PASSED


class SimWalker(Walker):
    """A person who walks a route or behind the robot during their plan step."""

    def __init__(self, name, config, scene):
        super().__init__(
            name,
            config["start"],
            config["yaw"],
            scene["walk_speed"],
            scene["follow_gap"],
        )
        self.config = config
        self.action = config["action"]
        self.wait_gap = scene["wait_gap"]
        self.stage = WAIT
        self.running = False  # the slot of the step is running now
        self.status = None  # the slot's last status
        self.attempts = None  # the slot's attempts so far

    def set_status(self, status, attempts=None):
        """Status and attempts of the walker's plan slot (step_status,
        step_attempts)."""
        # Running again after it ended (a retry, which restores the robot):
        # start over, as after a rewind. The engine retries within
        # milliseconds, so a new attempt also counts when "failed" was missed.
        retried = (
            status == RUNNING
            and self.stage == RUN
            and (
                self.status not in (None, PENDING, RUNNING)
                or (
                    attempts is not None
                    and self.attempts is not None
                    and attempts > self.attempts
                )
            )
        )
        self.status = status
        if attempts is not None:
            self.attempts = attempts
        self.running = status == RUNNING
        stage = stage_of(status, self.stage)
        if retried:
            self.stage = WAIT
            self._place(self.config["start"])
            self.yaw = self.config["yaw"]
        elif stage == self.stage:
            return
        previous, self.stage = self.stage, stage
        self.route, self.last_trail = [], None
        if stage == WAIT:  # rewound
            self._place(self.config["start"])
            self.yaw = self.config["yaw"]
        elif self.action == "follow_person":
            if stage == RUN:
                self.route = [list(p) for p in self.config["route"]]
            elif previous == WAIT:
                # Where the walk would end, facing the way it went.
                before, end = ([self.config["start"]] + self.config["route"])[-2:]
                self._place(end)
                self._face_away(before)

    def _face_away(self, point):
        if math.dist(point, (self.x, self.y)) > 0.0:
            self.yaw = math.atan2(self.y - point[1], self.x - point[0])

    def step(self, dt, robot=None):
        """Advance by dt seconds; `robot` is its (x, y), or None if unknown."""
        if self.stage == WAIT:
            return
        near = robot is not None and math.dist((self.x, self.y), robot) <= self.gap
        if self.action == "guide_person":
            if self.running and robot is not None:
                self._remember(robot)
            if self.running and self.route and not near:
                self._walk(self.speed * dt)
            else:
                self._face(robot)
            return
        waiting = (
            robot is not None and math.dist((self.x, self.y), robot) > self.wait_gap
        )
        if self.running and self.route and not waiting:
            self._walk(self.speed * dt)
        else:
            self._face(robot)
