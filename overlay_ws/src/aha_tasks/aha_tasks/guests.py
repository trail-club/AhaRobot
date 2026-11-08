"""Simulated guests of the Human Robot Interaction Challenge, without ROS or Gazebo.

A guest follows the subgoals of the task (rulebook 5.1): is hidden until it
comes to the door when the doorbell would ring, stays outside until the robot
starts the conversation, paces during it, follows the robot in and goes to the
seat it offers. The phase comes from the statuses of all subgoals, so a mock
subgoal that passes between two status messages still counts.
"""

import json
import math
from pathlib import Path

AWAY, ARRIVE, TALK, FOLLOW, SEAT = range(5)
# The start of subgoal <guest>.<name> begins the phase.
TRIGGERS = {SEAT: "seat", FOLLOW: "guide", TALK: "ask", ARRIVE: "doorbell"}
DOOR_TRIGGER = "open_door"
PENDING = "pending"

MESH = "model://person_standing/meshes/standing.dae"
MESH_POSE = "0 0 0.02 0.04 0 0"
MESH_PARTS = (
    "Body",
    "jeans01",
    "tshirt02",
    "mhair02",
    "Low-Poly",
    "Eyebrow001",
    "Teeth_Base",
)
COLOR_PARTS = {"shirt": "tshirt02", "trousers": "jeans01", "hair": "mhair02"}
MODEL_FRONT = -math.pi / 2  # the mesh faces -y
HIDDEN_Z = -5.0  # m; under the floor, because the 0.8 m walls hide nobody

MIN_STEP = 0.15  # m between remembered robot positions
SCENE_KEYS = {
    "world",
    "robot_model",
    "door",
    "walk_speed",
    "follow_gap",
    "pace",
    "guests",
}
GUEST_KEYS = {"away", "door", "threshold", "seat_via", "seat", "colors"}


def load_config(path):
    """The guest scene of a task; ValueError if it is incomplete."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != SCENE_KEYS:
        raise ValueError(f"{path}: needs exactly {sorted(SCENE_KEYS)}")
    for name, guest in config["guests"].items():
        if set(guest) != GUEST_KEYS:
            raise ValueError(f"{path}: {name} needs exactly {sorted(GUEST_KEYS)}")
        unknown = set(guest["colors"]) - set(COLOR_PARTS)
        if unknown:
            raise ValueError(f"{path}: {name} has unknown colors {sorted(unknown)}")
    return config


def statuses_of(status):
    """Subgoal id -> status from a /aha/tasks/status message."""
    return {subgoal["id"]: subgoal["status"] for subgoal in status["subgoals"]}


def reached(statuses, subgoal):
    return statuses.get(subgoal, PENDING) != PENDING


def guest_phase(name, statuses):
    for phase, trigger in TRIGGERS.items():
        if reached(statuses, f"{name}.{trigger}"):
            return phase
    return AWAY


def door_open(guests, statuses):
    return any(reached(statuses, f"{name}.{DOOR_TRIGGER}") for name in guests)


def guest_sdf(name, position, colors):
    """A static model without collision, so it cannot push the robot.

    The lidar and the cameras still see it. Parts with a color replace the
    texture of the mesh, which tells the guests apart.
    """
    parts = {COLOR_PARTS[part]: rgb for part, rgb in colors.items()}
    visuals = []
    for part in MESH_PARTS:
        material = ""
        if part in parts:
            color = " ".join(f"{c:.3f}" for c in parts[part]) + " 1"
            material = (
                f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse>"
                "<specular>0.05 0.05 0.05 1</specular></material>"
            )
        visuals.append(
            f'<visual name="{part}"><pose>{MESH_POSE}</pose><geometry><mesh>'
            f"<uri>{MESH}</uri><submesh><name>v2_{part}</name></submesh>"
            f"</mesh></geometry>{material}</visual>"
        )
    x, y = position
    return (
        '<?xml version="1.0"?><sdf version="1.8">'
        f'<model name="{name}"><static>true</static><pose>{x} {y} {HIDDEN_Z} 0 0 0</pose>'
        f'<link name="link">{"".join(visuals)}</link></model></sdf>'
    )


class Guest:
    """Position and heading (map-like world frame) of one guest over time."""

    def __init__(self, name, config, scene):
        self.name = name
        self.config = config
        self.speed = scene["walk_speed"]
        self.gap = scene["follow_gap"]
        self.pace = scene["pace"]
        self.x, self.y = config["away"]
        self.yaw = 0.0
        self.phase = AWAY
        self.route = []
        self.clock = 0.0
        self.last_trail = None

    @property
    def z(self):
        return HIDDEN_Z if self.phase == AWAY else 0.0

    @property
    def model_yaw(self):
        return math.remainder(self.yaw - MODEL_FRONT, math.tau)

    def set_phase(self, phase):
        if phase == self.phase:
            return
        previous, self.phase = self.phase, phase
        self.clock, self.last_trail = 0.0, None
        door, threshold = self.config["door"], self.config["threshold"]
        if phase < previous or phase == SEAT and previous < FOLLOW:
            # Rewound or skipped in the task, so there is nothing to walk through.
            self.route = []
            if phase == AWAY:
                self._place(self.config["away"])
            elif phase == SEAT:
                self._place(self.config["seat"][:2])
                self.yaw = self.config["seat"][2]
            else:
                self._place(door)
            if phase == FOLLOW:
                self.route = [threshold]
        elif phase == ARRIVE:
            self.route = [door]
        elif phase == TALK:
            self.route = [door] if previous == AWAY else self.route
        elif phase == FOLLOW:
            self.route = (self.route or [door]) + [threshold]
        elif phase == SEAT:
            self.route = [*self.config["seat_via"], self.config["seat"][:2]]

    def step(self, dt, robot=None):
        """Advance by dt seconds; `robot` is its (x, y), or None if unknown."""
        self.clock += dt
        if self.phase == FOLLOW and robot is not None:
            self._remember(robot)
        near = robot is not None and math.dist((self.x, self.y), robot) <= self.gap
        if self.route and not (self.phase == FOLLOW and near):
            self._walk(self.speed * dt)
            if not self.route and self.phase == SEAT:
                self.yaw = self.config["seat"][2]
        elif self.phase == ARRIVE:
            self._face(self.config["threshold"])  # waits looking in at the door
        elif self.phase == TALK:
            self._pace()
            self._face(robot)
        elif self.phase == FOLLOW:
            self._face(robot)

    def _place(self, point):
        self.x, self.y = point

    def _face(self, target):
        if target is not None:
            self.yaw = math.atan2(target[1] - self.y, target[0] - self.x)

    def _remember(self, robot):
        """The guest walks where the robot went, which keeps it out of the walls."""
        if self.last_trail is None or math.dist(self.last_trail, robot) >= MIN_STEP:
            self.route.append(list(robot))
            self.last_trail = robot

    def _walk(self, distance):
        while self.route and distance > 0.0:
            x, y = self.route[0]
            left = math.hypot(x - self.x, y - self.y)
            if left <= distance:
                self._place((x, y))
                self.route.pop(0)
                distance -= left
                continue
            self.yaw = math.atan2(y - self.y, x - self.x)
            self.x += (x - self.x) * distance / left
            self.y += (y - self.y) * distance / left
            return

    def _pace(self):
        """Slightly left and right of the door, across the way in (rulebook 5.1)."""
        (door_x, door_y), (in_x, in_y) = self.config["door"], self.config["threshold"]
        length = math.hypot(in_x - door_x, in_y - door_y)
        offset = self.pace["amplitude"] * math.sin(
            math.tau * self.clock / self.pace["period"]
        )
        self.x = door_x - (in_y - door_y) / length * offset
        self.y = door_y + (in_x - door_x) / length * offset
