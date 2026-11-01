"""CPU scripted policies; custom adapters use module:factory."""

import importlib
import math

from .api import Action, Observation
from .scene import approach_target, load_scene


class NoopPolicy:
    def reset(self, task_id: str, instruction: str):
        pass

    def act(self, observation: Observation) -> Action:
        return Action()


class ScriptedPolicy:
    """Known-layout diagnostic baseline using odometry and joint feedback."""

    def reset(self, task_id: str, instruction: str):
        if task_id not in tuple(
            f"{verb}_{obj}"
            for verb in ("approach", "pick", "place")
            for obj in ("apple", "object")
        ):
            raise ValueError(f"Scripted policy does not support {task_id}")
        self.task_id = task_id
        self.phase = "prepare"
        self.phase_time = None
        self.start_pose = None
        scene = load_scene()
        self.scene = scene
        self.settings = scene["scripted"]
        self.targets = [
            approach_target(scene, destination) for destination in (False, True)
        ]

    def _drive(self, pose, destination=False):
        # Transform the known world layout into the controller's odometry frame.
        spawn = self.scene["robot_spawn"]
        sx, sy, heading = self.start_pose
        rotation = heading - spawn["yaw"]
        c, s = math.cos(rotation), math.sin(rotation)
        tx, ty, yaw = self.targets[int(destination)]
        dx, dy = tx - spawn["x"], ty - spawn["y"]
        tx, ty = sx + c * dx - s * dy, sy + s * dx + c * dy
        x, y, current_yaw = pose
        distance = math.hypot(tx - x, ty - y)

        def wrap(angle):
            return math.atan2(math.sin(angle), math.cos(angle))

        if distance < 0.006:
            error = wrap(yaw + rotation - current_yaw)
            return 0.0, max(-0.15, min(0.15, 2 * error)), abs(error) < 0.008
        error = wrap(math.atan2(ty - y, tx - x) - current_yaw)
        direction = 1 if abs(error) < math.pi / 2 else -1
        if direction < 0:
            error = wrap(error + math.pi)
        velocity = (
            direction * min(0.10, max(0.012, distance * 0.8)) * max(0, math.cos(error))
        )
        return velocity, max(-0.25, min(0.25, 2 * error)), False

    def act(self, observation: Observation) -> Action:
        if self.start_pose is None:
            self.start_pose = observation.odometry
            self.phase_time = observation.sim_time
        age = observation.sim_time - self.phase_time
        joints = observation.joint_positions
        height = joints["joint_r1"]
        settings = self.settings
        open_limit = settings["open_position"] - settings["open_tolerance"]
        closed_limit = settings["open_position"] - 0.002
        opened = (
            joints["joint_r7r"] >= open_limit and joints["joint_r7l"] <= -open_limit
        )
        # Contact can stop the fingers well before their commanded positions.
        closed = (
            joints["joint_r7r"] <= closed_limit
            and joints["joint_r7l"] >= -closed_limit
            and all(
                abs(observation.joint_velocities.get(name, 0.0))
                < settings.get("finger_stall_velocity", 0.005)
                for name in ("joint_r7r", "joint_r7l")
            )
        )
        desired_height = settings["transport_lift"]
        opening = settings["open_position"]
        velocity = 0.0
        angular = 0.0

        def advance(phase):
            self.phase = phase
            self.phase_time = observation.sim_time

        if self.phase == "prepare":
            if abs(height - desired_height) < 0.008 and opened and age > 0.8:
                advance("approach")
        elif self.phase == "approach":
            velocity, angular, reached = self._drive(observation.odometry)
            if reached:
                velocity = angular = 0.0
                advance("done" if self.task_id.startswith("approach_") else "lower")
        elif self.phase == "lower":
            desired_height = settings["pick_lift"]
            if abs(height - desired_height) < 0.005 and age > 1.0:
                advance("close")
        elif self.phase == "close":
            desired_height = settings["pick_lift"]
            opening = settings["grasp_position"]
            if closed and age > 1.0:
                advance("lift")
        elif self.phase == "lift":
            opening = settings["grasp_position"]
            if abs(height - desired_height) < 0.008 and age > 2.0:
                advance("hold" if self.task_id.startswith("pick_") else "transfer")
        elif self.phase == "hold":
            opening = settings["grasp_position"]
        elif self.phase == "transfer":
            opening = settings["grasp_position"]
            velocity, angular, reached = self._drive(observation.odometry, True)
            if reached:
                velocity = angular = 0.0
                advance("place")
        elif self.phase == "place":
            desired_height = settings["pick_lift"]
            opening = settings["grasp_position"]
            if abs(height - desired_height) < 0.005 and age > 1.5:
                advance("release")
        elif self.phase == "release":
            desired_height = settings["pick_lift"]
            if opened and age > 1.0:
                advance("done")
        return Action(
            linear_velocity=velocity,
            angular_velocity=angular,
            joint_positions={
                "joint_head_pan": settings["head_pan"],
                "joint_head_tilt": settings["head_tilt"],
                "joint_r1": desired_height,
                "joint_r7r": opening,
                "joint_r7l": -opening,
            },
        )


def policy_factory(name: str):
    if name == "codex":
        from .codex_policy import CodexPolicy

        return CodexPolicy
    if name == "ground_truth_ik":
        from .ground_truth_ik import GroundTruthIKPolicy

        return GroundTruthIKPolicy
    if name == "scripted":
        return ScriptedPolicy
    if name == "noop":
        return NoopPolicy
    if ":" not in name:
        raise ValueError(
            "policy must be scripted, noop, codex, ground_truth_ik, or module:factory"
        )
    module, factory = name.rsplit(":", 1)
    factory = getattr(importlib.import_module(module), factory)
    if not callable(factory):
        raise TypeError("policy factory must be callable")
    return factory


def load_policy(name: str):
    policy = policy_factory(name)()
    if not callable(getattr(policy, "reset", None)) or not callable(
        getattr(policy, "act", None)
    ):
        raise TypeError(
            "policy must implement reset(task_id, instruction) and act(observation)"
        )
    return policy
