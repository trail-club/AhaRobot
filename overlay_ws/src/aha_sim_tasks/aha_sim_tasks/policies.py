"""CPU scripted policies; custom adapters use module:factory."""

import importlib
import math

from .api import Action, Observation


class NoopPolicy:
    def reset(self, task_id: str, instruction: str):
        pass

    def act(self, observation: Observation) -> Action:
        return Action()


class ScriptedPolicy:
    """Feedback phases use odometry, joint positions, and simulation time only."""

    def reset(self, task_id: str, instruction: str):
        if task_id not in ("approach_apple", "pick_apple", "place_apple"):
            raise ValueError(f"Scripted policy does not support {task_id}")
        self.task_id = task_id
        self.phase = "prepare"
        self.phase_time = None
        self.start_pose = None

    def act(self, observation: Observation) -> Action:
        if self.start_pose is None:
            self.start_pose = observation.odometry
            self.phase_time = observation.sim_time
        x, y, yaw = observation.odometry
        sx, sy, heading = self.start_pose
        distance = math.cos(heading) * (x - sx) + math.sin(heading) * (y - sy)
        age = observation.sim_time - self.phase_time
        joints = observation.joint_positions
        height = joints["joint_r1"]
        opened = joints["joint_r7r"] >= 0.055 and joints["joint_r7l"] <= -0.055
        # Contact can stop a finger before the commanded 0.035 m displacement.
        closed = joints["joint_r7r"] <= 0.047 and joints["joint_r7l"] >= -0.047
        desired_height = 0.18
        opening = 0.06
        velocity = 0.0
        heading_error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        angular = max(-0.15, min(0.15, 1.5 * heading_error))

        def advance(phase):
            self.phase = phase
            self.phase_time = observation.sim_time

        if self.phase == "prepare":
            if abs(height - desired_height) < 0.008 and opened and age > 0.8:
                advance("approach")
        elif self.phase == "approach":
            if distance >= 0.49:
                advance("done" if self.task_id == "approach_apple" else "lower")
            else:
                velocity = min(0.12, max(0.035, (0.49 - distance) * 1.5))
        elif self.phase == "lower":
            desired_height = 0.05
            if abs(height - desired_height) < 0.005 and age > 1.0:
                advance("close")
        elif self.phase == "close":
            desired_height = 0.05
            opening = 0.035
            if closed and age > 1.0:
                advance("lift")
        elif self.phase == "lift":
            opening = 0.035
            if abs(height - desired_height) < 0.008 and age > 2.0:
                advance("hold" if self.task_id == "pick_apple" else "transfer")
        elif self.phase == "hold":
            opening = 0.035
        elif self.phase == "transfer":
            opening = 0.035
            if distance >= 0.89:
                advance("place")
            else:
                velocity = min(0.10, max(0.025, (0.89 - distance) * 1.5))
        elif self.phase == "place":
            desired_height = 0.05
            opening = 0.035
            if abs(height - desired_height) < 0.005 and age > 1.5:
                advance("release")
        elif self.phase == "release":
            desired_height = 0.05
            if opened and age > 1.0:
                advance("done")
        return Action(
            linear_velocity=velocity,
            angular_velocity=angular if velocity else 0.0,
            joint_positions={
                "joint_r1": desired_height,
                "joint_r7r": opening,
                "joint_r7l": -opening,
            },
        )


def policy_factory(name: str):
    if name == "scripted":
        return ScriptedPolicy
    if name == "noop":
        return NoopPolicy
    if ":" not in name:
        raise ValueError("policy must be scripted, noop, or module:factory")
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
