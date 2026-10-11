"""CPU scripted policies; custom adapters use module:factory."""

import importlib
import math

from .api import Action, Observation
from .scene import load_scene, scripted_distances


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
        scene = load_scene()
        self.settings = scene["scripted"]
        self.approach_distance, self.transfer_distance = scripted_distances(scene)

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
        settings = self.settings
        open_limit = settings["open_position"] - settings["open_tolerance"]
        closed_limit = settings["grasp_position"] + settings["closed_tolerance"]
        opened = (
            joints["joint_r7r"] >= open_limit and joints["joint_r7l"] <= -open_limit
        )
        # Apple contact can stop a finger before the commanded grasp position.
        closed = (
            joints["joint_r7r"] <= closed_limit and joints["joint_r7l"] >= -closed_limit
        )
        desired_height = settings["transport_lift"]
        opening = settings["open_position"]
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
            if distance >= self.approach_distance:
                advance("done" if self.task_id == "approach_apple" else "lower")
            else:
                velocity = min(
                    0.12, max(0.035, (self.approach_distance - distance) * 1.5)
                )
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
                advance("hold" if self.task_id == "pick_apple" else "transfer")
        elif self.phase == "hold":
            opening = settings["grasp_position"]
        elif self.phase == "transfer":
            opening = settings["grasp_position"]
            if distance >= self.transfer_distance:
                advance("place")
            else:
                velocity = min(
                    0.10, max(0.025, (self.transfer_distance - distance) * 1.5)
                )
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
            angular_velocity=angular if velocity else 0.0,
            joint_positions={
                "joint_head_pan": settings["head_pan"],
                "joint_head_tilt": settings["head_tilt"],
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
