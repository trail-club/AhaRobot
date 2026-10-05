"""Independent task scoring from fresh Gazebo state, using simulation time."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .scene import approach_target, load_scene


@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    criterion: str
    timeout: float
    hold_time: float
    parameters: dict


def load_tasks(path: Path, scene=None) -> dict[str, Task]:
    scene = load_scene(path.with_name("scene.json")) if scene is None else scene
    tasks = {}
    for task_id, entry in json.loads(path.read_text()).items():
        entry = dict(entry)
        common = {
            key: entry.pop(key)
            for key in ("instruction", "criterion", "timeout", "hold_time")
        }
        if common["criterion"] not in ("approach", "pick", "place"):
            raise ValueError(f"Unknown criterion for {task_id}")
        for key in ("timeout", "hold_time"):
            if not math.isfinite(common[key]) or common[key] <= 0:
                raise ValueError(f"{task_id}: {key} must be finite and positive")
        # Named targets are resolved once from the same geometry as the world.
        if entry.get("robot_target") == "source_approach":
            entry["robot_target"] = approach_target(scene)
        if entry.get("object_target") == "destination_position":
            entry["object_target"] = list(scene["destination_position"])
        tasks[task_id] = Task(task_id=task_id, parameters=entry, **common)
    return tasks


@dataclass(frozen=True)
class WorldState:
    sim_time: float
    robot_pose: tuple[float, float, float]  # world x, y, yaw
    object_position: tuple[float, float, float]
    object_speed: float
    # Distinct fingers that touched during the sample window (0..2).
    finger_contacts: int
    robot_speed: float = 0.0


class Evaluator:
    def __init__(self, task: Task, start_time: float):
        self.task = task
        self.start_time = start_time
        self.stable_since = None
        self.previous_time = None
        self.was_lifted = False
        self.result = None
        self.metrics = {}

    def update(self, state: WorldState) -> str | None:
        if self.result is not None:
            return self.result
        if self.previous_time is not None and state.sim_time <= self.previous_time:
            return None  # A cached sample cannot advance the success hold time.
        if not all(
            math.isfinite(value)
            for value in (
                state.sim_time,
                *state.robot_pose,
                *state.object_position,
                state.object_speed,
                state.robot_speed,
            )
        ):
            self.stable_since = None
            return None
        if self.previous_time is not None and state.sim_time - self.previous_time > 0.5:
            self.stable_since = None
        self.previous_time = state.sim_time
        elapsed = state.sim_time - self.start_time
        self.metrics = {
            "elapsed_sim_seconds": elapsed,
            "robot_pose": state.robot_pose,
            "object_position": state.object_position,
            "object_speed": state.object_speed,
            "finger_contacts": state.finger_contacts,
            "grasped": state.finger_contacts == 2,
            "robot_speed": state.robot_speed,
            "was_lifted": self.was_lifted,
        }
        if elapsed >= self.task.timeout:
            self.result = "timeout"
            return self.result
        p = self.task.parameters
        lifted = state.finger_contacts == 2 and state.object_position[2] >= p.get(
            "min_height", math.inf
        )
        self.was_lifted = self.was_lifted or lifted
        if self.task.criterion == "approach":
            x, y, yaw = state.robot_pose
            tx, ty, tyaw = p["robot_target"]
            yaw_error = math.atan2(math.sin(yaw - tyaw), math.cos(yaw - tyaw))
            satisfied = (
                math.hypot(x - tx, y - ty) <= p["position_tolerance"]
                and abs(yaw_error) <= p["yaw_tolerance"]
                and state.robot_speed <= p["max_robot_speed"]
            )
        elif self.task.criterion == "pick":
            satisfied = lifted
        else:
            x, y, z = state.object_position
            tx, ty, tz = p["object_target"]
            satisfied = (
                self.was_lifted
                and state.finger_contacts == 0
                and abs(x - tx) <= p["xy_tolerance"]
                and abs(y - ty) <= p["xy_tolerance"]
                and abs(z - tz) <= p["z_tolerance"]
                and state.object_speed <= p["max_object_speed"]
            )
        if satisfied:
            if self.stable_since is None:
                self.stable_since = state.sim_time
            if state.sim_time - self.stable_since >= self.task.hold_time:
                self.result = "success"
        else:
            self.stable_since = None
        self.metrics["was_lifted"] = self.was_lifted
        return self.result
