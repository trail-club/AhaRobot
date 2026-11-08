"""Independent task scoring from fresh Gazebo state, using simulation time."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np

from .scene import approach_target, bounds_corners, load_scene


def make_evaluator(task, start_time, trace=None):
    if task.criterion == "laundry":
        from .laundry_evaluation import LaundryEvaluator

        return LaundryEvaluator(task, start_time, trace=trace)
    return Evaluator(task, start_time, trace=trace)


TASK_ALIASES = {
    f"{verb}_apple": f"{verb}_object" for verb in ("approach", "pick", "place")
}


def rotation_matrix(quaternion):
    x, y, z, w = quaternion
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return np.asarray(rotation)


@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    criterion: str
    timeout: float
    hold_time: float
    parameters: dict


def _laundry_parameters(entry, scene):
    """Scene-dependent geometry used by the laundry score sheet."""
    from .laundry_scene import table_frame

    (x, y, _), yaw = table_frame(scene)
    region = scene["laundry"]["regions"]["table"]
    entry.update(
        laundry_area=[list(p) for p in scene["laundry_area"]],
        table_center=[x, y],
        table_yaw=yaw,
        table_half=region["top_half"],
    )
    return entry


def load_tasks(path: Path, scene=None, laundry_scene=None) -> dict[str, Task]:
    scene = load_scene(path.with_name("scene.json")) if scene is None else scene
    tasks = {}
    for task_id, entry in json.loads(path.read_text()).items():
        entry = dict(entry)
        common = {
            key: entry.pop(key)
            for key in ("instruction", "criterion", "timeout", "hold_time")
        }
        if common["criterion"] not in ("approach", "pick", "place", "laundry"):
            raise ValueError(f"Unknown criterion for {task_id}")
        for key in ("timeout", "hold_time"):
            if not math.isfinite(common[key]) or common[key] <= 0:
                raise ValueError(f"{task_id}: {key} must be finite and positive")
        if common["criterion"] == "laundry":
            if laundry_scene is None:
                if scene.get("scene_type") == "laundry":
                    laundry_scene = scene
                else:
                    laundry_scene = load_scene(path.with_name(entry["scene"]))
            tasks[task_id] = Task(
                task_id=task_id,
                parameters=_laundry_parameters(entry, laundry_scene),
                **common,
            )
            continue
        if scene.get("scene_type") == "laundry":
            continue  # object tasks need a table-top scene
        common["instruction"] = common["instruction"].format(
            object=scene["object"]["label"]
        )
        # Named targets are resolved once from the same geometry as the world.
        if entry.get("robot_target") == "source_approach":
            entry["robot_target"] = approach_target(scene)
        if entry.get("object_target") == "destination_position":
            entry["object_target"] = [
                *scene["destination_table_position"][:2],
                scene["destination_position"][2],
            ]
        if "min_clearance" in entry:
            entry.update(
                object_size=scene["object"]["size"],
                object_points=scene["object"]["vertices"],
                source_surface_z=scene["source_table_position"][2]
                + scene["table_size"][2] / 2,
                destination_surface_z=scene["destination_table_position"][2]
                + scene["table_size"][2] / 2,
                require_support=True,
            )
        if entry.get("xy_tolerance") == "table_half_size":
            entry["xy_tolerance"] = [size / 2 for size in scene["table_size"][:2]]
        elif isinstance(entry.get("xy_tolerance"), (int, float)):
            entry["xy_tolerance"] = [entry["xy_tolerance"]] * 2
        tasks[task_id] = Task(task_id=task_id, parameters=entry, **common)
    for alias, canonical in TASK_ALIASES.items():
        if canonical in tasks and tasks[canonical].criterion != "laundry":
            from dataclasses import replace

            tasks[alias] = replace(tasks[canonical], task_id=alias)
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
    object_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    destination_contact: bool = False
    object_angular_speed: float = 0.0
    simultaneous_grasp: bool = True


class Evaluator:
    def __init__(self, task: Task, start_time: float, trace=None):
        self.task = task
        self.start_time = start_time
        self.stable_since = None
        self.previous_time = None
        self.was_lifted = False
        self.lift_since = None
        self.result = None
        self.metrics = {}
        self.trace = trace
        self.points = np.asarray(
            task.parameters.get(
                "object_points",
                bounds_corners(task.parameters.get("object_size", [0, 0, 0])),
            )
        )

    def finish(self, state=None, ended=False):
        """Close the episode early: only an already achieved success counts."""
        if self.result is None:
            self.result = "ended" if ended else "timeout"
        return self.result

    def update(self, state: WorldState) -> str | None:
        previous = self.previous_time
        result = self._update(state)
        if self.trace is not None and self.previous_time != previous:
            self.trace.write(
                json.dumps(
                    {"state": asdict(state), "metrics": self.metrics}, allow_nan=False
                )
                + "\n"
            )
        return result

    def _update(self, state: WorldState) -> str | None:
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
                *state.object_orientation,
                state.object_angular_speed,
            )
        ):
            self.stable_since = None
            self.lift_since = None
            return None
        if self.previous_time is not None and state.sim_time - self.previous_time > 0.5:
            self.stable_since = None
            self.lift_since = None
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
            "destination_contact": state.destination_contact,
            "object_angular_speed": state.object_angular_speed,
        }
        if elapsed >= self.task.timeout:
            self.result = "timeout"
            return self.result
        p = self.task.parameters
        points = self.points @ rotation_matrix(state.object_orientation).T + np.asarray(
            state.object_position
        )
        lower, upper = points.min(axis=0), points.max(axis=0)
        bottom = float(lower[2])
        if "min_clearance" in p:
            clearance = bottom - p["source_surface_z"]
            lifted = (
                state.finger_contacts == 2
                and state.simultaneous_grasp
                and clearance >= p["min_clearance"]
            )
            self.metrics["lift_clearance"] = clearance
        else:
            lifted = state.finger_contacts == 2 and state.object_position[2] >= p.get(
                "min_height", math.inf
            )
        if lifted:
            if self.lift_since is None:
                self.lift_since = state.sim_time
            self.was_lifted |= state.sim_time - self.lift_since >= p.get(
                "pick_hold_time", 0
            )
        else:
            self.lift_since = None
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
            x_tolerance, y_tolerance = p["xy_tolerance"]
            inside = (
                tx - x_tolerance <= lower[0]
                and upper[0] <= tx + x_tolerance
                and ty - y_tolerance <= lower[1]
                and upper[1] <= ty + y_tolerance
            )
            supported = state.destination_contact if p.get("require_support") else True
            height_error = (
                abs(bottom - p["destination_surface_z"])
                if "object_size" in p
                else abs(z - tz)
            )
            self.metrics.update(
                footprint_inside=bool(inside), destination_supported=supported
            )
            satisfied = (
                self.was_lifted
                and state.finger_contacts == 0
                and inside
                and supported
                and height_error <= p["z_tolerance"]
                and state.object_speed <= p["max_object_speed"]
                and state.object_angular_speed <= p.get("max_angular_speed", math.inf)
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
