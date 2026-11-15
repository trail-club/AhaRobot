#!/usr/bin/env python3
"""Place trials in Gazebo: hold an object, send PlaceObject, measure the result.

Runs in the development container with the overlay sourced. Without
--launch, a simulation started with the scenario's spawn pose must be running
(see README.md); with --launch, each scenario starts its own headless
simulation on ROS_DOMAIN_ID / GZ_PARTITION of this process and stops it.
Writes results/trials.csv (appends; --out to change the directory; a file
with other columns is rewritten with the current ones first).

Per scenario: fold both arms and raise them to carry (Runner.prepare_arms),
then spawn the scenario's obstacles once (they are not reset between
trials). Per trial: release the previous object, aim the head at the
support, hold the object at carry (sim_held_object), wait for two fresh plane
detections, send PlaceObject, and after --settle read the object, robot and
furniture poses from Gazebo's pose/info and the obstacle boxes of that goal
from move_group's planning scene. place_pose (base_footprint at the goal
start) is converted to world with the robot's Gazebo pose.

Scenarios with furniture_config launch the sim with world_tf:=true and
perception's furniture config of the arena; their trials can name a
furniture level, a clearance and the expected status. Probe trials call
/aha/perception/get_place_candidates directly (no object, no motion) and
record its status and view_point (converted to world with the spawn pose,
compared with the config's level center). min_gap is the smallest
horizontal gap between the placed object's and the obstacles' footprints.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rclpy
from aha_msgs.msg import PlaneArray
from aha_msgs.srv import GetPlaceCandidates
from aha_test_msgs.action import PlaceObject
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

from aha_perception.furniture import load_config
from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation import scene_objects as scene
from aha_test_manipulation import sim_trials
from aha_test_manipulation.sim_held_object import (
    GzSim,
    SimHolder,
    model_name,
    object_sdf,
)
from aha_test_manipulation.sim_trials import (
    ARENA_FURNITURE,
    PLACE_TIMEOUT_S,
    PlaceServerBusy,
    cancel_goal,
    fold_arms,
    stop_sim,
    wait_controllers,
)

HERE = Path(__file__).resolve().parent
UPRIGHT_DEG = 10.0
ON_SURFACE_TOL = 0.02  # m between the object bottom and the support top
RELEASE_HEIGHT = 0.01  # place_server default (goals send 0)
OBSTACLE_MASS = 0.2  # kg
IN_SCENE_TOL = 0.01  # m, clutter center to a planning scene obstacle box
OBSTACLE_PADDING = 0.01  # place_server default, removed from the box sizes
PROBE_DETECTIONS = 5  # fresh plane detections before a probe
STATUS_NAMES = {
    getattr(PlaceObject.Result, name): name[len("STATUS_") :]
    for name in dir(PlaceObject.Result)
    if name.startswith("STATUS_")
}
SERVICE_STATUS_NAMES = {
    getattr(GetPlaceCandidates.Response, name): name[len("STATUS_") :]
    for name in dir(GetPlaceCandidates.Response)
    if name.startswith("STATUS_")
}
CONTROLLERS = (
    "right_arm_controller",
    "left_arm_controller",
    "lift_controller",
    "head_controller",
)

BOX_S = ("box", (0.06, 0.06, 0.12))
BOX_L = ("box", (0.08, 0.08, 0.15))
CYL = ("cylinder", (0.065, 0.065, 0.12))
TALL = ("box", (0.06, 0.06, 0.30))  # taller than the 0.28 m under a shelf board
# Around the default preferred point (base_footprint (0.45, -0.25) = world
# (-2.25, -1.50)) of the dining table spawn.
DINING_CLUTTER = (
    ("obstacle_0", "box", (0.05, 0.05, 0.10), (-2.25, -1.50)),
    ("obstacle_1", "box", (0.05, 0.05, 0.10), (-2.15, -1.63)),
    ("obstacle_2", "cylinder", (0.06, 0.06, 0.12), (-2.37, -1.44)),
)


@dataclass
class Trial:
    grasp: str
    approach: str  # PlaceObject.approach
    shape: str
    size: tuple
    arm: str = "right"
    near: tuple | None = None  # base_footprint
    labels: tuple = ()
    restrict: float | None = None  # PlaceObject.surface_height with restrict_height
    furniture: str = ""  # PlaceObject.furniture / furniture_level
    level: int = 0
    clearance: float = 0.0
    surface_z: float | None = None  # intended support top, else the scenario's
    expect: str = "OK"  # expected status
    probe: bool = False  # only call GetPlaceCandidates (no object, no motion)


@dataclass
class Scenario:
    name: str
    world: str
    spawn: tuple  # x, y, yaw (world)
    look_at: tuple  # base_footprint
    surface_z: float  # world z of the intended support top
    bounds: tuple  # world x_min, x_max, y_min, y_max of the support top
    furniture: tuple
    trials: list = field(default_factory=list)
    # (name, shape, size, (x, y) world) standing on the support, spawned once
    obstacles: tuple = ()
    furniture_config: bool = False  # launch with world_tf and the arena furniture


def _trials(spec):
    return [Trial(*t[:4], **(t[4] if len(t) > 4 else {})) for t in spec]


S, T, A = geo.SIDE, geo.TOP, "auto"
SCENARIOS = {
    "dining_table": Scenario(
        "dining_table",
        "rcjo2026",
        (-2.0, -1.05, -math.pi / 2),
        (0.55, -0.15, 0.8),
        0.80,
        (-2.6, -1.4, -3.15, -1.35),
        ("dining_table",),
        _trials(
            [
                (S, S, *BOX_S),
                (S, S, *BOX_S),
                (S, S, *BOX_L),
                (S, S, *BOX_L),
                (S, S, *CYL),
                (S, S, *CYL),
                (S, T, *BOX_S),
                (S, T, *BOX_L),
                (T, T, *BOX_S),
                (T, T, *BOX_L),
                (T, T, *CYL),
                (S, A, *BOX_L),
                (S, S, *BOX_S, {"arm": "left"}),
                (T, A, *CYL, {"arm": "left"}),
                # No surface at 0.45 m: NO_SURFACE.
                (S, S, *BOX_S, {"restrict": 0.45, "expect": "NO_SURFACE"}),
                (S, S, *BOX_L, {"restrict": 0.80}),
                (T, T, *BOX_S, {"restrict": 0.80}),
            ]
        ),
    ),
    # Same table with three boxes around the default preferred point
    # (base_footprint (0.45, -0.25) = world (-2.25, -1.50)).
    "dining_clutter": Scenario(
        "dining_clutter",
        "rcjo2026",
        (-2.0, -1.05, -math.pi / 2),
        (0.55, -0.15, 0.8),
        0.80,
        (-2.6, -1.4, -3.15, -1.35),
        ("dining_table",),
        _trials(
            [
                (S, S, *BOX_S),
                (S, S, *BOX_L),
                (S, T, *BOX_S),
                (S, T, *BOX_L),
                (T, T, *CYL),
                (S, A, *CYL),
                (T, T, *BOX_S),
                (T, T, *BOX_L),
            ]
        ),
        DINING_CLUTTER,
    ),
    "shelf_060": Scenario(
        "shelf_060",
        "rcjo2026",
        (2.95, 2.90, math.pi / 2),
        (0.55, -0.25, 0.61),
        0.61,
        (2.91, 3.49, 3.3, 3.7),
        ("shelf",),
        _trials(
            [
                (S, S, *BOX_S, {"near": (0.5, -0.25, 0.61)}),
                (S, S, *BOX_L, {"near": (0.5, -0.25, 0.61)}),
                (S, S, *CYL, {"near": (0.5, -0.25, 0.61)}),
                (S, A, *BOX_S, {"near": (0.5, -0.25, 0.61)}),
                (T, T, *BOX_S, {"near": (0.5, -0.25, 0.61)}),
                # Only the 0.61 m board: placed there or NO_SPACE, never on 0.91.
                (S, S, *BOX_S, {"near": (0.5, -0.25, 0.61), "restrict": 0.61}),
                (S, S, *BOX_L, {"near": (0.5, -0.25, 0.61), "restrict": 0.61}),
                (S, S, *CYL, {"near": (0.5, -0.25, 0.61), "restrict": 0.61}),
                (S, A, *BOX_L, {"restrict": 0.61}),
            ]
        ),
    ),
    # 0.1 m further back: the head sees deeper under the 0.91 board.
    "shelf_060_back": Scenario(
        "shelf_060_back",
        "rcjo2026",
        (2.95, 2.80, math.pi / 2),
        (0.65, -0.25, 0.61),
        0.61,
        (2.91, 3.49, 3.3, 3.7),
        ("shelf",),
        _trials(
            [
                (S, S, *BOX_S, {"near": (0.6, -0.25, 0.61)}),
                (S, S, *BOX_L, {"near": (0.6, -0.25, 0.61)}),
                (S, S, *CYL, {"near": (0.6, -0.25, 0.61)}),
                (S, A, *BOX_L, {"near": (0.6, -0.25, 0.61)}),
                (S, S, *BOX_S, {"near": (0.6, -0.25, 0.61), "restrict": 0.61}),
                (S, S, *BOX_L, {"near": (0.6, -0.25, 0.61), "restrict": 0.61}),
                (S, S, *CYL, {"near": (0.6, -0.25, 0.61), "restrict": 0.61}),
                (S, A, *BOX_L, {"restrict": 0.61}),
            ]
        ),
    ),
    "shelf_090": Scenario(
        "shelf_090",
        "rcjo2026",
        (2.95, 2.90, math.pi / 2),
        (0.55, -0.25, 0.91),
        0.91,
        (2.91, 3.49, 3.3, 3.7),
        ("shelf",),
        _trials(
            [
                (S, S, *BOX_S, {"near": (0.5, -0.25, 0.91)}),
                (S, S, *BOX_L, {"near": (0.5, -0.25, 0.91)}),
                (S, S, *CYL, {"near": (0.5, -0.25, 0.91)}),
                (S, A, *BOX_L, {"near": (0.5, -0.25, 0.91)}),
                (S, T, *BOX_S, {"near": (0.5, -0.25, 0.91)}),
            ]
        ),
    ),
    # Shelf levels by name (arena furniture config, shelf level 3 = 0.61 m,
    # 4 = 0.91 m); the dining table is behind the robot (not observed). The
    # shelf_060_back spawn: from 2.90 only the front of the 0.61 board is seen.
    "shelf_furniture": Scenario(
        "shelf_furniture",
        "rcjo2026",
        (2.95, 2.80, math.pi / 2),
        (0.65, -0.25, 0.61),
        0.61,
        (2.91, 3.49, 3.3, 3.7),
        ("shelf",),
        _trials(
            [
                (S, S, *BOX_S, {"probe": True, "furniture": "shelf", "level": 3}),
                (
                    S,
                    S,
                    *BOX_S,
                    {
                        "probe": True,
                        "furniture": "dining_table",
                        "level": 1,
                        "expect": "NO_SURFACE",
                    },
                ),
                (
                    S,
                    S,
                    *BOX_S,
                    {"probe": True, "furniture": "cabinet", "expect": "ERROR"},
                ),
                (S, S, *BOX_S, {"furniture": "shelf", "level": 3}),
                (S, A, *BOX_L, {"furniture": "shelf", "level": 3}),
                (S, S, *CYL, {"furniture": "shelf", "level": 3}),
                (S, S, *BOX_S, {"furniture": "shelf", "level": 4, "surface_z": 0.91}),
                (S, S, *TALL, {"furniture": "shelf", "level": 3, "expect": "NO_SPACE"}),
                (
                    S,
                    S,
                    *BOX_S,
                    {"furniture": "dining_table", "level": 1, "expect": "NO_SURFACE"},
                ),
            ]
        ),
        furniture_config=True,
    ),
    # The dining_clutter boxes with the table named, without and with 5 cm of
    # clearance (min_gap).
    "dining_furniture": Scenario(
        "dining_furniture",
        "rcjo2026",
        (-2.0, -1.05, -math.pi / 2),
        (0.55, -0.15, 0.8),
        0.80,
        (-2.6, -1.4, -3.15, -1.35),
        ("dining_table",),
        _trials(
            [
                (S, S, *BOX_S, {"furniture": "dining_table"}),
                (S, S, *BOX_S, {"furniture": "dining_table", "clearance": 0.05}),
                (S, S, *BOX_L, {"furniture": "dining_table"}),
                (S, S, *BOX_L, {"furniture": "dining_table", "clearance": 0.05}),
                (T, T, *CYL, {"furniture": "dining_table", "clearance": 0.05}),
                (S, T, *BOX_S, {"furniture": "dining_table", "clearance": 0.05}),
            ]
        ),
        DINING_CLUTTER,
        furniture_config=True,
    ),
}

FIELDS = [
    "run",
    "scenario",
    "trial",
    "arm",
    "grasp",
    "approach",
    "approach_used",
    "shape",
    "size",
    "restrict_height",
    "furniture",
    "clearance",
    "expect",
    "as_expected",
    "action_success",
    "status",
    "message",
    "candidates_tried",
    "surface_height",
    "surface_id",
    "view_point",
    "view_err",
    "min_gap",
    "on_surface",
    "upright",
    "success",
    "tilt_deg",
    "err_xy",
    "err_z",
    "surface_err",
    "bottom_z",
    "hold_robot_yaw_deg",
    "robot_moved",
    "robot_yaw_change_deg",
    "furniture_moved",
    "obstacles_moved",
    "obstacle_moves",
    "scene_obstacles",
    "clutter_in_scene",
    "clutter_boxes",
    "wall_s",
    "sim_s",
]


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def open_csv(path: Path, fields=FIELDS):
    return sim_trials.open_csv(path, fields)


def footprint(shape: str, size, T: np.ndarray) -> np.ndarray:
    """(N, 2) world footprint polygon (counterclockwise) of an upright model."""
    if shape == "cylinder":
        angles = np.linspace(0.0, 2 * math.pi, 32, endpoint=False)
        local = 0.5 * size[0] * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        hx, hy = size[0] / 2, size[1] / 2
        local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return local @ T[:2, :2].T + T[:2, 3]


def polygon_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex polygons (0 if they overlap)."""

    def separated(p, q):
        """Whether an edge of p (counterclockwise) has all of q outside."""
        for k in range(len(p)):
            edge = p[(k + 1) % len(p)] - p[k]
            outward = np.array([edge[1], -edge[0]])
            if ((q - p[k]) @ outward > 0.0).all():
                return True
        return False

    if not (separated(a, b) or separated(b, a)):
        return 0.0

    def point_segment(points, p0, p1):
        d = p1 - p0
        t = np.clip((points - p0) @ d / max(float(d @ d), 1e-12), 0.0, 1.0)
        return np.linalg.norm(points - (p0 + t[:, None] * d), axis=1).min()

    return float(
        min(
            min(point_segment(b, p, q) for p, q in zip(a, np.roll(a, -1, axis=0))),
            min(point_segment(a, p, q) for p, q in zip(b, np.roll(b, -1, axis=0))),
        )
    )


def object_matrix(obj) -> np.ndarray:
    """4x4 pose of a CollisionObject's first primitive in its header frame."""
    T = np.eye(4)
    for pose in (obj.pose, obj.primitive_poses[0]):
        p, q = pose.position, pose.orientation
        T = T @ geo.transform(
            geo.matrix_from_quaternion(q.x, q.y, q.z, q.w), [p.x, p.y, p.z]
        )
    return T


class Runner:
    def __init__(self, settle: float, timeout: float):
        self.node = rclpy.create_node(
            "place_trials", parameter_overrides=[Parameter("use_sim_time", value=True)]
        )
        self.gz = GzSim(timeout=timeout)
        self.holder = SimHolder(self.node, self.gz)
        self.place = ActionClient(self.node, PlaceObject, "/aha/manipulation/place")
        self.look_at = self.node.create_publisher(
            PointStamped, "/aha/perception/look_at", 10
        )
        self.plane_count = 0
        self.node.create_subscription(
            PlaneArray, "/aha/perception/planes", self.on_planes, 10
        )
        self.scene_client = self.node.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        self.candidates_client = self.node.create_client(
            GetPlaceCandidates, "/aha/perception/get_place_candidates"
        )
        self.settle = settle
        self._furniture = None

    @property
    def furniture(self):
        """The arena furniture config (world frame)."""
        if self._furniture is None:
            share = Path(get_package_share_directory("aha_perception"))
            self._furniture = load_config(share / ARENA_FURNITURE)
        return self._furniture

    def on_planes(self, _msg):
        self.plane_count += 1

    def prepare_arms(self):
        """Fold both arms below table height, then raise them to carry."""
        fold_arms(self.node, self.holder)

    def spawn_obstacles(self, scenario: Scenario):
        for name, shape, size, (x, y) in scenario.obstacles:
            self.gz.remove(name)
            pose = geo.transform(None, [x, y, scenario.surface_z + size[2] / 2 + 0.002])
            if not self.gz.spawn(object_sdf(name, shape, size, OBSTACLE_MASS), pose):
                raise RuntimeError(f"could not spawn {name}")

    def wait_ready(self, timeout: float) -> bool:
        ok = self.place.wait_for_server(timeout_sec=timeout)
        ok = ok and self.holder.move_client.wait_for_server(timeout_sec=timeout)
        ok = ok and wait_controllers(self.node, CONTROLLERS, timeout)
        ok = ok and self.holder.spin_until(lambda: self.plane_count > 0, timeout)
        # The action servers exist from configure; give the spawners time to activate.
        self.holder.spin_until(lambda: False, 10.0)
        return ok

    def aim(self, point):
        msg = PointStamped()
        msg.header.frame_id = "base_footprint"
        msg.point.x, msg.point.y, msg.point.z = point
        self.look_at.publish(msg)

    def scene_obstacle_boxes(self) -> list[scene.Box]:
        """Obstacle boxes in move_group's planning scene (base_footprint)."""
        if not self.scene_client.wait_for_service(timeout_sec=5.0):
            return []
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        response = self.holder.wait_future(self.scene_client.call_async(request), 10.0)
        return [
            scene.Box(
                scene.OBSTACLE, object_matrix(obj), tuple(obj.primitives[0].dimensions)
            )
            for obj in response.scene.world.collision_objects
            if obj.id.startswith(scene.OBSTACLE_PREFIX) and obj.primitives
        ]

    def fresh_planes(self, count: int = 2, timeout: float = 60.0) -> bool:
        start = self.plane_count
        return self.holder.spin_until(
            lambda: self.plane_count >= start + count, timeout
        )

    def probe(self, scenario: Scenario, trial: Trial, row: dict) -> dict:
        """GetPlaceCandidates for the trial's furniture level, without motion."""
        self.aim(scenario.look_at)
        # Surfaces get an id on their second observation: wait a few more.
        self.fresh_planes(PROBE_DETECTIONS)
        request = GetPlaceCandidates.Request()
        request.object_size.x, request.object_size.y, request.object_size.z = trial.size
        request.furniture = trial.furniture
        request.furniture_level = trial.level
        request.clearance = trial.clearance
        request.max_candidates = 5
        if not self.candidates_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("get_place_candidates not available")
        response = self.holder.wait_future(
            self.candidates_client.call_async(request), 30.0
        )
        status = SERVICE_STATUS_NAMES.get(response.status, response.status)
        row.update(
            action_success=response.success,
            status=status,
            message=response.message,
            candidates_tried=len(response.candidates),
            as_expected=status == trial.expect,
        )
        view = response.view_point
        if view.header.frame_id:
            p = np.array([view.point.x, view.point.y, view.point.z, 1.0])
            row["view_point"] = (
                f"{view.header.frame_id}:{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}"
            )
            furniture = self.furniture.get(trial.furniture)
            if view.header.frame_id == "odom" and furniture is not None:
                x, y, yaw = scenario.spawn
                world_odom = geo.transform(geo.rot_z(yaw), [x, y, 0.0])
                expected = furniture.view_center(trial.level)
                row["view_err"] = round(
                    float(np.linalg.norm((world_odom @ p)[:3] - expected)), 4
                )
        return row

    def run(self, scenario: Scenario, index: int, trial: Trial) -> dict:
        row = {
            "scenario": scenario.name,
            "trial": index,
            "arm": trial.arm,
            "grasp": trial.grasp,
            "approach": "probe" if trial.probe else trial.approach,
            "shape": trial.shape,
            "size": "x".join(f"{v:.3f}" for v in trial.size),
            "restrict_height": "" if trial.restrict is None else trial.restrict,
            "furniture": (
                f"{trial.furniture} {trial.level}".strip() if trial.furniture else ""
            ),
            "clearance": trial.clearance or "",
            "expect": trial.expect,
        }
        if trial.probe:
            return self.probe(scenario, trial, row)
        for arm in ("right", "left"):
            self.gz.release(arm)
        start = self.gz.model_pose("aha_robot")
        self.aim(scenario.look_at)
        self.holder.hold(trial.arm, trial.shape, trial.size, trial.grasp)
        self.fresh_planes()
        before = self.gz.model_poses()
        hold_yaw = math.remainder(
            yaw_of(before["aha_robot"]) - yaw_of(start), 2 * math.pi
        )
        row["hold_robot_yaw_deg"] = round(math.degrees(hold_yaw), 2)
        goal = PlaceObject.Goal()
        goal.arm = trial.arm
        goal.object_size.x, goal.object_size.y, goal.object_size.z = trial.size
        goal.grasp = trial.grasp
        goal.approach = trial.approach
        goal.surface_labels = list(trial.labels)
        if trial.near:
            goal.near.header.frame_id = "base_footprint"
            goal.near.point.x, goal.near.point.y, goal.near.point.z = trial.near
        if trial.restrict is not None:
            goal.restrict_height = True
            goal.surface_height = trial.restrict
        goal.furniture = trial.furniture
        goal.furniture_level = trial.level
        goal.clearance = trial.clearance
        wall0, sim0 = time.monotonic(), self.node.get_clock().now()
        handle = self.holder.wait_future(self.place.send_goal_async(goal), 30.0)
        if not handle.accepted:
            raise PlaceServerBusy("place goal rejected")
        future = handle.get_result_async()
        timed_out = not self.holder.spin_until(future.done, PLACE_TIMEOUT_S)
        if timed_out:
            cancel_goal(self.holder, handle, future)
        result = future.result().result
        row["wall_s"] = round(time.monotonic() - wall0, 1)
        row["sim_s"] = round((self.node.get_clock().now() - sim0).nanoseconds * 1e-9, 1)
        self.holder.spin_until(lambda: False, self.settle)
        after = self.gz.model_poses()
        boxes = self.scene_obstacle_boxes()

        row.update(
            action_success=result.success,
            status=STATUS_NAMES.get(result.status, result.status),
            message=("timed out; " if timed_out else "") + result.message,
            candidates_tried=result.candidates_tried,
            approach_used=result.approach_used,
            surface_height=round(result.surface_height, 4) if result.success else "",
            surface_id=result.surface_id if result.success else "",
        )
        robot0, robot1 = before["aha_robot"], after["aha_robot"]
        row["scene_obstacles"] = len(boxes)
        names = [o[0] for o in scenario.obstacles]
        row["obstacle_moves"] = " ".join(
            f"{name}:{np.linalg.norm(after[name][:3, 3] - before[name][:3, 3]):.4f}"
            for name in names
            if name in before and name in after
        )
        if names:
            world_boxes = [scene.Box(b.kind, robot0 @ b.pose, b.size) for b in boxes]
            matched = {}
            for name in names:
                if name not in before:
                    continue
                center = before[name][:3, 3]
                inside = [
                    b
                    for b in world_boxes
                    if scene.box_distance(b, center) <= IN_SCENE_TOL
                ]
                if inside:
                    matched[name] = min(
                        inside, key=lambda b: np.linalg.norm(b.pose[:2, 3] - center[:2])
                    )
            row["clutter_in_scene"] = f"{len(matched)}/{len(names)}"
            # Unpadded box size: footprint sides (long first) x height.
            row["clutter_boxes"] = " ".join(
                f"{name}:"
                + "x".join(
                    f"{v:.3f}"
                    for v in (
                        *sorted(np.asarray(b.size[:2]) - 2 * OBSTACLE_PADDING)[::-1],
                        b.size[2] - OBSTACLE_PADDING,
                    )
                )
                for name, b in matched.items()
            )
        row["robot_moved"] = round(
            float(np.linalg.norm(robot1[:2, 3] - robot0[:2, 3])), 4
        )
        yaw_change = math.remainder(yaw_of(robot1) - yaw_of(robot0), 2 * math.pi)
        row["robot_yaw_change_deg"] = round(math.degrees(yaw_change), 2)
        for column, names in (
            ("furniture_moved", scenario.furniture),
            ("obstacles_moved", [o[0] for o in scenario.obstacles]),
        ):
            moved = [
                float(np.linalg.norm(after[name][:3, 3] - before[name][:3, 3]))
                for name in names
                if name in before and name in after
            ]
            row[column] = round(max(moved), 4) if moved else ""

        status = row["status"]
        row["as_expected"] = status == trial.expect
        obj = after.get(model_name(trial.arm))
        if obj is None:
            row.update(on_surface=False, upright=False, success=False)
            row["as_expected"] = row["as_expected"] and trial.expect != "OK"
            return row
        surface_z = scenario.surface_z if trial.surface_z is None else trial.surface_z
        height = trial.size[2]
        bottom = obj[:3, 3] - obj[:3, 2] * height / 2
        tilt = math.degrees(geo.tilt_from_vertical(obj[:3, :3]))
        x0, x1, y0, y1 = scenario.bounds
        on_surface = (
            abs(bottom[2] - surface_z) < ON_SURFACE_TOL
            and x0 <= bottom[0] <= x1
            and y0 <= bottom[1] <= y1
        )
        row.update(
            tilt_deg=round(tilt, 2),
            bottom_z=round(float(bottom[2]), 4),
            on_surface=on_surface,
            upright=tilt < UPRIGHT_DEG,
        )
        if result.success:
            p = result.place_pose.pose.position
            # place_pose is the bottom center at release, in base_footprint at the goal start
            target = robot0 @ np.array([p.x, p.y, p.z, 1.0])
            row["err_xy"] = round(float(np.linalg.norm(bottom[:2] - target[:2])), 4)
            row["err_z"] = round(float(bottom[2] - (target[2] - RELEASE_HEIGHT)), 4)
            row["surface_err"] = round(float(target[2] - RELEASE_HEIGHT - surface_z), 4)
            gaps = [
                polygon_gap(
                    footprint(trial.shape, trial.size, obj),
                    footprint(shape, size, after[name]),
                )
                for name, shape, size, _ in scenario.obstacles
                if name in after
            ]
            if gaps:
                row["min_gap"] = round(min(gaps), 4)
        row["success"] = bool(result.success and on_surface and row["upright"])
        if trial.expect == "OK":
            row["as_expected"] = row["success"]
        return row


def launch_sim(scenario: Scenario, log: Path, depth_noise: str = "d435i_filtered"):
    return sim_trials.launch_sim(
        log,
        scenario.world,
        scenario.spawn,
        depth_noise,
        # stress.py scenarios have no furniture_config.
        furniture=getattr(scenario, "furniture_config", False),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS)
    )
    parser.add_argument("--trials", type=int, nargs="+", help="trial indices to run")
    parser.add_argument(
        "--launch", action="store_true", help="start a sim per scenario"
    )
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument(
        "--settle", type=float, default=3.0, help="s (sim) before measuring"
    )
    parser.add_argument(
        "--run",
        default=time.strftime("%Y-%m-%d %H:%M"),
        help="label of the run column (default: start time)",
    )
    args = parser.parse_args()
    if (
        not args.launch
        and len({(SCENARIOS[s].world, SCENARIOS[s].spawn) for s in args.scenario}) > 1
    ):
        parser.error(
            "without --launch, pick scenarios of the running sim's world and spawn pose"
        )
    if args.launch and "GZ_PARTITION" not in os.environ:
        parser.error("set GZ_PARTITION (and ROS_DOMAIN_ID) for the launched sims")

    args.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    stream, writer = open_csv(args.out / "trials.csv")
    stopped = False  # a goal still runs in the shared sim (without --launch)
    with stream:
        for name in args.scenario:
            if stopped:
                break
            scenario = SCENARIOS[name]
            process = None
            if args.launch:
                log = (
                    Path(os.environ.get("TMPDIR", "/tmp")) / f"place_trials_{name}.log"
                )
                print(f"[{name}] launching the sim (log {log})", flush=True)
                process = launch_sim(scenario, log)
            runner = None
            try:
                runner = Runner(args.settle, 300.0)
                if not runner.wait_ready(600.0):
                    print(f"[{name}] sim not ready", flush=True)
                    continue
                try:
                    runner.prepare_arms()
                    runner.spawn_obstacles(scenario)
                except RuntimeError as error:
                    print(f"[{name}] {error}", flush=True)
                    continue
                runner.aim(scenario.look_at)
                runner.fresh_planes(3, 120.0)
                for index, trial in enumerate(scenario.trials):
                    if args.trials and index not in args.trials:
                        continue
                    try:
                        row = runner.run(scenario, index, trial)
                    except PlaceServerBusy as error:
                        row = {
                            "scenario": name,
                            "trial": index,
                            "message": f"stopped: {error}",
                            "run": args.run,
                        }
                        writer.writerow(row)
                        stream.flush()
                        # A launched sim ends with its scenario; a shared one
                        # keeps the running goal for the later scenarios too.
                        stopped = not args.launch
                        print(f"[{name}] stopped: {error}", flush=True)
                        break
                    except RuntimeError as error:
                        row = {
                            "scenario": name,
                            "trial": index,
                            "message": f"error: {error}",
                        }
                    row["run"] = args.run
                    writer.writerow(row)
                    stream.flush()
                    print(f"[{name}] {row}", flush=True)
            finally:
                if runner is not None:
                    runner.node.destroy_node()
                if process is not None:
                    stop_sim(process)
    rclpy.try_shutdown()
    return 1 if stopped else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # The gz-transport Python binding sometimes crashes during interpreter teardown.
    os._exit(code)
