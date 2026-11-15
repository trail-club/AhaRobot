#!/usr/bin/env python3
"""Stress trials of the place action in Gazebo (scenarios in stress_scenarios.py).

Each scenario spawns furniture and clutter relative to the robot's pose at its
start, holds an object, sends PlaceObject and records where the object and
every spawned model ended. Trials may add models, keep the placed object for
the next trials, or change the robot or the scene while the goal runs
(Perturb). Reuses the Runner of run_trials.py; scenarios with the same launch
(world, spawn, depth noise) share one simulation. Appends to
results/stress.csv; --evidence DIR keeps per trial the planes, place markers,
planning scene, camera images and the place server log (plot_evidence.py
draws them).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import rclpy
import run_trials as base
from aha_test_msgs.action import PlaceObject
from gz.msgs10.empty_pb2 import Empty as GzEmpty
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from stress_scenarios import SCENARIOS, Prop, prop_pose, prop_sdf, prop_surfaces
from visualization_msgs.msg import Marker, MarkerArray

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation.scene_objects import remove_object
from aha_test_manipulation.sim_held_object import model_name, object_sdf
from aha_test_manipulation.sim_trials import (
    PLACE_TIMEOUT_S,
    PlaceServerBusy,
    cancel_goal,
)

HERE = Path(__file__).resolve().parent
SURFACE_TOL = 0.02  # m between the object bottom and a support top
EDGE_TOL = 0.005  # m outside a support rectangle still counts as on it
MOVE_TOL = 0.002  # m, a model that moved more was touched
TILT_TOL = 2.0  # deg, same for the tilt
ROBOT_TOL = 0.003  # m
ROBOT_YAW_TOL = 0.5  # deg
FLOOR_Z = 0.03
DRIVE_SPEED = 0.05  # m/s
TURN_SPEED = 0.1  # rad/s
MARKERS_TOPIC = "/aha/perception/place/markers"
HOLD_ATTEMPTS = 3
SPAWN_ATTEMPTS = 2
POLL_PERIOD = 0.5  # s (wall) between model pose checks while a goal runs
PLACED_OBJECT_ID = "aha_placed_object"  # place_server.py

FIELDS = [
    "run",
    "scenario",
    "trial",
    "noise",
    "arm",
    "grasp",
    "approach",
    "approach_used",
    "shape",
    "size",
    "mass",
    "near",
    "restrict_height",
    "perturb",
    "perturb_result",
    "status",
    "message",
    "phases",
    "candidates_tried",
    "surface_height",
    "target",
    "ended_on",
    "on_target",
    "upright",
    "tilt_deg",
    "err_xy",
    "err_z",
    "surface_err",
    "bottom_z",
    "released",
    "robot_moved",
    "robot_yaw_change_deg",
    "props_moved",
    "prop_moves",
    "contact_phase",
    "scene_obstacles",
    "wall_s",
    "sim_s",
    "outcome",
    "note",
    # Added after the first runs: at the end, so older rows stay valid.
    "after_perturb_s",
    "returned",
]


def _fmt(values, digits=3):
    return ",".join(f"{v:.{digits}f}" for v in values)


def _matrix(T):
    return [[round(float(v), 5) for v in r] for r in T]


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _log_line(line: str) -> bool:
    """Lines of the launch log kept as evidence."""
    if "place_server" in line or "plane_detection" in line:
        return True
    return "move_group" in line and ("WARN" in line or "ERROR" in line)


def marker_summary(msg: MarkerArray) -> dict:
    """Candidates [rank, x, y, z, clearance] and obstacle boxes [x, y, z, length,
    width, height, yaw] of one place marker message."""
    candidates, clearance, obstacles = {}, {}, []
    for m in msg.markers:
        p = m.pose.position
        if m.ns == "center":
            candidates[m.id] = [p.x, p.y, p.z]
        elif m.ns == "clearance":
            clearance[m.id] = m.scale.x / 2
        elif m.ns == "obstacle":
            obstacles.append(
                [
                    p.x,
                    p.y,
                    p.z,
                    m.scale.x,
                    m.scale.y,
                    m.scale.z,
                    _yaw(m.pose.orientation),
                ]
            )
    frame = next(
        (m.header.frame_id for m in msg.markers if m.action != Marker.DELETEALL), ""
    )
    return {
        "frame": frame,
        "candidates": [
            [k, *candidates[k], clearance.get(k, 0.0)] for k in sorted(candidates)
        ],
        "obstacles": obstacles,
    }


def outcome(success, released, on_target, upright, contact) -> str:
    if success:
        name = "ok" if on_target and upright else "wrong_place"
    else:
        name = "released" if released else "fail"
    return name + ("+contact" if contact else "")


class StressRunner(base.Runner):
    def __init__(self, settle, timeout, evidence=None, log=None):
        super().__init__(settle, timeout)
        self.evidence = evidence
        self.log = log
        self.planes_msg = None
        self.markers = []
        self.phases = []
        self.images = {}
        self.tracked = {}  # spawned model name -> Prop
        self.trial_spawned = []
        self.spawned_poses = {}  # models spawned while a goal runs -> first pose
        self.kept = 0
        # Advertise the weld topics now: the first publish waits for subscribers.
        for arm in ("right", "left"):
            for action in ("attach", "detach"):
                topic = f"/aha/sim/held_object/{arm}/{action}"
                self.gz.publishers[topic] = self.gz.node.advertise(topic, GzEmpty)
        time.sleep(1.0)
        self.node.create_subscription(
            MarkerArray, MARKERS_TOPIC, self.markers.append, 10
        )
        self.cmd_vel = self.node.create_publisher(
            TwistStamped, "/diff_drive_controller/cmd_vel", 10
        )
        self.apply_scene_client = self.node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        if evidence is not None:
            for key, topic in (
                ("color", "/camera/color/image_raw"),
                ("depth", "/camera/depth_registered/image_rect"),
            ):
                self.node.create_subscription(
                    Image,
                    topic,
                    lambda msg, key=key: self.images.__setitem__(key, msg),
                    qos_profile_sensor_data,
                )

    def on_planes(self, msg):
        super().on_planes(msg)
        self.planes_msg = msg

    # --- models ------------------------------------------------------------------------

    def scenario_frame(self) -> np.ndarray:
        T = self.gz.model_pose("aha_robot")
        return geo.transform(geo.rot_z(base.yaw_of(T)), [T[0, 3], T[1, 3], 0.0])

    def spawn_props(self, props, frame):
        present = self.gz.model_poses()
        for prop in props:
            if prop.name in present:
                # A model of that name (e.g. the world's shelf) goes; a link name
                # stays in pose/info and would hide the prop's pose.
                self.remove_models([prop.name])
                if prop.name in self.gz.model_poses():
                    raise RuntimeError(f"{prop.name} is also a link name in the world")
            # The create service answers late under load although the model came.
            for _ in range(SPAWN_ATTEMPTS):
                if self.gz.spawn(prop_sdf(prop), prop_pose(prop, frame)) or (
                    self.holder.spin_until(
                        lambda name=prop.name: name in self.gz.model_poses(), 5.0
                    )
                ):
                    break
            else:
                raise RuntimeError(f"could not spawn {prop.name}")
            self.tracked[prop.name] = prop
        names = [p.name for p in props]
        if names:
            if not self.holder.spin_until(
                lambda: all(n in self.gz.model_poses() for n in names), 30.0
            ):
                raise RuntimeError(f"{names} did not appear")
            self.holder.spin_until(lambda: False, 1.0)

    def remove_models(self, names):
        for name in names:
            self.gz.remove(name)
            self.tracked.pop(name, None)
        self.holder.spin_until(
            lambda: not set(names) & set(self.gz.model_poses()), 20.0
        )

    def keep_placed(self, arm, shape, size, pose):
        """Replace the released held model by a copy, so the next hold leaves it."""
        self.gz.release(arm)
        name = f"kept_{self.kept}"
        self.kept += 1
        if self.gz.spawn(object_sdf(name, shape, size, 0.1), pose):
            self.tracked[name] = Prop(name, shape, tuple(size), static=False, mass=0.1)

    # --- perturbations -----------------------------------------------------------------

    def drive(self, distance=0.0, angle=0.0) -> str:
        if distance:
            duration = abs(distance) / DRIVE_SPEED
            v, w = math.copysign(DRIVE_SPEED, distance), 0.0
        else:
            duration = abs(angle) / TURN_SPEED
            v, w = 0.0, math.copysign(TURN_SPEED, angle)
        clock = self.node.get_clock()
        end = clock.now().nanoseconds + int(duration * 1e9)
        msg = TwistStamped()
        msg.header.frame_id = "base_footprint"
        while clock.now().nanoseconds < end:
            msg.header.stamp = clock.now().to_msg()
            msg.twist.linear.x, msg.twist.angular.z = v, w
            self.cmd_vel.publish(msg)
            self.holder.spin_until(lambda: False, 0.05)
        msg.twist.linear.x = msg.twist.angular.z = 0.0
        for _ in range(5):
            msg.header.stamp = clock.now().to_msg()
            self.cmd_vel.publish(msg)
            self.holder.spin_until(lambda: False, 0.05)
        if distance:
            return f"drove {distance:+.2f} m"
        return f"turned {math.degrees(angle):+.1f} deg"

    def first_candidate(self):
        for msg in reversed(self.markers):
            summary = marker_summary(msg)
            if summary["candidates"]:
                return summary["candidates"][0][1:4]
        return None

    def perturb(self, trial, goal, handle, robot0, extra) -> str:
        p = trial.perturb
        if p.kind == "drive":
            return self.drive(distance=p.value)
        if p.kind == "turn":
            return self.drive(angle=math.radians(p.value))
        if p.kind == "head":
            self.aim(p.value)
            return "head moved"
        if p.kind == "cancel":
            response = self.holder.wait_future(handle.cancel_goal_async(), 10.0)
            return f"cancel requested ({len(response.goals_canceling)} canceling)"
        if p.kind == "second_goal":
            second = self.holder.wait_future(self.place.send_goal_async(goal), 30.0)
            if second.accepted:
                extra.append(second)
            return f"second goal {'accepted' if second.accepted else 'rejected'}"
        if p.kind == "spawn":
            c = self.first_candidate()
            if c is None:
                return "no candidate marker"
            frame = robot0 @ geo.transform(
                None, [c[0] + p.prop.x, c[1] + p.prop.y, 0.0]
            )
            prop = replace(p.prop, x=0.0, y=0.0, z=c[2])
            self.spawn_props([prop], frame)
            self.trial_spawned.append(prop.name)
            self.spawned_poses[prop.name] = self.gz.model_pose(prop.name)
            return (
                f"spawned {prop.name} at candidate ({c[0]:.2f}, {c[1]:.2f})"
                f" + ({p.prop.x:.2f}, {p.prop.y:.2f})"
            )
        raise ValueError(f"unknown perturbation {p.kind!r}")

    # --- one trial ---------------------------------------------------------------------

    def goal_of(self, trial, frame, robot0):
        goal = PlaceObject.Goal()
        goal.arm = trial.arm
        goal.object_size.x, goal.object_size.y, goal.object_size.z = trial.size
        goal.grasp = trial.grasp
        goal.approach = trial.approach
        goal.surface_labels = list(trial.labels)
        if trial.near:
            # Scenario frame -> base_footprint now.
            point = np.linalg.inv(robot0) @ frame @ np.array([*trial.near, 1.0])
            goal.near.header.frame_id = "base_footprint"
            goal.near.point.x, goal.near.point.y, goal.near.point.z = (
                float(v) for v in point[:3]
            )
        if trial.restrict is not None:
            goal.restrict_height = True
            goal.surface_height = trial.restrict
        return goal

    def run_trial(self, scenario, index, trial, frame) -> dict:
        perturb = trial.perturb
        row = {
            "scenario": scenario.name,
            "trial": index,
            "noise": scenario.noise,
            "arm": trial.arm,
            "grasp": trial.grasp,
            "approach": trial.approach,
            "shape": trial.shape,
            "size": "x".join(f"{v:.3f}" for v in trial.size),
            "mass": trial.mass,
            "near": _fmt(trial.near) if trial.near else "",
            "restrict_height": "" if trial.restrict is None else trial.restrict,
            "perturb": ""
            if perturb is None
            else f"{perturb.kind}@{perturb.phase}"
            + ("" if perturb.value is None else f":{perturb.value}"),
            "target": trial.target or scenario.target,
            "note": trial.note,
        }
        for arm in ("right", "left"):
            self.gz.release(arm)
        self.trial_spawned = [p.name for p in trial.props]
        try:
            self.spawn_props(trial.props, frame)
            return self.place_and_measure(scenario, index, trial, frame, row)
        finally:
            self.remove_models([n for n in self.trial_spawned if n in self.tracked])

    def clear_placed(self):
        """Remove the place server's aha_placed_object and obstacle boxes from
        the planning scene.

        After a failed retreat and return the arm can stay in contact with
        them, and every plan from that state fails until the next place goal."""
        names = [PLACED_OBJECT_ID]
        if self.scene_client.wait_for_service(timeout_sec=5.0):
            query = GetPlanningScene.Request()
            query.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
            response = self.holder.wait_future(
                self.scene_client.call_async(query), 10.0
            )
            names += [
                obj.id
                for obj in response.scene.world.collision_objects
                if obj.id.startswith("aha_place_obstacle_")
            ]
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.world.collision_objects = [
            remove_object(name, "base_footprint") for name in names
        ]
        if self.apply_scene_client.wait_for_service(timeout_sec=5.0):
            self.holder.wait_future(self.apply_scene_client.call_async(request), 10.0)
            print(f"removed {len(names)} objects from the planning scene", flush=True)

    def hold(self, trial):
        """Hold the trial's object; retry when it was welded away from the grasp.

        The object falls from the moment it is spawned until the attach message
        arrives, so a slow attach welds it below the fingers."""
        error = "welded off the grasp"
        for _ in range(HOLD_ATTEMPTS):
            try:
                T = self.holder.hold(
                    trial.arm, trial.shape, trial.size, trial.grasp, mass=trial.mass
                )
            except RuntimeError as exc:  # e.g. a late answer of the create service
                error = str(exc)
                if "could not move" in error:
                    self.clear_placed()
            else:
                self.holder.spin_until(lambda: False, 0.5)
                actual = self.gz.model_pose(model_name(trial.arm))
                if (
                    actual is not None
                    and np.linalg.norm(actual[:3, 3] - T[:3, 3]) < 0.01
                ):
                    return
            print(f"holding failed ({error}), holding again", flush=True)
            self.gz.release(trial.arm)
        raise RuntimeError(f"could not hold the object: {error}")

    def place_and_measure(self, scenario, index, trial, frame, row) -> dict:
        self.aim(trial.look_at or scenario.look_at)
        self.hold(trial)
        self.fresh_planes()
        before = self.gz.model_poses()
        robot0 = before["aha_robot"]
        goal = self.goal_of(trial, frame, robot0)
        planes_msg, images = self.planes_msg, dict(self.images)
        log_start = self.log.stat().st_size if self.log and self.log.exists() else 0
        states0 = self.gz.state_count(trial.arm)
        self.markers.clear()
        self.phases = []
        self.spawned_poses = {}
        extra, perturbed, markers_at_trigger, perturbed_at = [], None, None, None
        # place_pose is in base_footprint at the detection: the robot pose when
        # the server last reported "perceiving" (again after a replan).
        perceiving = {"count": 0, "seen": 0}
        reference = robot0

        def on_feedback(msg):
            if not self.phases or self.phases[-1] != msg.feedback.phase:
                self.phases.append(msg.feedback.phase)
                if msg.feedback.phase == "perceiving":
                    perceiving["count"] += 1

        wall0, sim0 = time.monotonic(), self.node.get_clock().now()
        handle = self.holder.wait_future(
            self.place.send_goal_async(goal, feedback_callback=on_feedback), 30.0
        )
        if not handle.accepted:
            raise PlaceServerBusy("place goal rejected")
        future = handle.get_result_async()
        deadline = time.monotonic() + PLACE_TIMEOUT_S
        first_moves, next_poll, timed_out = {}, 0.0, False
        while not future.done():
            if time.monotonic() > deadline:
                timed_out = True
                cancel_goal(self.holder, handle, future)
                break
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if perceiving["count"] > perceiving["seen"]:
                perceiving["seen"] = perceiving["count"]
                pose = self.gz.model_pose("aha_robot")
                if pose is not None and perceiving["seen"] > 1:
                    reference = pose
            p = trial.perturb
            if p is not None and perturbed is None and p.phase in self.phases:
                markers_at_trigger = len(self.markers)
                perturbed = self.perturb(trial, goal, handle, robot0, extra)
                perturbed_at = time.monotonic()
            if time.monotonic() >= next_poll:
                next_poll = time.monotonic() + POLL_PERIOD
                self.note_first_moves(before, first_moves, trial)
        self.note_first_moves(before, first_moves, trial)
        row["contact_phase"] = " ".join(f"{n}@{ph}" for n, ph in first_moves.items())
        result = future.result().result
        row["wall_s"] = round(time.monotonic() - wall0, 1)
        if perturbed_at is not None:
            row["after_perturb_s"] = round(time.monotonic() - perturbed_at, 1)
        sim_s = (self.node.get_clock().now() - sim0).nanoseconds * 1e-9
        row["sim_s"] = round(sim_s, 1)
        for second in extra:
            self.holder.wait_future(second.get_result_async(), 900.0)
        self.holder.spin_until(lambda: False, self.settle)
        after = self.gz.model_poses()
        for name, pose in self.spawned_poses.items():
            if pose is not None:
                before.setdefault(name, pose)
        scene_dump = self.scene_dump() if self.evidence else None
        row.update(
            perturb_result=perturbed or "",
            status=base.STATUS_NAMES.get(result.status, result.status),
            message=("timed out; " if timed_out else "") + result.message,
            phases=">".join(self.phases),
            candidates_tried=result.candidates_tried,
            returned=result.returned,
            approach_used=result.approach_used,
            surface_height=round(result.surface_height, 4) if result.success else "",
            scene_obstacles=len(self.scene_obstacle_boxes()),
        )
        released = "detached" in self.gz.states[trial.arm][states0:]
        robot1 = after["aha_robot"]
        moved = float(np.linalg.norm(robot1[:2, 3] - robot0[:2, 3]))
        yaw_change = math.degrees(
            math.remainder(base.yaw_of(robot1) - base.yaw_of(robot0), 2 * math.pi)
        )
        row.update(
            released=released,
            robot_moved=round(moved, 4),
            robot_yaw_change_deg=round(yaw_change, 2),
        )

        moves = {}
        for name in self.tracked:
            if name in before and name in after:
                d = float(np.linalg.norm(after[name][:3, 3] - before[name][:3, 3]))
                relative = after[name][:3, :3] @ before[name][:3, :3].T
                tilt = math.degrees(geo.tilt_from_vertical(relative))
                if d > MOVE_TOL or tilt > TILT_TOL:
                    moves[name] = (d, tilt)
        row["props_moved"] = round(max((d for d, _ in moves.values()), default=0.0), 4)
        row["prop_moves"] = " ".join(
            f"{n}:{d * 1000:.0f}mm/{t:.0f}deg" for n, (d, t) in moves.items()
        )

        held = after.get(model_name(trial.arm))
        ended, on_target, upright = "missing", False, False
        if held is not None:
            bottom = held[:3, 3] - held[:3, 2] * trial.size[2] / 2
            tilt = math.degrees(geo.tilt_from_vertical(held[:3, :3]))
            upright = tilt < base.UPRIGHT_DEG
            ended = self.surface_at(bottom, after)
            if ended is None:
                if bottom[2] < FLOOR_Z:
                    ended = "floor"
                else:
                    ended = "other" if released else "hand"
            on_target = ended == row["target"]
            row.update(tilt_deg=round(tilt, 2), bottom_z=round(float(bottom[2]), 4))
            if result.success:
                p = result.place_pose.pose.position
                target = reference @ np.array([p.x, p.y, p.z, 1.0])
                release_z = target[2] - base.RELEASE_HEIGHT
                row["err_xy"] = round(float(np.linalg.norm(bottom[:2] - target[:2])), 4)
                row["err_z"] = round(float(bottom[2] - release_z), 4)
                top = self.surface_top(row["target"], after)
                if top is not None:
                    row["surface_err"] = round(float(release_z - top), 4)
        driven = trial.perturb is not None and trial.perturb.kind in ("drive", "turn")
        pushed = not driven and (moved > ROBOT_TOL or abs(yaw_change) > ROBOT_YAW_TOL)
        row.update(
            ended_on=ended,
            on_target=on_target,
            upright=upright,
            outcome=outcome(
                result.success, released, on_target, upright, bool(moves) or pushed
            ),
        )
        if self.evidence is not None:
            self.write_evidence(
                f"{scenario.name}_{index:02d}",
                trial,
                row,
                result,
                robot0,
                before,
                after,
                planes_msg,
                images,
                scene_dump,
                log_start,
                markers_at_trigger,
            )
        if trial.keep and released and held is not None:
            self.keep_placed(trial.arm, trial.shape, trial.size, held)
        p = trial.perturb
        if p is not None and p.kind == "drive":
            self.drive(distance=-p.value)  # back for the next trials
        elif p is not None and p.kind == "turn":
            self.drive(angle=-math.radians(p.value))
        return row

    def note_first_moves(self, before, first_moves, trial):
        """Record the goal phase in which each model (and the robot) first moved."""
        poses = self.gz.model_poses()
        phase = self.phases[-1] if self.phases else "sent"
        for name in [*self.tracked, *self.spawned_poses]:
            start = before.get(name, self.spawned_poses.get(name))
            if name in first_moves or start is None or name not in poses:
                continue
            if np.linalg.norm(poses[name][:3, 3] - start[:3, 3]) > MOVE_TOL:
                first_moves[name] = phase
        driven = trial.perturb is not None and trial.perturb.kind in ("drive", "turn")
        robot = poses.get("aha_robot")
        if not driven and "aha_robot" not in first_moves and robot is not None:
            if np.linalg.norm(robot[:2, 3] - before["aha_robot"][:2, 3]) > ROBOT_TOL:
                first_moves["aha_robot"] = phase

    def surface_at(self, point, poses):
        """Name of the spawned support top under ``point`` (world), or None."""
        best = None
        for name, prop in self.tracked.items():
            if name not in poses:
                continue
            local = np.linalg.inv(poses[name]) @ np.array([*point[:3], 1.0])
            for surface, z, x0, x1, y0, y1 in prop_surfaces(prop):
                dz = abs(local[2] - z)
                inside = (
                    x0 - EDGE_TOL <= local[0] <= x1 + EDGE_TOL
                    and y0 - EDGE_TOL <= local[1] <= y1 + EDGE_TOL
                )
                if dz < SURFACE_TOL and inside and (best is None or dz < best[0]):
                    best = (dz, surface)
        return best[1] if best else None

    def surface_top(self, surface, poses):
        """World z of the center of a named support top, or None."""
        for name, prop in self.tracked.items():
            if name not in poses:
                continue
            for label, z, x0, x1, y0, y1 in prop_surfaces(prop):
                if label == surface:
                    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2, z, 1.0])
                    return float((poses[name] @ center)[2])
        return None

    # --- evidence ----------------------------------------------------------------------

    def scene_dump(self):
        """World collision objects of move_group's planning scene (base_footprint)."""
        if not self.scene_client.wait_for_service(timeout_sec=5.0):
            return []
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        response = self.holder.wait_future(self.scene_client.call_async(request), 10.0)
        return [
            {
                "id": obj.id,
                "frame": obj.header.frame_id,
                "pose": _matrix(base.object_matrix(obj)),
                "size": list(obj.primitives[0].dimensions),
            }
            for obj in response.scene.world.collision_objects
            if obj.primitives
        ]

    def write_evidence(
        self,
        name,
        trial,
        row,
        result,
        robot0,
        before,
        after,
        planes_msg,
        images,
        scene_dump,
        log_start,
        markers_at_trigger,
    ):
        out = self.evidence / name
        out.mkdir(parents=True, exist_ok=True)
        models = [*self.tracked, model_name(trial.arm), "aha_robot"]
        planes = [
            {
                "label": plane.label,
                "height": plane.height,
                "area": plane.area,
                "normal": [plane.normal.x, plane.normal.y, plane.normal.z],
                "hull": [[p.x, p.y, p.z] for p in plane.hull.points],
            }
            for plane in (planes_msg.planes if planes_msg else [])
        ]
        place_pose = None
        if result.success:
            p = result.place_pose.pose.position
            place_pose = [p.x, p.y, p.z, _yaw(result.place_pose.pose.orientation)]
        data = {
            "row": row,
            "trial": asdict(trial),
            "props": {n: asdict(p) for n, p in self.tracked.items()},
            "robot0": _matrix(robot0),
            "before": {n: _matrix(before[n]) for n in models if n in before},
            "after": {n: _matrix(after[n]) for n in models if n in after},
            "planes": planes,
            "markers": [marker_summary(m) for m in self.markers],
            "markers_at_trigger": markers_at_trigger,
            "scene": scene_dump,
            "place_pose": place_pose,
        }
        (out / "result.json").write_text(json.dumps(data, indent=1, default=str))
        if "color" in images:
            msg = images["color"]
            rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            cv2.imwrite(str(out / "color.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if "depth" in images:
            msg = images["depth"]
            depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
            mm = np.nan_to_num(depth * 1000.0, nan=0.0, posinf=0.0).clip(0, 65535)
            cv2.imwrite(str(out / "depth.png"), mm.astype(np.uint16))
        if self.log and self.log.exists():
            with self.log.open(errors="replace") as stream:
                stream.seek(log_start)
                lines = [line for line in stream if _log_line(line)]
            (out / "server.log").write_text("".join(lines))

    # --- one scenario ------------------------------------------------------------------

    def run_scenario(self, scenario, indices, writer, stream, run_label):
        frame = self.scenario_frame()
        yaw = math.degrees(base.yaw_of(frame))
        print(
            f"[{scenario.name}] robot at {_fmt(frame[:2, 3])} yaw {yaw:.1f}", flush=True
        )
        try:
            try:
                self.spawn_props(scenario.props, frame)
            except RuntimeError as error:
                print(f"[{scenario.name}] skipped: {error}", flush=True)
                return
            self.aim(scenario.look_at)
            self.fresh_planes(3, 120.0)
            for index, trial in enumerate(scenario.trials):
                if indices and index not in indices:
                    continue
                try:
                    row = self.run_trial(scenario, index, trial, frame)
                except PlaceServerBusy as error:
                    writer.writerow(
                        {
                            "scenario": scenario.name,
                            "trial": index,
                            "message": f"stopped: {error}",
                            "outcome": "error",
                            "run": run_label,
                        }
                    )
                    stream.flush()
                    raise  # the later scenarios of this sim are not valid either
                except RuntimeError as error:
                    row = {
                        "scenario": scenario.name,
                        "trial": index,
                        "message": f"error: {error}",
                        "outcome": "error",
                    }
                row["run"] = run_label
                writer.writerow(row)
                stream.flush()
                print(f"[{scenario.name}] {row}", flush=True)
        finally:
            for arm in ("right", "left"):
                self.gz.release(arm)
            self.remove_models(list(self.tracked))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS)
    )
    parser.add_argument("--trials", type=int, nargs="+", help="trial indices to run")
    parser.add_argument(
        "--launch", action="store_true", help="start a sim per launch group"
    )
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--csv", default="stress.csv", help="file name in --out")
    parser.add_argument("--evidence", type=Path, help="per-trial evidence directory")
    parser.add_argument(
        "--log", type=Path, help="launch log of the running sim (without --launch)"
    )
    parser.add_argument(
        "--settle", type=float, default=3.0, help="s (sim) before measuring"
    )
    parser.add_argument(
        "--run",
        default=time.strftime("%Y-%m-%d %H:%M"),
        help="label of the run column (default: start time)",
    )
    parser.add_argument("--list", action="store_true", help="list the scenarios")
    args = parser.parse_args()
    if args.list:
        for name in args.scenario:
            s = SCENARIOS[name]
            print(f"{name:16s} {s.noise:15s} {len(s.trials)} trials")
        return 0
    groups = {}
    for name in args.scenario:
        s = SCENARIOS[name]
        groups.setdefault((s.world, s.spawn, s.noise), []).append(s)
    if not args.launch and len(groups) > 1:
        parser.error("without --launch, pick scenarios of one world, spawn and noise")
    if args.launch and "GZ_PARTITION" not in os.environ:
        parser.error("set GZ_PARTITION (and ROS_DOMAIN_ID) for the launched sims")

    args.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    stream, writer = base.open_csv(args.out / args.csv, FIELDS)
    with stream:
        for (_, _, noise), scenarios in groups.items():
            process, log = None, args.log
            if args.launch:
                tmp = Path(os.environ.get("TMPDIR", "/tmp"))
                log = tmp / f"place_stress_{scenarios[0].name}.log"
                print(f"launching the sim, {noise} (log {log})", flush=True)
                process = base.launch_sim(scenarios[0], log, noise)
            runner = None
            try:
                runner = StressRunner(args.settle, 300.0, args.evidence, log)
                if not runner.wait_ready(600.0):
                    print("sim not ready", flush=True)
                    continue
                try:
                    runner.prepare_arms()
                except RuntimeError as error:
                    print(f"{error}", flush=True)
                    continue
                try:
                    for scenario in scenarios:
                        runner.run_scenario(
                            scenario, args.trials, writer, stream, args.run
                        )
                except PlaceServerBusy as error:
                    print(f"stopped: {error}", flush=True)
            finally:
                if runner is not None:
                    runner.node.destroy_node()
                if process is not None:
                    base.stop_sim(process)
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # The gz-transport Python binding sometimes crashes during interpreter teardown.
    os._exit(code)
