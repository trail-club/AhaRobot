"""Decide with the LLM where to put the held object and put it there (ROS).

``PlaceLlmRunner.run`` (the robot already holds the object and stands in front
of the furniture):

1. observe: with ``back_off``, move the base straight back from the furniture
   to where the board above no longer hides the back of the lowest listed
   level and look in as few directions as fit the camera
   (aha_test_orchestration.viewpoint); else look at each listed level from here
   (perception furniture config, level height >= MIN_LEVEL_HEIGHT; the level
   center moved towards the robot). After each look command wait until the
   head joints are still (at least ``look_settle_s``), then for
   ``detections_per_look`` surface and object messages stamped after that,
   and (space_hint) ask /aha/perception/get_place_candidates whether the held
   object fits on the levels of that look. Objects are accumulated over the
   whole observation by track id (mean position, most frequent label). Then
   wait until the summary (aha_test_orchestration.scene) is unchanged over
   ``stable_updates`` surface messages and, after a back-off, return to the
   start pose.
2. decide: build the variant's prompt (aha_test_orchestration.place_decision), force
   a function call and validate it (one retry with the error).
3. execute: send /aha/manipulation/place with the fields of the decision. If
   it fails with the object still in the hand (RETRY_STATUSES), the failure
   goes back to the LLM as the result of its call and it chooses again
   (``place_retries``; a failed place is not accepted twice).

The summary is kept (``memory``) and, after a successful place, updated with
the placed object, so ``run(..., reuse=True)`` skips step 1 for the next
object (the cabinet only changes when the robot changes it).

The caller's rclpy node must not be spun by another executor (this class
spins it while waiting).
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import rclpy
from aha_msgs.msg import DetectedObjectArray, SurfaceArray
from aha_msgs.srv import GetPlaceCandidates
from aha_test_msgs.action import PlaceObject
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PointStamped
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener

from aha_perception.furniture import load_config
from aha_perception.object_list import load_object_list
from aha_test_orchestration.base_motion import BaseMover, MoveOptions, pose_of
from aha_test_orchestration.llm import InvalidAnswer, ask_tool, tool_results
from aha_test_orchestration.place_decision import (
    VARIANTS,
    Decision,
    PlaceRequest,
    choice_key,
    describe,
    failure_feedback,
    tool_for,
)
from aha_test_orchestration.scene import (
    MIN_LEVEL_HEIGHT,
    HeldObject,
    ObjectInfo,
    SceneSummary,
    SurfaceInfo,
    build_summary,
    next_object_id,
    split_groups,
    with_placed,
)
from aha_test_orchestration.viewpoint import ViewOptions, observe_pose, view_points
from aha_test_ui.publisher import UiPublisher
from aha_test_ui.records import note

FURNITURE_CONFIG = "config/furniture/rcjo2026_arena.yaml"  # aha_perception share
OBJECTS_CONFIG = "config/objects/robocup_objects.yaml"  # aha_perception share
STAMP_HISTORY = 50  # message stamps kept for wait_fresh
HEAD_JOINTS = ("joint_head_pan", "joint_head_tilt")
HEAD_HISTORY = 400  # head joint samples kept for the settle check
HEAD_TIMEOUT_S = 30.0  # wall s, a head that never settles (no joint states)
CANCEL_TIMEOUT_S = 60.0  # wall s for a timed-out place goal to stop
SPACE = {
    GetPlaceCandidates.Response.STATUS_OK: "yes",
    GetPlaceCandidates.Response.STATUS_NO_SPACE: "no",
    GetPlaceCandidates.Response.STATUS_NO_SURFACE: "not seen",
}
PLACE_STATUS = {
    getattr(PlaceObject.Result, n): n[len("STATUS_") :]
    for n in dir(PlaceObject.Result)
    if n.startswith("STATUS_")
}
# PlaceObject failures after which the object is still in the hand and
# another place can be chosen.
RETRY_STATUSES = frozenset(
    {"NO_SURFACE", "NO_SPACE", "UNREACHABLE", "EXECUTION_FAILED"}
)


class PlaceServerStuck(RuntimeError):
    """A timed-out place goal ignored the cancel: the server rejects new goals."""


@dataclass
class RunnerOptions:
    furniture: str = "shelf"
    furniture_config: str = ""  # empty: aha_perception's arena config
    objects_config: str = ""  # empty: aha_perception's object list
    arm: str = "right"
    grasp: str = "side"
    approach: str = "auto"
    # After each look command: at least look_settle_s of the node clock (sim
    # time), then until both head joints moved less than head_still (rad)
    # over head_window_s.
    look_settle_s: float = 0.3
    head_still: float = 0.002
    head_window_s: float = 0.3
    detections_per_look: int = 2
    # Observe from where every level is seen whole (aha_test_orchestration.viewpoint)
    # and come back before placing. The base moves without an obstacle check
    # (aha_test_orchestration.base_motion; a stand-in until Nav2 runs).
    back_off: bool = False
    view: ViewOptions = field(default_factory=ViewOptions)
    view_speed: float = 0.15  # m/s for the back-off moves
    view_max_move: float = 1.2  # m, longer back-off moves are refused
    camera_frame: str = "camera_color_optical_frame"
    stable_updates: int = 3
    observe_timeout_s: float = 120.0
    space_hint: bool = True
    footprint_margin: float = 0.05  # m around the furniture footprint (variant B)
    place_timeout_s: float = 900.0
    retries: int = 1  # LLM answers again after an invalid answer
    place_retries: int = 2  # LLM chooses again after a failed PlaceObject goal
    # With a next_to object: first only spots this close to it (and closer to
    # it than to other categories on the level); 0 = any spot, group first.
    next_to_distance: float = 0.15
    # Out of reach from here: move the base to the server's suggested pose
    # (else, or if that fails, place at the best reachable spot).
    move_base: bool = False
    base_move: MoveOptions = field(default_factory=MoveOptions)


def matrix(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = (t.x, t.y, t.z)
    return T


def surfaces_from_msg(msg: SurfaceArray) -> list[SurfaceInfo]:
    out = []
    for s in msg.surfaces:
        p = s.plane
        out.append(
            SurfaceInfo(
                int(s.id),
                s.label,
                float(p.height),
                (p.centroid.x, p.centroid.y, p.centroid.z),
                tuple((q.x, q.y) for q in p.hull.points),
                s.furniture,
                int(s.furniture_level),
                bool(s.visible),
            )
        )
    return out


class ObjectAccumulator:
    """Detections by track id: mean position and most frequent label."""

    def __init__(self):
        self.tracks = {}

    def clear(self):
        self.tracks.clear()

    def add(self, track_id: int, label: str, position, size):
        track = self.tracks.setdefault(
            track_id, {"labels": Counter(), "sum": np.zeros(3), "n": 0, "size": size}
        )
        track["labels"][label] += 1
        track["sum"] += np.asarray(position, dtype=float)
        track["n"] += 1
        track["size"] = tuple(size)

    def objects(self) -> list[ObjectInfo]:
        return [
            ObjectInfo(
                track_id,
                t["labels"].most_common(1)[0][0],
                tuple(float(v) for v in t["sum"] / t["n"]),
                tuple(float(v) for v in t["size"]),
            )
            for track_id, t in sorted(self.tracks.items())
        ]


def goal_from(
    request: PlaceRequest,
    held: HeldObject,
    arm: str = "right",
    grasp: str = "side",
    approach: str = "auto",
    avoid=(),
    near_max_distance: float = 0.0,
    group=(),
) -> PlaceObject.Goal:
    """PlaceObject goal; ``avoid`` and ``group`` (points in the near frame)
    and ``near_max_distance`` only apply with a near point."""
    goal = PlaceObject.Goal()
    goal.arm = arm
    goal.object_size.x, goal.object_size.y, goal.object_size.z = (
        float(v) for v in held.size
    )
    goal.grasp = grasp
    goal.approach = approach
    goal.surface_ids = [int(i) for i in request.surface_ids]
    if request.near is not None:
        goal.near.header.frame_id = request.near_frame
        goal.near.point.x, goal.near.point.y, goal.near.point.z = (
            float(v) for v in request.near
        )
        goal.avoid = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in avoid]
        goal.group = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in group]
        goal.near_max_distance = float(near_max_distance)
    goal.restrict_height = request.restrict_height
    goal.surface_height = float(request.surface_height)
    goal.furniture = request.furniture
    goal.furniture_level = int(request.furniture_level)
    return goal


class PlaceLlmRunner:
    def __init__(self, node, client, options: RunnerOptions | None = None):
        self.node = node
        self.client = client
        self.options = options or RunnerOptions()
        share = Path(get_package_share_directory("aha_perception"))
        self.furniture_config = load_config(
            self.options.furniture_config or share / FURNITURE_CONFIG
        )
        self.furniture = self.furniture_config.get(self.options.furniture)
        if self.furniture is None:
            raise ValueError(f"furniture {self.options.furniture!r} not in the config")
        self.object_list = load_object_list(
            self.options.objects_config or share / OBJECTS_CONFIG
        )
        self.surfaces_msg = None
        self.surface_count = 0
        self.surface_stamps = deque(maxlen=STAMP_HISTORY)
        self.object_stamps = deque(maxlen=STAMP_HISTORY)
        self.accumulator = ObjectAccumulator()
        self.memory = None  # last summary, updated with the robot's placements
        node.create_subscription(
            SurfaceArray, "/aha/perception/surfaces", self.on_surfaces, 10
        )
        node.create_subscription(
            DetectedObjectArray, "/aha/perception/objects", self.on_objects, 10
        )
        self.look_at = node.create_publisher(
            PointStamped, "/aha/perception/look_at", 10
        )
        self.candidates = node.create_client(
            GetPlaceCandidates, "/aha/perception/get_place_candidates"
        )
        self.place = ActionClient(node, PlaceObject, "/aha/manipulation/place")
        # State, events and the LLM conversation for the robot monitor (aha_test_ui).
        self.ui = UiPublisher(node, "place_runner")
        self.tf_buffer = Buffer()
        # Own node spun in its own thread: spin_once handles one callback per
        # call, so /tf queued behind the other subscriptions and lookups
        # returned seconds old poses. A thread spinning the caller's node
        # itself would race with the caller's spin_once (and other helpers
        # spinning it, e.g. sim_held_object).
        self.tf_node = rclpy.create_node(
            f"{node.get_name()}_tf",
            parameter_overrides=[
                Parameter(
                    "use_sim_time", value=node.get_parameter("use_sim_time").value
                )
            ],
        )
        self.tf_listener = TransformListener(
            self.tf_buffer, self.tf_node, spin_thread=True
        )
        # Head joints on the same thread (high rate, kept off the caller's node).
        self.head = deque(maxlen=HEAD_HISTORY)  # (node-clock ns, pan, tilt)
        self.tf_node.create_subscription(
            JointState, "/joint_states", self.on_joint_states, qos_profile_sensor_data
        )
        self.base = BaseMover(
            node,
            self.transform,
            lambda: rclpy.spin_once(node, timeout_sec=0.0),
            self.options.base_move,
        )

    # --- ROS plumbing -----------------------------------------------------------------

    def spin_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return predicate()

    def wait_future(self, future, timeout: float):
        if not self.spin_until(future.done, timeout):
            raise RuntimeError("timed out")
        return future.result()

    def on_surfaces(self, msg):
        self.surfaces_msg = msg
        self.surface_count += 1
        self.surface_stamps.append(Time.from_msg(msg.header.stamp).nanoseconds)

    def on_objects(self, msg):
        self.object_stamps.append(Time.from_msg(msg.header.stamp).nanoseconds)
        if self.surfaces_msg is None:
            return  # positions are kept in the surfaces' frame
        frame = self.surfaces_msg.header.frame_id
        T = None
        if msg.header.frame_id != frame:
            T = self.transform(frame, msg.header.frame_id)
            if T is None:
                return
        for det in msg.objects:
            p = det.pose.pose.position
            position = np.array([p.x, p.y, p.z])
            if T is not None:
                position = T[:3, :3] @ position + T[:3, 3]
            self.accumulator.add(
                det.track_id,
                det.class_name,
                position,
                (det.size.x, det.size.y, det.size.z),
            )

    def on_joint_states(self, msg):
        joints = dict(zip(msg.name, msg.position))
        if all(name in joints for name in HEAD_JOINTS):
            now = self.tf_node.get_clock().now().nanoseconds
            self.head.append((now, *(joints[name] for name in HEAD_JOINTS)))

    def head_settled(self, after_ns: int) -> bool:
        """Whether the head joints moved less than ``head_still`` over the
        last ``head_window_s``, all of it after ``after_ns``."""
        o = self.options
        now = self.node.get_clock().now().nanoseconds
        window = int(o.head_window_s * 1e9)
        if now - window < after_ns:
            return False
        samples = np.array([s[1:] for s in list(self.head) if s[0] >= now - window])
        if len(samples) < 2:
            return False
        return bool(np.all(np.ptp(samples, axis=0) < o.head_still))

    def wait_head(self) -> int:
        """Wait after a look command until the head is still; the node time
        (ns) from which detections count."""
        after = self.node.get_clock().now().nanoseconds + int(
            self.options.look_settle_s * 1e9
        )
        self.spin_until(lambda: self.head_settled(after), HEAD_TIMEOUT_S)
        return max(after, self.node.get_clock().now().nanoseconds)

    def transform(self, target: str, source: str) -> np.ndarray | None:
        try:
            return matrix(self.tf_buffer.lookup_transform(target, source, Time()))
        except TransformException:
            return None

    def aim(self, point, frame: str):
        msg = PointStamped()
        msg.header.frame_id = frame
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in point)
        self.look_at.publish(msg)

    def wait_fresh(self, after_ns: int, count: int, timeout: float) -> bool:
        """Wait for ``count`` surface and object messages stamped after ``after_ns``."""
        return self.spin_until(
            lambda: sum(s >= after_ns for s in self.surface_stamps) >= count
            and sum(s >= after_ns for s in self.object_stamps) >= count,
            timeout,
        )

    def look_point(self, number: int) -> np.ndarray:
        """Level center moved a quarter of the depth towards the robot (config frame).

        Lower levels are seen under the next board from the front."""
        point = self.furniture.view_center(number)
        T = self.transform(self.furniture_config.frame, "base_footprint")
        if T is not None:
            direction = T[:2, 3] - point[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                point[:2] += direction / norm * self.furniture.depth / 4
        return point

    def level_space(self, held: HeldObject, level: int) -> str:
        if not self.candidates.wait_for_service(timeout_sec=5.0):
            return "unknown"
        request = GetPlaceCandidates.Request()
        request.object_size.x, request.object_size.y, request.object_size.z = (
            float(v) for v in held.size
        )
        request.furniture = self.furniture.name
        request.furniture_level = level
        request.max_candidates = 1
        try:
            response = self.wait_future(self.candidates.call_async(request), 30.0)
        except RuntimeError:
            return "unknown"
        return SPACE.get(response.status, "unknown")

    # --- steps ------------------------------------------------------------------------

    def summary(self, space: dict | None = None) -> SceneSummary | None:
        msg = self.surfaces_msg
        if msg is None:
            return None
        frame = msg.header.frame_id
        robot = (0.0, 0.0, 0.0)
        T = self.transform(frame, "base_footprint")
        if T is not None:
            robot = (T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0]))
        footprint = None
        T = self.transform(frame, self.furniture_config.frame)
        if T is not None:
            corners = self.furniture.corners(self.options.footprint_margin)
            footprint = corners @ T[:2, :2].T + T[:2, 3]
        return build_summary(
            surfaces_from_msg(msg),
            self.accumulator.objects(),
            self.furniture,
            frame,
            robot,
            footprint,
            space,
        )

    def back_off(self, levels) -> dict:
        """Move back to the observe pose of ``levels`` (viewpoint.observe_pose);
        report with ok, the odom start pose, the looks [(point, levels)] in
        the config frame and the move."""
        o = self.options
        frame = self.furniture_config.frame
        world = self.transform(frame, "base_footprint")
        camera = self.transform("base_footprint", o.camera_frame)
        odom = self.transform("odom", frame)
        start = self.base.pose()
        if world is None or camera is None or odom is None or start is None:
            return {"ok": False, "message": "no TF for the back-off"}
        camera_height = float((world @ camera)[2, 3])
        target, distance, depth = observe_pose(
            self.furniture,
            pose_of(world),
            float(camera[0, 3]),
            camera_height,
            levels,
            o.view,
        )
        x, y, _ = (odom @ np.array([target[0], target[1], 0.0, 1.0]))[:3]
        yaw = target[2] + math.atan2(odom[1, 0], odom[0, 0])
        self.ui.status("moving base", f"backing off to {distance:.2f} m from the front")
        move = self.base.move_to(x, y, yaw, o.view_max_move, o.view_speed)
        if not move["ok"]:
            self.ui.event(f"back-off failed: {move.get('message', '')}", "warn")
        return {
            "ok": move["ok"],
            "message": move.get("message", ""),
            "start": start,
            "camera_distance": round(distance, 3),
            "looks": view_points(
                self.furniture, distance, depth, camera_height, levels, o.view
            ),
            "move": move,
        }

    def observe(self, held: HeldObject) -> tuple[SceneSummary | None, dict]:
        o = self.options
        start = time.monotonic()
        self.accumulator.clear()
        space = {}
        frame = self.furniture_config.frame
        levels = [
            v.number for v in self.furniture.levels if v.height >= MIN_LEVEL_HEIGHT
        ]
        back_off = self.back_off(levels) if o.back_off else None
        if back_off is not None and back_off["ok"]:
            looks = back_off["looks"]
        else:
            # Each level from here (also after a refused or partial back-off).
            looks = [(self.look_point(n), (n,)) for n in levels]
        stale_looks = 0
        for k, (point, numbers) in enumerate(looks):
            levels_text = ", ".join(str(n) for n in numbers)
            self.ui.status(
                "observing", f"look {k + 1}/{len(looks)}: level {levels_text}"
            )
            self.aim(point, frame)
            # Detections of images taken while the head moves are not counted.
            if not self.wait_fresh(self.wait_head(), o.detections_per_look, 60.0):
                stale_looks += 1
                self.ui.event(f"look {k + 1}: no fresh detections", "warn")
            if o.space_hint:
                for number in numbers:
                    space[number] = self.level_space(held, number)
        stable, last = 0, None
        deadline = start + o.observe_timeout_s
        summary = None
        while time.monotonic() < deadline and stable < o.stable_updates:
            count = self.surface_count
            # Only a new SurfaceArray can confirm the summary.
            if not self.spin_until(lambda: self.surface_count > count, 10.0):
                continue
            summary = self.summary(space)
            if summary is None:
                continue
            signature = summary.signature()
            stable = stable + 1 if signature == last else 0
            last = signature
        timing = {}
        if back_off is not None:
            record = {k: v for k, v in back_off.items() if k != "looks"}
            record["looks"] = len(looks)
            if back_off.get("move", {}).get("rounds"):
                # Back to where the robot stood (it places from there).
                self.ui.status("moving base", "back to the place pose")
                record["return"] = self.base.move_to(
                    *back_off["start"], o.view_max_move, o.view_speed
                )
                summary = self.summary(space) or summary
            timing["back_off"] = record
        if summary is None:
            summary = self.summary(space)
        return summary, {
            **timing,
            "observe_s": round(time.monotonic() - start, 2),
            "stable": stable >= o.stable_updates,
            "stale_looks": stale_looks,
        }

    def decide(
        self,
        summary: SceneSummary,
        held: HeldObject,
        variant_name: str,
        messages: list | None = None,
        failed: tuple = (),
    ):
        """Ask for a place; ``messages`` continues a conversation (ending with
        the tool result of a failed place), ``failed`` are the decisions that
        failed (answering one of them again is invalid)."""
        variant = VARIANTS[variant_name]
        if messages is None:
            messages = variant.messages(summary, held, self.object_list.categories())
        failed_keys = {choice_key(d): d for d in failed}

        def validate(arguments):
            decision = variant.validate(arguments, summary)
            if choice_key(decision) in failed_keys:
                raise InvalidAnswer(f"placing on {describe(decision)} already failed")
            return decision

        return ask_tool(
            self.client,
            messages,
            tool_for(variant, summary),
            validate,
            self.options.retries,
        )

    def category_of(self, label: str):
        return self.object_list.category(label) or None

    def send(self, goal) -> dict:
        """Send a PlaceObject goal and wait for the result (report dict)."""
        if not self.place.wait_for_server(timeout_sec=30.0):
            return {"status": "NO_SERVER", "success": False}
        start = time.monotonic()
        handle = self.wait_future(self.place.send_goal_async(goal), 30.0)
        if not handle.accepted:
            return {"status": "REJECTED", "success": False}
        future = handle.get_result_async()
        timed_out = not self.spin_until(future.done, self.options.place_timeout_s)
        if timed_out:
            # The server rejects goals while one runs: stop it before going on.
            self.ui.event(
                f"place timed out after {self.options.place_timeout_s:g} s, canceling",
                "warn",
            )
            handle.cancel_goal_async()
            if not self.spin_until(future.done, CANCEL_TIMEOUT_S):
                raise PlaceServerStuck(
                    "the place goal timed out and ignored the cancel"
                )
        result = future.result().result
        p = result.place_pose.pose.position
        message = result.message
        if timed_out:
            message = f"timed out after {self.options.place_timeout_s:g} s; {message}"
        report = {
            "status": PLACE_STATUS.get(result.status, str(result.status)),
            "success": bool(result.success),
            "message": message,
            "surface_id": int(result.surface_id),
            "surface_height": round(float(result.surface_height), 4),
            "place_pose": [round(v, 4) for v in (p.x, p.y, p.z)],
            "place_frame": result.place_pose.header.frame_id,
            "verified": bool(result.verified),
            "place_s": round(time.monotonic() - start, 2),
        }
        base = result.suggested_base_pose
        if base.header.frame_id:
            q = base.pose.orientation
            report["suggested_base_pose"] = [
                round(base.pose.position.x, 4),
                round(base.pose.position.y, 4),
                round(2 * math.atan2(q.z, q.w), 4),
            ]
        return report

    def execute(
        self, decision: Decision, held: HeldObject, summary: SceneSummary | None = None
    ) -> dict:
        """PlaceObject for the decision. With a next_to object, first only
        spots within ``next_to_distance`` of it and away from the other
        categories of its level; out of reach, move the base to the suggested
        pose (``move_base``) and try again, else place at the best reachable
        spot. The report of the last goal has ``base_move`` and
        ``compromise`` and ``goals`` (the statuses of all goals)."""
        o = self.options
        group, avoid, near_max = (), (), 0.0
        if decision.next_to is not None and summary is not None:
            group, avoid = split_groups(
                summary,
                decision.next_to,
                self.category_of,
                decision.level,
                decision.surface_id,
            )
            near_max = o.next_to_distance
        goal = goal_from(
            decision.request, held, o.arm, o.grasp, o.approach, avoid, near_max, group
        )
        place = self.send(goal)
        statuses, base_move, compromise = [place["status"]], None, False
        if place["status"] == "UNREACHABLE" and near_max > 0.0:
            pose = place.get("suggested_base_pose")
            if o.move_base and pose is not None:
                self.ui.status("moving base", "to the pose the place server suggested")
                base_move = self.base.move_to(*pose)
                self.ui.event(
                    "base moved for the place"
                    if base_move["ok"]
                    else f"base move failed: {base_move['message']}",
                    "info" if base_move["ok"] else "warn",
                )
                if base_move["ok"]:
                    place = self.send(goal)
                    statuses.append(place["status"])
            if place["status"] == "UNREACHABLE":
                goal.near_max_distance = 0.0
                compromise = True
                place = self.send(goal)
                statuses.append(place["status"])
        place.update(goals=statuses, base_move=base_move, compromise=compromise)
        return place

    def placed_object(self, summary: SceneSummary, place: dict, held: HeldObject):
        """ObjectInfo of the held object at the result's place pose (object
        bottom center) in the summary frame, or None without the TF."""
        T = self.transform(summary.frame, place["place_frame"])
        if T is None:
            return None
        bottom = T[:3, :3] @ np.asarray(place["place_pose"]) + T[:3, 3]
        center = bottom + np.array([0.0, 0.0, held.size[2] / 2])
        return ObjectInfo(
            next_object_id(summary),
            held.label,
            tuple(float(v) for v in center),
            tuple(float(v) for v in held.size),
        )

    def run(
        self,
        held: HeldObject,
        variant_name: str,
        execute: bool = True,
        reuse: bool = False,
    ) -> dict:
        """Observe (or with ``reuse`` take the kept summary), decide and place;
        after a failed place with the object still in the hand the LLM gets
        the failure and chooses again (``place_retries``). Returns a
        JSON-serializable report; ``decision`` and ``place`` are the last ones,
        ``attempts`` all of them."""
        report = {"variant": variant_name, "held": asdict(held)}
        if reuse and self.memory is not None:
            summary = self.memory
            report.update(observe_s=0.0, stable=True, stale_looks=0, reused=True)
        else:
            self.ui.status("observing", f"{self.furniture.name} for {held.label}")
            summary, timing = self.observe(held)
            report.update(timing, reused=False)
            self.memory = summary
        if summary is None:
            report["error"] = "no surfaces"
            self.ui.status("failed", "no surfaces")
            self.ui.event(f"{held.label}: no surfaces observed", "error")
            return report
        report["summary"] = summary_record(summary)
        attempts, failed, calls = [], [], []
        messages, error = None, ""
        chat_id = f"{held.label}-{time.strftime('%H%M%S')}"
        title = f"{held.label} (variant {variant_name})"
        notes = {}
        for _ in range(self.options.place_retries + 1):
            self.ui.status("deciding", f"asking the LLM where to put {held.label}")
            answer = self.decide(summary, held, variant_name, messages, tuple(failed))
            calls.extend(answer.calls)
            messages, error = answer.messages, answer.error
            decision = answer.value
            attempt = {"arguments": answer.arguments, "llm_error": answer.error}
            attempts.append(attempt)
            answered = len(messages) - 1  # the assistant message of this answer
            if decision is None:
                notes.setdefault(answered, []).append(
                    note(f"no valid decision: {error}", "error")
                )
                self.ui.chat(chat_id, title, messages, notes)
                break
            self.ui.chat(chat_id, title, messages, notes)
            self.ui.event(f"{held.label}: {describe(decision)}")
            attempt["decision"] = asdict(decision)
            if not execute:
                break
            self.ui.status("placing", f"{held.label} on {describe(decision)}")
            place = self.execute(decision, held, summary)
            attempt["place"] = place
            text = place_text(place)
            notes.setdefault(answered, []).append(
                note(text, "ok" if place["success"] else "error")
            )
            self.ui.chat(chat_id, title, messages, notes)
            self.ui.event(
                f"{held.label}: {text}", "info" if place["success"] else "warn"
            )
            if place["success"]:
                placed = self.placed_object(summary, place, held)
                if placed is not None:
                    self.memory = with_placed(summary, placed, decision.level)
                break
            if place["status"] not in RETRY_STATUSES:
                break
            failed.append(decision)
            messages = messages + tool_results(
                messages,
                failure_feedback(decision, place["status"], place.get("message", "")),
            )
        last = attempts[-1]
        self.ui.chat(chat_id, title, messages, notes, done=True)
        if "place" in last and last["place"]["success"]:
            self.ui.status("done", f"placed {held.label}")
        elif "decision" in last and not execute:
            self.ui.status("decided", held.label)
        else:
            self.ui.status("failed", held.label)
        report.update(
            messages=messages,
            arguments=last["arguments"],
            llm_error=error,
            llm_calls=len(calls),
            llm_s=round(sum(c.latency_s for c in calls), 3),
            prompt_tokens=sum(c.prompt_tokens for c in calls),
            completion_tokens=sum(c.completion_tokens for c in calls),
            attempts=attempts,
        )
        if "decision" not in last:
            report["error"] = f"no valid decision: {error}"
            return report
        report["decision"] = last["decision"]
        if "place" in last:
            report["place"] = last["place"]
        return report


def place_text(place: dict) -> str:
    """One line of a PlaceObject report for the monitor."""
    goals = place.get("goals") or [place["status"]]
    text = f"place {' > '.join(goals)}"
    if place.get("compromise"):
        text += " (best reachable spot)"
    move = place.get("base_move")
    if move:
        text += (
            ", base moved"
            if move.get("ok")
            else f", base move failed: {move.get('message', '')}"
        )
    if not place["success"] and place.get("message"):
        text += f": {place['message']}"
    return text


def summary_record(summary: SceneSummary) -> dict:
    return {
        "frame": summary.frame,
        "objects": [asdict(o) for o in summary.objects],
        "support": {str(k): v for k, v in summary.support.items()},
        "levels": [asdict(v) for v in summary.levels],
        "by_height": {str(k): list(v) for k, v in summary.by_height.items()},
        "surfaces": [
            {k: v for k, v in asdict(s).items() if k != "hull"}
            for s in summary.surfaces
        ],
    }
