#!/usr/bin/env python3
"""PlaceObject action server (/aha/manipulation/place).

Puts the object held in a gripper down on a surface from perception's
/aha/perception/get_place_candidates. Uses only move_group's ROS interfaces:
/get_planning_scene, /apply_planning_scene, /check_state_validity,
/compute_ik, /move_action, /compute_cartesian_path and /execute_trajectory.
The object-in-hand convention and the TCP poses come from
aha_test_manipulation.place_geometry.

Per goal: check the object against the gripper, aim the head
(/aha/perception/look_at) and wait for a detection after it settled, ask
perception for candidates (passing the goal's surface_ids, furniture,
furniture_level and clearance through) with the arms and the held object excluded
(aha_test_manipulation.self_filter), put the detection's planes and padded
obstacle boxes into the planning scene (aha_test_manipulation.place_scene, which
also keeps the robot's state valid against the boxes without removing any;
if the hand is at a real object, lift it first) and keep the candidates the
arm reaches (IK).

Per candidate, allowed approach and yaw: the descent column
(place_geometry.descent_column) clear of the perceived boxes known so far,
IK of the place and pre-place poses, joint-space plan to the pre-place pose
(retried), a fresh detection there to refresh the obstacle boxes and check
the spot again (things moved since), Cartesian approach, release
(optional release topic, then open the gripper slightly), Cartesian retreat,
the placed object into the planning scene, a joint-space move to
``return_state`` and (``verify_place``) a check: the head looks at the spot
and a fresh detection must have an obstacle box standing there, else the goal
aborts with STATUS_NOT_AT_SPOT (the wheel odometry does not see the base
being pushed by an arm in contact). Support boxes are grown by
``support_padding`` so that plans keep off the board edges. Before the
pre-place move, the descent and the release the base pose in ``odom_frame``
is compared with the one at the detection; if it
moved, the arm backs out and the goal starts over (``max_replans``). A cancel
stops the running MoveIt goal and leaves the arm where it is.

With a goal furniture and no ``near``, the head looks at perception's
view_point of the furniture level (from a first query) when it is within
``furniture_look_distance``, else at ``look_point`` at the level height (the
center of a large table is out of reach and the near part out of view). If
the level is then not in the detection (NO_SURFACE), the head turns to the
view point and perception is asked once more.

Candidate order: with the goal's ``near``, the spots closer to it than to
every ``avoid`` point (another group) first, each part nearest to ``near``
first (within the group those in near's row, ``next_to_row_tolerance``
along the base's x, before those in front of it); with
``near_max_distance`` only the group's spots within that distance, and if
none is reachable the goal fails with STATUS_UNREACHABLE and a
``suggested_base_pose`` (odom, current heading) that puts the best of them at
``base_spot`` (the caller may move the base and send the goal again). Without
``near``, per support
height the spots nearest to the arm side's preferred point (perception's
clearance order prefers far table centers). If no spot around ``near`` is
reachable, the preferred-point spots nearest to ``near`` are tried. A spot
found blocked at the pre-place pose (a fresh obstacle box inside
``place_geometry.descent_column``) is dropped together with the fresh
candidates within the object's diagonal of it; the fresh boxes are added to
the scene and the older ones stay (the arm or the head may hide them now).

The spots in the order they are tried, their state and the goal's near,
group and avoid points are published on /aha/manipulation/place/markers
(aha_test_manipulation.place_markers).

After a failure the arm (if it moved for this goal) backs out of the place
pose and returns to ``return_state`` with the object. Before any move to a
named state, the perceived boxes and the placed object that the arm would
collide with there are allowed (``PlaceScene.allow_at``). The scene objects
owned by the server (``aha_place_surface_*``, ``aha_place_wall_*``,
``aha_place_obstacle_*``, ``aha_held_object_<arm>``, ``aha_placed_object``)
are replaced at every goal; the placed object is kept in base_footprint, so
it is wrong once the base moves.
"""

import collections
import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from aha_msgs.msg import PlaneArray
from aha_msgs.srv import GetPlaceCandidates
from aha_test_msgs.action import PlaceObject
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation import place_markers as markers
from aha_test_manipulation import place_selection as selection
from aha_test_manipulation import scene_objects as scene
from aha_test_manipulation import self_filter
from aha_test_manipulation.moveit_requests import joint_goal, named_states
from aha_test_manipulation.place_scene import ARMS, PlaceScene
from aha_test_manipulation.place_selection import PlaceError
from aha_test_ui.publisher import UiPublisher

HEAD_JOINTS = ("joint_head_pan", "joint_head_tilt")
MAX_REFRESHES = 3  # fresh candidate lists after spots found blocked
DETECTION_RETRIES = 2  # queries again after an error (e.g. a too old detection)
WALL_CAP = 120.0  # s, any sim-time wait ends after this much wall time
MOTION_WALL_CAP = 900.0  # s, same for a MoveIt goal
HEAD_STILL = 0.002  # rad over HEAD_WINDOW (sim s)
HEAD_WINDOW = 0.3
HEAD_MIN_WAIT = 0.5  # sim s after the command before the head can count as settled
BASE_STILL = (0.002, math.radians(0.2))  # m, rad per BASE_SAMPLE
BASE_SAMPLE = 0.5  # sim s
STATE_TOLERANCE = 0.02  # rad / m, joints at a named state
Result = PlaceObject.Result


class Canceled(Exception):
    pass


class BaseMoved(Exception):
    def __init__(self, distance, angle):
        super().__init__(
            f"base moved {distance * 100:.1f} cm / {math.degrees(angle):.1f} deg "
            "since the detection"
        )


@dataclass
class Request:
    """A parsed goal."""

    goal: object
    arm: str
    grasp: str
    approaches: list
    size: tuple
    release_height: float
    surface_height: float | None
    clearance: float = 0.0
    near_max_distance: float = 0.0
    hand: np.ndarray | None = None  # T_tcp_object, the held object in the hand


@dataclass
class Perception:
    candidates: list
    status: int
    message: str
    planes: list
    obstacles: list
    stamp: object
    view_point: object = None  # PointStamped of the goal's furniture level, or None


@dataclass
class Blocked:
    """A spot found occupied at the pre-place pose; fresh candidates and
    obstacle boxes."""

    candidates: list
    obstacles: list


class PlaceServer(Node):
    def __init__(self):
        super().__init__("place_server")
        p = self.declare_parameter
        p("default_arm", "right")
        p("default_grasp", geo.SIDE)
        p("default_approach", "auto")
        p("planning_frame", "base_footprint")
        p("odom_frame", "odom")
        p("candidates_service", "/aha/perception/get_place_candidates")
        p("planes_topic", "/aha/perception/planes")
        p("look_at_topic", "/aha/perception/look_at")
        p("surface_labels", [""])  # [""] = perception default (table, shelf)
        p("max_candidates", 8)  # reachable candidates tried per detection
        p("candidate_pool", 30)  # candidates asked from perception
        p("object_padding", 0.03)  # added to the footprint (finger thickness)
        # Perception checks the footprint rectangle at the place yaws (see
        # candidates_request); finger_padding (m) is added across the fingers
        # (y): the fingers open release_margin (0.01) per side at the release.
        p("oriented_footprint", True)
        p("finger_padding", 0.02)
        # Without a goal near: x, |y| toward the arm side, per grasp (the top
        # grasp reaches about 0.45 m from the shoulder, the side grasp 0.75 m).
        p("preferred_point", [0.45, 0.25])
        p("preferred_point_top", [0.38, 0.2])
        p("reach_limits", [0.80, 0.55])  # side, top grasp: shoulder to spot
        # suggested_base_pose: x, |y| toward the arm side of the spot in the
        # base frame. Side: well out on the arm's side (approach about 30 deg
        # from the shoulder at y 0.13) so that the wrist, which sticks out on
        # the outer side of the hand, trails behind the object instead of
        # hitting neighbours and side panels; top: preferred_point_top.
        p("base_spot", [0.55, 0.45])
        p("base_spot_top", [0.38, 0.2])
        # With near: the group's spots at most this far from near along the
        # base's forward axis (beside it in its row) are tried before those
        # in front of it (0 = by distance only).
        p("next_to_row_tolerance", 0.04)
        # A spot is in near's group when its distance to near is below this
        # times the distance to every avoid point (0.6: 10 cm from near needs
        # 17 cm to the other groups; halfway spots were ambiguous).
        p("group_ratio", 0.6)
        # Log why each unreachable spot failed (IK again without collision
        # checking, then the contacts of that solution).
        p("explain_unreachable", True)
        p("aim_head", True)
        # Without a goal near: x, |y| toward the arm side. Farther than the
        # preferred point, so the view also covers the board above a shelf
        # board (its front face is not an obstacle box only if it is seen).
        p("look_point", [0.55, 0.15])
        p("look_height", 0.75)  # look point height without restrict_height
        # With a goal furniture and no near: look at the level center when it
        # is this close (m, horizontal), else look_point at the level height.
        p("furniture_look_distance", 0.9)
        p("head_settle_timeout", 4.0)  # sim s
        p("detection_timeout", 5.0)  # wall s, for a detection after the head settled
        # exclude_regions: growth of the arm link boxes and the held object box
        p("arm_padding", 0.02)
        p("held_padding", 0.02)
        p("late_check", True)  # fresh detection at the pre-place pose
        p("late_check_timeout", 5.0)  # wall s
        # Descent column growth (place_geometry.descent_column, with the
        # fingers): 0.005 + FINGER_WIDTH 0.015 = 0.02 per side across the
        # fingers, what perception keeps from a box at the checked yaw
        # (finger_padding / 2 + its candidate_box_padding).
        p("late_check_margin", 0.005)
        p("base_max_shift", 0.01)  # m since the detection
        p("base_max_turn", 1.0)  # deg
        p("base_settle_timeout", 5.0)  # sim s, wait for the base to stop
        p("max_replans", 2)
        p("release_height", 0.01)  # m above the surface when the goal has 0
        p("top_tilt", geo.DEFAULT_TOP_TILT)
        p("yaw_offsets", [0.0, 0.35, -0.35])  # rad, added to the approach direction
        # rad, tried after yaw_offsets, toward the arm's side (negative for
        # the right arm): swings the wrist behind the object (candidate_yaws).
        p("wrist_yaw_offsets", [0.61, 0.87])
        # place_geometry.MotionParams
        p("approach_distance", 0.10)
        p("lift_margin", 0.03)
        p("approach_height", 0.10)
        p("retreat_distance", 0.10)
        p("retreat_lift", 0.05)
        # scene_objects.SceneParams: plane boxes
        p("support_thickness", 0.03)
        p("surface_clearance", 0.005)
        p("support_padding", 0.03)
        p("wall_thickness", 0.02)
        # perception blocks the boxes grown by its candidate_box_padding (the
        # same 0.01), so a candidate's place pose stays clear of these boxes.
        p("obstacle_padding", 0.01)
        # Boxes touching the current state (place_scene.resolve_contacts):
        # cut clear by clip_margin, allowed below min_box_height; escape_lift
        # (0 = off) lifts the TCP first when the hand is at a real object.
        p("clip_margin", 0.01)
        p("escape_lift", 0.05)
        p("min_box_height", 0.02)
        p("plan_retries", 3)  # joint-space plans per pre-place pose
        p("planning_time", 3.0)
        p("velocity_scaling", 0.5)
        p("cartesian_velocity_scaling", 0.2)
        p("cartesian_step", 0.005)
        p("cartesian_min_fraction", 0.95)  # approach
        p("retreat_min_fraction", 0.5)
        p("ik_timeout", 0.1)  # wall s
        p("service_timeout", 10.0)  # wall s
        p("motion_timeout", 60.0)  # sim s per MoveIt goal
        p("gripper_open", geo.FINGER_MAX_OPENING)  # finger limit per side
        p("finger_clearance", 0.005)  # object fit check, per side and to the palm
        p("release_margin", 0.01)  # per side beyond the object at release
        p("release_topic", "")  # "{arm}" is replaced; empty = no release hook
        p("release_settle", 1.0)  # wall s after opening
        p("return_state", "carry")
        # After the return: look at the spot and require a perceived obstacle
        # box standing on the support within verify_margin of it.
        p("verify_place", True)
        p("verify_margin", 0.03)  # m, horizontal from the box
        p("verify_height_tolerance", 0.04)  # m, box bottom to the support height

        group = ReentrantCallbackGroup()
        self.busy = threading.Lock()
        self.cancel_event = threading.Event()
        self.active_motion = None  # client goal handle of the running MoveIt goal
        self.named_states = {}
        self.latest_planes = None
        self.plane_messages = collections.deque(maxlen=20)
        self.joints = {}
        self.head_history = collections.deque(maxlen=400)  # (sim ns, pan, tilt)
        self.release_pubs = {}
        self.last_feedback = None
        self.execution_failed = False
        self.moved = False  # a MoveIt goal ran for the current place goal
        self.at_place = None  # PlaceMotion while the held object is at the place pose
        self.released = None  # (request, motion) after the release until in the scene
        self.board = markers.SpotBoard()  # spots of the running goal (RViz)
        self.planning_scene = PlaceScene(self, group)

        self.candidates_client = self.create_client(
            GetPlaceCandidates, self.param("candidates_service"), callback_group=group
        )
        self.ik_client = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=group
        )
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=group
        )
        self.move_client = ActionClient(
            self, MoveGroup, "/move_action", callback_group=group
        )
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory", callback_group=group
        )
        self.gripper_pubs = {
            arm: self.create_publisher(Float64MultiArray, cfg["gripper_topic"], 10)
            for arm, cfg in ARMS.items()
        }
        self.look_pub = self.create_publisher(
            PointStamped, self.param("look_at_topic"), 10
        )
        self.create_subscription(
            PlaneArray,
            self.param("planes_topic"),
            self.on_planes,
            10,
            callback_group=group,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_states,
            qos_profile_sensor_data,
            callback_group=group,
        )
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String,
            "/robot_description_semantic",
            self.on_srdf,
            latched,
            callback_group=group,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/aha/manipulation/place/markers", latched
        )
        # Phase and results for the robot monitor (aha_test_ui).
        self.ui = UiPublisher(self, "place_server")
        self.ui.status("idle")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.server = ActionServer(
            self,
            PlaceObject,
            "/aha/manipulation/place",
            execute_callback=self.execute,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=group,
        )
        self.get_logger().info("place server ready on /aha/manipulation/place")

    # --- parameters and subscriptions -------------------------------------------------

    def param(self, name):
        return self.get_parameter(name).value

    def motion_params(self):
        return geo.MotionParams(
            approach_distance=self.param("approach_distance"),
            lift_margin=self.param("lift_margin"),
            approach_height=self.param("approach_height"),
            retreat_distance=self.param("retreat_distance"),
            retreat_lift=self.param("retreat_lift"),
        )

    def scene_params(self):
        return scene.SceneParams(
            support_thickness=self.param("support_thickness"),
            surface_clearance=self.param("surface_clearance"),
            support_padding=self.param("support_padding"),
            wall_thickness=self.param("wall_thickness"),
        )

    def on_planes(self, msg):
        self.latest_planes = msg
        self.plane_messages.append(msg)

    def plane_message(self, stamp, timeout=1.0):
        """The /aha/perception/planes message of a detection stamp, or None."""

        def find():
            return next(
                (m for m in reversed(self.plane_messages) if m.header.stamp == stamp),
                None,
            )

        self.wait_for(lambda: find() is not None, timeout)
        return find()

    def on_joint_states(self, msg):
        self.joints.update(zip(msg.name, msg.position))
        if all(name in self.joints for name in HEAD_JOINTS):
            now = self.get_clock().now().nanoseconds
            self.head_history.append((now, *(self.joints[n] for n in HEAD_JOINTS)))

    def on_srdf(self, msg):
        self.named_states = named_states(msg.data)

    def on_goal(self, _goal):
        if self.busy.locked():
            self.get_logger().warn("rejecting goal: a place is running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_cancel(self, _goal_handle):
        self.cancel_event.set()
        handle = self.active_motion
        if handle is not None:
            handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    # --- blocking helpers (the executor spins in other threads) -------------------------

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise Canceled()

    def wait_future(self, future, timeout, what):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout):
            raise PlaceError(f"{what}: no response in {timeout:.0f} s")
        return future.result()

    def wait_for(
        self,
        condition,
        timeout,
        sim=False,
        period=0.05,
        wall_cap=WALL_CAP,
        cancelable=True,
    ):
        """Poll ``condition`` until true or ``timeout`` (sim or wall s) runs out;
        a cancel raises Canceled unless not ``cancelable``."""
        clock = self.get_clock()
        start_wall, start = time.monotonic(), clock.now()
        while True:
            if cancelable:
                self.check_cancel()
            if condition():
                return True
            wall = time.monotonic() - start_wall
            elapsed = (clock.now() - start).nanoseconds * 1e-9 if sim else wall
            if elapsed > timeout or wall > wall_cap:
                return False
            time.sleep(period)

    def call(self, client, request, what, timeout=None):
        timeout = timeout or self.param("service_timeout")
        if not client.wait_for_service(timeout_sec=timeout):
            raise PlaceError(f"{what}: service {client.srv_name} not available")
        return self.wait_future(client.call_async(request), timeout, what)

    def run_action(self, client, goal, what):
        """Send a MoveIt goal and wait for its result; a cancel of the place
        goal cancels it and raises Canceled."""
        self.check_cancel()
        if not client.wait_for_server(timeout_sec=self.param("service_timeout")):
            raise PlaceError(f"{what}: action server not available")
        handle = self.wait_future(
            client.send_goal_async(goal), self.param("service_timeout"), what
        )
        if not handle.accepted:
            raise PlaceError(f"{what}: goal rejected")
        self.moved = True
        self.active_motion = handle
        try:
            if self.cancel_event.is_set():
                handle.cancel_goal_async()
            future = handle.get_result_async()
            # Node time: a slow simulation must not time out a normal motion.
            timeout = self.param("motion_timeout")
            if not self.wait_for(
                future.done,
                timeout,
                sim=True,
                wall_cap=MOTION_WALL_CAP,
                cancelable=False,
            ):
                handle.cancel_goal_async()
                raise PlaceError(
                    f"{what}: no result in {timeout:.0f} s",
                    Result.STATUS_EXECUTION_FAILED,
                )
        finally:
            self.active_motion = None
        self.check_cancel()
        return future.result().result

    def note_error(self, code):
        if code in selection.EXECUTION_ERRORS:
            self.execution_failed = True

    # --- TF ------------------------------------------------------------------------------

    def try_lookup(self, target, source, stamp=None, timeout=1.0):
        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                source,
                stamp or Time(),
                timeout=RclpyDuration(seconds=timeout),
            )
        except TransformException as error:
            return None, error
        t = transform.transform.translation
        q = transform.transform.rotation
        T = geo.transform(
            geo.matrix_from_quaternion(q.x, q.y, q.z, q.w), [t.x, t.y, t.z]
        )
        return T, None

    def lookup(self, target, source):
        T, error = self.try_lookup(target, source)
        if T is None:
            raise PlaceError(f"no TF {target} <- {source}: {error}")
        return T

    # --- planning scene ------------------------------------------------------------------

    def add_placed_object(self):
        """Keep the released object in the planning scene, after the retreat."""
        if self.released is not None:
            req, motion = self.released
            self.released = None
            self.planning_scene.add_placed(req, motion)

    # --- MoveIt requests -----------------------------------------------------------------

    def pose_stamped(self, T):
        msg = PoseStamped()
        msg.header.frame_id = self.param("planning_frame")
        msg.pose = scene.pose_msg(T)
        return msg

    def ik(self, cfg, T, avoid_collisions=True):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = cfg["group"]
        ik.ik_link_name = cfg["tcp"]
        ik.robot_state.is_diff = True
        ik.avoid_collisions = avoid_collisions
        ik.pose_stamped = self.pose_stamped(T)
        timeout = self.param("ik_timeout")
        ik.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        response = self.call(self.ik_client, request, "compute_ik")
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        state = response.solution.joint_state
        values = dict(zip(state.name, state.position))
        return {name: values[name] for name in cfg["joints"]}

    def move_to_joints(self, cfg, joints, what):
        goal = joint_goal(
            cfg["group"],
            joints,
            self.param("planning_time"),
            self.param("velocity_scaling"),
        )
        for attempt in range(max(1, self.param("plan_retries"))):
            code = self.run_action(self.move_client, goal, what).error_code.val
            if code == MoveItErrorCodes.SUCCESS:
                return True
            self.note_error(code)
            self.get_logger().warn(
                f"{what}: MoveIt error {code} (attempt {attempt + 1})"
            )
        return False

    def move_to_named(self, arm, name):
        cfg = ARMS[arm]
        joints = self.named_states.get((cfg["group"], name))
        if joints is None:
            raise PlaceError(f"named state {name!r} of {cfg['group']} not in the SRDF")
        self.planning_scene.allow_at(arm, joints, name)
        return self.move_to_joints(cfg, joints, f"move to {name}")

    def at_named_state(self, arm, state):
        joints = self.named_states.get((ARMS[arm]["group"], state))
        if not joints:
            return False
        return all(
            joint in self.joints and abs(self.joints[joint] - value) < STATE_TOLERANCE
            for joint, value in joints.items()
        )

    def cartesian(self, cfg, waypoints, what, min_fraction):
        request = GetCartesianPath.Request()
        request.header.frame_id = self.param("planning_frame")
        request.start_state.is_diff = True
        request.group_name = cfg["group"]
        request.link_name = cfg["tcp"]
        request.waypoints = [scene.pose_msg(T) for T in waypoints]
        request.max_step = float(self.param("cartesian_step"))
        request.avoid_collisions = True
        scaling = float(self.param("cartesian_velocity_scaling"))
        request.max_velocity_scaling_factor = scaling
        request.max_acceleration_scaling_factor = scaling
        response = self.call(self.cartesian_client, request, what)
        if response.fraction < min_fraction:
            self.get_logger().warn(
                f"{what}: Cartesian path fraction {response.fraction:.2f}"
            )
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        code = self.run_action(self.execute_client, goal, what).error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.execution_failed = True
            self.get_logger().warn(f"{what}: execution error {code}")
            return False
        return True

    # --- gripper and release ---------------------------------------------------------

    def set_gripper(self, arm, opening):
        self.gripper_pubs[arm].publish(
            Float64MultiArray(data=[float(opening), -float(opening)])
        )

    def release(self, arm, width):
        """Open the fingers ``release_margin`` beyond the object (y size, the
        finger direction), at most ``gripper_open``: wider fingers can reach
        into the padded boxes of obstacles near the spot."""
        topic = self.param("release_topic")
        if topic:
            topic = topic.replace("{arm}", arm)
            if topic not in self.release_pubs:
                self.release_pubs[topic] = self.create_publisher(Empty, topic, 10)
                time.sleep(0.3)  # let the subscription match
            self.release_pubs[topic].publish(Empty())
        opening = geo.finger_position(width, self.param("release_margin"))
        self.set_gripper(arm, min(opening, self.param("gripper_open")))
        time.sleep(self.param("release_settle"))

    # --- head, detections and the base --------------------------------------------------

    def preferred_point(self, req):
        px, py = self.param(
            "preferred_point_top" if req.grasp == geo.TOP else "preferred_point"
        )
        return px, -py if req.arm == "right" else py

    def aim_head(self, req, point=None, height=None):
        """Look at ``point`` (PointStamped), else the goal's near point, else
        ``look_point`` (toward the arm side, at the requested height, else
        ``height``, else ``look_height``) and wait for the head to settle.
        Returns the node time after it settled, or None if not aimed."""
        if self.look_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f"nothing subscribes {self.param('look_at_topic')}: head not aimed"
            )
            return None
        target = PointStamped()
        if point is not None:
            target.header.frame_id = point.header.frame_id
            target.point = point.point
        elif req.goal.near.header.frame_id:
            target.header.frame_id = req.goal.near.header.frame_id
            target.point = req.goal.near.point
        else:
            target.header.frame_id = self.param("planning_frame")
            x, y = self.param("look_point")
            target.point.x, target.point.y = x, -y if req.arm == "right" else y
            if req.surface_height is not None:
                height = req.surface_height
            elif height is None:
                height = self.param("look_height")
            target.point.z = height
        self.look_pub.publish(target)
        sent = self.get_clock().now().nanoseconds

        def settled():
            now = self.get_clock().now().nanoseconds
            if (now - sent) * 1e-9 < HEAD_MIN_WAIT:
                return False
            window = [
                s for s in self.head_history if (now - s[0]) * 1e-9 <= HEAD_WINDOW
            ]
            if len(window) < 2:
                return False
            spread = np.ptp(np.array([s[1:] for s in window]), axis=0)
            return bool((spread < HEAD_STILL).all())

        if not self.wait_for(settled, self.param("head_settle_timeout"), sim=True):
            self.get_logger().warn("head did not settle; perceiving anyway")
        p = target.point
        self.get_logger().info(
            f"head aimed at {target.header.frame_id} ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})"
        )
        return self.get_clock().now()

    def wait_detection(self, after, timeout):
        """Wait (wall s) for a plane detection stamped after ``after`` (node time)."""
        limit = after.nanoseconds

        def newer():
            msg = self.latest_planes
            return (
                msg is not None and Time.from_msg(msg.header.stamp).nanoseconds > limit
            )

        return self.wait_for(newer, timeout)

    def base_pose(self, stamp=None):
        T, _ = self.try_lookup(
            self.param("odom_frame"), self.param("planning_frame"), stamp, 0.5
        )
        return T

    def base_reference(self, stamp):
        """Base pose in odom at the detection stamp (else the latest), or None."""
        T = self.base_pose(Time.from_msg(stamp))
        if T is None:
            T = self.base_pose()
            if T is None:
                self.get_logger().warn(
                    f"no TF {self.param('odom_frame')} <- "
                    f"{self.param('planning_frame')}: base motion not checked"
                )
        return T

    def check_base(self, reference):
        if reference is None:
            return
        now = self.base_pose()
        if now is None:
            self.get_logger().warn(
                "no odom TF: base motion not checked", throttle_duration_sec=5.0
            )
            return
        shift, turn = self.param("base_max_shift"), self.param("base_max_turn")
        if geo.moved_beyond(reference, now, shift, math.radians(turn)):
            raise BaseMoved(*geo.pose_change(reference, now))

    def wait_base_still(self):
        """Wait until the base pose in odom stops changing (sim time)."""
        state = {"last": self.base_pose(), "t": self.get_clock().now()}

        def still():
            now = self.get_clock().now()
            if (now - state["t"]).nanoseconds * 1e-9 < BASE_SAMPLE:
                return False
            T = self.base_pose()
            last, state["last"], state["t"] = state["last"], T, now
            if T is None or last is None:
                return True
            distance, angle = geo.pose_change(last, T)
            return distance < BASE_STILL[0] and angle < BASE_STILL[1]

        return self.wait_for(still, self.param("base_settle_timeout"), sim=True)

    # --- perception ----------------------------------------------------------------------

    def exclude_regions(self, req):
        """Boxes of the held object and the links of both arms from the
        current TF (the other arm can stand over a table too)."""
        frame = self.param("planning_frame")
        poses = {}
        for arm in ARMS:
            for link in self_filter.arm_links(arm):
                T, _ = self.try_lookup(frame, link, timeout=0.2)
                if T is not None:
                    poses[link] = T
        tcp = self.lookup(frame, ARMS[req.arm]["tcp"])
        return self_filter.exclude_boxes(
            poses,
            tcp,
            req.hand,
            req.size,
            self.param("arm_padding"),
            self.param("held_padding"),
        )

    def candidates_request(self, req):
        """GetPlaceCandidates request with the padded object and the goal's
        support selection (surface ids, furniture, clearance). With
        ``oriented_footprint`` perception checks the footprint rectangle at the
        yaws of this server's place poses (x from the shoulder to the spot plus
        ``yaw_offsets``), padded by ``finger_padding`` across the fingers only:
        the hand and the arm are MoveIt's collision checks with their real
        shapes. Else the circle of the footprint padded by ``object_padding``."""
        goal = req.goal
        request = GetPlaceCandidates.Request()
        if self.param("oriented_footprint"):
            frame = self.param("planning_frame")
            shoulder = self.lookup(frame, ARMS[req.arm]["shoulder"])[:3, 3]
            request.approach_from.header.frame_id = frame
            request.approach_from.point.x, request.approach_from.point.y = (
                float(shoulder[0]),
                float(shoulder[1]),
            )
            request.yaw_offsets = [float(v) for v in self.param("yaw_offsets")]
            request.object_size.x = req.size[0]
            request.object_size.y = req.size[1] + self.param("finger_padding")
        else:
            padding = self.param("object_padding")
            request.object_size.x = req.size[0] + padding
            request.object_size.y = req.size[1] + padding
        request.object_size.z = req.size[2]
        request.surface_ids = list(goal.surface_ids)
        request.furniture = goal.furniture
        request.furniture_level = goal.furniture_level
        request.clearance = req.clearance
        return request

    def furniture_center(self, req):
        """Perception's view point of the goal's furniture level in the
        planning frame (asked before aiming the head); None without one."""
        request = self.candidates_request(req)
        request.max_candidates = 1
        response = self.call(self.candidates_client, request, "get_place_candidates")
        point = response.view_point
        if not point.header.frame_id:
            self.get_logger().warn(
                f"no view point of {req.goal.furniture}: {response.message}"
            )
            return None
        T, error = self.try_lookup(self.param("planning_frame"), point.header.frame_id)
        if T is None:
            self.get_logger().warn(f"view point not transformed: {error}")
            return None
        p = point.point
        return (T @ np.array([p.x, p.y, p.z, 1.0]))[:3]

    def furniture_look(self, req):
        """(look point, look height) for aim_head with a goal furniture and no
        near: the level center if within ``furniture_look_distance``, else
        ``look_point`` at the level height (a far center, e.g. of a large
        table, leaves the reachable part out of view)."""
        center = self.furniture_center(req)
        if center is None:
            return None, None
        if math.hypot(center[0], center[1]) > self.param("furniture_look_distance"):
            return None, float(center[2])
        point = PointStamped()
        point.header.frame_id = self.param("planning_frame")
        point.point.x, point.point.y, point.point.z = (float(v) for v in center)
        return point, None

    def query(self, req, use_near=True):
        """Candidates (to try, before the reach filter), planes and obstacles.
        Without the goal's ``near`` (or ``use_near`` false) the candidates are
        those nearest to the preferred point."""
        goal = req.goal
        near_frame = goal.near.header.frame_id if use_near else ""
        request = self.candidates_request(req)
        labels = [label for label in goal.surface_labels if label]
        if not labels:
            labels = [label for label in self.param("surface_labels") if label]
        request.surface_labels = labels
        if near_frame:
            request.near = goal.near
        request.max_candidates = self.param("candidate_pool")
        request.restrict_height = req.surface_height is not None
        if req.surface_height is not None:
            request.surface_height = req.surface_height
            request.height_tolerance = goal.height_tolerance
        request.exclude_regions = [
            scene.obstacle_msg(box) for box in self.exclude_regions(req)
        ]
        response = self.call(self.candidates_client, request, "get_place_candidates")
        near = None
        if response.success and not near_frame:
            near = self.candidates_near_arm(request, response, req)
        candidates, status = selection.select_candidates(
            response, near, req.surface_height, goal.height_tolerance
        )
        planes = list(response.planes.planes)
        matching = self.plane_message(response.stamp)
        if matching is not None:
            planes = list(matching.planes)  # also walls and other surfaces
        else:
            self.get_logger().warn(
                "no plane message for the detection; using the supports only"
            )
        return Perception(
            candidates,
            status,
            response.message,
            planes,
            list(response.obstacles),
            response.stamp,
            response.view_point if response.view_point.header.frame_id else None,
        )

    def candidates_near_arm(self, request, response, req):
        """Without a goal ``near``: per support height (of the restricted
        request, if any), the spots nearest to the preferred point (the
        clearance order prefers far table centers), sorted by horizontal
        distance to it."""
        frame = self.param("planning_frame")
        px, py = self.preferred_point(req)
        queries = []
        for height in selection.support_heights(response.planes.planes):
            request.near.header.frame_id = frame
            request.near.point.x, request.near.point.y, request.near.point.z = (
                px,
                py,
                height,
            )
            queries.append(
                (
                    height,
                    self.call(self.candidates_client, request, "get_place_candidates"),
                )
            )
        return selection.near_point_candidates(queries, (px, py))

    def near_point(self, req):
        """The goal's near point in the planning frame."""
        frame = self.param("planning_frame")
        near = req.goal.near
        point = np.array([near.point.x, near.point.y, near.point.z, 1.0])
        if near.header.frame_id != frame:
            point = self.lookup(frame, near.header.frame_id) @ point
        return point[:3]

    def avoid_points(self, req, points=None):
        """The goal's avoid points (or ``points``, near's frame) in the
        planning frame, xy."""
        frame = self.param("planning_frame")
        source = req.goal.near.header.frame_id
        T = np.eye(4) if source == frame else self.lookup(frame, source)
        points = req.goal.avoid if points is None else points
        return [tuple((T @ np.array([p.x, p.y, p.z, 1.0]))[:2]) for p in points]

    def group_order(self, req, candidates):
        """Candidates in the goal near point's group first, those beside it
        (``next_to_row_tolerance``) before those in front (see
        selection.order_for_group); unchanged without a near point."""
        if not req.goal.near.header.frame_id:
            return list(candidates)
        return selection.order_for_group(
            candidates,
            self.near_point(req)[:2],
            self.avoid_points(req),
            req.near_max_distance,
            self.param("next_to_row_tolerance"),
            self.param("group_ratio"),
            self.avoid_points(req, req.goal.group),
        )

    def reset_board(self, req):
        """Start the marker board of a detection with the goal's near, group
        and avoid points."""
        if not req.goal.near.header.frame_id:
            self.board.reset()
        else:
            near = self.near_point(req)
            self.board.reset(
                near, self.avoid_points(req, req.goal.group), self.avoid_points(req)
            )

    def show_queue(self, queue):
        self.board.queue([self.candidate_point(c) for c in queue])
        self.publish_board()

    def publish_board(self):
        self.marker_pub.publish(self.board.markers(self.param("planning_frame")))

    def suggest_base(self, req, spots):
        """PoseStamped (odom) of a base pose with the first spot at the arm's
        ``base_spot`` and the current heading, or None."""
        if not spots:
            return None
        bx, by = self.param("base_spot_top" if req.grasp == geo.TOP else "base_spot")
        offset = (bx, -by if req.arm == "right" else by)
        x, y = selection.base_for_spot(self.candidate_point(spots[0])[:2], offset)
        T, error = self.try_lookup(
            self.param("odom_frame"), self.param("planning_frame")
        )
        if T is None:
            self.get_logger().warn(f"no base pose suggested: {error}")
            return None
        p = T @ np.array([x, y, 0.0, 1.0])
        yaw = math.atan2(T[1, 0], T[0, 0])
        pose = PoseStamped()
        pose.header.frame_id = self.param("odom_frame")
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(p[0]), float(p[1])
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        self.get_logger().info(
            f"suggested base: move ({x:.2f}, {y:.2f}) m in the base frame "
            f"for the spot nearest to near"
        )
        return pose

    def candidate_transform(self, candidate):
        """planning_frame <- the candidate's frame at its stamp (the
        detection's; else the latest), or None if they are the same."""
        frame = self.param("planning_frame")
        source = candidate.pose.header.frame_id or frame
        if source == frame:
            return None
        stamp = Time.from_msg(candidate.pose.header.stamp)
        T, _ = self.try_lookup(frame, source, stamp, timeout=0.2)
        return T if T is not None else self.lookup(frame, source)

    def candidate_point(self, candidate):
        p = candidate.pose.pose.position
        point = np.array([p.x, p.y, p.z, 1.0])
        T = self.candidate_transform(candidate)
        return (point if T is None else T @ point)[:3]

    def candidate_yaws(self, req, candidate, point, shoulder):
        """Object yaws (planning frame) to try at a candidate: the one
        perception checked the footprint at (oriented) first, then the
        approach yaw from the shoulder plus each of ``yaw_offsets`` and of
        ``wrist_yaw_offsets`` (toward the arm side: the wrist sticks out on
        the arm's outer side of the hand, at the TCP height, and hit
        neighbours and side panels on that side; turning the yaw that way
        swings it behind the object). MoveIt checks the held object box at
        the yaws perception did not check."""
        approach = geo.approach_yaw(point[:2], shoulder)
        side = -1.0 if req.arm == "right" else 1.0
        offsets = list(self.param("yaw_offsets")) + [
            side * v for v in self.param("wrist_yaw_offsets")
        ]
        yaws = [approach + offset for offset in offsets]
        if candidate.oriented:
            # Heading of the pose's x axis (z is along the surface normal).
            q = candidate.pose.pose.orientation
            checked = math.atan2(
                2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.y * q.y + q.z * q.z)
            )
            T = self.candidate_transform(candidate)
            if T is not None:
                checked += math.atan2(T[1, 0], T[0, 0])
            yaws = [checked] + [
                y for y in yaws if abs(math.remainder(y - checked, 2 * math.pi)) > 0.05
            ]
        return yaws

    def motion(self, req, point, yaw, approach):
        return geo.place_motion(
            point,
            req.size,
            yaw,
            req.grasp,
            approach,
            req.release_height,
            self.param("top_tilt"),
            self.motion_params(),
            req.hand,
        )

    def reachable(self, req, candidates, obstacles=()):
        """The first ``max_candidates`` candidates (in order) within
        ``reach_limits`` of the shoulder whose place pose has an IK solution
        for an allowed approach and yaw offset with the descent column clear
        of ``obstacles`` (perceived boxes) at that yaw. The others checked on
        the way are dropped on the marker board."""
        cfg = ARMS[req.arm]
        shoulder = self.lookup(self.param("planning_frame"), cfg["shoulder"])[:2, 3]
        reach = self.param("reach_limits")[1 if req.grasp == geo.TOP else 0]

        def ok(candidate):
            self.check_cancel()
            point = self.candidate_point(candidate)
            if not selection.within_reach(point, shoulder, reach):
                self.explain(req, point, "beyond reach_limits")
                self.board.drop(point, "reach")
                return False
            yaws = self.candidate_yaws(req, candidate, point, shoulder)
            blocked = 0
            for yaw in yaws:
                for approach in req.approaches:
                    motion = self.motion(req, point, yaw, approach)
                    if self.column_blocked(req, motion, obstacles):
                        blocked += 1
                        continue
                    if self.ik(cfg, motion.tcp_place) is not None:
                        return True
            if blocked == len(yaws) * len(req.approaches):
                self.explain(req, point, "descent column blocked at every yaw")
                self.board.drop(point, "blocked")
                return False
            if self.param("explain_unreachable"):
                self.explain(req, point, self.ik_failure(req, cfg, point, yaws[0]))
            self.board.drop(point, "IK")
            return False

        out = selection.first_reachable(candidates, ok, self.param("max_candidates"))
        self.get_logger().info(
            f"{len(out)} reachable candidates (of {len(candidates)}, "
            f"reach {reach:.2f} m from the shoulder)"
        )
        return out

    def explain(self, req, point, reason):
        self.get_logger().info(
            f"spot ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) unreachable: {reason}"
        )

    def box_text(self, object_id):
        """ " (center x y z; size)" of a perceived box (as perceived, no
        padding), empty for other objects."""
        box = self.planning_scene.perceived.get(object_id)
        if box is None:
            return ""
        c = box.pose[:3, 3]
        return (
            f" ({c[0]:.3f} {c[1]:.3f} {c[2]:.3f}; "
            f"{box.size[0]:.3f} x {box.size[1]:.3f} x {box.size[2]:.3f})"
        )

    def ik_failure(self, req, cfg, point, yaw):
        """Why no place pose had an IK solution, per approach at the base yaw:
        no solution even ignoring collisions, or the collisions of one."""

        def is_object(name):
            return name.startswith(scene.PERCEIVED_PREFIXES) or name.startswith(
                "aha_placed_object"
            )

        reasons = []
        for approach in req.approaches:
            motion = self.motion(req, point, yaw, approach)
            joints = self.ik(cfg, motion.tcp_place, avoid_collisions=False)
            if joints is None:
                reasons.append(f"{approach}: no IK solution")
                continue
            hits = self.planning_scene.contacts(req.arm, is_object, joints)
            if not hits:
                reasons.append(f"{approach}: IK only without collision checking")
                continue
            text = ", ".join(
                f"{obj}{self.box_text(obj)} x {'/'.join(sorted(bodies))} at "
                f"({np.mean(points, axis=0)[0]:.3f} {np.mean(points, axis=0)[1]:.3f} "
                f"{np.mean(points, axis=0)[2]:.3f})"
                for obj, (bodies, points) in hits.items()
            )
            tcp = motion.tcp_place[:3, 3]
            reasons.append(
                f"{approach} (yaw {math.degrees(yaw):.0f} deg, tcp "
                f"{tcp[0]:.3f} {tcp[1]:.3f} {tcp[2]:.3f}): collides {text}"
            )
        return "; ".join(reasons)

    def column_blocked(self, req, motion, obstacles):
        """Indices of obstacle messages inside the motion's descent column."""
        T, size = geo.descent_column(
            motion, req.size, self.param("late_check_margin"), self.motion_params()
        )
        column = scene.Box(scene.OBSTACLE, T, size)
        return [
            i
            for i, obstacle in enumerate(obstacles)
            if scene.boxes_overlap(column, scene.obstacle_box(obstacle))
        ]

    def check_at_pre_place(self, goal_handle, req, index):
        """Fresh detection after reaching the pre-place pose: add its obstacle
        boxes to the scene. Returns its Perception, or None without one."""
        self.feedback(goal_handle, "checking", index)
        arrived = self.get_clock().now()
        if not self.wait_detection(arrived, self.param("late_check_timeout")):
            self.get_logger().warn(
                "no detection after reaching the pre-place pose: spot not re-checked"
            )
            return None
        perception = self.query(req)
        self.planning_scene.add_obstacles(perception.obstacles, req)
        return perception

    # --- goal ------------------------------------------------------------------------

    def parse_goal(self, goal):
        arm = goal.arm or self.param("default_arm")
        if arm not in ARMS:
            raise PlaceError(f"unknown arm {arm!r}")
        grasp = goal.grasp or self.param("default_grasp")
        if grasp not in geo.GRASPS:
            raise PlaceError(f"unknown grasp {grasp!r}")
        approach = goal.approach or self.param("default_approach")
        if approach == "auto":
            approaches = [geo.SIDE, geo.TOP]
        elif approach in geo.APPROACHES:
            approaches = [approach]
        else:
            raise PlaceError(f"unknown approach {approach!r}")
        size = (goal.object_size.x, goal.object_size.y, goal.object_size.z)
        if min(size) <= 0:
            raise PlaceError(
                "object_size must be positive", Result.STATUS_INVALID_OBJECT
            )
        error = geo.object_fit_error(
            grasp, size, self.param("gripper_open"), self.param("finger_clearance")
        )
        if error:
            raise PlaceError(error, Result.STATUS_INVALID_OBJECT)
        release_height = goal.release_height or self.param("release_height")
        surface_height = None
        if goal.restrict_height:
            surface_height = float(goal.surface_height)
            if not math.isfinite(surface_height):
                raise PlaceError("surface_height must be finite")
        clearance = float(goal.clearance)
        if not (math.isfinite(clearance) and clearance >= 0.0):
            raise PlaceError(f"clearance must be finite and >= 0, got {clearance}")
        if goal.furniture_level and not goal.furniture:
            raise PlaceError("furniture_level given without furniture")
        near_max_distance = float(goal.near_max_distance)
        if not (math.isfinite(near_max_distance) and near_max_distance >= 0.0):
            raise PlaceError(
                f"near_max_distance must be finite and >= 0, got {near_max_distance}"
            )
        if (
            goal.avoid or goal.group or near_max_distance > 0.0
        ) and not goal.near.header.frame_id:
            raise PlaceError("avoid, group and near_max_distance need near")
        return Request(
            goal,
            arm,
            grasp,
            approaches,
            size,
            float(release_height),
            surface_height,
            clearance,
            near_max_distance,
            self.object_in_hand(goal, arm, grasp, size),
        )

    def object_in_hand(self, goal, arm, grasp, size):
        """T_tcp_object: the goal's ``object_in_hand`` (a pose in a frame
        fixed to the hand, through the TF to the TCP), else the grasp's
        nominal pose (place_geometry.object_in_tcp)."""
        pose = goal.object_in_hand
        if not pose.header.frame_id:
            return geo.object_in_tcp(grasp, size[2], self.param("top_tilt"))
        q, p = pose.pose.orientation, pose.pose.position
        values = (q.x, q.y, q.z, q.w, p.x, p.y, p.z)
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if not all(math.isfinite(v) for v in values) or norm < 1e-6:
            raise PlaceError(
                "object_in_hand needs a finite pose and a nonzero quaternion"
            )
        T = geo.transform(
            geo.matrix_from_quaternion(q.x / norm, q.y / norm, q.z / norm, q.w / norm),
            [p.x, p.y, p.z],
        )
        tcp = ARMS[arm]["tcp"]
        if pose.header.frame_id != tcp:
            T = self.lookup(tcp, pose.header.frame_id) @ T
        return T

    def feedback(self, goal_handle, phase, index):
        if (phase, index) != self.last_feedback:
            self.get_logger().info(f"{phase} (candidate {index})")
            self.last_feedback = (phase, index)
            self.ui.status(phase, f"candidate {index + 1}")
        msg = PlaceObject.Feedback()
        msg.phase = phase
        msg.candidate_index = index
        goal_handle.publish_feedback(msg)

    def try_candidate(
        self, goal_handle, req, index, candidate, point, base_ref, obstacles=()
    ):
        """The motion at the place pose, a Blocked spot, or None. Yaws whose
        descent column meets a known obstacle box (``obstacles`` and those of
        the fresh detection) are skipped before the arm moves; the fresh
        detection at the pre-place pose catches what changed since."""
        cfg = ARMS[req.arm]
        shoulder = self.lookup(self.param("planning_frame"), cfg["shoulder"])[:2, 3]
        yaws = self.candidate_yaws(req, candidate, point, shoulder)
        fresh = None
        for approach in req.approaches:
            for yaw in yaws:
                self.check_cancel()
                motion = self.motion(req, point, yaw, approach)
                self.feedback(goal_handle, "planning", index)
                known = list(obstacles) + (list(fresh.obstacles) if fresh else [])
                if self.column_blocked(req, motion, known):
                    self.board.note("blocked")
                    continue
                if self.ik(cfg, motion.tcp_place) is None:
                    self.board.note("IK")
                    continue
                pre = self.ik(cfg, motion.pre_place)
                if pre is None:
                    self.board.note("IK")
                    continue
                label = (
                    f"candidate {index} {approach} yaw {math.degrees(motion.yaw):.0f}"
                )
                self.check_base(base_ref)
                if not self.move_to_joints(cfg, pre, f"{label} pre-place"):
                    self.board.note("plan")
                    continue
                if self.param("late_check"):
                    if fresh is None:
                        fresh = (
                            self.check_at_pre_place(goal_handle, req, index) or False
                        )
                    if fresh:
                        hits = self.column_blocked(req, motion, fresh.obstacles)
                        if hits:
                            self.get_logger().warn(
                                f"{label}: {len(hits)} obstacle boxes at the spot "
                                "in the fresh detection; next candidate"
                            )
                            return Blocked(fresh.candidates, fresh.obstacles)
                self.check_base(base_ref)
                self.feedback(goal_handle, "approaching", index)
                min_fraction = self.param("cartesian_min_fraction")
                if not self.cartesian(
                    cfg, motion.approach_waypoints, f"{label} approach", min_fraction
                ):
                    self.board.note("path")
                    continue
                self.at_place = motion
                self.get_logger().info(f"{label}: at the place pose")
                return motion
        return None

    def place_once(self, goal_handle, req, result, since=None):
        """Perceive, plan and place; raises BaseMoved to start over."""
        self.feedback(goal_handle, "perceiving", 0)
        aim = self.param("aim_head")
        look, height = None, None
        if aim and req.goal.furniture and not req.goal.near.header.frame_id:
            look, height = self.furniture_look(req)
        if aim:
            since = self.aim_head(req, look, height) or since
        perception = self.perceive(req, since)
        if (
            aim
            and perception.status == Result.STATUS_NO_SURFACE
            and perception.view_point is not None
        ):
            # The furniture level is not in view: look at its center.
            self.get_logger().info(
                f"{perception.message}; looking at {req.goal.furniture}"
            )
            since = self.aim_head(req, perception.view_point) or since
            perception = self.perceive(req, since)
        if not perception.candidates:
            raise PlaceError(
                f"no place candidates: {perception.message}", perception.status
            )
        base_ref = self.base_reference(perception.stamp)
        touching = self.planning_scene.replace(
            perception.planes, perception.obstacles, req, self.scene_params()
        )
        if touching:
            self.escape(req, touching)
        self.reset_board(req)
        obstacles = list(perception.obstacles)
        spots = self.group_order(req, perception.candidates)
        queue = self.reachable(req, spots, obstacles)
        if not queue and req.near_max_distance > 0.0:
            self.show_queue(queue)
            result.suggested_base_pose = self.suggest_base(req, spots) or PoseStamped()
            raise PlaceError(
                f"no reachable spot within {req.near_max_distance:.2f} m of near "
                f"({len(spots)} out of reach)",
                Result.STATUS_UNREACHABLE,
            )
        if not queue and req.goal.near.header.frame_id:
            # All spots around near are out of reach: those around the
            # preferred point, near's group and nearest to near first.
            fallback = self.query(req, use_near=False)
            self.get_logger().info("no reachable spot near the goal's near point")
            obstacles += fallback.obstacles
            queue = self.reachable(
                req, self.group_order(req, fallback.candidates), obstacles
            )
        self.show_queue(queue)
        if not queue:
            raise PlaceError(
                f"none of {len(perception.candidates)} candidates within reach",
                Result.STATUS_UNREACHABLE,
            )
        blocked, refreshes, index = [], 0, -1
        spacing = math.hypot(req.size[0], req.size[1])
        while queue:
            self.check_cancel()
            candidate = queue.pop(0)
            index += 1
            result.candidates_tried = index + 1
            point = self.candidate_point(candidate)
            self.board.start()
            self.publish_board()
            outcome = self.try_candidate(
                goal_handle, req, index, candidate, point, base_ref, obstacles
            )
            if isinstance(outcome, Blocked):
                self.board.finish(markers.FAILED, "blocked")
                blocked.append(point)
                obstacles += outcome.obstacles
                fresh = selection.drop_near(outcome.candidates, blocked, spacing)
                fresh = self.group_order(req, fresh)
                if fresh and refreshes < MAX_REFRESHES:
                    refreshes += 1
                    queue = self.reachable(req, fresh, obstacles)
                else:
                    queue = selection.drop_near(queue, blocked, spacing)
                self.show_queue(queue)
                continue
            if outcome is not None:
                result.surface_height = float(candidate.surface_height)
                result.surface_id = int(candidate.surface_id)
                self.finish(goal_handle, req, result, outcome, index, base_ref)
                return
            self.board.finish(markers.FAILED)
            self.publish_board()
        raise PlaceError(
            f"no reachable placement in {index + 1} candidates",
            selection.failure_status(self.execution_failed),
        )

    def perceive(self, req, since):
        """Query perception on a detection after ``since`` (node time, if
        any); an error (e.g. a too old detection) is asked again."""
        if since is not None and not self.wait_detection(
            since, self.param("detection_timeout")
        ):
            self.get_logger().warn("no new detection; using the latest")
        perception = self.query(req)
        for _ in range(DETECTION_RETRIES):
            if perception.status != Result.STATUS_ERROR:
                break
            self.get_logger().warn(f"{perception.message}; asking again")
            self.wait_detection(
                self.get_clock().now(), 3 * self.param("detection_timeout")
            )
            perception = self.query(req)
        return perception

    def escape(self, req, touching):
        """The hand is at real objects (their boxes had to be cut or
        allowed): lift the TCP ``escape_lift`` straight up first, then put the
        perceived boxes back, so that planning sees them whole."""
        lift = self.param("escape_lift")
        if lift <= 0.0:
            return
        tcp = self.lookup(self.param("planning_frame"), ARMS[req.arm]["tcp"])
        up = tcp.copy()
        up[2, 3] += lift
        self.get_logger().warn(
            f"hand at obstacle boxes {touching}: lifting {lift:.2f} m first"
        )
        if not self.cartesian(ARMS[req.arm], [up], "escape lift", 0.5):
            self.get_logger().warn("could not lift away from the obstacle boxes")
        self.planning_scene.restore(touching, req)

    def back_out(self, req):
        """From the place pose (object still held) back to the pre-place pose."""
        motion, self.at_place = self.at_place, None
        if motion is None:
            return
        path = list(reversed(motion.approach_waypoints[:-1])) + [motion.pre_place]
        try:
            if not self.cartesian(ARMS[req.arm], path, "back out", 0.5):
                self.get_logger().warn("could not back out of the place pose")
        except PlaceError as error:
            self.get_logger().warn(f"could not back out of the place pose: {error}")

    def execute(self, goal_handle):
        with self.busy:
            started = time.monotonic()
            self.cancel_event.clear()
            if goal_handle.is_cancel_requested:
                self.cancel_event.set()
            self.last_feedback = None
            self.execution_failed = False
            self.moved = False
            self.at_place = None
            self.released = None
            result = PlaceObject.Result()
            req = None
            try:
                req = self.parse_goal(goal_handle.request)
                replans, since = 0, None
                while True:
                    try:
                        self.place_once(goal_handle, req, result, since)
                        break
                    except BaseMoved as moved:
                        self.get_logger().warn(str(moved))
                        if replans >= self.param("max_replans"):
                            raise PlaceError(
                                f"{moved}; {replans} replans used",
                                Result.STATUS_BASE_MOVED,
                            ) from moved
                        replans += 1
                        self.back_out(req)
                        if not self.wait_base_still():
                            raise PlaceError(
                                f"{moved}; still moving", Result.STATUS_BASE_MOVED
                            ) from moved
                        since = self.get_clock().now()
                        self.get_logger().info(f"perceiving again (replan {replans})")
            except Canceled:
                return self.canceled(goal_handle, result, req)
            except PlaceError as error:
                return self.failed(goal_handle, result, req, error)
            if replans:
                result.message += f"; {replans} replans after base motion"
            result.message += f" ({time.monotonic() - started:.1f} s)"
            if not result.success:
                # Released, but the check after the return did not see it.
                self.get_logger().error(f"place failed: {result.message}")
                self.report(result)
                goal_handle.abort()
                return result
            self.get_logger().info(f"place done: {result.message}")
            self.report(result)
            goal_handle.succeed()
            return result

    def report(self, result):
        """The goal's result for the robot monitor."""
        status = next(
            (
                n[7:]
                for n in dir(Result)
                if n.startswith("STATUS_") and getattr(Result, n) == result.status
            ),
            str(result.status),
        )
        self.ui.status("idle", f"last: {status}")
        self.ui.event(
            f"{status}: {result.message}", "info" if result.success else "warn"
        )

    def canceled(self, goal_handle, result, req):
        placed = self.released is not None or result.success
        try:
            self.add_placed_object()
        except (PlaceError, Canceled) as error:
            self.get_logger().warn(f"placed object not added: {error}")
        result.success = False
        result.status = Result.STATUS_CANCELED
        result.message = "canceled" + ("; the object was released" if placed else "")
        if req is not None:
            result.returned = self.at_named_state(req.arm, self.param("return_state"))
        self.get_logger().info(f"place {result.message}")
        self.report(result)
        goal_handle.canceled()
        return result

    def failed(self, goal_handle, result, req, error):
        result.success = False
        result.status = error.status
        result.message = str(error)
        self.get_logger().error(f"place failed: {error}")
        state = self.param("return_state")
        if req is not None and self.moved:
            # The arm moved for this goal: back to the return state with the object.
            self.cancel_event.clear()  # a late cancel request must not stop it
            try:
                self.back_out(req)
                result.returned = self.recover(req.arm)
            except Canceled:
                result.returned = False
        else:
            arm = goal_handle.request.arm or self.param("default_arm")
            result.returned = arm in ARMS and self.at_named_state(arm, state)
        self.report(result)
        goal_handle.abort()
        return result

    def finish(self, goal_handle, req, result, motion, index, base_ref):
        self.check_cancel()
        self.check_base(base_ref)
        self.feedback(goal_handle, "releasing", index)
        self.release(req.arm, req.size[1])
        self.at_place = None
        self.released = (req, motion)
        self.planning_scene.detach_held(req.arm)
        result.success = True
        result.status = Result.STATUS_OK
        result.place_pose = self.pose_stamped(motion.place_pose)
        result.place_pose.header.stamp = self.get_clock().now().to_msg()
        result.approach_used = motion.approach
        notes = []
        self.feedback(goal_handle, "retreating", index)
        try:
            self.planning_scene.resolve_contacts(req)
            fraction = self.param("retreat_min_fraction")
            retreated = self.cartesian(
                ARMS[req.arm], motion.retreat_waypoints, "retreat", fraction
            )
            if not retreated and motion.retreat_fallback:
                retreated = self.cartesian(
                    ARMS[req.arm], motion.retreat_fallback, "retreat up", fraction
                )
                if retreated:
                    notes.append("retreated straight up")
            if not retreated:
                notes.append("retreat failed")
        except PlaceError as error:
            notes.append(f"retreat failed ({error})")
        try:
            self.add_placed_object()
        except PlaceError as error:
            notes.append(f"placed object not in the planning scene ({error})")
        self.feedback(goal_handle, "returning", index)
        state = self.param("return_state")
        try:
            result.returned = self.move_to_named(req.arm, state)
            if not result.returned:
                notes.append(f"return to {state} failed")
        except PlaceError as error:
            notes.append(f"return to {state} failed ({error})")
        if self.param("verify_place"):
            self.feedback(goal_handle, "verifying", index)
            seen = self.verify(req, motion, result.surface_height)
            if seen is None:
                notes.append("not verified (no detection)")
            else:
                result.verified = seen
                if not seen:
                    result.success = False
                    result.status = Result.STATUS_NOT_AT_SPOT
                    notes.append("no object at the spot after the return")
        if result.success:
            self.board.finish(markers.PLACED, "")
        else:
            self.board.finish(markers.FAILED, "not seen")
        self.publish_board()
        result.message = f"placed with {motion.approach} approach" + (
            "; " + ", ".join(notes) if notes else ""
        )

    def verify(self, req, motion, surface_height):
        """Whether a detection after the return sees an object standing at the
        place spot (perception's obstacle boxes); None without a detection."""
        point = motion.place_pose[:3, 3]
        since = None
        if self.param("aim_head"):
            target = PointStamped()
            target.header.frame_id = self.param("planning_frame")
            target.point.x, target.point.y, target.point.z = (float(v) for v in point)
            since = self.aim_head(req, target)
        since = since or self.get_clock().now()
        if not self.wait_detection(since, self.param("detection_timeout")):
            self.get_logger().warn("no detection after the return: place not verified")
            return None
        request = GetPlaceCandidates.Request()
        request.object_size.x, request.object_size.y, request.object_size.z = (
            float(v) for v in req.size
        )
        request.max_candidates = 1
        request.exclude_regions = [
            scene.obstacle_msg(box) for box in self.exclude_regions(req)
        ]
        response = self.call(self.candidates_client, request, "get_place_candidates")
        if not response.success:
            self.get_logger().warn(f"place not verified: {response.message}")
            return None
        seen = scene.object_at(
            response.obstacles,
            point,
            surface_height,
            self.param("verify_margin"),
            self.param("verify_height_tolerance"),
        )
        if not seen:
            self.get_logger().error(
                f"no object at the place spot ({point[0]:.2f}, {point[1]:.2f}) on "
                f"the {surface_height:.2f} m support after the return "
                f"({len(response.obstacles)} obstacle boxes)"
            )
        return seen

    def recover(self, arm):
        """Return to return_state with the object still in the hand (best effort)."""
        state = self.param("return_state")
        if self.at_named_state(arm, state):
            return True
        try:
            if self.move_to_named(arm, state):
                return True
            self.get_logger().warn("could not return after a failed place")
        except PlaceError as error:
            self.get_logger().warn(f"could not return after a failed place: {error}")
        return False


def main():
    rclpy.init()
    node = PlaceServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # SIGINT shuts the context down while the executor builds a wait set.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
