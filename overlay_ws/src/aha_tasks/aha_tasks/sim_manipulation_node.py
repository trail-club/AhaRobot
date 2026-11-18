"""Simulation only: pick, place and handover servers that move objects in Gazebo.

Stands in for the manipulation squad's servers in debug end-to-end runs
(task.launch.py e2e:=debug), as sim_teleport does for Nav2. No grasp is
planned; the object is moved by gz-transport (aha_tasks.sim_manipulation):

- pick (aha_msgs/action/PickObject, /aha/manipulation/pick): the object
  model of the target's label nearest the target (MATCH_DISTANCE), within
  `reach` of the base. The right arm reaches out over it, the model becomes
  a static copy without collision that follows the TCP (`tcp_frame`), and
  the arm goes to home, the carry pose (CARRY), where every goal ends (the
  actions' `at_home`).
- place (aha_msgs/action/PlaceObject, /aha/manipulation/place): the arm
  reaches out over `near` (within `reach`; over the top of a container to
  drop into), the static copy is set standing at `near` (`drop`: its centre
  at `near`), the arm goes back to home, and the dynamic model is created
  there, clear of the hand (a dropped object falls).
- handover (aha_msgs/action/HandOver, /aha/manipulation/handover): the arm
  points at the target at its height; with release "take" the simulated
  person takes the object `take_delay` s later (never if `take` is false,
  then the goal times out); the model is removed and the arm goes back to
  home.

Once the object is in the hand, a pick succeeds, also when the arm does not
get to carry (`at_home` false) or grasp_pose_used is not known; once it left
the hand, place and handover succeed, also when Gazebo then fails to
re-create or remove it (the message says so): a failure would leave the task
believing the hand holds otherwise.

The arm moves in straight lines in joint space (aha_tasks.gestures.Arm,
aha_manipulation.pointing) without collision checks, the lift before the
arm when going up and after it when going down; the hand passes HAND_ABOVE
over the objects it reaches for. `move_arm` false skips the motions. Only
the right arm is served. Model poses come from Gazebo's pose/info, targets
in `map` go to the world through the robot's Gazebo pose and the TF map ->
base_footprint (the detections' path in sim_object_detector.py). Waits are
on the simulation clock. On exit, held objects are put back into the
physics where they are.
"""

import math
from pathlib import Path
import threading
import time
import traceback

import numpy as np

from aha_tasks.sim_manipulation import (
    CARRY,
    REACH,
    aim_point,
    apply,
    find_model,
    held_pose,
    horizontal_distance,
    pose_matrix,
    resting_centre,
    static_copy,
    yaw_of,
)

REQUEST_TIMEOUT = 2000  # ms
WORLD_TIMEOUT = 60.0  # s (wall) to find the Gazebo world; then exit (respawn)
POSE_MAX_AGE = 2.0  # s (wall) of the robot's Gazebo pose
REMOVE_TIMEOUT = 5.0  # s (wall) until a removed model leaves pose/info
TF_TIMEOUT = 2.0  # s (wall)
CLOCK_STALL_TIMEOUT = 10.0  # s (wall) without the simulation clock advancing
MOVED = (0.001, 0.005)  # m, rad; a held object closer than this is not moved again
HAND_ABOVE = 0.08  # m from the hand to the top of an object it reaches over
DEFAULT_CACHE = Path.home() / ".cache" / "aharobot" / "sim_objects"


class Failed(Exception):
    """A goal fails with this message (and the status in `status`).

    `at_home`: whether the arm is at home (CARRY) at the end; None: the arm
    did not move."""

    def __init__(self, message, status=None, at_home=None):
        super().__init__(message)
        self.status, self.at_home = status, at_home


def failed_status(error):
    """The PlaceObject / HandOver status name of a failed goal: its own, else
    STATUS_ERROR before the arm moved (`at_home` None), else
    STATUS_EXECUTION_FAILED (the runner then reads `at_home`)."""
    if error.status:
        return error.status
    return "STATUS_ERROR" if error.at_home is None else "STATUS_EXECUTION_FAILED"


class Flag:
    """`is_set` of a goal's cancel request, for aha_tasks.gestures.Arm."""

    def __init__(self, handle):
        self.handle = handle

    def is_set(self):
        return self.handle.is_cancel_requested


class Held:
    def __init__(self, name, label, yaw_offset, size):
        self.name, self.label = name, label
        self.yaw_offset, self.size = yaw_offset, size
        self.sent = None  # last (x, y, z, yaw) sent to Gazebo


class Stand:
    """The state and Gazebo access of the stand-in."""

    def __init__(self, node, parameters):
        from ament_index_python.packages import get_package_share_directory
        from gz.transport13 import Node as GzNode
        from tf2_ros import Buffer, TransformListener

        from aha_perception.furniture import load_config
        from aha_perception.gazebo_poses import GazeboPoses
        from aha_perception.object_list import load_object_list
        from aha_tasks.sim_objects import Catalog

        self.node, self.log = node, node.get_logger()
        self.p = parameters
        perception = Path(get_package_share_directory("aha_perception"), "config")
        tasks = Path(get_package_share_directory("aha_tasks"), "config")
        tmc = Path(get_package_share_directory("tmc_wrs_gz_worlds"), "models")
        self.object_list = load_object_list(
            parameters["objects_config"]
            or perception / "objects" / "robocup_objects.yaml"
        )
        self.catalog = Catalog.load(
            tasks / "sim" / "object_models.json", self.object_list, tmc
        )
        # Only for the height of a container to drop into (frame world).
        self.furniture = load_config(
            parameters["furniture_config"]
            or perception / "furniture" / "rcjo2026_arena.yaml"
        )
        self.gazebo = GazeboPoses(parameters["world"], WORLD_TIMEOUT, 20.0)
        self.world = self.gazebo.world
        self.requests = GzNode()
        self.request_lock = threading.Lock()
        self.lock = threading.Lock()
        self.held = None  # Held
        self.placed = None  # (Held, centre, yaw) let go, still static
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)
        self.arm = None
        if parameters["move_arm"]:
            from aha_tasks.gestures import Arm

            self.arm = Arm(node)

    def param(self, name):
        """A parameter that may change at run time (reach, take, take_delay, ...)."""
        return self.node.get_parameter(name).value

    # -- Gazebo ---------------------------------------------------------------

    def request(self, service, message, reply_type):
        with self.request_lock:
            ok, reply = self.requests.request(
                f"/world/{self.world}/{service}",
                message,
                type(message),
                reply_type,
                REQUEST_TIMEOUT,
            )
        return ok and reply.data

    def poses(self):
        poses, age = self.gazebo.latest()
        base = poses.get(self.p["robot_model"])
        if base is None or age > POSE_MAX_AGE:
            raise Failed(f"No recent Gazebo pose of {self.p['robot_model']}")
        return poses, base

    def set_pose(self, name, x, y, z, yaw):
        from aha_tasks.guest_node import set_pose

        with self.request_lock:
            return set_pose(self.requests, self.world, name, x, y, z, yaw)

    def remove(self, name):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.entity_pb2 import Entity

        # A request without a reply in time may still be carried out (seen in
        # the simulation): pose/info decides.
        asked = self.request("remove", Entity(name=name, type=Entity.MODEL), Boolean)
        if not self.present(name, False, REMOVE_TIMEOUT):
            return
        what = "stays in Gazebo after its removal" if asked else "could not be removed"
        raise Failed(f"{name} {what}")

    def create(self, sdf, name):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.entity_factory_pb2 import EntityFactory

        for _ in range(3):
            if self.request("create", EntityFactory(sdf=sdf), Boolean):
                return
            if self.present(name, True, 1.0):  # created, the reply was lost
                return
        raise Failed(f"Could not create {name}")

    def present(self, name, expected, timeout):
        """Whether pose/info shows the model, waiting up to `timeout` s (wall)
        for it to be `expected` (present or gone) in a fresh message."""
        deadline = time.monotonic() + timeout
        while True:
            poses, age = self.gazebo.latest()
            fresh = age < 0.5
            if fresh and (name in poses) == expected:
                return expected
            if time.monotonic() > deadline:
                return name in poses if fresh else not expected
            time.sleep(0.1)

    def sdf(self, name, label, x, y, z, yaw):
        from aha_tasks.sim_objects import DROP, SceneObject

        entry = self.object_list.get(label)
        obj = SceneObject(
            name=name,
            label=label,
            category=entry.category,
            on="floor",
            level=0,
            role="",
            position=(x, y, z - DROP),  # spawn_pose adds DROP
            yaw=yaw,
            size=entry.size,
        )
        return self.catalog.sdf(obj, Path(self.p["cache"]))

    # -- frames ---------------------------------------------------------------

    def transform(self, target, source):
        from rclpy.time import Time
        from tf2_ros import TransformException

        from aha_perception.gazebo_poses import tf_matrix

        deadline = time.monotonic() + TF_TIMEOUT
        while True:
            try:
                return tf_matrix(self.tf.lookup_transform(target, source, Time()))
            except TransformException as error:
                if time.monotonic() > deadline:
                    raise Failed(f"No TF {target} <- {source}: {error}") from None
            time.sleep(0.05)

    def world_point(self, header_frame, xyz):
        """World xyz of a point in a ROS frame, through the robot's Gazebo pose."""
        _, world_base = self.poses()
        base_frame = self.p["base_frame"]
        base_point = apply(self.transform(base_frame, header_frame), xyz)
        return apply(world_base, base_point)

    def base_point(self, world_xyz):
        _, world_base = self.poses()
        return apply(np.linalg.inv(world_base), world_xyz)

    def world_tcp(self):
        _, world_base = self.poses()
        return world_base @ self.transform(self.p["base_frame"], self.p["tcp_frame"])

    def ros_pose(self, world_xyz, yaw=0.0):
        """PoseStamped in the global frame of a world point."""
        from geometry_msgs.msg import PoseStamped

        _, world_base = self.poses()
        global_base = self.transform(self.p["global_frame"], self.p["base_frame"])
        T = global_base @ np.linalg.inv(world_base) @ pose_matrix(*world_xyz, yaw)
        out = PoseStamped()
        out.header.frame_id = self.p["global_frame"]
        out.header.stamp = self.node.get_clock().now().to_msg()
        out.pose.position.x, out.pose.position.y, out.pose.position.z = (
            float(v) for v in T[:3, 3]
        )
        heading = yaw_of(T)
        out.pose.orientation.z = math.sin(heading / 2)
        out.pose.orientation.w = math.cos(heading / 2)
        return out

    # -- held object ------------------------------------------------------------

    def follow(self):
        """Keep the held object at the TCP; runs in its own thread."""
        period = 1.0 / max(float(self.p["rate"]), 1.0)
        while self.node.context.ok():
            time.sleep(period)
            with self.lock:
                held = self.held
            if held is None:
                continue
            try:
                pose = held_pose(self.world_tcp(), held.yaw_offset)
            except Failed:
                continue
            last = held.sent
            if (
                last is not None
                and math.dist(pose[:3], last[:3]) < MOVED[0]
                and abs(math.remainder(pose[3] - last[3], math.tau)) < MOVED[1]
            ):
                continue
            with self.lock:
                if self.held is not held:
                    continue
                if self.set_pose(held.name, *pose):
                    held.sent = pose

    def attach(self, name, label, world_object):
        """Make the model a static copy that follows the TCP."""
        entry = self.object_list.get(label) or self.object_list.for_model(name)
        x, y, z = world_object[:3, 3]
        yaw = yaw_of(world_object)
        sdf = static_copy(self.sdf(name, entry.label, x, y, z, yaw))
        with self.lock:
            # Before the model changes: a failure leaves the object as it is.
            offset = yaw - yaw_of(self.world_tcp())
            self.remove(name)
            try:
                self.create(sdf, name)
            except Failed:
                # Put the object back as it was, to be picked again.
                try:
                    self.create(self.sdf(name, entry.label, x, y, z, yaw), name)
                except Failed as error:
                    self.log.warning(f"{name} is gone from Gazebo: {error}")
                raise
            self.held = Held(name, entry.label, offset, entry.size)

    def put(self, centre):
        """Let go of the held object: its static copy stands at `centre` (world
        xyz) until settle() puts it into the physics, after the hand left."""
        with self.lock:
            held, self.held = self.held, None
            if held is None:
                return
            yaw = held.sent[3] if held.sent else 0.0
            self.set_pose(held.name, *centre, yaw)
            self.placed = (held, centre, yaw)

    def settle(self):
        """The object let go by put() as the dynamic model at its spot."""
        with self.lock:
            placed, self.placed = self.placed, None
            if placed is None:
                return
            held, centre, yaw = placed
            self.remove(held.name)
            self.create(self.sdf(held.name, held.label, *centre, yaw), held.name)

    def after_release(self, step, name):
        """Run a Gazebo step once the object left the hand (settle, drop): a
        failure only makes a note, as the goal's part is done and a failed
        goal would leave the task believing the object still held."""
        try:
            step()
        except Failed as error:
            self.log.warning(f"{name} let go, but: {error}")
            return f"; let go, but {error}"
        return ""

    def drop(self):
        """Remove the held object from the world (a person took it)."""
        with self.lock:
            held, self.held = self.held, None
            if held is not None:
                self.remove(held.name)

    def release(self):
        """On exit: a held object back into the physics where it is."""
        self.settle()
        with self.lock:
            held = self.held
        if held is None:
            return
        poses, _ = self.poses()
        current = poses.get(held.name)
        if current is not None:
            self.put(tuple(current[:3, 3]))
            self.settle()

    # -- arm ------------------------------------------------------------------

    def move(self, lift, joints, cancelled, what):
        """Up before reaching out, in before going down: the lift and the arm
        one after the other, so that the arm does not sweep through a table."""
        if self.arm is None:
            return
        from aha_manipulation.pointing import ARM_JOINTS, LIFT_JOINT

        # The Arm raises TimeoutError (no /joint_states, no controller answer,
        # not finished in time): a failure like any other, so that the callers'
        # state (held, put, dropped) stays right.
        try:
            current = self.arm.positions([LIFT_JOINT, *ARM_JOINTS])
            if lift > current[0]:
                steps = [(lift, current[1:]), (lift, joints)]
            else:
                steps = [(current[0], joints), (lift, joints)]
            for step_lift, step_joints in steps:
                ok, text = self.arm.move(step_lift, step_joints, cancelled)
                if not ok:
                    # A cancel stops the arm where it is (STATUS_CANCELED of
                    # place and handover), unlike a failed motion.
                    stopped = cancelled is not None and cancelled.is_set()
                    status = "STATUS_CANCELED" if stopped else None
                    raise Failed(f"Arm ({what}): {text}", status, at_home=False)
        except TimeoutError as error:
            raise Failed(f"Arm ({what}): {error}", at_home=False) from None

    def reach_out(self, base_xyz, height, cancelled, what):
        """point(); a motion that fails other than by a cancel (which stops
        the arm where it is) brings the arm back first: a server ends a goal
        with the arm at home."""
        try:
            return self.point(base_xyz, height, cancelled, what)
        except Failed as error:
            if error.status != "STATUS_CANCELED":
                error.at_home = self.back(cancelled)
            raise

    def point(self, base_xyz, height, cancelled, what):
        """Stretch the arm towards a point in base_footprint, the hand level
        at `height`; False if it cannot."""
        if self.arm is None:
            return True
        from aha_manipulation.pointing import (
            EE_DROP,
            SHOULDER_XY,
            lift_for_height,
            point_at,
        )

        target = aim_point((*base_xyz[:2], height), SHOULDER_XY)
        try:
            pointing = point_at(target, lift_for_height(height + EE_DROP))
        except ValueError as error:
            self.log.warning(f"Cannot point the arm at the {what}: {error}")
            return False
        self.move(pointing.lift, pointing.arm, cancelled, what)
        return True

    def back(self, cancelled):
        """The arm to home (CARRY), with or without an object; True when there."""
        if self.arm is None:
            return True
        try:
            self.move(CARRY[0], CARRY[1], cancelled, "home")
        except Failed as error:
            self.log.warning(str(error))
            return False
        return True

    def wait(self, seconds, cancelled):
        """Wait on the simulation clock; False if cancelled."""
        clock = self.node.get_clock()
        start = last = clock.now().nanoseconds * 1e-9
        progress = time.monotonic()
        while (now := clock.now().nanoseconds * 1e-9) - start < seconds:
            if cancelled.is_set():
                return False
            if now > last:
                last, progress = now, time.monotonic()
            elif time.monotonic() - progress > CLOCK_STALL_TIMEOUT:
                return True
            time.sleep(0.05)
        return True

    # -- goals ----------------------------------------------------------------

    def check_arm(self, arm):
        if arm not in ("", "right"):
            raise Failed(f"The stand-in serves only the right arm, not {arm!r}")

    def pick(self, goal, handle, feedback):
        self.check_arm(goal.arm)
        if self.held is not None:
            raise Failed(f"Already holding {self.held.name}")
        target = goal.target
        p = target.pose.pose.position
        point = self.world_point(target.header.frame_id, (p.x, p.y, p.z))
        poses, world_base = self.poses()
        found = find_model(poses, self.object_list, target.class_name, point)
        if found is None:
            raise Failed(f"No object model within reach of {target.class_name}")
        name, _ = found
        world_object = poses[name]
        distance = horizontal_distance(world_base, world_object[:3, 3])
        if distance > self.param("reach"):
            raise Failed(
                f"{name} is {distance:.2f} m away (reach {self.param('reach')} m)"
            )
        cancelled = Flag(handle)
        feedback("approaching", 0.2)
        entry = self.object_list.for_model(name)
        above = world_object[2, 3] + entry.size[2] / 2 + HAND_ABOVE
        if not self.reach_out(
            self.base_point(world_object[:3, 3]), above, cancelled, name
        ):
            raise Failed(f"The arm cannot reach out over {name}")  # not moved
        if cancelled.is_set():
            raise Failed("Cancelled", at_home=False)
        feedback("grasping", 0.5)
        try:
            self.attach(name, entry.label, world_object)
        except Failed as error:  # remove / create / TF: the arm comes back first
            error.at_home = self.back(cancelled)
            raise
        feedback("retreating", 0.8)
        back, note = True, ""
        try:
            self.move(CARRY[0], CARRY[1], cancelled, "carry")
        except Failed as error:
            # The object is in the hand now: a failed pick would leave the task
            # without it, and every later pick refused (Already holding). The
            # arm is not at home (`at_home` false).
            back, note = False, f"; the arm is not at carry ({error})"
            self.log.warning(f"Picked {name}{note}")
        self.log.info(f"Picked {name}")
        try:
            pose = self.ros_pose(world_object[:3, 3], yaw_of(world_object))
        except Failed as error:  # the object is in the hand all the same
            pose, note = None, f"{note}; no grasp_pose ({error})"
        return name, back, pose, note

    def place(self, goal, handle, feedback):
        self.check_arm(goal.arm)
        if self.held is None:
            raise Failed("Nothing held", "STATUS_ERROR")
        if not goal.near.header.frame_id:
            raise Failed("The stand-in needs a near point", "STATUS_ERROR")
        n = goal.near.point
        near = self.world_point(goal.near.header.frame_id, (n.x, n.y, n.z))
        _, world_base = self.poses()
        distance = horizontal_distance(world_base, near)
        if distance > self.param("reach"):
            raise Failed(
                f"The spot is {distance:.2f} m away (reach {self.param('reach')} m)",
                "STATUS_UNREACHABLE",
            )
        cancelled = Flag(handle)
        feedback("approaching")
        size_z = self.held.size[2]
        centre = resting_centre(near, size_z, goal.drop)
        above = centre[2] + size_z / 2 + HAND_ABOVE
        container = self.furniture.get(goal.furniture) if goal.drop else None
        if container is not None:  # over its top, not into its side
            above = max(above, container.heights[-1] + size_z + HAND_ABOVE)
        if not self.reach_out(self.base_point(centre), above, cancelled, "spot"):
            raise Failed("The arm cannot reach out over the spot", "STATUS_UNREACHABLE")
        if cancelled.is_set():
            raise Failed("Cancelled", "STATUS_CANCELED", at_home=False)
        feedback("releasing")
        name = self.held.name
        self.put(centre)
        feedback("returning")
        back = self.back(cancelled)
        note = self.after_release(self.settle, name)
        self.log.info(f"Placed {name} at {[round(v, 3) for v in centre]}")
        bottom = (centre[0], centre[1], centre[2] - size_z / 2)
        try:
            pose = self.ros_pose(bottom)
        except Failed as error:
            pose, note = None, f"{note}; no place_pose ({error})"
        return name, back, pose, note

    def hand_over(self, goal, handle, feedback):
        self.check_arm(goal.arm)
        if self.held is None:
            raise Failed("Nothing held", "STATUS_ERROR")
        t = goal.target.point
        target = self.base_point(
            self.world_point(goal.target.header.frame_id, (t.x, t.y, t.z))
        )
        cancelled = Flag(handle)
        feedback("reaching")
        if not self.reach_out(target, target[2], cancelled, "person"):
            raise Failed("The arm cannot reach out there", "STATUS_UNREACHABLE")
        if cancelled.is_set():
            raise Failed("Cancelled", "STATUS_CANCELED", at_home=False)
        name = self.held.name
        if goal.release != "now":
            feedback("waiting")
            timeout = goal.timeout if goal.timeout > 0 else self.param("take_timeout")
            delay = self.param("take_delay") if self.param("take") else math.inf
            if not self.wait(min(delay, timeout), cancelled):
                raise Failed("Cancelled", "STATUS_CANCELED", at_home=False)
            if delay > timeout:
                feedback("returning")
                back = self.back(cancelled)
                raise Failed(
                    f"Nobody took {name} within {timeout:.0f} s", "STATUS_TIMEOUT", back
                )
        feedback("releasing")
        note = self.after_release(self.drop, name)
        feedback("returning")
        back = self.back(cancelled)
        self.log.info(f"Handed over {name}")
        return name, back, note


def execute_goal(stand, kind, what, run, handle, log):
    """Run one goal of action `kind`: `run(goal, handle, feedback, result)`
    fills in a success, or raises Failed. Any other error may come with the
    arm out (e.g. a model that cannot be made after reaching out): the arm
    goes back home first, and the goal fails with STATUS_EXECUTION_FAILED."""

    def feedback(phase, progress=None):
        message = kind.Feedback(phase=phase)
        if progress is not None:
            message.progress = float(progress)
        handle.publish_feedback(message)

    result = kind.Result()
    try:
        run(handle.request, handle, feedback, result)
    except Failed as error:
        log.warning(f"{what}: {error}")
        status, at_home, message = (
            failed_status(error),
            error.at_home is not False,
            str(error),
        )
    except Exception as error:
        log.error(f"{what}: {traceback.format_exc()}")
        status, message = "STATUS_EXECUTION_FAILED", f"{type(error).__name__}: {error}"
        at_home = stand.back(Flag(handle))
    else:
        result.success = True
        handle.succeed()
        return result
    if hasattr(result, "status"):  # not PickObject
        result.status = getattr(kind.Result, status)
    result.at_home, result.message = at_home, message
    if handle.is_cancel_requested:
        handle.canceled()
    else:
        handle.abort()
    return result


def main(argv=None):
    from aha_msgs.action import HandOver, PickObject, PlaceObject
    import rclpy
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.node import Node

    from aha_tasks.handling import HANDOVER_ACTION, PICK_ACTION, PLACE_ACTION

    rclpy.init(args=argv)
    node = Node("sim_manipulation")
    defaults = {
        "world": "",  # empty: the first world on the partition
        "robot_model": "aha_robot",
        "base_frame": "base_footprint",
        "global_frame": "map",
        "tcp_frame": "link_ree",
        "objects_config": "",
        "furniture_config": "",
        "cache": str(DEFAULT_CACHE),
        "reach": REACH,
        "move_arm": True,
        "take": True,
        "take_delay": 2.0,  # s (simulation clock)
        "take_timeout": 20.0,  # s, a goal without a timeout
        "rate": 20.0,  # Hz (wall) of the held object's updates
    }
    parameters = {k: node.declare_parameter(k, v).value for k, v in defaults.items()}
    stand = Stand(node, parameters)
    log = node.get_logger()
    group = ReentrantCallbackGroup()
    busy = threading.Lock()

    def accept(_):
        if not busy.acquire(blocking=False):
            return GoalResponse.REJECT
        busy.release()
        return GoalResponse.ACCEPT

    def serve(kind, what, run):
        def execute(handle):
            with busy:
                return execute_goal(stand, kind, what, run, handle, log)

        return execute

    def away(back):
        return "" if back else "; the arm is not at home"

    def pick(goal, handle, feedback, result):
        name, back, pose, note = stand.pick(goal, handle, feedback)
        result.message, result.at_home = f"Picked {name}{note}", back
        if pose is not None:
            result.grasp_pose_used = pose

    def place(goal, handle, feedback, result):
        name, back, pose, note = stand.place(goal, handle, feedback)
        result.status = PlaceObject.Result.STATUS_OK
        result.message, result.at_home = f"Placed {name}{away(back)}{note}", back
        if pose is not None:
            result.place_pose = pose

    def hand_over(goal, handle, feedback, result):
        name, back, note = stand.hand_over(goal, handle, feedback)
        result.status = HandOver.Result.STATUS_RELEASED
        result.message, result.at_home = f"Handed over {name}{away(back)}{note}", back

    for kind, name, what, run in (
        (PickObject, PICK_ACTION, "Pick", pick),
        (PlaceObject, PLACE_ACTION, "Place", place),
        (HandOver, HANDOVER_ACTION, "Handover", hand_over),
    ):
        ActionServer(
            node,
            kind,
            name,
            serve(kind, what, run),
            goal_callback=accept,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=group,
        )
    threading.Thread(target=stand.follow, daemon=True).start()
    log.info(
        f"Serving {PICK_ACTION}, {PLACE_ACTION}, {HANDOVER_ACTION} in {stand.world}"
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            stand.release()  # back into the physics where it is
        except Exception as error:  # Gazebo may be gone already
            log.warning(f"Held object not released: {error}")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
