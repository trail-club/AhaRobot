"""Simulation only: pick, place and handover servers that move objects in Gazebo.

Stands in for the manipulation squad's servers in debug end-to-end runs
(task.launch.py e2e:=debug), as sim_teleport does for Nav2. No grasp is
planned; the object is moved by gz-transport (aha_tasks.sim_manipulation):

- pick (aha_msgs/action/PickObject, /aha/manipulation/pick): the object
  model of the target's label nearest the target (MATCH_DISTANCE), within
  `reach` of the base. The right arm reaches out over it, the model becomes
  a static copy without collision that follows the TCP (`tcp_frame`), and
  the arm goes to the carry pose. Once the object is in the hand the pick
  succeeds, also when the arm does not get to carry (cancel, controller):
  the message says so.
- place (aha_msgs/action/PlaceObject, /aha/manipulation/place): the arm
  reaches out over `near` (within `reach`; over the top of a container to
  drop into), the static copy is set standing at `near` (`drop`: its centre
  at `near`), the arm goes back, and the dynamic model is created there,
  clear of the hand (a dropped object falls).
- handover (aha_msgs/action/HandOver, /aha/manipulation/handover): the arm
  points at the target at its height; with release "take" the simulated
  person takes the object `take_delay` s later (never if `take` is false,
  then the goal times out); the model is removed and the arm goes back.

The arm moves in straight lines in joint space (aha_tasks.gestures.Arm,
aha_manipulation.pointing) without collision checks, the lift before the
arm when going up and after it when going down; the hand passes HAND_ABOVE
over the objects it reaches for. `move_arm` false skips the motions. Only the right arm is served. Model poses come from
Gazebo's pose/info, targets in `map` go to the world through the robot's
Gazebo pose and the TF map -> base_footprint (the detections' path in
sim_object_detector.py). Waits are on the simulation clock. On exit, held
objects are put back into the physics where they are.
"""

import math
from pathlib import Path
import threading
import time

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
    """A goal fails with this message (and the status in `status`)."""

    def __init__(self, message, status=None, returned=False):
        super().__init__(message)
        self.status, self.returned = status, returned


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
        self.rest = None  # (lift, arm joints) before the pick
        self.busy = False
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

        if not self.request("remove", Entity(name=name, type=Entity.MODEL), Boolean):
            raise Failed(f"Could not remove {name}")
        deadline = time.monotonic() + REMOVE_TIMEOUT
        while time.monotonic() < deadline:
            poses, age = self.gazebo.latest()
            if name not in poses and age < 0.5:
                return
            time.sleep(0.1)
        raise Failed(f"{name} stays in Gazebo after its removal")

    def create(self, sdf, name):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.entity_factory_pb2 import EntityFactory

        for _ in range(3):
            if self.request("create", EntityFactory(sdf=sdf), Boolean):
                return
            time.sleep(0.3)
        raise Failed(f"Could not create {name}")

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
            self.remove(name)
            self.create(sdf, name)
            offset = yaw - yaw_of(self.world_tcp())
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

    def remember_rest(self):
        if self.arm is None or self.rest is not None:
            return
        from aha_manipulation.pointing import ARM_JOINTS, LIFT_JOINT

        values = self.arm.positions([LIFT_JOINT, *ARM_JOINTS])
        self.rest = (values[0], tuple(values[1:]))

    def move(self, lift, joints, cancelled, what):
        """Up before reaching out, in before going down: the lift and the arm
        one after the other, so that the arm does not sweep through a table."""
        if self.arm is None:
            return
        from aha_manipulation.pointing import ARM_JOINTS, LIFT_JOINT

        current = self.arm.positions([LIFT_JOINT, *ARM_JOINTS])
        if lift > current[0]:
            steps = [(lift, current[1:]), (lift, joints)]
        else:
            steps = [(current[0], joints), (lift, joints)]
        for step_lift, step_joints in steps:
            ok, text = self.arm.move(step_lift, step_joints, cancelled)
            if not ok:
                raise Failed(f"Arm ({what}): {text}")

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
        """The arm to the carry pose while it holds an object, else to its pose
        before the pick; True when there."""
        if self.arm is None:
            return True
        holding = self.held is not None
        if not holding and self.rest is None:
            return True
        lift, joints = CARRY if holding else self.rest
        try:
            self.move(lift, joints, cancelled, "back")
        except Failed as error:
            self.log.warning(str(error))
            return False
        if not holding:
            self.rest = None
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
        self.remember_rest()
        entry = self.object_list.for_model(name)
        above = world_object[2, 3] + entry.size[2] / 2 + HAND_ABOVE
        self.point(self.base_point(world_object[:3, 3]), above, cancelled, name)
        if cancelled.is_set():
            raise Failed("Cancelled")
        feedback("grasping", 0.5)
        self.attach(name, entry.label, world_object)
        feedback("retreating", 0.8)
        note = ""
        try:
            self.move(CARRY[0], CARRY[1], cancelled, "carry")
        except Failed as error:
            # The object is in the hand now: a failed pick would leave the task
            # without it, and every later pick refused (Already holding).
            note = f"; the arm is not at carry ({error})"
            self.log.warning(f"Picked {name}{note}")
        self.log.info(f"Picked {name}")
        return name, self.ros_pose(world_object[:3, 3], yaw_of(world_object)), note

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
        self.point(self.base_point(centre), above, cancelled, "spot")
        if cancelled.is_set():
            raise Failed("Cancelled", "STATUS_CANCELED")
        feedback("releasing")
        name = self.held.name
        self.put(centre)
        feedback("returning")
        back = self.back(cancelled)
        self.settle()
        self.log.info(f"Placed {name} at {[round(v, 3) for v in centre]}")
        bottom = (centre[0], centre[1], centre[2] - size_z / 2)
        return name, back, self.ros_pose(bottom)

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
        if not self.point(target, target[2], cancelled, "person"):
            raise Failed("The arm cannot reach out there", "STATUS_UNREACHABLE")
        if cancelled.is_set():
            raise Failed("Cancelled", "STATUS_CANCELED")
        name = self.held.name
        if goal.release != "now":
            feedback("waiting")
            timeout = goal.timeout if goal.timeout > 0 else self.param("take_timeout")
            delay = self.param("take_delay") if self.param("take") else math.inf
            if not self.wait(min(delay, timeout), cancelled):
                raise Failed("Cancelled", "STATUS_CANCELED")
            if delay > timeout:
                feedback("returning")
                back = self.back(cancelled)
                raise Failed(
                    f"Nobody took {name} within {timeout:.0f} s", "STATUS_TIMEOUT", back
                )
        feedback("releasing")
        self.drop()
        feedback("returning")
        back = self.back(cancelled)
        self.log.info(f"Handed over {name}")
        return name, back


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

    def serve(run):
        def execute(handle):
            with busy:
                return run(handle)

        return execute

    def publish(handle, kind):
        def feedback(phase, progress=None):
            message = kind.Feedback(phase=phase)
            if progress is not None:
                message.progress = float(progress)
            handle.publish_feedback(message)

        return feedback

    def finish(handle, result, ok, cancelled=False):
        if cancelled:
            handle.canceled()
        elif ok:
            handle.succeed()
        else:
            handle.abort()
        return result

    def run_pick(handle):
        result = PickObject.Result()
        try:
            name, pose, note = stand.pick(
                handle.request, handle, publish(handle, PickObject)
            )
            result.success, result.message = True, f"Picked {name}{note}"
            result.grasp_pose_used = pose
            return finish(handle, result, True)
        except Failed as error:
            result.message = str(error)
            log.warning(f"Pick: {error}")
            return finish(handle, result, False, handle.is_cancel_requested)

    def run_place(handle):
        R = PlaceObject.Result
        result = R()
        try:
            name, back, pose = stand.place(
                handle.request, handle, publish(handle, PlaceObject)
            )
            result.success, result.status = True, R.STATUS_OK
            result.message = f"Placed {name}" + (
                "" if back else "; the arm is not back"
            )
            result.place_pose = pose
            return finish(handle, result, True)
        except Failed as error:
            result.status = getattr(R, error.status or "STATUS_EXECUTION_FAILED")
            result.message = str(error)
            log.warning(f"Place: {error}")
            return finish(handle, result, False, handle.is_cancel_requested)

    def run_handover(handle):
        R = HandOver.Result
        result = R()
        try:
            name, back = stand.hand_over(
                handle.request, handle, publish(handle, HandOver)
            )
            result.success, result.status = True, R.STATUS_RELEASED
            result.message = f"Handed over {name}" + (
                "" if back else "; the arm is not back"
            )
            result.returned = back
            return finish(handle, result, True)
        except Failed as error:
            result.status = getattr(R, error.status or "STATUS_ERROR")
            result.returned = error.returned
            result.message = str(error)
            log.warning(f"Handover: {error}")
            return finish(handle, result, False, handle.is_cancel_requested)

    for kind, name, run in (
        (PickObject, PICK_ACTION, run_pick),
        (PlaceObject, PLACE_ACTION, run_place),
        (HandOver, HANDOVER_ACTION, run_handover),
    ):
        ActionServer(
            node,
            kind,
            name,
            serve(run),
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
