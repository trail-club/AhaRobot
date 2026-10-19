"""Simulation only: put an object into a gripper through Gazebo's DetachableJoint.

The test bed URDF (aha_test_description, sim:=true) welds a model named
``aha_held_object_<arm>`` (link ``link``) to ``link_<r|l>6`` and listens on
``/aha/sim/held_object/<arm>/{attach,detach}``; ``.../state`` reports
"attached" / "detached". Plugin behaviour seen in Gazebo Harmonic (gz-sim
8.15):
  - the first model of that name after start is attached in the step it
    appears, at its relative pose at that moment;
  - a model respawned after a detach is attached only by the attach topic
    (the plugin looks it up by name again);
  - deleting the attached model publishes no state and the plugin stays
    attached; a respawned model then needs detach, then attach;
  - detach while detached and attach while attached do nothing.

``SimHolder.hold`` detaches and deletes the previous object, moves the arm to
a named state, opens the gripper, spawns the object (0.1 kg, friction 1.0 by
default) where ``place_geometry.object_in_tcp`` puts it for the grasp from a
TCP TF stamped after the arm stopped and the robot's Gazebo pose, attaches it
and closes the fingers to its width plus ``FINGER_MARGIN`` per side; with
``physical`` it squeezes the object and detaches it (friction holds it). The
held object's Gazebo pose is then checked against the hand and the hold is
redone if it is off. The object is a box or cylinder, or a mesh with that
bounding box (``ycb_mesh``: a tmc_wrs_gz_worlds YCB model with its origin
moved to the object center). The caller's rclpy node must not be
spun by another executor (this module spins it while waiting).
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import Buffer, TransformListener

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation.moveit_requests import joint_goal, named_states

MODEL_NAME = "aha_held_object_{arm}"
TOPIC = "/aha/sim/held_object/{arm}"
FINGER_MARGIN = 0.002  # gap per side between a finger and the object
# Finger target inside the object per side (physical hold); 4 mm threw the
# object out of the hand (penetration forces of the position-controlled
# fingers).
SQUEEZE = 0.0015
GRIP_SETTLE = 2.0  # sim s for the fingers to close
HOLD_SETTLE = 1.0  # sim s before the held object pose is checked
HOLD_TOLERANCE = 0.01  # m, held object from its grasp pose


def model_name(arm: str) -> str:
    return MODEL_NAME.format(arm=arm)


def _inertia(shape: str, size, mass: float) -> tuple[float, float, float]:
    x, y, z = size
    if shape == "cylinder":
        r = max(x, y) / 2
        side = mass * (3 * r * r + z * z) / 12
        return side, side, mass * r * r / 2
    return (
        mass * (y * y + z * z) / 12,
        mass * (x * x + z * z) / 12,
        mass * (x * x + y * y) / 12,
    )


@dataclass(frozen=True)
class Mesh:
    """Mesh of an object (e.g. a YCB model): Gazebo URIs of the visual and
    the collision mesh (empty: collide the box or cylinder), the mesh origin
    in the object frame (center of the bounding box) and the mass (kg)."""

    uri: str
    offset: tuple[float, float, float]
    collision: str = ""
    mass: float = 0.0


def ycb_mesh(model_dir: Path, height: float, collision: bool = True) -> Mesh:
    """Mesh of a tmc_wrs_gz_worlds YCB model (``model-1_4.sdf``, recentred:
    the origin is the footprint center on the base plane) with its origin
    moved to the object center ``height`` / 2 above the base; ``collision``
    false collides the box or cylinder instead of the collision mesh."""
    root = ET.parse(Path(model_dir) / "model-1_4.sdf").getroot()
    visual = root.find(".//visual")
    pose = visual.find("pose")
    xyz = [float(v) for v in pose.text.split()[:3]] if pose is not None else [0.0] * 3
    mesh = root.find(".//collision/geometry/mesh/uri")
    mass = root.find(".//inertial/mass")
    return Mesh(
        visual.find("geometry/mesh/uri").text.strip(),
        (xyz[0], xyz[1], xyz[2] - height / 2),
        mesh.text.strip() if collision and mesh is not None else "",
        float(mass.text) if mass is not None else 0.0,
    )


def object_sdf(
    name: str,
    shape: str,
    size,
    mass: float = 0.1,
    mu: float = 1.0,
    mesh: Mesh | None = None,
) -> str:
    """SDF of a box (x, y, z) or an upright cylinder (diameter max(x, y),
    height z) with its origin at the center; with ``mesh`` that mesh (its
    bounding box ``size``) instead, the inertia still of the shape."""
    if shape == "cylinder":
        geometry = f"<cylinder><radius>{max(size[0], size[1]) / 2}</radius><length>{size[2]}</length></cylinder>"
    elif shape == "box":
        geometry = f"<box><size>{size[0]} {size[1]} {size[2]}</size></box>"
    else:
        raise ValueError(f"unknown shape {shape!r}")
    ixx, iyy, izz = _inertia(shape, size, mass)
    visual = collision = f"<geometry>{geometry}</geometry>"
    material = (
        "<material><ambient>0.9 0.4 0.1 1</ambient>"
        "<diffuse>0.9 0.4 0.1 1</diffuse></material>"
    )
    if mesh is not None:
        pose = "<pose>{} {} {} 0 0 0</pose>".format(*mesh.offset)
        visual = f"{pose}<geometry><mesh><uri>{mesh.uri}</uri></mesh></geometry>"
        if mesh.collision:
            collision = (
                f"{pose}<geometry><mesh><uri>{mesh.collision}</uri></mesh></geometry>"
                "<max_contacts>10</max_contacts>"
            )
        material = ""
    return (
        f"<sdf version='1.9'><model name='{name}'><link name='link'>"
        f"<inertial><mass>{mass}</mass><inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy>"
        f"<izz>{izz}</izz></inertia></inertial>"
        f"<collision name='collision'>{collision}<surface><friction>"
        f"<ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface></collision>"
        f"<visual name='visual'>{visual}{material}</visual>"
        "</link></model></sdf>"
    )


class GzSim:
    """gz-transport access to the running world (GZ_PARTITION from the environment).

    A blocking service request from the Python binding times out while
    subscription callbacks are pending (pose/info is busy), even after the
    request was carried out, so requests go through the ``gz service`` CLI and
    poses are read with a short-lived subscription.
    """

    def __init__(self, world: str | None = None, timeout: float = 15.0):
        from gz.msgs10.stringmsg_pb2 import StringMsg
        from gz.transport13 import Node

        self.node = Node()
        self.world = world or self._find_world(timeout)
        self.lock = threading.Lock()
        self.poses = {}
        self.pose_stamp = 0
        self.states = {"right": [], "left": []}
        self.publishers = {}
        for arm in self.states:
            self.node.subscribe(
                StringMsg, TOPIC.format(arm=arm) + "/state", self._state_callback(arm)
            )

    def _find_world(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for topic in self.node.topic_list():
                parts = topic.split("/")
                if len(parts) == 4 and parts[1] == "world" and parts[3] == "clock":
                    return parts[2]
            time.sleep(0.5)
        raise RuntimeError("no Gazebo world found (check GZ_PARTITION)")

    def _on_poses(self, msg):
        poses = {}
        for pose in msg.pose:
            q = pose.orientation
            R = geo.matrix_from_quaternion(q.x, q.y, q.z, q.w)
            poses[pose.name] = geo.transform(
                R, [pose.position.x, pose.position.y, pose.position.z]
            )
        with self.lock:
            self.poses = poses
            self.pose_stamp += 1

    def _state_callback(self, arm):
        def callback(msg):
            with self.lock:
                self.states[arm].append(msg.data)

        return callback

    def model_poses(self, timeout: float = 5.0) -> dict:
        """World poses (4x4) by name from one pose/info message (models are top level)."""
        from gz.msgs10.pose_v_pb2 import Pose_V

        topic = f"/world/{self.world}/pose/info"
        with self.lock:
            start = self.pose_stamp
        self.node.subscribe(Pose_V, topic, self._on_poses)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                with self.lock:
                    if self.pose_stamp > start:
                        return dict(self.poses)
                time.sleep(0.02)
        finally:
            self.node.unsubscribe(topic)
        raise RuntimeError(f"no message on {topic}")

    def model_pose(self, name: str, timeout: float = 5.0):
        """Latest world pose (4x4) of a top-level model, or None."""
        return self.model_poses(timeout).get(name)

    def _request(self, service: str, request, timeout_ms: int = 5000) -> bool:
        from google.protobuf import text_format

        command = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world}/{service}",
            "--reqtype",
            f"gz.msgs.{type(request).__name__}",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(timeout_ms),
            "--req",
            text_format.MessageToString(request, as_one_line=True),
        ]
        try:
            out = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_ms / 1000 + 10
            )
        except subprocess.TimeoutExpired:
            return False
        return "data: true" in out.stdout

    def spawn(self, sdf: str, T: np.ndarray, timeout_ms: int = 5000) -> bool:
        """Create a model; also true if the reply timed out but the model
        appeared (loading a mesh can take longer than the service reply)."""
        from gz.msgs10.entity_factory_pb2 import EntityFactory

        request = EntityFactory()
        request.sdf = sdf
        _set_pose(request.pose, T)
        if self._request("create", request, timeout_ms):
            return True
        name = ET.fromstring(sdf).find("model").get("name")
        return self.model_pose(name) is not None

    def remove(self, name: str) -> bool:
        from gz.msgs10.entity_pb2 import Entity

        request = Entity()
        request.name = name
        request.type = Entity.MODEL
        return self._request("remove", request)

    def set_physics_step(self, step: float) -> bool:
        """Set the world's physics step (s) and an update rate of 1 / step
        (the real-time cap stays 1)."""
        from gz.msgs10.physics_pb2 import Physics

        request = Physics()
        request.max_step_size = step
        request.real_time_update_rate = 1.0 / step
        request.real_time_factor = 1.0
        return self._request("set_physics", request)

    def publish_empty(self, topic: str):
        from gz.msgs10.empty_pb2 import Empty

        if topic not in self.publishers:
            self.publishers[topic] = self.node.advertise(topic, Empty)
            time.sleep(0.5)  # let subscribers connect
        self.publishers[topic].publish(Empty())

    def state_count(self, arm: str) -> int:
        with self.lock:
            return len(self.states[arm])

    def wait_state(self, arm: str, value: str, since: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if value in self.states[arm][since:]:
                    return True
            time.sleep(0.05)
        return False

    def release(self, arm: str):
        """Detach (no-op when nothing is attached) and delete the held object model."""
        self.publish_empty(TOPIC.format(arm=arm) + "/detach")
        time.sleep(0.3)
        name = model_name(arm)
        if self.model_pose(name) is None:
            return
        self.remove(name)
        deadline = time.monotonic() + 10.0
        while self.model_pose(name) is not None:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name} was not removed")
            time.sleep(0.2)


def _set_pose(msg, T):
    msg.position.x, msg.position.y, msg.position.z = (float(v) for v in T[:3, 3])
    x, y, z, w = geo.quaternion_from_matrix(T[:3, :3])
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (
        x,
        y,
        z,
        w,
    )


class SimHolder:
    """Put an object into a gripper of the simulated robot (see the module docstring)."""

    def __init__(
        self,
        node,
        gz: GzSim,
        robot_name: str = "aha_robot",
        base_frame="base_footprint",
    ):
        self.node = node
        self.gz = gz
        self.robot_name = robot_name
        self.base_frame = base_frame
        self.states = {}
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        node.create_subscription(
            String,
            "/robot_description_semantic",
            lambda msg: self.states.update(named_states(msg.data)),
            latched,
        )
        self.move_client = ActionClient(node, MoveGroup, "/move_action")
        self.grippers = {
            arm: node.create_publisher(
                Float64MultiArray, f"/{arm}_gripper_controller/commands", 10
            )
            for arm in ("right", "left")
        }
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

    def spin_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return predicate()

    def wait_clock(self, seconds: float, timeout: float = 120.0) -> bool:
        """Spin for ``seconds`` of the node's clock (sim time in Gazebo, which
        may run several times slower than the wall clock)."""
        start = self.node.get_clock().now().nanoseconds
        return self.spin_until(
            lambda: self.node.get_clock().now().nanoseconds - start >= seconds * 1e9,
            timeout,
        )

    def wait_future(self, future, timeout: float):
        if not self.spin_until(future.done, timeout):
            raise RuntimeError("timed out")
        return future.result()

    def named_state(self, arm: str, state: str) -> dict:
        group = f"{arm}_arm"
        if not self.spin_until(lambda: (group, state) in self.states, 10.0):
            raise RuntimeError(f"named state {state!r} of {group} not available")
        return dict(self.states[(group, state)])

    def move_named(
        self, arm: str, state: str, retries: int = 3, scaling: float = 0.5
    ) -> bool:
        return self.move_joints(arm, self.named_state(arm, state), retries, scaling)

    def move_joints(
        self, arm: str, joints: dict, retries: int = 3, scaling: float = 0.5
    ) -> bool:
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("/move_action not available")
        goal = joint_goal(f"{arm}_arm", joints, 3.0, scaling)
        for _ in range(retries):
            handle = self.wait_future(self.move_client.send_goal_async(goal), 10.0)
            if not handle.accepted:
                continue
            result = self.wait_future(handle.get_result_async(), 90.0).result
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return True
            self.node.get_logger().warn(
                f"{arm} arm: MoveIt error {result.error_code.val}"
            )
        return False

    def set_gripper(self, arm: str, position: float):
        self.grippers[arm].publish(Float64MultiArray(data=[position, -position]))

    def tcp_in_base(self, arm: str, after: int = 0) -> np.ndarray:
        """TCP pose in the base frame from a TF stamped at or after ``after``
        (node clock ns): this module spins one callback at a time, so queued
        /tf could otherwise give the hand of seconds ago (an object spawned
        there and attached floated below the fingers)."""
        tcp = f"link_{arm[0]}ee"
        latest = {}

        def fresh():
            if not self.tf_buffer.can_transform(self.base_frame, tcp, Time()):
                return False
            t = self.tf_buffer.lookup_transform(self.base_frame, tcp, Time())
            latest["t"] = t
            return Time.from_msg(t.header.stamp).nanoseconds >= after

        if not self.spin_until(fresh, 20.0):
            raise RuntimeError(f"no fresh TF {self.base_frame} -> {tcp}")
        t = latest["t"]
        q = t.transform.rotation
        p = t.transform.translation
        return geo.transform(
            geo.matrix_from_quaternion(q.x, q.y, q.z, q.w), [p.x, p.y, p.z]
        )

    def hold(
        self,
        arm: str,
        shape: str,
        size,
        grasp: str = geo.SIDE,
        state: str | None = "carry",
        tilt: float = geo.DEFAULT_TOP_TILT,
        mass: float = 0.1,
        mu: float = 1.0,
        physical: bool = False,
        attempts: int = 3,
        mesh: Mesh | None = None,
        hand: np.ndarray | None = None,
    ) -> np.ndarray:
        """Put a new object into the gripper; returns its world pose (4x4).

        Welded (DetachableJoint) by default. With ``physical`` the fingers
        squeeze it (``SQUEEZE`` per side) and the joint is released, so only
        friction holds it, as on the robot. The object pose is checked in
        Gazebo after ``HOLD_SETTLE`` s and the hold is redone (``attempts``) if
        it is off its grasp pose by more than ``HOLD_TOLERANCE``. ``hand``
        (T_tcp_object) puts it elsewhere in the hand than the grasp's nominal
        pose (e.g. slid down), as PlaceObject's ``object_in_hand``."""
        geo.check_grasp(grasp)
        width = max(size[0], size[1]) if shape == "cylinder" else size[1]
        if hand is None:
            hand = geo.object_in_tcp(grasp, size[2], tilt)
        error = math.inf
        for _ in range(attempts):
            self.gz.release(arm)
            if state and not self.move_named(arm, state):
                raise RuntimeError(f"could not move {arm} arm to {state}")
            self.set_gripper(arm, geo.FINGER_MAX_OPENING)
            self.spin_until(lambda: False, 1.0)
            after = self.node.get_clock().now().nanoseconds
            world_base = self.gz.model_pose(self.robot_name)
            if world_base is None:
                raise RuntimeError(f"no Gazebo pose for {self.robot_name}")
            tcp = self.tcp_in_base(arm, after)
            T_object = world_base @ tcp @ hand
            since = self.gz.state_count(arm)
            if not self.gz.spawn(
                object_sdf(model_name(arm), shape, size, mass, mu, mesh), T_object
            ):
                raise RuntimeError("spawning the object failed")
            self.gz.publish_empty(TOPIC.format(arm=arm) + "/attach")
            if not self.gz.wait_state(arm, "attached", since, 5.0):
                raise RuntimeError("the object was not attached")
            margin = -SQUEEZE if physical else FINGER_MARGIN
            self.set_gripper(arm, geo.finger_position(width, margin))
            # Sim time: at a real time factor of 0.2 one wall second left the
            # fingers half open and a released object fell out.
            self.wait_clock(GRIP_SETTLE)
            if physical:
                self.gz.publish_empty(TOPIC.format(arm=arm) + "/detach")
            self.wait_clock(HOLD_SETTLE)
            placed = self.gz.model_pose(model_name(arm))
            if placed is None:
                raise RuntimeError(f"{model_name(arm)} is gone")
            # Against the hand now (world base again: the hand may have moved).
            expected = (
                self.gz.model_pose(self.robot_name)
                @ self.tcp_in_base(arm, self.node.get_clock().now().nanoseconds)
                @ hand
            )
            error = float(np.linalg.norm(placed[:3, 3] - expected[:3, 3]))
            if error <= HOLD_TOLERANCE:
                return placed
            self.node.get_logger().warn(
                f"held object {error * 100:.1f} cm off its grasp pose; holding again"
            )
        raise RuntimeError(f"held object {error * 100:.1f} cm off its grasp pose")


def tilt_deg(T: np.ndarray) -> float:
    return math.degrees(geo.tilt_from_vertical(T[:3, :3]))
