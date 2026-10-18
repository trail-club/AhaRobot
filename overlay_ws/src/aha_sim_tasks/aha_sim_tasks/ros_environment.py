"""Evaluator-side ROS adapter; only public observations cross policy IPC."""

import math
import json
import time
from collections import deque
from functools import partial
from types import MappingProxyType

from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformException, TransformListener

from .api import (
    Action,
    CameraObservation,
    clamp_joint_roundoff,
    CONTROLLER_JOINTS,
    GRIPPER_JOINTS,
    GroundTruth,
    JOINT_LIMITS,
    Observation,
)
from .evaluation import WorldState


def yaw(quaternion):
    return math.atan2(
        2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1 - 2 * (quaternion.y**2 + quaternion.z**2),
    )


class RosEnvironment(Node):
    def __init__(
        self,
        task,
        head_image_topic="",
        *,
        cameras=False,
        camera_timeout=10.0,
        ground_truth=False,
        video_recorder=None,
        sensor_model=None,
        sensor_seed=0,
    ):
        if ground_truth and (cameras or head_image_topic):
            raise ValueError("Ground-truth IK does not consume sensor observations")
        super().__init__(
            "task_evaluator",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.task = task
        self.truth_enabled = ground_truth
        self.ground_truth = None
        self.positions = {}
        self.velocities = {}
        self.odom = None
        self.image = None
        self.cameras_enabled = cameras
        self.camera_timeout = camera_timeout
        self.sensor_model = sensor_model or {}
        self.sensor_seed = sensor_seed
        self.camera_samples = {}
        self.camera_raw_stamps = {}
        self.video_recorder = video_recorder
        self.camera_filters = []
        self.world = None
        self.received = {}
        self.previous_object = None
        self.previous_robot = None
        self.last_targets = {}
        self.last_trajectory_times = {}
        self.active_controllers = set()
        self.controller_future = None
        self.controller_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        self.create_timer(0.5, self.poll_controllers)
        self.base = self.create_publisher(
            TwistStamped, "/diff_drive_controller/cmd_vel", 10
        )
        self.trajectories = {
            name: self.create_publisher(
                JointTrajectory, f"/{name}/joint_trajectory", 10
            )
            for name in CONTROLLER_JOINTS
        }
        self.grippers = {
            name: self.create_publisher(Float64MultiArray, f"/{name}/commands", 10)
            for name in GRIPPER_JOINTS
        }
        if ground_truth:
            self.truth_joints = self.create_publisher(
                JointState, "/evaluation/joint_states", 10
            )
        if not ground_truth:
            self.create_subscription(
                JointState, "/joint_states", self.on_joints, qos_profile_sensor_data
            )
            self.create_subscription(
                Odometry,
                "/diff_drive_controller/odom",
                self.on_odom,
                qos_profile_sensor_data,
            )
        self.create_subscription(String, "/evaluation/state", self.on_state, 10)
        if head_image_topic:
            self.create_subscription(
                Image, head_image_topic, self.on_image, qos_profile_sensor_data
            )
        self.required = {"state"} if ground_truth else {"joints", "odom", "state"}
        if head_image_topic:
            self.required.add("image")
        if cameras:
            self.tf_buffer = Buffer(node=self)
            self.tf_listener = TransformListener(self.tf_buffer, self)
            for name, topics in {
                "head": (
                    (Image, "/camera/color/image_raw"),
                    (CameraInfo, "/camera/color/camera_info"),
                    (Image, "/camera/depth_registered/image_rect"),
                ),
                "left_wrist": (
                    (Image, "/camera/left_wrist/image_raw"),
                    (CameraInfo, "/camera/left_wrist/camera_info"),
                ),
                "right_wrist": (
                    (Image, "/camera/right_wrist/image_raw"),
                    (CameraInfo, "/camera/right_wrist/camera_info"),
                ),
            }.items():
                # Images use reliable streams in the camera bridge.
                # Best-effort readers can lose different frames from each stream
                # during a physics burst, leaving no exact pair after pausing.
                camera_qos = QoSProfile(depth=10)
                subscribers = [
                    Subscriber(self, message_type, topic, qos_profile=camera_qos)
                    for message_type, topic in topics
                ]
                for subscriber, (_, topic) in zip(subscribers, topics):
                    subscriber.registerCallback(
                        partial(self.on_camera_component, topic)
                    )
                synchronizer = TimeSynchronizer(subscribers, queue_size=10)
                synchronizer.registerCallback(partial(self.on_camera, name))
                self.camera_filters.append((subscribers, synchronizer))
                self.required.add("camera:" + name)

    def mark(self, source):
        self.received[source] = time.monotonic()

    def poll_controllers(self):
        if self.controller_future is not None:
            if not self.controller_future.done():
                return
            response = self.controller_future.result()
            self.active_controllers = {
                controller.name
                for controller in response.controller
                if controller.state == "active"
            }
            self.controller_future = None
        if self.controller_client.service_is_ready():
            self.controller_future = self.controller_client.call_async(
                ListControllers.Request()
            )

    def on_joints(self, message):
        self.positions = dict(zip(message.name, message.position))
        self.velocities = dict(zip(message.name, message.velocity))
        self.mark("joints")

    def on_odom(self, message):
        self.odom = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw(message.pose.pose.orientation),
        )
        self.mark("odom")

    def on_image(self, message):
        self.image = message
        self.mark("image")

    def on_camera(self, name, image, camera_info, depth_image=None):
        if depth_image is not None and getattr(self, "sensor_model", {}):
            from .sensor_noise import noisy_depth

            depth_image = noisy_depth(depth_image, self.sensor_model, self.sensor_seed)
        messages = [image, camera_info]
        if depth_image is not None:
            messages.append(depth_image)
        if any(message.header != image.header for message in messages):
            return
        samples = self.camera_samples.setdefault(name, deque(maxlen=10))
        if samples and Time.from_msg(image.header.stamp) <= Time.from_msg(
            samples[-1][0].header.stamp
        ):
            return
        # Retain more than the 0.5 s age window at 15 Hz while TF catches up.
        samples.append((image, camera_info, depth_image))
        self.mark("camera:" + name)

    def on_camera_component(self, topic, message):
        self.camera_raw_stamps[topic] = (
            Time.from_msg(message.header.stamp).nanoseconds * 1e-9
        )
        if self.video_recorder is not None:
            self.video_recorder.image(topic, message)

    def camera_observations(self):
        if not self.cameras_enabled:
            return {}
        if len(self.camera_samples) != 3:
            return None
        now = self.get_clock().now().nanoseconds

        def available_samples(samples):
            for image, info, depth in reversed(samples):
                stamp = Time.from_msg(image.header.stamp)
                if not -100_000_000 <= now - stamp.nanoseconds <= 500_000_000:
                    continue
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "base_link", image.header.frame_id, stamp
                    )
                except TransformException:
                    continue
                yield (
                    stamp.nanoseconds,
                    CameraObservation(image, info, transform, depth),
                )

        candidates = {
            name: available_samples(samples)
            for name, samples in self.camera_samples.items()
        }
        selected = {name: next(samples, None) for name, samples in candidates.items()}
        while all(sample is not None for sample in selected.values()):
            oldest = min(stamp for stamp, _ in selected.values())
            ahead = [
                name
                for name, (stamp, _) in selected.items()
                if stamp - oldest > 100_000_000
            ]
            if not ahead:
                return {name: camera for name, (_, camera) in selected.items()}
            # Only advance views too far ahead of the oldest available sample.
            # Each view is searched newest-first without a Cartesian bundle search.
            for name in ahead:
                selected[name] = next(candidates[name], None)
        return None

    def observation_ready(self):
        return self.camera_observations() is not None

    def camera_diagnostics(self):
        """Explain unavailable timestamped camera observations at a paused boundary."""
        now = self.get_clock().now().nanoseconds * 1e-9
        result = {"sim_time": now, "raw_stamps": self.camera_raw_stamps, "cameras": {}}
        for name, samples in self.camera_samples.items():
            latest = []
            for image, _, _ in list(samples)[-3:]:
                stamp = Time.from_msg(image.header.stamp)
                data = {
                    "stamp": stamp.nanoseconds * 1e-9,
                    "age": now - stamp.nanoseconds * 1e-9,
                }
                try:
                    self.tf_buffer.lookup_transform(
                        "base_link", image.header.frame_id, stamp
                    )
                    data["tf"] = "available"
                except TransformException as error:
                    data["tf"] = str(error)
                latest.append(data)
            result["cameras"][name] = latest
        return result

    def on_state(self, message):
        sample = json.loads(message.data)
        if sample["schema_version"] not in (1, 2) or sample["frame_id"] != "world":
            raise ValueError("Unsupported evaluation state schema or frame")
        stamp = sample["stamp"]
        sim_time = stamp["sec"] + stamp["nanosec"] * 1e-9
        position = tuple(sample["object_position"])
        robot_pose = tuple(sample["robot_pose"])
        contacts = sample["finger_contacts"]
        orientation = tuple(sample.get("object_orientation", (0, 0, 0, 1)))
        destination_contact = sample.get("destination_contact", False)
        simultaneous = sample.get("simultaneous_grasp", contacts == 2)
        if (
            sample["schema_version"] == 2
            and not {"object_orientation", "destination_contact", "simultaneous_grasp"}
            <= sample.keys()
        ):
            raise ValueError("Incomplete evaluation state")
        if (
            len(position) != 3
            or len(robot_pose) != 3
            or type(contacts) is not int
            or not 0 <= contacts <= 2
            or len(orientation) != 4
            or not all(math.isfinite(v) for v in orientation)
            or abs(sum(v * v for v in orientation) - 1) > 1e-3
            or type(destination_contact) is not bool
            or type(simultaneous) is not bool
            or not all(
                math.isfinite(value) for value in (sim_time, *position, *robot_pose)
            )
        ):
            raise ValueError("Invalid evaluation state")
        # Reject delayed/duplicate samples before changing any speed history.
        if self.world is not None and sim_time <= self.world.sim_time:
            return
        truth = None
        if getattr(self, "truth_enabled", False):
            payload = sample.get("ground_truth")
            if payload is None:
                raise ValueError("Gazebo ground truth is missing")
            pose = tuple(payload["robot_world_pose"])
            positions = payload["joint_positions"]
            if (
                len(pose) != 7
                or not all(
                    math.isfinite(value) for value in (*pose, *positions.values())
                )
                or abs(sum(value * value for value in pose[3:]) - 1) > 1e-3
            ):
                raise ValueError("Invalid Gazebo ground truth")
            # Physics may not have filled the newly requested joint components yet.
            if not JOINT_LIMITS.keys() <= positions.keys():
                return
            truth = GroundTruth(pose, position, contacts)
        speed = math.inf
        if self.previous_object is not None:
            previous_time, previous_position = self.previous_object
            if sim_time > previous_time:
                speed = math.dist(position, previous_position) / (
                    sim_time - previous_time
                )
        angular_speed = math.inf
        previous_orientation = getattr(self, "previous_orientation", None)
        if previous_orientation is not None:
            previous_time, previous_q = previous_orientation
            cosine = min(1.0, abs(sum(a * b for a, b in zip(previous_q, orientation))))
            angular_speed = 2 * math.acos(cosine) / (sim_time - previous_time)
        self.previous_orientation = (sim_time, orientation)
        self.previous_object = (sim_time, position)
        robot_xy = robot_pose[:2]
        robot_speed = math.inf
        if self.previous_robot is not None:
            previous_time, previous_xy = self.previous_robot
            if sim_time > previous_time:
                robot_speed = math.dist(robot_xy, previous_xy) / (
                    sim_time - previous_time
                )
        self.previous_robot = (sim_time, robot_xy)
        self.world = WorldState(
            sim_time,
            robot_pose,
            position,
            speed,
            contacts,
            robot_speed,
            orientation,
            destination_contact,
            angular_speed,
            simultaneous,
        )
        if truth is not None:
            self.ground_truth = truth
            self.positions = dict(positions)
            self.velocities = {}
            self.odom = robot_pose
            joint_message = JointState(
                name=list(positions),
                position=[
                    clamp_joint_roundoff(name, value)
                    for name, value in positions.items()
                ],
            )
            joint_message.header.stamp.sec = stamp["sec"]
            joint_message.header.stamp.nanosec = stamp["nanosec"]
            self.truth_joints.publish(joint_message)
        self.mark("state")

    def ready(self):
        return (
            self.fresh()
            and {
                "joint_state_broadcaster",
                "diff_drive_controller",
                *CONTROLLER_JOINTS,
                *GRIPPER_JOINTS,
            }.issubset(self.active_controllers)
            and all(name in self.positions for name in JOINT_LIMITS)
            and self.base.get_subscription_count() > 0
            and all(
                pub.get_subscription_count() > 0 for pub in self.trajectories.values()
            )
            and all(pub.get_subscription_count() > 0 for pub in self.grippers.values())
            and self.world is not None
            and math.isfinite(self.world.object_speed)
            and abs(self.get_clock().now().nanoseconds * 1e-9 - self.world.sim_time)
            <= 0.2
            and self.observation_ready()
        )

    def stale_sources(self):
        now = time.monotonic()
        return {
            source: round(now - self.received.get(source, -math.inf), 3)
            for source in self.required
            if now - self.received.get(source, -math.inf)
            >= (self.camera_timeout if source.startswith("camera:") else 2)
        }

    def fresh(self):
        return not self.stale_sources()

    def observation(self):
        cameras = self.camera_observations()
        if cameras is None:
            raise RuntimeError(
                "Camera images or their timestamped transforms are unavailable"
            )
        return Observation(
            self.world.sim_time
            if self.truth_enabled
            else self.get_clock().now().nanoseconds * 1e-9,
            self.task.task_id,
            self.task.instruction,
            MappingProxyType(dict(self.positions)),
            MappingProxyType(dict(self.velocities)),
            self.odom,
            cameras["head"].image if cameras else self.image,
            MappingProxyType(cameras),
            self.ground_truth,
        )

    def apply(self, action):
        if not isinstance(action, Action):
            raise TypeError("policy.act must return aha_sim_tasks.api.Action")
        action.validate()
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        sim_time = command.header.stamp.sec + command.header.stamp.nanosec * 1e-9
        command.twist.linear.x = action.linear_velocity
        command.twist.angular.z = action.angular_velocity
        self.base.publish(command)
        for controller, names in CONTROLLER_JOINTS.items():
            if not any(name in action.joint_positions for name in names):
                continue
            # Gazebo feedback can stray just outside a joint bound numerically.
            # Clamp held positions; explicit policy commands were validated above.
            targets = tuple(
                action.joint_positions.get(
                    name,
                    max(
                        JOINT_LIMITS[name][0],
                        min(
                            JOINT_LIMITS[name][1],
                            self.last_targets.get(name, self.positions[name]),
                        ),
                    ),
                )
                for name in names
            )
            unchanged = all(
                self.last_targets.get(name) == target
                for name, target in zip(names, targets)
            )
            if unchanged:
                reached = all(
                    abs(self.positions[name] - target) <= 0.005
                    for name, target in zip(names, targets)
                )
                if reached or sim_time - self.last_trajectory_times[controller] < 3.0:
                    continue
            trajectory = JointTrajectory(joint_names=list(names))
            point = JointTrajectoryPoint(positions=list(targets))
            point.time_from_start.sec = 2
            trajectory.points = [point]
            self.trajectories[controller].publish(trajectory)
            self.last_targets.update(zip(names, targets))
            self.last_trajectory_times[controller] = sim_time
        for controller, names in GRIPPER_JOINTS.items():
            if any(name in action.joint_positions for name in names):
                targets = [
                    action.joint_positions.get(
                        name, self.last_targets.get(name, self.positions[name])
                    )
                    for name in names
                ]
                self.grippers[controller].publish(Float64MultiArray(data=targets))
                self.last_targets.update(zip(names, targets))

    def stop(self):
        self.apply(Action())
        for controller, names in CONTROLLER_JOINTS.items():
            if all(name in self.positions for name in names):
                trajectory = JointTrajectory(joint_names=list(names))
                point = JointTrajectoryPoint(
                    positions=[self.positions[name] for name in names]
                )
                point.time_from_start.nanosec = 200_000_000
                trajectory.points = [point]
                self.trajectories[controller].publish(trajectory)
