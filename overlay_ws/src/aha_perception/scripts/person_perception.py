#!/usr/bin/env python3
"""Track people in the head RGB-D camera with SAM 3.1 on the DGX Spark.

Each processed color image goes to aha_perception.person_worker on the DGX,
which returns person masks. The node measures each mask in the aligned depth,
transforms it to `target_frame` at the image time and tracks people. One image
is in flight at a time, so the processing rate is the DGX round trip, capped
by `max_rate`.

Remembered people are crops by name. A second thread asks the DGX VLM whether
a visible unnamed track is one of the remembered people no track holds, one
track at a time; detection continues meanwhile.
"""

import shlex
import threading
import time

from aha_msgs.msg import Person, PersonArray
from aha_msgs.srv import DescribePerson, RememberPerson
from geometry_msgs.msg import Point
import message_filters
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from aha_perception.dgx import DgxWorker, RequestError, WorkerError, WorkerOptions
from aha_perception.people import (
    MAX_REFERENCES,
    Detection,
    Gallery,
    PersonTracker,
    decode_mask,
    describe_crop,
    encode_image,
    measure,
    rotation_matrix,
)

PEOPLE_TOPIC = "/aha/perception/people"
MARKERS_TOPIC = "/aha/perception/people/markers"
REMEMBER_SERVICE = "/aha/perception/remember_person"
DESCRIBE_SERVICE = "/aha/perception/describe_person"


def image_array(msg):
    """Image message as a numpy array: rgb8 / bgr8 -> HxWx3 RGB, depth -> metres."""
    if msg.encoding in ("rgb8", "bgr8"):
        data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        rgb = data[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(rgb[..., ::-1] if msg.encoding == "bgr8" else rgb)
    if msg.encoding == "32FC1":
        data = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)
        return data[:, : msg.width]
    if msg.encoding in ("16UC1", "mono16"):
        data = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)
        depth = data[:, : msg.width].astype(np.float32) / 1000.0
        depth[depth == 0] = np.nan
        return depth
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def point(vector):
    return Point(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))


class PersonPerception(Node):
    def __init__(self):
        super().__init__("person_perception")
        p = self.declare_parameter
        defaults = WorkerOptions()
        self.color_topic = p("color_topic", "/camera/color/image_raw").value
        self.depth_topic = p("depth_topic", "/camera/depth_registered/image_rect").value
        self.info_topic = p("info_topic", "/camera/color/camera_info").value
        # Empty: the camera frame. A fixed frame (map / odom) keeps tracks still
        # while the robot moves.
        self.target_frame = p("target_frame", "map").value
        self.max_rate = p("max_rate", 5.0).value
        self.request_timeout = p("request_timeout", 30.0).value
        self.vlm_timeout = p("vlm_timeout", 180.0).value
        self.startup_timeout = p("startup_timeout", 1800.0).value
        self.tf_timeout = p("tf_timeout", 0.3).value
        self.options = WorkerOptions(
            host=p("ssh_host", defaults.host).value,
            port=p("ssh_port", defaults.port).value,
            user=p("ssh_user", defaults.user).value,
            identity_file=p("identity_file", defaults.identity_file).value,
            image=p("image", defaults.image).value,
            checkpoint=p("checkpoint", defaults.checkpoint).value,
            threshold=p("threshold", defaults.threshold).value,
            prompt=p("prompt", defaults.prompt).value,
            vlm_url=p("vlm_url", defaults.vlm_url).value,
            vlm_model=p("vlm_model", defaults.vlm_model).value,
        )
        # Tests: a local argv that speaks the worker protocol instead of SSH.
        self.worker_command = shlex.split(p("worker_command", "").value)
        self.tracker = PersonTracker(
            gate_m=p("gate", 0.8).value,
            forget_after=p("forget_after", 3.0).value,
            smoothing=p("smoothing", 0.5).value,
        )
        self.gallery = Gallery()

        self.lock = threading.Lock()  # tracker, gallery
        self.frame_ready = threading.Condition()
        self.frame = None
        self.stopping = threading.Event()
        self.error = None
        self.worker = None
        self.last_stamp = None
        self.stats = None  # per 10 s report, from the first processed image

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.people_pub = self.create_publisher(PersonArray, PEOPLE_TOPIC, 10)
        self.marker_pub = self.create_publisher(MarkerArray, MARKERS_TOPIC, 10)
        group = ReentrantCallbackGroup()
        self.create_service(
            RememberPerson, REMEMBER_SERVICE, self.on_remember, callback_group=group
        )
        self.create_service(
            DescribePerson, DESCRIBE_SERVICE, self.on_describe, callback_group=group
        )
        # Best effort loses most fragmented 640x480 images from the reliable
        # simulation bridge and rosbridge; a best-effort camera needs false.
        reliable = p("reliable", True).value
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE
            if reliable
            else ReliabilityPolicy.BEST_EFFORT,
        )
        subscribers = [
            message_filters.Subscriber(self, Image, topic, qos_profile=qos)
            for topic in (self.color_topic, self.depth_topic)
        ]
        subscribers.append(
            message_filters.Subscriber(
                self, CameraInfo, self.info_topic, qos_profile=qos
            )
        )
        # The camera publishes color, aligned depth and info with one stamp.
        self.sync = message_filters.TimeSynchronizer(subscribers, 10)
        self.sync.registerCallback(self.on_frame)
        self.threads = [
            threading.Thread(target=target, daemon=True)
            for target in (self.process, self.match_people)
        ]
        for thread in self.threads:
            thread.start()

    # -- Input ---------------------------------------------------------------

    def on_frame(self, color, depth, info):
        with self.frame_ready:
            self.frame = (color, depth, info)
            self.frame_ready.notify()

    def next_frame(self):
        with self.frame_ready:
            while not self.stopping.is_set():
                if self.frame is not None:
                    frame, self.frame = self.frame, None
                    return frame
                self.frame_ready.wait(0.5)
        return None

    # -- Processing ------------------------------------------------------------

    def process(self):
        try:
            self.worker = DgxWorker(
                self.options,
                log=lambda text: self.get_logger().info(text),
                command=self.worker_command or None,
            )
            info = self.worker.start(self.startup_timeout)
            self.get_logger().info(
                f"DGX person worker ready (prompt {info.get('prompt')!r}, "
                f"VLM {'on' if info.get('vlm') else 'off'}); "
                f"publishing {PEOPLE_TOPIC} in frame {self.target_frame or 'camera'}"
            )
            next_time = 0.0
            while not self.stopping.is_set():
                frame = self.next_frame()
                if frame is None:
                    break
                wait = next_time - time.monotonic()
                if wait > 0:
                    self.stopping.wait(wait)
                    with self.frame_ready:  # a newer frame may have arrived
                        frame = self.frame or frame
                        self.frame = None
                next_time = time.monotonic() + 1.0 / self.max_rate
                try:
                    self.process_frame(*frame)
                except (
                    TimeoutError,
                    ValueError,
                    TransformException,
                    RequestError,
                ) as exc:
                    self.get_logger().warning(
                        f"Skipped an image: {exc}", throttle_duration_sec=5.0
                    )
        except Exception as exc:
            if not self.stopping.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
                self.get_logger().error(f"Person perception stopped: {self.error}")

    def process_frame(self, color, depth_msg, info):
        stamp = color.header.stamp
        seconds = stamp_seconds(stamp)
        if self.last_stamp is not None and seconds < self.last_stamp - 1.0:
            # The simulation restarted; old tracks belong to another episode.
            with self.lock:
                self.tracker.clear()
        self.last_stamp = seconds
        rgb, depth = image_array(color), image_array(depth_msg)
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth is not aligned to the color image")
        started = time.monotonic()
        response = self.worker.request(
            {"type": "detect", "image": encode_image(rgb)}, self.request_timeout
        )
        round_trip = time.monotonic() - started
        if tuple(response["shape"]) != rgb.shape[:2]:
            raise ValueError("the DGX worker answered for another image size")
        rotation, translation = self.camera_pose(color.header.frame_id, stamp)
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        detections = []
        for person in response["people"]:
            mask = decode_mask(person, rgb.shape[:2])
            measured = measure(mask, depth, fx, fy, cx, cy)
            if measured is None:
                continue
            detections.append(
                Detection(
                    centre=rotation @ measured.centre + translation,
                    head=rotation @ measured.head + translation,
                    score=person["score"],
                    bbox=person["bbox"],
                    crop=describe_crop(rgb, mask),
                )
            )
        with self.lock:
            seen = self.tracker.update(detections, seconds)
            self.publish(
                stamp, self.target_frame or color.header.frame_id, {t.id for t in seen}
            )
        self.report(round_trip, response.get("seconds", {}).get("sam", 0.0))

    def camera_pose(self, camera_frame, stamp):
        if not self.target_frame or self.target_frame == camera_frame:
            return np.eye(3), np.zeros(3)
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            camera_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=self.tf_timeout),
        ).transform
        q, t = transform.rotation, transform.translation
        return rotation_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def report(self, round_trip, sam):
        if self.stats is None:
            self.stats = {
                "frames": 0,
                "round_trip": 0.0,
                "sam": 0.0,
                "since": time.monotonic() - round_trip,
            }
        stats = self.stats
        stats["frames"] += 1
        stats["round_trip"] += round_trip
        stats["sam"] += sam
        elapsed = time.monotonic() - stats["since"]
        if elapsed >= 10.0:
            n = stats["frames"]
            self.get_logger().info(
                f"{n / elapsed:.1f} images/s, round trip {stats['round_trip'] / n:.2f} s "
                f"(SAM 3.1 {stats['sam'] / n:.2f} s), {len(self.tracker.tracks)} tracked"
            )
            self.stats = None

    # -- Output ----------------------------------------------------------------

    def publish(self, stamp, frame, seen):
        message = PersonArray()
        message.header.stamp = stamp
        message.header.frame_id = frame
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for track in sorted(self.tracker.tracks.values(), key=lambda t: t.id):
            person = Person(
                id=track.id,
                name=track.name,
                score=float(track.score),
                position=point(track.centre),
                head=point(track.head),
                visible=track.id in seen,
                bbox=[int(v) for v in track.bbox],
                last_seen=Time(seconds=track.last_seen).to_msg(),
            )
            message.people.append(person)
            markers.markers.extend(self.track_markers(track, person.visible))
        for marker in markers.markers:
            marker.header.frame_id = frame
            marker.header.stamp = stamp
        self.people_pub.publish(message)
        self.marker_pub.publish(markers)

    @staticmethod
    def track_markers(track, visible):
        color = (0.1, 0.9, 0.3) if visible else (0.6, 0.6, 0.6)
        body = Marker(ns="people", id=track.id, type=Marker.SPHERE)
        body.pose.position = point(track.centre)
        body.pose.orientation.w = 1.0
        body.scale.x = body.scale.y = body.scale.z = 0.3
        body.color.r, body.color.g, body.color.b, body.color.a = (*color, 0.8)
        head = Marker(ns="heads", id=track.id, type=Marker.SPHERE)
        head.pose.position = point(track.head)
        head.pose.orientation.w = 1.0
        head.scale.x = head.scale.y = head.scale.z = 0.15
        head.color = body.color
        label = Marker(ns="labels", id=track.id, type=Marker.TEXT_VIEW_FACING)
        label.pose.position = point(track.head)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.15
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"{track.id} {track.name}".strip()
        return [body, head, label]

    # -- Services --------------------------------------------------------------

    def on_remember(self, request, response):
        name = request.name.strip()
        with self.lock:
            track = self.tracker.tracks.get(request.id)
            if not name:
                response.message = "name must not be empty"
            elif track is None:
                response.message = f"person {request.id} is not tracked"
            else:
                try:
                    self.gallery.remember(name, track, self.tracker.tracks.values())
                    response.success = True
                    response.message = f"Remembered person {track.id} as {name}"
                except ValueError as exc:
                    response.message = str(exc)
        return response

    def on_describe(self, request, response):
        with self.lock:
            track = self.tracker.tracks.get(request.id)
            crop = None if track is None else track.crop
        if crop is None:
            response.message = f"person {request.id} is not tracked"
            return response
        if self.worker is None or not self.worker.ready.is_set():
            response.message = "the DGX worker is not ready"
            return response
        try:
            reply = self.worker.request(
                {"type": "describe", "image": encode_image(crop)}, self.vlm_timeout
            )
        except (TimeoutError, WorkerError) as exc:
            response.message = str(exc)
            return response
        response.success = True
        response.features = reply["features"]
        response.message = (
            f"{len(reply['features'])} feature(s) in {reply['seconds']:.1f} s"
        )
        return response

    # -- Names ------------------------------------------------------------------

    def match_people(self):
        """Name visible unnamed tracks by asking the VLM, one track at a time."""
        while not self.stopping.wait(0.2):
            worker = self.worker
            if worker is None or not worker.info.get("vlm") or worker.failure:
                continue
            with self.lock:
                job = self.gallery.next_match(self.tracker)
            if job is None:
                continue
            track_id, crop, references = job
            try:
                name = self.match(crop, references)
            except (TimeoutError, WorkerError) as exc:
                self.get_logger().warning(
                    f"Matching person {track_id} failed: {exc}",
                    throttle_duration_sec=5.0,
                )
                continue
            if name is None:
                continue
            with self.lock:
                named = self.gallery.give(name, track_id, self.tracker)
            if named:
                self.get_logger().info(f"Person {track_id} is {name}")

    def match(self, crop, references):
        """The remembered name the VLM sees in `crop`, or None."""
        image = encode_image(crop)
        for start in range(0, len(references), MAX_REFERENCES):
            chunk = references[start : start + MAX_REFERENCES]
            reply = self.worker.request(
                {
                    "type": "match",
                    "image": image,
                    "references": [encode_image(c) for _, c in chunk],
                },
                self.vlm_timeout,
            )
            if reply["index"] is not None:
                return chunk[reply["index"]][0]
        return None

    def stop(self):
        self.stopping.set()
        with self.frame_ready:
            self.frame_ready.notify_all()
        for thread in self.threads:
            thread.join(timeout=5)
        if self.worker is not None and not self.worker.close():
            self.get_logger().error(
                "DGX cleanup was incomplete; see the messages above"
            )


def main():
    rclpy.init()
    node = PersonPerception()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    code = 0
    try:
        while rclpy.ok() and node.error is None:
            executor.spin_once(timeout_sec=0.2)
        code = 1 if node.error else 0
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
