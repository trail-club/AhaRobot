"""Show perceived objects on the robot's screen, one at a time (PPC rule 16).

For each object, show_objects turns the head to it, waits for an image taken
after the head settled, frames the object in the whole image (the scene stays
visible; aha_tasks.object_view) and shows it with Interaction.show while saying
what it is. The referee is asked to look at the screen, not to touch or answer.

The objects are memory keys holding records of perceive_objects. A missing or
null key is skipped; a record without a position (a label-only mock) is only
spoken. The mock that the engine put in a key after the perception failed and
was skipped is not a perception: it is neither shown nor spoken, only named in
the result, and the skill does not fail for it, since the operator already
handled that failure (the mock of an unselected perception is shown, to try
the show alone). An object that cannot be pictured is spoken instead, and the
skill fails so that the operator sees it.
"""

import math
import threading
import time
from types import SimpleNamespace

from aha_tasks.core.engine import SKIPPED
from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.interaction import Cancelled, Interaction
from aha_tasks.object_view import (
    draw_object,
    encode_jpeg,
    image_array,
    image_box,
    object_sentence,
)

IMAGE_TOPIC = "/camera/color/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
CAMERA_FRAME = "camera_color_optical_frame"  # when an image has no frame_id
SETTLE = 1.5  # s (wait clock) after turning the head before taking the image
IMAGE_TIMEOUT = 5.0  # s (wall clock) for an image taken after the head settled
TF_TIMEOUT = 1.0  # s (wall clock) for the transform at the image time
CLOCK_STALL_TIMEOUT = 10.0  # s (wall clock) without the wait clock advancing
DWELL = 2.0  # s (wall clock) a picture stays after its sentence
LOOK_FIRST = "Please look at my screen."


def is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_xyz(value):
    return isinstance(value, list) and len(value) == 3 and all(map(is_number, value))


def skipped_mock(spec, statuses, key, value):
    """Id of a skipped subgoal whose mock put `value` (not None) in `key`, or None."""
    for subgoal in spec.subgoals:
        if statuses.get(subgoal.id) == SKIPPED and subgoal.mock.get(key) == value:
            return subgoal.id
    return None


def parse_record(key, value):
    """(label, place) of a memory value; place is None unless it can be pictured.

    place = (frame, position, size, yaw); yaw is None when the record has none.
    A size component may be 0 (perceive_objects rounds to mm, e.g. the height
    of a flat dish tab): image_box still frames it with its padding. A string
    is a label only (a mock).
    """
    if isinstance(value, str):
        return value.strip() or key.rsplit(".", 1)[-1], None
    if not isinstance(value, dict):
        raise ValueError(f"{key}: not an object record: {value!r}")
    label = value.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{key}: the record has no label: {value!r}")
    frame, position, size = value.get("frame"), value.get("position"), value.get("size")
    if not (
        isinstance(frame, str)
        and frame
        and is_xyz(position)
        and is_xyz(size)
        and all(v >= 0 for v in size)
    ):
        return label.strip(), None
    yaw = float(value["yaw"]) if is_number(value.get("yaw")) else None
    return label.strip(), (frame, [float(v) for v in position], size, yaw)


class HeadCamera:
    """Head colour images, camera info and TF, for the skills of one runner.

    The images are subscribed only between open() and close(), while a skill
    shows objects: the runner's single executor would otherwise take in every
    image (640x480 rgb8, about 0.9 MB at 15-30 Hz) for the rest of the task.
    TF comes from `tf`, the buffer of the runner's objects (aha_tasks.objects,
    made when the perceive skills get ready) when there is one, so that the
    node does not take in /tf twice; otherwise a listener of its own stays.
    """

    def __init__(self, node, tf=None):
        self.node = node
        self.lock = threading.Lock()
        self.image, self.info = None, None
        self.subscriptions = []
        self.listener = None
        if tf is None:
            from tf2_ros import Buffer, TransformListener

            tf = Buffer()
            self.listener = TransformListener(tf, node)
        self.tf = tf

    def open(self):
        """Subscribe to the colour image and its camera info."""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.subscriptions = [
            self.node.create_subscription(
                Image, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data
            ),
            self.node.create_subscription(
                CameraInfo, INFO_TOPIC, self._on_info, qos_profile_sensor_data
            ),
        ]

    def close(self):
        """Stop the subscriptions of open() and drop the last image."""
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
        self.subscriptions = []
        with self.lock:
            self.image, self.info = None, None

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "head_camera", None) is None:
            objects = getattr(runtime, "objects", None)
            runtime.head_camera = cls(runtime.node, getattr(objects, "tf", None))
        return runtime.head_camera

    def _on_image(self, message):
        with self.lock:
            self.image = message

    def _on_info(self, message):
        with self.lock:
            self.info = message

    def wait_clock(self):
        """Simulation seconds when active; otherwise monotonic wall seconds."""
        clock = self.node.get_clock()
        if not getattr(clock, "ros_time_is_active", False):
            return time.monotonic()
        return clock.now().nanoseconds * 1e-9

    def pause(self, seconds, cancelled):
        """Wait `seconds` on the wait clock: True, or False once the wait clock
        has not advanced for CLOCK_STALL_TIMEOUT wall seconds (no /clock)."""
        start = last = self.wait_clock()
        progress = time.monotonic()
        while True:
            if cancelled.is_set():
                raise Cancelled
            now = self.wait_clock()
            if now >= start + seconds:
                return True
            wall = time.monotonic()
            if now > last:
                last, progress = now, wall
            elif wall - progress >= CLOCK_STALL_TIMEOUT:
                return False
            time.sleep(0.05)

    def new_image(self, cancelled):
        """(image, info) of an image taken from now on, or None after IMAGE_TIMEOUT."""
        after = self.node.get_clock().now().nanoseconds  # the time of the stamps
        deadline = time.monotonic() + IMAGE_TIMEOUT
        while time.monotonic() < deadline:
            if cancelled.is_set():
                raise Cancelled
            with self.lock:
                image, info = self.image, self.info
            if image is not None and info is not None:
                stamp = image.header.stamp
                if stamp.sec * 1_000_000_000 + stamp.nanosec >= after:
                    return image, info
            time.sleep(0.02)
        return None

    def transform(self, target, source, stamp, cancelled):
        """((x, y, z, w), (x, y, z)) from `source` into `target` at `stamp`, or None.

        After TF_TIMEOUT without the transform at that time, the latest one is
        used: the head has settled, so it has not moved since the image.
        """
        from rclpy.time import Time
        from tf2_ros import TransformException

        for at in (Time.from_msg(stamp), Time()):  # Time(): the latest
            deadline = time.monotonic() + TF_TIMEOUT
            while time.monotonic() < deadline:
                if cancelled.is_set():
                    raise Cancelled
                try:
                    t = self.tf.lookup_transform(target, source, at).transform
                except TransformException:
                    time.sleep(0.05)
                    continue
                r, p = t.rotation, t.translation
                return (r.x, r.y, r.z, r.w), (p.x, p.y, p.z)
        return None


class ShowObjects(Skill):
    """Show each perceived object framed in the head camera image, and say it.

    params: {"objects": ["table.cutlery", ...], "surface": "dining_table",
    "dwell": 2.0}; dwell is optional.
    """

    @classmethod
    def validate(cls, params, spec):
        unknown = set(params) - {"objects", "surface", "dwell"}
        missing = {"objects", "surface"} - set(params)
        if unknown or missing:
            raise ValueError(
                f"show_objects needs objects and surface, may have dwell: {params}"
            )
        objects = params["objects"]
        if (
            not isinstance(objects, list)
            or not objects
            or not all(isinstance(key, str) and key.strip() for key in objects)
        ):
            raise ValueError(
                f"objects must be a nonempty list of memory keys: {params}"
            )
        if len(set(objects)) != len(objects):
            raise ValueError(f"objects has a key twice: {params}")
        if not isinstance(params["surface"], str) or not params["surface"].strip():
            raise ValueError(f"surface must be a nonempty label: {params}")
        dwell = params.get("dwell", DWELL)
        if not is_number(dwell) or dwell < 0:
            raise ValueError(f"dwell must be a number >= 0: {params}")

    def ready(self):
        Interaction.of(self.runtime).find_tts()

    def execute(self, ctx):
        interaction = Interaction.of(self.runtime)
        surface = ctx.params["surface"]
        dwell = ctx.params.get("dwell", DWELL)
        items, shown, spoken, failed = [], [], [], []
        mocked = {}  # skipped subgoal -> keys holding its mock
        for key in ctx.params["objects"]:
            value = ctx.memory.get(key)
            if value is None or value == "":
                ctx.log(f"{key}: not perceived")
                continue
            skipped = skipped_mock(self.runtime.spec, ctx.statuses, key, value)
            if skipped:
                mocked.setdefault(skipped, []).append(key)
                continue
            try:
                label, place = parse_record(key, value)
            except ValueError as error:
                ctx.log(str(error))
                failed.append(f"{key} (not an object record)")
                continue
            items.append((key, label, place))
        if not items and not failed and not mocked:
            return SkillResult(True, "No perceived objects to show")
        displayed = False  # a framed image is on the screen
        camera = None
        if any(place is not None for _, _, place in items):
            camera = HeadCamera.of(self.runtime)
            camera.open()
        try:
            for key, label, place in items:
                trash = "trash" in key.split(".")
                picture = None
                if place is None:
                    spoken.append(label)
                else:
                    picture, note = self.picture(camera, place, label, ctx.cancelled)
                    if picture is None:
                        ctx.log(f"{key}: not shown: {note}")
                        failed.append(f"{label} ({note})")
                    elif note:
                        ctx.log(f"{key}: {note}")
                if picture is None:
                    if displayed:  # the previous frame must not seem to be this object
                        interaction.hide()
                        displayed = False
                    text = object_sentence(label, surface, trash, shown=False)
                    self.check_speech(interaction.say(text, ctx.cancelled), ctx)
                    continue
                text = object_sentence(label, surface, trash)
                if not shown:
                    text = f"{LOOK_FIRST} {text}"
                displayed = True  # show publishes the image before speaking
                self.check_speech(
                    interaction.show(picture, "jpeg", text, ctx.cancelled), ctx
                )
                shown.append(label)
                self.hold(dwell, ctx.cancelled)
        except Cancelled:
            return SkillResult(False, "Cancelled")
        finally:
            if displayed:
                interaction.hide()
            if camera is not None:
                camera.close()
        parts = [f"shown: {', '.join(shown)}"] if shown else []
        if spoken:
            parts.append(f"spoken only: {', '.join(spoken)}")
        if failed:
            parts.append(f"could not show: {', '.join(failed)}")
        for skipped, keys in mocked.items():
            parts.append(f"not perceived ({skipped} skipped): {', '.join(keys)}")
        return SkillResult(not failed, "; ".join(parts))

    def check_speech(self, result, ctx):
        """Raise Cancelled after an aborted sentence; log a failed one."""
        ok, message = result
        if ctx.cancelled.is_set():
            raise Cancelled
        if not ok:  # the caption is on the screen even without the TTS
            ctx.log(f"Speech: {message}")

    def hold(self, seconds, cancelled):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cancelled.is_set():
                raise Cancelled
            time.sleep(0.05)

    def picture(self, camera, place, label, cancelled):
        """(JPEG bytes, note) of the object framed in a new head image, or (None, why)."""
        frame, position, size, yaw = place
        x, y, z = position
        self.runtime.gaze.look_at(frame, SimpleNamespace(x=x, y=y, z=z))
        if not camera.pause(SETTLE, cancelled):
            return None, (
                f"wait clock stopped ({CLOCK_STALL_TIMEOUT:.0f} s without /clock)"
            )
        found = camera.new_image(cancelled)
        if found is None:
            return None, f"no image on {IMAGE_TOPIC} within {IMAGE_TIMEOUT:.0f} s"
        image, info = found
        source = image.header.frame_id or CAMERA_FRAME
        transform = camera.transform(source, frame, image.header.stamp, cancelled)
        if transform is None:
            return None, f"no TF from {frame} to {source}"
        rotation, translation = transform
        try:
            rgb = image_array(
                image.data, image.width, image.height, image.step, image.encoding
            )
        except ValueError as error:
            return None, str(error)
        box, cut = image_box(
            position,
            size,
            rotation,
            translation,
            info.k,
            image.width,
            image.height,
            yaw=yaw,
        )
        if box is None:
            return None, "out of the camera view"
        note = "partly out of the camera view" if cut else ""
        return encode_jpeg(draw_object(rgb, box, label)), note
