"""show_objects against a fake camera, gaze and screen (no ROS graph)."""

import importlib.util
import io
import os
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
import aha_tasks.show_objects as show_objects  # noqa: E402
from aha_tasks.show_objects import (  # noqa: E402
    LOOK_FIRST,
    Cancelled,
    HeadCamera,
    ShowObjects,
    parse_record,
)

WIDTH, HEIGHT = 640, 480
K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
FORWARD = (0.5, -0.5, 0.5, 0.5)  # map (camera at the origin, looking +x) -> optical
OBJECTS = ["table.cup", "table.plate", "table.trash"]


def record(label, x, y, z=0.0, size=(0.1, 0.1, 0.1)):
    return {
        "label": label,
        "category": "tableware",
        "position": [x, y, z],
        "frame": "map",
        "size": list(size),
        "track": 1,
        "confidence": 1.0,
    }


class FakeCamera(HeadCamera):
    """The camera at the map origin looking along +x, without ROS."""

    def __init__(self):
        self.image_available = True
        self.transform_value = (FORWARD, (0.0, 0.0, 0.0))
        self.on_pause = None
        self.stalled = False
        self.calls = []

    def open(self):
        self.calls.append("open")

    def close(self):
        self.calls.append("close")

    def pause(self, seconds, cancelled):
        if self.on_pause:
            self.on_pause()
        if cancelled.is_set():
            raise Cancelled
        return not self.stalled

    def stamp_now(self):
        return 0

    def frame(self, after, timeout, cancelled):
        if not self.image_available:
            return None
        image = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="camera_color_optical_frame",
                stamp=SimpleNamespace(sec=0, nanosec=0),
            ),
            width=WIDTH,
            height=HEIGHT,
            step=WIDTH * 3,
            encoding="rgb8",
            data=bytes(WIDTH * HEIGHT * 3),
        )
        return image, SimpleNamespace(k=K)

    def transform(self, target, source, stamp, cancelled):
        return self.transform_value


class FakeScreen:
    def __init__(self):
        self.events = []
        self.cancel_on_show = False  # the skill is aborted while it speaks

    def say(self, text, cancelled):
        self.events.append(("say", text))
        return True, ""

    def show(self, image, image_format, text, cancelled):
        self.events.append(("show", text, image_format, image))
        if self.cancel_on_show:
            cancelled.set()
            return False, "Cancelled"
        return True, ""

    def hide(self):
        self.events.append(("hide",))


class FakeGaze:
    def __init__(self):
        self.targets = []

    def look_at(self, frame, point):
        self.targets.append((frame, (point.x, point.y, point.z)))


class ShowObjectsTests(unittest.TestCase):
    def setUp(self):
        self.camera = FakeCamera()
        self.screen = FakeScreen()
        self.gaze = FakeGaze()
        runtime = SimpleNamespace(
            node=None, gaze=self.gaze, interaction=self.screen, head_camera=self.camera
        )
        self.skill = ShowObjects(runtime)
        self.logs = []

    def run_skill(self, memory, objects=OBJECTS, surface="dining_table", cancel=None):
        cancelled = threading.Event()
        if cancel:
            cancel(cancelled)
        params = {"objects": objects, "surface": surface, "dwell": 0.0}
        ShowObjects.validate(params, None)
        ctx = SkillContext("table.show", params, memory, cancelled, self.logs.append)
        return self.skill.execute(ctx)

    def test_each_record_is_framed_and_said(self):
        memory = {
            "table.cup": record("mug", 2.0, 0.3),
            "table.plate": record("plate", 2.0, -0.3),
            "table.trash": record("pringles", 2.5, 0.0),
        }
        result = self.run_skill(memory)
        self.assertTrue(result.success, result.message)
        shows = [e for e in self.screen.events if e[0] == "show"]
        self.assertEqual(len(shows), 3)
        self.assertEqual(
            shows[0][1], f"{LOOK_FIRST} I found the mug on the dining table."
        )
        self.assertEqual(shows[1][1], "I found the plate on the dining table.")
        self.assertEqual(
            shows[2][1],
            "I found the pringles on the dining table. That goes in the trash.",
        )
        for _, _, image_format, data in shows:
            self.assertEqual(image_format, "jpeg")
            self.assertEqual(Image.open(io.BytesIO(data)).size, (WIDTH, HEIGHT))
        self.assertEqual(self.screen.events[-1], ("hide",))
        self.assertEqual(
            self.gaze.targets,
            [
                ("map", (2.0, 0.3, 0.0)),
                ("map", (2.0, -0.3, 0.0)),
                ("map", (2.5, 0.0, 0.0)),
            ],
        )

    def test_missing_and_mock_labels_are_spoken(self):
        memory = {"table.cup": None, "table.plate": "plate", "table.trash": ""}
        result = self.run_skill(memory)
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            self.screen.events, [("say", "I recognized the plate on the dining table.")]
        )
        self.assertIn("spoken only: plate", result.message)

    def test_nothing_perceived(self):
        result = self.run_skill({})
        self.assertTrue(result.success)
        self.assertEqual(self.screen.events, [])

    def test_no_image_is_spoken_and_fails(self):
        self.camera.image_available = False
        result = self.run_skill({"table.cup": record("mug", 2.0, 0.0)})
        self.assertFalse(result.success)
        self.assertIn("could not show: mug (no image", result.message)
        self.assertEqual(
            self.screen.events, [("say", "I recognized the mug on the dining table.")]
        )

    def test_out_of_view_fails(self):
        result = self.run_skill({"table.cup": record("mug", -2.0, 0.0)})
        self.assertFalse(result.success)
        self.assertIn("out of the camera view", result.message)

    def test_no_transform_fails(self):
        self.camera.transform_value = None
        result = self.run_skill({"table.cup": record("mug", 2.0, 0.0)})
        self.assertFalse(result.success)
        self.assertIn("no TF from map", result.message)

    def test_partly_out_is_still_shown(self):
        result = self.run_skill({"table.cup": record("mug", 1.0, -0.62)})
        self.assertTrue(result.success, result.message)
        self.assertTrue(any("partly out" in line for line in self.logs))

    def test_cancel_clears_the_screen(self):
        memory = {
            "table.cup": record("mug", 2.0, 0.3),
            "table.plate": record("plate", 2.0, -0.3),
        }
        holder = {}
        self.camera.on_pause = lambda: (
            holder["cancelled"].set() if len(self.gaze.targets) == 2 else None
        )
        result = self.run_skill(
            memory, cancel=lambda event: holder.setdefault("cancelled", event)
        )
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Cancelled")
        self.assertEqual(self.screen.events[-1], ("hide",))
        self.assertEqual(len([e for e in self.screen.events if e[0] == "show"]), 1)

    def test_cancel_during_the_first_speech_clears_the_screen(self):
        self.screen.cancel_on_show = True
        result = self.run_skill({"table.cup": record("mug", 2.0, 0.3)})
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Cancelled")
        self.assertEqual([e[0] for e in self.screen.events], ["show", "hide"])

    def test_frame_is_removed_before_an_object_only_spoken(self):
        """The previous frame must not stay up while another object is named."""
        memory = {
            "table.cup": record("mug", 2.0, 0.3),
            "table.plate": "plate",
            "table.trash": record("pringles", -2.0, 0.0),  # behind the camera
            "table.other1": record("apple", 2.0, -0.3),
        }
        result = self.run_skill(memory, objects=list(memory))
        self.assertFalse(result.success)
        self.assertEqual(
            [e[0] for e in self.screen.events],
            ["show", "hide", "say", "say", "show", "hide"],
        )
        self.assertEqual(
            [e[1] for e in self.screen.events if e[0] != "hide"],
            [
                f"{LOOK_FIRST} I found the mug on the dining table.",
                "I recognized the plate on the dining table.",
                "I recognized the pringles on the dining table. That goes in the trash.",
                "I found the apple on the dining table.",
            ],
        )

    def test_stopped_wait_clock_fails(self):
        """Without /clock the head may still turn: no image is taken then."""
        self.camera.stalled = True
        result = self.run_skill({"table.cup": record("mug", 2.0, 0.3)})
        self.assertFalse(result.success)
        self.assertIn("mug (wait clock stopped", result.message)
        self.assertEqual(
            self.screen.events, [("say", "I recognized the mug on the dining table.")]
        )

    def test_images_are_subscribed_only_while_showing(self):
        self.run_skill({"table.cup": record("mug", 2.0, 0.3)})
        self.assertEqual(self.camera.calls, ["open", "close"])
        self.camera.calls.clear()
        self.run_skill({"table.cup": "mug"})  # nothing to picture
        self.assertEqual(self.camera.calls, [])

    def test_images_are_unsubscribed_on_cancel(self):
        self.screen.cancel_on_show = True
        self.run_skill({"table.cup": record("mug", 2.0, 0.3)})
        self.assertEqual(self.camera.calls, ["open", "close"])

    def test_bad_record_fails(self):
        result = self.run_skill({"table.cup": {"position": [1, 2, 3]}})
        self.assertFalse(result.success)
        self.assertIn("table.cup (not an object record)", result.message)


class FakeClock:
    def __init__(self, now):
        self.now_value = now
        self.ros_time_is_active = True

    def now(self):
        return SimpleNamespace(nanoseconds=int(self.now_value() * 1e9))


def bare_camera(now):
    """A HeadCamera on a fake node clock, without ROS subscriptions or TF."""
    camera = HeadCamera.__new__(HeadCamera)
    clock = FakeClock(now)
    camera.node = SimpleNamespace(get_clock=lambda: clock)
    return camera


class PauseTests(unittest.TestCase):
    def test_advancing_clock(self):
        camera = bare_camera(time.monotonic)
        self.assertTrue(camera.pause(0.1, threading.Event()))

    def test_stopped_clock_is_reported(self):
        camera = bare_camera(lambda: 5.0)
        with patch.object(show_objects, "CLOCK_STALL_TIMEOUT", 0.2):
            started = time.monotonic()
            self.assertFalse(camera.pause(1.5, threading.Event()))
        self.assertLess(time.monotonic() - started, 1.0)

    def test_cancel(self):
        camera = bare_camera(lambda: 5.0)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            camera.pause(1.5, cancelled)


@unittest.skipUnless(importlib.util.find_spec("sensor_msgs"), "needs ROS messages")
class SubscriptionTests(unittest.TestCase):
    def test_open_and_close(self):
        created, destroyed = [], []
        node = SimpleNamespace(
            create_subscription=lambda kind, topic, callback, qos: created.append(topic)
            or topic,
            destroy_subscription=destroyed.append,
        )
        camera = HeadCamera.__new__(HeadCamera)
        camera.node, camera.lock, camera.subscriptions = node, threading.Lock(), []
        camera.image = camera.info = "old"
        camera.open()
        camera.open()  # once only
        self.assertEqual(created, [show_objects.IMAGE_TOPIC, show_objects.INFO_TOPIC])
        self.assertIsNone(camera.image)
        camera._on_image("image")
        camera.close()
        self.assertEqual(destroyed, created)
        self.assertEqual((camera.subscriptions, camera.image), ([], None))


class ParamsTests(unittest.TestCase):
    def test_valid(self):
        ShowObjects.validate({"objects": ["a.b"], "surface": "floor"}, None)
        ShowObjects.validate({"objects": ["a.b"], "surface": "floor", "dwell": 0}, None)

    def test_invalid(self):
        for params in (
            {"objects": ["a.b"]},
            {"objects": [], "surface": "floor"},
            {"objects": "a.b", "surface": "floor"},
            {"objects": ["a.b", "a.b"], "surface": "floor"},
            {"objects": ["a.b"], "surface": ""},
            {"objects": ["a.b"], "surface": "floor", "dwell": -1},
            {"objects": ["a.b"], "surface": "floor", "dwell": True},
            {"objects": ["a.b"], "surface": "floor", "extra": 1},
        ):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    ShowObjects.validate(params, None)

    def test_record_without_place_is_a_label(self):
        self.assertEqual(parse_record("a.b", {"label": "mug"}), ("mug", None))
        self.assertEqual(
            parse_record("a.b", {**record("mug", 1, 2), "size": [0, 1, 1]})[1], None
        )
        label, place = parse_record("a.b", record("mug", 1, 2, 3))
        self.assertEqual(
            (label, place[0], place[1], place[3]), ("mug", "map", [1.0, 2.0, 3.0], None)
        )
        self.assertEqual(
            parse_record("a.b", {**record("mug", 1, 2), "yaw": 0.5})[1][3], 0.5
        )


if __name__ == "__main__":
    unittest.main()
