"""The robot's screen without ROS: questions, captions, images and its HTTP API."""

import http.client
import json
import os
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.display import (  # noqa: E402
    CAPTION_LINES,
    MAX_CHOICES,
    Screen,
    check_question,
    image_type,
)
from aha_tasks import monitor  # noqa: E402
from aha_tasks.monitor import StatusHub, make_server  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "web"


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.screen = Screen()

    def payload(self):
        return self.screen.view()

    def test_a_touch_answers_the_question_on_the_screen(self):
        question = self.screen.ask(" Is this the cup? ", ["Yes", "No"])
        self.assertEqual(
            self.payload()["question"],
            {"id": question.id, "prompt": "Is this the cup?", "choices": ["Yes", "No"]},
        )
        self.assertEqual(self.screen.touch(question.id, 1), (True, "No"))
        self.assertTrue(question.done.is_set())
        self.assertEqual(question.index, 1)
        self.assertIsNone(question.ended)
        self.assertIsNone(self.payload()["question"])

    def test_a_late_or_wrong_touch_is_refused(self):
        question = self.screen.ask("", ["Start"])
        self.assertFalse(self.screen.touch(question.id + 1, 0)[0])
        self.assertFalse(self.screen.touch(question.id, 1)[0])
        self.assertFalse(self.screen.touch(question.id, -1)[0])
        self.assertFalse(question.done.is_set())
        self.assertTrue(self.screen.touch(question.id, 0)[0])
        self.assertFalse(self.screen.touch(question.id, 0)[0])  # a second tap

    def test_a_new_question_replaces_the_shown_one(self):
        first = self.screen.ask("First", ["A"])
        second = self.screen.ask("Second", ["B"])
        self.assertTrue(first.done.is_set())
        self.assertEqual(first.ended, "Replaced by another question")
        self.assertFalse(self.screen.close(first))  # already off the screen
        self.assertEqual(self.payload()["question"]["id"], second.id)
        self.assertTrue(self.screen.close(second))
        self.assertIsNone(self.payload()["question"])

    def test_shutdown_ends_the_question_and_later_ones(self):
        question = self.screen.ask("Ready to start", ["Start"])
        self.screen.shutdown()
        self.assertTrue(question.done.is_set())
        self.assertEqual(question.ended, "The screen is shutting down")
        self.assertIsNone(self.payload()["question"])
        later = self.screen.ask("Again?", ["Yes"])
        self.assertTrue(later.done.is_set())
        self.assertEqual(later.ended, "The screen is shutting down")
        self.assertIsNone(self.payload()["question"])
        self.assertFalse(self.screen.touch(later.id, 0)[0])

    def test_buttons_are_checked(self):
        self.assertIsNone(check_question(["Yes", "No"]))
        self.assertIsNotNone(check_question([]))
        self.assertIsNotNone(check_question(["x"] * (MAX_CHOICES + 1)))
        self.assertIsNotNone(check_question(["Yes", " "]))
        self.assertIsNotNone(check_question(["Yes", "Yes"]))
        with self.assertRaises(ValueError):
            self.screen.ask("", [])

    def test_a_blank_caption_clears_them(self):
        self.screen.say("Hello")
        self.screen.say("   ")
        self.assertEqual(self.payload()["captions"], [])
        self.screen.say(" Hello ")
        self.screen.say("World")
        captions = self.payload()["captions"]
        self.assertEqual([line["text"] for line in captions], ["Hello", "World"])
        self.assertLess(captions[0]["id"], captions[1]["id"])

    def test_captions_drop_the_oldest_line(self):
        for i in range(CAPTION_LINES + 2):
            self.screen.say(f"line {i}")
        texts = [line["text"] for line in self.payload()["captions"]]
        self.assertEqual(texts, [f"line {i}" for i in range(2, CAPTION_LINES + 2)])

    def test_one_image_at_a_time(self):
        self.screen.show_image(b"first", "image/png")
        first = self.payload()["image"]["id"]
        self.screen.show_image(b"second", "image/jpeg")
        second = self.payload()["image"]["id"]
        self.assertIsNone(self.screen.image(first))
        self.assertEqual(self.screen.image(second), ("image/jpeg", b"second"))
        self.screen.show_image(b"", None)
        self.assertIsNone(self.payload()["image"])
        self.assertIsNone(self.screen.image(second))

    def test_image_formats(self):
        self.assertEqual(image_type("jpeg"), "image/jpeg")
        self.assertEqual(image_type("bgr8; jpeg compressed bgr8"), "image/jpeg")
        self.assertEqual(image_type("PNG"), "image/png")
        self.assertIsNone(image_type("rgb8"))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.hub = StatusHub(online=False, status=None)
        self.screen = Screen(self.hub)  # shared as in task_monitor
        self.server = self.serve(self.screen)

    def serve(self, screen):
        server = make_server(
            "127.0.0.1", 0, self.hub, lambda c: (True, ""), WEB, screen
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5.0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def request(self, method, path, body=None, headers=None, server=None):
        port = (server or self.server).server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(connection.close)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        return response, response.read()

    def touch(self, body, content_type="application/json"):
        response, data = self.request(
            "POST",
            "/display/api/touch",
            json.dumps(body),
            {"Content-Type": content_type},
        )
        return response.status, json.loads(data)

    def test_pages_are_served(self):
        response, body = self.request("GET", "/display/")
        self.assertEqual(response.status, 200)
        self.assertIn(b"display.js", body)
        for name in ("display.js", "display.css"):
            response, _ = self.request("GET", f"/display/{name}")
            self.assertEqual(response.status, 200, name)
        response, body = self.request("GET", "/")
        self.assertIn(b'src="display/?embedded"', body)
        response, _ = self.request("GET", "/display/?embedded")
        self.assertEqual(response.status, 200)
        response, _ = self.request("GET", "/display")
        self.assertEqual(response.status, 301)
        self.assertEqual(response.getheader("Location"), "display/")
        response, _ = self.request("GET", "/display/../index.html")
        self.assertEqual(response.status, 404)

    def test_monitor_status_carries_the_screen(self):
        self.screen.say("Hello")
        response, body = self.request("GET", "/api/status")
        payload = json.loads(body)
        self.assertEqual((payload["online"], payload["status"]), (False, None))
        self.assertEqual(payload["screen"]["captions"][0]["text"], "Hello")

    def test_touch_answers_the_question(self):
        question = self.screen.ask("Ready to start", ["Start"])
        self.assertEqual(
            self.touch({"question": question.id, "choice": 0}),
            (200, {"accepted": True, "message": "Start"}),
        )
        self.assertEqual(question.index, 0)
        status, reply = self.touch({"question": question.id, "choice": 0})
        self.assertEqual(status, 200)
        self.assertFalse(reply["accepted"])

    def test_touch_needs_a_json_request(self):
        question = self.screen.ask("", ["Start"])
        status, _ = self.touch({"question": question.id, "choice": 0}, "text/plain")
        self.assertEqual(status, 415)
        for body in ({"question": question.id}, {"question": "1", "choice": 0}, []):
            status, reply = self.touch(body)
            self.assertEqual(status, 400, body)
            self.assertFalse(reply["accepted"])
        status, _ = self.touch({"question": question.id, "choice": True})
        self.assertEqual(status, 400)
        self.assertFalse(question.done.is_set())

    def test_image_is_served_while_shown(self):
        self.screen.show_image(b"\x89PNG", "image/png")
        number = self.screen.view()["image"]["id"]
        response, body = self.request("GET", f"/display/api/image/{number}")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertEqual(body, b"\x89PNG")
        for path in (f"/display/api/image/{number + 1}", "/display/api/image/x"):
            response, _ = self.request("GET", path)
            self.assertEqual(response.status, 404, path)

    def test_events_stream_the_screen(self):
        question = self.screen.ask("Is this the cup?", ["Yes", "No"])
        port = self.server.server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("GET", "/display/api/events")
        response = connection.getresponse()
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        line = response.fp.readline().decode()
        self.assertTrue(line.startswith("data: "), line)
        self.assertEqual(json.loads(line[6:])["question"]["id"], question.id)
        response.fp.readline()  # blank line after the event
        # A runner status change is not sent to the screen; its next change is.
        self.hub.update(online=True)
        self.screen.say("Hello")
        line = response.fp.readline().decode()
        self.assertEqual(json.loads(line[6:])["captions"][0]["text"], "Hello")

    def test_keepalive_is_not_delayed_by_status_changes(self):
        self.addCleanup(setattr, monitor, "HEARTBEAT", monitor.HEARTBEAT)
        monitor.HEARTBEAT = 1.0
        port = self.server.server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("GET", "/display/api/events")
        response = connection.getresponse()
        self.assertTrue(response.fp.readline().startswith(b"data: "))
        start = time.monotonic()
        response.fp.readline()
        # A status change late in the interval must not restart the wait.
        timer = threading.Timer(0.8, self.hub.update, kwargs={"online": True})
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(response.fp.readline(), b": keepalive\n")
        self.assertLess(time.monotonic() - start, 1.4)

    def test_one_stream_for_the_gui_includes_the_screen(self):
        port = self.server.server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("GET", "/api/events")
        response = connection.getresponse()
        line = response.fp.readline().decode()
        self.assertEqual(json.loads(line[6:])["screen"]["question"], None)
        response.fp.readline()
        question = self.screen.ask("Ready to start", ["Start"])
        line = response.fp.readline().decode()
        self.assertEqual(json.loads(line[6:])["screen"]["question"]["id"], question.id)

    def test_the_embedded_screen_does_not_open_a_stream(self):
        response, body = self.request("GET", "/")
        self.assertIn(b'src="display/?embedded"', body)
        script = (WEB / "display" / "display.js").read_text()
        self.assertIn('has("embedded")', script)
        self.assertIn("forwardScreen", (WEB / "app.js").read_text())

    def test_without_a_screen_there_is_no_display(self):
        server = self.serve(None)
        for path in ("/display/", "/display/api/events"):
            response, _ = self.request("GET", path, server=server)
            self.assertEqual(response.status, 404, path)
        response, _ = self.request(
            "POST",
            "/display/api/touch",
            "{}",
            {"Content-Type": "application/json"},
            server=server,
        )
        self.assertEqual(response.status, 404)


if __name__ == "__main__":
    unittest.main()
