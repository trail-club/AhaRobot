"""say / show / ask through ROS against the real screen, and the skills that use them.

ROS runs on ROS_DOMAIN_ID 95 (AHA_TEST_DOMAIN_ID to change), so the tests do not
reach a running robot, sim or TTS through the real names.
"""

import math
import os
import select
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks import interaction as interaction_module  # noqa: E402
from aha_tasks.display import Screen, ScreenBridge  # noqa: E402
from aha_tasks.interaction import (  # noqa: E402
    LISTEN_ACTION,
    NO_ASR,
    SPEAK_ACTION,
    Interaction,
    Say,
    WaitStartSignal,
)

DOMAIN_ID = int(os.environ.get("AHA_TEST_DOMAIN_ID", "95"))


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "Timed out"
        time.sleep(0.01)


def context(params=None, logs=None):
    return SkillContext(
        subgoal="start.signal",
        params=params or {},
        memory={},
        cancelled=threading.Event(),
        log=(logs if logs is not None else []).append,
    )


class FakeInteraction:
    def __init__(self, said=(True, "said"), touched=(0, "Start")):
        self.said, self.touched = said, touched
        self.calls = []

    def say(self, text, cancelled):
        self.calls.append(("say", text))
        return self.said

    def ask(self, prompt, choices, cancelled, timeout=0.0):
        self.calls.append(("ask", prompt, choices))
        return self.touched


class SkillTests(unittest.TestCase):
    def skill(self, cls, interaction):
        return cls(SimpleNamespace(node=None, interaction=interaction))

    def test_say_needs_exactly_a_text(self):
        Say.validate({"text": "Hello"}, None)
        for params in ({}, {"text": " "}, {"text": 3}, {"text": "Hi", "to": "you"}):
            with self.assertRaises(ValueError, msg=params):
                Say.validate(params, None)

    def test_say_reports_the_outcome(self):
        interaction = FakeInteraction(said=(False, "TTS: busy"))
        result = self.skill(Say, interaction).execute(context({"text": " Hello "}))
        self.assertEqual(interaction.calls, [("say", "Hello")])
        self.assertFalse(result.success)
        self.assertEqual(result.message, "TTS: busy")

    def test_start_signal_announces_then_waits_for_the_button(self):
        interaction = FakeInteraction()
        result = self.skill(WaitStartSignal, interaction).execute(context())
        self.assertTrue(result.success, result.message)
        self.assertEqual(
            interaction.calls,
            [
                (
                    "say",
                    "I am ready to start. Please touch the Start button on my screen.",
                ),
                ("ask", WaitStartSignal.PROMPT, ["Start"]),
            ],
        )

    def test_the_announcement_names_the_button_shown(self):
        interaction = FakeInteraction()
        skill = self.skill(WaitStartSignal, interaction)
        self.assertTrue(skill.execute(context({"button": "Go"})).success)
        self.assertEqual(
            interaction.calls[0],
            ("say", "I am ready to start. Please touch the Go button on my screen."),
        )

    def test_start_signal_params(self):
        WaitStartSignal.validate({}, None)
        WaitStartSignal.validate({"text": "Touch Go.", "button": "Go"}, None)
        for params in ({"button": ""}, {"text": None}, {"timeout": 3}):
            with self.assertRaises(ValueError, msg=params):
                WaitStartSignal.validate(params, None)
        interaction = FakeInteraction()
        self.skill(WaitStartSignal, interaction).execute(
            context({"text": "Touch Go.", "button": "Go"})
        )
        self.assertEqual(interaction.calls[1], ("ask", WaitStartSignal.PROMPT, ["Go"]))

    def test_start_signal_waits_even_if_it_cannot_speak(self):
        interaction = FakeInteraction(said=(False, "TTS: busy"))
        logs = []
        result = self.skill(WaitStartSignal, interaction).execute(context(logs=logs))
        self.assertTrue(result.success)
        self.assertIn("TTS: busy", logs[0])

    def test_start_signal_fails_without_a_touch(self):
        interaction = FakeInteraction(touched=(None, "No screen"))
        result = self.skill(WaitStartSignal, interaction).execute(context())
        self.assertFalse(result.success)
        self.assertEqual(result.message, "No screen")


class RosTests(unittest.TestCase):
    """The runner's Interaction against the task monitor's ScreenBridge."""

    @classmethod
    def setUpClass(cls):
        rclpy.init(domain_id=DOMAIN_ID)
        cls.screen = Screen()
        cls.display_node = Node("test_robot_display")
        cls.bridge = ScreenBridge(cls.display_node, cls.screen)
        cls.node = Node("test_task_runner")
        cls.interaction = Interaction(cls.node)
        cls.executor = MultiThreadedExecutor()
        cls.executor.add_node(cls.display_node)
        cls.executor.add_node(cls.node)
        cls.spinner = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.spinner.start()

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown()
        cls.spinner.join(timeout=5.0)
        cls.bridge.server.destroy()  # destroy_node() leaves an action server
        cls.node.destroy_node()
        cls.display_node.destroy_node()
        rclpy.try_shutdown()

    def setUp(self):
        self.cancelled = threading.Event()
        self.screen.say("")  # clear captions

    def ask_in_background(self, *args, **kwargs):
        answer = {}

        def ask():
            answer["value"] = self.interaction.ask(*args, self.cancelled, **kwargs)

        thread = threading.Thread(target=ask)
        thread.start()
        self.addCleanup(thread.join, 10.0)
        return answer, thread

    def shown(self):
        return self.screen.view()["question"]

    def test_touch_returns_the_choice(self):
        answer, thread = self.ask_in_background("Pick one", ["Yes", "No"])
        wait_until(lambda: self.shown() is not None)
        self.assertEqual(self.shown()["choices"], ["Yes", "No"])
        self.assertEqual(self.screen.touch(self.shown()["id"], 1), (True, "No"))
        thread.join(10.0)
        self.assertEqual(answer["value"], (1, "No"))
        self.assertIsNone(self.shown())

    def test_no_touch_within_the_timeout(self):
        start = time.monotonic()
        index, message = self.interaction.ask(
            "Ready to start", ["Start"], self.cancelled, timeout=0.3
        )
        self.assertIsNone(index)
        self.assertIn("No touch within 0.3 s", message)
        self.assertGreaterEqual(time.monotonic() - start, 0.3)
        self.assertIsNone(self.shown())

    def test_cancel_removes_the_question(self):
        answer, thread = self.ask_in_background("Ready to start", ["Start"])
        wait_until(lambda: self.shown() is not None)
        self.cancelled.set()
        thread.join(10.0)
        self.assertEqual(answer["value"], (None, "Cancelled"))
        wait_until(lambda: self.shown() is None)

    def test_a_later_question_replaces_the_earlier(self):
        first, first_thread = self.ask_in_background("First", ["A"])
        wait_until(lambda: self.shown() is not None)
        second, second_thread = self.ask_in_background("Second", ["B"])
        wait_until(lambda: self.shown()["prompt"] == "Second")
        first_thread.join(10.0)
        self.assertEqual(first["value"], (None, "Replaced by another question"))
        self.screen.touch(self.shown()["id"], 0)
        second_thread.join(10.0)
        self.assertEqual(second["value"], (0, "B"))

    def without_screen(self):
        from aha_msgs.action import Ask

        interaction = Interaction(self.node)
        interaction.screen = ActionClient(self.node, Ask, "/test/no_screen/ask")
        self.addCleanup(interaction.screen.destroy)
        return interaction

    def test_cancel_while_looking_for_the_screen(self):
        interaction = self.without_screen()
        threading.Timer(0.2, self.cancelled.set).start()
        start = time.monotonic()
        answer = interaction.ask("Ready to start", ["Start"], self.cancelled)
        self.assertEqual(answer, (None, "Cancelled"))
        self.assertLess(time.monotonic() - start, 1.0)

    def test_no_screen(self):
        interaction = self.without_screen()
        self.addCleanup(
            setattr,
            interaction_module,
            "SERVER_TIMEOUT",
            interaction_module.SERVER_TIMEOUT,
        )
        interaction_module.SERVER_TIMEOUT = 0.3
        index, message = interaction.ask("Ready to start", ["Start"], self.cancelled)
        self.assertIsNone(index)
        self.assertIn("No screen", message)

    def test_unusable_buttons_are_rejected(self):
        index, message = self.interaction.ask("", ["Yes", "Yes"], self.cancelled)
        self.assertIsNone(index)
        self.assertEqual(message, "the goal was rejected")

    def test_say_without_tts_shows_the_caption(self):
        # The fake TTS of another test may take a moment to leave the graph.
        wait_until(lambda: not self.interaction.speaker.server_is_ready())
        ok, message = self.interaction.say("Hello, I am AhaRobot.", self.cancelled)
        self.assertTrue(ok)
        self.assertIn("no TTS", message)
        wait_until(lambda: self.screen.view()["captions"])
        self.assertEqual(
            self.screen.view()["captions"][-1]["text"], "Hello, I am AhaRobot."
        )

    def test_show_puts_the_image_and_the_caption_on_the_screen(self):
        png = b"\x89PNG\r\n\x1a\nnot decoded"
        wait_until(lambda: not self.interaction.speaker.server_is_ready())
        wait_until(lambda: self.interaction.images.get_subscription_count() > 0)
        ok, message = self.interaction.show(
            png, "png", "This is the cup.", self.cancelled
        )
        self.assertTrue(ok, message)
        wait_until(lambda: self.screen.view()["image"] is not None)
        number = self.screen.view()["image"]["id"]
        self.assertEqual(self.screen.image(number), ("image/png", png))
        wait_until(lambda: self.screen.view()["captions"])
        self.assertEqual(self.screen.view()["captions"][-1]["text"], "This is the cup.")
        self.assertIsNone(self.shown())  # nothing to touch
        self.interaction.hide()
        wait_until(lambda: self.screen.view()["image"] is None)

    def test_show_needs_a_jpeg_or_png(self):
        for image, image_format in ((b"", "png"), (b"BM", "bmp"), (b"x", "")):
            with self.assertRaises(ValueError, msg=image_format):
                self.interaction.show(image, image_format, "Look.", self.cancelled)

    def test_say_waits_for_the_tts(self):
        from aha_msgs.action import Speak

        spoken = []

        def speak(goal_handle):
            spoken.append(goal_handle.request.text)
            goal_handle.succeed()
            return Speak.Result(success=True, message="done")

        tts = Node("test_tts")
        server = ActionServer(tts, Speak, SPEAK_ACTION, speak)
        self.executor.add_node(tts)
        try:
            wait_until(self.interaction.speaker.server_is_ready)
            ok, message = self.interaction.say("Welcome.", self.cancelled)
        finally:
            self.executor.remove_node(tts)
            server.destroy()
            tts.destroy_node()
        self.assertTrue(ok, message)
        self.assertEqual(message, "TTS: done")
        self.assertEqual(spoken, ["Welcome."])

    def test_listen_fails_without_an_asr(self):
        self.assertFalse(self.interaction.find_asr(self.cancelled))
        text, message = self.interaction.listen(["Charlie"], self.cancelled)
        self.assertIsNone(text)
        self.assertTrue(message.startswith(NO_ASR), message)

    def test_listen_returns_the_utterance(self):
        from aha_msgs.action import Listen

        replies = [(True, " Yes, I am Charlie. ", ""), (False, "", "no speech")]
        goals = []

        def listen(goal_handle):
            request = goal_handle.request
            goals.append((list(request.hotwords), request.language, request.timeout))
            success, text, message = replies.pop(0)
            goal_handle.succeed()
            result = Listen.Result(success=success, message=message)
            result.transcript.text = text
            return result

        asr = Node("test_asr")
        server = ActionServer(asr, Listen, LISTEN_ACTION, listen)
        self.executor.add_node(asr)
        try:
            # Discovered before the first listen (ListenCommand.ready).
            self.assertTrue(self.interaction.find_asr(self.cancelled))
            heard = self.interaction.listen(["Charlie"], self.cancelled, timeout=3.0)
            failed = self.interaction.listen([], self.cancelled)
        finally:
            self.executor.remove_node(asr)
            server.destroy()
            asr.destroy_node()
        self.assertEqual(heard, ("Yes, I am Charlie.", ""))
        self.assertEqual(failed, (None, "ASR: no speech"))
        self.assertEqual(goals, [(["Charlie"], "en", 3.0), ([], "en", 10.0)])

    def test_start_signal_through_the_screen(self):
        skill = WaitStartSignal(SimpleNamespace(node=self.node))
        skill.runtime.interaction = self.interaction
        result = {}
        thread = threading.Thread(
            target=lambda: result.update(value=skill.execute(context()))
        )
        thread.start()
        wait_until(lambda: self.shown() is not None)
        self.assertEqual(self.shown()["choices"], ["Start"])
        wait_until(lambda: self.screen.view()["captions"])
        self.assertEqual(
            self.screen.view()["captions"][-1]["text"],
            WaitStartSignal.TEXT.format(button="Start"),
        )
        self.screen.touch(self.shown()["id"], 0)
        thread.join(10.0)
        self.assertTrue(result["value"].success, result["value"].message)


class FakeFuture:
    def __init__(self):
        self.value, self.callbacks = None, []

    def done(self):
        return self.value is not None

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def set(self, value):
        self.value = value
        for callback in self.callbacks:
            callback(self)


def finished(value):
    future = FakeFuture()
    future.set(value)
    return future


class FakeClient:
    """An action client whose first `lost` goals get no response."""

    def __init__(self, result, lost=1):
        self.result, self.lost = result, lost
        self.ready = True
        self.sent = []  # futures of the goal handles
        self.goals = []
        self.cancelled = []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        future = FakeFuture()
        self.sent.append(future)
        self.goals.append(goal)
        if len(self.sent) > self.lost:
            future.set(self.handle())
        return future

    def handle(self):
        handle = SimpleNamespace(accepted=True)
        handle.get_result_async = lambda: finished(SimpleNamespace(result=self.result))
        handle.cancel_goal_async = lambda: self.cancelled.append(handle)
        return handle


class ResendTests(unittest.TestCase):
    """A goal sent right after discovery, or its response, can be lost."""

    def setUp(self):
        from aha_msgs.action import Ask, Speak
        from std_msgs.msg import String

        for name in ("RESPONSE_TIMEOUT", "SERVER_TIMEOUT"):
            value = getattr(interaction_module, name)
            self.addCleanup(setattr, interaction_module, name, value)
            setattr(interaction_module, name, 0.2)
        self.Ask = Ask
        interaction = Interaction.__new__(Interaction)
        interaction.node = SimpleNamespace(context=SimpleNamespace(ok=lambda: True))
        interaction.Ask, interaction.Speak, interaction.String = Ask, Speak, String
        interaction.captions = SimpleNamespace(publish=lambda message: None)
        interaction.tts_searched = False
        self.interaction = interaction

    def ask(self):
        return self.interaction.ask("Ready to start", ["Start"], threading.Event())

    def test_an_unanswered_question_is_sent_again(self):
        touched = self.Ask.Result(success=True, index=0, choice="Start")
        screen = self.interaction.screen = FakeClient(touched)
        self.assertEqual(self.ask(), (0, "Start"))
        self.assertEqual(len(screen.sent), 2)
        # The same request, so the screen answers it with a touch of the first.
        self.assertTrue(screen.goals[0].request)
        self.assertEqual(screen.goals[1].request, screen.goals[0].request)
        self.assertEqual(self.ask(), (0, "Start"))  # the next question: a new one
        self.assertNotEqual(screen.goals[2].request, screen.goals[0].request)
        screen.sent[0].set(screen.handle())  # a late response to the first
        self.assertEqual(len(screen.cancelled), 1)

    def test_resending_stops_when_the_screen_goes_away(self):
        screen = self.interaction.screen = FakeClient(None, lost=math.inf)
        timer = threading.Timer(0.3, setattr, (screen, "ready", False))
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(self.ask(), (None, "No screen: /aha/display/ask went away"))
        self.assertEqual(len(screen.sent), 2)

    def test_say_waits_for_a_tts_still_being_discovered(self):
        from aha_msgs.action import Speak

        done = Speak.Result(success=True, message="done")
        speaker = self.interaction.speaker = FakeClient(done, lost=0)
        speaker.ready = False
        timer = threading.Timer(0.1, setattr, (speaker, "ready", True))
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(
            self.interaction.say("Welcome.", threading.Event()), (True, "TTS: done")
        )

    def test_without_a_tts_only_the_first_say_waits(self):
        self.addCleanup(
            setattr,
            interaction_module,
            "TTS_DISCOVERY",
            interaction_module.TTS_DISCOVERY,
        )
        interaction_module.TTS_DISCOVERY = 0.3
        speaker = self.interaction.speaker = FakeClient(None)
        speaker.ready = False
        for most, least in ((1.0, 0.3), (0.1, 0.0)):
            started = time.monotonic()
            ok, message = self.interaction.say("Welcome.", threading.Event())
            elapsed = time.monotonic() - started
            self.assertEqual(
                (ok, message), (True, "Shown on the screen; no TTS is running")
            )
            self.assertGreaterEqual(elapsed, least)
            self.assertLess(elapsed, most)
        self.assertEqual(speaker.sent, [])

    def test_speech_is_not_sent_again(self):
        speaker = self.interaction.speaker = FakeClient(None)
        ok, message = self.interaction.say("Welcome.", threading.Event())
        self.assertEqual((ok, message), (False, "TTS: no answer from the server"))
        self.assertEqual(len(speaker.sent), 1)


class FakeGoal:
    """An Ask goal handle; `checking` runs when the bridge checks for a cancel."""

    def __init__(self, choices, timeout=0.0, cancel=False, checking=None, request=""):
        from aha_msgs.action import Ask

        self.request = Ask.Goal(
            prompt="", choices=choices, timeout=timeout, request=request
        )
        self.cancel = cancel
        self.checking = checking or (lambda: None)
        self.ended = None

    @property
    def is_cancel_requested(self):
        self.checking()
        return self.cancel

    def succeed(self):
        self.ended = "succeeded"

    def abort(self):
        self.ended = "aborted"

    def canceled(self):
        self.ended = "canceled"


class RaceTests(unittest.TestCase):
    """A touch right before a timeout or cancel ends the goal: the touch wins."""

    def bridge(self):
        from aha_msgs.action import Ask

        bridge = ScreenBridge.__new__(ScreenBridge)
        bridge.screen = Screen()
        bridge.Ask = Ask
        bridge.node = SimpleNamespace(context=SimpleNamespace(ok=lambda: True))
        return bridge

    def touch_now(self, bridge):
        def touch():
            question = bridge.screen.view()["question"]
            if question:
                bridge.screen.touch(question["id"], 1)

        return touch

    def test_a_resent_goal_gets_the_touch_of_its_first_try(self):
        """The client never heard of the first goal; the referee touched it."""
        bridge = self.bridge()
        first = FakeGoal(
            ["Yes", "No"], checking=self.touch_now(bridge), request="ask-1"
        )
        self.assertEqual(bridge._answer(first).choice, "No")
        again = FakeGoal(["Yes", "No"], request="ask-1")
        result = bridge._answer(again)
        self.assertEqual(
            (again.ended, result.success, result.choice), ("succeeded", True, "No")
        )
        self.assertIsNone(bridge.screen.view()["question"])

    def test_a_touch_at_the_timeout(self):
        bridge = self.bridge()
        goal = FakeGoal(["Yes", "No"], timeout=1e-6, checking=self.touch_now(bridge))
        result = bridge._answer(goal)
        self.assertEqual(
            (goal.ended, result.success, result.choice), ("succeeded", True, "No")
        )

    def test_a_touch_at_the_cancel(self):
        bridge = self.bridge()
        goal = FakeGoal(["Yes", "No"], cancel=True, checking=self.touch_now(bridge))
        result = bridge._answer(goal)
        self.assertEqual((goal.ended, result.index), ("succeeded", 1))

    def test_no_touch_at_the_timeout(self):
        bridge = self.bridge()
        goal = FakeGoal(["Yes"], timeout=1e-6)
        result = bridge._answer(goal)
        self.assertEqual((goal.ended, result.success), ("aborted", False))
        self.assertIsNone(bridge.screen.view()["question"])


# A screen in its own process, to be killed like a crashed task monitor.
# It prints "asked" once a question is on the screen.
SCREEN_PROCESS = """
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from aha_tasks.display import Screen, ScreenBridge


def report(hub):
    version = None
    while True:
        version, payload = hub.wait(version, None)
        if payload["screen"]["question"]:
            print("asked", flush=True)
            return


rclpy.init()
node = Node("test_crashing_display")
screen = Screen()
threading.Thread(target=report, args=(screen.hub,), daemon=True).start()
ScreenBridge(node, screen)
executor = MultiThreadedExecutor()
executor.add_node(node)
executor.spin()
"""


def read_line(stream, timeout):
    ready, _, _ = select.select([stream], [], [], timeout)
    assert ready, "Timed out"
    return stream.readline().strip()


class ShutdownTests(unittest.TestCase):
    """The task monitor stops while the start button waits for a touch."""

    def test_a_crashed_screen_ends_the_question(self):
        self.addCleanup(
            setattr,
            interaction_module,
            "SERVER_TIMEOUT",
            interaction_module.SERVER_TIMEOUT,
        )
        interaction_module.SERVER_TIMEOUT = 1.0
        source = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID))
        env["PYTHONPATH"] = os.pathsep.join(
            [source] + [p for p in [env.get("PYTHONPATH")] if p]
        )
        screen = subprocess.Popen(
            [sys.executable, "-c", SCREEN_PROCESS],
            env=env,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(screen.stdout.close)
        self.addCleanup(screen.wait, 10.0)
        self.addCleanup(screen.kill)
        rclpy.init(domain_id=DOMAIN_ID)
        self.addCleanup(rclpy.try_shutdown)
        node = Node("test_crash_runner")
        interaction = Interaction(node)
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()
        self.addCleanup(executor.shutdown)
        wait_until(interaction.screen.server_is_ready, 20.0)
        answer = {}
        thread = threading.Thread(
            target=lambda: answer.update(
                value=interaction.ask("Ready to start", ["Start"], threading.Event())
            )
        )
        thread.start()
        # Sent right after discovery, the goal may be lost and then resent.
        self.assertEqual(read_line(screen.stdout, 20.0), "asked")
        self.assertNotIn("value", answer)
        screen.kill()  # SIGKILL: no abort reaches the runner
        start = time.monotonic()
        # The server leaves the graph when its DDS lease (about 20 s) runs out.
        thread.join(60.0)
        self.assertEqual(
            answer.get("value"), (None, "No screen: /aha/display/ask went away")
        )
        self.assertLess(time.monotonic() - start, 40.0)

    def test_the_runner_gets_the_abort_and_the_executor_stops(self):
        rclpy.init(domain_id=DOMAIN_ID)
        self.addCleanup(rclpy.try_shutdown)
        screen = Screen()
        display_node = Node("test_stopping_display")
        bridge = ScreenBridge(display_node, screen)
        node = Node("test_stopping_runner")
        interaction = Interaction(node)
        # Separate executors, as the monitor and the runner are separate processes.
        executor = MultiThreadedExecutor()
        executor.add_node(display_node)
        runner = MultiThreadedExecutor()
        runner.add_node(node)
        spinners = [
            threading.Thread(target=executor.spin, daemon=True),
            threading.Thread(target=runner.spin, daemon=True),
        ]
        for spinner in spinners:
            spinner.start()
        answer = {}
        thread = threading.Thread(
            target=lambda: answer.update(
                value=interaction.ask("Ready to start", ["Start"], threading.Event())
            )
        )
        thread.start()
        wait_until(lambda: screen.view()["question"] is not None)
        start = time.monotonic()
        bridge.close()  # as task_monitor does before executor.shutdown()
        executor.shutdown()
        self.assertLess(time.monotonic() - start, 2.0)
        thread.join(10.0)
        self.assertEqual(answer["value"], (None, "The screen is shutting down"))
        runner.shutdown()
        for spinner in spinners:
            spinner.join(timeout=5.0)
        bridge.server.destroy()
        node.destroy_node()
        display_node.destroy_node()


if __name__ == "__main__":
    unittest.main()
