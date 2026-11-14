"""Speaking and asking through the robot's speaker and touch screen.

`say` shows the text as a caption on the screen (aha_tasks.display) and speaks
it when the TTS (/aha/perception/audio/speak) is up; the simulation has no TTS,
so there it only shows the caption. `ask` shows buttons on the screen and waits
for a touch.
"""

import math
import time

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.display import ASK_ACTION, CAPTION_TOPIC

SPEAK_ACTION = "/aha/perception/audio/speak"
SERVER_TIMEOUT = 5.0  # s (wall clock) to find a server, or to give up on a lost one
LIVENESS_PERIOD = 0.25  # s between checks that a server still runs
RESPONSE_TIMEOUT = 5.0  # s for a server to accept or reject a goal
SPEAK_TIMEOUT = 60.0  # s for one utterance, including other queued utterances


class Cancelled(Exception):
    pass


class Lost(Exception):
    """The server of a goal left the ROS graph, e.g. the task monitor crashed."""


class Interaction:
    """Speech and touch shared by the skills of one runner; see `of`."""

    def __init__(self, node):
        from aha_msgs.action import Ask, Speak
        from rclpy.action import ActionClient
        from std_msgs.msg import String

        self.node = node
        self.Ask, self.Speak, self.String = Ask, Speak, String
        self.captions = node.create_publisher(String, CAPTION_TOPIC, 10)
        self.speaker = ActionClient(node, Speak, SPEAK_ACTION)
        self.screen = ActionClient(node, Ask, ASK_ACTION)

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "interaction", None) is None:
            runtime.interaction = cls(runtime.node)
        return runtime.interaction

    def say(self, text, cancelled):
        """Caption `text`, and speak it if a TTS runs: (ok, message) when done."""
        self.captions.publish(self.String(data=text))
        if not self.speaker.server_is_ready():
            return True, "Shown on the screen; no TTS is running"
        result, message = self._run(
            self.speaker, self.Speak.Goal(text=text), SPEAK_TIMEOUT, cancelled
        )
        if result is None:
            return False, f"TTS: {message}"
        return result.success, f"TTS: {result.message}"

    def ask(self, prompt, choices, cancelled, timeout=0.0):
        """Buttons on the screen: (index of the touched one, its label).

        Without a touch the index is None and the label the reason. `timeout` > 0
        gives up after that many seconds (wall clock).
        """
        try:
            self._poll(self.screen.server_is_ready, SERVER_TIMEOUT, cancelled)
        except Cancelled:
            return None, "Cancelled"
        except TimeoutError:
            return None, f"No screen: {ASK_ACTION} is not served"
        goal = self.Ask.Goal(
            prompt=prompt, choices=list(choices), timeout=float(timeout)
        )
        limit = timeout + RESPONSE_TIMEOUT if timeout > 0 else math.inf
        result, message = self._run(self.screen, goal, limit, cancelled)
        if message == LOST:
            return None, f"No screen: {ASK_ACTION} went away"
        if result is None:
            return None, message
        if not result.success:
            return None, result.message
        return result.index, result.choice

    def _run(self, client, goal, timeout, cancelled):
        """(result, "") of `goal`, or (None, why); the goal is cancelled on giving up."""
        sent = client.send_goal_async(goal)
        try:
            handle = self._wait(sent, RESPONSE_TIMEOUT, cancelled)
        except (Cancelled, TimeoutError) as error:
            sent.add_done_callback(cancel_if_accepted)
            return None, reason(error)
        if not handle.accepted:
            return None, "the goal was rejected"
        # A server that dies never answers, so give up once it is gone.
        try:
            response = self._wait(
                handle.get_result_async(), timeout, cancelled, client.server_is_ready
            )
        except Lost:
            return None, LOST
        except (Cancelled, TimeoutError) as error:
            handle.cancel_goal_async()
            return None, reason(error)
        return response.result, ""

    def _wait(self, future, timeout, cancelled, alive=None):
        self._poll(future.done, timeout, cancelled, alive)
        return future.result()

    def _poll(self, done, timeout, cancelled, alive=None):
        """Return once `done()`; raise Cancelled, TimeoutError or Lost before that.

        Lost: `alive()` stayed false for SERVER_TIMEOUT.
        """
        now = time.monotonic()
        deadline = now + timeout
        checked, last_alive = now, now
        while not done():
            if cancelled.is_set():
                raise Cancelled
            if not self.node.context.ok():
                raise RuntimeError("ROS shut down")
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError
            if alive is not None and now - checked >= LIVENESS_PERIOD:
                checked = now
                if alive():
                    last_alive = now
                elif now - last_alive >= SERVER_TIMEOUT:
                    raise Lost
            time.sleep(0.02)


def cancel_if_accepted(future):
    handle = future.result()
    if handle is not None and handle.accepted:
        handle.cancel_goal_async()


LOST = "the server went away"


def reason(error):
    return "Cancelled" if isinstance(error, Cancelled) else "no answer from the server"


def text(params, key, default=None):
    value = params.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty text: {params}")
    return value.strip()


class Say(Skill):
    """Say a text: a caption on the screen, and speech with a TTS.

    params: {"text": "<English sentence>"}
    """

    @classmethod
    def validate(cls, params, spec):
        if set(params) != {"text"}:
            raise ValueError(f"say needs exactly a text: {params}")
        text(params, "text")

    def ready(self):
        Interaction.of(self.runtime)  # finds the TTS before the first sentence

    def execute(self, ctx):
        ok, message = Interaction.of(self.runtime).say(
            text(ctx.params, "text"), ctx.cancelled
        )
        return SkillResult(ok, message)


class WaitStartSignal(Skill):
    """Manual start signal (rulebook 3.6.8) as a button on the touch screen.

    Says that the robot is ready and what to touch, then waits for the touch.
    params (optional): {"text": "<announcement>", "button": "<label>"}
    """

    TEXT = "I am ready to start. Please touch the Start button on my screen."
    BUTTON = "Start"
    PROMPT = "Ready to start"

    @classmethod
    def validate(cls, params, spec):
        unknown = set(params) - {"text", "button"}
        if unknown:
            raise ValueError(f"wait_start_signal may have text and button: {params}")
        text(params, "text", cls.TEXT)
        text(params, "button", cls.BUTTON)

    def ready(self):
        Interaction.of(self.runtime)

    def execute(self, ctx):
        interaction = Interaction.of(self.runtime)
        button = text(ctx.params, "button", self.BUTTON)
        ok, message = interaction.say(
            text(ctx.params, "text", self.TEXT), ctx.cancelled
        )
        if ctx.cancelled.is_set():
            return SkillResult(False, "Cancelled")
        if not ok:  # the button still shows what to do
            ctx.log(f"Could not announce the start signal: {message}")
        index, label = interaction.ask(self.PROMPT, [button], ctx.cancelled)
        if index is None:
            return SkillResult(False, label)
        return SkillResult(True, f"Start signal: {label} touched")
