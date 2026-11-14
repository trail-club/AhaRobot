"""The robot's touch screen: speech captions, one image and touch questions.

`Screen` holds what the screen shows, without ROS, as the "screen" entry of a
StatusHub; the task monitor shares its own hub, serves the screen under
/display/ and pushes every change to the pages by SSE. `ScreenBridge` is the
ROS side: the Ask action, the caption topic and the image topic.
"""

from collections import deque
import itertools
import math
import threading
import time

from aha_tasks.monitor import StatusHub

ASK_ACTION = "/aha/display/ask"
CAPTION_TOPIC = "/aha/display/caption"
IMAGE_TOPIC = "/aha/display/image"
MAX_CHOICES = 6
CAPTION_LINES = 3  # latest utterances kept on the screen
POLL = 0.05  # s between checks for cancel and timeout while a question is open
CLOSE_TIMEOUT = 2.0  # s to wait for open Ask goals when the screen closes
RESULT_GRACE = 0.2  # s for the executor to send the results of ended goals
# CompressedImage.format contains one of these, e.g. "jpeg" or "bgr8; png compressed".
IMAGE_TYPES = (("png", "image/png"), ("jpeg", "image/jpeg"), ("jpg", "image/jpeg"))
REPLACED = "Replaced by another question"
CLOSED = "The screen is shutting down"


def check_question(choices):
    """Why the screen cannot show these buttons, or None."""
    if not 1 <= len(choices) <= MAX_CHOICES:
        return f"needs 1 to {MAX_CHOICES} choices, got {len(choices)}"
    if not all(choice.strip() for choice in choices):
        return "choices must be nonempty"
    if len(set(choices)) != len(choices):
        return "choices must differ"
    return None


def image_type(compressed_format):
    """MIME type for a CompressedImage.format, or None if browsers cannot show it."""
    name = compressed_format.lower()
    for key, mime in IMAGE_TYPES:
        if key in name:
            return mime
    return None


class Question:
    def __init__(self, number, prompt, choices):
        self.id = number
        self.prompt = prompt
        self.choices = choices
        self.done = threading.Event()  # touched, or ended without a touch
        self.index = None
        self.ended = None  # why it ended without a touch

    def end(self, why):
        self.ended = why
        self.done.set()

    def view(self):
        return {"id": self.id, "prompt": self.prompt, "choices": self.choices}


class Screen:
    """What the screen shows; every change updates hub.payload["screen"]."""

    def __init__(self, hub=None, lines=CAPTION_LINES):
        self.hub = hub if hub is not None else StatusHub()
        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        self._question = None
        self._open = True
        self._captions = deque(maxlen=lines)
        self._image = None  # (number, MIME type, bytes)
        self._publish()

    def view(self):
        """{"question", "captions", "image"}: what the page shows."""
        return self.hub.payload["screen"]

    def _publish(self):
        self.hub.update(
            screen={
                "question": self._question.view() if self._question else None,
                "captions": list(self._captions),
                "image": {"id": self._image[0]} if self._image else None,
            }
        )

    def say(self, text):
        """Add a caption of the robot's utterance; an empty text clears them."""
        text = text.strip()
        with self._lock:
            if text:
                self._captions.append({"id": next(self._numbers), "text": text})
            else:
                self._captions.clear()
            self._publish()

    def show_image(self, data, mime):
        """Show one image (bytes of `mime`); empty data removes it."""
        with self._lock:
            self._image = (next(self._numbers), mime, bytes(data)) if data else None
            self._publish()

    def image(self, number):
        """(MIME type, bytes) while image `number` is on the screen, else None."""
        with self._lock:
            if self._image is None or self._image[0] != number:
                return None
            return self._image[1], self._image[2]

    def ask(self, prompt, choices):
        """Show a question; the one on the screen ends as replaced.

        After `shutdown` the question ends at once.
        """
        choices = list(choices)
        error = check_question(choices)
        if error:
            raise ValueError(error)
        with self._lock:
            question = Question(next(self._numbers), prompt.strip(), choices)
            if not self._open:
                question.end(CLOSED)
                return question
            if self._question is not None:
                self._question.end(REPLACED)
            self._question = question
            self._publish()
            return question

    def shutdown(self):
        """End the question on the screen and refuse new ones.

        Call before stopping the executor: an Ask goal waits for its question.
        """
        with self._lock:
            self._open = False
            if self._question is not None:
                self._question.end(CLOSED)
                self._question = None
                self._publish()

    def close(self, question):
        """Remove `question` if it is still on the screen; True if it was.

        False means it already ended: touched, replaced or the screen closed.
        """
        with self._lock:
            if self._question is not question:
                return False
            self._question = None
            self._publish()
            return True

    def touch(self, number, index):
        """A touch on button `index` of question `number`: (accepted, message)."""
        with self._lock:
            question = self._question
            if question is None or question.id != number:
                return False, "This question is no longer on the screen"
            if not 0 <= index < len(question.choices):
                return False, f"No button {index}"
            question.index = index
            question.done.set()
            self._question = None
            self._publish()
            return True, question.choices[index]


class ScreenBridge:
    """ROS side of `screen`. Ask goals block, so spin with a multi-threaded executor."""

    def __init__(self, node, screen):
        from aha_msgs.action import Ask
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        self.node = node
        self.screen = screen
        self.Ask = Ask
        self.GoalResponse = GoalResponse
        self._goals = 0  # Ask goals being executed
        self._idle = threading.Condition()
        self.server = ActionServer(
            node,
            Ask,
            ASK_ACTION,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        node.create_subscription(
            String, CAPTION_TOPIC, lambda message: screen.say(message.data), 10
        )
        node.create_subscription(CompressedImage, IMAGE_TOPIC, self._on_image, 1)

    def close(self):
        """End the open questions and let their results reach the clients.

        Call while the executor still spins, before executor.shutdown(): an Ask
        goal waits for its question, and rclpy sends the result of an ended goal
        in a later executor task, which shutdown would drop.
        """
        self.screen.shutdown()
        with self._idle:
            self._idle.wait_for(lambda: self._goals == 0, CLOSE_TIMEOUT)
        time.sleep(RESULT_GRACE)

    def _on_image(self, message):
        if not message.data:
            self.screen.show_image(b"", None)
            return
        mime = image_type(message.format)
        if mime is None:
            self.node.get_logger().warning(
                f"Ignoring an image in format {message.format!r}; use jpeg or png"
            )
            return
        self.screen.show_image(message.data, mime)

    def _on_goal(self, goal):
        error = check_question(list(goal.choices))
        if error:
            self.node.get_logger().warning(f"Rejecting a question: {error}")
            return self.GoalResponse.REJECT
        return self.GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        with self._idle:
            self._goals += 1
        try:
            return self._answer(goal_handle)
        finally:
            with self._idle:
                self._goals -= 1
                self._idle.notify_all()

    def _answer(self, goal_handle):
        goal = goal_handle.request
        result = self.Ask.Result(index=-1)
        question = self.screen.ask(goal.prompt, goal.choices)
        timeout = float(goal.timeout)
        deadline = time.monotonic() + timeout if timeout > 0 else math.inf
        try:
            while not question.done.wait(POLL):
                if goal_handle.is_cancel_requested:
                    why = None
                elif not self.node.context.ok():
                    why = CLOSED
                elif time.monotonic() >= deadline:
                    why = f"No touch within {timeout:g} s"
                else:
                    continue
                # A touch may have come since the check; the screen's lock decides.
                if not self.screen.close(question):
                    break
                if why is None:
                    goal_handle.canceled()
                    result.message = "Cancelled"
                else:
                    goal_handle.abort()
                    result.message = why
                return result
        finally:
            self.screen.close(question)
        if question.ended:
            goal_handle.abort()
            result.message = question.ended
            return result
        goal_handle.succeed()
        result.success = True
        result.index = question.index
        result.choice = question.choices[question.index]
        return result
