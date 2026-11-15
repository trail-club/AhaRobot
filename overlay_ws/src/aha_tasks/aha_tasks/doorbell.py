"""Skill on aha_perception's doorbell detector: wait for the doorbell or a knock.

aha_audio_doorbell publishes aha_msgs/SoundEvent on /aha/perception/audio/doorbell.
The skill succeeds on the first event that arrives after it starts (rulebook 5.1:
the robot waits at the start position and reacts at once) and fails after
`timeout` s. Times are monotonic wall seconds: the audio runs in real time even in
simulation.
"""

import threading
import time

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.people import check_keys, positive

DOORBELL_TOPIC = "/aha/perception/audio/doorbell"
LABELS = ("doorbell", "knock")
KEEP = 20  # events remembered


class Doorbell:
    """ROS access to the detector, shared by the skills of one runner."""

    def __init__(self, node):
        from aha_msgs.msg import SoundEvent

        self.node = node
        self.changed = threading.Condition()
        self.events = []  # (monotonic receive time, label, score)
        node.create_subscription(SoundEvent, DOORBELL_TOPIC, self.on_event, 10)

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "doorbell", None) is None:
            runtime.doorbell = cls(runtime.node)
        return runtime.doorbell

    def on_event(self, message):
        with self.changed:
            self.events.append((time.monotonic(), message.label, float(message.score)))
            del self.events[:-KEEP]
            self.changed.notify_all()

    def detectors(self):
        return self.node.count_publishers(DOORBELL_TOPIC)

    def wait(self, since, labels, deadline, cancelled):
        """The first (time, label, score) after `since` with a wanted label, or None."""
        with self.changed:
            while True:
                for event in self.events:
                    if event[0] >= since and event[1] in labels:
                        return event
                left = deadline - time.monotonic()
                if left <= 0 or cancelled.is_set():
                    return None
                self.changed.wait(min(left, 0.1))


class DetectDoorbell(Skill):
    """Wait at the start position for the doorbell (or a knock).

    params: {"timeout": 120.0, "labels": ["doorbell", "knock"]}; both optional.
    Output: `doorbell.heard` = the label that was heard.
    """

    @classmethod
    def validate(cls, params, spec):
        check_keys(params, (), ("timeout", "labels"))
        positive(params, "timeout", 120.0)
        labels = params.get("labels", list(LABELS))
        if not isinstance(labels, list) or not labels or set(labels) - set(LABELS):
            raise ValueError(
                f"labels must be a nonempty subset of {list(LABELS)}: {labels!r}"
            )

    def execute(self, ctx):
        bell = Doorbell.of(self.runtime)
        if bell.detectors() == 0:
            return SkillResult(
                False,
                f"No publisher on {DOORBELL_TOPIC}; is aha_audio_doorbell running?",
            )
        timeout = positive(ctx.params, "timeout", 120.0)
        labels = ctx.params.get("labels", list(LABELS))
        start = time.monotonic()
        ctx.log(f"Waiting up to {timeout:.0f} s for: {', '.join(labels)}")
        event = bell.wait(start, labels, start + timeout, ctx.cancelled)
        if event is None:
            if ctx.cancelled.is_set():
                return SkillResult(False, "Cancelled")
            return SkillResult(
                False, f"No {' or '.join(labels)} within {timeout:.0f} s"
            )
        heard_at, label, score = event
        return SkillResult(
            True,
            f"Heard {label} (score {score:.2f}) after {heard_at - start:.1f} s",
            {"doorbell.heard": label},
        )
