"""TF listener that detects /tf stamps going back (sim restart or reset).

On the wall clock the TF buffer cannot see the sim clock going back and keeps
the old, later transforms (rejecting new ones as TF_OLD_DATA), so nodes clear
it when a /tf stamp goes back by more than TF_JUMP_BACK_S for a frame.
"""

from tf2_ros import TransformListener

TF_JUMP_BACK_S = 1.0
TF_RESET_MIN_INTERVAL_S = 5.0


class JumpDetectingTransformListener(TransformListener):
    """TransformListener that reports /tf stamps going back per child frame.

    on_jump() returns True if it reset the buffer; otherwise the message is
    dropped and the old stamps are kept, so the jump is reported again.
    """

    def __init__(self, buffer, node, on_jump, **kwargs):
        self._on_jump = on_jump
        self._latest = {}
        super().__init__(buffer, node, **kwargs)

    def forget(self):
        self._latest.clear()

    def callback(self, data):
        stamps = [
            (t.child_frame_id, t.header.stamp.sec + t.header.stamp.nanosec * 1e-9)
            for t in data.transforms
        ]
        latest = self._latest
        if any(s < latest.get(f, s) - TF_JUMP_BACK_S for f, s in stamps):
            if not self._on_jump():
                return
        latest.update(stamps)
        super().callback(data)
