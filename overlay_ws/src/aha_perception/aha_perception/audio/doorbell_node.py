"""aha_audio_doorbell: microphone -> BEATs-SED (GPU) -> doorbell / knock events.

Topics
  /aha/perception/audio/doorbell    aha_msgs/SoundEvent           start of each detection;
                                    the publisher exists only while the node listens
  /aha/perception/audio/models      std_msgs/String (JSON)        loaded model (latched)
  /aha/perception/audio/tts_active  std_msgs/Bool (sub)           ignore the audio while True
  /aha/sim/audio                    std_msgs/Float32MultiArray (sub, input: topic)
                                    16 kHz mono stream of the simulation
Parameters
  input (mic | topic), device, input_device, capture_rate, mute_tail_ms. The model,
  window, classes and thresholds are the `doorbell.beats_sed` entry of models.yaml.
  Inference runs in this process (no `target`).
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
from aha_msgs.msg import SoundEvent
from rclpy.time import Time
from std_msgs.msg import Bool, Float32MultiArray

from .audio_io import SAMPLE_RATE, STALL_S, MicStream
from .common import TOPIC_TTS_ACTIVE, AudioModelNode, spin_multithreaded
from .doorbell import HOP, LABELS, MODEL, BeatsSed, Rule, Window

TOPIC_DOORBELL = "/aha/perception/audio/doorbell"
TOPIC_SIM_AUDIO = "/aha/sim/audio"
FRAME = int(HOP * SAMPLE_RATE)


class TopicStream:
    """MicStream's interface over the simulated audio topic. The simulation may
    start after this node and restart while it runs, so a silent topic does not
    end the node: `on_flow(True)` when the audio starts, `on_flow(False)` when none
    has come for STALL_S."""

    def __init__(self, node, on_flow):
        self._q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=200)
        self._last = time.monotonic()
        self._flowing = False
        self._on_flow = on_flow
        self._log = node.get_logger()
        node.create_subscription(Float32MultiArray, TOPIC_SIM_AUDIO, self._on_audio, 50)

    def _on_audio(self, msg):
        try:
            self._q.put_nowait((np.asarray(msg.data, np.float32), time.time()))
        except queue.Full:
            pass

    def start(self):
        self._last = time.monotonic()

    def close(self):
        pass

    def frames(self, stop: threading.Event):
        pending = np.zeros(0, np.float32)
        while not stop.is_set():
            try:
                block, t = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._flowing and time.monotonic() - self._last > STALL_S:
                    self._flowing = False
                    pending = np.zeros(0, np.float32)
                    self._log.warn(f"no audio on {TOPIC_SIM_AUDIO}; not listening")
                    self._on_flow(False)
                continue
            self._last = time.monotonic()
            if not self._flowing:
                self._flowing = True
                self._log.info(f"audio on {TOPIC_SIM_AUDIO}; listening")
                self._on_flow(True)
            pending = np.concatenate([pending, block])
            while len(pending) >= FRAME:
                out, pending = pending[:FRAME], pending[FRAME:]
                yield out, t


class DoorbellNode(AudioModelNode):
    def __init__(self):
        super().__init__("aha_audio_doorbell")
        self.declare_parameter("input", "mic")  # mic | topic
        self.declare_parameter("input_device", "")
        self.declare_parameter("capture_rate", SAMPLE_RATE)
        self.declare_parameter("mute_tail_ms", 300)
        self.entry = self.registry.entry("doorbell", MODEL)
        # detect_doorbell takes the publisher as the sign that the node listens:
        # it exists only once the model is loaded and while audio comes in.
        self.pub_event = None
        self._pub_lock = threading.Lock()
        self.create_subscription(Bool, TOPIC_TTS_ACTIVE, self._on_tts_active, 10)
        self._tts_active = False
        self._muted_until = 0.0
        self._stop = threading.Event()
        self._fresh = threading.Condition()
        self._latest = -1.0  # wall time of the newest audio in the window
        self.stream = None

    def start(self) -> None:
        spec = self.entry.spec
        with self.registry.loading(self.entry, self.device) as extra:
            self.engine = BeatsSed(spec, self.device)
            extra["classes"] = spec["classes"]
        self.publish_models()
        self.window = Window(spec["window"])
        self.rules = {target: Rule(**spec["rules"][target]) for target in LABELS}
        source = self.get_parameter("input").value
        if source == "mic":
            self.stream = MicStream(
                self.get_parameter("input_device").value,
                self.get_parameter("capture_rate").value,
                FRAME,
            )
        elif source == "topic":
            self.stream = TopicStream(self, self._listen)
        else:
            raise ValueError(f"input must be mic or topic, not {source!r}")
        self.stream.start()
        if source == "mic":
            self._listen(True)  # a stalled microphone ends the node instead
        for name, loop in (
            ("capture", self._capture_loop),
            ("score", self._score_loop),
        ):
            threading.Thread(
                target=self.watched, args=(loop,), name=name, daemon=True
            ).start()
        self.get_logger().info(
            f"ready: {self.engine.tag} on {source}, window {spec['window']} s"
        )

    def _listen(self, on: bool) -> None:
        """Create or remove the event publisher; old audio leaves the window."""
        with self._pub_lock:
            if on and self.pub_event is None:
                self.pub_event = self.create_publisher(SoundEvent, TOPIC_DOORBELL, 10)
            elif not on and self.pub_event is not None:
                self.destroy_publisher(self.pub_event)
                self.pub_event = None
        with self._fresh:
            self.window.clear()
            self._latest = -1.0  # the score loop resets the rules
            self._fresh.notify_all()

    # ---------------------------------------------------------------- audio
    def _on_tts_active(self, msg: Bool) -> None:
        self._tts_active = msg.data
        if not msg.data:
            self._muted_until = (
                time.time() + self.get_parameter("mute_tail_ms").value / 1000
            )

    def _muted(self) -> bool:
        if self._tts_active and self.count_publishers(TOPIC_TTS_ACTIVE) == 0:
            self._tts_active = False  # the TTS node died while speaking
        return self._tts_active or time.time() < self._muted_until

    def _capture_loop(self) -> None:
        for frame, t in self.stream.frames(self._stop):
            with self._fresh:
                if self._muted():
                    self.window.clear()  # the robot's own voice must not stay in the window
                    self._latest = -1.0
                else:
                    self.window.push(frame)
                    self._latest = t
                self._fresh.notify_all()

    def _score_loop(self) -> None:
        scored = -1.0
        while not self._stop.is_set():
            with self._fresh:
                self._fresh.wait_for(
                    lambda: self._latest != scored or self._stop.is_set(), 0.5
                )
                t, audio = self._latest, self.window.buffer.copy()
            if t == scored:
                continue
            scored = t
            if t < 0:  # muted
                for rule in self.rules.values():
                    rule.reset()
                continue
            scores = self.engine.score(audio)
            for target, rule in self.rules.items():
                smoothed = rule.push(scores[target], t)
                if smoothed is not None:
                    self._publish(target, smoothed, rule.threshold, t)

    def _publish(self, target: str, score: float, threshold: float, t: float) -> None:
        msg = SoundEvent(
            label=LABELS[target],
            score=float(score),
            threshold=float(threshold),
            model=self.engine.tag,
        )
        msg.header.stamp = Time(seconds=t).to_msg()
        msg.header.frame_id = "microphone"
        with self._pub_lock:
            if self.pub_event is None:  # the audio stopped meanwhile
                return
            self.pub_event.publish(msg)
        self.get_logger().info(
            f"{LABELS[target]} (score {score:.3f} >= {threshold:.3f})"
        )

    def destroy_node(self):
        self._stop.set()
        try:
            if self.stream is not None:
                self.stream.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    spin_multithreaded(DoorbellNode, args)


if __name__ == "__main__":
    main()
