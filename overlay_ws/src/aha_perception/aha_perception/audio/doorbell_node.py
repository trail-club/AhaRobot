"""aha_audio_doorbell: microphone -> BEATs-SED (GPU) -> doorbell / knock events.

Topics
  /aha/perception/audio/doorbell    aha_msgs/SoundEvent           start of each detection
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
    """MicStream's interface over the simulated audio topic. A silent topic only
    warns: the simulation may start after this node."""

    def __init__(self, node):
        self._q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=200)
        self._last = time.monotonic()
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
                if time.monotonic() - self._last > STALL_S:
                    self._log.warn(
                        f"no audio on {TOPIC_SIM_AUDIO}", throttle_duration_sec=10.0
                    )
                continue
            self._last = time.monotonic()
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
        self.pub_event = None  # created in start(), once the node can detect
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
            self.stream = TopicStream(self)
        else:
            raise ValueError(f"input must be mic or topic, not {source!r}")
        self.stream.start()
        # detect_doorbell takes a publisher as the sign that the detector listens.
        self.pub_event = self.create_publisher(SoundEvent, TOPIC_DOORBELL, 10)
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
