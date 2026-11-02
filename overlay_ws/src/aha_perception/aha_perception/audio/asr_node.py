"""aha_audio_asr: mic -> Silero VAD -> Smart Turn -> ASR (en: Parakeet / ja: Haas).

Topics
  /aha/perception/audio/transcript  aha_msgs/SpeechTranscript   認識結果 (常時)
  /aha/perception/audio/debug       std_msgs/String (JSON)      デバッグ UI 向けイベント
  /aha/perception/audio/models      std_msgs/String (JSON)      読み込んだモデルの記録 (latched)
  /aha/perception/audio/tts_active  std_msgs/Bool (sub)         True の間はマイク入力を捨てる
Actions
  /aha/perception/audio/listen      aha_msgs/action/Listen
Parameters
  language (ja|en|ja-qwen), hotwords (string[]) は実行中に変更可。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from aha_msgs.action import Listen
from aha_msgs.msg import SpeechTranscript
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import GoalResponse
from rclpy.time import Time
from std_msgs.msg import Bool

from .asr_engines import ASREngine, load_engine
from .audio_io import SAMPLE_RATE, MicStream
from .common import (
    ACTION_LISTEN,
    SEGMENTS_DIR,
    TOPIC_TRANSCRIPT,
    TOPIC_TTS_ACTIVE,
    AudioModelNode,
    segment_filename,
    spin_multithreaded,
)
from .remote import load_remote_engine
from .segmenter import FRAME, Segment, Segmenter, SegmenterConfig, SileroVad, SmartTurn


@dataclass
class _ListenRequest:
    language: str
    hotwords: list[str]
    done: threading.Event = field(default_factory=threading.Event)
    transcript: SpeechTranscript | None = None


class AsrNode(AudioModelNode):
    def __init__(self):
        super().__init__("aha_audio_asr")
        self.declare_parameter("language", "en")
        self.declare_parameter("preload_languages", ["en", "ja"])
        self.declare_parameter("hotwords", [""])  # rclpy cannot type an empty list
        self.declare_parameter("input_device", "")
        self.declare_parameter("capture_rate", SAMPLE_RATE)
        self.declare_parameter("listen_timeout", 10.0)
        self.declare_parameter("mute_tail_ms", 300)
        self.declare_parameter("segments_dir", SEGMENTS_DIR)
        self.declare_parameter("keep_segments", 50)
        for name, value in vars(SegmenterConfig()).items():
            self.declare_parameter(f"segmenter.{name}", value)

        self.registry.language(self.language)  # fail fast on typos

        self.pub_transcript = self.create_publisher(
            SpeechTranscript, TOPIC_TRANSCRIPT, 10
        )
        self.create_subscription(Bool, TOPIC_TTS_ACTIVE, self._on_tts_active, 10)
        self._stop = threading.Event()
        self.mic = None

    def start(self) -> None:
        # --- models -------------------------------------------------------
        if self.remote:
            self.start_worker(["asr"])

        import torch

        torch.set_num_threads(1)  # プロセス全体に効く (Silero VAD も ASR エンジンも)
        self.engines: dict[str, ASREngine] = {}
        self._loading = False
        cfg = SegmenterConfig(
            **{
                name: self.get_parameter(f"segmenter.{name}").value
                for name in vars(SegmenterConfig())
            }
        )
        self.segmenter = Segmenter(
            SileroVad(self.registry), SmartTurn(self.registry), cfg
        )
        for lang in self.get_parameter("preload_languages").value:
            self._engine_for(lang)
        engine = self._engine_for(self.language)

        # --- runtime state ------------------------------------------------
        self._segments: queue.Queue[tuple[int, Segment]] = queue.Queue(maxsize=8)
        self._seg_files: deque[str] = deque()
        self._muted_until = 0.0
        self._tts_active = False
        self._recognizing = False
        self._request: _ListenRequest | None = None
        self._request_lock = threading.Lock()
        self.segments_dir = self.get_parameter("segments_dir").value
        os.makedirs(self.segments_dir, exist_ok=True)

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._action = self.serve_action(
            Listen, ACTION_LISTEN, self._execute_listen, self._on_goal
        )

        self.mic = MicStream(
            self.get_parameter("input_device").value,
            self.get_parameter("capture_rate").value,
            FRAME,
        )
        for name, loop in (("capture", self._capture_loop), ("asr", self._asr_loop)):
            threading.Thread(
                target=self.watched, args=(loop,), name=name, daemon=True
            ).start()
        self.mic.start()
        self.create_timer(1.0, self._publish_state)
        self.get_logger().info(f"ready: language={self.language}, asr={engine.tag}")

    @property
    def language(self) -> str:
        return self.get_parameter("language").value

    # ------------------------------------------------------------------ models
    def _engine_for(self, lang: str) -> ASREngine:
        key = self.registry.language(lang)["asr"]
        if key not in self.engines:
            self._loading = True  # Listen の上限時刻を読み込みの分だけ延ばす
            try:
                if self.remote:
                    self.engines[key] = load_remote_engine(
                        self.registry,
                        key,
                        self.worker,
                        self.target,
                        self.startup_timeout,
                        self.request_timeout,
                        self.get_logger(),
                    )
                else:
                    self.engines[key] = load_engine(
                        self.registry, key, self.device, self.get_logger()
                    )
            finally:
                self._loading = False
            self.publish_models()
        return self.engines[key]

    def _loaded_engine(self, lang: str) -> ASREngine | None:
        return self.engines.get(self.registry.language(lang)["asr"])

    def _language_error(self, lang: str) -> str | None:
        try:
            self.registry.language(lang)
        except KeyError as e:
            return str(e)
        return None

    # --------------------------------------------------------------- params
    def _on_set_parameters(self, params) -> SetParametersResult:
        lang = next((p.value for p in params if p.name == "language"), None)
        if lang is None or lang == self.language:
            return SetParametersResult(successful=True)
        if (error := self._language_error(lang)) is not None:
            return SetParametersResult(successful=False, reason=error)
        # 未ロードなら ASR スレッドの初回認識時に読み込まれる
        self.get_logger().info(f"language -> {lang}")
        return SetParametersResult(successful=True)

    def _hotwords(self, extra: list[str]) -> list[str]:
        base = self.get_parameter("hotwords").value or []
        return [w for w in dict.fromkeys([*base, *extra]) if w.strip()]

    # ---------------------------------------------------------------- audio
    def _on_tts_active(self, msg: Bool) -> None:
        self._tts_active = msg.data
        if not msg.data:
            self._muted_until = (
                time.time() + self.get_parameter("mute_tail_ms").value / 1000
            )

    def _muted(self) -> bool:
        return self._tts_active or time.time() < self._muted_until

    def _capture_loop(self) -> None:
        batch, last_flush, seg_id = [], time.time(), 0
        for frame, t in self.mic.frames(self._stop):
            info, seg = self.segmenter.push(frame, t, muted=self._muted())
            info["t"] = round(t, 3)
            batch.append(info)
            if seg is not None:
                seg_id += 1
                try:
                    self._segments.put_nowait((seg_id, seg))
                except queue.Full:
                    self.get_logger().warn("ASR is falling behind; dropping a segment")
            if t - last_flush >= 0.1:
                self.debug(
                    {"type": "frames", "frames": batch, "dropped": self.mic.dropped}
                )
                batch, last_flush = [], t

    def _save_segment(self, seg_id: int, seg: Segment) -> str:
        """デバッグ UI で再生するために WAV を書き、keep_segments 件を超えた古いものを消す."""
        import soundfile as sf

        wav = os.path.join(self.segments_dir, segment_filename(seg_id))
        sf.write(wav, seg.audio, SAMPLE_RATE)
        self._seg_files.append(wav)
        while len(self._seg_files) > self.get_parameter("keep_segments").value:
            try:
                os.remove(self._seg_files.popleft())
            except OSError:
                pass
        return wav

    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            try:
                seg_id, seg = self._segments.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._request_lock:
                req = self._request
            language = req.language if req and req.language else self.language
            hotwords = self._hotwords(req.hotwords if req else [])

            try:
                wav = self._save_segment(seg_id, seg)
            except Exception as e:  # the WAV is for the debug UI only
                self.get_logger().warn(f"could not save segment {seg_id}: {e!r}")
                wav = ""

            self._recognizing = True
            try:
                engine = self._engine_for(language)
                t0 = time.time()
                text = engine.transcribe(seg.audio, language, hotwords)
                asr_seconds = time.time() - t0
            except Exception as e:  # keep listening even if one decode fails
                self.get_logger().error(f"ASR failed on segment {seg_id}: {e!r}")
                self.debug({"type": "asr_error", "id": seg_id, "error": repr(e)})
                continue
            finally:
                self._recognizing = False

            latency = time.time() - seg.speech_end_time
            msg = SpeechTranscript(
                text=text,
                language=self.registry.language_code(language),
                model=engine.tag,
                audio_duration=float(len(seg.audio) / SAMPLE_RATE),
                latency=float(latency),
            )
            msg.header.stamp = Time(seconds=seg.speech_end_time).to_msg()
            msg.header.frame_id = "microphone"
            if text:
                self.pub_transcript.publish(msg)
            self.debug(
                {
                    "type": "asr",
                    "id": seg_id,
                    "text": text,
                    "language": language,
                    "model": engine.tag,
                    "wav": os.path.basename(wav),
                    "speech_seconds": round(seg.speech_seconds, 2),
                    "end_reason": seg.end_reason,
                    "turn_prob": round(seg.turn_prob, 3),
                    "endpoint_delay": round(seg.endpoint_delay, 3),
                    "asr_seconds": round(asr_seconds, 3),
                    "latency": round(latency, 3),
                    "hotwords": hotwords,
                    "hotwords_applied": bool(hotwords) and engine.supports_hotwords,
                }
            )
            self.get_logger().info(
                f'[{language}] "{text}" (latency {latency * 1000:.0f} ms, asr {asr_seconds * 1000:.0f} ms)'
            )
            if req is not None and text:
                req.transcript = msg
                req.done.set()

    # ---------------------------------------------------------------- action
    def _on_goal(self, goal) -> GoalResponse:
        if goal.language and (error := self._language_error(goal.language)) is not None:
            self.get_logger().warn(f"reject listen goal: {error}")
            return GoalResponse.REJECT
        with self._request_lock:
            if self._request is not None:
                self.get_logger().warn("reject listen goal: another listen is active")
                return GoalResponse.REJECT
            self._request = _ListenRequest(goal.language, list(goal.hotwords))
        return GoalResponse.ACCEPT

    def _execute_listen(self, goal_handle):
        with self._request_lock:
            req = self._request
        timeout = goal_handle.request.timeout
        if timeout <= 0:
            timeout = self.get_parameter("listen_timeout").value
        deadline = time.time() + timeout
        # 話している途中は延長するが、雑音で発話判定が続いても終わるよう上限を設ける
        hard_deadline = deadline + self.segmenter.cfg.max_segment_s + 5.0
        result = Listen.Result()
        feedback = Listen.Feedback()
        last = time.time()
        try:
            while not req.done.wait(0.1):
                now = time.time()
                if self._loading:  # 未ロードの言語は初回の認識で読み込む (数十秒かかる)
                    hard_deadline += now - last
                last = now
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.message = "canceled"
                    return result
                feedback.state = (
                    "recognizing"
                    if self._recognizing
                    else "hearing"
                    if self.segmenter.in_speech
                    else "waiting"
                )
                if feedback.state != "waiting":
                    deadline = max(
                        deadline, time.time() + 0.5
                    )  # 話している途中では切らない
                if time.time() > min(deadline, hard_deadline):
                    goal_handle.abort()
                    result.message = f"no transcript within {timeout:.1f}s"
                    return result
                goal_handle.publish_feedback(feedback)
            goal_handle.succeed()
            result.success = True
            result.transcript = req.transcript
            return result
        finally:
            with self._request_lock:
                self._request = None

    # ---------------------------------------------------------------- debug
    def _publish_state(self) -> None:
        with self._request_lock:
            listening = self._request is not None
        self.debug(
            {
                "type": "asr_state",
                "language": self.language,
                "model": getattr(self._loaded_engine(self.language), "tag", None),
                "hotwords": self._hotwords([]),
                "muted": self._muted(),
                "listen_active": listening,
                "languages": self.registry.languages,
            }
        )

    def destroy_node(self):
        self._stop.set()
        try:
            if self.mic is not None:
                self.mic.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    spin_multithreaded(AsrNode, args)


if __name__ == "__main__":
    main()
