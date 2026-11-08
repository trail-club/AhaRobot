"""Silero VAD + Smart Turn による発話区間の切り出し.

32 ms (512 samples @16 kHz) のフレームを push すると、発話が終わった時点で
Segment を返す。「無音が turn_check_ms 続いたら Smart Turn に話し終わりか聞き、
確率が閾値以上なら即確定、そうでなければ max_silence_ms まで待って確定」する。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from .audio_io import SAMPLE_RATE
from .model_registry import Registry

FRAME = 512  # Silero VAD の入力長 (16 kHz)
FRAME_MS = FRAME * 1000 / SAMPLE_RATE  # 32 ms
TURN_WINDOW_S = 8  # Smart Turn が見る末尾の長さ


@dataclass
class SegmenterConfig:
    start_threshold: float = 0.5  # VAD 確率がこれ以上で「声あり」
    end_threshold: float = 0.35  # 発話中、これ未満で「無音」
    min_start_ms: float = 64.0  # 声ありがこれだけ続いたら発話開始
    preroll_ms: float = 300.0  # 発話開始前に遡って含める長さ
    min_speech_ms: float = 250.0  # これより短い発話は捨てる
    turn_check_ms: float = 200.0  # 無音がこれだけ続いたら Smart Turn に聞く
    turn_recheck_ms: float = 300.0  # 以後この間隔で聞き直す
    turn_threshold: float = 0.5  # Smart Turn 確率がこれ以上で確定
    max_silence_ms: float = 1200.0  # Smart Turn が確定しなくてもこの無音で確定
    max_segment_s: float = 15.0  # これ以上長い発話は強制確定
    tail_ms: float = 200.0  # 確定時に残す末尾無音


@dataclass
class Segment:
    audio: np.ndarray  # float32 16 kHz mono
    speech_end_time: float  # wall time of the last voiced frame
    speech_seconds: float
    end_reason: str  # "turn" | "silence" | "max_length"
    turn_prob: float  # last Smart Turn probability (-1 if never asked)
    endpoint_delay: float  # seconds from speech end to segment close


class SmartTurn:
    def __init__(self, registry: Registry):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        entry = registry.entry("turn", "smart_turn")
        with registry.loading(entry, "cpu"):
            path = entry.fetch_file()
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # onnxruntime-gpu の公式 wheel は sm_120 (RTX 50) 非対応の報告があるため CPU 固定。
            self.session = ort.InferenceSession(
                str(path), sess_options=so, providers=["CPUExecutionProvider"]
            )
            self.features = WhisperFeatureExtractor(chunk_length=TURN_WINDOW_S)

    def __call__(self, audio: np.ndarray) -> float:
        inputs = self.features(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=TURN_WINDOW_S * SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        feats = inputs.input_features.squeeze(0).astype(np.float32)[None]
        return float(self.session.run(None, {"input_features": feats})[0][0].item())


class SileroVad:
    def __init__(self, registry: Registry):
        import torch
        from silero_vad import load_silero_vad

        entry = registry.entry("vad", "silero")
        with registry.loading(entry, "cpu"):
            self.model = load_silero_vad()
        self._torch = torch

    def __call__(self, frame: np.ndarray) -> float:
        with self._torch.inference_mode():
            return float(self.model(self._torch.from_numpy(frame), SAMPLE_RATE).item())

    def reset(self) -> None:
        self.model.reset_states()


class Segmenter:
    def __init__(self, vad: SileroVad, turn: SmartTurn, cfg: SegmenterConfig):
        self.vad = vad
        self.turn = turn
        self.cfg = cfg
        self._preroll = collections.deque(maxlen=max(1, int(cfg.preroll_ms / FRAME_MS)))
        self._turn_frames = int(TURN_WINDOW_S * 1000 / FRAME_MS)
        self._muted = False
        self._reset()

    def _reset(self) -> None:
        self.in_speech = False
        self._frames: list[np.ndarray] = []
        self._voiced_run = 0
        self._silence_frames = 0
        self._voiced_frames = 0
        self._last_voice_time = 0.0
        self._next_turn_check_ms = self.cfg.turn_check_ms
        self._turn_prob = -1.0

    def push(
        self, frame: np.ndarray, now: float, muted: bool = False
    ) -> tuple[dict, Segment | None]:
        """Feed one FRAME. Returns (debug info for this frame, finished segment or None).

        `muted` の間 (ロボット自身が話している間など) は VAD を回さず、
        ミュートに入った時点で途中までの発話を捨てる。
        """
        cfg = self.cfg
        info = {
            "rms": float(np.sqrt(np.mean(frame * frame))),
            "peak": float(np.max(np.abs(frame))),
        }
        if muted:
            if not self._muted:
                self._preroll.clear()
                self._reset()
                self.vad.reset()
            self._muted = True
            info.update(vad=0.0, speech=False, muted=True)
            return info, None
        self._muted = False

        prob = self.vad(frame)
        info["vad"] = prob

        if not self.in_speech:
            self._preroll.append(frame)
            self._voiced_run = (
                self._voiced_run + 1 if prob >= cfg.start_threshold else 0
            )
            if self._voiced_run * FRAME_MS >= cfg.min_start_ms:
                self.in_speech = True
                self._frames = list(self._preroll)
                self._preroll.clear()
                self._voiced_frames = self._voiced_run
                self._last_voice_time = now
            info["speech"] = self.in_speech
            return info, None

        self._frames.append(frame)
        if prob >= cfg.end_threshold:
            self._voiced_frames += 1
            self._silence_frames = 0
            self._last_voice_time = now
            self._next_turn_check_ms = cfg.turn_check_ms
        else:
            self._silence_frames += 1

        silence_ms = self._silence_frames * FRAME_MS
        reason = None
        if silence_ms >= self._next_turn_check_ms:
            self._turn_prob = self.turn(
                np.concatenate(self._frames[-self._turn_frames :])
            )
            info["turn"] = self._turn_prob
            self._next_turn_check_ms += cfg.turn_recheck_ms
            if self._turn_prob >= cfg.turn_threshold:
                reason = "turn"
        if reason is None and silence_ms >= cfg.max_silence_ms:
            reason = "silence"
        if reason is None and len(self._frames) * FRAME_MS / 1000 >= cfg.max_segment_s:
            reason = "max_length"

        info["speech"] = True
        if reason is None:
            return info, None

        segment = self._finish(reason, now)
        info["speech"] = False
        info["end"] = reason if segment else "too_short"
        return info, segment

    def _finish(self, reason: str, now: float) -> Segment | None:
        cfg = self.cfg
        keep_tail = int(cfg.tail_ms / FRAME_MS)
        frames = self._frames
        if self._silence_frames > keep_tail:
            frames = frames[: len(frames) - (self._silence_frames - keep_tail)]
        speech_seconds = self._voiced_frames * FRAME_MS / 1000
        seg = None
        if speech_seconds * 1000 >= cfg.min_speech_ms:
            seg = Segment(
                audio=np.concatenate(frames),
                speech_end_time=self._last_voice_time,
                speech_seconds=speech_seconds,
                end_reason=reason,
                turn_prob=self._turn_prob,
                endpoint_delay=now - self._last_voice_time,
            )
        self._reset()
        return seg
