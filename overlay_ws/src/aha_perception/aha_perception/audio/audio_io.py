"""Microphone capture / speaker playback via sounddevice (PortAudio).

コンテナでは ALSA の default がホストの PulseAudio / PipeWire に向く
(docker/audio/asound.conf)。デバイスは名前の部分一致か index で指定できる。
"""

from __future__ import annotations

import queue
import threading
import time
from math import gcd

import numpy as np

SAMPLE_RATE = 16000                           # ASR / VAD パイプラインのサンプルレート


def _device(spec: str):
    if spec == '':
        return None
    return int(spec) if spec.isdigit() else spec


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


class MicStream:
    """Delivers fixed-size float32 mono frames at SAMPLE_RATE with capture timestamps."""

    def __init__(self, device: str, capture_rate: int, frame: int):
        import sounddevice as sd

        self.capture_rate = capture_rate
        self.frame = frame
        self._q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=200)
        self.dropped = 0
        block = int(round(frame * capture_rate / SAMPLE_RATE))
        self._stream = sd.InputStream(
            device=_device(device), samplerate=capture_rate, channels=1, dtype='float32',
            blocksize=block, callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        try:
            self._q.put_nowait((indata[:, 0].copy(), time.time()))
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def frames(self, stop: threading.Event):
        """Yield (frame, wall_time_of_frame_end) until `stop` is set."""
        pending = np.zeros(0, dtype=np.float32)
        while not stop.is_set():
            try:
                block, t = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            pending = np.concatenate([pending, resample(block, self.capture_rate, SAMPLE_RATE)])
            while len(pending) >= self.frame:
                out, pending = pending[:self.frame], pending[self.frame:]
                yield out, t


class Speaker:
    """Blocking playback in small blocks so that it can be interrupted."""

    def __init__(self, device: str, playback_rate: int, block_ms: int = 50):
        import sounddevice as sd

        self.rate = playback_rate
        self.block = int(playback_rate * block_ms / 1000)
        self._stream = sd.OutputStream(
            device=_device(device), samplerate=playback_rate, channels=1, dtype='float32')
        self._stream.start()

    def play(self, audio: np.ndarray, src_rate: int, cancelled) -> bool:
        """Play `audio`; returns False if `cancelled()` became true midway."""
        audio = resample(audio.astype(np.float32), src_rate, self.rate)
        for i in range(0, len(audio), self.block):
            if cancelled():
                return False
            self._stream.write(audio[i:i + self.block])
        return True

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
