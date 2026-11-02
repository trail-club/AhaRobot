"""Microphone capture / speaker playback via sounddevice (PortAudio).

コンテナでは ALSA の default がホストの PulseAudio / PipeWire に向く
(docker/audio/asound.conf)。デバイスは名前の部分一致か index で指定できる。
"""

from __future__ import annotations

import base64
import queue
import threading
import time
from math import ceil, gcd

import numpy as np

SAMPLE_RATE = 16000  # ASR / VAD パイプラインのサンプルレート


def encode_pcm(audio: np.ndarray) -> str:
    """Base64 of little-endian float32 samples, for the worker protocol."""
    return base64.b64encode(np.asarray(audio, dtype="<f4").tobytes()).decode("ascii")


def decode_pcm(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text, validate=True), dtype="<f4").astype(
        np.float32
    )


def _device(spec: str):
    if spec == "":
        return None
    return int(spec) if spec.isdigit() else spec


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


class StreamResampler:
    """Resamples consecutive blocks as one signal.

    resample_poly treats the input as zero outside what it is given, so resampling
    each block alone leaves a click at every block edge. This holds back the samples
    the filter still needs and returns only output it computed from real neighbours.
    """

    def __init__(self, src: int, dst: int):
        g = gcd(src, dst)
        self.up, self.down = dst // g, src // g
        half_filter = 10 * max(self.up, self.down)  # resample_poly's default
        # A whole number of `down` steps, so that the output lines up exactly.
        self.context = ceil(half_filter / self.up / self.down) * self.down
        self.pending = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> np.ndarray:
        if self.up == self.down:
            return block
        from scipy.signal import resample_poly

        self.pending = np.concatenate([self.pending, block])
        edges = 2 * self.context
        usable = (len(self.pending) - edges) // self.down * self.down
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        out = resample_poly(self.pending[: usable + edges], self.up, self.down)
        self.pending = self.pending[usable:]
        start = self.context * self.up // self.down
        return out[start : start + usable * self.up // self.down].astype(np.float32)


class MicStream:
    """Delivers fixed-size float32 mono frames at SAMPLE_RATE with capture timestamps."""

    def __init__(self, device: str, capture_rate: int, frame: int):
        import sounddevice as sd

        self.resampler = StreamResampler(capture_rate, SAMPLE_RATE)
        self.frame = frame
        self._q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=200)
        self.dropped = 0
        block = int(round(frame * capture_rate / SAMPLE_RATE))
        self._stream = sd.InputStream(
            device=_device(device),
            samplerate=capture_rate,
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=self._callback,
        )

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
            pending = np.concatenate([pending, self.resampler.push(block)])
            while len(pending) >= self.frame:
                out, pending = pending[: self.frame], pending[self.frame :]
                yield out, t


class Speaker:
    """Blocking playback in small blocks so that it can be interrupted."""

    def __init__(self, device: str, playback_rate: int, block_ms: int = 50):
        import sounddevice as sd

        self.rate = playback_rate
        self.block = int(playback_rate * block_ms / 1000)
        self._stream = sd.OutputStream(
            device=_device(device),
            samplerate=playback_rate,
            channels=1,
            dtype="float32",
        )
        self._stream.start()
        self._sounding_until = 0.0

    def play(self, audio: np.ndarray, src_rate: int, cancelled) -> bool:
        """Play `audio`; returns False if `cancelled()` became true midway.

        Returns once the last block is queued, so the device still holds up to
        `latency` seconds of sound; call `drain` after the last chunk."""
        audio = resample(audio.astype(np.float32), src_rate, self.rate)
        for i in range(0, len(audio), self.block):
            if cancelled():
                # Drop what is queued so playback stops now.
                self._stream.abort()
                self._stream.start()
                self._sounding_until = 0.0
                return False
            self._stream.write(audio[i : i + self.block])
            self._sounding_until = time.monotonic() + self._stream.latency
        return True

    def drain(self) -> None:
        """Wait until the queued sound has been heard."""
        time.sleep(max(0.0, self._sounding_until - time.monotonic()))

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
