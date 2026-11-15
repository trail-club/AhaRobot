#!/usr/bin/env python3
"""audio worker protocol without CUDA or models.

ASR answers with the length of the audio in tenths of a second, so tests can tell
that the audio arrived whole; TTS returns 0.1 s of audio per character. Text
"fail" and audio of exactly 0.5 s fail one request; audio of 0.7 s stops the worker.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_perception.audio import worker  # noqa: E402
from aha_perception.audio.model_registry import Registry  # noqa: E402


class FakeEngine:
    supports_hotwords = True

    def transcribe(self, audio, language, hotwords):
        if len(audio) == 8000:
            raise RuntimeError("fake decode failure")
        if len(audio) == 11200:
            os._exit(3)
        return f"{language} {round(len(audio) / 1600)} {' '.join(hotwords)}".strip()


class FakeTts:
    voices = {"af_heart": "af_heart.pt"}
    default_voice = "af_heart"
    sample_rate = 24000
    tag = "fake/tts@0000000"

    def __init__(self, registry, device):
        registry.loaded["kokoro"] = {
            "tag": self.tag,
            "device": device,
            "load_seconds": 0.1,
        }

    def synthesize(self, text, voice):
        if text == "fail":
            raise RuntimeError("fake synthesis failure")
        yield np.full(2400 * len(text), 0.25, dtype=np.float32)


def load_asr(registry, key, device, logger):
    registry.loaded[key] = {"tag": key, "device": device, "load_seconds": 0.1}
    return FakeEngine()


def load_inference():
    registry = Registry(path="fake", raw={})
    return worker.Inference(
        registry, "cuda:0", ["asr", "tts"], load_asr=load_asr, load_tts=FakeTts
    )


if __name__ == "__main__":
    raise SystemExit(worker.serve(sys.stdin, worker.protocol_stream(), load_inference))
