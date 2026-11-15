"""English TTS with Kokoro-82M (no ROS dependency: tts_node と推論ワーカーで共用)."""

from __future__ import annotations

import re
import threading

import numpy as np

from .model_registry import Registry

SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


class Turns:
    """Speak goals take turns in the order they arrive (threading.Lock wakes waiters in
    no particular order). A goal cancelled while waiting gives up its turn."""

    def __init__(self):
        self.cond = threading.Condition()
        self.next = 0
        self.serving = 0
        self.finished: set[int] = set()

    def take(self) -> int:
        with self.cond:
            self.next += 1
            return self.next - 1

    def wait(self, turn: int, cancelled) -> bool:
        """True when `turn` may play; False if cancelled first."""
        with self.cond:
            while self.serving != turn:
                if cancelled():
                    return False
                self.cond.wait(0.1)
            return True

    def finish(self, turn: int) -> None:
        with self.cond:
            self.finished.add(turn)
            while self.serving in self.finished:
                self.finished.remove(self.serving)
                self.serving += 1
            self.cond.notify_all()


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]


class KokoroTts:
    def __init__(self, registry: Registry, device: str):
        from kokoro import KModel, KPipeline

        entry = registry.entry("tts", "kokoro")
        spec = entry.spec
        with registry.loading(entry, device) as extra:
            local = entry.fetch(
                allow_patterns=[
                    "config.json",
                    "*.pth",
                    f'voices/{spec["lang_code"]}*.pt',
                ]
            )
            weights = next(local.glob("*.pth"))
            model = (
                KModel(
                    repo_id=entry.repo,
                    config=str(local / "config.json"),
                    model=str(weights),
                )
                .to(device)
                .eval()
            )
            self.pipeline = KPipeline(
                lang_code=spec["lang_code"], repo_id=entry.repo, model=model
            )
            self.voices = {
                p.stem: str(p) for p in sorted((local / "voices").glob("*.pt"))
            }
            self.default_voice = spec["voice"]
            self.sample_rate = int(spec["sample_rate"])
            self.tag = entry.tag
            list(self.synthesize("Ready.", self.default_voice))  # warm up
            extra.update(
                weights=weights.name,
                voices=list(self.voices),
                default_voice=self.default_voice,
            )

    def synthesize(self, text: str, voice: str):
        """Yield float32 audio chunks for `text`."""
        for result in self.pipeline(text, voice=self.voices[voice]):
            if result.audio is not None:
                yield result.audio.detach().cpu().numpy().astype(np.float32)
