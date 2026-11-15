"""Doorbell / knock detection over a 16 kHz stream (no ROS, no torch at import).

Every HOP s, the BEATs of PretrainedSED (AudioSet strong labels, the submodule
upstream/PretrainedSED) scores the last `window` s on the GPU. A target's score
is the max of its classes' probabilities over the frames between `frames[0]` and
`frames[1]` s before the window end. Post-processing is the one of the benchmark
(tools/perception/doorbell_bench/evaluate.py): a moving average over `avg` hops
must stay at or above the threshold for `n` hops; an event is the start of such a
run, unless the previous run started less than `refractory` s before.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RATE = 16000
HOP = 0.2  # s between scores
FRAME = 0.04  # s per output frame of the model
LABELS = {"chime": "doorbell", "knock": "knock"}  # target -> SoundEvent.label
MODEL = "beats_sed"  # key under doorbell: in models.yaml


@dataclass
class Rule:
    """Post-processing of one target's scores."""

    threshold: float
    avg: int = 1
    n: int = 1
    refractory: float = 5.0  # s
    _scores: deque = field(default_factory=deque, repr=False)
    _smoothed: deque = field(default_factory=deque, repr=False)
    _above: bool = False
    _last_start: float = -np.inf

    def push(self, score: float, t: float) -> float | None:
        """Add the score of the window ending at t; the smoothed score on an event."""
        self._scores.append(score)
        while len(self._scores) > self.avg:
            self._scores.popleft()
        smoothed = float(np.mean(self._scores))
        self._smoothed.append(smoothed)
        while len(self._smoothed) > self.n:
            self._smoothed.popleft()
        above = len(self._smoothed) == self.n and min(self._smoothed) >= self.threshold
        start = above and not self._above
        self._above = above
        if not start:
            return None
        quiet = t - self._last_start >= self.refractory - 1e-6
        self._last_start = t
        return smoothed if quiet else None

    def reset(self) -> None:
        """Forget the recent scores (e.g. while the robot itself speaks)."""
        self._scores.clear()
        self._smoothed.clear()
        self._above = False


class Window:
    """The last `seconds` of a 16 kHz stream."""

    def __init__(self, seconds: float):
        self.size = int(round(seconds * RATE))
        self.buffer = np.zeros(self.size, np.float32)

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, np.float32)[-self.size :]
        self.buffer = np.concatenate([self.buffer[len(samples) :], samples])

    def clear(self) -> None:
        self.buffer[:] = 0.0


def target_columns(
    labels: list[str], classes: dict[str, list[str]]
) -> dict[str, list[int]]:
    """Output indices of each target's classes; error if a class is unknown."""
    index = {name: i for i, name in enumerate(labels)}
    missing = [
        name for names in classes.values() for name in names if name not in index
    ]
    if missing:
        raise ValueError(f"classes not in the model's labels: {missing}")
    return {
        target: [index[name] for name in names] for target, names in classes.items()
    }


class BeatsSed:
    """PretrainedSED's BEATs with its AudioSet-strong checkpoint, on the GPU.

    `entry` is the models.yaml entry: `code` (the submodule, relative to
    $AHA_REPO_ROOT, default /app), `checkpoint`, `url`, `sha256`, `classes` and
    `frames`. The checkpoint is cached under $HF_HOME/aharobot-doorbell."""

    def __init__(self, entry: dict, device: str):
        import torch

        self.torch = torch
        self.device = device
        code = Path(os.environ.get("AHA_REPO_ROOT", "/app")) / entry["code"]
        if not (code / "models" / "prediction_wrapper.py").is_file():
            raise FileNotFoundError(
                f"{code}: run git submodule update --init {entry['code']}"
            )
        sys.path.insert(0, str(code))  # PretrainedSED's top-level `models`, `config`
        from data_util import audioset_classes
        from models import prediction_wrapper
        from models.beats.BEATs_wrapper import BEATsWrapper

        # load_checkpoint() reads <RESOURCES_FOLDER>/<checkpoint>.pt
        prediction_wrapper.RESOURCES_FOLDER = str(_checkpoint(entry).parent)
        self.model = prediction_wrapper.PredictionsWrapper(
            BEATsWrapper(), checkpoint=entry["checkpoint"]
        )
        self.model.eval().to(device)
        self.columns = target_columns(
            list(audioset_classes.as_strong_train_classes), entry["classes"]
        )
        far, near = entry["frames"]
        self.view = slice(-round(far / FRAME), -round(near / FRAME) or None)
        self.tag = f"{entry['checkpoint']}@{entry['sha256'][:7]}"

    def score(self, audio: np.ndarray) -> dict[str, float]:
        torch = self.torch
        with torch.inference_mode():
            mel = self.model.mel_forward(torch.from_numpy(audio)[None].to(self.device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                strong, _ = self.model(mel)
            frames = torch.sigmoid(strong.float())[0, :, self.view]
            probs = frames.amax(dim=1).cpu().numpy()
        return {
            target: float(probs[cols].max()) for target, cols in self.columns.items()
        }


def _checkpoint(entry: dict) -> Path:
    """The checkpoint file, downloaded once and checked against its sha256."""
    import torch

    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = home / "aharobot-doorbell" / f"{entry['checkpoint']}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = f"{path}.part"
        torch.hub.download_url_to_file(entry["url"], part, progress=False)
        os.replace(part, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != entry["sha256"]:
        raise RuntimeError(f"{path}: sha256 does not match models.yaml")
    return path
