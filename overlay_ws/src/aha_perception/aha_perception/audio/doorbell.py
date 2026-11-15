"""Doorbell / knock detection over a 16 kHz stream (no ROS, no torch at import).

Every HOP s the last `window` s of audio go through a pretrained AudioSet model
(the `engine` of the models.yaml entry, see ENGINES); a target's score is the max
of its classes' probabilities. Post-processing is the one of the
benchmark (tools/perception/doorbell_bench/evaluate.py): a moving average over
`avg` hops must stay at or above the threshold for `n` hops; an event is the start
of such a run, unless the previous run started less than `refractory` s before.
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
LABELS = {"chime": "doorbell", "knock": "knock"}  # target -> SoundEvent.label


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


class TaggerEngine:
    """An AudioSet tagger from Hugging Face (transformers, `trust_remote_code`).

    The registry entry gives `repo`, `revision` and `classes` ({target: [AudioSet
    display names]}); the model must take raw 16 kHz audio through its feature
    extractor and return probabilities (CED does) or logits (`sigmoid: true`)."""

    def __init__(self, entry: dict, device: str):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.device = device
        repo, rev = entry["repo"], entry["revision"]
        self.extractor = AutoFeatureExtractor.from_pretrained(
            repo, revision=rev, trust_remote_code=True
        )
        self.model = (
            AutoModelForAudioClassification.from_pretrained(
                repo, revision=rev, trust_remote_code=True
            )
            .eval()
            .to(device)
        )
        names = self.model.config.id2label
        self.labels = [names[i] for i in range(len(names))]
        self.columns = target_columns(self.labels, entry["classes"])
        self.sigmoid = bool(entry.get("sigmoid", False))
        self.tag = f"{repo}@{rev[:7]}"

    def score(self, audio: np.ndarray) -> dict[str, float]:
        torch = self.torch
        with torch.inference_mode():
            inputs = self.extractor(
                torch.from_numpy(audio)[None], sampling_rate=RATE, return_tensors="pt"
            )
            out = (
                self.model(**{k: v.to(self.device) for k, v in inputs.items()})
                .logits[0]
                .float()
            )
            if self.sigmoid:
                out = torch.sigmoid(out)
            out = out.cpu().numpy()
        return {target: float(out[cols].max()) for target, cols in self.columns.items()}


class FrameEngine:
    """A frame-level AudioSet-strong model of PretrainedSED (`code`, a submodule of
    this repository): 10 s in, 40 ms frames out. A target's score is the max of its
    classes over the frames between `frames[0]` and `frames[1]` s before the end.

    The checkpoint comes from `url` and must match `sha256`."""

    FRAME = 0.04  # s

    def __init__(self, entry: dict, device: str):
        import torch

        self.torch = torch
        self.device = device
        code = Path(os.environ.get("AHA_REPO_ROOT", "/app")) / entry["code"]
        if not (code / "models" / "prediction_wrapper.py").is_file():
            raise FileNotFoundError(
                f"{code}: run git submodule update --init {entry['code']}"
            )
        sys.path.insert(
            0, str(code)
        )  # PretrainedSED imports its top-level `models`, `config`
        from data_util import audioset_classes
        from models import prediction_wrapper

        cache = (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "aharobot-doorbell"
        )
        cache.mkdir(parents=True, exist_ok=True)
        name = entry["checkpoint"]
        path = cache / f"{name}.pt"
        if not path.exists():
            torch.hub.download_url_to_file(
                entry["url"], str(path) + ".part", progress=False
            )
            os.replace(str(path) + ".part", path)
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"{path}: sha256 does not match models.yaml")
        prediction_wrapper.RESOURCES_FOLDER = str(
            cache
        )  # load_checkpoint finds the file there
        self.model = prediction_wrapper.PredictionsWrapper(
            _frame_backbone(entry["model"]), checkpoint=name
        )
        self.model.eval().to(device)
        self.labels = list(audioset_classes.as_strong_train_classes)
        self.columns = target_columns(self.labels, entry["classes"])
        far, near = entry["frames"]
        self.view = slice(
            -int(round(far / self.FRAME)), -int(round(near / self.FRAME)) or None
        )
        self.tag = f"{name}@{entry['sha256'][:7]}"

    def score(self, audio: np.ndarray) -> dict[str, float]:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(audio)[None].to(self.device)
            mel = self.model.mel_forward(x)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
            ):
                strong, _ = self.model(mel)
            probs = (
                torch.sigmoid(strong.float())[0, :, self.view].amax(dim=1).cpu().numpy()
            )
        return {
            target: float(probs[cols].max()) for target, cols in self.columns.items()
        }


def _frame_backbone(name: str):
    if name == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper

        return ATSTWrapper()
    if name == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper

        return BEATsWrapper()
    raise ValueError(f"unsupported PretrainedSED model {name!r} (ATST-F, BEATs)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


ENGINES = {"tagger": TaggerEngine, "frame": FrameEngine}
