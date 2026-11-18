"""Paths, constants and audio helpers shared by the doorbell benchmark scripts."""

from __future__ import annotations

import json
import os
from math import gcd
from pathlib import Path

import numpy as np

STORE_RATE = 32000  # every clip and stream is stored at this rate (float16 .npy)
HOP = 0.2  # s between the ends of consecutive analysis windows
SPLITS = ("dev", "eval")

TARGETS = ("chime", "knock")
# AudioSet display names whose max is a target's score, per class set. Names missing
# from a model's label set are skipped. The sets follow the ontology, not the data:
#   strict  the target classes themselves
#   door    plus other door signals (Doorbell's siblings under Alarm, and Tap)
#   tonal   plus bells, struck tonal instruments and electronic tones (chime), and
#           other short impacts on wood (knock)
CLASS_SETS = {
    "strict": {"chime": ("Doorbell", "Ding-dong"), "knock": ("Knock",)},
    "door": {
        "chime": ("Doorbell", "Ding-dong", "Ding", "Buzzer", "Telephone bell ringing"),
        "knock": ("Knock", "Tap"),
    },
    "tonal": {
        "chime": (
            "Doorbell",
            "Ding-dong",
            "Ding",
            "Buzzer",
            "Telephone bell ringing",
            "Bell",
            "Chime",
            "Tubular bells",
            "Glockenspiel",
            "Marimba, xylophone",
            "Tuning fork",
            "Jingle, tinkle",
            "Ping",
            "Beep, bleep",
            "Sine wave",
            "Ringtone",
        ),
        "knock": ("Knock", "Tap", "Wood block", "Thump, thud"),
    },
}


def data_root() -> Path:
    return Path(os.environ.get("DOORBELL_DATA", "/data"))


def results_root() -> Path:
    return Path(os.environ.get("DOORBELL_RESULTS", "/data/results"))


def save_audio(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak
    np.save(path, x.astype(np.float16))


def load_audio(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def active_frames(x: np.ndarray, rate: int, floor_db: float = 30.0) -> np.ndarray:
    """Boolean mask of 10 ms frames within `floor_db` of the loudest frame."""
    n = rate // 100
    frames = len(x) // n
    if frames == 0:
        return np.zeros(0, dtype=bool)
    power = np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12
    db = 10 * np.log10(power)
    return db >= db.max() - floor_db


def onset_seconds(x: np.ndarray, rate: int) -> float:
    """First 10 ms frame within 25 dB of the loudest one (sound onset estimate)."""
    mask = active_frames(x, rate, floor_db=25.0)
    hits = np.flatnonzero(mask)
    return float(hits[0] / 100) if len(hits) else 0.0


def active_power(x: np.ndarray, rate: int) -> float:
    """Mean power over the active frames, the reference for SNR."""
    n = rate // 100
    mask = active_frames(x, rate)
    if not mask.any():
        return float(np.mean(x**2) + 1e-12)
    frames = x[: len(mask) * n].reshape(len(mask), n)
    return float(np.mean(frames[mask] ** 2) + 1e-12)
