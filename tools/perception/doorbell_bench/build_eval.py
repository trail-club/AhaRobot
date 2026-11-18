"""Render the evaluation streams from `clips.jsonl` and the synthetic doorbells.

Positive streams hold one event: PRE seconds of background, the event (its onset
lands at PRE), then TAIL seconds. Each event is rendered in every condition:

  clean   the event over -70 dBFS white noise
  snr10   ambient background (other FSD50K clips of the same split) at +10 dB
  snr0    the same at 0 dB
  room    shoebox room (pyroomacoustics, RT60 0.3-0.8 s): the event 3-8 m away,
          the background 1-3 m away, +5 dB at the microphone

SNR is the event's active power (frames within 30 dB of its peak) over the
background's mean power. Negative streams concatenate the ambient, hard_chime and
hard_knock clips of one split and group into streams of at most NEG_SECONDS, each
clip at a random level. Output: `streams/<split>/<id>.npy` and `streams.jsonl`.
"""

from __future__ import annotations

import argparse
import random
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import synth
from common import (
    STORE_RATE,
    active_power,
    data_root,
    load_audio,
    onset_seconds,
    read_jsonl,
    save_audio,
    write_jsonl,
)

PRE = 6.0  # s of background before the onset
TAIL = 1.5  # s after the (cropped) event
EVENT_MAX = 6.0  # s of the event kept after its onset
NEG_SECONDS = 120.0
CONDITIONS = ("clean", "snr10", "snr0", "room")
EVENT_DBFS = -25.0  # active level of the event before mixing
SYNTH_PER_FAMILY = 24  # per split


def _db_to_amp(db: float) -> float:
    return 10 ** (db / 20)


def _scale_active(x: np.ndarray, dbfs: float) -> np.ndarray:
    return x * _db_to_amp(dbfs) / np.sqrt(active_power(x, STORE_RATE))


class Background:
    """Random ambient audio of one split, pieced together from whole clips."""

    def __init__(self, root, clips: list[dict]):
        self.root = root
        self.clips = clips

    def take(self, need: int, rng: random.Random) -> tuple[np.ndarray, list[str]]:
        """`need` samples of background and the clips they came from."""
        parts, used = [], []
        while sum(len(p) for p in parts) < need:
            clip = rng.choice(self.clips)
            x = load_audio(self.root / "clips" / clip["split"] / f"{clip['fname']}.npy")
            if len(x) < STORE_RATE // 2:
                continue
            parts.append(x / np.sqrt(np.mean(x**2) + 1e-12))
            used.append(clip["fname"])
        return np.concatenate(parts)[:need], used


def _room(rng: random.Random):
    import pyroomacoustics as pra

    while True:
        dim = [rng.uniform(6, 10), rng.uniform(4, 7), rng.uniform(2.5, 3.0)]
        rt60 = rng.uniform(0.3, 0.8)
        door = [0.4, rng.uniform(0.5, dim[1] - 0.5), 1.5]
        mic = [rng.uniform(1.0, dim[0] - 0.5), rng.uniform(0.5, dim[1] - 0.5), 1.2]
        dist = float(np.linalg.norm(np.subtract(door, mic)))
        if not 3.0 <= dist <= 8.0:
            continue
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(1.0, 3.0)
        talker = [mic[0] + r * np.cos(ang), mic[1] + r * np.sin(ang), 1.5]
        if not (0.3 < talker[0] < dim[0] - 0.3 and 0.3 < talker[1] < dim[1] - 0.3):
            continue
        absorption, max_order = pra.inverse_sabine(rt60, dim)
        room = pra.ShoeBox(
            dim,
            fs=STORE_RATE,
            materials=pra.Material(absorption),
            max_order=min(max_order, 40),
        )
        room.add_source(door)
        room.add_source(talker)
        room.add_microphone(mic)
        room.compute_rir()
        rirs = [np.asarray(room.rir[0][k], dtype=np.float32) for k in (0, 1)]
        return rirs, {
            "rt60": round(rt60, 2),
            "distance": round(dist, 2),
            "talker": round(r, 2),
        }


def _conv(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    return fftconvolve(x, rir)[: len(x) + len(rir)].astype(np.float32)


_AMBIENT: dict[str, list[dict]] = {}


def _init(ambient):
    _AMBIENT.update(ambient)


def render_positive(job):
    root, event, condition, seed = job
    rng = random.Random(seed)
    x = event["audio"]
    x = _scale_active(x, EVENT_DBFS + rng.uniform(-5, 5))
    n = int((PRE + len(x) / STORE_RATE + TAIL) * STORE_RATE)
    onset = PRE
    info = {}
    if condition == "room":
        rirs, info = _room(rng)
        direct = int(np.argmax(np.abs(rirs[0])))
        x = _conv(x, rirs[0])[direct:]
        x = x[: n - int(PRE * STORE_RATE)]
    out = np.zeros(n, dtype=np.float32)
    i = int(PRE * STORE_RATE)
    out[i : i + len(x)] += x[: n - i]
    used = []
    if condition == "clean":
        out += (
            np.random.default_rng(seed).normal(0, _db_to_amp(-70), n).astype(np.float32)
        )
    else:
        bg, used = Background(root, _AMBIENT[event["split"]]).take(n, rng)
        if condition == "room":
            bg = _conv(bg, rirs[1])[:n]
        snr = {"snr10": 10.0, "snr0": 0.0, "room": 5.0}[condition]
        p_event = active_power(x, STORE_RATE)
        bg = bg * np.sqrt(p_event / (np.mean(bg**2) + 1e-12) / 10 ** (snr / 10))
        out += bg[:n]
        info["snr"] = snr
    sid = f"{event['id']}-{condition}"
    save_audio(root / "streams" / event["split"] / f"{sid}.npy", out)
    return {
        "id": sid,
        "split": event["split"],
        "kind": "pos",
        "target": event["target"],
        "source": event["source"],
        "group": event["group"],
        "condition": condition,
        "onset": round(onset, 3),
        "event_duration": round(len(event["audio"]) / STORE_RATE, 3),
        "duration": round(n / STORE_RATE, 3),
        "clips": [event["fname"]] + used[:20],
        **info,
    }


def render_negative(job):
    root, split, group, sub, clips, index, seed = job
    rng = random.Random(seed)
    parts, used = [], []
    for clip in clips:
        x = load_audio(root / "clips" / split / f"{clip['fname']}.npy")
        if len(x) < STORE_RATE // 10:
            continue
        parts.append(_scale_active(x, rng.uniform(-35, -18)))
        used.append(clip["fname"])
    out = np.concatenate(parts)
    out += (
        np.random.default_rng(seed)
        .normal(0, _db_to_amp(-70), len(out))
        .astype(np.float32)
    )
    sid = f"neg-{split}-{group}-{sub}-{index:03d}"
    save_audio(root / "streams" / split / f"{sid}.npy", out)
    return {
        "id": sid,
        "split": split,
        "kind": "neg",
        "target": None,
        "source": "fsd50k",
        "group": group,
        "subgroup": sub,
        "condition": "neg",
        "duration": round(len(out) / STORE_RATE, 3),
        "clips": used,
    }


def events(root, clips: list[dict]):
    for c in clips:
        if c["group"] not in ("doorbell", "knock"):
            continue
        x = load_audio(root / "clips" / c["split"] / f"{c['fname']}.npy")
        start = max(0, int((c["onset"] - 0.05) * STORE_RATE))
        x = x[start : start + int(EVENT_MAX * STORE_RATE)]
        yield {
            "id": f"{c['group']}-{c['fname']}",
            "split": c["split"],
            "target": "chime" if c["group"] == "doorbell" else "knock",
            "source": "fsd50k",
            "group": c["subgroup"] if c["group"] == "doorbell" else "knock",
            "fname": c["fname"],
            "audio": x,
        }
    for split_index, split in enumerate(("dev", "eval")):
        for family in synth.FAMILIES:
            for k in range(SYNTH_PER_FAMILY):
                seed = split_index * 1000 + k
                x = synth.make(family, seed)
                start = max(0, int((onset_seconds(x, STORE_RATE) - 0.05) * STORE_RATE))
                yield {
                    "id": f"synth-{family}-{seed}",
                    "split": split,
                    "target": "chime",
                    "source": "synth",
                    "group": family,
                    "fname": f"{family}-{seed}",
                    "audio": x[start : start + int(EVENT_MAX * STORE_RATE)],
                }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = data_root()
    clips = read_jsonl(root / "clips.jsonl")
    ambient = defaultdict(list)
    for c in clips:
        if c["group"] == "ambient":
            ambient[c["split"]].append(c)

    pos_jobs = []
    for event in events(root, clips):
        for condition in CONDITIONS:
            seed = zlib.crc32(f"{args.seed}-{event['id']}-{condition}".encode())
            pos_jobs.append((root, event, condition, seed))

    neg_jobs = []
    pools = defaultdict(list)
    for c in clips:
        if c["group"] in ("ambient", "hard_chime", "hard_knock"):
            key = (
                c["split"],
                c["group"],
                c["subgroup"] if c["group"] == "ambient" else "all",
            )
            pools[key].append(c)
    for (split, group, sub), pool in sorted(pools.items()):
        rng = random.Random(f"{args.seed}-{split}-{group}-{sub}")
        rng.shuffle(pool)
        chunk, seconds, index = [], 0.0, 0
        for c in pool + [None]:
            if c is None or seconds + c["duration"] > NEG_SECONDS:
                if chunk:
                    neg_jobs.append(
                        (root, split, group, sub, chunk, index, rng.randrange(2**31))
                    )
                    index += 1
                chunk, seconds = [], 0.0
            if c is not None:
                chunk.append(c)
                seconds += c["duration"]

    with ProcessPoolExecutor(
        args.jobs, initializer=_init, initargs=(dict(ambient),)
    ) as ex:
        rows = list(ex.map(render_positive, pos_jobs, chunksize=8))
        rows += list(ex.map(render_negative, neg_jobs))
    write_jsonl(root / "streams.jsonl", rows)

    hours = Counter()
    count = Counter()
    for r in rows:
        key = (r["split"], r["kind"], r["target"] or r["group"])
        hours[key] += r["duration"] / 3600
        count[key] += 1
    for key in sorted(count, key=str):
        print(*key, count[key], f"{hours[key]:.2f} h")


if __name__ == "__main__":
    main()
