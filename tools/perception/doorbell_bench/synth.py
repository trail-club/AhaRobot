"""Synthetic doorbells, to cover chime types that FSD50K lacks (no Japanese intercoms).

Every family draws its parameters from `random.Random(seed)`, so a clip is fixed by
(family, seed). Timbres: struck bar (free-free bar modes), bell (hum/prime/tierce/
quint/nominal partials), FM bell, and plain electronic tones.

  ding_dong    two descending notes, struck-bar or FM timbre (Western door chime)
  pinpon       Japanese intercom style: a higher then a lower electronic tone, 1-2 times
  westminster  four notes of the Westminster Quarters, bell or bar timbre
  melody       3-6 notes of a pentatonic tune, electronic tones
  buzzer       electric buzzer (square/saw at 90-250 Hz), 1-2 bursts
  bell_ring    electric bell: a striker at 12-25 Hz on a metal bell
  single_ding  one strike of a desk or shop bell
"""

from __future__ import annotations

import random

import numpy as np

from common import STORE_RATE

FAMILIES = (
    "ding_dong",
    "pinpon",
    "westminster",
    "melody",
    "buzzer",
    "bell_ring",
    "single_ding",
)
BAR = ((1.0, 1.0), (2.756, 0.5), (5.404, 0.25), (8.933, 0.12))
BELL = (
    (0.5, 0.4),
    (1.0, 1.0),
    (1.19, 0.6),
    (1.5, 0.35),
    (2.0, 0.5),
    (2.5, 0.2),
    (3.0, 0.15),
)


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * STORE_RATE)) / STORE_RATE


def _env(t: np.ndarray, attack: float, decay: float) -> np.ndarray:
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / decay)


def _partials(
    f0: float, seconds: float, decay: float, partials, rng: random.Random
) -> np.ndarray:
    t = _t(seconds)
    out = np.zeros_like(t)
    for ratio, amp in partials:
        f = f0 * ratio * (1 + rng.uniform(-0.004, 0.004))
        if f > STORE_RATE * 0.45:
            continue
        out += (
            amp
            * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            * _env(t, 0.002, decay / ratio**0.6)
        )
    return out


def _fm_bell(f0: float, seconds: float, decay: float, rng: random.Random) -> np.ndarray:
    t = _t(seconds)
    ratio = rng.choice((1.4, 2.0, 3.5))
    index = rng.uniform(1.5, 4.0) * np.exp(-t / (decay * 0.5))
    return np.sin(
        2 * np.pi * f0 * t + index * np.sin(2 * np.pi * f0 * ratio * t)
    ) * _env(t, 0.003, decay)


def _tone(
    f0: float, seconds: float, wave: str, attack: float, decay: float
) -> np.ndarray:
    t = _t(seconds)
    phase = 2 * np.pi * f0 * t
    if wave == "sine":
        x = np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
    elif wave == "triangle":
        x = 2 / np.pi * np.arcsin(np.sin(phase))
    else:  # soft square: odd harmonics up to 7
        x = sum(np.sin(k * phase) / k for k in (1, 3, 5, 7))
    return x * _env(t, attack, decay)


def _place(parts: list[tuple[float, np.ndarray]], tail: float = 0.3) -> np.ndarray:
    end = max(start + len(x) / STORE_RATE for start, x in parts) + tail
    out = np.zeros(int(end * STORE_RATE))
    for start, x in parts:
        i = int(start * STORE_RATE)
        out[i : i + len(x)] += x
    return out


def _note(
    f: float, seconds: float, timbre: str, decay: float, rng: random.Random
) -> np.ndarray:
    if timbre == "bar":
        return _partials(f, seconds, decay, BAR, rng)
    if timbre == "bell":
        return _partials(f, seconds, decay, BELL, rng)
    return _fm_bell(f, seconds, decay, rng)


def ding_dong(rng: random.Random) -> np.ndarray:
    f1 = rng.uniform(500, 1100)
    f2 = f1 / 2 ** (rng.choice((3, 4, 5, 7, 2)) / 12)
    timbre = rng.choice(("bar", "fm"))
    decay = rng.uniform(0.5, 1.8)
    gap = rng.uniform(0.35, 0.8)
    parts = [
        (0.0, _note(f1, 2.5, timbre, decay, rng)),
        (gap, _note(f2, 3.0, timbre, decay * 1.2, rng)),
    ]
    return _place(parts)


def pinpon(rng: random.Random) -> np.ndarray:
    f_hi = rng.uniform(600, 1300)
    f_lo = f_hi / 2 ** (rng.choice((3, 4, 5)) / 12)
    wave = rng.choice(("sine", "sine", "triangle", "square"))
    gap = rng.uniform(0.25, 0.55)
    d_hi, d_lo = rng.uniform(0.15, 0.4), rng.uniform(0.5, 1.2)
    one = [
        (0.0, _tone(f_hi, 1.2, wave, 0.005, d_hi)),
        (gap, _tone(f_lo, 2.5, wave, 0.005, d_lo)),
    ]
    parts = list(one)
    if rng.random() < 0.6:  # "ピンポーン、ピンポーン"
        again = gap + rng.uniform(0.9, 1.6)
        parts += [(again + s, x) for s, x in one]
    return _place(parts)


def westminster(rng: random.Random) -> np.ndarray:
    base = rng.uniform(330, 700)
    semis = rng.choice(
        ((4, 2, 0, -5), (0, 4, 2, -5), (0, 2, 4, 0), (4, 0, 2, -5), (-5, 2, 4, 0))
    )
    timbre = rng.choice(("bell", "bar"))
    step = rng.uniform(0.45, 0.8)
    decay = rng.uniform(0.8, 2.0)
    parts = [
        (i * step, _note(base * 2 ** (s / 12), 3.0, timbre, decay, rng))
        for i, s in enumerate(semis)
    ]
    return _place(parts)


def melody(rng: random.Random) -> np.ndarray:
    base = rng.uniform(400, 900)
    scale = (0, 2, 4, 7, 9, 12)
    wave = rng.choice(("sine", "triangle", "square"))
    step = rng.uniform(0.15, 0.4)
    n = rng.randint(3, 6)
    parts = [
        (
            i * step,
            _tone(base * 2 ** (rng.choice(scale) / 12), 1.0, wave, 0.005, step * 1.5),
        )
        for i in range(n)
    ]
    return _place(parts)


def buzzer(rng: random.Random) -> np.ndarray:
    f0 = rng.uniform(90, 250)
    out = []
    t0 = 0.0
    for _ in range(rng.choice((1, 2))):
        seconds = rng.uniform(0.5, 2.0)
        t = _t(seconds)
        saw = 2 * (t * f0 % 1) - 1
        square = np.sign(np.sin(2 * np.pi * f0 * t))
        x = saw if rng.random() < 0.5 else square
        x = x * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(20, 60) * t))  # rattle
        x *= np.clip(t / 0.01, 0, 1) * np.clip((seconds - t) / 0.02, 0, 1)
        out.append((t0, 0.5 * x))
        t0 += seconds + rng.uniform(0.3, 0.7)
    return _place(out, tail=0.2)


def bell_ring(rng: random.Random) -> np.ndarray:
    f0 = rng.uniform(700, 1800)
    rate = rng.uniform(12, 25)
    seconds = rng.uniform(1.0, 3.0)
    strike = _partials(f0, 0.3, 0.25, BELL, rng)
    parts = [(k / rate, strike) for k in range(int(seconds * rate))]
    return _place(parts, tail=0.4)


def single_ding(rng: random.Random) -> np.ndarray:
    f0 = rng.uniform(900, 3000)
    timbre = rng.choice(("bell", "fm"))
    return _place([(0.0, _note(f0, 2.5, timbre, rng.uniform(0.4, 1.5), rng))])


def make(family: str, seed: int) -> np.ndarray:
    rng = random.Random(f"{family}-{seed}")
    x = globals()[family](rng)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
    return x.astype(np.float32)
