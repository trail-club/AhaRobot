"""Simulated room audio for the doorbell detector, without ROS.

The simulation has no acoustics, so a microphone stream is synthesized: quiet
pink noise plus the doorbell or knocks that a guest makes at the door. The sounds
are generated, so no recording is needed; their levels are relative to full scale
at the robot.

Each chime is one of CHIMES with random pitch, timbre and timing (the families of
tools/perception/doorbell_bench/synth.py that are chimes; buzzers and electric
bells are left out because the detector misses them). Knocks are short impacts on
the damped modes of a random door panel.
"""

import numpy as np

RATE = 16000  # Hz, the detector's input rate
NOISE_DBFS = -55.0  # RMS of the background
SOUND_DBFS = -20.0  # peak-normalized level of a doorbell or knock
SOUNDS = ("chime", "knock")
BAR_MODES = ((1.0, 1.0), (2.756, 0.5), (5.404, 0.25), (8.933, 0.12))  # struck bar
BELL_MODES = (
    (0.5, 0.4),
    (1.0, 1.0),
    (1.19, 0.6),
    (1.5, 0.35),
    (2.0, 0.5),
    (2.5, 0.2),
    (3.0, 0.15),
)
CHIMES = ("ding_dong", "pinpon", "westminster", "melody", "single_ding")


def _db(db):
    return 10.0 ** (db / 20.0)


def _t(seconds):
    return np.arange(int(seconds * RATE)) / RATE


def _env(t, attack, decay):
    return np.clip(t / attack, 0.0, 1.0) * np.exp(-t / decay)


def _modes(f0, seconds, decay, modes, rng):
    t = _t(seconds)
    out = np.zeros_like(t)
    for ratio, amp in modes:
        f = f0 * ratio * (1 + rng.uniform(-0.004, 0.004))
        if f < RATE * 0.45:
            out += (
                amp
                * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                * _env(t, 0.002, decay / ratio**0.6)
            )
    return out


def _fm_bell(f0, seconds, decay, rng):
    t = _t(seconds)
    index = rng.uniform(1.5, 4.0) * np.exp(-t / (decay * 0.5))
    modulator = np.sin(2 * np.pi * f0 * rng.choice((1.4, 2.0, 3.5)) * t)
    return np.sin(2 * np.pi * f0 * t + index * modulator) * _env(t, 0.003, decay)


def _note(f0, seconds, decay, timbre, rng):
    if timbre == "bar":
        return _modes(f0, seconds, decay, BAR_MODES, rng)
    if timbre == "bell":
        return _modes(f0, seconds, decay, BELL_MODES, rng)
    return _fm_bell(f0, seconds, decay, rng)


def _tone(f0, seconds, wave, decay):
    t = _t(seconds)
    phase = 2 * np.pi * f0 * t
    if wave == "sine":
        x = np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
    elif wave == "triangle":
        x = 2 / np.pi * np.arcsin(np.sin(phase))
    else:  # soft square
        x = sum(np.sin(k * phase) / k for k in (1, 3, 5, 7))
    return x * _env(t, 0.005, decay)


def _place(parts, tail=0.3):
    end = max(start + len(x) / RATE for start, x in parts) + tail
    out = np.zeros(int(end * RATE))
    for start, x in parts:
        i = int(start * RATE)
        out[i : i + len(x)] += x
    return out


def ding_dong(rng):
    """Two descending struck-bar or FM-bell notes (Western door chime)."""
    f1 = rng.uniform(500.0, 1100.0)
    f2 = f1 / 2 ** (rng.choice((2, 3, 4, 5, 7)) / 12)
    timbre, decay, gap = (
        rng.choice(("bar", "fm")),
        rng.uniform(0.5, 1.8),
        rng.uniform(0.35, 0.8),
    )
    return _place(
        [
            (0.0, _note(f1, 2.5, decay, timbre, rng)),
            (gap, _note(f2, 3.0, decay * 1.2, timbre, rng)),
        ]
    )


def pinpon(rng):
    """Japanese intercom: a higher, then a lower electronic tone, once or twice."""
    f_hi = rng.uniform(600.0, 1300.0)
    f_lo = f_hi / 2 ** (rng.choice((3, 4, 5)) / 12)
    wave, gap = (
        rng.choice(("sine", "sine", "triangle", "square")),
        rng.uniform(0.25, 0.55),
    )
    one = [
        (0.0, _tone(f_hi, 1.2, wave, rng.uniform(0.15, 0.4))),
        (gap, _tone(f_lo, 2.5, wave, rng.uniform(0.5, 1.2))),
    ]
    parts = list(one)
    if rng.random() < 0.6:
        again = gap + rng.uniform(0.9, 1.6)
        parts += [(again + start, x) for start, x in one]
    return _place(parts)


def westminster(rng):
    """Four notes of the Westminster Quarters, bell or bar timbre."""
    base = rng.uniform(330.0, 700.0)
    semis = [(4, 2, 0, -5), (0, 4, 2, -5), (0, 2, 4, 0), (4, 0, 2, -5), (-5, 2, 4, 0)][
        rng.integers(5)
    ]
    timbre, step, decay = (
        rng.choice(("bell", "bar")),
        rng.uniform(0.45, 0.8),
        rng.uniform(0.8, 2.0),
    )
    return _place(
        [
            (k * step, _note(base * 2 ** (s / 12), 3.0, decay, timbre, rng))
            for k, s in enumerate(semis)
        ]
    )


def melody(rng):
    """Three to six notes of a pentatonic tune in electronic tones."""
    base, wave, step = (
        rng.uniform(400.0, 900.0),
        rng.choice(("sine", "triangle", "square")),
        rng.uniform(0.15, 0.4),
    )
    notes = rng.choice((0, 2, 4, 7, 9, 12), size=rng.integers(3, 7))
    return _place(
        [
            (k * step, _tone(base * 2 ** (s / 12), 1.0, wave, step * 1.5))
            for k, s in enumerate(notes)
        ]
    )


def single_ding(rng):
    """One strike of a desk or shop bell."""
    f0 = rng.uniform(900.0, 3000.0)
    return _place(
        [(0.0, _note(f0, 2.5, rng.uniform(0.4, 1.5), rng.choice(("bell", "fm")), rng))]
    )


def _panel(rng, modes=24, fmin=80.0, fmax=4000.0, decay=0.03):
    """Impulse response of a wooden door: damped modes, the high ones shorter."""
    t = _t(0.25)
    out = np.zeros_like(t)
    for f in np.exp(rng.uniform(np.log(fmin), np.log(fmax), modes)):
        phase = rng.uniform(0.0, 2 * np.pi)
        out += (
            (200.0 / f) ** 0.5
            * np.sin(2 * np.pi * f * t + phase)
            * np.exp(-t / (decay * (200.0 / f) ** 0.4))
        )
    return out


def knock(rng):
    """Two groups of three knocks: 1.5 ms knuckle impacts on one door panel."""
    door = _panel(rng)
    impact = np.sin(np.pi * _t(0.0015) / 0.0015)
    out = np.zeros(int(2.6 * RATE))
    for group in range(2):
        for k in range(3):
            i = int((0.05 + 1.1 * group + 0.18 * k + rng.uniform(-0.015, 0.015)) * RATE)
            hit = np.convolve(impact * rng.uniform(0.8, 1.0), door)
            out[i : i + len(hit)] += hit[: len(out) - i]
    return out


def make(sound, rng, family=None):
    """(samples, kind): a knock, or a chime of `family` (random if None)."""
    if sound not in SOUNDS:
        raise ValueError(f"unknown sound {sound!r}; one of {SOUNDS}")
    if sound == "knock":
        kind, x = "knock", knock(rng)
    else:
        kind = family or str(rng.choice(CHIMES))
        x = globals()[kind](rng)
    return (x / np.max(np.abs(x)) * _db(SOUND_DBFS)).astype(np.float32), kind


class Mixer:
    """Endless background with sounds added at their start; read in chunks."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)  # None (the node): different every run
        self.pending = []  # [samples, offset into the next chunk]
        self.pink = np.zeros(3)

    def add(self, sound):
        """Start a sound now; returns what it is (the chime family or "knock")."""
        samples, kind = make(sound, self.rng)
        self.pending.append(samples)
        return kind

    def _noise(self, n):
        # Pink-ish noise: white noise through a fixed 3-pole low-pass sum.
        white = self.rng.normal(0.0, 1.0, n)
        out = np.empty(n)
        b = self.pink
        for i, w in enumerate(white):
            b[0] = 0.99765 * b[0] + w * 0.0990460
            b[1] = 0.96300 * b[1] + w * 0.2965164
            b[2] = 0.57000 * b[2] + w * 1.0526913
            out[i] = b[0] + b[1] + b[2] + w * 0.1848
        return out / np.sqrt(np.mean(out**2)) * _db(NOISE_DBFS)

    def read(self, n):
        out = self._noise(n)
        rest = []
        for sound in self.pending:
            take = sound[:n]
            out[: len(take)] += take
            if len(sound) > n:
                rest.append(sound[n:])
        self.pending = rest
        return out.astype(np.float32)
