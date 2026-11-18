"""detect_doorbell against fake detector events, and the simulated sounds (no ROS)."""

import os
import sys
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.doorbell import DetectDoorbell, Doorbell  # noqa: E402
from aha_tasks import sim_sounds  # noqa: E402


class FakeDoorbell(Doorbell):
    def __init__(self, publishers=1):
        self.changed = threading.Condition()
        self.events = []
        self.publishers = publishers

    def detectors(self):
        return self.publishers

    def ring(self, label, delay, score=0.9):
        def later():
            time.sleep(delay)
            self.on_event(SimpleNamespace(label=label, score=score))

        threading.Thread(target=later, daemon=True).start()


def context(params=None, cancelled=None):
    return SkillContext(
        subgoal="guest1.doorbell",
        params=params or {},
        memory={},
        cancelled=cancelled or threading.Event(),
        log=lambda message: None,
    )


def skill(bell):
    return DetectDoorbell(SimpleNamespace(doorbell=bell))


class DetectDoorbellTests(unittest.TestCase):
    def test_params(self):
        DetectDoorbell.validate({}, None)
        DetectDoorbell.validate({"timeout": 30}, None)
        for bad in ({"timeout": 0}, {"labels": ["knock"]}, {"wait": 3}):
            with self.subTest(bad), self.assertRaises(ValueError):
                DetectDoorbell.validate(bad, None)

    def test_succeeds_on_the_next_event(self):
        bell = FakeDoorbell()
        bell.ring("doorbell", 0.2)
        result = skill(bell).execute(context({"timeout": 5}))
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.outputs, {"doorbell.heard": "doorbell"})

    def test_ignores_events_from_before_the_start(self):
        bell = FakeDoorbell()
        bell.on_event(SimpleNamespace(label="doorbell", score=0.9))
        result = skill(bell).execute(context({"timeout": 0.3}))
        self.assertFalse(result.success)
        self.assertIn("within", result.message)

    def test_a_knock_counts_too(self):
        bell = FakeDoorbell()
        bell.ring("knock", 0.1)
        result = skill(bell).execute(context({"timeout": 5}))
        self.assertEqual(result.outputs, {"doorbell.heard": "knock"})

    def test_fails_without_a_detector(self):
        result = skill(FakeDoorbell(publishers=0)).execute(context())
        self.assertFalse(result.success)
        self.assertIn("aha_audio_doorbell", result.message)

    def test_fails_at_once_when_the_detector_stops(self):
        bell = FakeDoorbell()
        threading.Timer(0.2, lambda: setattr(bell, "publishers", 0)).start()
        started = time.monotonic()
        result = skill(bell).execute(context({"timeout": 5}))
        self.assertFalse(result.success)
        self.assertIn("stopped", result.message)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_cancel_stops_the_wait(self):
        cancelled = threading.Event()
        threading.Timer(0.2, cancelled.set).start()
        start = time.monotonic()
        result = skill(FakeDoorbell()).execute(context({"timeout": 30}, cancelled))
        self.assertFalse(result.success)
        self.assertLess(time.monotonic() - start, 2.0)


class SimSoundTests(unittest.TestCase):
    def test_sounds_have_the_set_peak(self):
        rng = np.random.default_rng(0)
        for sound in sim_sounds.SOUNDS:
            x, _ = sim_sounds.make(sound, rng)
            self.assertAlmostEqual(
                20 * np.log10(np.max(np.abs(x))), sim_sounds.SOUND_DBFS, places=3
            )
        with self.assertRaises(ValueError):
            sim_sounds.make("siren", rng)

    def test_every_sound_has_a_truth_label(self):
        self.assertEqual(set(sim_sounds.TRUTH_LABELS), set(sim_sounds.SOUNDS))
        self.assertEqual(set(sim_sounds.TRUTH_LABELS.values()), {"doorbell", "knock"})

    def test_every_chime_family_and_random_choice(self):
        rng = np.random.default_rng(3)
        for family in sim_sounds.CHIMES:
            x, kind = sim_sounds.make("chime", rng, family)
            self.assertEqual(kind, family)
            self.assertTrue(
                np.all(np.isfinite(x))
                and 0.5 * sim_sounds.RATE < len(x) < 8 * sim_sounds.RATE
            )
        kinds = {sim_sounds.make("chime", rng)[1] for _ in range(40)}
        self.assertGreater(len(kinds), 2)
        self.assertLessEqual(kinds, set(sim_sounds.CHIMES))

    def test_mixer_adds_a_sound_over_the_background(self):
        mixer = sim_sounds.Mixer(seed=1)
        quiet = mixer.read(sim_sounds.RATE)
        self.assertEqual(len(quiet), sim_sounds.RATE)
        self.assertAlmostEqual(
            20 * np.log10(np.sqrt(np.mean(quiet**2))), sim_sounds.NOISE_DBFS, delta=1.0
        )
        self.assertIn(mixer.add("chime"), sim_sounds.CHIMES)
        chunks = [mixer.read(1600) for _ in range(40)]  # 4 s in 0.1 s pieces
        loud = np.concatenate(chunks)
        self.assertGreater(np.max(np.abs(loud)), 10 * np.max(np.abs(quiet)))
        self.assertLess(np.max(np.abs(mixer.read(1600))), 3 * np.max(np.abs(quiet)))


if __name__ == "__main__":
    unittest.main()
