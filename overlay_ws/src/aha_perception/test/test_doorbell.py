"""Doorbell post-processing and window (aha_perception.audio.doorbell, no torch)."""

import numpy as np
import pytest

from aha_perception.audio.doorbell import HOP, RATE, Rule, Window, target_columns


def events(rule, scores):
    out = []
    for k, s in enumerate(scores):
        if rule.push(s, (k + 1) * HOP) is not None:
            out.append(k)
    return out


def test_an_event_is_the_start_of_a_run_above_the_threshold():
    assert events(Rule(threshold=0.5), [0.1, 0.6, 0.7, 0.9, 0.2]) == [1]


def test_n_consecutive_hops_are_needed():
    rule = Rule(threshold=0.5, n=3)
    assert events(rule, [0.6, 0.6, 0.1, 0.6, 0.6, 0.6, 0.6]) == [5]


def test_moving_average_over_the_available_hops():
    rule = Rule(threshold=0.5, avg=2)
    assert events(rule, [0.9]) == [0]  # one hop so far: its own value
    rule = Rule(threshold=0.5, avg=2)
    assert events(rule, [0.0, 0.9, 0.9]) == [2]  # 0.45, then 0.9


def test_a_start_soon_after_the_previous_one_is_the_same_event():
    gap = int(5.0 / HOP)
    scores = [0.9, 0.1] * 3 + [0.1] * gap + [0.9]
    assert events(Rule(threshold=0.5), scores) == [0, len(scores) - 1]


def test_reset_forgets_the_run():
    rule = Rule(threshold=0.5, refractory=0.0)
    assert rule.push(0.9, 0.2) is not None
    rule.reset()
    assert rule.push(0.9, 0.4) is not None


def test_window_keeps_the_last_seconds():
    window = Window(1.0)
    window.push(np.ones(RATE // 2))
    assert window.buffer[: RATE // 2].sum() == 0
    window.push(np.full(2 * RATE, 2.0))
    assert np.all(window.buffer == 2.0) and len(window.buffer) == RATE
    window.clear()
    assert not window.buffer.any()


def test_target_columns_names_unknown_classes():
    labels = ["Speech", "Doorbell", "Ding-dong", "Knock"]
    assert target_columns(
        labels, {"chime": ["Doorbell", "Ding-dong"], "knock": ["Knock"]}
    ) == {
        "chime": [1, 2],
        "knock": [3],
    }
    with pytest.raises(ValueError, match="Dingdong"):
        target_columns(labels, {"chime": ["Dingdong"]})


def test_a_checkpoint_pinned_by_hash_is_tagged_with_it():
    from aha_perception.audio.model_registry import ModelEntry

    spec = {"checkpoint": "ATST-F_strong_1", "sha256": "fbf2577958e3"}
    assert ModelEntry("atst_f", "doorbell", spec).tag == "ATST-F_strong_1@fbf2577"
    hf = {"repo": "mispeech/ced-base", "revision": "db3e14a8db4c"}
    assert ModelEntry("ced_base", "doorbell", hf).tag == "mispeech/ced-base@db3e14a"


def test_the_ledger_has_complete_doorbell_entries():
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "config" / "audio" / "models.yaml"
    entries = yaml.safe_load(path.read_text())["doorbell"]
    assert {"beats_sed", "atst_f", "ced_base"} <= set(entries)
    for key, entry in entries.items():
        assert entry["engine"] in ("tagger", "frame"), key
        assert set(entry["classes"]) == set(entry["rules"]) == {"chime", "knock"}, key
        for rule in entry["rules"].values():
            Rule(**rule)
