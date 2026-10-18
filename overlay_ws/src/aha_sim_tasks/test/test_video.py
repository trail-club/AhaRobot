"""Decode the recorded movie to verify views, sample timing, and finalization."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.api import Action  # noqa: E402
from aha_sim_tasks.video import VideoRecorder  # noqa: E402


def image(color, stamp):
    pixels = np.full((6, 8, 3), color, dtype=np.uint8)
    return SimpleNamespace(
        height=6,
        width=8,
        step=24,
        encoding="rgb8",
        data=pixels.tobytes(),
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=int(stamp), nanosec=round((stamp % 1) * 1e9))
        ),
    )


def frames(path):
    capture = cv2.VideoCapture(str(path))
    result = []
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == 10
        while True:
            ok, frame = capture.read()
            if not ok:
                return result
            result.append(frame)
    finally:
        capture.release()


def test_video_records_motion_between_decisions_and_retains_simulation_time(tmp_path):
    recorder = VideoRecorder(tmp_path, "place_apple", "codex")
    recorder.image("/camera/color/image_raw", image((255, 0, 0), 10))
    recorder.image("/camera/left_wrist/image_raw", image((0, 255, 0), 10))
    recorder.image("/camera/right_wrist/image_raw", image((0, 0, 255), 10))
    recorder.start(10)
    directory = tmp_path / "codex_steps"
    directory.mkdir()
    (directory / "decisions.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "situation": "Apple ahead.",
                "decision_note": "Approach with open fingers.",
            }
        )
        + "\n"
    )
    recorder.action(Action(linear_velocity=0.1))
    recorder.image("/camera/color/image_raw", image((255, 255, 255), 10.2))
    recorder.capture(10.2)
    count = recorder.frame_count
    recorder.capture(10.2)  # A paused model must not create repeated frozen frames.
    assert recorder.frame_count == count
    recorder.capture(10.5)
    summary = recorder.close(10.5, "timeout")
    assert summary["frames"] == 16  # Six samples, then one terminal playback second.
    assert summary["status"] == "timeout"
    decoded = frames(tmp_path / "episode.mp4")
    assert len(decoded) == summary["frames"]
    assert decoded[0].shape == (720, 1920, 3)
    # BGR pixels: original head red, then white at t=10.2, and distinct wrist views.
    np.testing.assert_allclose(decoded[1][250, 300], (0, 0, 255), atol=8)
    np.testing.assert_allclose(decoded[2][250, 300], (255, 255, 255), atol=8)
    np.testing.assert_allclose(decoded[2][250, 940], (0, 255, 0), atol=8)
    np.testing.assert_allclose(decoded[2][250, 1580], (255, 0, 0), atol=8)
    index = [
        json.loads(line)
        for line in (tmp_path / "video_frames.jsonl").read_text().splitlines()
    ]
    assert [row["sim_time"] for row in index[:6]] == [10 + i / 10 for i in range(6)]
    assert index[1]["camera_stamps"]["head"] == 10
    assert index[2]["camera_stamps"]["head"] == 10.2
    assert index[2]["decision"] == 1


def test_partial_video_is_playable_after_an_episode_error(tmp_path):
    recorder = VideoRecorder(tmp_path, "place_apple", "codex")
    for topic in (
        "/camera/color/image_raw",
        "/camera/left_wrist/image_raw",
        "/camera/right_wrist/image_raw",
    ):
        recorder.image(topic, image((100, 100, 100), 1))
    recorder.start(1)
    summary = recorder.close(1, "error")
    assert summary["status"] == "error"
    assert len(frames(tmp_path / "episode.mp4")) == 11
    recorder.close()  # Cleanup is idempotent.


def test_missing_encoder_fails_before_policy_runs(tmp_path, monkeypatch):
    released = []
    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        lambda *args: SimpleNamespace(
            isOpened=lambda: False,
            release=lambda: released.append(True),
        ),
    )
    with pytest.raises(RuntimeError, match="H.264 video encoder unavailable"):
        VideoRecorder(tmp_path, "place_apple", "codex")
    assert released == [True]
