"""Exercise multimodal CLI inputs, decisions, validation, and episode artifacts."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.api import CameraObservation, GroundTruth, Observation  # noqa: E402
from aha_sim_tasks.codex_policy import (  # noqa: E402
    CodexPolicy,
    image_array,
    parse_decision,
)


def camera():
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0), frame_id="optical"
    )
    # Two BGR pixels followed by row padding, then another padded row.
    image = SimpleNamespace(
        encoding="bgr8",
        height=2,
        width=2,
        step=8,
        is_bigendian=False,
        data=bytes([0, 0, 255, 0, 255, 0, 99, 99] * 2),
        header=header,
    )
    depth = SimpleNamespace(
        encoding="32FC1",
        height=2,
        width=2,
        step=8,
        is_bigendian=True,
        data=np.array([[1, np.nan], [np.inf, -1]], dtype=">f4").tobytes(),
        header=header,
    )
    info = SimpleNamespace(
        width=2,
        height=2,
        distortion_model="plumb_bob",
        k=[1] * 9,
        d=[],
        r=[1] * 9,
        p=[1] * 12,
    )
    transform = SimpleNamespace(
        header=SimpleNamespace(frame_id="base_link"),
        child_frame_id="optical",
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=1, y=2, z=3),
            rotation=SimpleNamespace(x=0, y=0, z=0, w=1),
        ),
    )
    return CameraObservation(image, info, transform, depth)


def observation():
    return Observation(
        1,
        "place_apple",
        "Place the apple on the green table.",
        {"joint_r1": 0.1},
        {"joint_r1": 0},
        (0, 0, 0),
        cameras={name: camera() for name in ("head", "left_wrist", "right_wrist")},
    )


def decision():
    return {
        "action": {
            "linear_velocity": 0.05,
            "angular_velocity": 0,
            "joint_positions": [{"name": "joint_r1", "position": 0.18}],
        },
        "situation": "The apple is visible ahead.",
        "decision_note": "Move closer while raising the gripper.",
    }


def test_each_act_runs_cli_with_images_and_records_notes(tmp_path, monkeypatch):
    executable = tmp_path / "codex"
    executable.write_text(
        "#!/usr/bin/python3\nimport json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "assert args[args.index('--model')+1] == 'gpt-6.1-sol'\n"
        "assert 'model_reasoning_effort=\"medium\"' in args\n"
        "assert args.count('--image') == 3\n"
        "assert '--output-schema' in args and '--ephemeral' in args\n"
        "prompt = sys.stdin.read()\n"
        "assert 'Current observation:' in prompt\n"
        "assert 'finger_contacts' not in prompt and 'object_position' not in prompt\n"
        f"pathlib.Path(args[args.index('--output-last-message')+1]).write_text({json.dumps(json.dumps(decision()))})\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "policy_ros_logs"))
    monkeypatch.setenv("AHA_POLICY_STEP_DURATION", "2")
    policy = CodexPolicy()
    policy.reset("place_apple", observation().instruction)
    first = policy.act(observation())
    assert first.joint_positions == {"joint_r1": 0.18}
    assert type(first.angular_velocity) is float
    assert policy.act(replace(observation(), sim_time=3)) == first
    directory = tmp_path / "codex_steps"
    records = [
        json.loads(line)
        for line in (directory / "decisions.jsonl").read_text().splitlines()
    ]
    assert [item["sim_time"] for item in records] == [1, 3]
    assert all(item["decision_note"] == decision()["decision_note"] for item in records)
    assert decision()["situation"] in (directory / "0002/prompt.txt").read_text()
    data = json.loads((directory / "0001/observation.json").read_text())
    assert "ground_truth" not in data
    assert data["cameras"]["head"]["depth_metres"]["samples"] == [
        [1, None],
        [None, None],
    ]
    assert np.asarray(Image.open(directory / "0001/head.png")).tolist()[0] == [
        [255, 0, 0],
        [0, 255, 0],
    ]
    assert np.load(directory / "0001/head_depth.npy")[0, 0] == 1


@pytest.mark.parametrize(
    "kind", ["unknown_joint", "out_of_range", "duplicate", "nan", "no_notes"]
)
def test_invalid_decisions_fail_without_scripted_fallback(kind):
    value = decision()
    joints = value["action"]["joint_positions"]
    if kind == "unknown_joint":
        joints[0]["name"] = "unknown"
    elif kind == "out_of_range":
        joints[0]["position"] = 2
    elif kind == "duplicate":
        joints.append(dict(joints[0]))
    elif kind == "nan":
        value["action"]["linear_velocity"] = float("nan")
    else:
        value["decision_note"] = ""
    with pytest.raises(ValueError):
        parse_decision(value)


def test_image_stride_validation_and_depth_byte_order():
    frame = camera()
    assert image_array(frame.depth_image, depth=True)[0, 0] == 1
    frame.image.step = 5
    with pytest.raises(ValueError, match="stride"):
        image_array(frame.image)


def test_ground_truth_or_missing_cameras_are_rejected(tmp_path, monkeypatch):
    from aha_sim_tasks.codex_policy import save_observation

    with pytest.raises(ValueError, match="standard sensor"):
        save_observation(
            replace(observation(), ground_truth=GroundTruth((0,) * 7, (0,) * 3, 2)),
            tmp_path,
        )
    with pytest.raises(ValueError, match="both wrist"):
        save_observation(replace(observation(), cameras={}), tmp_path)
