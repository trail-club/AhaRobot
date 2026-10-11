"""Task ordering, validation and failure propagation without ROS."""

import copy
import json
from pathlib import Path
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from aha_tasks.flow import execute_flow, load_flow  # noqa: E402
from aha_tasks.runner import main  # noqa: E402

CONFIG = PACKAGE / "config" / "human_interaction.json"


def test_two_guests_then_host_and_drop():
    flow = load_flow(CONFIG)
    visited, reports = [], []

    def navigate(pose):
        visited.append(pose)
        return True

    assert execute_flow(flow, navigate, reports.append)
    assert visited == [
        flow["locations"][name]
        for name in (
            "waiting",
            "entrance",
            "study_chair_left",
            "waiting",
            "entrance",
            "study_chair_right",
            "host",
            "bag_drop",
        )
    ]
    assert sum(message.startswith("SKIPPED") for message in reports) == 10
    assert reports[-1].startswith("COMPLETED")


@pytest.mark.parametrize("failed_goal", range(8))
def test_failure_never_advances(failed_goal):
    flow, calls, reports = load_flow(CONFIG), [], []

    def navigate(pose):
        calls.append(pose)
        return len(calls) <= failed_goal

    assert not execute_flow(flow, navigate, reports.append)
    assert len(calls) == failed_goal + 1
    assert reports[-1].startswith("FAILED")
    assert not any(message.startswith("COMPLETED") for message in reports)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(frame_id="odom"),
        lambda d: d["locations"].update(entrance=[0, float("nan"), 0]),
        lambda d: d["locations"].update(entrance=[0, 0]),
        lambda d: d["locations"].update(entrance=[True, 0, 0]),
        lambda d: d["steps"][0].update(navigate="missing"),
        lambda d: d["steps"][0].update(skip="accidental navigation skip"),
        lambda d: d["steps"][1].update(skip=""),
        lambda d: d["steps"][1].update(id=d["steps"][0]["id"]),
        lambda d: d.update(steps=[]),
    ],
)
def test_bad_config_rejected_before_motion(tmp_path, mutation):
    data = copy.deepcopy(load_flow(CONFIG))
    mutation(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_flow(path)


def test_dry_run_does_not_import_ros(capsys):
    assert main(["--config", str(CONFIG), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output and "bag_drop" in output
    assert "COMPLETED" not in output
