"""Episode cleanup must include detached processes, without touching other runs."""

import os
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_sim_tasks.processes import (  # noqa: E402
    partition_processes,
    policy_partition,
    stop_group,
)


WORKER = """
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGINT, signal.SIG_IGN)
Path(sys.argv[1]).write_text('ready')
time.sleep(60)
"""
PARENT = """
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]],
                         env={**os.environ, 'GZ_PARTITION': sys.argv[5]},
                         start_new_session=True)
while not Path(sys.argv[2]).exists():
    time.sleep(0.01)
Path(sys.argv[3]).write_text(str(child.pid))
if sys.argv[4] == 'wait':
    time.sleep(60)
"""


@pytest.mark.parametrize("parent_exited", [False, True])
@pytest.mark.parametrize("separate_partition", [False, True])
def test_cleanup_stops_detached_worker_even_after_parent_exits(
    tmp_path, parent_exited, separate_partition
):
    partition = "aha-evaluation-test-" + uuid.uuid4().hex
    worker_partition = policy_partition(partition) if separate_partition else partition
    extra_partitions = (worker_partition,) if separate_partition else ()
    other_partition = partition + "-other"
    other = subprocess.Popen(
        [
            sys.executable,
            "-c",
            WORKER.replace("signal.SIG_IGN", "signal.default_int_handler"),
            str(tmp_path / "other-ready"),
        ],
        env={**os.environ, "GZ_PARTITION": other_partition},
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            PARENT,
            WORKER,
            str(tmp_path / "ready"),
            str(tmp_path / "pid"),
            "exit" if parent_exited else "wait",
            worker_partition,
        ],
        env={**os.environ, "GZ_PARTITION": partition},
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "pid").exists():
            assert time.monotonic() < deadline, "Detached worker did not start"
            time.sleep(0.01)
        worker = int((tmp_path / "pid").read_text())
        assert os.getpgid(worker) == worker
        if parent_exited:
            parent.wait(timeout=5)
        assert worker in partition_processes(worker_partition)
        stop_group(parent, partition, other_partitions=extra_partitions)
        assert not partition_processes(partition)
        assert not partition_processes(worker_partition)
        assert other.poll() is None
    finally:
        stop_group(parent, partition, other_partitions=extra_partitions)
        stop_group(other, other_partition)


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("camera_state", ["missing_tf", "recovered_tf", "disabled"])
def test_episode_bounds_unusable_camera_bundles(
    monkeypatch, tmp_path, pending, camera_state
):
    rclpy = pytest.importorskip("rclpy")
    from aha_sim_tasks import ros_environment, runner
    from aha_sim_tasks.evaluation import Task, WorldState

    wall_time = 0.0
    stopped = []
    actions = []
    adapter = ros_environment.RosEnvironment
    node = SimpleNamespace(
        camera_timeout=2,
        required={
            "joints",
            "odom",
            "state",
            "camera:head",
            "camera:left_wrist",
            "camera:right_wrist",
        },
        received={},
        world=WorldState(0, (0, 0, 0), (0, 0, 0), 0, 0),
        ready=lambda: True,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=round(wall_time * 1e9))
        ),
        observation=lambda: object(),
        stop=lambda: stopped.append("node"),
        destroy_node=lambda: stopped.append("destroy"),
    )
    node.stale_sources = lambda: adapter.stale_sources(node)
    node.fresh = lambda: adapter.fresh(node)
    node.observation_ready = lambda: (
        camera_state == "disabled"
        or wall_time < 1
        or (camera_state == "recovered_tf" and 2 <= wall_time < 3)
    )

    def spin_once(environment, timeout_sec):
        nonlocal wall_time
        if stopped:
            return
        wall_time += 0.25
        environment.world = WorldState(wall_time, (0, 0, 0), (0, 0, 0), 0, 0)
        # Images and other streams keep arriving even while TF is unusable.
        environment.received = {source: wall_time for source in environment.required}

    policy = SimpleNamespace(
        ready=True,
        pending=object() if pending else None,
        poll=lambda: None,
        act=actions.append,
        close=lambda: stopped.append("policy"),
    )
    simulator = SimpleNamespace(
        poll=lambda: None,
        send_signal=lambda sig: stopped.append("simulator"),
        wait=lambda timeout: None,
    )
    clock = SimpleNamespace(monotonic=lambda: wall_time, sleep=lambda seconds: None)
    monkeypatch.setattr(runner, "time", clock)
    monkeypatch.setattr(ros_environment, "time", clock)
    monkeypatch.setattr(ros_environment, "RosEnvironment", lambda *args, **kwargs: node)
    monkeypatch.setattr(runner, "PolicyProcess", lambda *args, **kwargs: policy)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: simulator)
    monkeypatch.setattr(rclpy, "init", lambda: None)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(rclpy, "spin_once", spin_once)
    monkeypatch.setenv("GZ_PARTITION", "camera-deadline-test")
    args = SimpleNamespace(
        policy="noop",
        policy_ros_domain_id=88,
        policy_startup_timeout=120,
        policy_timeout=30,
        headless=True,
        cameras=camera_state != "disabled",
        camera_view=False,
        camera_timeout=2,
        head_image_topic="",
        wall_timeout=20,
        output=tmp_path / "result.json",
    )
    task = Task("pick", "Pick an apple", "pick", 6, 1, {"min_height": 0.24})
    assert runner.run_episode(args, task) == 1
    result = json.loads(args.output.read_text())
    if camera_state == "disabled":
        assert result["status"] == "timeout"
        assert "reason" not in result
    else:
        assert result["status"] == "error"
        assert "Camera bundle" in result["reason"]
        assert "timestamped transforms" in result["reason"]
        expected = 2.75 if camera_state == "missing_tf" else 4.75
        assert result["wall_seconds"] == expected
    assert {"node", "destroy", "policy", "simulator"} <= set(stopped)
    assert bool(actions) is not pending
