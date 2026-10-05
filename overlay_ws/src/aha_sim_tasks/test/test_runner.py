"""Episode cleanup must include detached processes, without touching other runs."""

import os
from pathlib import Path
import subprocess
import sys
import time
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
