"""Stop episode process groups and descendants identified by their partitions."""

import os
from pathlib import Path
import signal
import subprocess
import time


def policy_partition(partition):
    return partition + "-policy"


def partition_processes(partition):
    """Find this episode's processes, including Gazebo's detached GUI/server."""
    if not partition:
        return set()
    marker = f"GZ_PARTITION={partition}".encode()
    processes = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            if marker in (entry / "environ").read_bytes().split(b"\0"):
                processes.add(int(entry.name))
        except (OSError, ProcessLookupError):
            continue
    return processes


def signal_partition(partition, sig):
    for pid in partition_processes(partition):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def stop_group(process, partition, *, other_partitions=(), timeout=15):
    partitions = (partition, *other_partitions)
    # Gazebo's combined GUI/server CLI creates separate process groups, so a
    # group signal alone can leave a controller manager in the ROS domain.
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    for marker in partitions:
        signal_partition(marker, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for marker in partitions:
        signal_partition(marker, signal.SIGKILL)
    process.wait()
    deadline = time.monotonic() + 3
    while any(partition_processes(marker) for marker in partitions):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Simulator processes did not stop: {partition}")
        for marker in partitions:
            signal_partition(marker, signal.SIGKILL)
        time.sleep(0.05)
