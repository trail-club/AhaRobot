"""Start a run on the task_runner, wait until it ends and print where the time went.

For scripted end-to-end checks: waits until the runner is ready, starts a run
(selection and failure policy as in the GUI), follows /aha/tasks/status and
prints each executed subgoal's wall time, including the move that first brings
the robot to the memory's location, the time per skill and the total. Exits 0
when the run completed. The summary is computed from the status log without
ROS (``summarize``), so it can also be applied to a saved status.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import threading
import time

from aha_tasks.runner import COMMAND_SERVICE, STATUS_TOPIC

READY = ("idle", "completed", "failed", "aborted")
ENDED = ("completed", "failed", "aborted", "error")
START = re.compile(r"^Starting (\S+) \(attempt \d+\)$")
RESULT = re.compile(r"^(SUCCEEDED|FAILED|CANCELLED|SKIPPED|UNSUPPORTED) (\S+?)(?::|$)")
READY_MESSAGE = "Ready"


class LogBook:
    """Log entries of a runner, kept beyond the 200 of one status message."""

    def __init__(self):
        self.entries = []
        self._seen = set()

    def add(self, entries):
        for entry in entries:
            key = (entry["time"], entry["message"], entry.get("subgoal"))
            if key not in self._seen:
                self._seen.add(key)
                self.entries.append(entry)
        self.entries.sort(key=lambda e: e["time"])


def summarize(status, entries):
    """Wall time per executed subgoal and per skill of the latest run.

    A subgoal's time runs from its "Starting" entry to its result entry, so it
    includes restoring robot.location before the skill and the retries.
    """
    run = status.get("run") or {}
    started, finished = run.get("started_at"), run.get("finished_at")
    skills = {s["id"]: s["skill"] for s in status["subgoals"]}
    records = {s["id"]: s for s in status["subgoals"]}
    begin, rows = {}, {}
    for entry in entries:
        if started is not None and entry["time"] < started:
            continue
        message = entry["message"]
        match = START.match(message)
        if match:
            begin.setdefault(match.group(1), entry["time"])
            continue
        match = RESULT.match(message)
        if match and match.group(2) in begin:
            subgoal = match.group(2)
            rows[subgoal] = {
                "id": subgoal,
                "skill": skills.get(subgoal, "?"),
                "status": records.get(subgoal, {}).get(
                    "status", match.group(1).lower()
                ),
                "seconds": round(entry["time"] - begin[subgoal], 2),
                "attempts": records.get(subgoal, {}).get("attempts", 0),
                "message": records.get(subgoal, {}).get("message", ""),
            }
    order = [s["id"] for s in status["subgoals"]]
    rows = [rows[i] for i in order if i in rows]
    per_skill = {}
    for row in rows:
        per_skill[row["skill"]] = round(
            per_skill.get(row["skill"], 0.0) + row["seconds"], 2
        )
    counts = {}
    for subgoal in status["subgoals"]:
        counts[subgoal["status"]] = counts.get(subgoal["status"], 0) + 1
    total = None
    if started is not None and finished is not None:
        total = round(finished - started, 2)
    return {
        "state": status["state"],
        "total_seconds": total,
        "subgoals": rows,
        "per_skill": dict(sorted(per_skill.items(), key=lambda kv: -kv[1])),
        "counts": counts,
    }


def ready_seconds(entries):
    """Seconds from the runner's first log entry to "Ready", or None."""
    times = [e["time"] for e in entries if e["message"] == READY_MESSAGE]
    if not entries or not times:
        return None
    return round(times[0] - entries[0]["time"], 2)


def format_summary(summary):
    lines = [f"{'subgoal':32} {'skill':22} {'status':11} {'s':>8}  message"]
    for row in summary["subgoals"]:
        lines.append(
            f"{row['id']:32} {row['skill']:22} {row['status']:11} "
            f"{row['seconds']:8.1f}  {row['message'][:70]}"
        )
    lines.append(
        "per skill: "
        + ", ".join(f"{k} {v:.1f} s" for k, v in summary["per_skill"].items())
    )
    lines.append(
        "statuses: "
        + ", ".join(f"{v} {k}" for k, v in sorted(summary["counts"].items()))
    )
    total = summary["total_seconds"]
    lines.append(
        f"run {summary['state']} in {total:.1f} s"
        if total is not None
        else f"run {summary['state']}"
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subgoals",
        default="",
        help="Selection as in task_runner (ids, globs, ranges); empty = runner default",
    )
    parser.add_argument(
        "--on-failure",
        default="",
        help="pause | retry | skip | abort; empty = runner default",
    )
    parser.add_argument(
        "--ready-timeout", type=float, default=600.0, help="s (wall clock)"
    )
    parser.add_argument(
        "--timeout", type=float, default=0.0, help="s for the run; 0 = none"
    )
    parser.add_argument(
        "--output", type=Path, help="Write the summary, status and log as JSON"
    )
    args, ros_args = parser.parse_known_args(argv)

    from aha_msgs.srv import TaskCommand
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init(args=ros_args)
    node = Node("run_task")
    lock = threading.Lock()
    latest = {"status": None}
    book = LogBook()

    def on_status(message):
        status = json.loads(message.data)
        with lock:
            latest["status"] = status
            book.add(status.get("log", []))

    node.create_subscription(
        String,
        STATUS_TOPIC,
        on_status,
        QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        ),
    )
    client = node.create_client(TaskCommand, COMMAND_SERVICE)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    log = node.get_logger()

    def status():
        with lock:
            return latest["status"]

    def wait(condition, timeout=None):
        """The first status that meets `condition` within `timeout` s (None: no limit)."""
        deadline = float("inf") if timeout is None else time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = status()
            if current is not None and condition(current):
                return current
            time.sleep(0.2)
        return None

    def start(deadline):
        """Start a run once a runner is ready; the accepted reply or None."""
        request = TaskCommand.Request(
            command="start",
            subgoals=[s for s in args.subgoals.split(",") if s],
            on_failure=args.on_failure,
        )
        while time.monotonic() < deadline:
            current = wait(
                lambda s: s["state"] in READY + ("error",),
                deadline - time.monotonic(),
            )
            if current is None or current["state"] == "error":
                log.error(f"Runner not ready: {current and current['message']}")
                return None
            if not client.wait_for_service(timeout_sec=10.0):
                log.error(f"{COMMAND_SERVICE} is not available")
                return None
            future = client.call_async(request)
            while not future.done():
                time.sleep(0.05)
            reply = future.result()
            if reply.accepted:
                return reply
            # A restarted runner answers before it is ready, after the old
            # runner's status was seen.
            if not reply.message.startswith("Cannot start while"):
                log.error(f"Start rejected: {reply.message}")
                return None
            time.sleep(1.0)
        log.error("Runner not ready in time")
        return None

    code = 1
    try:
        requested = time.time()
        reply = start(time.monotonic() + args.ready_timeout)
        if reply is None:
            return 1
        with lock:
            seconds = ready_seconds(book.entries)
        if seconds is not None:
            log.info(f"Runner ready {seconds:.1f} s after it started")
        log.info(reply.message)
        # The runner stamps the run with its clock on this machine.
        final = wait(
            lambda s: (s.get("run") or {}).get("started_at", 0.0) >= requested - 1.0
            and s["state"] in ENDED,
            args.timeout or None,
        )
        if final is None:
            log.error("The run did not end in time")
            final = status()
        with lock:
            entries = list(book.entries)
        summary = summarize(final, entries)
        print(format_summary(summary), flush=True)
        if args.output:
            args.output.write_text(
                json.dumps(
                    {"summary": summary, "status": final, "log": entries},
                    indent=1,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        code = 0 if final["state"] == "completed" else 1
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spinner.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
