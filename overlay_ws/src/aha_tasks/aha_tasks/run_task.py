"""Start a run on the task_runner, wait until it stops and print where the time went.

For scripted end-to-end checks: waits until the runner is ready, starts a run
(selection and failure policy as in the GUI), follows /aha/tasks/status and
prints each executed subgoal's wall time, including the move that first brings
the robot to the memory's location, the time per skill and the total. The run
stops when it ends, pauses after a failure (policy pause: decide in the GUI)
or when the runner restarts; exit 0 only when it completed. The decisions and
the summary are ROS free (``start_run``, ``follow_run``, ``summarize``).
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys
import threading
import time

from aha_tasks.core.engine import STARTABLE
from aha_tasks.runner import COMMAND_SERVICE, STATUS_TOPIC

# States in which the run waits for nobody: it ended or needs a decision.
STOPPED = ("completed", "failed", "aborted", "paused", "error")
CALL_TIMEOUT = 10.0  # s (wall clock) for the runner to answer a command
APPEAR_TIMEOUT = 10.0  # s (wall clock) for a started run to show in the status
START = re.compile(r"^Starting (\S+) \(attempt \d+\)$")
RESULT = re.compile(r"^(SUCCEEDED|FAILED|CANCELLED|SKIPPED|UNSUPPORTED) (\S+?)(?::|$)")
READY_MESSAGE = "Ready"
WAITING_MESSAGE = "Waiting for active Nav2"


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
    """Seconds from the latest "Waiting for active Nav2" to the "Ready" after it."""
    waiting = ready = None
    for entry in entries:
        if entry["message"] == WAITING_MESSAGE:
            waiting = entry["time"]
        elif entry["message"] == READY_MESSAGE and waiting is not None:
            ready, waiting = round(entry["time"] - waiting, 2), None
    return ready


def start_run(call, wait_ready, deadline, now=time.monotonic, pause=time.sleep):
    """Start a run once a runner is ready; (accepted reply, "") or (None, error).

    wait_ready(timeout) returns the first startable or error status, or None;
    call(timeout) the reply to the start command, or None when no answer came
    in time (no service, or the runner exited). Retries until `deadline`.
    """
    while (remaining := deadline - now()) > 0:
        status = wait_ready(remaining)
        if status is None:
            break
        if status["state"] == "error":
            return None, f"Runner error: {status['message']}"
        reply = call(min(CALL_TIMEOUT, max(deadline - now(), 0.0)))
        if reply is not None:
            if reply.accepted:
                return reply, ""
            # A restarted runner answers before it is ready, while the old
            # runner's status is still the latest one seen.
            if not reply.message.startswith("Cannot start while"):
                return None, f"Start rejected: {reply.message}"
        pause(1.0)
    return None, "Runner not ready in time"


def follow_run(
    get_status, requested, timeout=None, now=time.monotonic, pause=time.sleep
):
    """Follow the statuses of a run started at `requested` (epoch s).

    Returns (outcome, last status of the run or None). The outcome is the
    stopped state (STOPPED), "restarted" when a status of another runner
    follows the run's, "lost" when the run never shows, or "timeout".
    """
    begin = now()
    last = None
    while True:
        status = get_status()
        run = (status or {}).get("run") or {}
        # The runner stamps the run with its clock on this machine.
        if run.get("started_at", -math.inf) >= requested - 1.0:
            last = status
            if status["state"] in STOPPED:
                return status["state"], last
        elif last is not None and status is not None:
            return "restarted", last
        if last is None and now() - begin > APPEAR_TIMEOUT:
            return "lost", None
        if timeout is not None and now() - begin > timeout:
            return "timeout", last
        pause(0.2)


OUTCOMES = {
    "paused": "paused after a failure; decide in the GUI (retry, rewind, skip, abort)",
    "restarted": "the runner restarted during the run",
    "lost": "the started run did not show in the status",
    "timeout": "the run did not stop within --timeout",
}


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

    def wait_ready(timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            current = status()
            if current is not None and current["state"] in STARTABLE + ("error",):
                return current
            time.sleep(0.2)
        return None

    request = TaskCommand.Request(
        command="start",
        subgoals=[s for s in args.subgoals.split(",") if s],
        on_failure=args.on_failure,
    )

    def call(timeout):
        end = time.monotonic() + timeout
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        while not future.done():
            if time.monotonic() > end:
                client.remove_pending_request(future)
                return None
            time.sleep(0.05)
        return future.result()

    code = 1
    try:
        requested = time.time()
        reply, error = start_run(
            call, wait_ready, time.monotonic() + args.ready_timeout
        )
        if reply is None:
            log.error(error)
            return 1
        with lock:
            seconds = ready_seconds(book.entries)
        if seconds is not None:
            log.info(f"Runner ready {seconds:.1f} s after it waited for Nav2")
        log.info(reply.message)
        outcome, final = follow_run(status, requested, args.timeout or None)
        if outcome in OUTCOMES:
            log.error(OUTCOMES[outcome])
        if final is not None:
            with lock:
                entries = list(book.entries)
            summary = summarize(final, entries)
            print(format_summary(summary), flush=True)
            if args.output:
                args.output.write_text(
                    json.dumps(
                        {
                            "outcome": outcome,
                            "summary": summary,
                            "status": final,
                            "log": entries,
                        },
                        indent=1,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
        code = 0 if outcome == "completed" else 1
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
