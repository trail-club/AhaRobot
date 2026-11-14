"""Debug and final end-to-end runs: the launch switch, teleport checks and run summaries."""

import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.core import load_spec  # noqa: E402
from aha_tasks.run_task import (  # noqa: E402
    APPEAR_TIMEOUT,
    LogBook,
    follow_run,
    ready_seconds,
    start_run,
    summarize,
)
from aha_tasks.skills import resolve_pose  # noqa: E402
from aha_tasks.teleport import (  # noqa: E402
    footprint_blocked,
    pose_error,
    wait_sim,
    yaw_of,
)

PACKAGE = Path(__file__).resolve().parents[1]
MAP = PACKAGE.parent / "aha_navigation" / "maps"
LAUNCH = PACKAGE / "launch" / "task.launch.py"


def served(pixel, occupied=0.65, free=0.25):
    """map_server's trinary value of a pixel (rcjo2026.yaml thresholds)."""
    occupancy = (255 - pixel) / 255.0
    if occupancy > occupied:
        return 100
    return 0 if occupancy < free else -1


def demo_map():
    """The RCJO2026 map as map_server serves it on /map."""
    raw = (MAP / "rcjo2026.pgm").read_bytes()
    _, width, height, _ = raw.split(maxsplit=4)[:4]
    width, height = int(width), int(height)
    pixels = raw[-width * height :]
    data = []
    for j in range(height):  # row 0 of the grid is the bottom row of the image
        row = pixels[(height - 1 - j) * width : (height - j) * width]
        data += [served(p) for p in row]
    return {
        "data": data,
        "width": width,
        "height": height,
        "resolution": 0.05,
        "origin": [-5.5, -4.5],
    }


def entry(time, message, subgoal=None):
    return {"time": time, "level": "info", "subgoal": subgoal, "message": message}


class TeleportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = demo_map()

    def test_task_locations_are_free_for_the_robot(self):
        for path in sorted((PACKAGE / "config").glob("*.json")):
            spec = load_spec(path)
            for label in spec.locations:
                x, y, _ = resolve_pose(spec, label)
                with self.subTest(task=spec.task, location=label):
                    self.assertIsNone(footprint_blocked(self.grid, x, y, 0.40))

    def test_walls_and_outside_are_refused(self):
        # The entrance door at x = -5.0 is occupied on the map; the grid ends at
        # x = -5.5.
        self.assertIn("occupied", footprint_blocked(self.grid, -5.0, 1.5, 0.40))
        self.assertIn("unknown", footprint_blocked(self.grid, -7.0, 1.5, 0.40))
        self.assertIn("unknown", footprint_blocked(self.grid, 50.0, 50.0, 0.40))

    def test_pose_error_wraps_the_yaw(self):
        distance, angle = pose_error(
            [1.0, 2.0, math.pi - 0.01], [1.0, 2.1, -math.pi + 0.01]
        )
        self.assertAlmostEqual(distance, 0.1)
        self.assertAlmostEqual(angle, 0.02)

    def test_yaw_of_a_quaternion(self):
        q = SimpleNamespace(x=0.0, y=0.0, z=math.sin(0.6), w=math.cos(0.6))
        self.assertAlmostEqual(yaw_of(q), 1.2)


class RunSummaryTests(unittest.TestCase):
    def status(self):
        subgoals = [
            ("a.go", "navigate", "succeeded", 1),
            ("a.find", "find_person", "skipped", 2),
            ("a.ask", "ask", "unsupported", 0),
            ("b.go", "navigate", "succeeded", 1),
            ("b.mock", "navigate", "bypassed", 0),
        ]
        return {
            "state": "failed",
            "run": {"number": 2, "started_at": 100.0, "finished_at": 160.0},
            "subgoals": [
                {"id": i, "skill": k, "status": s, "attempts": n, "message": ""}
                for i, k, s, n in subgoals
            ],
        }

    def test_wall_time_includes_restoring_and_retries(self):
        entries = [
            entry(50.0, "Ready"),
            entry(60.0, "Starting a.go (attempt 1)"),  # an earlier run
            entry(100.0, "Run 2: 4 subgoal(s), on failure skip"),
            entry(100.5, "Starting a.go (attempt 1)"),
            entry(130.5, "SUCCEEDED a.go: Arrived at a", "a.go"),
            entry(130.6, "Starting a.find (attempt 1)"),
            entry(131.0, "Restoring robot.location = a"),
            entry(140.6, "FAILED a.find: No person", "a.find"),
            entry(140.7, "Starting a.find (attempt 2)"),
            entry(150.6, "FAILED a.find: No person", "a.find"),
            entry(150.7, "SKIPPED a.find: Skipped after failure: No person"),
            entry(150.8, "UNSUPPORTED a.ask: Skill 'ask' is not implemented"),
            entry(150.9, "Starting b.go (attempt 1)"),
            entry(159.9, "SUCCEEDED b.go", "b.go"),
        ]
        book = LogBook()
        book.add(entries[5:])
        book.add(entries)  # repeated entries of later status messages
        summary = summarize(self.status(), book.entries)
        rows = {row["id"]: row for row in summary["subgoals"]}
        self.assertEqual(list(rows), ["a.go", "a.find", "b.go"])
        self.assertAlmostEqual(rows["a.go"]["seconds"], 30.0)
        self.assertAlmostEqual(rows["a.find"]["seconds"], 20.1)
        self.assertEqual(rows["a.find"]["status"], "skipped")
        self.assertEqual(summary["per_skill"], {"navigate": 39.0, "find_person": 20.1})
        self.assertEqual(summary["total_seconds"], 60.0)
        self.assertEqual(summary["counts"]["bypassed"], 1)
        self.assertIsNone(ready_seconds(book.entries))  # its wait was cut off

    def test_ready_after_the_latest_wait_for_nav2(self):
        entries = [entry(10.0, "Waiting for active Nav2"), entry(32.5, "Ready")]
        self.assertEqual(ready_seconds(entries), 22.5)
        self.assertIsNone(ready_seconds(entries[:1]))
        restarted = entries + [
            entry(40.0, "Waiting for active Nav2"),
            entry(42.0, "Ready"),
        ]
        self.assertEqual(ready_seconds(restarted), 2.0)


class FakeClock:
    """Wall clock that sleep() advances; the simulation runs at `rtf`."""

    def __init__(self, rtf=1.0):
        self.wall = self.sim = 0.0
        self.rtf = rtf

    def now(self):
        return self.wall

    def sim_now(self):
        return self.sim

    def sleep(self, seconds):
        self.wall += seconds
        self.sim += self.rtf * seconds


def reply(accepted, message=""):
    return SimpleNamespace(accepted=accepted, message=message)


def runner_status(state, started_at=None):
    run = None if started_at is None else {"started_at": started_at}
    return {"state": state, "message": "", "run": run}


class StartTests(unittest.TestCase):
    def start(self, replies, statuses=None, deadline=60.0):
        clock = FakeClock()
        replies = iter(replies)
        statuses = iter(statuses or [])

        def call(timeout):
            answer = next(replies, None)
            if answer is None:  # no answer: the whole timeout passes
                clock.sleep(timeout)
            return answer

        def wait_ready(timeout):
            return next(statuses, runner_status("idle"))

        return start_run(call, wait_ready, deadline, clock.now, clock.sleep), clock

    def test_a_restarting_runner_is_asked_again(self):
        (answer, error), _ = self.start(
            [reply(False, "Cannot start while preparing"), reply(True, "Started")]
        )
        self.assertTrue(answer.accepted)
        self.assertEqual(error, "")

    def test_a_runner_that_never_answers_ends_at_the_deadline(self):
        (answer, error), clock = self.start([], deadline=30.0)
        self.assertIsNone(answer)
        self.assertIn("not ready", error)
        self.assertLess(clock.now(), 32.0)

    def test_rejections_and_errors_end_the_start(self):
        (answer, error), _ = self.start([reply(False, "Unknown failure policy: x")])
        self.assertIsNone(answer)
        self.assertIn("Unknown failure policy", error)
        (answer, error), _ = self.start([], [runner_status("error")])
        self.assertIn("Runner error", error)


class FollowTests(unittest.TestCase):
    def follow(self, statuses, timeout=None):
        clock = FakeClock()
        statuses = list(statuses)

        def get_status():
            return statuses.pop(0) if len(statuses) > 1 else statuses[0]

        return follow_run(get_status, 100.0, timeout, clock.now, clock.sleep)

    def test_a_pause_stops_the_follow(self):
        outcome, last = self.follow(
            [
                runner_status("idle", 50.0),  # the previous run, still latest
                runner_status("running", 100.2),
                runner_status("paused", 100.2),
            ]
        )
        self.assertEqual(outcome, "paused")
        self.assertEqual(last["state"], "paused")

    def test_a_restarted_runner_stops_the_follow(self):
        outcome, last = self.follow(
            [runner_status("running", 100.2), runner_status("preparing")]
        )
        self.assertEqual(outcome, "restarted")
        self.assertEqual(last["state"], "running")

    def test_completion_loss_and_timeout(self):
        self.assertEqual(
            self.follow([runner_status("completed", 100.0)])[0], "completed"
        )
        self.assertEqual(self.follow([runner_status("idle")]), ("lost", None))
        outcome, _ = self.follow([runner_status("running", 100.0)], timeout=5.0)
        self.assertEqual(outcome, "timeout")
        self.assertGreater(APPEAR_TIMEOUT, 0.0)


class SimulationWaitTests(unittest.TestCase):
    def test_a_slow_simulation_only_takes_longer(self):
        clock = FakeClock(rtf=0.1)
        outcome = wait_sim(clock.sim_now, 7.7, wall_now=clock.now, sleep=clock.sleep)
        self.assertEqual(outcome, "elapsed")
        self.assertGreater(clock.now(), 70.0)

    def test_a_stopped_clock_ends_the_wait(self):
        clock = FakeClock(rtf=0.0)
        outcome = wait_sim(
            clock.sim_now, 1.0, stall=10.0, wall_now=clock.now, sleep=clock.sleep
        )
        self.assertEqual(outcome, "stalled")
        self.assertLess(clock.now(), 11.0)

    def test_a_condition_or_a_cancel_ends_the_wait(self):
        clock = FakeClock(rtf=0.5)
        self.assertEqual(
            wait_sim(
                clock.sim_now,
                10.0,
                done=lambda: clock.sim > 2.0,
                wall_now=clock.now,
                sleep=clock.sleep,
            ),
            "done",
        )
        self.assertEqual(
            wait_sim(
                clock.sim_now,
                cancelled=lambda: True,
                wall_now=clock.now,
                sleep=clock.sleep,
            ),
            "cancelled",
        )


class LaunchSwitchTests(unittest.TestCase):
    """The e2e switch of task.launch.py; needs the ROS launch packages."""

    @classmethod
    def setUpClass(cls):
        try:
            spec = importlib.util.spec_from_file_location("task_launch", LAUNCH)
            cls.launch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.launch)
        except ImportError as error:  # host without ROS
            raise unittest.SkipTest(f"no ROS launch: {error}")

    def test_final_is_the_real_stack(self):
        self.assertEqual(self.launch.e2e_choices("final"), ("nav2", "model"))
        self.assertEqual(
            self.launch.e2e_choices("final", "nav2", "model"), ("nav2", "model")
        )

    def test_debug_defaults_to_the_stand_ins(self):
        self.assertEqual(self.launch.e2e_choices("debug"), ("teleport", "truth"))
        self.assertEqual(
            self.launch.e2e_choices("debug", "nav2", "model"), ("nav2", "model")
        )

    def test_final_refuses_stand_ins(self):
        for navigation, perception in (("teleport", ""), ("", "truth")):
            with self.subTest(navigation=navigation, perception=perception):
                with self.assertRaises(ValueError):
                    self.launch.e2e_choices("final", navigation, perception)

    def test_unknown_values_are_refused(self):
        for args in (("fast",), ("debug", "walk"), ("debug", "", "sam")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.launch.e2e_choices(*args)


if __name__ == "__main__":
    unittest.main()
