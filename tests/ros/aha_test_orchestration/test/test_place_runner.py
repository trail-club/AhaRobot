"""Message conversions of place_runner (needs the ROS messages: colcon test)."""

import json

import numpy as np
from aha_msgs.msg import Surface, SurfaceArray
from conftest import CATEGORIES
from geometry_msgs.msg import Point32
from test_llm import ScriptedClient, call_message

from aha_test_orchestration.place_decision import VARIANTS
from aha_test_orchestration.place_runner import (
    ObjectAccumulator,
    PlaceLlmRunner,
    RunnerOptions,
    goal_from,
    surfaces_from_msg,
)
from aha_test_ui import records


def test_goal_variant_a(summary, held):
    decision = VARIANTS["A"].validate({"surface_id": 3, "near_object_id": 2}, summary)
    goal = goal_from(decision.request, held, "right", "side", "auto")
    assert list(goal.surface_ids) == [3]
    assert goal.near.header.frame_id == "odom"
    assert (goal.near.point.x, goal.near.point.y) == (0.6, -0.1)
    assert not goal.restrict_height and goal.furniture == ""
    assert abs(goal.object_size.z - 0.12) < 1e-9 and goal.arm == "right"


def test_goal_variant_b(summary, held):
    decision = VARIANTS["B"].validate({"height": 0.91}, summary)
    goal = goal_from(decision.request, held)
    assert goal.restrict_height and abs(goal.surface_height - 0.91) < 1e-6
    assert goal.near.header.frame_id == "" and len(goal.surface_ids) == 0


def test_goal_variant_c(summary, held):
    decision = VARIANTS["C"].validate(
        {"furniture": "shelf", "level": 5, "next_to": "empty"}, summary
    )
    goal = goal_from(decision.request, held)
    assert goal.furniture == "shelf" and goal.furniture_level == 5
    assert goal.near.header.frame_id == ""
    decision = VARIANTS["C"].validate(
        {"furniture": "shelf", "level": 4, "next_to": "O3"}, summary
    )
    goal = goal_from(decision.request, held)
    assert goal.near.header.frame_id == "odom" and goal.furniture_level == 4


def test_surfaces_from_msg():
    msg = SurfaceArray()
    s = Surface(id=7, label="shelf", furniture="shelf", furniture_level=3, visible=True)
    s.plane.height = 0.61
    s.plane.hull.points = [
        Point32(x=0.5, y=-0.3),
        Point32(x=0.9, y=-0.3),
        Point32(x=0.9, y=0.3),
    ]
    msg.surfaces = [s]
    (info,) = surfaces_from_msg(msg)
    assert info.id == 7 and info.level == 3 and info.furniture == "shelf"
    assert abs(info.height - 0.61) < 1e-9 and len(info.hull) == 3


def test_accumulator_mean_and_majority():
    acc = ObjectAccumulator()
    acc.add(4, "orange", (1.0, 0.0, 0.9), (0.07, 0.07, 0.07))
    acc.add(4, "orange", (1.2, 0.0, 0.9), (0.07, 0.07, 0.07))
    acc.add(4, "lemon", (1.1, 0.0, 0.9), (0.07, 0.07, 0.07))
    (obj,) = acc.objects()
    assert obj.label == "orange" and abs(obj.position[0] - 1.1) < 1e-9
    acc.clear()
    assert acc.objects() == []


class Categories:
    def categories(self):
        return CATEGORIES


class FakeUi:
    """aha_test_ui.publisher.UiPublisher keeping the records."""

    def __init__(self):
        self.states, self.events, self.chats = [], [], []

    def status(self, state, detail=""):
        self.states.append(records.status("place_runner", state, detail))

    def event(self, text, level="info"):
        self.events.append(records.event("place_runner", text, level))

    def chat(self, chat_id, title, messages, notes=None, done=False):
        self.chats.append(
            records.chat("place_runner", chat_id, title, messages, notes, done)
        )


def offline_runner(summary, replies, places):
    """A runner without ROS: kept summary, scripted LLM and PlaceObject results."""
    runner = PlaceLlmRunner.__new__(PlaceLlmRunner)
    runner.options = RunnerOptions(place_retries=2)
    runner.memory = summary
    runner.client = ScriptedClient(replies)
    runner.object_list = Categories()
    runner.transform = lambda target, source: np.eye(4)
    runner.ui = FakeUi()
    sent = []

    def execute(decision, held, summary=None):
        sent.append(decision)
        return places.pop(0)

    runner.execute = execute
    return runner, sent


def failure(status):
    return {"status": status, "success": False, "message": "no reachable spot"}


def success(xyz):
    return {
        "status": "OK",
        "success": True,
        "message": "placed",
        "place_pose": list(xyz),
        "place_frame": "base_footprint",
    }


def test_failed_place_goes_back_to_the_llm(summary, held):
    replies = [
        call_message(
            '{"reason": "r", "furniture": "shelf", "level": 4, "next_to": "O3"}'
        ),
        # The failed place again is invalid, then another one.
        call_message(
            '{"reason": "r", "furniture": "shelf", "level": 4, "next_to": "O3"}', "c2"
        ),
        call_message(
            '{"reason": "r", "furniture": "shelf", "level": 3, "next_to": "O1"}', "c3"
        ),
    ]
    runner, sent = offline_runner(
        summary, replies, [failure("UNREACHABLE"), success((0.6, -0.25, 0.61))]
    )
    report = runner.run(held, "C", reuse=True)
    assert report["reused"] and report["observe_s"] == 0.0
    assert [(d.level, d.next_to) for d in sent] == [(4, 3), (3, 1)]
    assert report["place"]["success"] and report["decision"]["level"] == 3
    assert len(report["attempts"]) == 2 and report["llm_calls"] == 3
    # The failure reached the LLM as the result of its call.
    second = runner.client.requests[1][0]
    assert second[-1]["role"] == "tool" and "UNREACHABLE" in second[-1]["content"]
    third = runner.client.requests[2][0]
    assert "already failed" in third[-1]["content"]
    # The monitor got the conversation (the place results as notes after the
    # answers), the states and the events.
    record = json.loads(json.dumps(runner.ui.chats[-1]))
    assert record["done"] and record["title"].startswith(held.label)
    page = records.chat_html(record)
    assert page.index("place UNREACHABLE") < page.index("already failed")
    assert page.index("already failed") < page.index("place OK")
    assert [s["state"] for s in runner.ui.states][-1] == "done"
    assert any("UNREACHABLE" in e["text"] for e in runner.ui.events)
    # The kept summary has the placed cola on level 3 (center above the bottom).
    level = runner.memory.level(3)
    assert level.object_ids == (1, 2, 6)
    assert runner.memory.object(6).label == "cola"
    assert abs(runner.memory.object(6).position[2] - (0.61 + 0.06)) < 1e-9


class Base:
    def __init__(self, ok=True):
        self.ok, self.moves = ok, []

    def move_to(self, x, y, yaw):
        self.moves.append((x, y, yaw))
        return {"ok": self.ok, "message": "" if self.ok else "blocked"}


def runner_with_goals(results, move_base=True, base_ok=True):
    runner = PlaceLlmRunner.__new__(PlaceLlmRunner)
    runner.options = RunnerOptions(move_base=move_base)
    runner.object_list = type(
        "Objects", (), {"category": lambda self, label: CATEGORY.get(label, "")}
    )()
    runner.base = Base(base_ok)
    runner.ui = FakeUi()
    goals = []

    def send(goal):
        goals.append((list(goal.avoid), goal.near_max_distance, list(goal.group)))
        return dict(results.pop(0))

    runner.send = send
    return runner, goals


CATEGORY = {label: c for c, labels in CATEGORIES.items() for label in labels}
UNREACHABLE = {
    "status": "UNREACHABLE",
    "success": False,
    "suggested_base_pose": [3.0, 2.9, 1.57],
}
OK = {"status": "OK", "success": True}


def next_to_orange(summary):
    return VARIANTS["C"].validate(
        {"furniture": "shelf", "level": 4, "next_to": "O3"}, summary
    )


def test_next_to_goal_avoids_the_other_category_of_the_level(summary, held):
    runner, goals = runner_with_goals([OK])
    place = runner.execute(next_to_orange(summary), held, summary)
    ((avoid, near_max, group),) = goals
    tuna = summary.object(4).position
    assert [(p.x, p.y, p.z) for p in avoid] == [tuna] and near_max == 0.15
    assert group == []
    assert place["goals"] == ["OK"] and not place["compromise"]
    assert place["base_move"] is None


def test_next_to_goal_carries_the_other_members_of_the_group(summary, held):
    runner, goals = runner_with_goals([OK])
    tea = VARIANTS["C"].validate(
        {"furniture": "shelf", "level": 3, "next_to": "O1"}, summary
    )
    runner.execute(tea, held, summary)
    ((avoid, _, group),) = goals
    lemonade = summary.object(2).position
    assert avoid == [] and [(p.x, p.y, p.z) for p in group] == [lemonade]


def test_out_of_reach_moves_the_base_and_tries_again(summary, held):
    runner, goals = runner_with_goals([UNREACHABLE, OK])
    place = runner.execute(next_to_orange(summary), held, summary)
    assert runner.base.moves == [(3.0, 2.9, 1.57)]
    assert place["goals"] == ["UNREACHABLE", "OK"] and place["success"]
    assert [g[1] for g in goals] == [0.15, 0.15]


def test_without_moving_the_best_reachable_spot_is_taken(summary, held):
    for move_base, base_ok in ((False, True), (True, False)):
        runner, goals = runner_with_goals([UNREACHABLE, OK], move_base, base_ok)
        place = runner.execute(next_to_orange(summary), held, summary)
        assert place["compromise"] and place["success"]
        assert [g[1] for g in goals] == [0.15, 0.0]
    # An empty-part decision has no near point: one plain goal.
    runner, goals = runner_with_goals([OK])
    empty = VARIANTS["C"].validate(
        {"furniture": "shelf", "level": 5, "next_to": "empty"}, summary
    )
    runner.execute(empty, held, summary)
    assert goals == [([], 0.0, [])]


def test_no_retry_once_the_object_is_released(summary, held):
    replies = [
        call_message(
            '{"reason": "r", "furniture": "shelf", "level": 4, "next_to": "O3"}'
        )
    ]
    runner, sent = offline_runner(summary, replies, [failure("NOT_AT_SPOT")])
    report = runner.run(held, "C", reuse=True)
    assert len(sent) == 1 and not report["place"]["success"]
    assert report["place"]["status"] == "NOT_AT_SPOT"
    assert runner.memory is summary  # nothing placed


class Done:
    """A future that is done, or becomes done when the test sets ``is_done``."""

    def __init__(self, value=None, done=True):
        self.value, self.is_done = value, done

    def done(self):
        return self.is_done

    def result(self):
        return self.value


def test_timed_out_place_goal_is_canceled_before_going_on():
    from aha_test_msgs.action import PlaceObject

    canceled = PlaceObject.Result(status=PlaceObject.Result.STATUS_CANCELED)
    result = Done(type("Wrapped", (), {"result": canceled})(), done=False)

    class Handle:
        accepted = True

        def get_result_async(self):
            return result

        def cancel_goal_async(self):
            result.is_done = True  # the server stops and answers CANCELED

    class Client:
        def wait_for_server(self, timeout_sec):
            return True

        def send_goal_async(self, goal):
            return Done(Handle())

    runner = PlaceLlmRunner.__new__(PlaceLlmRunner)
    runner.options = RunnerOptions(place_timeout_s=0.01)
    runner.place = Client()
    runner.ui = FakeUi()
    runner.spin_until = lambda predicate, timeout: predicate()
    report = runner.send(PlaceObject.Goal())
    assert report["status"] == "CANCELED" and not report["success"]
    assert report["message"].startswith("timed out")
    assert any("canceling" in e["text"] for e in runner.ui.events)


def test_observation_without_new_surfaces_is_not_stable(summary, held):
    levels = [type("Level", (), {"number": n, "height": 0.3 * n})() for n in (3, 4)]
    runner = PlaceLlmRunner.__new__(PlaceLlmRunner)
    runner.options = RunnerOptions(space_hint=False, observe_timeout_s=0.05)
    runner.accumulator = ObjectAccumulator()
    runner.furniture_config = type("Config", (), {"frame": "world"})()
    runner.furniture = type("Furniture", (), {"levels": levels})()
    runner.ui = FakeUi()
    runner.look_point = lambda number: np.zeros(3)
    runner.aim = lambda point, frame: None
    runner.wait_head = lambda: 0
    runner.wait_fresh = lambda after, count, timeout: False  # perception stopped
    runner.surface_count = 0
    runner.spin_until = lambda predicate, timeout: predicate()
    runner.summary = lambda space: summary
    observed, timing = runner.observe(held)
    assert observed is summary
    assert not timing["stable"] and timing["stale_looks"] == 2
