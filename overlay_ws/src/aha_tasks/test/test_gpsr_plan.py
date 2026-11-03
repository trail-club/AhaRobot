"""GPSR plan validation and step dispatch, with fake skills (no ROS graph)."""

from itertools import groupby
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
import unittest

PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE)
from aha_tasks.core import Skill, SkillResult, TaskEngine, parse_spec  # noqa: E402
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.plan import RunPlanStep, validate_plan  # noqa: E402
from aha_tasks.runner import load_task  # noqa: E402
from aha_tasks.skills import LOCATION, go_to  # noqa: E402

GPSR = os.path.join(PACKAGE, "config", "gpsr.json")


class FakeSkill(Skill):
    """Implemented skill that records its calls. params: {"item": <str>}."""

    def __init__(self, runtime=None, result=None, outputs=None):
        super().__init__(runtime)
        self.result = result or SkillResult(True, "done", outputs or {})
        self.contexts = []

    @classmethod
    def validate(cls, params, spec):
        if not isinstance(params.get("item"), str):
            raise ValueError(f"needs an item: {params}")

    def expected(self, params):
        return {"fake.item": params["item"]}

    def execute(self, ctx):
        self.contexts.append(ctx)
        return self.result


class FakeNavigation:
    """Fails the goals in ``fail`` (0-based positions in the sequence of goals)."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.goals = []

    def navigate(self, pose, cancelled=None):
        self.goals.append(pose)
        ok = len(self.goals) - 1 not in self.fail
        return ok, "" if ok else "blocked"


def small_spec(slots=2):
    subgoals = [
        {"id": f"exec.{i}", "skill": "run_plan_step", "params": {"index": i}}
        for i in range(slots)
    ]
    return parse_spec(
        {
            "task": "small",
            "locations": {"a": [0.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]},
            "skills": {
                "navigate": "aha_tasks.skills:Navigate",
                "run_plan_step": "aha_tasks.plan:RunPlanStep",
                "fake": f"{__name__}:FakeSkill",
                "todo": None,
            },
            "subgoals": subgoals,
        }
    )


def step(action, params=None, command=1, **extra):
    return {"command": command, "action": action, "params": params or {}, **extra}


def slot(skills=None, navigation=None, spec=None):
    runtime = SimpleNamespace(
        spec=spec or small_spec(), navigation=navigation, skills=skills or {}
    )
    return RunPlanStep(runtime)


def context(plan, index=0, memory=None):
    return SkillContext(
        subgoal="exec.0",
        params={"index": index},
        memory={"plan": plan, **(memory or {})},
        cancelled=threading.Event(),
        log=lambda text: None,
    )


def gpsr_engine(navigation=None, overrides=None, restore=False, policy="abort", **kw):
    """Engine over the real task definition; skills as the runner builds them."""
    spec, classes = load_task(GPSR)
    runtime = SimpleNamespace(
        node=None, spec=spec, navigation=navigation or FakeNavigation()
    )
    skills = {name: cls and cls(runtime) for name, cls in classes.items()}
    skills.update(overrides or {})
    runtime.skills = skills
    restorers = {}
    if restore:
        restorers[LOCATION] = lambda value, cancelled: go_to(runtime, value, cancelled)
    engine = TaskEngine(spec, skills, restorers, default_policy=policy, **kw)
    engine.set_ready()
    return engine, runtime


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "Timed out"
        time.sleep(0.01)


def start(engine, selection=None):
    ok, message = engine.start(selection)
    assert ok, message


def run(engine, selection=None):
    start(engine, selection)
    wait_until(engine.finished)
    return engine.status()


def records(status):
    return {subgoal["id"]: subgoal for subgoal in status["subgoals"]}


class ValidatePlanTests(unittest.TestCase):
    def setUp(self):
        self.spec = small_spec()

    def rejects(self, plan, text):
        with self.assertRaises(ValueError) as caught:
            validate_plan(plan, self.spec)
        self.assertIn(text, str(caught.exception))

    def test_accepts_known_steps(self):
        validate_plan([], self.spec)
        validate_plan(
            [
                step("navigate", {"location": "a"}, command=3),
                step("todo", {"location": "b"}, mock={"robot.location": "b"}),
            ],
            self.spec,
        )

    def test_rejects_malformed_plans(self):
        self.rejects(None, "list")
        self.rejects({"action": "navigate"}, "list")
        self.rejects(["navigate"], "step 1")
        self.rejects([{"command": 1, "action": "todo"}], "needs")
        self.rejects([step("todo", extra=1)], "needs")
        self.rejects([step("todo"), {"command": 1, "params": {}}], "step 2")
        self.rejects([{"command": 1, "action": "todo", "params": []}], "objects")
        self.rejects([step("todo", mock=[])], "objects")

    def test_rejects_bad_command_numbers(self):
        for command in (0, 4, "1", True, None, 1.0, [1]):
            self.rejects([step("todo", command=command)], "command")

    def test_rejects_unknown_actions(self):
        self.rejects([step("teleport")], "Unknown action")
        self.rejects([step(["todo"])], "Unknown action")

    def test_rejects_actions_not_for_plans(self):
        spec = parse_spec(
            {
                "task": "flow",
                "skills": {
                    "run_plan_step": "aha_tasks.plan:RunPlanStep",
                    "listen_command": None,
                    "plan_commands": None,
                    "report_results": None,
                    "wait_door_open": None,
                },
                "subgoals": [
                    {
                        "id": "exec.0",
                        "skill": "run_plan_step",
                        "params": {"index": 0},
                    }
                ],
            }
        )
        for action in spec.skills:
            with self.assertRaisesRegex(ValueError, "cannot be used"):
                validate_plan([step(action)], spec)

    def test_rejects_unknown_locations(self):
        self.rejects([step("todo", {"location": "moon"})], "Unknown location")
        self.rejects([step("todo", {"location": ["a"]})], "Unknown location")
        self.rejects([step("todo", mock={"robot.location": "moon"})], "moon")

    def test_rejects_plans_longer_than_the_slots(self):
        self.rejects([step("todo")] * 3, "3 steps do not fit 2 slots")

    def test_runs_the_params_check_of_implemented_skills(self):
        self.rejects([step("navigate", {"location": "a", "speed": 1})], "exactly")
        self.rejects([step("fake", {})], "needs an item")
        validate_plan([step("fake", {"item": "x"})], self.spec)


class RunPlanStepValidateTests(unittest.TestCase):
    def test_params_need_an_integer_index(self):
        spec = small_spec()
        RunPlanStep.validate({"index": 1}, spec)
        for params in ({}, {"index": "1"}, {"index": True}, {"index": 1, "x": 2}):
            with self.assertRaises(ValueError):
                RunPlanStep.validate(params, spec)

    def test_slots_must_cover_the_plan_in_order(self):
        data = {
            "task": "t",
            "skills": {"run_plan_step": "aha_tasks.plan:RunPlanStep"},
            "subgoals": [
                {"id": "a", "skill": "run_plan_step", "params": {"index": 0}},
                {"id": "b", "skill": "run_plan_step", "params": {"index": 0}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "in order"):
            RunPlanStep.validate({"index": 0}, parse_spec(data))

    def test_a_bad_mock_plan_is_rejected_at_load(self):
        with open(GPSR, encoding="utf-8") as file:
            data = json.load(file)
        plan_goal = next(s for s in data["subgoals"] if s["id"] == "plan.generate")
        plan_goal["mock"]["plan"][0]["params"]["location"] = "moon"
        with self.assertRaisesRegex(ValueError, "plan.generate.*step 1.*moon"):
            RunPlanStep.validate({"index": 0}, parse_spec(data))


class RunPlanStepTests(unittest.TestCase):
    def test_empty_slot_succeeds_without_dispatch(self):
        fake = FakeSkill()
        result = slot({"fake": fake}).execute(
            context([step("fake", {"item": "x"})], index=1)
        )
        self.assertTrue(result.success)
        self.assertEqual(result.outputs, {})
        self.assertIn("No step 2", result.message)
        self.assertEqual(fake.contexts, [])
        self.assertTrue(slot().execute(context([])).success)

    def test_missing_or_invalid_plan_fails(self):
        skill = slot({"fake": FakeSkill()})
        ctx = context(None)
        del ctx.memory["plan"]
        self.assertFalse(skill.execute(ctx).success)
        result = skill.execute(context([step("teleport")]))
        self.assertFalse(result.success)
        self.assertIn("Invalid plan", result.message)

    def test_a_bad_later_step_stops_the_first_slot_before_dispatch(self):
        fake = FakeSkill()
        result = slot({"fake": fake}).execute(
            context([step("fake", {"item": "x"}), step("teleport")])
        )
        self.assertFalse(result.success)
        self.assertIn("step 2", result.message)
        self.assertEqual(fake.contexts, [])

    def test_unimplemented_step_passes_with_its_mock(self):
        plan = [step("todo", mock={"robot.holding": "cola"}, command=2), step("todo")]
        skill = slot({"todo": None})
        result = skill.execute(context(plan))
        self.assertTrue(result.success)
        self.assertIn("todo is not implemented", result.message)
        self.assertIn("command 2", result.message)
        self.assertEqual(result.outputs, {"robot.holding": "cola"})
        self.assertEqual(skill.execute(context(plan, index=1)).outputs, {})

    def test_mock_location_of_unimplemented_step_is_physically_reached(self):
        plan = [step("todo", mock={"robot.location": "b"})]
        navigation = FakeNavigation()
        result = slot(navigation=navigation).execute(context(plan))
        self.assertTrue(result.success)
        self.assertEqual(navigation.goals, [[1.0, 0.0, 0.0]])
        self.assertEqual(result.outputs, {"robot.location": "b"})

        result = slot(navigation=FakeNavigation(fail=[0])).execute(context(plan))
        self.assertFalse(result.success)
        self.assertIn("blocked", result.message)

    def test_dispatches_params_and_context(self):
        fake = FakeSkill()
        ctx = context(
            [step("fake", {"item": "x"}), step("fake", {"item": "cola"}, command=3)],
            index=1,
            memory={"other": 1},
        )
        result = slot({"fake": fake}).execute(ctx)
        self.assertTrue(result.success, result.message)
        self.assertIn("command 3", result.message)
        seen = fake.contexts[0]
        self.assertEqual(seen.params, {"item": "cola"})
        self.assertEqual(seen.subgoal, ctx.subgoal)
        self.assertEqual(seen.memory["other"], 1)
        self.assertIs(seen.cancelled, ctx.cancelled)
        self.assertIs(seen.log, ctx.log)

    def test_outputs_and_expected_are_returned_for_the_engine(self):
        fake = FakeSkill(outputs={"fake.seen": 3, "fake.item": "output wins"})
        result = slot({"fake": fake}).execute(context([step("fake", {"item": "x"})]))
        self.assertEqual(result.outputs, {"fake.seen": 3, "fake.item": "output wins"})
        fake = FakeSkill()
        result = slot({"fake": fake}).execute(context([step("fake", {"item": "x"})]))
        self.assertEqual(result.outputs, {"fake.item": "x"})

    def test_mock_is_ignored_once_the_skill_is_implemented(self):
        plan = [step("fake", {"item": "x"}, mock={"robot.holding": "cola"})]
        result = slot({"fake": FakeSkill()}).execute(context(plan))
        self.assertNotIn("robot.holding", result.outputs)

    def test_failure_propagates_with_no_outputs(self):
        fake = FakeSkill(result=SkillResult(False, "dropped", {"fake.seen": 1}))
        result = slot({"fake": fake}).execute(context([step("fake", {"item": "x"})]))
        self.assertFalse(result.success)
        self.assertIn("dropped", result.message)
        self.assertEqual(result.outputs, {})

    def test_bool_results_are_normalized(self):
        fake = FakeSkill()
        fake.result = False
        result = slot({"fake": fake}).execute(context([step("fake", {"item": "x"})]))
        self.assertFalse(result.success)


class RunPlanStepAssumedTests(unittest.TestCase):
    def assumed(self, plan, index=0, skills=None):
        return slot(skills).assumed({"index": index}, {"plan": plan})

    def test_unimplemented_step_assumes_its_mock(self):
        plan = [step("todo", mock={"robot.holding": "cola"})]
        self.assertEqual(self.assumed(plan), {"robot.holding": "cola"})

    def test_implemented_step_assumes_expected_and_mock(self):
        plan = [step("fake", {"item": "x"}, mock={"robot.holding": "cola"})]
        self.assertEqual(
            self.assumed(plan, skills={"fake": FakeSkill()}),
            {"fake.item": "x", "robot.holding": "cola"},
        )

    def test_nothing_is_assumed_without_a_step(self):
        self.assertEqual(self.assumed([step("todo")], index=1), {})
        self.assertEqual(slot().assumed({"index": 0}, {}), {})
        self.assertEqual(self.assumed([step("teleport", mock={"x": 1})]), {})


class GpsrDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.spec, _ = load_task(GPSR)
        self.mock = next(s for s in self.spec.subgoals if s.id == "plan.generate").mock
        self.plan = self.mock["plan"]

    def test_mock_plan_is_valid_and_splits_three_commands(self):
        validate_plan(self.plan, self.spec)
        self.assertEqual({s["command"] for s in self.plan}, {1, 2, 3})

    def test_mock_plan_interleaves_the_commands(self):
        runs = groupby(s["command"] for s in self.plan)
        self.assertGreater(len(list(runs)), 3)

    def test_slots_cover_twenty_steps(self):
        slots = [s for s in self.spec.subgoals if s.skill == "run_plan_step"]
        self.assertEqual(len(slots), 20)
        self.assertEqual(slots[0].id, "exec.step01")
        self.assertEqual(slots[-1].id, "exec.step20")

    def test_surfaces_are_known_locations(self):
        for item in self.plan:
            if "surface" in item["params"]:
                self.assertIn(item["params"]["surface"], self.spec.locations)

    def test_runs_without_person_perception(self):
        self.assertNotIn("find_person", self.spec.skills)
        implemented = {k for k, v in self.spec.skills.items() if v}
        self.assertEqual(implemented, {"navigate", "run_plan_step"})


class GpsrEngineTests(unittest.TestCase):
    def test_all_subgoals_pass_and_steps_are_dispatched(self):
        navigation = FakeNavigation()
        holding = []
        engine, runtime = gpsr_engine(
            navigation,
            publish=lambda status: holding.append(
                status["memory"].get("robot.holding")
            ),
        )
        status = run(engine)
        self.assertEqual(status["state"], "completed", status["message"])
        by_id = records(status)
        self.assertEqual(by_id["start.wait_door"]["status"], "unsupported")
        self.assertEqual(by_id["exec.step01"]["status"], "succeeded")
        self.assertIn("navigate", by_id["exec.step01"]["message"])
        self.assertIn("pick_object is not implemented", by_id["exec.step02"]["message"])
        self.assertIn("No step 13", by_id["exec.step13"]["message"])
        self.assertEqual(by_id["finish.report"]["status"], "unsupported")
        memory = status["memory"]
        self.assertEqual(memory["robot.location"], "instruction_point")
        self.assertIsNone(memory["robot.holding"])
        self.assertEqual(memory["cmd2.count"], 3)
        self.assertEqual(len(memory["plan"]), 12)
        # Transitions of robot.holding during the run.
        changes = [h for i, h in enumerate(holding) if i == 0 or holding[i - 1] != h]
        self.assertEqual(changes, [None, "cola", None])
        locations = runtime.spec.locations
        self.assertEqual(
            navigation.goals,
            [
                locations[name]
                for name in (
                    "instruction_point",
                    "kitchen_table",
                    "bedroom",
                    "office",
                    "desk",
                    "instruction_point",
                    "instruction_point",
                )
            ],
        )

    def test_failure_rolls_back_memory_and_location(self):
        # Goals: instruction point, kitchen table, then the failing bedroom.
        navigation = FakeNavigation(fail=[2])
        engine, runtime = gpsr_engine(navigation, restore=True, policy="pause")
        start(engine)
        wait_until(lambda: engine.state == "paused")
        status = engine.status()
        self.assertEqual(records(status)["exec.step03"]["status"], "failed")
        self.assertIn("blocked", records(status)["exec.step03"]["message"])
        memory = status["memory"]
        self.assertEqual(memory["robot.location"], "kitchen_table")
        self.assertEqual(memory["robot.holding"], "cola")
        self.assertEqual(len(memory["plan"]), 12)
        self.assertEqual(navigation.goals[-1], runtime.spec.locations["kitchen_table"])
        self.assertEqual(status["realized"], {"robot.location": "kitchen_table"})
        self.assertTrue(engine.shutdown(timeout=5.0))

    def test_rewinding_past_the_plan_takes_it_back(self):
        first = [step("navigate", {"location": "bedroom"})]
        second = [step("navigate", {"location": "office"})]
        answers = [first, second]

        class FakePlanner(Skill):
            def execute(self, ctx):
                return SkillResult(True, "", {"plan": answers.pop(0)})

        navigation = FakeNavigation(fail=[1])  # the first plan's only goal
        engine, runtime = gpsr_engine(
            navigation,
            {"plan_commands": FakePlanner(None)},
            policy="pause",
        )
        start(engine)
        wait_until(lambda: engine.state == "paused")
        status = engine.status()
        self.assertEqual(status["memory"]["plan"], first)
        self.assertEqual(status["memory"]["robot.location"], "instruction_point")
        ok, message = engine.command("rewind", "plan.generate")
        self.assertTrue(ok, message)
        wait_until(engine.finished)
        status = engine.status()
        self.assertEqual(status["state"], "completed", status["message"])
        self.assertEqual(status["memory"]["plan"], second)
        locations = runtime.spec.locations
        self.assertEqual(
            navigation.goals[-2:], [locations["office"], locations["instruction_point"]]
        )

    def test_abort_policy_ends_with_the_checkpoint(self):
        engine, _ = gpsr_engine(FakeNavigation(fail=[1]))
        status = run(engine)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["memory"]["robot.location"], "instruction_point")
        self.assertIn("plan", status["memory"])

    def test_partial_run_assumes_earlier_steps(self):
        seen = []

        class Probe(Skill):
            def execute(self, ctx):
                seen.append(ctx.memory)
                return True

        # Step 5 is answer_question; steps 1-4 are bypassed as they are not selected.
        navigation = FakeNavigation()
        engine, runtime = gpsr_engine(
            navigation, {"answer_question": Probe(None)}, restore=True
        )
        status = run(engine, "exec.step05")
        self.assertEqual(status["state"], "completed", status["message"])
        by_id = records(status)
        self.assertEqual(by_id["exec.step04"]["status"], "bypassed")
        self.assertEqual(by_id["exec.step05"]["status"], "succeeded")
        self.assertEqual(seen[0]["robot.location"], "bedroom")
        self.assertEqual(seen[0]["robot.holding"], "cola")
        self.assertEqual(len(seen[0]["plan"]), 12)
        self.assertEqual(navigation.goals, [runtime.spec.locations["bedroom"]])

    def test_skipped_navigation_assumes_its_destination(self):
        seen = []

        class Probe(Skill):
            def execute(self, ctx):
                seen.append(ctx.memory)
                return True

        # Goals: instruction point, kitchen table, bedroom, then the failing office.
        navigation = FakeNavigation(fail=[3])
        engine, runtime = gpsr_engine(
            navigation,
            {"find_described_person": Probe(None)},
            restore=True,
            policy="skip",
        )
        status = run(engine)
        by_id = records(status)
        self.assertEqual(by_id["exec.step06"]["status"], "skipped")
        self.assertEqual(by_id["exec.step07"]["status"], "succeeded")
        self.assertEqual(seen[1]["robot.location"], "office")
        locations = runtime.spec.locations
        self.assertEqual(navigation.goals[3:5], [locations["office"]] * 2)

    def test_skipped_step_assumes_its_mock(self):
        seen = []

        class Dropping(Skill):
            def execute(self, ctx):
                return SkillResult(False, "dropped")

        class Probe(Skill):
            def execute(self, ctx):
                seen.append(ctx.memory)
                return True

        engine, _ = gpsr_engine(
            overrides={"pick_object": Dropping(None), "answer_question": Probe(None)},
            policy="skip",
        )
        status = run(engine)
        self.assertEqual(records(status)["exec.step02"]["status"], "skipped")
        self.assertEqual(seen[0]["robot.holding"], "cola")

    def test_selecting_later_subgoals_keeps_the_plan_and_location(self):
        engine, _ = gpsr_engine()
        status = run(engine, "finish.return")
        self.assertEqual(status["memory"]["robot.location"], "instruction_point")
        self.assertEqual(len(status["memory"]["plan"]), 12)


if __name__ == "__main__":
    unittest.main()
