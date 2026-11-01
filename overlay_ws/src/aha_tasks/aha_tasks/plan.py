"""Run a plan in fixed subgoal slots: validate it and dispatch its steps to skills.

A plan is the list ``memory["plan"]`` of ``{"command": 1-3, "action": "<skill>",
"params": {...}, "mock": {...}}``; ``mock`` is optional and holds the state to assume
while the skill is not implemented or when the step is not run. Each
``run_plan_step`` subgoal runs one step, so the engine's checkpoints, rollback and
retries apply per step.
"""

import copy

from aha_tasks.core.skill import Skill, SkillContext, SkillResult, normalize, resolve
from aha_tasks.skills import LOCATION, go_to, resolve_pose

PLAN = "plan"  # memory key
RUN_PLAN_STEP = "run_plan_step"
COMMANDS = (1, 2, 3)
REQUIRED_KEYS = {"command", "action", "params"}
# Task flow, not something a plan can ask for.
NOT_PLANNABLE = {
    RUN_PLAN_STEP,
    "listen_command",
    "plan_commands",
    "report_results",
    "wait_door_open",
}


def plan_slots(spec):
    return [subgoal for subgoal in spec.subgoals if subgoal.skill == RUN_PLAN_STEP]


def _validate_step(step, spec):
    if (
        not isinstance(step, dict)
        or set(step) - (REQUIRED_KEYS | {"mock"})
        or REQUIRED_KEYS - set(step)
    ):
        raise ValueError(f"needs {sorted(REQUIRED_KEYS)}, may have mock: {step!r}")
    if step["command"] not in COMMANDS or type(step["command"]) is not int:
        raise ValueError(f"command must be one of {COMMANDS}: {step['command']!r}")
    action, params, mock = step["action"], step["params"], step.get("mock", {})
    if not isinstance(action, str) or action not in spec.skills:
        raise ValueError(f"Unknown action: {action!r}")
    if action in NOT_PLANNABLE:
        raise ValueError(f"{action} cannot be used in a plan")
    if not isinstance(params, dict) or not isinstance(mock, dict):
        raise ValueError(f"params and mock must be objects: {step!r}")
    if "location" in params:
        location = params["location"]
        if not isinstance(location, str) or location not in spec.locations:
            raise ValueError(f"Unknown location: {location!r}")
    if mock.get(LOCATION) is not None:
        resolve_pose(spec, mock[LOCATION])
    skill = resolve(spec.skills[action])
    if skill is not None:
        skill.validate(params, spec)


def validate_plan(plan, spec):
    """Raise ValueError unless every step can be dispatched and fits the slots."""
    if not isinstance(plan, list):
        raise ValueError(f"{PLAN} must be a list of steps: {plan!r}")
    capacity = len(plan_slots(spec))
    if len(plan) > capacity:
        raise ValueError(f"{len(plan)} steps do not fit {capacity} slots")
    for number, step in enumerate(plan, 1):
        try:
            _validate_step(step, spec)
        except ValueError as error:
            raise ValueError(f"step {number}: {error}") from None


class RunPlanStep(Skill):
    """Run a step of the plan. params: {"index": <0-based position in the plan>}."""

    @classmethod
    def validate(cls, params, spec):
        if set(params) != {"index"} or type(params["index"]) is not int:
            raise ValueError(
                f"{RUN_PLAN_STEP} needs exactly an integer index: {params}"
            )
        indices = [subgoal.params.get("index") for subgoal in plan_slots(spec)]
        if indices != list(range(len(indices))):
            raise ValueError(f"Slots must run indices 0, 1, ... in order: {indices}")
        for subgoal in spec.subgoals:
            if PLAN in subgoal.mock:
                try:
                    validate_plan(subgoal.mock[PLAN], spec)
                except ValueError as error:
                    raise ValueError(f"mock of {subgoal.id}: {error}") from None

    def assumed(self, params, memory):
        plan = memory.get(PLAN)
        try:
            validate_plan(plan, self.runtime.spec)
        except ValueError:
            return {}  # execute reports it if the slot runs
        index = params["index"]
        if index >= len(plan):
            return {}
        step = plan[index]
        mock = step.get("mock", {})
        skill = self.runtime.skills.get(step["action"])
        if skill is None:
            return mock
        return {**skill.assumed(step["params"], memory), **mock}

    def execute(self, ctx):
        plan = ctx.memory.get(PLAN)
        try:
            # The whole plan, so that a bad step is rejected before any motion.
            validate_plan(plan, self.runtime.spec)
        except ValueError as error:
            return SkillResult(False, f"Invalid plan: {error}")
        index = ctx.params["index"]
        if index >= len(plan):
            return SkillResult(True, f"No step {index + 1} in the plan")
        step = plan[index]
        action, params = step["action"], step["params"]
        label = f"Step {index + 1} (command {step['command']}): {action}"
        ctx.log(f"{label} {params}")
        skill = self.runtime.skills.get(action)
        if skill is None:
            return self._assume_mock(ctx, label, step.get("mock", {}))
        result = normalize(
            skill.execute(
                SkillContext(
                    subgoal=ctx.subgoal,
                    params=copy.deepcopy(params),
                    memory=ctx.memory,
                    cancelled=ctx.cancelled,
                    log=ctx.log,
                )
            )
        )
        message = f"{label}: {result.message}" if result.message else label
        if not result.success:
            return SkillResult(False, message)
        return SkillResult(True, message, {**skill.expected(params), **result.outputs})

    def _assume_mock(self, ctx, label, mock):
        # Unlike a mocked subgoal's location, which the engine restores with Nav2,
        # the location of a succeeded subgoal counts as reached: move now.
        location = mock.get(LOCATION)
        if location is not None and self.runtime.navigation is not None:
            moved = go_to(self.runtime, location, ctx.cancelled)
            if not moved.success:
                return SkillResult(False, f"{label}: {moved.message}")
        return SkillResult(
            True, f"{label} is not implemented; mock state assumed", mock
        )
