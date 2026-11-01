"""Built-in skills and the robot.location restorer, using labelled map locations."""

from aha_tasks.core.skill import Skill, SkillResult
from aha_tasks.core.spec import is_pose

LOCATION = "robot.location"  # memory key: location label or [x, y, yaw] in map


def resolve_pose(spec, value):
    if isinstance(value, str):
        if value not in spec.locations:
            raise ValueError(f"Unknown location: {value}")
        return spec.locations[value]
    if is_pose(value):
        return value
    raise ValueError(f"{LOCATION} must be a location label or [x, y, yaw]: {value!r}")


def check_locations(spec):
    """Every assumed robot.location must be reachable before any motion."""
    states = [("initial_state", spec.initial_state)]
    states += [(subgoal.id, subgoal.mock) for subgoal in spec.subgoals]
    for source, state in states:
        if state.get(LOCATION) is not None:
            try:
                resolve_pose(spec, state[LOCATION])
            except ValueError as error:
                raise ValueError(f"{source}: {error}") from None


def go_to(runtime, value, cancelled):
    """Restorer for LOCATION and the body of the navigate skill."""
    ok, message = runtime.navigation.navigate(
        resolve_pose(runtime.spec, value), cancelled
    )
    return SkillResult(ok, message or f"Arrived at {value}")


class Navigate(Skill):
    """Move the base with Nav2. params: {"location": "<label>"}."""

    @classmethod
    def validate(cls, params, spec):
        location = params.get("location")
        if set(params) != {"location"} or not isinstance(location, str):
            raise ValueError(f"navigate needs exactly a location label: {params}")
        if location not in spec.locations:
            raise ValueError(f"Unknown navigation label: {location}")

    def expected(self, params):
        return {LOCATION: params["location"]}

    def execute(self, ctx):
        return go_to(self.runtime, ctx.params["location"], ctx.cancelled)
