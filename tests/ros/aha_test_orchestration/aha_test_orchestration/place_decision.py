"""LLM inputs and answers for "where to put the held object" (ROS free).

Three splits of the work between the LLM and the robot, all answered through
a forced call of the function ``choose_place``:

  A (full context): every tracked horizontal surface (id, label, furniture
    level, height, centroid, extent) and every detected object (id, label,
    center, size) as raw numbers. Answer: surface id and the id of an object
    to put it next to or a target point. Robot: PlaceObject on that surface
    near the object / point.
  B (height only): the level heights of the furniture config and the labels
    of the objects seen at each height. Answer: a height. Robot: PlaceObject
    restricted to that height, no near point.
  B2: B's input and answer; the robot sends the furniture level of that
    height instead (no near point).
  C (id summary): furniture levels with the ids and labels of the objects on
    each and whether perception found a free spot for the held object.
    Answer: furniture, level and the object to put it next to (or "empty").
    Robot: PlaceObject on that furniture level near that object.

``Variant.validate`` turns the arguments into a ``Decision`` (with the
``PlaceRequest`` for PlaceObject) or raises InvalidAnswer with a message for
the retry.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from aha_test_orchestration.llm import InvalidAnswer
from aha_test_orchestration.scene import HeldObject, SceneSummary

TOOL_NAME = "choose_place"
HEIGHT_TOLERANCE = 0.03  # m, answered height to a level height (B)
POINT_MARGIN = 0.10  # m, answered point outside the surface extent (A)
SYSTEM_PROMPT = """You decide where a home service robot puts the object it is holding.
Task (RoboCup@Home Pick and Place): store the object in the cabinet.
Rules:
- Put it next to objects of the same category on the cabinet.
- If no object of the same category is on the cabinet, start a new group on a shelf level that has no objects.
Categories of the known objects:
{categories}
Answer by calling the function {tool} once."""


@dataclass(frozen=True)
class PlaceRequest:
    """PlaceObject goal fields chosen from the answer (others from the held object)."""

    surface_ids: tuple[int, ...] = ()
    near: tuple[float, float, float] | None = None
    near_frame: str = ""
    restrict_height: bool = False
    surface_height: float = 0.0
    furniture: str = ""
    furniture_level: int = 0


@dataclass(frozen=True)
class Decision:
    variant: str
    level: int  # furniture level of the answer, 0 if not on a level
    request: PlaceRequest
    surface_id: int = 0
    next_to: int | None = None  # object id
    point: tuple[float, float] | None = None
    height: float | None = None


def describe(decision: Decision) -> str:
    """The chosen place in the words of the variant's prompt."""
    if decision.variant == "A":
        where = f"surface {decision.surface_id}"
        if decision.next_to is not None:
            return f"{where} next to object {decision.next_to}"
        x, y = decision.point
        return f"{where} at ({x:.2f}, {y:.2f})"
    if decision.variant in ("B", "B2"):
        return f"the {decision.height:.2f} m level"
    where = f"level {decision.level}"
    if decision.next_to is not None:
        return f"{where} next to O{decision.next_to}"
    return f"{where} (empty part)"


def choice_key(decision: Decision) -> tuple:
    """Equal for answers that choose the same place."""
    return (decision.level, decision.surface_id, decision.next_to, decision.point)


def failure_feedback(decision: Decision, status: str, message: str) -> str:
    """Tool result after a failed PlaceObject goal, asking for another place."""
    return (
        f"The robot could not put the object on {describe(decision)}: {status} "
        f"({message}). The object is still in the hand. Choose another place "
        f"by calling {TOOL_NAME} again; do not choose a failed place again."
    )


def size_text(size) -> str:
    return " x ".join(f"{v:.3f}" for v in size) + " m"


def system_prompt(categories: dict) -> str:
    lines = "\n".join(f"- {c}: {', '.join(labels)}" for c, labels in categories.items())
    return SYSTEM_PROMPT.format(categories=lines, tool=TOOL_NAME)


def _tool(description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "One short sentence."},
                    **properties,
                },
                "required": ["reason", *required],
            },
        },
    }


def _int(arguments: dict, key: str) -> int:
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InvalidAnswer(f"{key} must be an integer")
    try:
        number = float(value)
    except ValueError as e:
        raise InvalidAnswer(f"{key} must be an integer, got {value!r}") from e
    if not number.is_integer():
        raise InvalidAnswer(f"{key} must be an integer, got {value!r}")
    return int(number)


def _float(arguments: dict, key: str) -> float:
    value = arguments.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise InvalidAnswer(f"{key} must be a number, got {value!r}") from e
    if isinstance(value, bool) or not math.isfinite(number):
        raise InvalidAnswer(f"{key} must be a number, got {value!r}")
    return number


class Variant:
    name = ""
    tool: dict = {}

    def user_prompt(self, summary: SceneSummary, held: HeldObject) -> str:
        raise NotImplementedError

    def validate(self, arguments: dict, summary: SceneSummary) -> Decision:
        raise NotImplementedError

    def messages(self, summary: SceneSummary, held: HeldObject, categories) -> list:
        return [
            {"role": "system", "content": system_prompt(categories)},
            {"role": "user", "content": self.user_prompt(summary, held)},
        ]


class FullContext(Variant):
    """A: raw surfaces and objects; answer surface id and object id / point."""

    name = "A"
    tool = _tool(
        "Where to put the held object.",
        {
            "surface_id": {
                "type": "integer",
                "description": "Id of the surface to put the object on.",
            },
            "near_object_id": {
                "type": "integer",
                "description": "Id of an object on that surface to put it next to, -1 for none.",
            },
            "x": {
                "type": "number",
                "description": "Target x in the frame when near_object_id is -1.",
            },
            "y": {
                "type": "number",
                "description": "Target y in the frame when near_object_id is -1.",
            },
        },
        ["surface_id", "near_object_id"],
    )

    def user_prompt(self, summary, held):
        x, y, yaw = summary.robot_pose
        lines = [
            f"Held object: {held.label} ({size_text(held.size)}).",
            f"Frame: {summary.frame} (m, z up from the floor). Robot base: x {x:.2f}, "
            f"y {y:.2f}, yaw {math.degrees(yaw):.0f} deg (x is forward at yaw 0).",
            "Tracked horizontal surfaces:",
        ]
        for s in summary.surfaces:
            x0, x1, y0, y1 = s.extent()
            where = f", furniture {s.furniture} level {s.level}" if s.furniture else ""
            seen = "" if s.visible else ", not in view now"
            lines.append(
                f"- surface {s.id}: label {s.label}{where}, height {s.height:.3f}, "
                f"centroid ({s.centroid[0]:.2f}, {s.centroid[1]:.2f}), "
                f"x [{x0:.2f}, {x1:.2f}], y [{y0:.2f}, {y1:.2f}]{seen}"
            )
        lines.append("Detected objects (center x, y, z; size):")
        for o in summary.objects:
            px, py, pz = o.position
            lines.append(
                f"- object {o.id}: {o.label} at ({px:.2f}, {py:.2f}, {pz:.3f}), "
                f"size {size_text(o.size)}"
            )
        if not summary.objects:
            lines.append("- none")
        lines.append(
            "Choose the surface and the spot for the held object: the id of an object "
            "to put it next to, or -1 and a target point x, y."
        )
        return "\n".join(lines)

    def validate(self, arguments, summary):
        surface_id = _int(arguments, "surface_id")
        surface = summary.surface(surface_id)
        if surface is None:
            ids = ", ".join(str(s.id) for s in summary.surfaces)
            raise InvalidAnswer(f"unknown surface_id {surface_id}; valid ids: {ids}")
        near_id = _int(arguments, "near_object_id")
        point = None
        if near_id >= 0:
            obj = summary.object(near_id)
            if obj is None:
                ids = ", ".join(str(o.id) for o in summary.objects) or "none"
                raise InvalidAnswer(
                    f"unknown near_object_id {near_id}; valid ids: {ids} or -1"
                )
            on = summary.support.get(near_id, 0)
            if on not in (0, surface_id):
                raise InvalidAnswer(
                    f"object {near_id} is on surface {on}, not on surface {surface_id}"
                )
            near = (obj.position[0], obj.position[1], surface.height)
            next_to = near_id
        else:
            if arguments.get("x") is None or arguments.get("y") is None:
                raise InvalidAnswer("give x and y when near_object_id is -1")
            point = (_float(arguments, "x"), _float(arguments, "y"))
            x0, x1, y0, y1 = surface.extent()
            m = POINT_MARGIN
            if not (x0 - m <= point[0] <= x1 + m and y0 - m <= point[1] <= y1 + m):
                raise InvalidAnswer(
                    f"point ({point[0]:.2f}, {point[1]:.2f}) is outside surface "
                    f"{surface_id} (x [{x0:.2f}, {x1:.2f}], y [{y0:.2f}, {y1:.2f}])"
                )
            near = (point[0], point[1], surface.height)
            next_to = None
        return Decision(
            self.name,
            summary.level_of_surface(surface_id),
            PlaceRequest(
                surface_ids=(surface_id,), near=near, near_frame=summary.frame
            ),
            surface_id=surface_id,
            next_to=next_to,
            point=point,
        )


class HeightOnly(Variant):
    """B: level heights and the labels seen at each; answer a height."""

    name = "B"
    tool = _tool(
        "Height of the level to put the held object on.",
        {
            "height": {
                "type": "number",
                "description": "Height in m of one of the listed levels.",
            }
        },
        ["height"],
    )

    def user_prompt(self, summary, held):
        lines = [
            f"Held object: {held.label}.",
            f'Levels of the cabinet "{summary.furniture}" (height of the level top '
            "above the floor) and the objects seen at each height:",
        ]
        for level in summary.levels:
            labels = [
                summary.object(i).label for i in summary.by_height.get(level.number, ())
            ]
            lines.append(
                f"- {level.height:.2f} m: {', '.join(labels) if labels else 'no objects'}"
            )
        lines.append("Choose the height of the level to put the held object on.")
        return "\n".join(lines)

    def validate(self, arguments, summary):
        height = _float(arguments, "height")
        level = min(summary.levels, key=lambda v: abs(v.height - height), default=None)
        if level is None or abs(level.height - height) > HEIGHT_TOLERANCE:
            heights = ", ".join(f"{v.height:.2f}" for v in summary.levels)
            raise InvalidAnswer(
                f"height {height:.2f} is not a listed level; choose one of {heights}"
            )
        return Decision(
            self.name,
            level.number,
            self.request(summary, level),
            height=level.height,
        )

    def request(self, summary, level) -> PlaceRequest:
        return PlaceRequest(restrict_height=True, surface_height=level.height)


class HeightToLevel(HeightOnly):
    """B2: B's input and answer; the robot sends the furniture level of the height."""

    name = "B2"

    def request(self, summary, level) -> PlaceRequest:
        return PlaceRequest(furniture=summary.furniture, furniture_level=level.number)


class IdSummary(Variant):
    """C: levels with object ids / labels and free space; answer level and object."""

    name = "C"
    tool = _tool(
        "Furniture level and the object to put the held object next to.",
        {
            "furniture": {"type": "string", "description": "Furniture name."},
            "level": {"type": "integer", "description": "Level number."},
            "next_to": {
                "type": "string",
                "description": 'Object id on that level such as "O3", or "empty".',
            },
        },
        ["furniture", "level", "next_to"],
    )

    def user_prompt(self, summary, held):
        spaces = any(level.space != "unknown" for level in summary.levels)
        header = f'Cabinet "{summary.furniture}", levels from the bottom'
        if spaces:
            header += (
                ' ("space": whether the robot found a free spot for the held '
                "object there)"
            )
        lines = [f"Held object: {held.label} ({size_text(held.size)}).", header + ":"]
        for level in summary.levels:
            objects = [summary.object(i) for i in level.object_ids]
            content = ", ".join(f"O{o.id} {o.label}" for o in objects) or "no objects"
            space = f"; space: {level.space}" if spaces else ""
            lines.append(
                f"- level {level.number} ({level.height:.2f} m): {content}{space}"
            )
        lines.append(
            "Choose the level and the object to put the held object next to "
            '("empty" if no object of the same category is on the cabinet).'
        )
        return "\n".join(lines)

    def tool_for(self, summary):
        tool = {**self.tool}
        tool["function"] = {**tool["function"]}
        parameters = {**tool["function"]["parameters"]}
        properties = {**parameters["properties"]}
        properties["furniture"] = {
            **properties["furniture"],
            "enum": [summary.furniture],
        }
        parameters["properties"] = properties
        tool["function"]["parameters"] = parameters
        return tool

    def validate(self, arguments, summary):
        furniture = str(arguments.get("furniture", "")).strip()
        if furniture != summary.furniture:
            raise InvalidAnswer(
                f"unknown furniture {furniture!r}; use {summary.furniture!r}"
            )
        number = _int(arguments, "level")
        level = summary.level(number)
        if level is None:
            numbers = ", ".join(str(v.number) for v in summary.levels)
            raise InvalidAnswer(f"unknown level {number}; valid levels: {numbers}")
        next_to = str(arguments.get("next_to", "")).strip()
        near, object_id = None, None
        if next_to.lower() != "empty":
            match = re.fullmatch(r"[Oo]?\s*(\d+)", next_to)
            obj = summary.object(int(match.group(1))) if match else None
            if obj is None:
                raise InvalidAnswer(
                    f'unknown next_to {next_to!r}; give an object id like "O3" or "empty"'
                )
            on = summary.level_of_object(obj.id)
            if on != number:
                where = f"level {on}" if on else "no level"
                raise InvalidAnswer(f"O{obj.id} is on {where}, not on level {number}")
            near, object_id = tuple(obj.position), obj.id
        return Decision(
            self.name,
            number,
            PlaceRequest(
                near=near,
                near_frame=summary.frame if near else "",
                furniture=furniture,
                furniture_level=number,
            ),
            next_to=object_id,
        )


VARIANTS = {
    v.name: v for v in (FullContext(), HeightOnly(), HeightToLevel(), IdSummary())
}


def tool_for(variant: Variant, summary: SceneSummary) -> dict:
    if isinstance(variant, IdSummary):
        return variant.tool_for(summary)
    return variant.tool
