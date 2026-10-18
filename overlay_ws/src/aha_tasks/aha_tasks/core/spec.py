"""Task definitions as ordered subgoals, validated before any motion."""

from dataclasses import dataclass
import fnmatch
import json
import math
from pathlib import Path
import re

ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")
TOP_LEVEL = {
    "task",
    "title",
    "rulebook",
    "time_limit",
    "locations",
    "initial_state",
    "skills",
    "subgoals",
}
SUBGOAL_KEYS = {"id", "title", "group", "skill", "params", "mock", "points", "note"}


@dataclass(frozen=True)
class Subgoal:
    id: str
    title: str
    group: str
    skill: str
    params: dict
    mock: dict
    points: int
    note: str


@dataclass(frozen=True)
class TaskSpec:
    task: str
    title: str
    rulebook: dict
    time_limit: float | None
    locations: dict
    initial_state: dict
    skills: dict
    subgoals: tuple

    def index(self, subgoal_id):
        for i, subgoal in enumerate(self.subgoals):
            if subgoal.id == subgoal_id:
                return i
        raise ValueError(f"Unknown subgoal: {subgoal_id}")


def is_pose(value):
    """``[x, y, yaw]`` in the map frame (m, rad)."""
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(type(v) in (int, float) and math.isfinite(v) for v in value)
    )


def _object(data, key):
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def parse_spec(data):
    if not isinstance(data, dict):
        raise ValueError("Task definition must be an object")
    unknown = set(data) - TOP_LEVEL
    if unknown:
        raise ValueError(f"Unknown task keys: {sorted(unknown)}")
    task = data.get("task")
    if not isinstance(task, str) or not ID_PATTERN.fullmatch(task):
        raise ValueError("task must be an identifier")
    title = data.get("title", task)
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    time_limit = data.get("time_limit")
    if time_limit is not None and (
        type(time_limit) not in (int, float) or not 0 < time_limit < math.inf
    ):
        raise ValueError("time_limit must be a positive number of seconds")

    locations = _object(data, "locations")
    for label, pose in locations.items():
        if not label or not is_pose(pose):
            raise ValueError(f"Location {label!r} must be finite [x, y, yaw]")

    skills = data.get("skills")
    if not isinstance(skills, dict) or not skills:
        raise ValueError("skills must be a nonempty object")
    for name, target in skills.items():
        if target is not None and (
            not isinstance(target, str) or not re.fullmatch(r"[\w.]+:\w+", target)
        ):
            raise ValueError(f"Skill {name} must be null or 'module:attribute'")

    raw = data.get("subgoals")
    if not isinstance(raw, list) or not raw:
        raise ValueError("subgoals must be a nonempty array")
    subgoals, ids = [], set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Every subgoal needs a string id")
        sid = item["id"]
        if not ID_PATTERN.fullmatch(sid) or sid in ids:
            raise ValueError(f"Subgoal ids must be unique identifiers: {sid!r}")
        ids.add(sid)
        unknown = set(item) - SUBGOAL_KEYS
        if unknown:
            raise ValueError(f"Unknown keys in {sid}: {sorted(unknown)}")
        if item.get("skill") not in skills:
            raise ValueError(f"Subgoal {sid} uses an undeclared skill")
        for key in ("params", "mock"):
            if not isinstance(item.get(key, {}), dict):
                raise ValueError(f"{key} of {sid} must be an object")
        points = item.get("points", 0)
        if type(points) is not int or points < 0:
            raise ValueError(f"points of {sid} must be a nonnegative integer")
        for key in ("title", "group", "note"):
            if not isinstance(item.get(key, ""), str):
                raise ValueError(f"{key} of {sid} must be a string")
        subgoals.append(
            Subgoal(
                id=sid,
                title=item.get("title") or sid,
                group=item.get("group", ""),
                skill=item["skill"],
                params=item.get("params", {}),
                mock=item.get("mock", {}),
                points=points,
                note=item.get("note", ""),
            )
        )
    return TaskSpec(
        task=task,
        title=title,
        rulebook=_object(data, "rulebook"),
        time_limit=time_limit,
        locations=locations,
        initial_state=_object(data, "initial_state"),
        skills=skills,
        subgoals=tuple(subgoals),
    )


def load_spec(path):
    return parse_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def select(spec, expression=None):
    """Resolve ``id``, ``glob*``, ``first..last`` (either end optional) tokens.

    ``expression`` is a comma-separated string or a list of tokens; empty means all.
    Returns subgoal ids in task order.
    """
    if expression is None:
        tokens = []
    elif isinstance(expression, str):
        tokens = expression.split(",")
    else:
        tokens = list(expression)
    tokens = [token.strip() for token in tokens if token.strip()]
    ids = [subgoal.id for subgoal in spec.subgoals]
    if not tokens:
        return ids
    chosen = set()
    for token in tokens:
        if ".." in token:
            first, last = token.split("..", 1)
            start = spec.index(first) if first else 0
            end = spec.index(last) if last else len(ids) - 1
            if start > end:
                raise ValueError(f"Empty subgoal range: {token}")
            matched = ids[start : end + 1]
        elif any(c in token for c in "*?["):
            matched = fnmatch.filter(ids, token)
        else:
            matched = [ids[spec.index(token)]]
        if not matched:
            raise ValueError(f"No subgoal matches {token!r}")
        chosen.update(matched)
    return [sid for sid in ids if sid in chosen]
