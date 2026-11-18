"""Object list: labels, categories, sizes and simulation model names.

The YAML (config/objects/robocup_objects.yaml) has one entry per label::

  objects:
    - {label: tuna can, category: food, shape: cylinder, size: [0.085, 0.085, 0.045]}

``models`` (optional) lists fnmatch patterns of Gazebo model names detected
as the label; the default is the label with "_" for spaces and the same
followed by "_*".

Optional mappings next to ``objects`` describe the labels for tasks that
answer about objects (aha_tasks GPSR)::

  weights: {tuna can: 0.171}            # kg
  aliases: {tuna can: [tuna fish can]}  # other names of a label in commands
  groups: {kitchen item: [cutlery, tableware]}  # names of sets of categories / labels

ROS free (PyYAML).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

SHAPES = ("box", "cylinder")
_KEYS = {"label", "category", "shape", "size", "models"}
_TOP_KEYS = {"objects", "weights", "aliases", "groups"}


class ObjectListError(ValueError):
    """Invalid object list."""


@dataclass(frozen=True)
class ObjectType:
    label: str
    category: str
    shape: str
    size: tuple[float, float, float]  # x, y footprint and z height (m)
    models: tuple[str, ...]  # fnmatch patterns of Gazebo model names
    weight: float | None = None  # kg, None if unknown
    aliases: tuple[str, ...] = ()  # other names in commands

    def matches(self, model: str) -> bool:
        return any(fnmatchcase(model, pattern) for pattern in self.models)


@dataclass(frozen=True)
class ObjectList:
    objects: tuple[ObjectType, ...]
    # Name of a set -> the categories and labels in it (e.g. "kitchen item").
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.objects]

    def get(self, label: str) -> ObjectType | None:
        return next((o for o in self.objects if o.label == label), None)

    def category(self, label: str) -> str:
        """Category of a label, empty if unknown."""
        entry = self.get(label)
        return entry.category if entry else ""

    def categories(self) -> dict[str, list[str]]:
        """Labels by category, in the order of the list."""
        out: dict[str, list[str]] = {}
        for o in self.objects:
            out.setdefault(o.category, []).append(o.label)
        return out

    def for_model(self, model: str) -> ObjectType | None:
        """First entry whose model patterns match a Gazebo model name."""
        return next((o for o in self.objects if o.matches(model)), None)


def default_models(label: str) -> tuple[str, str]:
    base = label.replace(" ", "_")
    return base, f"{base}_*"


def parse_object_list(data, source: str = "object list") -> ObjectList:
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise ObjectListError(f"{source}: expected a mapping with an 'objects' list")
    entries = []
    for k, item in enumerate(data["objects"]):
        what = f"{source}: objects[{k}]"
        if not isinstance(item, dict):
            raise ObjectListError(f"{what}: expected a mapping")
        unknown = set(item) - _KEYS
        missing = {"label", "category", "shape", "size"} - set(item)
        if unknown or missing:
            raise ObjectListError(
                f"{what}: unknown keys {sorted(unknown)}, missing {sorted(missing)}"
            )
        label = str(item["label"]).strip()
        if not label:
            raise ObjectListError(f"{what}: empty label")
        shape = str(item["shape"])
        if shape not in SHAPES:
            raise ObjectListError(f"{what}: shape must be one of {SHAPES}")
        size = item["size"]
        if (
            not isinstance(size, list)
            or len(size) != 3
            or not all(isinstance(v, (int, float)) and v > 0 for v in size)
        ):
            raise ObjectListError(f"{what}: size must be three positive numbers")
        models = item.get("models") or default_models(label)
        entries.append(
            ObjectType(
                label,
                str(item["category"]),
                shape,
                tuple(float(v) for v in size),
                tuple(str(m) for m in models),
            )
        )
    labels = [e.label for e in entries]
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        raise ObjectListError(f"{source}: duplicate labels {duplicates}")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ObjectListError(f"{source}: unknown keys {sorted(unknown)}")
    weights = _label_mapping(data, "weights", labels, source)
    aliases = _label_mapping(data, "aliases", labels, source)
    for label, weight in weights.items():
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not 0 < weight < float("inf")
        ):
            raise ObjectListError(f"{source}: weights.{label} must be positive (kg)")
    for label, names in aliases.items():
        if not _names(names):
            raise ObjectListError(f"{source}: aliases.{label} must be a list of names")
    entries = [
        replace(
            e,
            weight=float(weights[e.label]) if e.label in weights else None,
            aliases=tuple(n.strip() for n in aliases.get(e.label, ())),
        )
        for e in entries
    ]
    groups = data.get("groups") or {}
    if not isinstance(groups, dict):
        raise ObjectListError(f"{source}: groups must be a mapping")
    known = set(labels) | {e.category for e in entries}
    for name, members in groups.items():
        if not str(name).strip() or not _names(members):
            raise ObjectListError(f"{source}: groups.{name} must be a list of names")
        missing = [m for m in members if m not in known]
        if missing:
            raise ObjectListError(
                f"{source}: groups.{name}: {missing} are not categories or labels"
            )
    return ObjectList(
        tuple(entries),
        tuple((str(n).strip(), tuple(m)) for n, m in groups.items()),
    )


def _label_mapping(data, key, labels, source):
    mapping = data.get(key) or {}
    if not isinstance(mapping, dict):
        raise ObjectListError(f"{source}: {key} must be a mapping of labels")
    unknown = sorted(set(mapping) - set(labels))
    if unknown:
        raise ObjectListError(f"{source}: {key} of unknown labels {unknown}")
    return mapping


def _names(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, str) and v.strip() for v in value)
    )


def load_object_list(path) -> ObjectList:
    path = Path(path)
    with path.open() as stream:
        return parse_object_list(yaml.safe_load(stream), str(path))
