"""Object list: labels, categories, sizes and simulation model names.

The YAML (config/objects/robocup_objects.yaml) has one entry per label::

  objects:
    - {label: tuna can, category: food, shape: cylinder, size: [0.085, 0.085, 0.045]}

``models`` (optional) lists fnmatch patterns of Gazebo model names detected
as the label; the default is the label with "_" for spaces and the same
followed by "_*". ROS free (PyYAML).
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

SHAPES = ("box", "cylinder")
_KEYS = {"label", "category", "shape", "size", "models"}


class ObjectListError(ValueError):
    """Invalid object list."""


@dataclass(frozen=True)
class ObjectType:
    label: str
    category: str
    shape: str
    size: tuple[float, float, float]  # x, y footprint and z height (m)
    models: tuple[str, ...]  # fnmatch patterns of Gazebo model names

    def matches(self, model: str) -> bool:
        return any(fnmatchcase(model, pattern) for pattern in self.models)


@dataclass(frozen=True)
class ObjectList:
    objects: tuple[ObjectType, ...]

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
    return ObjectList(tuple(entries))


def load_object_list(path) -> ObjectList:
    path = Path(path)
    with path.open() as stream:
        return parse_object_list(yaml.safe_load(stream), str(path))
