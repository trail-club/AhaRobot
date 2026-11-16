from pathlib import Path

import pytest

from aha_perception.object_list import (
    ObjectListError,
    load_object_list,
    parse_object_list,
)

CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "objects" / "robocup_objects.yaml"
)


def test_package_list_loads():
    objects = load_object_list(CONFIG)
    assert objects.category("apple") == "fruit"
    assert objects.category("unknown") == ""
    assert "drink" in objects.categories()
    for label in ("fork", "knife", "spoon", "mug", "plate", "bowl", "cereal", "milk"):
        assert objects.get(label) is not None, label
    for entry in objects.objects:
        if entry.label in ("plate", "bowl"):
            continue  # grasped by the rim
        # Every other listed object fits between the fingers (0.11 m).
        assert min(entry.size[:2]) <= 0.11, entry.label


def test_package_labels_not_shadowed():
    # A label's own model names are not taken by another label's patterns
    # (e.g. strawberry_* would take strawberry_jello).
    objects = load_object_list(CONFIG)
    for entry in objects.objects:
        base = entry.label.replace(" ", "_")
        for name in (base, f"{base}_2"):
            assert objects.for_model(name).label == entry.label, name


def test_model_patterns():
    objects = parse_object_list(
        {
            "objects": [
                {
                    "label": "tuna can",
                    "category": "food",
                    "shape": "cylinder",
                    "size": [0.08, 0.08, 0.04],
                },
                {
                    "label": "cola",
                    "category": "drink",
                    "shape": "cylinder",
                    "size": [0.066, 0.066, 0.12],
                    "models": ["coke*"],
                },
            ]
        }
    )
    assert objects.for_model("tuna_can").label == "tuna can"
    assert objects.for_model("tuna_can_2").label == "tuna can"
    assert objects.for_model("tuna_cans") is None
    assert objects.for_model("coke_zero").label == "cola"
    assert objects.for_model("cola") is None
    assert objects.for_model("aha_held_object_right") is None


@pytest.mark.parametrize(
    "item",
    [
        {"label": "x", "category": "c", "shape": "sphere", "size": [0.1, 0.1, 0.1]},
        {"label": "x", "category": "c", "shape": "box", "size": [0.1, 0.1]},
        {"label": "x", "category": "c", "shape": "box", "size": [0.1, 0.1, 0.0]},
        {
            "label": "x",
            "category": "c",
            "shape": "box",
            "size": [0.1, 0.1, 0.1],
            "color": 1,
        },
        {"category": "c", "shape": "box", "size": [0.1, 0.1, 0.1]},
    ],
)
def test_invalid_entries(item):
    with pytest.raises(ObjectListError):
        parse_object_list({"objects": [item]})


def test_duplicate_labels():
    item = {"label": "x", "category": "c", "shape": "box", "size": [0.1, 0.1, 0.1]}
    with pytest.raises(ObjectListError):
        parse_object_list({"objects": [item, item]})


def test_package_weights_aliases_and_groups():
    objects = load_object_list(CONFIG)
    # Every label has a weight for heaviest / lightest.
    assert all(o.weight for o in objects.objects)
    assert objects.get("green tea").weight > objects.get("red bull").weight
    assert "tuna" in objects.get("tuna can").aliases
    groups = dict(objects.groups)
    assert "cutlery" in groups["kitchen item"]


ITEM = {"label": "x", "category": "c", "shape": "box", "size": [0.1, 0.1, 0.1]}


def test_weights_aliases_and_groups():
    objects = parse_object_list(
        {
            "objects": [ITEM, dict(ITEM, label="y")],
            "weights": {"x": 0.25},
            "aliases": {"y": ["why", "wye"]},
            "groups": {"things": ["c"], "pair": ["x", "y"]},
        }
    )
    assert objects.get("x").weight == 0.25
    assert objects.get("y").weight is None
    assert objects.get("y").aliases == ("why", "wye")
    assert dict(objects.groups) == {"things": ("c",), "pair": ("x", "y")}


@pytest.mark.parametrize(
    "extra",
    [
        {"weights": {"z": 0.1}},
        {"weights": {"x": 0}},
        {"weights": {"x": True}},
        {"weights": [0.1]},
        {"aliases": {"x": "why"}},
        {"aliases": {"x": [""]}},
        {"aliases": {"z": ["zed"]}},
        {"groups": {"things": ["unknown"]}},
        {"groups": {"things": []}},
        {"weight": {"x": 0.1}},
    ],
)
def test_invalid_weights_aliases_and_groups(extra):
    with pytest.raises(ObjectListError):
        parse_object_list({"objects": [ITEM], **extra})
