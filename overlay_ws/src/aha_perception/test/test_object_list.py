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
    for entry in objects.objects:
        # Every listed object fits between the fingers (0.11 m).
        assert min(entry.size[:2]) <= 0.11


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
