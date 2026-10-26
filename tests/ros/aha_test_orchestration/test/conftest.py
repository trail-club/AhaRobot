"""A shelf scene shared by the tests (odom = world, shelf open towards -x)."""

import pytest

from aha_perception.furniture import Furniture
from aha_test_orchestration.scene import (
    HeldObject,
    ObjectInfo,
    SurfaceInfo,
    build_summary,
)

SHELF = Furniture(
    "shelf", "shelf", 0.7, 0.0, 0.0, 0.4, 0.6, (0.02, 0.31, 0.61, 0.91, 1.20)
)
CATEGORIES = {
    "fruit": ["apple", "orange", "lemon"],
    "food": ["tuna can", "spam"],
    "drink": ["green tea", "lemonade", "cola"],
    "snack": ["pringles"],
}


def board(surface_id, height, level, furniture="shelf", label="shelf"):
    hull = ((0.5, -0.3), (0.9, -0.3), (0.9, 0.3), (0.5, 0.3))
    return SurfaceInfo(
        surface_id, label, height, (0.7, 0.0, height), hull, furniture, level
    )


def item(object_id, label, x, y, level_height, height=0.08):
    return ObjectInfo(
        object_id, label, (x, y, level_height + height / 2), (0.07, 0.07, height)
    )


@pytest.fixture
def surfaces():
    floor = SurfaceInfo(
        1, "floor", 0.0, (0.0, 0.0, 0.0), ((-2, -2), (2, -2), (2, 2), (-2, 2))
    )
    wall = SurfaceInfo(9, "wall", 0.5, (1.0, 0.0, 0.5), ())
    return [floor, board(3, 0.61, 3), board(4, 0.91, 4), board(5, 1.20, 5), wall]


@pytest.fixture
def objects():
    return [
        item(1, "green tea", 0.6, -0.2, 0.61, 0.2),
        item(2, "lemonade", 0.6, -0.1, 0.61, 0.12),
        item(3, "orange", 0.6, -0.2, 0.91),
        item(4, "tuna can", 0.6, 0.2, 0.91, 0.045),
        item(5, "apple", 2.0, 2.0, 0.80),  # on a table elsewhere
    ]


@pytest.fixture
def summary(surfaces, objects):
    footprint = SHELF.corners(0.05)
    return build_summary(
        surfaces,
        objects,
        SHELF,
        "odom",
        (0.0, 0.0, 0.0),
        footprint,
        {2: "no", 3: "yes", 4: "yes", 5: "yes"},
    )


@pytest.fixture
def held():
    return HeldObject("cola", (0.066, 0.066, 0.12), "cylinder")
