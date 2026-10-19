"""Shelf scene and held-object cases of run_trials.py (ROS free).

World rcjo2026 (model://shelf at world (3.2, 3.5), open side towards world
-y; boards from x 2.90 to 3.50 and y 3.30 to 3.70). The robot stands at the
spawn of place_sim's shelf_060_back (2.95, 2.80, facing +y), where levels 3
(0.61 m), 4 (0.91 m) and 5 (1.20 m) are reachable with the right arm; on
level 2 (0.31 m) only a strip at the front is seen (no place candidates).
Labels and sizes come from aha_perception's object list.
"""

from __future__ import annotations

from dataclasses import dataclass

SPAWN = (2.95, 2.80, 1.5708)
FURNITURE = "shelf"
LEVEL_HEIGHTS = {2: 0.31, 3: 0.61, 4: 0.91, 5: 1.20}
BOARDS = ((2.90, 3.50), (3.30, 3.70))  # world x and y range of the boards
EMPTY_LEVEL = 5
NEXT_TO = 0.15  # m, center distance to a same-category object counted as "next to"


@dataclass(frozen=True)
class Prop:
    model: str  # Gazebo model name (object list pattern <label>_<n>)
    label: str
    x: float  # world
    y: float
    level: int


# One category per level, level 5 empty. From the spawn the right arm places
# only in a band of the boards: about 0.15 m deep (robot x < 0.66) and robot y
# -0.2 to -0.45 (world x 3.15 to 3.40). The drink and fruit groups stand at the
# right of that band; fruit 0.18 m behind the front edge of level 4 so that the
# strip in front stays free (on level 3 the next board hides deeper props).
# The food group at the left end of level 4 is out of the band: it is only a
# second group on that level (no held food case).
PROPS = (
    Prop("cleanser_1", "cleanser", 3.40, 3.36, 2),
    Prop("sponge_1", "sponge", 3.30, 3.36, 2),
    Prop("green_tea_1", "green tea", 3.41, 3.40, 3),
    Prop("lemonade_1", "lemonade", 3.34, 3.40, 3),
    Prop("orange_1", "orange", 3.40, 3.48, 4),
    Prop("lemon_1", "lemon", 3.33, 3.48, 4),
    Prop("tuna_can_1", "tuna can", 2.965, 3.48, 4),
    Prop("spam_1", "spam", 3.055, 3.48, 4),
)


@dataclass(frozen=True)
class Ycb:
    model: str  # tmc_wrs_gz_worlds model directory
    shape: str  # box | cylinder: the grasp width and inertia
    size: tuple[float, float, float]  # collision mesh bounding box (m)
    yaw: float = 0.0  # rad, as a prop (held objects are not turned)


# --ycb: YCB models for the labels (visual and collision meshes, mass; drinks
# by cans and bottles of similar shape; the snacks stay boxes and
# cylinders). Sizes measured from the collision meshes; the robot is given
# these as the held object's size. The props stand in rows along world x;
# spam and the mustard bottle are turned to put their narrow side (y)
# towards the neighbour, else they overlap it (gaps then: tuna can - spam
# 1.9 cm, mustard - soup can 0.4 cm, orange - lemon 0.3 cm, cleanser -
# sponge 1.1 cm).
YCB = {
    "apple": Ycb("ycb_013_apple", "cylinder", (0.075, 0.075, 0.072)),
    "orange": Ycb("ycb_017_orange", "cylinder", (0.072, 0.074, 0.071)),
    "lemon": Ycb("ycb_014_lemon", "cylinder", (0.060, 0.059, 0.053)),
    "peach": Ycb("ycb_015_peach", "cylinder", (0.062, 0.062, 0.058)),
    "tuna can": Ycb("ycb_007_tuna_fish_can", "cylinder", (0.085, 0.085, 0.033)),
    "spam": Ycb("ycb_005_spam", "box", (0.098, 0.057, 0.081), 1.5708),
    "green tea": Ycb("ycb_006_mustard_bottle", "box", (0.097, 0.066, 0.192), 1.5708),
    "lemonade": Ycb("ycb_010_tomato_soup_can", "cylinder", (0.067, 0.067, 0.101)),
    "cola": Ycb("ycb_010_tomato_soup_can", "cylinder", (0.067, 0.067, 0.101)),
    "milk": Ycb("ycb_002_sugar_box", "box", (0.049, 0.094, 0.176)),
    "cleanser": Ycb("ycb_021_bleach_cleanser", "box", (0.102, 0.067, 0.251)),
    "sponge": Ycb("ycb_026_sponge", "box", (0.077, 0.111, 0.018)),
}


@dataclass(frozen=True)
class Case:
    name: str
    label: str  # held object
    category: str
    level: int  # correct level
    next_to_category: str = ""  # same-category group on that level, "" for a new group


CASES = (
    Case("apple", "apple", "fruit", 4, "fruit"),
    Case("peach", "peach", "fruit", 4, "fruit"),
    Case("cola", "cola", "drink", 3, "drink"),
    Case("milk", "milk", "drink", 3, "drink"),
    # Taller than about 0.12 m: the hand cannot reach the place pose on level 5.
    Case("pringles", "pringles", "snack", EMPTY_LEVEL),
    Case("cookies", "cookies", "snack", EMPTY_LEVEL),
    Case("chocolate_jello", "chocolate jello", "snack", EMPTY_LEVEL),
    Case("strawberry_jello", "strawberry jello", "snack", EMPTY_LEVEL),
)


def on_boards(x: float, y: float, margin: float = 0.02) -> bool:
    """Whether the world point is above the boards (within margin)."""
    (x0, x1), (y0, y1) = BOARDS
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin


def level_of_height(z: float, tolerance: float = 0.03) -> int:
    """Shelf level whose top is within tolerance of an object bottom, else 0."""
    best = min(LEVEL_HEIGHTS, key=lambda n: abs(LEVEL_HEIGHTS[n] - z))
    return best if abs(LEVEL_HEIGHTS[best] - z) <= tolerance else 0
