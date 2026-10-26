import math

import numpy as np
from conftest import CATEGORIES, SHELF, board, item

from aha_test_orchestration.scene import (
    build_summary,
    next_object_id,
    other_groups,
    polygon_distance,
    split_groups,
    support_of,
    with_placed,
    without_object,
)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_polygon_distance():
    assert polygon_distance(SQUARE, (0.5, 0.5)) == 0.0
    assert math.isclose(polygon_distance(SQUARE, (1.5, 0.5)), 0.5)
    assert math.isclose(polygon_distance(SQUARE[::-1], (0.5, -0.2)), 0.2)
    assert polygon_distance(np.zeros((0, 2)), (0, 0)) == math.inf


def test_support_by_height_and_hull(surfaces):
    assert support_of(item(1, "x", 0.6, 0.0, 0.61), surfaces) == 3
    assert support_of(item(1, "x", 0.6, 0.0, 0.91), surfaces) == 4
    # Just outside the hull (margin 0.03) and far outside.
    assert support_of(item(1, "x", 0.92, 0.0, 0.91), surfaces) == 4
    assert support_of(item(1, "x", 1.2, 0.0, 0.91), surfaces) == 0
    # Floating between the boards.
    assert support_of(item(1, "x", 0.6, 0.0, 0.75), surfaces) == 0
    # On the floor.
    assert support_of(item(1, "x", 0.0, 0.0, 0.0), surfaces) == 1


def test_levels_and_objects(summary):
    assert [v.number for v in summary.levels] == [2, 3, 4, 5]
    level = {v.number: v for v in summary.levels}
    assert level[2].surface_ids == () and level[2].object_ids == ()
    assert level[3].object_ids == (1, 2)
    assert level[4].object_ids == (3, 4)
    assert level[5].object_ids == ()
    assert level[2].space == "no" and level[5].space == "yes"
    assert summary.level_of_object(5) == 0
    assert summary.level_of_surface(4) == 4
    assert summary.level_of_surface(1) == 0
    assert all(s.label != "wall" for s in summary.surfaces)


def test_height_groups_inside_footprint(summary):
    assert summary.by_height[3] == (1, 2)
    assert summary.by_height[4] == (3, 4)
    # The apple at 0.80 m is outside the shelf footprint.
    assert 5 not in sum(summary.by_height.values(), ())


def test_height_groups_without_surfaces():
    # B does not need the tracked surfaces.
    summary = build_summary([], [item(1, "cola", 0.6, 0.0, 0.61)], SHELF)
    assert summary.by_height[3] == (1,)
    assert summary.levels[1].object_ids == ()


def test_signature_changes(surfaces, objects):
    a = build_summary(surfaces, objects, SHELF)
    b = build_summary(surfaces, objects[:-1] + [item(5, "apple", 0.6, 0.0, 1.2)], SHELF)
    assert a.signature() != b.signature()
    assert a.signature() == build_summary(surfaces, objects, SHELF).signature()


def test_unassociated_surface_has_no_level(objects):
    surfaces = [board(3, 0.61, 0, furniture="")]
    summary = build_summary(surfaces, objects, SHELF)
    assert summary.support[1] == 3
    assert summary.level(3).object_ids == ()


def test_objects_of_an_undetected_board_stay_on_their_level(surfaces, objects):
    # Board 4 (0.91 m) not detected: its objects keep level 4 by height.
    without_4 = [s for s in surfaces if s.id != 4]
    summary = build_summary(without_4, objects, SHELF, footprint=SHELF.corners(0.05))
    level = summary.level(4)
    assert level.object_ids == (3, 4)
    assert level.surface_ids == () and not level.board_seen
    assert summary.level(3).board_seen
    assert summary.level_conflicts == {}


def test_height_wins_over_a_wrong_board_association(surfaces, objects):
    # Board 4 associated with level 3 by mistake.
    wrong = [board(4, 0.91, 3) if s.id == 4 else s for s in surfaces]
    summary = build_summary(wrong, objects, SHELF, footprint=SHELF.corners(0.05))
    assert summary.level(4).object_ids == (3, 4)
    assert summary.level(3).object_ids == (1, 2)
    assert summary.level_conflicts == {3: (4, 3), 4: (4, 3)}


def test_other_groups_are_the_other_categories_of_the_level(summary):
    category = {label: c for c, labels in CATEGORIES.items() for label in labels}.get
    # Level 4: orange (3, fruit) and tuna can (4, food).
    assert other_groups(summary, 3, category, level=4) == (summary.object(4).position,)
    assert other_groups(summary, 4, category, level=4) == (summary.object(3).position,)
    # Level 3: green tea and lemonade are both drinks.
    assert other_groups(summary, 1, category, level=3) == ()
    # By support surface when the level is unknown.
    assert other_groups(summary, 3, category, surface_id=4) == (
        summary.object(4).position,
    )
    # An unknown label only groups with the same label.
    assert other_groups(summary, 3, lambda label: None, level=4) == (
        summary.object(4).position,
    )
    assert other_groups(summary, 99, category, level=4) == ()


def test_split_groups_into_the_target_group_and_the_others(summary):
    category = {label: c for c, labels in CATEGORIES.items() for label in labels}.get
    # Level 3: green tea (1) and lemonade (2) are both drinks.
    same, other = split_groups(summary, 1, category, level=3)
    assert same == (summary.object(2).position,) and other == ()
    # Level 4: orange (3) has no other fruit, the tuna can (4) is food.
    same, other = split_groups(summary, 3, category, level=4)
    assert same == () and other == (summary.object(4).position,)


def test_placed_and_taken_objects_update_the_summary(summary):
    obj = item(next_object_id(summary), "cola", 0.6, -0.25, 0.61, 0.12)
    assert obj.id == 6
    placed = with_placed(summary, obj, 3)
    assert placed.level(3).object_ids == (1, 2, 6)
    assert placed.object(6) == obj and placed.support[6] == 0
    assert placed.by_height[3] == (1, 2, 6)
    # The space hints were for the previous held object.
    assert {v.space for v in placed.levels} == {"unknown"}
    taken = without_object(placed, 2)
    assert taken.level(3).object_ids == (1, 6)
    assert taken.object(2) is None and 2 not in taken.support
    assert taken.by_height[3] == (1, 6)
    assert summary.level(3).object_ids == (1, 2)  # unchanged
