import pytest
from conftest import CATEGORIES, item

from aha_test_orchestration.llm import InvalidAnswer
from aha_test_orchestration.place_decision import (
    VARIANTS,
    choice_key,
    describe,
    failure_feedback,
    system_prompt,
    tool_for,
)
from aha_test_orchestration.scene import with_placed

A, B, C = VARIANTS["A"], VARIANTS["B"], VARIANTS["C"]


def test_system_prompt_lists_categories():
    text = system_prompt(CATEGORIES)
    assert "- drink: green tea, lemonade, cola" in text
    assert "choose_place" in text


def test_prompt_a_is_raw(summary, held):
    text = A.user_prompt(summary, held)
    assert "surface 4: label shelf, furniture shelf level 4, height 0.910" in text
    assert "object 3: orange at (0.60, -0.20, 0.950)" in text
    assert "object 5: apple" in text  # every object, also off the shelf
    assert "surface 9" not in text  # walls are not supports
    messages = A.messages(summary, held, CATEGORIES)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_prompt_b_has_heights_and_labels_only(summary, held):
    text = B.user_prompt(summary, held)
    assert "- 0.61 m: green tea, lemonade" in text
    assert "- 0.91 m: orange, tuna can" in text
    assert "- 1.20 m: no objects" in text
    assert "apple" not in text and "O1" not in text and "surface" not in text


def test_prompt_c_has_levels_ids_and_space(summary, held):
    text = C.user_prompt(summary, held)
    assert "- level 3 (0.61 m): O1 green tea, O2 lemonade; space: yes" in text
    assert "- level 2 (0.31 m): no objects; space: no" in text
    assert "apple" not in text
    tool = tool_for(C, summary)
    assert tool["function"]["parameters"]["properties"]["furniture"]["enum"] == [
        "shelf"
    ]
    assert "enum" not in C.tool["function"]["parameters"]["properties"]["furniture"]


def test_prompt_c_without_space_hints(summary, held):
    cleared = with_placed(summary, item(6, "cola", 0.6, -0.25, 0.61, 0.12), 3)
    text = C.user_prompt(cleared, held)
    assert "- level 3 (0.61 m): O1 green tea, O2 lemonade, O6 cola" in text
    assert "space" not in text


def test_failure_feedback_names_the_place(summary):
    c = C.validate({"furniture": "shelf", "level": 4, "next_to": "O3"}, summary)
    text = failure_feedback(c, "UNREACHABLE", "none of 3 candidates within reach")
    assert "level 4 next to O3: UNREACHABLE (none of 3 candidates" in text
    assert "still in the hand" in text and "choose_place" in text
    empty = C.validate({"furniture": "shelf", "level": 5, "next_to": "empty"}, summary)
    assert describe(empty) == "level 5 (empty part)"
    assert describe(VARIANTS["B"].validate({"height": 0.61}, summary)) == (
        "the 0.61 m level"
    )
    a = VARIANTS["A"].validate({"surface_id": 3, "near_object_id": 2}, summary)
    assert describe(a) == "surface 3 next to object 2"
    assert choice_key(c) != choice_key(empty)
    same = C.validate({"furniture": "shelf", "level": 4, "next_to": "3"}, summary)
    assert choice_key(c) == choice_key(same)


def test_validate_a_object_and_point(summary):
    decision = A.validate({"surface_id": 3, "near_object_id": 2}, summary)
    assert decision.level == 3 and decision.next_to == 2
    assert decision.request.surface_ids == (3,)
    assert decision.request.near == (0.6, -0.1, 0.61)
    assert decision.request.near_frame == "odom"
    decision = A.validate(
        {"surface_id": 5, "near_object_id": -1, "x": 0.6, "y": 0.1}, summary
    )
    assert decision.level == 5 and decision.next_to is None
    assert decision.request.near == (0.6, 0.1, 1.2)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"surface_id": 7, "near_object_id": 2}, "unknown surface_id 7"),
        ({"surface_id": 3, "near_object_id": 42}, "unknown near_object_id 42"),
        ({"surface_id": 3, "near_object_id": 3}, "object 3 is on surface 4"),
        ({"surface_id": 3, "near_object_id": -1}, "give x and y"),
        (
            {"surface_id": 3, "near_object_id": -1, "x": 3.0, "y": 0.0},
            "outside surface 3",
        ),
        (
            {"surface_id": "three", "near_object_id": -1},
            "surface_id must be an integer",
        ),
    ],
)
def test_validate_a_errors(summary, arguments, message):
    with pytest.raises(InvalidAnswer, match=message):
        A.validate(arguments, summary)


def test_validate_b(summary):
    decision = B.validate({"height": 0.6}, summary)
    assert decision.level == 3 and decision.height == 0.61
    assert decision.request.restrict_height and decision.request.surface_height == 0.61
    assert decision.request.near is None and not decision.request.surface_ids
    with pytest.raises(InvalidAnswer, match="not a listed level"):
        B.validate({"height": 0.75}, summary)
    with pytest.raises(InvalidAnswer, match="must be a number"):
        B.validate({"height": "high"}, summary)


def test_validate_b2_uses_the_level(summary, held):
    b2 = VARIANTS["B2"]
    assert b2.user_prompt(summary, held) == B.user_prompt(summary, held)
    decision = b2.validate({"height": 0.92}, summary)
    assert decision.level == 4 and decision.variant == "B2"
    assert decision.request.furniture == "shelf"
    assert decision.request.furniture_level == 4
    assert not decision.request.restrict_height and decision.request.near is None


def test_validate_c(summary):
    decision = C.validate({"furniture": "shelf", "level": 3, "next_to": "O2"}, summary)
    assert decision.next_to == 2
    assert (
        decision.request.furniture == "shelf" and decision.request.furniture_level == 3
    )
    assert decision.request.near == pytest.approx((0.6, -0.1, 0.67))
    decision = C.validate(
        {"furniture": "shelf", "level": "5", "next_to": "empty"}, summary
    )
    assert decision.level == 5 and decision.request.near is None
    assert decision.request.near_frame == ""
    assert (
        C.validate({"furniture": "shelf", "level": 4, "next_to": "3"}, summary).next_to
        == 3
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"furniture": "cabinet", "level": 3, "next_to": "O1"}, "unknown furniture"),
        ({"furniture": "shelf", "level": 1, "next_to": "empty"}, "unknown level 1"),
        ({"furniture": "shelf", "level": 3, "next_to": "O9"}, "unknown next_to"),
        ({"furniture": "shelf", "level": 3, "next_to": "O3"}, "O3 is on level 4"),
        ({"furniture": "shelf", "level": 3, "next_to": "O5"}, "O5 is on no level"),
    ],
)
def test_validate_c_errors(summary, arguments, message):
    with pytest.raises(InvalidAnswer, match=message):
        C.validate(arguments, summary)
