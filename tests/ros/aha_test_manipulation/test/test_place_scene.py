"""PlaceScene.add_placed with the planning scene calls stubbed (no ROS)."""

from types import SimpleNamespace

import numpy as np
import pytest

from aha_test_manipulation.place_scene import PLACED_OBJECT_ID, PlaceScene

SIZE = (0.08, 0.08, 0.10)


class Log:
    def __init__(self):
        self.lines = []

    def warn(self, text, **_):
        self.lines.append(text)


def scene_stub(touch_below):
    """A stand-in whose hand touches the placed box when the box is wider than
    ``touch_below`` (m): the hand is that far from the object's center."""
    log = Log()
    stub = SimpleNamespace(
        frame="base_footprint",
        boxes=[],
        allowed=[],
        node=SimpleNamespace(get_logger=lambda: log),
        log=log,
    )

    def apply(objects, *_):
        stub.boxes.append(list(objects[0].primitives[0].dimensions))

    def contacts(arm, is_object):
        if stub.boxes[-1][0] / 2 > touch_below:
            return {PLACED_OBJECT_ID: ({"link_r7l"}, [])}
        return {}

    stub.apply = apply
    stub.contacts = contacts
    stub.allow = lambda object_id, bodies: stub.allowed.append((object_id, bodies))
    stub.add_placed = PlaceScene.add_placed.__get__(stub)
    return stub


def place(stub, padding):
    req = SimpleNamespace(arm="right", size=SIZE)
    motion = SimpleNamespace(object_pose=np.eye(4))
    stub.add_placed(req, motion, padding)


def test_padded_box_when_the_hand_is_clear():
    stub = scene_stub(touch_below=0.2)
    place(stub, 0.03)
    assert stub.boxes == [pytest.approx([0.14, 0.14, 0.16])]
    assert stub.allowed == []


def test_hand_inside_the_padding_keeps_the_object_size():
    # A retreat stopped 0.05 m from the center: inside the padded box (0.07)
    # but clear of the object (0.04).
    stub = scene_stub(touch_below=0.05)
    place(stub, 0.03)
    assert stub.boxes[-1] == pytest.approx(list(SIZE))
    assert stub.allowed == []
    assert any("kept at the object's size" in line for line in stub.log.lines)


def test_hand_at_the_object_allows_only_the_touching_links():
    stub = scene_stub(touch_below=0.01)
    place(stub, 0.03)
    assert stub.boxes[-1] == pytest.approx(list(SIZE))
    ((object_id, bodies),) = stub.allowed
    assert object_id == PLACED_OBJECT_ID and "link_r7l" in bodies
