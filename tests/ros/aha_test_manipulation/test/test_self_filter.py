"""Boxes around the arms and the held object (aha_test_manipulation.self_filter)."""

import math

import numpy as np
import pytest

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation import scene_objects as scene
from aha_test_manipulation import self_filter


def test_arm_links():
    links = self_filter.arm_links("right")
    assert links[0] == "link_r1" and links[-2:] == ["link_r7r", "link_r7l"]
    assert all(self_filter.link_bounds(link) for link in links)
    assert len(self_filter.arm_links("left")) == len(links)
    assert self_filter.link_bounds("base_link") is None
    # The left upper arm is the right one mirrored in y.
    (_, r_y0, _), (_, r_y1, _) = self_filter.link_bounds("link_r2")
    (_, l_y0, _), (_, l_y1, _) = self_filter.link_bounds("link_l2")
    assert (l_y0, l_y1) == (-r_y1, -r_y0)
    assert self_filter.link_bounds("link_l6") == self_filter.link_bounds("link_r6")


def test_bounds_box_follows_the_link_pose():
    T = geo.transform(geo.rot_z(math.pi / 2), [1.0, 0.0, 0.5])
    box = self_filter.bounds_box(T, ((0.0, -0.1, 0.0), (0.2, 0.1, 0.1)), padding=0.02)
    # Center (0.1, 0, 0.05) in the link frame, turned by 90 deg about z.
    np.testing.assert_allclose(box.pose[:3, 3], [1.0, 0.1, 0.55], atol=1e-12)
    np.testing.assert_allclose(box.pose[:3, :3], T[:3, :3])
    assert box.size == pytest.approx((0.24, 0.24, 0.14))
    assert scene.box_corners(box)[:, 2].min() == pytest.approx(0.48)


def test_held_object_box_matches_the_grasp():
    size = (0.06, 0.08, 0.12)
    tcp = geo.transform(geo.rot_z(-0.3), [0.22, -0.30, 0.92])
    hand = geo.object_in_tcp(geo.SIDE, size[2], 0.05)
    box = self_filter.held_object_box(tcp, hand, size, padding=0.02)
    expected = tcp @ hand
    np.testing.assert_allclose(box.pose, expected)
    assert box.size == pytest.approx((0.10, 0.12, 0.16))
    # Upright: the lowest point is the padded bottom.
    bottom = expected[2, 3] - size[2] / 2 - 0.02
    assert scene.box_corners(box)[:, 2].min() == pytest.approx(bottom)


def test_exclude_boxes_cover_the_object_and_arm_links():
    size = (0.06, 0.06, 0.12)
    tcp = geo.transform(None, [0.22, -0.30, 0.92])
    poses = {
        "link_r6": geo.transform(None, [0.1, -0.3, 0.9]),
        "link_r7r": geo.transform(None, [0.2, -0.3, 0.9]),
        "camera_link": geo.transform(None, [0.0, 0.0, 1.4]),
    }
    hand = geo.object_in_tcp(geo.SIDE, size[2], 0.05)
    boxes = self_filter.exclude_boxes(poses, tcp, hand, size, 0.02, 0.02)
    assert len(boxes) == 3  # held object, link_r6, link_r7r
    held = self_filter.held_object_box(tcp, hand, size, 0.02)
    np.testing.assert_allclose(boxes[0].pose, held.pose)
    lo, hi = self_filter.link_bounds("link_r6")
    assert boxes[1].size == pytest.approx(tuple(np.subtract(hi, lo) + 0.04))
