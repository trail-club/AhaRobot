"""Boxes around the arm, the hand and the held object, left out of perception.

The camera sees the arm, the gripper and the object in it; above a table or
next to an object they merge with real objects into one obstacle box, or
become boxes of their own. The place server sends these boxes as
``GetPlaceCandidates.exclude_regions`` so perception drops their points first.

Link boxes are the axis-aligned bounds of the astra_description meshes in
each link frame (the collision geometry), placed with the link's TF and grown
by a padding. The held object box is at its pose in the hand (the goal's, or
the grasp's nominal ``place_geometry.object_in_tcp``).
"""

from __future__ import annotations

import numpy as np

from aha_test_manipulation.scene_objects import OBSTACLE, Box


def _mirror(bounds):
    """Bounds of the left arm's link mirrored in the link frame's y."""
    (x0, y0, z0), (x1, y1, z1) = bounds
    return (x0, -y1, z0), (x1, -y0, z1)


_RIGHT = {
    "1": ((-0.004, -0.0375, -0.086), (0.0781, 0.0375, 0.086)),
    "2": ((-0.018, -0.0703, -0.095), (0.3257, 0.018, 0.095)),
    "3": ((-0.018, -0.018, -0.095), (0.283, 0.018, 0.095)),
    "4": ((-0.018, -0.018, -0.064), (0.018, 0.0656, 0.018)),
    "5": ((-0.018, -0.024, -0.032), (0.042, 0.0179, 0.018)),
    "6": ((-0.018, -0.1, 0.019), (0.067, 0.1, 0.1179)),
    "7r": ((-0.168, -0.01, -0.0702), (0.01, 0.018, 0.029)),
    "7l": ((-0.168, -0.01, -0.029), (0.01, 0.018, 0.0702)),
}
# (min, max) corner of each arm link mesh in its own frame (m). The left
# links 2, 4 and 5 are mirrored (link_l5 is also 4 mm shorter in z).
LINK_BOUNDS = {f"link_r{k}": b for k, b in _RIGHT.items()}
LINK_BOUNDS.update(
    {f"link_l{k}": (_mirror(b) if k in ("2", "4") else b) for k, b in _RIGHT.items()}
)
LINK_BOUNDS["link_l5"] = ((-0.018, -0.0179, -0.0278), (0.042, 0.024, 0.018))


def arm_links(arm: str) -> list[str]:
    """Lift, arm, wrist, palm and finger links of ``arm`` ("right" / "left")."""
    side = arm[0]
    return [f"link_{side}{k}" for k in _RIGHT]


def link_bounds(link: str):
    """(min, max) corner of an arm link mesh in the link frame, or None."""
    return LINK_BOUNDS.get(link)


def bounds_box(T_base_link: np.ndarray, bounds, padding: float = 0.0) -> Box:
    """Box (base frame) of the (min, max) corners in a link frame, grown by
    ``padding`` on every side."""
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    T = np.asarray(T_base_link, dtype=float).copy()
    T[:3, 3] = T[:3, :3] @ ((lo + hi) / 2) + T[:3, 3]
    size = tuple(float(v) for v in (hi - lo) + 2 * padding)
    return Box(OBSTACLE, T, size)


def held_object_box(
    T_base_tcp: np.ndarray, T_tcp_object: np.ndarray, size, padding: float = 0.0
) -> Box:
    """Box (base frame) of the held object at T_tcp_object in the hand,
    grown by ``padding`` on every side."""
    T = np.asarray(T_base_tcp, dtype=float) @ np.asarray(T_tcp_object, dtype=float)
    return Box(OBSTACLE, T, tuple(float(v) + 2 * padding for v in size))


def exclude_boxes(
    link_poses: dict,
    T_base_tcp: np.ndarray,
    T_tcp_object: np.ndarray,
    size,
    link_padding: float,
    object_padding: float,
) -> list[Box]:
    """The held object box and a box per arm link in ``link_poses``
    ({link name: 4x4 pose in base}); links without known bounds are skipped."""
    boxes = [held_object_box(T_base_tcp, T_tcp_object, size, object_padding)]
    for link, T in link_poses.items():
        bounds = link_bounds(link)
        if bounds is not None:
            boxes.append(bounds_box(T, bounds, link_padding))
    return boxes
