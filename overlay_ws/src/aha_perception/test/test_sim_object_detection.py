import math

import numpy as np

from aha_perception.sim_object_detection import (
    corruption,
    depth_at,
    object_visible,
    project,
)

K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
SIZE = (0.07, 0.07, 0.10)


def pose(x, y, z):
    """Upright object (z up) in the optical frame (z forward, y down)."""
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    T[:3, 3] = (x, y, z)
    return T


def test_project():
    assert project(K, (0.0, 0.0, 1.0)) == (320.0, 240.0)
    assert project(K, (0.1, -0.1, 1.0)) == (370.0, 190.0)
    assert project(K, (0.0, 0.0, -1.0)) is None


def test_depth_at_ignores_invalid():
    depth = np.full((10, 10), np.nan)
    assert math.isnan(depth_at(depth, 5, 5))
    depth[5, 5] = 1.0
    depth[5, 6] = 3.0
    depth[4, 5] = 1.2
    assert depth_at(depth, 5, 5) == 1.2
    assert math.isnan(depth_at(depth, -1, 5))


def test_visible_front_surface():
    depth = np.full((480, 640), 2.0, np.float32)
    # The front face 3.5 cm before the center.
    depth[230:250, 310:330] = 1.0 - 0.035
    assert object_visible(depth, K, pose(0.0, 0.0, 1.0), SIZE)


def test_occluded_and_out_of_view():
    depth = np.full((480, 640), 0.5, np.float32)  # something 0.5 m away everywhere
    assert not object_visible(depth, K, pose(0.0, 0.0, 1.0), SIZE)
    depth[:] = 2.0
    assert not object_visible(depth, K, pose(5.0, 0.0, 1.0), SIZE)
    assert not object_visible(depth, K, pose(0.0, 0.0, -1.0), SIZE)


def test_top_face_seen_when_center_hidden():
    depth = np.full((480, 640), 0.5, np.float32)
    T = pose(0.0, 0.0, 1.0)
    top = T[:3, :3] @ np.array([0.0, 0.0, SIZE[2] / 2 - 0.005]) + T[:3, 3]
    u, v = project(K, top)
    depth[
        int(round(v)) - 3 : int(round(v)) + 4, int(round(u)) - 3 : int(round(u)) + 4
    ] = top[2]
    assert object_visible(depth, K, T, SIZE)


def test_corruption_is_fixed_per_seed():
    labels = ["apple", "orange", "cola"]
    assert corruption(0, "apple_1", "apple", labels, 0.0, 0.0) == (False, "apple")
    assert corruption(0, "apple_1", "apple", labels, 1.0, 0.0)[0]
    missed, label = corruption(3, "apple_1", "apple", labels, 0.0, 1.0)
    assert not missed and label in ("orange", "cola")
    assert corruption(3, "apple_1", "apple", labels, 0.5, 0.5) == corruption(
        3, "apple_1", "apple", labels, 0.5, 0.5
    )
    draws = [corruption(s, "m", "apple", labels, 0.1, 0.0)[0] for s in range(2000)]
    assert 0.07 < sum(draws) / len(draws) < 0.13
