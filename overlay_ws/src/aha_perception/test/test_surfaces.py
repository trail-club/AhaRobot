import dataclasses

import numpy as np
import pytest

from aha_perception.planes import PlaneLabel
from aha_perception.planes.geometry import (
    clip_convex_polygons,
    convex_hull_2d,
    polygon_area,
)
from aha_perception.planes.segmentation import make_plane
from aha_perception.surfaces import (
    CameraView,
    SurfaceTracker,
    TrackerOptions,
    convex_hull,
    intersection_area,
)

UP = np.array([0.0, 0.0, 1.0])
CAMERA = np.array([0.0, 0.0, 1.43])
IDENTITY = np.eye(4)
STEP = 0.02


def horizontal(x0, x1, y0, y1, z, label=PlaneLabel.TABLE, transform=IDENTITY):
    """Rectangle at height z; ``transform`` maps it into the plane's frame."""
    x, y = np.meshgrid(np.arange(x0, x1 + 1e-9, STEP), np.arange(y0, y1 + 1e-9, STEP))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])
    points = points @ transform[:3, :3].T + transform[:3, 3]
    plane = make_plane(points, np.arange(len(points)), CAMERA, UP)
    plane.label = label
    return plane


def wall(x, y0, y1, z0=0.0, z1=2.0):
    y, z = np.meshgrid(np.arange(y0, y1 + 1e-9, STEP), np.arange(z0, z1 + 1e-9, STEP))
    points = np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])
    plane = make_plane(points, np.arange(len(points)), CAMERA, UP)
    plane.label = PlaneLabel.WALL
    return plane


def relabel(plane, label):
    return dataclasses.replace(plane, label=label)


def make_view(camera_pose, pan_deg=0.0, depth=None):
    return CameraView(
        camera_to_tracking=camera_pose(pan_deg=pan_deg),
        fx=465.7,
        fy=465.7,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
        depth=depth,
    )


def by_id(tracker):
    return {s.id: s for s in tracker.surfaces()}


TABLE = (1.0, 1.6, -0.4, 0.4, 0.72)


def confirmed(tracker, planes, stamp=0.0):
    tracker.update(planes, IDENTITY, stamp)
    return tracker.update(planes, IDENTITY, stamp + 0.2)


def test_ids_are_confirmed_on_the_second_detection_and_stay():
    tracker = SurfaceTracker()
    floor = horizontal(0.5, 3.0, -1.5, 1.5, 0.0, PlaneLabel.FLOOR)
    table = horizontal(*TABLE)
    assert tracker.update([floor, table], IDENTITY, 0.0) == [0, 0]
    assert tracker.surfaces() == []
    floor_id, table_id = tracker.update([floor, table], IDENTITY, 0.2)
    assert sorted([floor_id, table_id]) == [1, 2]

    rng = np.random.default_rng(0)
    for k in range(10):
        dx, dy = rng.uniform(-0.02, 0.02, 2)
        dz = rng.uniform(-0.01, 0.01)
        moved = horizontal(1.0 + dx, 1.6 + dx, -0.4 + dy, 0.4 + dy, 0.72 + dz)
        ids = tracker.update([moved, floor], IDENTITY, 0.4 + 0.2 * k)
        assert ids == [table_id, floor_id]
    surfaces = by_id(tracker)
    assert surfaces[table_id].label == "table"
    assert surfaces[floor_id].label == "floor"
    assert surfaces[table_id].observations == 12
    assert surfaces[table_id].height == pytest.approx(0.72, abs=0.01)
    assert all(s.visible for s in surfaces.values())


def test_base_motion_is_compensated_by_the_tracking_transform():
    tracker = SurfaceTracker()
    (table_id,) = confirmed(tracker, [horizontal(*TABLE)])
    # The base moved 0.5 m forward and turned 30 deg left: the table, fixed in
    # the tracking frame, is seen elsewhere in the base frame.
    yaw = np.radians(30.0)
    base_to_odom = np.eye(4)
    base_to_odom[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    base_to_odom[:2, 3] = (0.5, 0.0)
    seen = horizontal(*TABLE, transform=np.linalg.inv(base_to_odom))
    assert tracker.update([seen], base_to_odom, 1.0) == [table_id]
    assert by_id(tracker)[table_id].centroid[:2] == pytest.approx([1.3, 0.0], abs=0.02)


def test_two_tables_at_the_same_height_get_different_ids():
    tracker = SurfaceTracker()
    left = horizontal(1.0, 1.6, 0.05, 0.65, 0.72)  # 0.1 m apart
    right = horizontal(1.0, 1.6, -0.65, -0.05, 0.72)
    left_id, right_id = confirmed(tracker, [left, right])
    assert left_id and right_id and left_id != right_id
    for k in range(6):
        planes = [left] if k % 2 else [right]
        expected = [left_id] if k % 2 else [right_id]
        assert tracker.update(planes, IDENTITY, 1.0 + 0.2 * k) == expected
    assert set(by_id(tracker)) == {left_id, right_id}


def test_partial_view_keeps_the_id_and_the_unseen_part_of_the_hull(camera_pose):
    tracker = SurfaceTracker()
    full = horizontal(1.0, 1.6, -0.3, 1.8, 0.72)
    (table_id,) = confirmed(tracker, [full])
    full_area = by_id(tracker)[table_id].area
    # The camera sees y < ~0.8 m; the detected part ends at 0.5 m.
    partial = horizontal(1.0, 1.6, -0.3, 0.5, 0.72)
    view = make_view(camera_pose)
    assert tracker.update([partial], IDENTITY, 1.0, view) == [table_id]
    surface = by_id(tracker)[table_id]
    assert surface.area == pytest.approx(full_area, rel=0.05)
    assert surface.hull[:, 1].max() == pytest.approx(1.8, abs=0.03)
    # Without a view the hull follows the latest observation.
    assert tracker.update([partial], IDENTITY, 1.2) == [table_id]
    assert by_id(tracker)[table_id].area == pytest.approx(0.6 * 0.8, rel=0.05)


def test_surfaces_out_of_view_are_kept_until_max_unseen(camera_pose):
    tracker = SurfaceTracker(TrackerOptions(max_unseen_s=120.0))
    (table_id,) = confirmed(tracker, [horizontal(*TABLE)])
    away = make_view(camera_pose, pan_deg=180.0)
    for k in range(30):
        tracker.update([], IDENTITY, 1.0 + k, away)
    surface = by_id(tracker)[table_id]
    assert not surface.visible
    assert surface.last_seen == pytest.approx(0.2)
    tracker.update([], IDENTITY, 120.0, away)
    assert table_id in by_id(tracker)
    tracker.update([], IDENTITY, 120.3, away)
    assert by_id(tracker) == {}


def test_surfaces_expected_in_view_are_deleted_after_max_misses(camera_pose):
    tracker = SurfaceTracker(TrackerOptions(max_misses=5))
    (table_id,) = confirmed(tracker, [horizontal(*TABLE)])
    ahead = make_view(camera_pose)
    for k in range(4):
        tracker.update([], IDENTITY, 1.0 + 0.2 * k, ahead)
        assert table_id in by_id(tracker)
    tracker.update([], IDENTITY, 2.0, ahead)
    assert by_id(tracker) == {}


def test_misses_need_consecutive_expected_detections(camera_pose):
    tracker = SurfaceTracker(TrackerOptions(max_misses=3))
    table = horizontal(*TABLE)
    (table_id,) = confirmed(tracker, [table])
    ahead = make_view(camera_pose)
    for k in range(10):
        planes = [table] if k % 2 else []
        tracker.update(planes, IDENTITY, 1.0 + 0.2 * k, ahead)
    assert table_id in by_id(tracker)


def test_occluded_surfaces_are_not_expected(camera_pose):
    tracker = SurfaceTracker(TrackerOptions(max_misses=2))
    (table_id,) = confirmed(tracker, [horizontal(*TABLE)])
    occluder = np.full((480, 640), 0.5)  # something 0.5 m in front of the camera
    view = make_view(camera_pose, depth=occluder)
    for k in range(10):
        tracker.update([], IDENTITY, 1.0 + 0.2 * k, view)
    assert table_id in by_id(tracker)
    # Seeing through the table (the floor behind it) is evidence it is gone.
    through = make_view(camera_pose, depth=np.full((480, 640), 4.0))
    tracker.update([], IDENTITY, 4.0, through)
    tracker.update([], IDENTITY, 4.2, through)
    assert by_id(tracker) == {}


def test_tentative_tracks_survive_one_miss_only():
    tracker = SurfaceTracker(TrackerOptions(tentative_max_misses=2))
    table = horizontal(*TABLE)
    # Detected every other frame: confirmed at the second detection.
    assert tracker.update([table], IDENTITY, 0.0) == [0]
    tracker.update([], IDENTITY, 0.2)
    assert tracker.update([table], IDENTITY, 0.4) == [1]
    other = horizontal(2.0, 2.5, 1.0, 1.5, 0.45)
    assert tracker.update([table, other], IDENTITY, 0.6) == [1, 0]
    tracker.update([table], IDENTITY, 0.8)
    tracker.update([table], IDENTITY, 1.0)
    assert tracker.update([table, other], IDENTITY, 1.2) == [1, 0]
    assert tracker.update([table, other], IDENTITY, 1.4) == [1, 2]


def test_confirm_hits_one_gives_ids_immediately():
    tracker = SurfaceTracker(TrackerOptions(confirm_hits=1))
    assert tracker.update([horizontal(*TABLE)], IDENTITY, 0.0) == [1]


def test_fragments_of_one_surface_share_its_id():
    tracker = SurfaceTracker()
    (table_id,) = confirmed(tracker, [horizontal(*TABLE)])
    left = horizontal(1.0, 1.6, 0.02, 0.4, 0.72)
    right = horizontal(1.0, 1.6, -0.4, -0.02, 0.72)
    assert tracker.update([left, right], IDENTITY, 1.0) == [table_id, table_id]
    assert list(by_id(tracker)) == [table_id]


def test_adjacent_pieces_of_a_wall_keep_its_id():
    tracker = SurfaceTracker()
    (wall_id,) = confirmed(tracker, [wall(3.0, -1.5, 0.0, 0.5, 2.0)])
    # The next view of the wall starts 3 cm beyond the hull seen so far.
    assert tracker.update([wall(3.0, 0.03, 1.5, 0.5, 2.0)], IDENTITY, 1.0) == [wall_id]
    assert tracker.update([wall(3.0, 1.52, 3.0, 0.5, 2.0)], IDENTITY, 1.2) == [wall_id]
    # A piece 0.1 m beyond is another surface.
    assert tracker.update([wall(3.0, 3.1, 4.0, 0.5, 2.0)], IDENTITY, 1.4) == [0]


def test_floor_fragments_are_one_surface():
    tracker = SurfaceTracker()
    left = horizontal(0.5, 3.0, 0.5, 1.5, 0.0, PlaneLabel.FLOOR)
    right = horizontal(0.5, 3.0, -1.5, -0.5, 0.01, PlaneLabel.FLOOR)
    floor_id, other_id = confirmed(tracker, [left, right])
    assert floor_id == other_id != 0
    far = horizontal(4.0, 5.0, -3.0, -2.0, 0.0, PlaneLabel.FLOOR)
    assert tracker.update([far], IDENTITY, 1.0) == [floor_id]


def test_duplicate_tracks_merge_into_the_older_id():
    tracker = SurfaceTracker()
    left = horizontal(1.0, 1.6, 0.05, 0.4, 0.72)
    right = horizontal(1.0, 1.6, -0.4, -0.05, 0.72)
    left_id, right_id = confirmed(tracker, [left, right])
    assert left_id != right_id
    full = horizontal(*TABLE)
    assert tracker.update([full], IDENTITY, 1.0) == [min(left_id, right_id)]
    surfaces = by_id(tracker)
    assert list(surfaces) == [min(left_id, right_id)]
    assert surfaces[min(left_id, right_id)].observations == 5


def test_stacked_boards_get_levels_from_the_lowest():
    tracker = SurfaceTracker()
    boards = [
        horizontal(1.0, 1.35, -0.4, 0.4, z, PlaneLabel.SHELF) for z in (0.3, 0.65, 1.0)
    ]
    table = horizontal(2.0, 2.6, 1.0, 1.6, 0.72)
    floor = horizontal(0.5, 3.0, -1.5, 1.5, 0.0, PlaneLabel.FLOOR)
    ids = confirmed(tracker, [boards[1], floor, boards[2], table, boards[0]])
    surfaces = by_id(tracker)
    levels = [surfaces[i].level for i in ids]
    assert levels == [2, 0, 3, 0, 1]


def test_label_is_the_majority_of_recent_observations():
    tracker = SurfaceTracker(TrackerOptions(label_window=10))
    table = horizontal(*TABLE)
    other = relabel(table, PlaneLabel.OTHER)
    tracker.update([table], IDENTITY, 0.0)
    (table_id,) = tracker.update([table], IDENTITY, 0.2)
    tracker.update([other], IDENTITY, 0.4)
    assert by_id(tracker)[table_id].label == "table"
    for k in range(3):
        tracker.update([other], IDENTITY, 0.6 + 0.2 * k)
    assert by_id(tracker)[table_id].label == "other"


def test_wall_and_floor_are_separate_surfaces():
    tracker = SurfaceTracker()
    floor = horizontal(0.5, 3.0, -1.5, 1.5, 0.0, PlaneLabel.FLOOR)
    front = wall(3.0, -1.5, 1.5)
    floor_id, wall_id = confirmed(tracker, [floor, front])
    assert floor_id != wall_id
    surfaces = by_id(tracker)
    assert surfaces[wall_id].vertical and not surfaces[floor_id].vertical
    assert surfaces[wall_id].level == 0
    # A parallel wall 0.5 m behind is another surface.
    behind = wall(3.5, -1.5, 1.5)
    assert tracker.update([floor, behind], IDENTITY, 1.0) == [floor_id, 0]
    assert tracker.update([floor, behind], IDENTITY, 1.2)[1] not in (0, wall_id)


def test_reset_forgets_surfaces_and_ids_continue():
    tracker = SurfaceTracker()
    (first,) = confirmed(tracker, [horizontal(*TABLE)])
    tracker.reset()
    assert tracker.surfaces() == []
    (second,) = confirmed(tracker, [horizontal(*TABLE)], stamp=5.0)
    assert second > first


def test_intersection_area_matches_the_clipped_polygon():
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, _ = convex_hull_2d(rng.normal(0.0, 1.0, (12, 2)))
        b, _ = convex_hull_2d(rng.normal(rng.uniform(-1.5, 1.5, 2), 1.0, (12, 2)))
        expected = abs(polygon_area(clip_convex_polygons(a, b)))
        assert intersection_area(a, b) == pytest.approx(expected, abs=1e-9)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert intersection_area(square, square + 0.5) == pytest.approx(0.25)
    assert intersection_area(square, square + 2.0) == 0.0


def test_convex_hull_matches_qhull():
    rng = np.random.default_rng(4)
    for n in (3, 10, 300):
        points = rng.normal(0.0, 1.0, (n, 2))
        hull, area = convex_hull(points)
        expected, expected_area = convex_hull_2d(points)
        assert area == pytest.approx(expected_area)
        assert polygon_area(hull) > 0.0  # counter-clockwise
        assert {tuple(p) for p in hull} == {tuple(p) for p in expected}
    line = np.column_stack([np.linspace(0, 1, 5), np.zeros(5)])
    assert convex_hull(line)[1] == 0.0


def test_stamp_going_back_resets_the_tracker():
    tracker = SurfaceTracker()
    confirmed(tracker, [horizontal(*TABLE)], stamp=100.0)
    assert tracker.update([horizontal(*TABLE)], IDENTITY, 10.0) == [0]
    assert tracker.surfaces() == []
