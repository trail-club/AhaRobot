"""Marker board of the place server (aha_test_manipulation.place_markers)."""

from visualization_msgs.msg import Marker

from aha_test_manipulation import place_markers as pm


def by_ns(array, ns):
    return [m for m in array.markers if m.ns == ns]


def test_numbers_follow_the_try_order_and_survive_a_new_queue():
    board = pm.SpotBoard()
    board.reset((0.6, -0.4, 0.61), group=[(0.6, -0.33, 0.0)], avoid=[(0.6, 0.1, 0.0)])
    board.drop((0.8, -0.3, 0.61), "reach")
    board.queue([(0.6, -0.25, 0.61), (0.55, -0.3, 0.61)])
    assert board.start() == 1
    board.note("IK")
    board.finish(pm.FAILED)
    assert board.start() == 2
    board.finish(pm.FAILED, "blocked")
    # A fresh list after a blocked spot: numbered after the tried ones.
    board.queue([(0.5, -0.35, 0.61)])
    assert board.start() == 3
    board.finish(pm.PLACED, "")
    assert board.start() == 0
    array = board.markers("base_footprint")
    assert array.markers[0].action == Marker.DELETEALL
    assert [m.text for m in by_ns(array, "number")] == ["1 IK", "2 blocked", "3"]
    assert [s.state for s in board.spots] == [pm.FAILED, pm.FAILED, pm.PLACED]
    placed = by_ns(array, "spot")[2].color
    assert abs(placed.g - pm.COLORS[pm.PLACED][1]) < 1e-6
    assert len(by_ns(array, "dropped")) == 1
    (near,) = by_ns(array, "near")
    (group,) = by_ns(array, "group")
    assert group.pose.position.z == near.pose.position.z == 0.61
    assert len(by_ns(array, "avoid")) == 1


def test_queue_replaces_only_the_pending_spots():
    board = pm.SpotBoard()
    board.reset()
    board.queue([(0.5, 0.0, 0.7), (0.6, 0.0, 0.7)])
    board.start()
    board.queue([(0.7, 0.0, 0.7)])
    assert [s.state for s in board.spots] == [pm.TRYING, pm.PENDING]
    assert board.current().point == (0.5, 0.0, 0.7)
    assert by_ns(board.markers("base_footprint"), "near") == []
