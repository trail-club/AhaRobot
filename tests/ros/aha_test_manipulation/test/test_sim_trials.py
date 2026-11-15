"""ROS-free parts of sim_trials (needs the ROS messages: colcon test)."""

import pytest

from aha_test_manipulation.sim_trials import PlaceServerBusy, cancel_goal, open_csv


class Future:
    def __init__(self):
        self.finished = False

    def done(self):
        return self.finished


class Holder:
    def spin_until(self, predicate, timeout):
        return predicate()


def test_canceled_goal_that_stops_is_waited_for():
    future = Future()

    class Handle:
        def cancel_goal_async(self):
            future.finished = True

    cancel_goal(Holder(), Handle(), future)
    assert future.finished


def test_goal_that_ignores_the_cancel_means_a_busy_server():
    class Handle:
        def cancel_goal_async(self):
            pass

    with pytest.raises(PlaceServerBusy):
        cancel_goal(Holder(), Handle(), Future())


def test_csv_rewrite_keeps_rows_when_columns_change(tmp_path):
    path = tmp_path / "trials.csv"
    stream, writer = open_csv(path, ["a", "b", "c"])
    writer.writerow({"a": 1, "b": 2, "c": 3})
    stream.close()
    stream, writer = open_csv(path, ["a", "c"])  # b removed
    writer.writerow({"a": 4, "c": 5})
    stream.close()
    stream, writer = open_csv(path, ["a", "c", "d"])  # d added
    writer.writerow({"a": 6, "c": 7, "d": 8})
    stream.close()
    assert path.read_text().splitlines() == ["a,c,d", "1,3,", "4,5,", "6,7,8"]
    assert list(tmp_path.iterdir()) == [path]
