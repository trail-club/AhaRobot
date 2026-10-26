"""FrozenWatch on synthetic trajectory feedback (ROS free)."""

from aha_test_manipulation.frozen_watch import FrozenWatch


def feed(watch, samples):
    """The first reason, or None, over (t, desired, actual) samples."""
    for t, desired, actual in samples:
        reason = watch.update(t, desired, actual)
        if reason:
            return reason
    return None


def test_moving_readings_are_not_frozen():
    watch = FrozenWatch(0.5, 0.05)
    samples = [(0.05 * k, [0.01 * k], [0.01 * k - 0.02]) for k in range(40)]
    assert feed(watch, samples) is None


def test_constant_readings_while_the_command_moves_are_frozen():
    watch = FrozenWatch(0.5, 0.05)
    samples = [(0.05 * k, [0.1 * k], [0.3]) for k in range(20)]
    reason = feed(watch, samples)
    assert reason and "frozen readings" in reason


def test_an_arm_at_rest_with_the_command_stopped_is_not_frozen():
    # Off the goal by a steady 0.04 rad: the readings stay, the command too.
    watch = FrozenWatch(0.5, 0.05)
    samples = [(0.05 * k, [0.5], [0.46]) for k in range(40)]
    assert feed(watch, samples) is None


def test_only_the_motion_since_the_readings_last_changed_counts():
    watch = FrozenWatch(0.5, 0.05)
    moving = [(0.05 * k, [0.02 * k], [0.02 * k]) for k in range(20)]
    # Readings stop while a slow final approach moves the command 0.02 rad on.
    settle = [(1.0 + 0.05 * k, [0.38 + 0.001 * k], [0.38]) for k in range(1, 21)]
    assert feed(watch, moving + settle) is None


def test_a_single_frozen_joint_is_flagged():
    # The wrist reading stops (servo read errors) while the others follow.
    watch = FrozenWatch(0.5, 0.05)
    samples = [(0.05 * k, [0.02 * k, 0.03 * k], [0.02 * k, 0.0015]) for k in range(30)]
    reason = None
    for t, desired, actual in samples:
        reason = watch.update(t, desired, actual, ["joint_r2", "joint_r5"]) or reason
    assert reason and "joint_r5" in reason and "joint_r2" not in reason
    assert watch.frozen == [1]


def test_reset_forgets_the_last_readings():
    watch = FrozenWatch(0.5, 0.05)
    assert feed(watch, [(0.05 * k, [0.1 * k], [0.3]) for k in range(20)])
    watch.reset()
    assert watch.update(10.0, [5.0], [0.3]) is None
