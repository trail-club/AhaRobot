"""Ramp sequences and contact detection for the gripper handover tool
(aha_test_manipulation.gripper_ramp)."""

import pytest

from aha_test_manipulation import gripper_ramp as ramp


def test_close_target_is_half_width_minus_squeeze():
    assert ramp.close_target(0.03, 0.003) == pytest.approx(0.012)


def test_close_target_clamps_to_the_finger_range():
    assert ramp.close_target(1.0, 0.003) == ramp.FINGER_RANGE[1]
    assert ramp.close_target(0.0, 0.05) == ramp.FINGER_RANGE[0]


def test_estimated_width_inverts_close_target():
    width, squeeze = 0.03, 0.003
    target = ramp.close_target(width, squeeze)
    assert ramp.estimated_width(target, squeeze) == pytest.approx(width)


def test_simple_ramp_reaches_target_and_steps_monotonically_toward_it():
    r = ramp.SimpleRamp(start=0.03, target=0.0, duration=1.0, rate=20.0)
    assert r.total_steps == 20
    values = []
    while not r.done:
        values.append(r.next_command())
    assert values[-1] == pytest.approx(0.0)
    assert all(a >= b for a, b in zip(values, values[1:]))  # closing: decreasing


def test_simple_ramp_single_step_when_duration_is_zero():
    r = ramp.SimpleRamp(start=0.03, target=0.0, duration=0.0, rate=20.0)
    assert r.total_steps == 1
    assert r.next_command() == pytest.approx(0.0)
    assert r.done


def test_simple_ramp_clamps_target_to_the_finger_range():
    r = ramp.SimpleRamp(start=0.0, target=1.0, duration=0.1, rate=20.0)
    while not r.done:
        value = r.next_command()
    assert value == ramp.FINGER_RANGE[1]


def test_contact_ramp_runs_to_target_without_contact():
    # The measured position always keeps up with the command: never blocked.
    r = ramp.ContactRamp(start=0.03, target=0.0, duration=1.0, rate=20.0, squeeze=0.003)
    measured = r.start
    while not r.done:
        measured = r.next_command(measured)
    assert measured == pytest.approx(0.0)
    assert not r.holding
    assert r.contact_position is None


def test_contact_ramp_stops_and_holds_measured_minus_squeeze_on_contact():
    # The object stops the fingers at 0.012 m; the state keeps reporting that
    # while the command keeps trying to advance below it.
    contact_at = 0.012
    r = ramp.ContactRamp(start=0.03, target=0.0, duration=1.0, rate=20.0, squeeze=0.003)
    measured = r.start
    command = measured
    for _ in range(200):
        command = r.next_command(measured)
        measured = max(command, contact_at)
        if r.done:
            break
    assert r.holding
    assert r.contact_position == pytest.approx(contact_at)
    assert command == pytest.approx(contact_at - 0.003)


def test_contact_ramp_requires_consecutive_samples_not_a_single_blip():
    r = ramp.ContactRamp(
        start=0.03,
        target=0.0,
        duration=1.0,
        rate=20.0,
        squeeze=0.003,
        gap=0.004,
        samples=3,
    )
    # One glitchy sample (reported well above the last command), then the
    # state goes back to closely tracking the command: must not latch.
    r.next_command(0.04)
    measured = r.start
    for _ in range(10):
        command = r.next_command(measured)
        measured = command  # the state keeps up with the command exactly
    assert not r.holding


def test_contact_ramp_lagging_but_moving_finger_is_not_contact():
    # The real fingers lag the ramp by more than the gap while still closing.
    r = ramp.ContactRamp(start=0.03, target=0.0, duration=1.0, rate=20.0, squeeze=0.003)
    measured = r.start
    for _ in range(19):
        command = r.next_command(measured)
        measured = max(command + 0.008, measured - 0.001)  # 8 mm behind, moving
    assert not r.holding


def test_contact_ramp_ignores_measured_none_like_simple_ramp():
    r = ramp.ContactRamp(start=0.03, target=0.0, duration=1.0, rate=20.0, squeeze=0.003)
    values = []
    while not r.done:
        values.append(r.next_command(None))
    assert values[-1] == pytest.approx(0.0)
    assert not r.holding


@pytest.mark.parametrize(
    "samples,expected",
    [
        ([], False),
        ([(0.0, 0.01)], False),
        ([(0.0, 0.01), (1.0, 0.01)], False),  # not long enough yet
        ([(0.0, 0.01), (2.5, 0.01)], True),  # identical for > STALE_WINDOW
        ([(0.0, 0.01), (1.0, 0.02), (2.5, 0.03)], False),  # moving: not stale
    ],
)
def test_is_stale(samples, expected):
    assert ramp.is_stale(samples) is expected


def test_finger_joint_names():
    assert ramp.finger_joint("right") == "joint_r7r"
    assert ramp.finger_joint("left") == "joint_l7r"
