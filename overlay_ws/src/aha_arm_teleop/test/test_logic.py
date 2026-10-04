"""arm_node へ出す指令が、シリアル教示の step / lead / 可動域を崩していないか。"""

import math

from aha_arm_teleop.logic import (
    GRIPPER_GEAR_R,
    GRIPPER_ZERO_M,
    RAD_PER_COUNT,
    Teleop,
    encoder_raw,
    si_from_raw,
)

LEFT_ARM = ("joint_l2", "joint_l3", "joint_l4", "joint_l5", "joint_l6")
RIGHT_ARM = ("joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6")


def _center(tele: Teleop) -> None:
    for name in LEFT_ARM + RIGHT_ARM:
        tele.update_feedback(name, 0.0)
    tele.update_feedback("joint_l7r", GRIPPER_ZERO_M)
    tele.update_feedback("joint_r7r", GRIPPER_ZERO_M)
    # 反対側の指は指令に使わない。来ても無視する。
    tele.update_feedback("joint_l7l", -GRIPPER_ZERO_M)
    tele.update_feedback("joint_r7l", -GRIPPER_ZERO_M)


def _arm(outputs, topic):
    found = [c for c in outputs.commands if c.topic == topic]
    return found[-1] if found else None


def test_raw_roundtrip_matches_arm_controller_si():
    raw = encoder_raw(si_from_raw(100, False), False)
    assert raw == 100
    grip = encoder_raw(si_from_raw(1448, True), True)
    assert grip == 1448
    # 中点 2048 は腕 0 rad、グリッパ 0.03 m。
    assert si_from_raw(2048, False) == 0.0
    assert math.isclose(si_from_raw(2048, True), GRIPPER_ZERO_M)


def test_no_command_until_that_arm_is_complete():
    tele = Teleop()
    for name in LEFT_ARM:
        tele.update_feedback(name, 0.0)
    out = tele.tick(["w"], 0.0, 0.0)
    assert out.commands == []
    tele.update_feedback("joint_l7r", GRIPPER_ZERO_M)
    out = tele.tick([], 0.01, 0.01)
    command = _arm(out, "/left/arm/joint_command")
    assert command is not None
    assert command.names == LEFT_ARM


def test_tap_uses_serial_step_and_inverts_into_ros_radians():
    tele = Teleop()
    _center(tele)
    out = tele.tick(["w"], 0.0, 0.0)
    command = _arm(out, "/left/arm/joint_command")
    assert command.names == LEFT_ARM
    # +8 count は raw が増える向き。SI は減る。他の関節は中点のまま。
    assert command.positions[0] == -8 * RAD_PER_COUNT
    assert command.positions[1:] == (0.0, 0.0, 0.0, 0.0)
    assert _arm(out, "/right/arm/joint_command") is None
    assert _arm(out, "/left/arm/gripper_joint_command") is None


def test_right_arm_is_uppercase_and_independent():
    tele = Teleop()
    _center(tele)
    out = tele.tick(["W"], 0.0, 0.0)
    command = _arm(out, "/right/arm/joint_command")
    assert command.names == RIGHT_ARM
    assert command.positions[0] == -8 * RAD_PER_COUNT
    assert _arm(out, "/left/arm/joint_command") is None


def test_gripper_command_is_only_the_positive_finger_in_meters():
    tele = Teleop()
    _center(tele)
    out = tele.tick(["y"], 0.0, 0.0)
    command = _arm(out, "/left/arm/gripper_joint_command")
    assert command.names == ("joint_l7r",)
    assert math.isclose(
        command.positions[0],
        GRIPPER_ZERO_M - 8 * RAD_PER_COUNT * GRIPPER_GEAR_R,
    )
    out = tele.tick(["Y"], 0.1, 0.0)
    right = _arm(out, "/right/arm/gripper_joint_command")
    assert right.names == ("joint_r7r",)
    assert math.isclose(right.positions[0], command.positions[0])


def test_second_tap_inside_repeat_window_starts_hold_and_lead_caps_it():
    tele = Teleop(lead=80, rate=250)
    _center(tele)
    tele.tick(["w"], 0.0, 0.0)
    now = 0.0
    goal = 0.0
    peak = 0.0
    while now < 1.0:
        now += 0.1
        out = tele.tick(["w"], now, 0.1)
        command = _arm(out, "/left/arm/joint_command")
        if command is not None:
            goal = -command.positions[0] / RAD_PER_COUNT
            peak = max(peak, goal)
        assert goal <= 80 + 1e-6
    # 実位置が追従しないと stall が先行分を捨てる。捨てる前に lead ちょうどまで進む。
    assert peak == 80


def test_single_tap_does_not_keep_moving():
    tele = Teleop()
    _center(tele)
    tele.tick(["e"], 0.0, 0.0)
    out = tele.tick([], 0.5, 0.5)
    assert _arm(out, "/left/arm/joint_command") is None
    assert tele.arms["left"].joints[1].goal == 8


def test_space_snaps_goal_to_actual_and_publishes_both():
    tele = Teleop()
    _center(tele)
    tele.tick(["w"], 0.0, 0.0)
    # 実機が指令まで来た。
    tele.update_feedback("joint_l2", -8 * RAD_PER_COUNT)
    out = tele.tick([" "], 0.2, 0.0)
    command = _arm(out, "/left/arm/joint_command")
    assert math.isclose(command.positions[0], -8 * RAD_PER_COUNT)
    grip = _arm(out, "/left/arm/gripper_joint_command")
    assert math.isclose(grip.positions[0], GRIPPER_ZERO_M)


def test_origin_margin_blocks_the_wrap_direction_on_the_wide_wrist():
    tele = Teleop()
    _center(tele)
    # joint_*4 (wrist12) だけ raw 100。+方向の余裕はあるが -方向は原点マージン内。
    tele.update_feedback("joint_l4", si_from_raw(100, False))
    # 基準を取り直す。トルク OFF → 方向キーで resync。
    tele.tick(["0"], 0.0, 0.0)
    tele.tick(["f"], 0.1, 0.0)  # resync だけで動かない
    blocked = tele.tick(["f"], 0.2, 0.0)
    assert tele.arms["left"].joints[2].goal == 0
    assert "無効" in blocked.message
    moved = tele.tick(["r"], 0.3, 0.0)
    command = _arm(moved, "/left/arm/joint_command")
    assert command.names[2] == "joint_l4"
    assert command.positions[2] < si_from_raw(100, False)


def test_firmware_limit_is_applied_before_the_packet_would_be_dropped():
    tele = Teleop()
    _center(tele)
    start = math.pi / 2 - 3 * RAD_PER_COUNT
    tele.update_feedback("joint_l2", start)
    tele.tick(["0"], 0.0, 0.0)
    tele.tick(["s"], 0.1, 0.0)  # resync
    out = tele.tick(["s"], 0.3, 0.0)
    command = _arm(out, "/left/arm/joint_command")
    # 8 step 欲しいが、arm_node の +π/2 まで残り 3 step しかない。
    assert tele.arms["left"].joints[0].goal == -3
    assert command.positions[0] <= math.pi / 2 + 1e-9
    assert math.isclose(command.positions[0], math.pi / 2, abs_tol=1e-9)


def test_stall_drops_the_lead_and_blocks_that_direction():
    tele = Teleop(lead=80, rate=250, stall_time=0.5)
    _center(tele)
    now = 0.0
    tele.tick(["w"], now, 0.0)
    saw_stop = False
    for _ in range(20):
        now += 0.1
        out = tele.tick(["w"], now, 0.1)
        if "動かない" in out.message:
            saw_stop = True
            break
    assert saw_stop
    joint = tele.arms["left"].joints[0]
    assert joint.goal == 0
    assert joint.blocked_dir == 1
    held = tele.tick(["w"], now + 0.05, 0.0)
    assert joint.goal == 0
    assert "停止中" in held.message
    back = tele.tick(["s"], now + 0.1, 0.0)
    assert joint.goal == -8
    assert _arm(back, "/left/arm/joint_command") is not None


def test_jump_disconnects_until_torque_cycles():
    tele = Teleop(max_jump=250)
    _center(tele)
    tele.tick(["u"], 0.0, 0.0)
    tele.tick(["u"], 0.05, 0.05)
    tele.update_feedback("joint_l6", -300 * RAD_PER_COUNT)
    out = tele.tick([], 0.06, 0.01)
    joint = tele.arms["left"].joints[4]
    assert joint.faulted
    assert "飛んだ" in out.message
    stuck = tele.tick(["u"], 0.1, 0.0)
    assert joint.goal == int(joint.actual)
    assert "切り離し" in stuck.message
    tele.tick(["0"], 0.2, 0.0)
    reenabled = tele.tick(["u"], 0.3, 0.0)
    assert not joint.faulted
    assert any(t.value == 1 and t.topic == "/left/arm/torque_enable" for t in reenabled.torque)
    moved = tele.tick(["u"], 0.5, 0.0)
    assert joint.goal == 8
    assert _arm(moved, "/left/arm/joint_command") is not None


def test_zero_and_quit_release_both_arms():
    tele = Teleop()
    _center(tele)
    out = tele.tick(["0"], 0.0, 0.0)
    assert {(t.topic, t.value) for t in out.torque} == {
        ("/left/arm/torque_enable", 0),
        ("/right/arm/torque_enable", 0),
    }
    out = tele.tick(["q"], 0.1, 0.0)
    assert out.quit
    assert {t.value for t in out.torque} == {0}


def test_help_does_not_move_the_arm():
    tele = Teleop()
    _center(tele)
    out = tele.tick(["?"], 0.0, 0.0)
    assert "w/s" in out.banner
    assert out.commands == []


def test_bracket_changes_step_and_rate():
    tele = Teleop()
    _center(tele)
    out = tele.tick(["]"], 0.0, 0.0)
    assert tele.step == 13
    assert tele.rate == 350
    assert "13" in out.message
    tele.tick(["["], 0.1, 0.0)
    assert tele.step == 8
    assert tele.rate == 250


def test_rom_cap_matches_measured_half_range():
    tele = Teleop(lead=5000, rate=5000, step=200)
    _center(tele)
    # joint0 の片側は 920。lead を広げてもそこを超えない。
    now = 0.0
    tele.tick(["w"], now, 0.0)
    for _ in range(30):
        now += 0.1
        tele.tick(["w"], now, 0.1)
    assert tele.arms["left"].joints[0].goal <= 920
    assert tele.arms["left"].joints[0].goal == 920
