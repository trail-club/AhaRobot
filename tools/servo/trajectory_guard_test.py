"""可動域・状態更新・追従誤差・終点静止の監視を検証する。ROS・実機に接続しない。

実行例（tools/servo）:
    python3 -m unittest trajectory_guard_test -v
"""

import math
import unittest

from trajectory_guard import TrajectoryGuard

NAMES = [f"joint_r{i}" for i in range(2, 7)]


def make_guard(max_error=2, timeout=0.5):
    return TrajectoryGuard(NAMES, [-90] * 5, [90] * 5, timeout, max_error, 1)


def receive(guard, degrees, stamp, now=None, names=NAMES):
    """計測時刻stamp［秒］の角度［度］を、受信時刻now［秒］に届けたものとして渡す。"""
    now = stamp if now is None else now
    guard.receive(
        names,
        [math.radians(value) for value in degrees],
        round(stamp * 1e9),
        now,
        round(now * 1e9),
    )


class LimitTest(unittest.TestCase):
    def test_bounds_and_start(self):
        guard = make_guard()
        for value in (91, math.nan, math.inf):
            with self.assertRaises(ValueError):
                guard.check_positions([value, 0, 0, 0, 0])
        receive(guard, [0] * 5, 1)
        self.assertEqual(
            guard.check_tracking([0] * 5, 1, 10**9, starting=True), [0] * 5
        )
        with self.assertRaisesRegex(RuntimeError, "開始姿勢"):
            guard.check_tracking([2, 0, 0, 0, 0], 1, 10**9, starting=True)

    def test_limit_rounding_and_invalid_configuration(self):
        guard = TrajectoryGuard(NAMES, [-3] * 5, [3] * 5, 0.5, 2, 1)
        values = [math.degrees(math.radians(3))] * 5
        guard.check_positions(values)
        receive(guard, [3] * 5, 1)
        self.assertEqual(guard.snapshot(1, 10**9), values)
        with self.assertRaises(ValueError):
            guard.check_positions([3.00001] * 5)
        for minimum, timeout in [([-3], 0.5), ([-3] * 5, math.nan), ([-3] * 5, -1)]:
            with self.assertRaises(ValueError):
                TrajectoryGuard(NAMES, minimum, [3] * 5, timeout, 2, 1)

    def test_command_clock_reverse_is_rejected(self):
        guard = make_guard()
        guard.record_command(10**9, [0] * 5)
        for stamp in (10**9, 999_999_999):
            with self.assertRaises(RuntimeError):
                guard.check_command(stamp, [0] * 5)
            with self.assertRaises(RuntimeError):
                guard.record_command(stamp, [0] * 5)


class StateTest(unittest.TestCase):
    def setUp(self):
        self.guard = make_guard()

    def test_partial_frozen_and_reordered(self):
        receive(self.guard, [0] * 5, 1, names=NAMES[::-1])
        self.assertEqual(self.guard.snapshot(1.1, 1_100_000_000), [0] * 5)
        # 同じ計測時刻の再送は更新とみなさない。
        receive(self.guard, [0] * 5, 1, now=1.4)
        with self.assertRaises(RuntimeError):
            self.guard.snapshot(1.6, 1_400_000_000)
        receive(self.guard, [0], 1.6, names=NAMES[:1])
        with self.assertRaisesRegex(RuntimeError, "joint_r3"):
            self.guard.snapshot(1.6, 1_600_000_000)

    def test_duplicate_does_not_refresh_timeout(self):
        receive(self.guard, [0] * 5, 1)
        receive(self.guard, [0] * 5, 1, now=1.2)
        self.assertEqual(self.guard.received[NAMES[0]], 1)
        with self.assertRaises(RuntimeError):
            self.guard.snapshot(1.5, 1_500_000_000)

    def test_bad_data_latches_fault(self):
        for names, positions, stamp in [
            (NAMES, [math.nan] * 5, 1_000_000_000),
            (NAMES, [0.0], 1_000_000_000),
            (NAMES, [0.0] * 5, 0),
            (NAMES, [0.0] * 5, 2_000_000_000),
        ]:
            with self.subTest(positions=positions, stamp=stamp):
                guard = make_guard()
                guard.receive(names, positions, stamp, 10, 1_000_000_000)
                self.assertTrue(guard.fault)
                # 正常値が届いても、一度検知した異常は解除しない。
                guard.receive(NAMES, [0.0] * 5, 1_000_000_000, 10, 1_000_000_000)
                with self.assertRaises(RuntimeError):
                    guard.snapshot(10, 1_000_000_000)

    def test_rewind_with_fresh_timestamp(self):
        receive(self.guard, [0] * 5, 1)
        # 鮮度条件は満たすが、既に採用した時刻より1ナノ秒古い。
        self.guard.receive(NAMES, [0.0] * 5, 999_999_999, 1.01, 1_010_000_000)
        with self.assertRaisesRegex(RuntimeError, "時刻が逆行"):
            self.guard.snapshot(1.01, 1_010_000_000)

    def test_invalid_message_is_not_partially_applied(self):
        receive(self.guard, [0] * 5, 1)
        receive(self.guard, [1, math.nan, 0, 0, 0], 1.01)
        self.assertIsNotNone(self.guard.fault)
        self.assertEqual(self.guard.values["joint_r2"], 0)
        self.assertEqual(self.guard.stamps["joint_r2"], 10**9)


class TrackingTest(unittest.TestCase):
    def test_tracking_uses_measurement_time(self):
        guard = make_guard()
        receive(guard, [0] * 5, 1)
        guard.check_tracking([0] * 5, 1, 10**9, starting=True)
        guard.record_command(1_010_000_000, [0] * 5)
        guard.record_command(1_100_000_000, [10] * 5)
        # 遅れて届いた旧時刻の0度を、最新の10度指令と比較して誤判定しない。
        receive(guard, [0] * 5, 1.05, now=1.2)
        guard.check_aligned_tracking(1.2, 1_200_000_000)
        self.assertEqual(
            guard.reference_for(NAMES[0], 1_050_000_000), (1_010_000_000, 0)
        )
        receive(guard, [0] * 5, 1.15, now=1.21)
        with self.assertRaisesRegex(RuntimeError, "追従誤差"):
            guard.check_aligned_tracking(1.21, 1_210_000_000)


class SettleTest(unittest.TestCase):
    def setUp(self):
        self.guard = make_guard()
        receive(self.guard, [0] * 5, 1)

    def test_requires_fresh_stationary_measurements(self):
        self.guard.begin_settle([0] * 5, 1_050_000_000, 1, 1, 0.2)
        receive(self.guard, [0] * 5, 1.1)
        self.assertFalse(self.guard.settled(1.35, 1_350_000_000))
        receive(self.guard, [0] * 5, 1.35)
        self.assertTrue(self.guard.settled(1.35, 1_350_000_000))
        # 角度は許容差内でも、角度変化率が大きければ静止確認をやり直す。
        receive(self.guard, [0.5] * 5, 1.36)
        self.assertFalse(self.guard.settled(1.36, 1_360_000_000))

    def test_old_endpoint_sample_cannot_pass(self):
        receive(self.guard, [0] * 5, 1.1)
        self.guard.begin_settle([0] * 5, 1_200_000_000, 0.1, 1, 0.05)
        receive(self.guard, [0] * 5, 1.15, now=1.25)
        self.assertFalse(self.guard.settled(1.25, 1_250_000_000))

    def test_needs_new_updates_from_all_joints(self):
        self.guard.begin_settle([0] * 5, 1_100_000_000, 0.1, 1, 0.05)
        receive(self.guard, [0] * 5, 1.11)
        self.assertFalse(self.guard.settled(1.18, 1_180_000_000))
        receive(self.guard, [0], 1.19, names=NAMES[:1])
        self.assertFalse(self.guard.settled(1.19, 1_190_000_000))
        receive(self.guard, [0] * 5, 1.2)
        self.assertTrue(self.guard.settled(1.2, 1_200_000_000))

    def test_encoder_step_dither_is_stationary(self):
        # 実機のエンコーダ1ステップ（約0.088度）の揺れを100Hzで受信する。
        step = 360 / 4096
        self.guard.begin_settle([0] * 5, 1_000_000_000, 1, 1, 0.5)
        for k in range(1, 60):
            receive(self.guard, [step * (k % 2)] * 5, 1 + k * 0.01)
        self.assertTrue(self.guard.settled(1.59, 1_590_000_000))

    def test_slow_drift_is_not_stationary(self):
        # 許容差内でも、2度/秒で流れ続ける間は静止とみなさない。
        self.guard.begin_settle([0] * 5, 1_000_000_000, 1, 1, 0.2)
        for k in range(1, 40):
            receive(self.guard, [-0.8 + 0.02 * k] * 5, 1 + k * 0.01)
            self.assertFalse(self.guard.settled(1 + k * 0.01, 10**9 + k * 10**7))

    def test_leaving_tolerance_restarts_window(self):
        self.guard.begin_settle([0] * 5, 1_000_000_000, 0.5, 1, 0.2)
        for k in range(1, 30):
            receive(self.guard, [0] * 5, 1 + k * 0.01)
        receive(self.guard, [0.6] * 5, 1.3)
        receive(self.guard, [0] * 5, 1.31)
        self.assertFalse(self.guard.settled(1.31, 1_310_000_000))
        for k in range(32, 52):
            receive(self.guard, [0] * 5, k * 0.01 + 1)
        self.assertTrue(self.guard.settled(1.51, 1_510_000_000))


if __name__ == "__main__":
    unittest.main()
