"""Hardware-in-the-loop test for the head pan/tilt servos (Feetech STS3215).

Skipped unless AHA_HARDWARE=1 (and when scservo_sdk is not installed).
Run from the repo root with the head powered and Serial Forwarding enabled:

    AHA_HARDWARE=1 uv run --no-project --with pyserial --with feetech-servo-sdk \\
      python -m unittest tests/perception/test_head_servo.py -v
"""

import os
import time
import unittest

try:
    from scservo_sdk import COMM_SUCCESS, SCS_MAKEWORD, PacketHandler, PortHandler
except ImportError:
    PortHandler = None


HARDWARE = os.environ.get("AHA_HARDWARE") == "1"
PORT = os.environ.get("AHA_HEAD_PORT", "/dev/ttyUSB0")
BAUD = int(os.environ.get("AHA_HEAD_BAUD", "115200"))

PAN_ID = 12
TILT_ID = 13
SERVO_IDS = (PAN_ID, TILT_ID)
SERVO_NAMES = {PAN_ID: "pan", TILT_ID: "tilt"}
MOVE_DEG = {PAN_ID: 30.0, TILT_ID: 20.0}

STEPS_PER_REV = 4096
POS_MIN, POS_MAX = 0, 4095

# Upper bound: Waveshare Servo Driver with ESP32 spec for ST servos (6-12.6V).
VOLT_RANGE = (10.0, 12.6)
TEMP_MAX = 60

TOLERANCE = 15
LOAD_ABORT = 800
LOAD_ABORT_COUNT = 3
SPEED = 400
ACC = 30
TORQUE_LIMIT = 500
MOVE_TIMEOUT = 3.0
POLL_INTERVAL = 0.05
RETRIES = 3

ADDR_MODE = 33
ADDR_TORQUE_ENABLE = 40
ADDR_ACC = 41
ADDR_GOAL_POS = 42
ADDR_GOAL_SPD = 46
ADDR_TORQUE_LIMIT = 48
ADDR_STATE = 56  # pos(2) speed(2) load(2), always read as one block
ADDR_VOLT = 62
ADDR_TEMP = 63


def signed(value, bits):
    return -(value & ((1 << bits) - 1)) if value & (1 << bits) else value


def deg_to_steps(deg):
    return round(deg * STEPS_PER_REV / 360.0)


# Commands outside 0..4095 wrap around and make the servo spin the wrong way.
def clamp_pos(pos):
    return max(POS_MIN, min(POS_MAX, pos))


@unittest.skipUnless(HARDWARE, "hardware test: set AHA_HARDWARE=1 to run")
@unittest.skipIf(
    PortHandler is None, "scservo_sdk (feetech-servo-sdk) is not installed"
)
class HeadServoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ph = PortHandler(PORT)
        try:
            opened = cls.ph.openPort() and cls.ph.setBaudRate(BAUD)
        except Exception as e:
            opened = False
            reason = repr(e)
        else:
            reason = "openPort/setBaudRate returned False"
        if not opened:
            raise AssertionError(
                f"{PORT} を {BAUD}bps で開けません ({reason})。"
                "USB 接続と権限 (dialout)、AHA_HEAD_PORT を確認してください"
            )
        cls.pk = PacketHandler(0)

    @classmethod
    def tearDownClass(cls):
        for sid in SERVO_IDS:
            cls._torque_off(sid)
        try:
            cls.ph.closePort()
        except Exception:
            pass

    @classmethod
    def _torque_off(cls, sid):
        try:
            cls.pk.write1ByteTxRx(cls.ph, sid, ADDR_TORQUE_ENABLE, 0)
        except Exception:
            pass

    @classmethod
    def _retry(cls, fn, *args):
        """Call an SDK read and return its data, or None after RETRIES failures."""
        for _ in range(RETRIES):
            try:
                data, result, _error = fn(cls.ph, *args)
            except (IndexError, TypeError):
                continue
            if result == COMM_SUCCESS:
                return data
            time.sleep(0.01)
        return None

    @classmethod
    def _write(cls, size, sid, addr, value):
        fn = cls.pk.write1ByteTxRx if size == 1 else cls.pk.write2ByteTxRx
        for _ in range(RETRIES):
            try:
                result, _error = fn(cls.ph, sid, addr, value)
            except (IndexError, TypeError):
                continue
            if result == COMM_SUCCESS:
                return True
            time.sleep(0.01)
        return False

    def _ping(self, sid):
        return self._retry(self.pk.ping, sid) is not None

    def _read1(self, sid, addr):
        return self._retry(self.pk.read1ByteTxRx, sid, addr)

    def _state(self, sid):
        """Return (pos, load) from a single 6-byte read at ADDR_STATE, or None."""
        d = self._retry(self.pk.readTxRx, sid, ADDR_STATE, 6)
        if d is None or len(d) < 6:
            return None
        return SCS_MAKEWORD(d[0], d[1]), signed(SCS_MAKEWORD(d[4], d[5]), 10)

    def _write_ok(self, size, sid, addr, value):
        self.assertTrue(
            self._write(size, sid, addr, value),
            f"ID{sid}: アドレス {addr} への書き込みに失敗しました",
        )

    def test_1_servos_respond(self):
        missing = [sid for sid in SERVO_IDS if not self._ping(sid)]
        self.assertFalse(
            missing,
            f"ID {missing} が ping に応答しません。12V 電源が入っているか、"
            "基板で Start Serial Forwarding を押したか確認してください",
        )

    def test_2_status_is_sane(self):
        for sid in SERVO_IDS:
            with self.subTest(servo=SERVO_NAMES[sid]):
                volt = self._read1(sid, ADDR_VOLT)
                temp = self._read1(sid, ADDR_TEMP)
                mode = self._read1(sid, ADDR_MODE)
                self.assertIsNotNone(volt, f"ID{sid}: 電圧を読めません")
                self.assertIsNotNone(temp, f"ID{sid}: 温度を読めません")
                self.assertIsNotNone(mode, f"ID{sid}: モードを読めません")
                v = volt / 10.0
                self.assertTrue(
                    VOLT_RANGE[0] <= v <= VOLT_RANGE[1],
                    f"ID{sid}: 電圧 {v:.1f}V が {VOLT_RANGE[0]}-{VOLT_RANGE[1]}V の範囲外",
                )
                self.assertLess(temp, TEMP_MAX, f"ID{sid}: 温度 {temp}C が高すぎます")
                self.assertEqual(
                    mode, 0, f"ID{sid}: モード {mode} (位置モード 0 ではない)"
                )

    def test_3_each_servo_moves_and_returns(self):
        for sid in SERVO_IDS:
            with self.subTest(servo=SERVO_NAMES[sid]):
                try:
                    self._move_and_return(sid)
                finally:
                    self._torque_off(sid)

    def _move_and_return(self, sid):
        state = self._state(sid)
        self.assertIsNotNone(state, f"ID{sid}: 開始位置を読めません")
        start = state[0]
        delta = deg_to_steps(MOVE_DEG[sid])

        self._write_ok(1, sid, ADDR_ACC, ACC)
        self._write_ok(2, sid, ADDR_GOAL_SPD, SPEED)
        self._write_ok(2, sid, ADDR_TORQUE_LIMIT, TORQUE_LIMIT)
        # Goal must equal the current position before torque on, or the servo jumps.
        self._write_ok(2, sid, ADDR_GOAL_POS, start)
        self._write_ok(1, sid, ADDR_TORQUE_ENABLE, 1)

        for target in (clamp_pos(start + delta), clamp_pos(start - delta), start):
            pos, err, peak = self._go_to(sid, target)
            print(
                f"\n  ID{sid} ({SERVO_NAMES[sid]}): target={target} reached={pos} "
                f"err={err} peak_load={peak}",
                end="",
                flush=True,
            )
            self.assertLessEqual(
                err, TOLERANCE, f"ID{sid}: 目標 {target} に届きません (誤差 {err})"
            )
            self.assertLess(
                peak, LOAD_ABORT, f"ID{sid}: 負荷のピーク {peak} が大きすぎます"
            )

    def _go_to(self, sid, target):
        """Command target and wait for it. Returns (pos, error, peak |load|)."""
        self._write_ok(2, sid, ADDR_GOAL_POS, target)
        peak = 0
        overload = 0
        pos = None
        deadline = time.monotonic() + MOVE_TIMEOUT
        while True:
            time.sleep(POLL_INTERVAL)
            state = self._state(sid)
            if state is None:
                self._torque_off(sid)
                self.fail(f"ID{sid}: 移動中に状態を読めなくなりました")
            pos, load = state
            peak = max(peak, abs(load))
            overload = overload + 1 if abs(load) > LOAD_ABORT else 0
            if overload >= LOAD_ABORT_COUNT:
                self._write(2, sid, ADDR_GOAL_POS, pos)
                self._torque_off(sid)
                self.fail(
                    f"ID{sid}: 負荷 {abs(load)} が {LOAD_ABORT_COUNT} 回連続で {LOAD_ABORT} 超え。"
                    f"pos={pos} で停止してトルクOFFしました (機械端や干渉を確認)"
                )
            err = abs(pos - target)
            if err <= TOLERANCE:
                return pos, err, peak
            if time.monotonic() >= deadline:
                self.fail(
                    f"ID{sid}: {MOVE_TIMEOUT:.0f} 秒以内に目標 {target} へ届きません "
                    f"(pos={pos}, 残り誤差 {err}, peak_load={peak})"
                )


if __name__ == "__main__":
    unittest.main()
