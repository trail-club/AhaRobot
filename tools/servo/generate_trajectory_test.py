"""実機に接続しない軌道計算の検証。unittestのdiscoverで実行する。"""

import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from generate_trajectory import RampProfile
from servo_common import japanese_pyplot

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RampProfileTest(unittest.TestCase):
    def test_extreme_inputs_and_long_duration(self):
        for values in [(-1e308, 1e308, 1, 1), (0, 1, 1e-320, 1)]:
            with self.assertRaises(ValueError):
                RampProfile(*values)
        profile = RampProfile(0, 1, 1, 1e-320)
        self.assertTrue(all(math.isfinite(v) for v in profile.at(profile.duration / 4)))

    def test_known_trapezoid(self):
        profile = RampProfile(0, 20, 10, 20)
        self.assertEqual(profile.shape, "trapezoidal")
        self.assertEqual(profile.duration, 2.5)
        for time_s, expected in [
            (0, (0, 0)),
            (0.5, (2.5, 10)),
            (2, (17.5, 10)),
            (2.5, (20, 0)),
        ]:
            self.assertEqual(profile.at(time_s), expected)

    def test_short_move_never_reaches_requested_speed(self):
        profile = RampProfile(0, 2, 10, 20)
        self.assertEqual(profile.shape, "triangular")
        self.assertAlmostEqual(profile.peak_velocity, math.sqrt(40))
        self.assertAlmostEqual(profile.at(profile.duration / 2)[0], 1)
        self.assertLess(profile.peak_velocity, profile.max_velocity)

    def test_reverse_limits_and_integral(self):
        profile = RampProfile(20, -10, 8, 4)
        rows = profile.samples(200)
        for (_, position, velocity), (_, next_position, _) in zip(rows, rows[1:]):
            self.assertTrue(-10 <= position <= 20)
            self.assertLessEqual(next_position, position)
            self.assertTrue(-8 <= velocity <= 0)
        distance = sum(
            (b[0] - a[0]) * (a[2] + b[2]) / 2 for a, b in zip(rows, rows[1:])
        )
        self.assertAlmostEqual(distance, -30, places=4)
        for a, b in zip(rows, rows[1:]):
            self.assertLessEqual(abs((b[2] - a[2]) / (b[0] - a[0])), 4 + 1e-8)

    def test_hold_and_zero_distance(self):
        profile = RampProfile(3, 3, 10, 20)
        self.assertEqual(profile.shape, "stationary")
        self.assertEqual(profile.samples(50), [(0, 3, 0)])
        self.assertEqual(profile.at(-1), (3, 0))
        self.assertEqual(profile.at(1), (3, 0))

    def test_endpoint_and_sampling(self):
        profile = RampProfile(0, 3, 7, 11)
        rows = profile.samples(50)
        self.assertEqual(rows[0], (0, 0, 0))
        self.assertEqual(rows[-1], (profile.duration, 3, 0))
        for a, b in zip(rows[:-1], rows[1:-1]):
            self.assertAlmostEqual(b[0] - a[0], 0.02)
        self.assertGreater(rows[-1][0], rows[-2][0])
        self.assertLessEqual(rows[-1][0] - rows[-2][0], 0.02)
        self.assertEqual(profile.at(profile.duration + 1), (3, 0))

    def test_phase_continuity(self):
        for end in (2, 20):
            profile = RampProfile(0, end, 10, 20)
            for boundary in (
                profile.acceleration_time,
                profile.acceleration_time + profile.cruise_time,
            ):
                before = profile.at(boundary - 1e-8)
                after = profile.at(boundary + 1e-8)
                for a, b in zip(before, after):
                    self.assertAlmostEqual(a, b, places=5)

    def test_reject_invalid_inputs(self):
        for values in [
            (0, 1, 0, 1),
            (0, 1, 1, -1),
            (math.nan, 1, 1, 1),
            (0, math.inf, 1, 1),
        ]:
            with self.assertRaises(ValueError):
                RampProfile(*values)
        profile = RampProfile(0, 20, 10, 20)
        for rate in (0, -1, math.nan, math.inf, 1e9):
            with self.assertRaises(ValueError):
                profile.samples(rate)

    def test_final_sample_is_not_duplicated(self):
        # 所要時間×周波数が丸め誤差で整数をわずかに超えても、終了時刻は1行だけ。
        for end, rate in [(20, 50), (3, 50), (0.3, 1000), (7, 30)]:
            profile = RampProfile(0, end, 10, 20)
            times = [row[0] for row in profile.samples(rate)]
            self.assertEqual(times.count(profile.duration), 1, (end, rate))
            self.assertTrue(all(b > a for a, b in zip(times, times[1:])))


class CliTest(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("generate_trajectory.py"))]
            + [str(arg) for arg in args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )

    def test_help(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--start-deg", "--end-deg", "--output-dir"):
            self.assertIn(option, result.stdout)

    def test_csv_and_png(self):
        try:
            japanese_pyplot()
        except ModuleNotFoundError as error:
            if not error.name.startswith("matplotlib"):
                raise
            self.skipTest("matplotlibがないためPNG生成をスキップしました")
        except ValueError as error:
            self.skipTest(str(error))
        cases = [(0, 20, "台形速度"), (0, 2, "三角速度"), (20, 0, "台形速度")]
        cases.append((3, 3, "移動なし"))
        with tempfile.TemporaryDirectory() as directory:
            for index, (start, end, label) in enumerate(cases):
                with self.subTest(start=start, end=end):
                    output = Path(directory) / f"case_{index}"
                    result = self.run_script(
                        "--start-deg",
                        start,
                        "--end-deg",
                        end,
                        "--max-velocity-deg-s",
                        10,
                        "--acceleration-deg-s2",
                        20,
                        "--output-dir",
                        output,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(label, result.stdout)
                    self.assertNotIn("Glyph", result.stderr)
                    with (output / "profile.csv").open(newline="") as stream:
                        reader = csv.DictReader(stream)
                        rows = [{k: float(v) for k, v in row.items()} for row in reader]
                    self.assertEqual(
                        reader.fieldnames, ["time_s", "position_deg", "velocity_deg_s"]
                    )
                    self.assertEqual(
                        rows[0], dict(time_s=0, position_deg=start, velocity_deg_s=0)
                    )
                    self.assertEqual(rows[-1]["position_deg"], end)
                    self.assertEqual(rows[-1]["velocity_deg_s"], 0)
                    png = (output / "profile.png").read_bytes()
                    self.assertEqual(png[:8], PNG_SIGNATURE)


if __name__ == "__main__":
    unittest.main()
