"""CSVの検査・欠損の描画・入力保護を検証する。ROS接続は不要。"""

import contextlib
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import plot_joint_states as plotting


HEADER = "elapsed_s,joint_name,position_deg\n"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PlotTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "states.csv"
        self.output = Path(self.directory.name) / "graphs" / "states.png"

    def write_csv(self, body):
        self.source.write_text(body, encoding="utf-8")

    def pyplot_for_test(self):
        """フォントに依存しない描画テスト。日本語フォントは別途検証する。"""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as error:
            if error.name == "matplotlib":
                self.skipTest("matplotlibが未導入です")
            raise
        return plt

    def test_bom_names_units_and_extra_columns(self):
        self.write_csv(
            "\ufeffelapsed_s,joint_name,position_deg,position_rad\n"
            "0, joint_r2 ,90,1.5708\n1,joint_r2,100,1.7453\n"
            "0,joint_r3,5,0.0873\n1,joint_r3,6,0.1047\n"
        )
        series, invalid = plotting.read_recording(self.source)
        self.assertEqual(invalid, 0)
        self.assertEqual(
            series,
            {
                "joint_r2": ([0, 1], [90, 100]),
                "joint_r3": ([0, 1], [5, 6]),
            },
        )

    def test_nonfinite_values_become_gaps(self):
        self.write_csv(
            HEADER + "0,joint_r2,0\n1,joint_r2,nan\n"
            "2,joint_r2,inf\n3,joint_r2,-inf\n4,joint_r2,10\n"
        )
        series, invalid = plotting.read_recording(self.source)
        self.assertEqual(invalid, 3)
        times, angles = series["joint_r2"]
        self.assertEqual(times, [0, 1, 2, 3, 4])
        self.assertEqual((angles[0], angles[-1]), (0, 10))
        self.assertTrue(all(math.isnan(value) for value in angles[1:4]))

    def test_invalid_csv_is_rejected_before_drawing(self):
        bodies = {
            "空ファイル": "",
            "記録なし": HEADER,
            "列不足": "elapsed_s,joint_name\n0,joint_r2\n",
            "列名重複": "elapsed_s,joint_name,position_deg,position_deg\n",
            "余分な値": HEADER + "0,joint_r2,0,unexpected\n",
            "角度欠損": HEADER + "0,joint_r2\n",
            "角度空欄": HEADER + "0,joint_r2,\n",
            "数値でない角度": HEADER + "0,joint_r2,error\n",
            "関節名空欄": HEADER + "0,   ,0\n",
            "負の時刻": HEADER + "-1,joint_r2,0\n",
            "非有限時刻": HEADER + "nan,joint_r2,0\n",
            "時刻逆行": HEADER + "2,joint_r2,0\n1,joint_r2,10\n",
            "すべて欠損": HEADER + "0,joint_r2,nan\n1,joint_r2,inf\n",
        }
        for label, body in bodies.items():
            with self.subTest(label=label):
                self.write_csv(body)
                with patch.object(plotting, "japanese_pyplot") as load_plot:
                    with self.assertRaises(ValueError):
                        plotting.plot_recording(self.source, self.output)
                    load_plot.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_equal_times_are_allowed(self):
        self.write_csv(HEADER + "0,joint_r2,0\n0,joint_r2,1\n")
        series, _ = plotting.read_recording(self.source)
        self.assertEqual(series["joint_r2"], ([0, 0], [0, 1]))

    def test_plot_contains_gap_and_is_png(self):
        self.write_csv(
            HEADER + "0,joint_r2,0\n1,joint_r2,nan\n2,joint_r2,10\n"
            "0,joint_r3,5\n2,joint_r3,5\n"
        )
        plt = self.pyplot_for_test()
        fig, axis = plt.subplots()
        self.addCleanup(plt.close, fig)
        with patch.object(plotting, "japanese_pyplot", return_value=plt), patch.object(
            plt, "subplots", return_value=(fig, axis)
        ), contextlib.redirect_stdout(io.StringIO()) as log, warnings.catch_warnings():
            # このテストは欠損とPNG生成のみを検証する。
            warnings.filterwarnings("ignore", message=r"Glyph .* missing from .*font")
            self.assertEqual(plotting.plot_recording(self.source, self.output), 1)
        lines = {line.get_label(): line for line in axis.lines}
        self.assertEqual(set(lines), {"joint_r2", "joint_r3"})
        self.assertEqual(list(lines["joint_r2"].get_xdata()), [0, 1, 2])
        self.assertTrue(math.isnan(lines["joint_r2"].get_ydata()[1]))
        self.assertEqual(list(lines["joint_r3"].get_ydata()), [5, 5])
        self.assertIn("不正な角度1件", log.getvalue())
        self.assertEqual(self.output.read_bytes()[:8], PNG_SIGNATURE)
        self.assertFalse(plt.fignum_exists(fig.number))

    def test_input_is_protected_by_function(self):
        self.write_csv(HEADER + "0,joint_r2,0\n")
        original = self.source.read_bytes()
        for output in (self.source, self.source.parent / "." / self.source.name):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "別のファイル"):
                    plotting.plot_recording(self.source, output)
                self.assertEqual(self.source.read_bytes(), original)

    def test_input_links_are_protected(self):
        self.write_csv(HEADER + "0,joint_r2,0\n")
        original = self.source.read_bytes()
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                alias = self.source.with_name(kind + ".png")
                try:
                    if kind == "symlink":
                        alias.symlink_to(self.source)
                    else:
                        os.link(self.source, alias)
                except (OSError, NotImplementedError) as error:
                    self.skipTest(f"リンクを作成できない環境です: {error}")
                with self.assertRaisesRegex(ValueError, "別のファイル"):
                    plotting.plot_recording(self.source, alias)
                self.assertEqual(self.source.read_bytes(), original)

    def test_save_failure_closes_figure(self):
        self.write_csv(HEADER + "0,joint_r2,0\n")
        plt = self.pyplot_for_test()
        fig, axis = plt.subplots()
        self.addCleanup(plt.close, fig)
        with patch.object(plotting, "japanese_pyplot", return_value=plt), patch.object(
            plt, "subplots", return_value=(fig, axis)
        ), patch.object(fig, "savefig", side_effect=OSError("保存失敗")):
            with self.assertRaisesRegex(OSError, "保存失敗"):
                plotting.plot_recording(self.source, self.output)
        self.assertFalse(plt.fignum_exists(fig.number))

    def test_cli_help_and_input_protection(self):
        script = Path(plotting.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)
        self.write_csv(HEADER + "0,joint_r2,0\n")
        original = self.source.read_bytes()
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(self.source),
                "--output",
                str(self.source),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("別のファイル", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.source.read_bytes(), original)

    def test_real_japanese_font_rendering(self):
        """日本語フォントの実描画は、依存環境が揃っている場合だけ確認する。"""
        try:
            plt = plotting.japanese_pyplot()
        except ModuleNotFoundError as error:
            if error.name.startswith("matplotlib"):
                self.skipTest(f"描画に必要なモジュールがありません: {error.name}")
            raise
        except ValueError as error:
            if "日本語フォントがありません" in str(error):
                self.skipTest(str(error))
            raise
        self.write_csv(HEADER + "0,joint_r2,0\n1,joint_r2,10\n")
        with patch.object(
            plotting, "japanese_pyplot", return_value=plt
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(plotting.plot_recording(self.source, self.output), 0)
        self.assertFalse(any("Glyph" in str(item.message) for item in caught))
        self.assertEqual(self.output.read_bytes()[:8], PNG_SIGNATURE)


if __name__ == "__main__":
    unittest.main()
