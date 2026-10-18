"""関節角度の対応付け・欠損記録・タイムアウトと、ROS通信を検証する。"""

import csv
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

from record_joint_states import JointRecorder

SCRIPT = Path(__file__).with_name("record_joint_states.py")


class JointRecorderTest(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        self.recorder = JointRecorder(self.stream, ["joint_r2", "joint_r3"], 10)

    def rows(self):
        return list(csv.DictReader(io.StringIO(self.stream.getvalue())))

    def test_names_units_and_times(self):
        problem = self.recorder.receive(
            ["joint_r3", "joint_r2"],
            [0.0, math.pi / 2],
            100,
            123,
            11,
            100000000999,
        )
        self.assertEqual(problem, "")
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["joint_name"], "joint_r2")
        self.assertEqual(rows[1]["joint_name"], "joint_r3")
        self.assertAlmostEqual(float(rows[0]["position_rad"]), math.pi / 2)
        self.assertAlmostEqual(float(rows[0]["position_deg"]), 90)
        self.assertEqual(rows[0]["elapsed_s"], "1")
        self.assertEqual(rows[0]["stamp_sec"], "100")
        self.assertEqual(rows[0]["stamp_nanosec"], "123")
        self.assertEqual(rows[0]["received_ros_ns"], "100000000999")

    def test_partial_updates_and_timeout(self):
        self.recorder.receive(["joint_r2"], [0.0], 1, 0, 11, 0)
        self.assertEqual(self.recorder.stale_joints(12, 2), ["joint_r3"])
        self.recorder.receive(["other"], [0.0], 2, 0, 12, 0)
        self.assertEqual(self.recorder.stale_joints(13, 2), ["joint_r2", "joint_r3"])
        self.assertEqual(len(self.rows()), 1)

    def test_invalid_and_old_data_do_not_reset_timeout(self):
        self.recorder.receive(["joint_r2"], [0.0], 1, 0, 10, 0)
        for names, positions, stamp in [
            (["joint_r2"], [], 2),
            (["joint_r2"] * 2, [0.0] * 2, 2),
            (["joint_r2"], [math.nan], 2),
            (["joint_r2"], [0.0], 1),
            (["joint_r2"], [0.0], 0),
        ]:
            with self.subTest(names=names, positions=positions, stamp=stamp):
                self.assertTrue(
                    self.recorder.receive(names, positions, stamp, 0, 11, 0)
                )
        rows = self.rows()
        self.assertEqual(len(rows), 2)  # 有効な角度1行と、欠損1行。
        self.assertTrue(math.isnan(float(rows[1]["position_rad"])))
        self.assertTrue(math.isnan(float(rows[1]["position_deg"])))
        self.assertEqual(self.recorder.count, 1)
        self.assertEqual(self.recorder.invalid_count, 1)
        self.assertEqual(self.recorder.last_received["joint_r2"], 10)
        self.assertIn("joint_r2", self.recorder.stale_joints(12, 2))

    def test_gap_duplicate_and_recovery(self):
        self.recorder.receive(["joint_r2"], [0.0], 1, 0, 10, 0)
        self.assertTrue(self.recorder.receive(["joint_r2"], [math.nan], 2, 0, 11, 0))
        # 欠損行と同じ時刻の有効値も、重複として除外する。
        self.assertTrue(
            self.recorder.receive(["joint_r2"], [math.pi / 2], 2, 0, 11.5, 0)
        )
        self.assertEqual(
            self.recorder.receive(["joint_r2"], [math.pi / 2], 3, 0, 12, 0), ""
        )
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        self.assertTrue(math.isnan(float(rows[1]["position_deg"])))
        self.assertAlmostEqual(float(rows[2]["position_deg"]), 90)
        self.assertEqual(self.recorder.count, 2)
        self.assertEqual(self.recorder.invalid_count, 1)
        self.assertEqual(self.recorder.last_stamp["joint_r2"], 3_000_000_000)
        self.assertEqual(self.recorder.last_received["joint_r2"], 12)
        self.assertNotIn("joint_r2", self.recorder.stale_joints(13, 2))

    def test_nonfinite_values_never_become_valid_reception(self):
        # ラジアン自体は有限でも、度への変換で無限大になる値を含める。
        for stamp, position in enumerate(
            (math.nan, math.inf, -math.inf, 1e308), start=1
        ):
            with self.subTest(position=position):
                self.assertTrue(
                    self.recorder.receive(["joint_r2"], [position], stamp, 0, 11, 0)
                )
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertTrue(math.isnan(float(row["position_rad"])))
            self.assertTrue(math.isnan(float(row["position_deg"])))
        self.assertEqual(self.recorder.count, 0)
        self.assertEqual(self.recorder.invalid_count, 4)
        self.assertNotIn("joint_r2", self.recorder.last_stamp)
        self.assertEqual(self.recorder.last_received["joint_r2"], 10)
        self.assertIn("joint_r2", self.recorder.stale_joints(12, 2))

    def test_invalid_nanosecond_does_not_change_recording(self):
        for nanosec in (-1, 1_000_000_000):
            with self.subTest(nanosec=nanosec):
                self.assertTrue(
                    self.recorder.receive(["joint_r2"], [0.0], 1, nanosec, 11, 0)
                )
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.recorder.count, 0)
        self.assertEqual(self.recorder.invalid_count, 0)
        self.assertEqual(self.recorder.last_seen_stamp, {})
        self.assertEqual(self.recorder.last_received["joint_r2"], 10)


class CliTest(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )

    def test_help(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--joints", "--output"):
            self.assertIn(option, result.stdout)

    def test_no_data_and_no_overwrite(self):
        try:
            import rclpy  # noqa: F401
            from sensor_msgs.msg import JointState  # noqa: F401
        except ModuleNotFoundError as error:
            if error.name.split(".")[0] not in ("rclpy", "sensor_msgs"):
                raise
            self.skipTest("ROS 2がないためスキップしました")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "empty.csv"
            args = (
                "--topic",
                f"/test/cli_no_data/case_{uuid.uuid4().hex}",
                "--joints",
                "joint_r2",
                "--timeout",
                "0.2",
                "--duration",
                "1",
                "--output",
                output,
            )
            result = self.run_script(*args)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("新しい有効な値が届いていません", result.stdout)
            with output.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertIn("position_deg", reader.fieldnames or [])
                self.assertEqual(list(reader), [])  # ヘッダーだけが保存される。
            original = output.read_bytes()
            result = self.run_script(*args)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("CSVを作成できません", result.stderr)
            self.assertEqual(output.read_bytes(), original)


class RosRecordingTest(unittest.TestCase):
    """ROS 2の環境を読み込んだ開発コンテナ内で実行する通信テスト。"""

    def test_receive_then_disconnect(self):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
        except ModuleNotFoundError as error:
            if error.name in ("rclpy", "sensor_msgs", "sensor_msgs.msg"):
                self.skipTest("ROS 2がないため通信テストをスキップしました")
            raise  # ROS内部の依存関係が壊れている場合は失敗として扱う。

        context = Context()
        node = None
        try:
            rclpy.init(args=[], context=context)
            suffix = uuid.uuid4().hex
            node = rclpy.create_node(
                f"test_joint_state_source_{suffix}", context=context
            )
            # topic名の各部分は数字で始められないため、英字を前に付ける。
            topic = f"/test_record_joint_states/case_{suffix}/feedback"
            publisher = node.create_publisher(
                JointState, topic, qos_profile_sensor_data
            )
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "states.csv"
                log_path = Path(directory) / "recorder.log"
                # このテストでは起動・接続の余裕を持たせ、受信待ち上限を5秒にする。
                receive_timeout = 5
                with log_path.open("w", encoding="utf-8") as log_stream:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--topic",
                            topic,
                            "--joints",
                            "joint_r2",
                            "--timeout",
                            str(receive_timeout),
                            "--duration",
                            "30",
                            "--output",
                            str(output),
                        ],
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )

                    def diagnostic():
                        return log_path.read_text(encoding="utf-8", errors="replace")

                    def recorded_rows():
                        try:
                            text = output.read_text(encoding="utf-8")
                        except FileNotFoundError:
                            return []
                        # 書き込み途中の最終行は、次の読み取りで確認する。
                        if text and not text.endswith("\n"):
                            text = text.rpartition("\n")[0]
                        return list(csv.DictReader(io.StringIO(text)))

                    def publish(angle_deg):
                        message = JointState()
                        message.header.stamp = node.get_clock().now().to_msg()
                        message.name = ["joint_r2"]
                        message.position = [math.radians(angle_deg)]
                        publisher.publish(message)

                    def publish_until_recorded(angle_deg, timeout):
                        deadline = time.monotonic() + timeout
                        while time.monotonic() < deadline:
                            self.assertIsNone(
                                process.poll(),
                                f"受信確認前に記録プロセスが終了しました。\n{diagnostic()}",
                            )
                            # 接続を待つ間も送信する。毎回新しい時刻を付ける。
                            publish(angle_deg)
                            rows = recorded_rows()
                            if (
                                publisher.get_subscription_count() > 0
                                and rows
                                and rows[-1].get("joint_name") == "joint_r2"
                                and math.isclose(
                                    float(rows[-1]["position_deg"]),
                                    angle_deg,
                                    abs_tol=1e-7,
                                )
                            ):
                                return
                            time.sleep(0.05)
                        self.fail(
                            f"{angle_deg:g}度のCSV記録を確認できませんでした。\n"
                            f"{diagnostic()}"
                        )

                    try:
                        publish_until_recorded(0, timeout=8)
                        for angle_deg in range(1, 20):
                            publish(angle_deg)
                            time.sleep(0.05)
                        # BEST_EFFORTでは欠落が起こり得るため、最後の値を確認するまで送る。
                        publish_until_recorded(19, timeout=5)
                        # ここから送信を止め、記録プロセス自身のタイムアウトを確認する。
                        try:
                            process.wait(timeout=receive_timeout + 3)
                        except subprocess.TimeoutExpired:
                            self.fail(
                                f"送信停止後も記録プロセスが終了しません。\n{diagnostic()}"
                            )
                        log = diagnostic()
                        self.assertEqual(process.returncode, 2, log)
                        self.assertIn("新しい有効な値が届いていません", log)
                        rows = recorded_rows()
                        # 配信行数ではなく、開始値と終了値の受信を検証する。
                        self.assertGreaterEqual(len(rows), 2)
                        self.assertAlmostEqual(float(rows[0]["position_deg"]), 0)
                        self.assertAlmostEqual(float(rows[-1]["position_deg"]), 19)
                        self.assertTrue(
                            all(row["joint_name"] == "joint_r2" for row in rows)
                        )
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=5)
        finally:
            if node is not None:
                node.destroy_node()
            if context.ok():
                context.shutdown()


if __name__ == "__main__":
    unittest.main()
