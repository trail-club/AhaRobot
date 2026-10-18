"""軌道送信・状態監視・異常時の中断とログ保存を検証する。実機には接続しない。

SenderTestはROS通信を模擬し、RosSenderTestはテスト専用topicで実際のROS通信を使う。
ROS 2または関節指令の型がない環境では、RosSenderTestをスキップする。
本テストの成功は、実際の制御器・実機の動作確認を意味しない。

実行例（tools/servo）:
    ROS_DOMAIN_ID=97 python3 -m unittest send_trajectory_test -v
"""

import csv
import io
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import send_trajectory
from generate_trajectory import RampProfile

try:
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from astra_controller_interfaces.msg import JointCommand
except ModuleNotFoundError as error:
    # ROS内部の依存関係の不具合は、スキップせずテスト失敗として扱う。
    if error.name.split(".")[0] not in (
        "rclpy",
        "sensor_msgs",
        "astra_controller_interfaces",
    ):
        raise
    rclpy = None

SCRIPT = Path(__file__).with_name("send_trajectory.py")


def arm_names(arm="right"):
    return [f"joint_{arm[0]}{i}" for i in range(2, 7)]


def sender_args(output, arm="right", **options):
    """送信スクリプトの引数。optionsで既定値を上書き・追加する（例: rate_hz="20"）。"""
    values = {
        "arm": arm,
        "joint": arm_names(arm)[0],
        "hold_deg": ["0"] * 5,
        "end_deg": "2",
        "max_velocity_deg_s": "10",
        "acceleration_deg_s2": "100",
        "min_deg": ["-90"] * 5,
        "max_deg": ["90"] * 5,
        "max_error_deg": "10",
        "output": str(output),
        **options,
    }
    args = []
    for key, value in values.items():
        args.append("--" + key.replace("_", "-"))
        args.extend(value if isinstance(value, list) else [str(value)])
    return args


def read_complete_csv(path):
    """書き込み途中の最終行を除いて読む。記録ファイルがまだなければ空を返す。"""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    if text and not text.endswith("\n"):
        text = text.rpartition("\n")[0]
    return list(csv.DictReader(io.StringIO(text)))


def stamp_ns(row):
    return int(row["stamp_sec"]) * 10**9 + int(row["stamp_nanosec"])


class CommandPositionsTest(unittest.TestCase):
    def test_keeps_other_joints(self):
        profile = RampProfile(0, 20, 10, 20)
        hold = [1, 2, 3, 4, 5]
        for index in range(5):
            result = send_trajectory.command_positions(
                profile, hold, index, profile.duration
            )
            self.assertEqual(len(result), 5)
            for i, value in enumerate(result):
                self.assertAlmostEqual(
                    math.degrees(value), 20 if i == index else hold[i]
                )


class SimulatedRos:
    """仮想時計・指令受信・関節状態を用意する。通信品質の試験には使わない。"""

    def __init__(self, mode, names):
        self.mode = mode
        self.names = names
        self.now = 0.0
        self.callback = None
        self.sent = []
        self.last_feedback = -1.0
        self.injected = False
        self.endpoint_stamp = None
        self.command_qos = None

    def stamp(self, seconds=None):
        ns = 10**9 + round((self.now if seconds is None else seconds) * 1e9)
        return SimpleNamespace(sec=ns // 10**9, nanosec=ns % 10**9)

    def publish(self, message):
        self.sent.append((self.now, tuple(message.position_cmd)))
        if math.isclose(math.degrees(message.position_cmd[0]), 2, abs_tol=1e-7):
            if self.endpoint_stamp is None:
                self.endpoint_stamp = message.header.stamp

    def spin(self, node, timeout_sec):
        self.now += max(timeout_sec, 0.001)
        if self.mode == "late" and len(self.sent) >= 2 and not self.injected:
            self.now += 0.08
            self.injected = True
        if self.mode == "interrupt" and len(self.sent) >= 2 and not self.injected:
            self.injected = True
            raise KeyboardInterrupt
        if self.callback is None or self.now - self.last_feedback < 0.01:
            return
        self.last_feedback = self.now
        sample_time = self.now - (0.03 if self.mode == "delayed" else 0)
        positions = (0.0,) * 5
        for sent_at, values in self.sent:
            if sent_at <= sample_time:
                positions = values
        if self.mode == "endpoint_offset" and self.endpoint_stamp is not None:
            positions = (math.radians(3),) + positions[1:]
        stamp = self.stamp(sample_time)
        if self.mode == "old_endpoint" and self.endpoint_stamp is not None:
            ns = self.endpoint_stamp.sec * 10**9 + self.endpoint_stamp.nanosec - 1
            stamp = SimpleNamespace(sec=ns // 10**9, nanosec=ns % 10**9)
        self.callback(
            SimpleNamespace(
                name=self.names, position=positions, header=SimpleNamespace(stamp=stamp)
            )
        )

    def other_publishers(self):
        if self.mode == "other_publisher":
            return 1
        return 1 if self.mode == "intruder" and len(self.sent) >= 2 else 0

    def modules(self):
        def module(name, **attrs):
            result = ModuleType(name)
            result.__dict__.update(attrs)
            return result

        def count():
            return 0 if self.mode == "disconnect" and len(self.sent) >= 2 else 1

        def create_publisher(message_type, topic, qos):
            self.command_qos = qos
            return SimpleNamespace(publish=self.publish, get_subscription_count=count)

        def subscribe(message_type, topic, callback, qos):
            self.callback = callback

        clock = SimpleNamespace(
            now=lambda: SimpleNamespace(
                nanoseconds=10**9 + round(self.now * 1e9), to_msg=self.stamp
            )
        )
        node = SimpleNamespace(
            get_clock=lambda: clock,
            create_publisher=create_publisher,
            create_subscription=subscribe,
            destroy_node=lambda: None,
            count_publishers=lambda topic: 1 + self.other_publishers(),
        )

        class JointCommand:
            def __init__(self):
                self.header = SimpleNamespace(stamp=None)

        class ExternalShutdownException(Exception):
            pass

        return {
            "rclpy": module(
                "rclpy",
                init=lambda **kw: None,
                ok=lambda: True,
                create_node=lambda name: node,
                spin_once=self.spin,
                shutdown=lambda: None,
            ),
            "rclpy.executors": module(
                "rclpy.executors", ExternalShutdownException=ExternalShutdownException
            ),
            "rclpy.qos": module(
                "rclpy.qos",
                QoSProfile=lambda **kw: SimpleNamespace(**kw),
                ReliabilityPolicy=SimpleNamespace(RELIABLE="RELIABLE"),
                qos_profile_sensor_data=object(),
            ),
            "rclpy.signals": module(
                "rclpy.signals", SignalHandlerOptions=SimpleNamespace(NO=0)
            ),
            "sensor_msgs": module("sensor_msgs"),
            "sensor_msgs.msg": module("sensor_msgs.msg", JointState=object),
            "astra_controller_interfaces": module("astra_controller_interfaces"),
            "astra_controller_interfaces.msg": module(
                "astra_controller_interfaces.msg", JointCommand=JointCommand
            ),
        }


class SenderTest(unittest.TestCase):
    """模擬ROSで送信処理を動かし、終了理由と記録を確認する。"""

    def run_sender(self, mode, arm="right"):
        world = SimulatedRos(mode, arm_names(arm))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "commands.csv"
            argv = ["send_trajectory.py"] + sender_args(
                output,
                arm,
                max_error_deg="0.01" if mode == "delayed" else "10",
                settle_tolerance_deg="0.005",
                settle_seconds="0.05",
                settle_timeout="0.3",
            )
            with (
                patch.dict(sys.modules, world.modules()),
                patch.object(sys, "argv", argv),
                patch.object(send_trajectory.time, "monotonic", lambda: world.now),
                patch.object(send_trajectory.time, "sleep", lambda seconds: None),
                patch.object(send_trajectory.signal, "signal", return_value=None),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                result = send_trajectory.main()
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            rows = read_complete_csv(output)
            feedback = read_complete_csv(output.with_suffix(".feedback.csv"))
            return result, summary, rows, feedback, world

    def test_normal_completion_and_reliable_commands(self):
        for arm in ("right", "left"):
            with self.subTest(arm=arm):
                result, summary, rows, feedback, world = self.run_sender("normal", arm)
                self.assertEqual(result, 0, summary)
                self.assertTrue(summary["trial_valid"])
                self.assertEqual(world.command_qos.reliability, "RELIABLE")
                self.assertTrue(any(row["event"] == "endpoint_hold" for row in rows))
                self.assertTrue(feedback)
                first_joint = f"{arm_names(arm)[0]}_rad"
                self.assertAlmostEqual(float(rows[-1][first_joint]), math.radians(2))
                trajectory = [row for row in rows if row["event"] == "trajectory"]
                for i, row in enumerate(trajectory[:-1]):
                    self.assertAlmostEqual(float(row["profile_time_s"]), i / 50)
                stamps = [stamp_ns(row) for row in trajectory]
                self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])))
                # 指令と実測の経過秒は同じ起動時基準。最初の指令は開始から1周期以内。
                start = summary["trajectory_start_elapsed_s"]
                first = float(trajectory[0]["elapsed_s"])
                self.assertTrue(start <= first < start + 1 / 50, (start, first))

    def test_delayed_measurements_complete_without_false_error(self):
        result, summary, _, feedback, _ = self.run_sender("delayed")
        self.assertEqual(result, 0, summary)
        errors = [
            abs(float(row["tracking_error_deg"]))
            for row in feedback
            if row["tracking_error_deg"]
        ]
        self.assertTrue(errors)
        self.assertLessEqual(max(errors), 0.01)

    def test_failures_set_reason_and_hold_request(self):
        # (模擬モード, 終了コード, 終了理由に含む文言, 保持指令を送るか)
        cases = [
            ("old_endpoint", 2, "終点到達・静止を確認できません", True),
            ("endpoint_offset", 2, "終点到達・静止を確認できません", True),
            ("late", 2, "送信遅延", True),
            ("disconnect", 2, "切断", False),
            ("other_publisher", 2, "他の指令送信元", False),
            ("intruder", 2, "他の指令送信元", True),
            ("interrupt", 130, "手動中断", True),
        ]
        for mode, code, reason, hold in cases:
            with self.subTest(mode=mode):
                result, summary, rows, _, _ = self.run_sender(mode)
                self.assertEqual(result, code, summary)
                self.assertFalse(summary["trial_valid"])
                self.assertIn(reason, summary["reason"])
                held = any(row["event"] == "abort_hold" for row in rows)
                self.assertEqual(held, hold, summary)
                if hold:
                    self.assertIn("停止保証なし", summary["stop_action"])


class CliTest(unittest.TestCase):
    """ROS接続前に終わる入力検査とファイル保護。ROS 2がなくても実行できる。"""

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
        for option in ("--arm", "--joint", "--output"):
            self.assertIn(option, result.stdout)

    def test_invalid_plan_rejected_before_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "commands.csv"
            for hold, end, message in [
                (["0"] * 5, "91", "可動域外"),
                (["0", "0", "0", "0", "91"], "1", "可動域外"),
                (["0"] * 5, "nan", "有限"),
            ]:
                with self.subTest(hold=hold, end=end):
                    result = self.run_script(
                        *sender_args(output, hold_deg=hold, end_deg=end)
                    )
                    self.assertEqual(result.returncode, 2)
                    # ROS未導入など、別の理由で終了コード2になっても合格にしない。
                    self.assertIn(message, result.stderr)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_does_not_overwrite_any_recording_file(self):
        for suffix in (".csv", ".feedback.csv", ".summary.json"):
            with self.subTest(
                suffix=suffix
            ), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "commands.csv"
                protected = output.with_suffix(suffix)
                original = b"existing recording: do not overwrite\n"
                protected.write_bytes(original)
                result = self.run_script(*sender_args(output))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("未使用の保存先", result.stderr)
                self.assertEqual(protected.read_bytes(), original)
                self.assertEqual(list(Path(directory).iterdir()), [protected])


# 別の原因による失敗を、対象の異常検出として数えない。
REASON_PATTERNS = {
    "drop": r"新しい関節状態|関節状態の時刻が古い",
    "partial": r"新しい関節状態.*joint_r3",
    "frozen": r"新しい関節状態|関節状態の時刻が古い",
    "error": r"joint_r2: 追従誤差",
    "other_error": r"joint_r6: 追従誤差",
    "pulse": r"joint_r2: 追従誤差",
    "malformed": r"関節状態の配列長",
    "range": r"joint_r2: 実測角度が可動域外",
    "rewind": r"joint_r2: 関節状態の時刻が逆行",
    "nan": r"joint_r2: 実測角度が可動域外または不正",
    "start": r"joint_r2: 開始姿勢",
    "interrupt": r"手動中断またはROS終了",
}
# 実測値の異常で中断し、最新実測による保持指令を送るモード。
HOLD_MODES = ("error", "other_error", "pulse", "interrupt")


class FaultInjector:
    """受信した指令をそのまま実測として返す模擬関節。4指令目以降に異常を注入する。"""

    def __init__(self, mode, node, publisher, feedback_path):
        self.mode = mode
        self.node = node
        self.publisher = publisher
        self.feedback_path = feedback_path
        self.commands = []
        self.frozen_stamp = None
        self.rewind_stamp_ns = None
        self.pulse_confirmed = False

    def recorded(self, predicate):
        return any(
            row.get("joint_name") == "joint_r2" and predicate(row)
            for row in read_complete_csv(self.feedback_path)
        )

    def publish(self, message_type):
        failing = len(self.commands) >= 4
        if self.mode == "drop" and failing:
            return
        msg = message_type()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = arm_names()
        msg.position = (
            list(self.commands[-1].position_cmd) if self.commands else [0.0] * 5
        )
        if self.mode == "start":
            msg.position[0] = math.radians(5)
        if failing and not self.inject(msg):
            return
        self.publisher.publish(msg)

    def inject(self, msg):
        """異常を書き込む。送信を見送る場合はFalseを返す。"""
        mode = self.mode
        if mode == "pulse":
            # 送信側CSVで受信を確認するまで、新しい時刻で20度を送り直し、その後は正常値に戻す。
            self.pulse_confirmed = self.pulse_confirmed or self.recorded(
                lambda row: math.isclose(float(row["position_deg"]), 20, abs_tol=1e-7)
            )
            msg.position = [0.0] * 5
            if not self.pulse_confirmed:
                msg.position[0] = math.radians(20)
        elif mode == "malformed":
            msg.position = msg.position[:1]
        elif mode == "range":
            msg.position[0] = math.radians(100)
        elif mode == "rewind":
            if self.rewind_stamp_ns is None:
                stamps = [
                    stamp_ns(row)
                    for row in read_complete_csv(self.feedback_path)
                    if row.get("joint_name") == "joint_r2"
                ]
                if not stamps:
                    return False
                # 送信済み時刻ではなく、送信側が記録した計測時刻を基準にする。
                self.rewind_stamp_ns = max(stamps) - 1
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(
                self.rewind_stamp_ns, 10**9
            )
        elif mode == "partial":
            msg.name, msg.position = msg.name[:1], msg.position[:1]
        elif mode in ("error", "other_error"):
            msg.position[0 if mode == "error" else 4] = math.radians(20)
        elif mode == "nan":
            msg.position[0] = math.nan
        elif mode == "frozen":
            self.frozen_stamp = self.frozen_stamp or msg.header.stamp
            msg.header.stamp = self.frozen_stamp
        return True


@unittest.skipIf(
    rclpy is None, "ROS 2または関節指令の型がないため通信テストをスキップしました"
)
class RosSenderTest(unittest.TestCase):
    """テスト専用topicを使って、実際のROS通信と子プロセスの終了を確認する。"""

    def test_faults_and_interrupt_save_logs(self):
        context = Context()
        executor = None
        try:
            rclpy.init(args=[], context=context)
            executor = SingleThreadedExecutor(context=context)
            for mode, reason_pattern in REASON_PATTERNS.items():
                with self.subTest(
                    mode=mode
                ), tempfile.TemporaryDirectory() as directory:
                    # topic名の各部分は数字で始められないため、英字を前に付ける。
                    suffix = "case_" + uuid.uuid4().hex
                    node = rclpy.create_node(f"guard_test_{suffix}", context=context)
                    try:
                        self.run_fault_case(
                            mode,
                            reason_pattern,
                            Path(directory),
                            f"/guard_test/{suffix}",
                            node,
                            executor,
                        )
                    finally:
                        node.destroy_node()
        finally:
            if executor is not None:
                executor.shutdown()
            if context.ok():
                context.shutdown()

    def run_fault_case(self, mode, reason_pattern, directory, topic, node, executor):
        output = directory / "commands.csv"
        log_path = directory / "sender.log"
        publisher = node.create_publisher(
            JointState, topic + "/states", qos_profile_sensor_data
        )
        injector = FaultInjector(
            mode, node, publisher, output.with_suffix(".feedback.csv")
        )
        # 実機のarm_nodeと同じくBEST_EFFORTで購読する（RELIABLEの送信側と接続できる）。
        node.create_subscription(
            JointCommand,
            topic + "/command",
            injector.commands.append,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        node.create_timer(0.01, lambda: injector.publish(JointState))
        args = sender_args(
            output,
            end_deg="6",
            max_velocity_deg_s="3",
            acceleration_deg_s2="6",
            # 送信遅延はSenderTestで検証する。ここでは負荷による遅れで別の理由にしない。
            rate_hz="10",
            max_lateness_seconds="0.1",
            topic=topic + "/command",
            state_topic=topic + "/states",
            state_timeout="1.0",
            max_error_deg="30" if mode in ("drop", "partial", "frozen") else "2",
        )
        # PIPEを満杯にしないよう、子プロセスのログは一時ファイルへ保存する。
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), *args],
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                interrupted = False
                deadline = time.monotonic() + 12
                while process.poll() is None and time.monotonic() < deadline:
                    rclpy.spin_once(node, executor=executor, timeout_sec=0.01)
                    if mode == "interrupt" and len(injector.commands) >= 4:
                        if not interrupted:
                            process.send_signal(signal.SIGINT)
                            interrupted = True
                log = log_path.read_text(encoding="utf-8", errors="replace")
                self.assertIsNotNone(
                    process.poll(),
                    f"制限時間内に送信プロセスが終了しませんでした。\n{log}",
                )
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)

        summary = json.loads(output.with_suffix(".summary.json").read_text("utf-8"))
        rows = read_complete_csv(output)
        recorded = read_complete_csv(injector.feedback_path)
        detail = f"{log}\n{summary}"
        self.assertEqual(process.returncode, 130 if mode == "interrupt" else 2, detail)
        self.assertEqual(summary["exit_code"], process.returncode, detail)
        self.assertFalse(summary["trial_valid"], detail)
        self.assertRegex(summary["reason"], reason_pattern, detail)
        self.assertEqual(summary["settings"]["state_topic"], topic + "/states")
        self.assertEqual(
            summary["trajectory_messages"],
            sum(row["event"] == "trajectory" for row in rows),
        )
        self.assertTrue(recorded, detail)
        if mode == "start":
            self.assertEqual(rows, [], detail)
            self.assertEqual(injector.commands, [], detail)
            return
        self.assertGreater(summary["trajectory_messages"], 0, detail)
        held = any(row["event"] == "abort_hold" for row in rows)
        self.assertEqual(held, mode in HOLD_MODES, detail)
        if mode in HOLD_MODES:
            self.assertEqual(rows[-1]["event"], "abort_hold", detail)
        if mode == "interrupt":
            self.assertTrue(interrupted, detail)
        if mode == "pulse":
            self.assertTrue(injector.pulse_confirmed, detail)
        if mode == "rewind":
            self.assertIsNotNone(injector.rewind_stamp_ns, detail)
            self.assertTrue(
                any(stamp_ns(row) == injector.rewind_stamp_ns for row in recorded),
                detail,
            )


if __name__ == "__main__":
    unittest.main()
