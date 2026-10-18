#!/usr/bin/env python3
"""台形速度の目標角度をROSで送信し、全5関節を監視します。

既定topicは模擬試験用です。送信先を実機topicへ変更すると実機を動かします。
指令CSV・実測CSV・終了理由JSONを保存します。停止要求は保持指令によるもので、
通信断や制御器の故障時の停止を保証するものではありません。
"""

import argparse
import csv
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from generate_trajectory import RampProfile
from servo_common import NS_PER_S, positive_float, stamp_to_ns
from trajectory_guard import TrajectoryGuard

# 送信周期が低い場合も、この間隔で状態を監視する。
POLL_SECONDS = 0.02
FEEDBACK_FIELDS = (
    "elapsed_s",
    "received_ros_ns",
    "stamp_sec",
    "stamp_nanosec",
    "joint_name",
    "position_rad",
    "position_deg",
    "reference_stamp_ns",
    "reference_deg",
    "tracking_error_deg",
)


def command_positions(profile, hold_deg, index, elapsed):
    """対象関節だけ軌道で置き換え、5関節をROSのrad単位で返す。"""
    positions = [math.radians(value) for value in hold_deg]
    positions[index] = math.radians(profile.at(elapsed)[0])
    return positions


# 正の有限値を取る引数: (名前, 既定値（Noneなら必須）, 説明)
POSITIVE_OPTIONS = (
    ("--max-velocity-deg-s", None, "速度の上限［度/秒］"),
    ("--acceleration-deg-s2", None, "加減速の大きさ［度/秒²］"),
    ("--max-error-deg", None, "全関節に適用する追従誤差の上限［度］"),
    ("--rate-hz", 50, "送信周波数［Hz］"),
    ("--wait-timeout", 5, "受信先と状態の待ち時間［秒］"),
    ("--state-timeout", 0.5, "状態の受信・計測時刻の許容経過秒"),
    ("--start-tolerance-deg", 1, "開始姿勢と保持指令の許容差［度］"),
    ("--settle-tolerance-deg", 1, "全関節の終点許容差［度］"),
    ("--settle-velocity-deg-s", 1, "静止とみなす変動幅÷確認時間の上限［度/秒］"),
    ("--settle-seconds", 0.5, "終点の許容差と角度変化率を満たし続ける時間［秒］"),
    ("--settle-timeout", 5, "終点到達・静止の確認待ち上限［秒］"),
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--arm", choices=("left", "right"), required=True, help="左腕または右腕")
    add("--joint", required=True, help="動かす関節名（例: joint_r2）")
    add("--end-deg", type=float, required=True, help="対象関節の終了角度［度］")
    for name, help_text in (
        ("--hold-deg", "関節2〜6の開始・保持角度［度］"),
        ("--min-deg", "関節2〜6の可動域下限［度］"),
        ("--max-deg", "関節2〜6の可動域上限［度］"),
    ):
        add(name, nargs=5, type=float, required=True, help=help_text)
    for name, default, help_text in POSITIVE_OPTIONS:
        if default is not None:
            help_text += f"（既定: {default}）"
        add(
            name,
            type=positive_float,
            default=default,
            required=default is None,
            help=help_text,
        )
    add(
        "--max-lateness-seconds",
        type=positive_float,
        help="送信の許容遅延［秒］（既定: 送信周期の半分、1周期以下）",
    )
    add("--topic", default="/test/arm/joint_command", help="指令の送信先topic")
    add("--state-topic", default="/test/arm/joint_states", help="実測の関節状態topic")
    add("--output", type=Path, required=True, help="送信記録CSV（上書き不可）")
    return parser


@dataclass
class Plan:
    """送信前に検査済みの試験条件。角度は度で持つ。"""

    names: list
    index: int
    profile: RampProfile
    guard: TrajectoryGuard
    endpoint: list


def make_plan(parser, args):
    """入力を検査して試験条件を作る。不正な入力はROS接続・ファイル作成の前に拒否する。"""
    period = 1.0 / args.rate_hz
    if not math.isfinite(period):
        parser.error("送信周波数から有限の周期を計算できません")
    if args.max_lateness_seconds is None:
        args.max_lateness_seconds = period * 0.5
    if args.max_lateness_seconds > period:
        parser.error("送信の許容遅延は1周期以下にしてください")
    if args.settle_seconds >= args.settle_timeout:
        parser.error("静止確認時間は終点の確認待ち上限より短くしてください")
    if args.settle_tolerance_deg > args.max_error_deg:
        parser.error("終点許容差は追従誤差の上限以下にしてください")
    names = [f"joint_{args.arm[0]}{number}" for number in range(2, 7)]
    if args.joint not in names:
        parser.error(f"対象関節は {', '.join(names)} から指定してください")
    if not all(math.isfinite(value) for value in args.hold_deg):
        parser.error("保持角度には有限値を指定してください")
    index = names.index(args.joint)
    endpoint = list(args.hold_deg)
    endpoint[index] = args.end_deg
    try:
        profile = RampProfile(
            args.hold_deg[index],
            args.end_deg,
            args.max_velocity_deg_s,
            args.acceleration_deg_s2,
        )
        guard = TrajectoryGuard(
            names,
            args.min_deg,
            args.max_deg,
            args.state_timeout,
            args.max_error_deg,
            args.start_tolerance_deg,
        )
        guard.check_positions(args.hold_deg)
        # 単調な片道軌道なので開始・終点の両方が範囲内なら軌道全体も範囲内。
        guard.check_positions(endpoint)
        # 軌道生成ツールと同じサンプル数の上限で、送信回数を検査する。
        profile.samples(args.rate_hz)
    except ValueError as error:
        parser.error(str(error))
    return Plan(names, index, profile, guard, endpoint)


def output_paths(parser, output):
    """指令CSV・実測CSV・summaryの保存先を返す。既存ファイルは上書きしない。"""
    paths = [
        output,
        output.with_suffix(".feedback.csv"),
        output.with_suffix(".summary.json"),
    ]
    if len(set(paths)) != 3 or any(path.exists() for path in paths):
        parser.error("出力CSVと付属ログには未使用の保存先を指定してください")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"保存先を作成できません: {error}")
    return paths


def load_ros(parser):
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from astra_controller_interfaces.msg import JointCommand
    except ImportError:
        parser.error(
            "ROS 2とastra_controller_interfacesを読み込んだ環境で実行してください"
        )
    return SimpleNamespace(
        rclpy=rclpy,
        ExternalShutdownException=ExternalShutdownException,
        QoSProfile=QoSProfile,
        ReliabilityPolicy=ReliabilityPolicy,
        qos_profile_sensor_data=qos_profile_sensor_data,
        SignalHandlerOptions=SignalHandlerOptions,
        JointState=JointState,
        JointCommand=JointCommand,
    )


class CsvLog:
    """1行ごとにflushするCSV。異常終了時も書けた行までを残す。"""

    def __init__(self, stream, header):
        self.stream = stream
        self.writer = csv.writer(stream)
        self.write(header)

    def write(self, row):
        self.writer.writerow(row)
        self.stream.flush()


class TrajectoryTrial:
    """1回の片道軌道試験。送信・監視・記録と、中断時の保持指令を受け持つ。"""

    def __init__(self, args, plan, ros, streams):
        command_stream, feedback_stream, self.summary_stream = streams
        self.args = args
        self.plan = plan
        self.guard = plan.guard
        self.ros = ros
        self.command_log = CsvLog(
            command_stream,
            (
                "elapsed_s",
                "profile_time_s",
                "stamp_sec",
                "stamp_nanosec",
                *[f"{name}_rad" for name in plan.names],
                "event",
            ),
        )
        self.feedback_log = CsvLog(feedback_stream, FEEDBACK_FIELDS)
        self.node = None
        self.publisher = None
        self.sent = {"trajectory": 0, "endpoint_hold": 0, "abort_hold": 0}
        self.max_lateness = 0.0
        # 指令・実測CSVの経過秒は、どちらも起動時を基準にする。
        self.origin = time.monotonic()
        self.started = None
        self.settle_started = None
        self.tracking_fault = None
        self.stop_requested = False
        self.phase = "開始前"
        self.result = 2
        self.reason = "初期化中に終了"
        self.stop_action = "未送信"

    def run(self):
        rclpy = self.ros.rclpy
        previous_sigint = None
        try:
            # Ctrl+CでROSを先に破棄せず、可能なら保持指令とログを残す。
            rclpy.init(args=[], signal_handler_options=self.ros.SignalHandlerOptions.NO)
            # C拡張の実行途中にKeyboardInterruptを投げず、Python側で中断する。
            previous_sigint = signal.signal(signal.SIGINT, self.request_stop)
            self.connect()
            self.wait_until_ready()
            endpoint_command = self.send_trajectory()
            self.hold_until_settled(*endpoint_command)
            self.result, self.reason = 0, "終点到達・静止を確認して完了"
            self.stop_action = "終点指令で終了"
        except (KeyboardInterrupt, self.ros.ExternalShutdownException):
            self.result, self.reason = 130, "手動中断またはROS終了"
        except (RuntimeError, ValueError, OSError) as error:
            self.result, self.reason = 2, str(error)
        finally:
            try:
                if self.result != 0:
                    self.abort_hold()
                json.dump(
                    self.summary(), self.summary_stream, ensure_ascii=False, indent=2
                )
                self.summary_stream.flush()
            finally:
                if self.node is not None:
                    self.node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                if previous_sigint is not None:
                    signal.signal(signal.SIGINT, previous_sigint)
        return self.result

    def connect(self):
        ros = self.ros
        self.node = ros.rclpy.create_node("send_trajectory")
        self.publisher = self.node.create_publisher(
            ros.JointCommand,
            self.args.topic,
            ros.QoSProfile(depth=1, reliability=ros.ReliabilityPolicy.RELIABLE),
        )
        self.node.create_subscription(
            ros.JointState,
            self.args.state_topic,
            self.on_state,
            ros.qos_profile_sensor_data,
        )

    def ros_now(self):
        return self.node.get_clock().now().nanoseconds

    def spin(self, timeout):
        self.ros.rclpy.spin_once(self.node, timeout_sec=timeout)

    def request_stop(self, signum, frame):
        self.stop_requested = True

    def check_interrupted(self):
        if self.stop_requested:
            raise KeyboardInterrupt
        if not self.ros.rclpy.ok():
            raise self.ros.ExternalShutdownException()

    def wait_until_ready(self):
        """受信先と全関節の状態が揃い、実測姿勢が開始姿勢に一致するまで待つ。"""
        print(
            f"受信先と5関節の状態を待っています: {self.args.topic} / {self.args.state_topic}",
            flush=True,
        )
        deadline = time.monotonic() + self.args.wait_timeout
        while True:
            self.check_interrupted()
            self.spin(POLL_SECONDS)
            if self.guard.fault:
                raise RuntimeError(self.guard.fault)
            if self.publisher.get_subscription_count() and len(self.guard.values) == 5:
                self.check_other_publishers()
                self.guard.check_tracking(
                    self.args.hold_deg, time.monotonic(), self.ros_now(), starting=True
                )
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "受信先または5関節の状態が揃わないため、送信せず終了します"
                )

    def send_trajectory(self):
        """開始時刻を基準に予定時刻の目標を送る。終点の指令と時刻を返す。"""
        profile, rate = self.plan.profile, self.args.rate_hz
        self.phase = "軌道送信中"
        self.started = time.monotonic()
        tick = 0
        while True:
            # 遅れた周期を飛ばして先の角度へ進めない。予定時刻の値を送信する。
            profile_time = min(tick / rate, profile.duration)
            self.wait_until(self.started + profile_time)
            positions = command_positions(
                profile, self.args.hold_deg, self.plan.index, profile_time
            )
            stamp = self.publish(positions, profile_time, "trajectory")
            if profile_time >= profile.duration:
                return positions, stamp
            tick += 1

    def hold_until_settled(self, positions, endpoint_stamp):
        """終点指令を繰り返し、全関節の到達・静止を新しい計測値で確認する。"""
        args = self.args
        self.guard.begin_settle(
            self.plan.endpoint,
            endpoint_stamp,
            args.settle_tolerance_deg,
            args.settle_velocity_deg_s,
            args.settle_seconds,
        )
        self.phase = "終点保持中"
        self.settle_started = time.monotonic()
        tick = 1
        while not self.wait_until(self.settle_started + tick / args.rate_hz):
            self.publish(positions, self.plan.profile.duration, "endpoint_hold")
            tick += 1

    def wait_until(self, deadline):
        """状態を監視しながら送信予定時刻まで待つ。

        終点保持中に到達・静止を確認した場合はTrueを返す。送信が予定より
        許容値を超えて遅れた場合は異常として扱う。
        """
        while True:
            self.check_interrupted()
            self.spin(0.0)
            self.check_interrupted()
            if self.tracking_fault:
                raise RuntimeError(self.tracking_fault)
            now, ros_now = time.monotonic(), self.ros_now()
            self.guard.check_aligned_tracking(now, ros_now)
            if self.publisher.get_subscription_count() == 0:
                raise RuntimeError(f"{self.phase}に受信先の切断を検知しました")
            self.check_other_publishers()
            if self.settle_started is not None:
                if self.guard.settled(now, ros_now):
                    return True
                if now - self.settle_started >= self.args.settle_timeout:
                    raise RuntimeError("終点到達・静止を確認できませんでした")
            remaining = deadline - now
            if remaining <= 0:
                break
            self.spin(min(remaining, POLL_SECONDS))
        lateness = max(0.0, time.monotonic() - deadline)
        self.max_lateness = max(self.max_lateness, lateness)
        if lateness > self.args.max_lateness_seconds:
            raise RuntimeError(
                f"{self.phase}の送信遅延が許容値を超えました（{lateness:.6f}秒）"
            )
        return False

    def check_other_publishers(self):
        """同じ指令topicに、このツール以外の送信元がないことを確認する。

        upstreamのik_node・teleop_web_node・moveit_relay_nodeも腕の指令topicへ
        送信するため、同時に動いていると試験の指令と混ざる。検出はROSの探索に
        依存し、起動直後の送信元を見落とす場合がある。
        """
        others = self.node.count_publishers(self.args.topic) - 1
        if others > 0:
            raise RuntimeError(
                f"{self.args.topic} に他の指令送信元が{others}件あります。"
                "ik_nodeなどを停止してから実行してください"
            )

    def publish(self, positions, profile_time, event):
        """5関節の指令（rad）を送り、指令CSVと監視用の履歴に残す。"""
        degrees = [math.degrees(value) for value in positions]
        message = self.ros.JointCommand()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = self.plan.names
        message.position_cmd = positions
        stamp = stamp_to_ns(message.header.stamp)
        self.guard.check_command(stamp, degrees)
        self.publisher.publish(message)
        # 保存中の例外でも、既に指令を送った事実を中断処理へ伝える。
        self.sent[event] += 1
        self.guard.record_command(stamp, degrees)
        self.command_log.write(
            (
                time.monotonic() - self.origin,
                profile_time,
                message.header.stamp.sec,
                message.header.stamp.nanosec,
                *positions,
                event,
            )
        )
        return stamp

    def on_state(self, message):
        now, ros_now = time.monotonic(), self.ros_now()
        stamp = message.header.stamp
        if not 0 <= stamp.nanosec < NS_PER_S:
            self.guard.fault = "関節状態のnanosecが不正です"
        stamp_ns = stamp_to_ns(stamp)
        # 不正値を含め、受信した生値を残す。採用の可否は監視側で判断する。
        for name, position in zip(message.name, message.position):
            if name in self.plan.names:
                self.feedback_log.write(
                    (
                        now - self.origin,
                        ros_now,
                        stamp.sec,
                        stamp.nanosec,
                        name,
                        position,
                        math.degrees(position),
                        *self.reference_columns(name, stamp_ns, position),
                    )
                )
        self.guard.receive(message.name, message.position, stamp_ns, now, ros_now)
        # 次の正常値で上書きされる前に、受信した瞬間の異常を保持する。
        if self.started is not None and self.tracking_fault is None:
            try:
                self.guard.check_aligned_tracking(now, ros_now)
            except RuntimeError as error:
                self.tracking_fault = str(error)

    def reference_columns(self, name, stamp_ns, position):
        """実測CSV用に、対応する指令の時刻・角度と追従誤差を返す。不明なら空欄。"""
        if self.guard.initial_positions is None:
            return "", "", ""
        try:
            reference_stamp, reference = self.guard.reference_for(name, stamp_ns)
        except RuntimeError:
            # 履歴不足は監視処理で異常として扱い、生値の保存は続ける。
            return "", "", ""
        if reference_stamp is None:
            reference_stamp = ""
        return reference_stamp, reference, math.degrees(position) - reference

    def abort_hold(self):
        """送信開始後の中断時、最新実測位置の保持指令を1回送る。停止は保証しない。"""
        if not self.sent["trajectory"] or self.node is None or not self.ros.rclpy.ok():
            return
        self.stop_action = "保持指令なし（状態不明または受信先なし）"
        try:
            actual = self.guard.snapshot(time.monotonic(), self.ros_now())
            if self.publisher.get_subscription_count():
                self.publish(
                    [math.radians(value) for value in actual], "", "abort_hold"
                )
                self.stop_action = "最新実測位置の保持指令を送信（到達・停止保証なし）"
                time.sleep(0.05)
        except (RuntimeError, ValueError, OSError, KeyboardInterrupt) as error:
            self.stop_action = f"保持指令を完了できません: {error}"

    def summary(self):
        return {
            "exit_code": self.result,
            "reason": self.reason,
            "stop_action": self.stop_action,
            "trajectory_messages": self.sent["trajectory"],
            "endpoint_hold_messages": self.sent["endpoint_hold"],
            "max_send_lateness_s": self.max_lateness,
            "trial_valid": self.result == 0,
            "tracking_reference": "計測ROS時刻以前にpublishした最後の指令（補間なし）",
            "limitations": [
                "publish成功は制御器が指令を適用したことの確認ではありません",
                "保持指令による中断は実機の停止を保証しません",
                "この試験だけでは関節の速度・加速度上限は確定しません",
            ],
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(self.args).items()
            },
            "elapsed_origin": "起動時（指令・実測CSV共通）",
            "trajectory_start_elapsed_s": None
            if self.started is None
            else self.started - self.origin,
        }


def main():
    parser = build_parser()
    args = parser.parse_args()
    plan = make_plan(parser, args)
    paths = output_paths(parser, args.output)
    ros = load_ros(parser)
    with ExitStack() as files:
        try:
            streams = [
                files.enter_context(path.open("x", newline="", encoding="utf-8"))
                for path in paths
            ]
        except OSError as error:
            parser.error(f"記録ファイルを作成できません: {error}")
        trial = TrajectoryTrial(args, plan, ros, streams)
        result = trial.run()
    print(
        f"{trial.reason} / 送信数: {trial.sent['trajectory']} / "
        f"{trial.stop_action} / ログ: {paths[2]}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
