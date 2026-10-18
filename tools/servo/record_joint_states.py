#!/usr/bin/env python3
"""ROSの関節状態topicを購読し、回転関節の角度をCSVへ記録します。

動作指令・校正・トルク変更は行いません。関節ごとに有効値の受信途絶を監視します。
"""

import argparse
import csv
import math
import time
from pathlib import Path

from servo_common import NS_PER_S, positive_float


class JointRecorder:
    """関節名で値を対応付け、関節ごとに新しい有効値の受信を監視する。"""

    def __init__(self, stream, joints, started):
        self.stream = stream
        self.joints = joints
        self.started = started
        self.last_received = dict.fromkeys(joints, started)
        self.last_stamp = {}  # 最後に受信した有効値の時刻。
        self.last_seen_stamp = {}  # 欠損値も含む時刻。重複・逆行の検出に使用。
        self.count = 0  # 有効な角度の行数。
        self.invalid_count = 0  # CSVに残した欠損行数。
        self.writer = csv.writer(stream)
        self.writer.writerow(
            (
                "elapsed_s",
                "received_ros_ns",
                "stamp_sec",
                "stamp_nanosec",
                "joint_name",
                "position_rad",
                "position_deg",
            )
        )
        stream.flush()

    def receive(self, names, positions, sec, nanosec, received, received_ros_ns):
        # 配列の順番は固定と仮定しない。対応付けできないメッセージは記録しない。
        if len(names) != len(positions) or len(set(names)) != len(names):
            return "関節名と角度の数が合わないか、関節名が重複しています"
        if not 0 <= nanosec < NS_PER_S:
            return "メッセージの時刻が不正です"
        stamp = sec * NS_PER_S + nanosec
        positions_by_name = dict(zip(names, positions))
        problems = []
        for joint in self.joints:
            if joint not in positions_by_name:
                continue
            if joint in self.last_seen_stamp and stamp <= self.last_seen_stamp[joint]:
                problems.append(f"{joint}: 時刻が重複または逆行しています")
                continue

            position = positions_by_name[joint]
            valid = math.isfinite(position)
            degrees = math.degrees(position) if valid else math.nan
            valid = valid and math.isfinite(degrees)
            # 非有限値はnanに統一して残す。描画側がこの場所で線を切ります。
            self.writer.writerow(
                (
                    received - self.started,
                    received_ros_ns,
                    sec,
                    nanosec,
                    joint,
                    position if valid else math.nan,
                    degrees if valid else math.nan,
                )
            )
            self.last_seen_stamp[joint] = stamp
            if not valid:
                self.invalid_count += 1
                problems.append(f"{joint}: 角度が非有限値のため欠損として記録しました")
                # 欠損が続いても、有効値の受信待ち時間を延長しません。
                continue
            self.last_stamp[joint] = stamp
            self.last_received[joint] = received
            self.count += 1
        self.stream.flush()
        return " / ".join(problems)

    def stale_joints(self, now, timeout):
        # ROS時刻が停止・巻き戻りしても、通信の途絶を検知できる時計を使う。
        return [
            joint for joint in self.joints if now - self.last_received[joint] >= timeout
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/joint_states",
        help="関節状態のtopic（既定: /joint_states）",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        required=True,
        help="記録する回転関節名（複数指定可）",
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=30,
        help="起動からの記録時間［秒］（既定: 30）",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=2,
        help="関節ごとの有効値受信待ち上限［秒］（既定: 2）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="保存するCSV（既存ファイルは上書きしない）",
    )
    args = parser.parse_args()
    if len(set(args.joints)) != len(args.joints):
        parser.error("関節名を重複させないでください")

    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
    except ImportError:
        parser.error(
            "ROS 2の環境が必要です。ROSの環境を読み込んだコンテナ内で実行してください"
        )

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("x", newline="", encoding="utf-8")
    except OSError as error:
        parser.error(f"CSVを作成できません: {error}")
    node = None
    result = 0
    try:
        rclpy.init(args=[])
        node = rclpy.create_node("record_joint_states")
        recorder = JointRecorder(stream, args.joints, time.monotonic())

        def callback(message):
            problem = recorder.receive(
                message.name,
                message.position,
                message.header.stamp.sec,
                message.header.stamp.nanosec,
                time.monotonic(),
                node.get_clock().now().nanoseconds,
            )
            if problem:
                node.get_logger().warning(problem, throttle_duration_sec=2.0)

        node.create_subscription(
            JointState, args.topic, callback, qos_profile_sensor_data
        )
        print(
            f"記録開始: {args.topic} → {args.output}（{', '.join(args.joints)}）",
            flush=True,
        )
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                now = time.monotonic()
                stale = recorder.stale_joints(now, args.timeout)
                if stale:
                    print(
                        f"記録中断: {args.timeout:g}秒間、新しい有効な値が届いていません: {', '.join(stale)}",
                        flush=True,
                    )
                    result = 2
                    break
                if now - recorder.started >= args.duration:
                    if any(joint not in recorder.last_stamp for joint in args.joints):
                        print(
                            "記録終了: 一度も有効値を受信できなかった関節があります",
                            flush=True,
                        )
                        result = 2
                    break
        except (KeyboardInterrupt, ExternalShutdownException):
            print("手動で記録を終了しました", flush=True)
        print(
            f"保存した行数: 有効 {recorder.count} / 欠損 {recorder.invalid_count} / CSV: {args.output}",
            flush=True,
        )
    finally:
        stream.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
