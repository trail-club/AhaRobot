"""軌道試験の可動域・状態更新・追従誤差・終点静止を監視する（ROS非依存）。"""

from collections import deque
import math


def within_limits(value, low, high):
    """度とradの往復変換による丸めだけを許容する。"""
    return math.isfinite(value) and low - 1e-9 <= value <= high + 1e-9


class TrajectoryGuard:
    def __init__(self, names, minimum, maximum, timeout, max_error, start_tolerance):
        if (
            len(names) != 5
            or len(set(names)) != 5
            or len(minimum) != 5
            or len(maximum) != 5
        ):
            raise ValueError("重複しない5関節と、その下限・上限が必要です")
        if not all(
            math.isfinite(v) and v > 0 for v in (timeout, max_error, start_tolerance)
        ):
            raise ValueError("待ち時間と誤差上限には正の有限値が必要です")
        for low, high in zip(minimum, maximum):
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("可動域は有限値で、下限より上限を大きくしてください")
        self.names = tuple(names)
        self.minimum = tuple(minimum)
        self.maximum = tuple(maximum)
        self.timeout = timeout
        self.max_error = max_error
        self.start_tolerance = start_tolerance
        self.values = {}
        self.received = {}
        self.stamps = {}
        self.fault = None
        self.commands = deque()
        self.first_command_stamp = None
        self.initial_positions = None
        self.settle_config = None
        # 終点指令以降の、関節ごとの（計測時刻, 角度）。許容差を外れると空に戻す。
        self.settle_samples = {}

    def check_positions(self, positions):
        if len(positions) != len(self.names):
            raise ValueError("5関節分の角度が必要です")
        for name, value, low, high in zip(
            self.names, positions, self.minimum, self.maximum
        ):
            if not within_limits(value, low, high):
                raise ValueError(
                    f"{name}: 角度が可動域外または不正です"
                    f"（{value}度、範囲{low}〜{high}度）"
                )

    def receive(self, names, positions_rad, stamp_ns, received, ros_now_ns):
        """新しい状態を採用する。同じ計測時刻では受信待ち時間を延長しない。"""
        if self.fault or not set(names).intersection(self.names):
            return
        try:
            if len(names) != len(positions_rad) or len(names) != len(set(names)):
                raise ValueError("関節状態の配列長または関節名の重複が不正です")
            if not math.isfinite(received):
                raise ValueError("受信経過時刻が不正です")
            age = (ros_now_ns - stamp_ns) / 1e9
            if not -0.05 <= age < self.timeout:
                raise ValueError("関節状態の時刻が古いか、受信側より未来です")
            updates = []
            for name, position in zip(names, positions_rad):
                if name not in self.names:
                    continue
                index = self.names.index(name)
                value = math.degrees(position)
                if not within_limits(value, self.minimum[index], self.maximum[index]):
                    raise ValueError(f"{name}: 実測角度が可動域外または不正です")
                if name in self.stamps and stamp_ns < self.stamps[name]:
                    raise ValueError(f"{name}: 関節状態の時刻が逆行しました")
                if stamp_ns == self.stamps.get(name):
                    continue
                updates.append((name, value))
            # 一つでも不正なら、同じメッセージの一部だけを更新しない。
            for name, value in updates:
                self.values[name] = value
                self.stamps[name] = stamp_ns
                self.received[name] = received
                self.add_settle_sample(name, stamp_ns, value)
        except (ValueError, OverflowError, RuntimeError) as error:
            self.fault = str(error)

    def snapshot(self, now, ros_now_ns):
        if self.fault:
            raise RuntimeError(self.fault)
        missing = [
            name
            for name in self.names
            if name not in self.values
            or not 0 <= now - self.received[name] < self.timeout
            or not -0.05 <= (ros_now_ns - self.stamps[name]) / 1e9 < self.timeout
        ]
        if missing:
            raise RuntimeError(f"新しい関節状態がありません: {', '.join(missing)}")
        return [self.values[name] for name in self.names]

    def check_tracking(self, expected, now, ros_now_ns, starting=False):
        """明示した姿勢との比較。開始姿勢の確認にも使用する。"""
        self.check_positions(expected)
        actual = self.snapshot(now, ros_now_ns)
        threshold = self.start_tolerance if starting else self.max_error
        for name, target, measured in zip(self.names, expected, actual):
            if abs(target - measured) > threshold + 1e-9:
                phase = "開始姿勢の差" if starting else "追従誤差"
                raise RuntimeError(
                    f"{name}: {phase}が上限を超えました"
                    f"（{abs(target - measured):.3f}度 > {threshold:g}度）"
                )
        if starting:
            self.initial_positions = tuple(expected)
        return actual

    def check_command(self, stamp_ns, positions):
        """publish前に、指令の可動域と時刻の単調増加を確認する。"""
        self.check_positions(positions)
        if self.commands and stamp_ns <= self.commands[-1][0]:
            raise RuntimeError("指令のROS時刻が重複・逆行しました")

    def record_command(self, stamp_ns, positions):
        """publish済みの指令を保存する。計測時刻との比較用に前後の履歴を残す。"""
        self.check_command(stamp_ns, positions)
        if self.first_command_stamp is None:
            self.first_command_stamp = stamp_ns
            if self.initial_positions is None:
                self.initial_positions = tuple(positions)
        self.commands.append((stamp_ns, tuple(positions)))
        cutoff = stamp_ns - int((self.timeout + 1.0) * 1e9)
        # 保持範囲の直前の1点も残す。未配信・古いデータを推測して補わない。
        while len(self.commands) > 2 and self.commands[1][0] < cutoff:
            self.commands.popleft()

    def reference_for(self, name, stamp_ns):
        """計測時刻以前に送信した最後の指令を返す（補間なし）。"""
        index = self.names.index(name)
        if self.first_command_stamp is None or stamp_ns < self.first_command_stamp:
            if self.initial_positions is None:
                raise RuntimeError("開始姿勢の確認が完了していません")
            return None, self.initial_positions[index]
        for stamp, positions in reversed(self.commands):
            if stamp <= stamp_ns:
                return stamp, positions[index]
        raise RuntimeError("計測時刻に対応する指令履歴がありません")

    def check_aligned_tracking(self, now, ros_now_ns):
        """関節ごとの計測時刻に対応する送信履歴との誤差を監視する。"""
        actual = self.snapshot(now, ros_now_ns)
        for name, measured in zip(self.names, actual):
            _, expected = self.reference_for(name, self.stamps[name])
            error = abs(measured - expected)
            if error > self.max_error + 1e-9:
                raise RuntimeError(
                    f"{name}: 追従誤差が上限を超えました"
                    f"（{error:.3f}度 > {self.max_error:g}度）"
                )
        return actual

    def begin_settle(self, endpoint, stamp_ns, tolerance, max_speed, seconds):
        """最初の終点指令の時刻を基準に、到達・静止の確認を始める。"""
        self.check_positions(endpoint)
        if not all(math.isfinite(v) and v > 0 for v in (tolerance, max_speed, seconds)):
            raise ValueError("終点判定の誤差・角度変化率・時間には正の有限値が必要です")
        if tolerance > self.max_error:
            raise ValueError("終点許容差は追従誤差の上限以下にしてください")
        self.settle_config = (tuple(endpoint), stamp_ns, tolerance, max_speed, seconds)
        self.settle_samples = {name: deque() for name in self.names}

    def add_settle_sample(self, name, stamp_ns, value):
        """終点指令以降の計測値を、静止確認の時間窓に加える。"""
        if self.settle_config is None:
            return
        endpoint, start_ns, tolerance, _, seconds = self.settle_config
        if stamp_ns < start_ns:
            return
        samples = self.settle_samples[name]
        if abs(value - endpoint[self.names.index(name)]) > tolerance + 1e-9:
            samples.clear()
            return
        samples.append((stamp_ns, value))
        # 時間窓の始まりより前の点は1つだけ残し、窓の長さを判定できるようにする。
        window_start = stamp_ns - round(seconds * 1e9)
        while len(samples) > 1 and samples[1][0] <= window_start:
            samples.popleft()

    def settled(self, now, ros_now_ns):
        """全関節が終点指令以降、許容差内で確認時間以上静止したか確認する。

        エンコーダの1ステップ（約0.088度）の揺れで誤判定しないよう、隣り合う
        計測値の差ではなく、確認時間内の角度の変動幅で静止を判定する。
        """
        if self.settle_config is None:
            return False
        self.snapshot(now, ros_now_ns)
        _, _, _, max_speed, seconds = self.settle_config
        max_spread = max_speed * seconds
        for samples in self.settle_samples.values():
            # 時間が過ぎただけでは合格にしない。全関節の新しい計測値で確認する。
            if not samples or samples[-1][0] - samples[0][0] < round(seconds * 1e9):
                return False
            values = [value for _, value in samples]
            if max(values) - min(values) > max_spread + 1e-9:
                return False
        return True
