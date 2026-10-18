#!/usr/bin/env python3
"""実機に接続せず、静止状態から動き始めて停止する目標角度・速度を計算します。"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from servo_common import japanese_pyplot


SHAPE_LABELS = {
    "stationary": "移動なし",
    "trapezoidal": "台形速度",
    "triangular": "三角速度",
}


@dataclass(frozen=True)
class RampProfile:
    """加速と減速の大きさが等しい軌道。角度は度、時間は秒で扱う。"""

    # 入力する角度・速度・加速度。速度は向きによらず正の大きさを指定する。
    start_deg: float
    end_deg: float
    max_velocity: float
    acceleration: float

    def __post_init__(self):
        values = (self.start_deg, self.end_deg, self.max_velocity, self.acceleration)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("角度・速度・加速度には有限の数値を指定してください")
        if self.max_velocity <= 0 or self.acceleration <= 0:
            raise ValueError("速度と加速度には0より大きい値を指定してください")
        if not math.isfinite(self.distance):
            raise ValueError("指定値が大きすぎて計算できません")
        if self.distance and self.peak_velocity == 0:
            raise ValueError("指定値が小さすぎて計算できません")
        if not math.isfinite(self.duration) or (self.distance and self.duration <= 0):
            raise ValueError("指定値から有限の動作時間を計算できません")

    @property
    def distance(self):
        return abs(self.end_deg - self.start_deg)

    @property
    def direction(self):
        return 1 if self.end_deg >= self.start_deg else -1

    @property
    def peak_velocity(self):
        # 短距離では上限速度に届かないため、距離から到達可能な速度を求める。
        return min(
            self.max_velocity, math.sqrt(self.distance) * math.sqrt(self.acceleration)
        )

    @property
    def acceleration_time(self):
        return self.peak_velocity / self.acceleration

    @property
    def cruise_time(self):
        if self.distance == 0:
            return 0.0
        return max(0.0, self.distance / self.peak_velocity - self.acceleration_time)

    @property
    def duration(self):
        return 2 * self.acceleration_time + self.cruise_time

    @property
    def shape(self):
        if self.distance == 0:
            return "stationary"
        return "trapezoidal" if self.cruise_time > 0 else "triangular"

    def at(self, time_s):
        """指定時刻の（角度［度］、速度［度/秒］）を返す。動作前後は静止。"""
        if not math.isfinite(time_s):
            raise ValueError("時刻には有限の数値を指定してください")
        if time_s <= 0:
            return self.start_deg, 0.0
        if time_s >= self.duration:
            return self.end_deg, 0.0
        ta = self.acceleration_time
        if time_s < ta:
            # 加速区間：移動角度 = 加速度 × 時間² ÷ 2
            velocity = self.acceleration * time_s
            distance = 0.5 * velocity * time_s
        elif time_s < ta + self.cruise_time:
            distance = self.peak_velocity * (time_s - 0.5 * ta)
            velocity = self.peak_velocity
        else:
            # 減速区間：終了までの残り時間から角度と速度を逆算する。
            remaining = self.duration - time_s
            velocity = self.acceleration * remaining
            distance = self.distance - 0.5 * velocity * remaining
        return self.start_deg + self.direction * distance, self.direction * velocity

    def samples(self, rate_hz):
        """一定周期で値を計算する。最後には正確な終了時刻の値も含める。"""
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("サンプル周波数には0より大きい有限の値を指定してください")
        count = self.duration * rate_hz
        if not math.isfinite(count) or count > 1_000_000:
            raise ValueError(
                "サンプル数が多すぎます。動作時間かサンプル周波数を減らしてください"
            )
        times = (i / rate_hz for i in range(math.ceil(count)))
        # 丸め誤差で終了時刻と同じ値が出ても、終了時刻の行は1つだけにする。
        rows = [(t, *self.at(t)) for t in times if t < self.duration]
        rows.append((self.duration, self.end_deg, 0.0))
        return rows


def save_plot(profile, rows, path):
    plt = japanese_pyplot()
    # サンプル間隔が粗くても、加速・一定速度・減速の切り替わりを描く。
    times = sorted(
        {row[0] for row in rows}
        | {profile.acceleration_time, profile.acceleration_time + profile.cruise_time}
    )
    values = [profile.at(t) for t in times]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    for axis, index, label in zip(axes, (0, 1), ("角度 [度]", "速度 [度/秒]")):
        axis.plot(
            times,
            [value[index] for value in values],
            marker="." if len(times) == 1 else None,
        )
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("時間 [秒]")
    fig.suptitle(
        f"目標の動き（実機接続なし）: {SHAPE_LABELS[profile.shape]}\n{profile.start_deg:g} → {profile.end_deg:g} 度 | 到達速度 {profile.peak_velocity:g} 度/秒 | 加速度 {profile.acceleration:g} 度/秒²"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="使い方を表示して終了")
    parser.add_argument("--start-deg", type=float, required=True, help="開始角度［度］")
    parser.add_argument("--end-deg", type=float, required=True, help="終了角度［度］")
    parser.add_argument(
        "--max-velocity-deg-s",
        type=float,
        required=True,
        help="速度の上限［度/秒］（正の値）",
    )
    parser.add_argument(
        "--acceleration-deg-s2",
        type=float,
        required=True,
        help="加速・減速の大きさ［度/秒²］（正の値）",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=50,
        help="1秒あたりのサンプル数［Hz］（既定: 50）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="CSVとグラフの保存先（同名ファイルは上書き）",
    )
    args = parser.parse_args()
    try:
        profile = RampProfile(
            args.start_deg,
            args.end_deg,
            args.max_velocity_deg_s,
            args.acceleration_deg_s2,
        )
        rows = profile.samples(args.rate_hz)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "profile.png"
    try:
        save_plot(profile, rows, plot_path)
    except ValueError as error:
        parser.error(str(error))
    except ImportError:
        parser.error(
            "グラフの作成にはmatplotlibが必要です（python3 -m pip install matplotlib）"
        )
    csv_path = args.output_dir / "profile.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "position_deg", "velocity_deg_s"))
        writer.writerows(rows)
    print(
        f"{SHAPE_LABELS[profile.shape]}: 所要時間={profile.duration:g} 秒、指定速度={profile.max_velocity:g} 度/秒、到達速度={profile.peak_velocity:g} 度/秒"
    )
    print(f"CSV: {csv_path}\nグラフ: {plot_path}")


if __name__ == "__main__":
    main()
