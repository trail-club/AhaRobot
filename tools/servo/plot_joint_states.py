#!/usr/bin/env python3
"""記録済みの関節角度CSVから日本語のPNGグラフを作ります。ROS接続は不要です。"""

import argparse
import csv
import math
from pathlib import Path

from servo_common import japanese_pyplot


def read_recording(source):
    """CSVを関節ごとにまとめる。非有限の角度は欠損（nan）として残す。"""
    series = {}
    invalid_count = 0
    valid_count = 0
    # Excelなどで付加されたUTF-8のBOMも受け付ける。
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"joint_name", "elapsed_s", "position_deg"}
        if not required.issubset(fields):
            raise ValueError("関節角度の記録CSVではありません（必要な列がありません）")
        if len(fields) != len(set(fields)):
            raise ValueError("CSVの列名が重複しています")
        for row in reader:
            prefix = f"CSVの{reader.line_num}行目: "
            if None in row:
                raise ValueError(prefix + "列数が見出しより多くなっています")
            joint = (row["joint_name"] or "").strip()
            try:
                elapsed = float(row["elapsed_s"])
                position = float(row["position_deg"])
            except (TypeError, ValueError) as error:
                raise ValueError(prefix + "時刻または角度が欠損・不正です") from error
            if not joint or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(prefix + "関節名または時刻が不正です")
            times, angles = series.setdefault(joint, ([], []))
            # 関節ごとの順序を確認する。異なる関節の行は混在してよい。
            if times and elapsed < times[-1]:
                raise ValueError(prefix + "経過時間が逆行しています")
            if math.isfinite(position):
                valid_count += 1
            else:
                position = math.nan
                invalid_count += 1
            times.append(elapsed)
            angles.append(position)
    if not valid_count:
        raise ValueError("CSVに有効な関節角度の記録がありません")
    return series, invalid_count


def check_output_path(source, output):
    """別名のシンボリックリンク・ハードリンクを含め、入力の上書きを防ぐ。"""
    same_path = source.resolve() == output.resolve()
    if not same_path and source.exists() and output.exists():
        same_path = source.samefile(output)
    if same_path:
        raise ValueError("入力CSVと出力先は別のファイルにしてください")


def plot_recording(source, output):
    """PNGを保存し、欠損として記録された角度の件数を返す。"""
    source, output = Path(source), Path(output)
    check_output_path(source, output)
    series, invalid_count = read_recording(source)
    plt = japanese_pyplot()
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    try:
        for joint, (times, angles) in series.items():
            # nanを除去せず渡すことで、欠損の前後を線で結ばない。
            axis.plot(times, angles, label=joint, marker=".", markersize=2)
        axis.set(
            title="受信した関節角度",
            xlabel="記録開始からの時間 [秒]",
            ylabel="角度 [度]",
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="png", dpi=150)
    finally:
        # 保存に失敗した場合も描画用のメモリを解放する。
        plt.close(fig)
    if invalid_count:
        print(f"注意: 不正な角度{invalid_count}件をグラフの欠損として表示しました")
    return invalid_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="record_joint_states.pyで保存したCSV")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="出力PNG（同名ファイルは上書き。ただし入力CSVは保護）",
    )
    args = parser.parse_args()
    try:
        plot_recording(args.csv, args.output)
    except (OSError, ValueError, ImportError, csv.Error) as error:
        parser.error(str(error))
    print(f"グラフ: {args.output}")


if __name__ == "__main__":
    main()
