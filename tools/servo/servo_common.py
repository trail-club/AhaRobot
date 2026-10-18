"""軌道試験ツールで共通に使う入力検査・時刻変換・日本語グラフ設定。"""

import argparse
import math

NS_PER_S = 1_000_000_000


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("0より大きい有限の値を指定してください")
    return number


def stamp_to_ns(stamp):
    """ROSのheader時刻（sec, nanosec）をナノ秒の整数にする。"""
    return stamp.sec * NS_PER_S + stamp.nanosec


def japanese_pyplot():
    """日本語フォントを設定した、画面表示不要の描画機能を返す。"""
    # --helpやCSVの検査では、描画ライブラリを読み込まない。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # コンテナのNotoフォント、macOS、Windowsの順に日本語フォントを探す。
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        "Noto Sans CJK JP",
        "IPAexGothic",
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
    )
    font = next((name for name in candidates if name in available), None)
    if font is None:
        raise ValueError(
            "日本語フォントがありません。fonts-noto-cjkを導入するか、開発イメージを再ビルドしてください"
        )
    plt.rcParams["font.family"] = font
    return plt
