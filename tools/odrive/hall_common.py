"""hall_record.py の記録を解析するための共通処理。"""

from __future__ import annotations

import collections
import csv

# fw 0.5.6 decode_hall (hall_polarity = 0) の状態 -> 番号
CNT = {1: 0, 3: 1, 2: 2, 6: 3, 4: 4, 5: 5}


def load(outdir: str, name: str, steady_from: float, margin: float = 0.5):
    """定速区間 (steady_from + margin 以降) の時刻とホール状態を返す。"""
    T, H = [], []
    with open(f"{outdir}/{name}.csv") as f:
        for row in csv.DictReader(f):
            t = float(row["t"])
            if t >= steady_from + margin:
                T.append(t)
                H.append(int(row["hall"]))
    return T, H


def aux_rows(outdir: str, name: str, steady_from: float, margin: float = 0.5):
    """補助データ (shadow 等) を持つ行だけを返す。"""
    with open(f"{outdir}/{name}.csv") as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["shadow"] and float(r["t"]) >= steady_from + margin
        ]


def reference(T, H, half_ms: float = 1.5):
    """前後 half_ms の正規状態の多数決。オフライン解析用の参照系列。"""
    out = []
    j0 = j1 = 0
    n = len(T)
    cnt = collections.Counter()
    for i in range(n):
        while j1 < n and T[j1] <= T[i] + half_ms / 1000:
            if H[j1] in CNT:
                cnt[H[j1]] += 1
            j1 += 1
        while T[j0] < T[i] - half_ms / 1000:
            if H[j0] in CNT:
                cnt[H[j0]] -= 1
            j0 += 1
        prev = out[-1] if out else H[i]
        best = max(cnt.items(), key=lambda kv: kv[1]) if cnt else None
        if best is None or best[1] <= 0 or cnt.get(prev, 0) >= best[1]:
            out.append(prev)  # 同数なら直前を保つ (切り替わり付近のばたつき防止)
        else:
            out.append(best[0])
    return out


def fw_delta(new_cnt: int, cur_cnt: int) -> int:
    """fw 0.5.6 のカウント増分。差 3 は向きによらず +3 になる。"""
    d = (new_cnt - cur_cnt) % 6
    return d - 6 if d > 3 else d
