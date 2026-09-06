#!/usr/bin/env python3
"""キーボードで各関節を動かす。対向駆動の関節は 4 個まとめて動かす。

ファームを焼く前でも、制御基板をシリアル転送モードにしておけばホストから
直接動かせる (手順は README)。対向取付の符号は実測値
(docs/servo-bringup.md) をそのまま使う。符号が違うとサーボ同士が押し合うので、
機体を変えたら `teach_calibrate.py --verify` で測り直して JOINTS を更新すること。

キーは 1 回叩くと `--step` 分だけ動き、押しっぱなしのあいだは `--rate`
step/秒 で動き続ける。離すと 1 ループ (約 30ms) で止まる。端末のキーリピートは
そのままだと入力バッファに溜まり、離したあとも溜まった分だけ動き続けるので、
毎ループ標準入力を全部読み切ってから 1 回の指令にまとめている。

目標値は実位置から `--lead` step 以上は先行させない。これがないと、機械端や
過負荷で関節が動けないあいだもキーを押した分だけ目標が進み続け、戻すのに
同じ回数だけ逆キーを押す羽目になる。過負荷を検出したときは目標を実位置まで
引き戻すので、その場ですぐ逆へ動かせる。

Usage:  keyboard_teleop.py [--port /dev/ttyUSB0] [--step 20]
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

from scservo_sdk import (
    COMM_SUCCESS,
    SCS_HIBYTE,
    SCS_LOBYTE,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

R_ACC, R_GOAL_POS, R_GOAL_SPD, R_TORQUE, R_TORQUE_LIMIT = 41, 42, 46, 40, 48
R_POS, R_LOAD = 56, 60

BLOCK_COOLDOWN = 0.6  # 動けないと判定した向きを止めておく秒数
STALL_MOVE = 3  # これ以下しか動かなければ止まっているとみなす (step)
ORIGIN_MARGIN = 150  # エンコーダ原点にこれより近いサーボがある関節は動かさない
# 位置指令は 0-4095 しか出せないので、中点から片側に出せるのは高々 2047 step。
# 機構がそれ以上回っても指令できないため、可動量はここで頭打ちにする。
MAX_ROM = 1950
LOAD_SAMPLES = 3  # 負荷はこの回数連続で超えたときだけ効かせる
HOLD_GRACE = 0.15  # 最後のキー入力からこの秒数だけ動き続ける (キーリピート間隔より長く)


class Joint:
    """1 つの関節。dual 構成では複数サーボを符号付きで同時に動かす。"""

    def __init__(
        self, name: str, servos: dict[int, int], keys: tuple[str, str], limit: int
    ):
        self.name = name
        self.servos = servos  # {サーボID: 符号(+1/-1)}
        self.keys = keys
        # 機構が 1 回転を超えて回っても、位置指令は中点から片側 2047 step までしか
        # 出せない。ここで頭打ちにしておかないと範囲外の指令になる。
        self.rom = min(limit, MAX_ROM)  # 実測可動域の約半分
        self.lim_pos = limit  # 起動時にサーボの余裕から決め直す
        self.lim_neg = limit
        self.goal = 0  # 指令中のオフセット
        self.actual = 0  # 実位置から求めたオフセット
        self.start: dict[int, int] = {}
        self.blocked_dir = 0  # 過負荷で進めない向き (+1/-1/0)
        self.blocked_at = 0.0
        self.load = 0
        self.hold_dir = 0  # 押しっぱなしで動かしている向き
        self.hold_until = 0.0
        self.over = 0  # 負荷が連続で閾値を超えた回数
        self.stall_ref = 0  # 停止判定の基準にした実位置
        self.stall_at = 0.0


# 符号は実機の実測値 (|r| = 1.00)。limit は teach_calibrate.py で測った可動域の
# 半分から余裕を引いた値。どちらも docs/servo-bringup.md を参照。
# wrist12 は機構の可動域が 415 度あって 1 回転に収まらないので MAX_ROM で頭打ちに
# なる。両端の約 30 度ずつは位置モードでは指令できない。
JOINTS = [
    Joint("joint0", {4: +1, 5: -1, 6: -1, 7: +1}, ("w", "s"), 920),
    Joint("joint1", {8: +1, 9: -1, 10: -1, 11: +1}, ("e", "d"), 990),
    Joint("wrist12", {12: +1}, ("r", "f"), 2362),
    Joint("wrist13", {13: +1}, ("t", "g"), 1360),
    Joint("wrist14", {14: +1}, ("u", "j"), 1650),
    Joint("gripper", {15: +1}, ("y", "h"), 1310),
]

HELP = """
  w/s  joint0 (ID4-7 を 4 個同時)      r/f  wrist12
  e/d  joint1 (ID8-11 を 4 個同時)     t/g  wrist13
  u/j  wrist14                        y/h  gripper (ID15)
  [ ]  ステップ幅 -/+        space  その場で停止        0  トルクOFF
  ?    このヘルプ            q      終了 (トルクOFF)
"""


class Bus:
    def __init__(self, port: str, baud: int):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            sys.exit(f"ポートを開けません: {port}")
        self.ph.setBaudRate(baud)
        self.pk = PacketHandler(0)
        self.sync = GroupSyncWrite(self.ph, self.pk, R_GOAL_POS, 2)
        self.last_raw: dict[int, int] = {}
        self.last_cmd: dict[int, int] = {}

    def r2(self, i: int, a: int, tries: int = 4) -> int | None:
        for _ in range(tries):
            v, c, _ = self.pk.read2ByteTxRx(self.ph, i, a)
            if c == COMM_SUCCESS:
                return v
        return None

    def w1(self, i: int, a: int, v: int) -> bool:
        return self.pk.write1ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS

    def w2(self, i: int, a: int, v: int) -> bool:
        return self.pk.write2ByteTxRx(self.ph, i, a, v)[0] == COMM_SUCCESS

    def close(self) -> None:
        self.ph.closePort()


def signed(v: int, bits: int = 10) -> int:
    """Feetech は最上位ビットを符号に使う (load は 10bit)。"""
    return -(v & ((1 << bits) - 1)) if v & (1 << bits) else v


def wrapped(d: int) -> int:
    """0/4095 をまたいだ差分を [-2048, 2047] に畳む。"""
    return (d + 2048) % 4096 - 2048


def apply(bus: Bus, joints: list[Joint]) -> None:
    """関節の goal を各サーボの目標位置へ書く。

    SyncWrite なら応答待ちが無く、何個でも 1 パケットで送れる。個別書き込みだと
    1 回あたり往復 16ms 前後かかり、対向 4 個の関節では更新が 15Hz まで落ちて
    長押しの追従が目に見えて遅れる。
    """
    bus.sync.clearParam()
    for j in joints:
        for sid, sign in j.servos.items():
            # 範囲外を絶対に送らない。負の値を送ると SCS_LOBYTE/HIBYTE が
            # 0xFF.. を吐き、サーボが桁違いの位置と解釈して暴走する。
            v = max(0, min(4095, j.start[sid] + sign * j.goal))
            bus.last_cmd[sid] = v
            bus.sync.addParam(sid, [SCS_LOBYTE(v), SCS_HIBYTE(v)])
    bus.sync.txPacket()


def set_limits(j: Joint) -> None:
    """サーボの残り可動量から、この関節を動かせる量を決める。

    STS3215 の位置指令は 0-4095 の絶対値で、範囲外は出せない。原点近くに
    止まっているサーボがあると、その 1 個だけ動けずに残りと押し合って関節が
    暴れる。関節の可動量を「全サーボの余裕の最小値」に切り詰めて防ぐ。
    """
    pos = min(
        (4095 - j.start[sid]) if sign > 0 else j.start[sid]
        for sid, sign in j.servos.items()
    )
    neg = min(
        j.start[sid] if sign > 0 else (4095 - j.start[sid])
        for sid, sign in j.servos.items()
    )
    j.lim_pos = min(j.rom, pos)
    j.lim_neg = min(j.rom, neg)
    # 原点をまたぐと位置が 0 と 4095 の間で飛び、サーボが逆回りに全力で回る。
    # 原点のすぐ近くに止まっているサーボがある向きへは動かさない。
    if pos < ORIGIN_MARGIN:
        j.lim_pos = 0
    if neg < ORIGIN_MARGIN:
        j.lim_neg = 0


def tight_servo(j: Joint, positive: bool) -> int:
    """その向きで最初に頭打ちになるサーボID。警告表示用。"""

    def room(item: tuple[int, int]) -> int:
        sid, sign = item
        forward = sign > 0 if positive else sign < 0
        return 4095 - j.start[sid] if forward else j.start[sid]

    return min(j.servos.items(), key=room)[0]


def measure(bus: Bus, j: Joint, with_load: bool = True) -> None:
    """実位置と負荷を読む。全サーボ読むと遅いので関節あたり 1 個。"""
    sid, sign = next(iter(j.servos.items()))
    pos = bus.r2(sid, R_POS, tries=2)
    if pos is not None:
        bus.last_raw[sid] = pos
        j.actual = sign * wrapped(pos - j.start[sid])
    if with_load:
        load = bus.r2(sid, R_LOAD, tries=2)
        if load is not None:
            j.load = signed(load)


def drain_keys() -> list[str]:
    """溜まっている入力を全部読む。

    端末のキーリピートは 1 秒あたり 30 回ほど入力を送ってくる。1 ループで 1 文字
    しか読まないと入力バッファに溜まり続け、キーを離したあとも溜まった分だけ
    動き続ける (押している期間と動く期間がずれる)。毎ループ読み切ってから
    まとめて 1 回の指令にする。
    """
    keys = []
    while select.select([sys.stdin], [], [], 0)[0]:
        c = sys.stdin.read(1)
        if not c:
            break
        keys.append(c)
        if len(keys) > 200:  # 異常入力で回り続けないように
            break
    return keys


def clamp_goal(j: Joint, want: float, lead: int) -> int:
    """実位置からの先行量と関節の可動量で目標を制限する。"""
    want = max(j.actual - lead, min(j.actual + lead, want))
    return int(max(-j.lim_neg, min(j.lim_pos, want)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--step", type=int, default=20, help="1 回叩いたときの移動量 (step)")
    p.add_argument(
        "--rate", type=int, default=400, help="押しっぱなしのときの速度 (step/秒)"
    )
    p.add_argument(
        "--speed",
        type=int,
        default=0,
        help="サーボの速度上限。0 なら --rate の 1.5 倍を使う",
    )
    p.add_argument("--acc", type=int, default=30)
    p.add_argument("--torque-limit", type=int, default=500, help="0-1000")
    p.add_argument("--lead", type=int, default=80, help="目標が実位置を先行してよい量")
    p.add_argument(
        "--load-stop", type=int, default=700, help="この負荷が続いたら進行を止める"
    )
    p.add_argument(
        "--stall-time",
        type=float,
        default=0.5,
        help="目標が先行しているのに実位置が動かない状態がこの秒数続いたら止める",
    )
    p.add_argument("--max-offset", type=int, help="関節ごとの既定可動量を上書きする")
    p.add_argument("--debug-log", help="毎周の状態を CSV で書き出す (不具合調査用)")
    a = p.parse_args()

    # 速すぎると位置モードの対向 4 個が行き過ぎて押し合い、実位置が数十 step
    # 振動する。ランプ速度をわずかに上回る程度が滑らか。
    speed = a.speed if a.speed > 0 else max(300, int(a.rate * 1.5))

    if a.max_offset is not None:
        for j in JOINTS:
            j.rom = a.max_offset

    bus = Bus(a.port, a.baud)
    all_ids = [sid for j in JOINTS for sid in j.servos]

    # 現在位置を目標にしてからトルクを入れる。順序を逆にすると跳ねる。
    print("現在位置を読み出し中…")
    for j in JOINTS:
        for sid in j.servos:
            pos = bus.r2(sid, R_POS)
            if pos is None:
                bus.close()
                sys.exit(
                    f"ID{sid} の位置が読めません。シリアル転送が有効か確認してください。"
                )
            j.start[sid] = pos
    for sid in all_ids:
        bus.w1(sid, R_ACC, a.acc)
        bus.w2(sid, R_GOAL_SPD, speed)
        bus.w2(sid, R_TORQUE_LIMIT, a.torque_limit)
    apply(bus, JOINTS)
    for sid in all_ids:
        bus.w1(sid, R_TORQUE, 1)

    blocked_joints: list[str] = []
    print(f"{'関節':<9}{'+方向':>8}{'-方向':>8}   制限しているサーボ")
    for j in JOINTS:
        set_limits(j)
        note = ""
        if j.lim_pos < j.rom:
            note += f" +側: ID{tight_servo(j, True)}"
        if j.lim_neg < j.rom:
            note += f" -側: ID{tight_servo(j, False)}"
        if j.lim_pos == 0 or j.lim_neg == 0:
            note += "  ★原点に近すぎるため、この向きは無効"
            blocked_joints.append(j.name)
        print(f"{j.name:<9}{j.lim_pos:>8}{j.lim_neg:>8}  {note}")

    if blocked_joints:
        print(
            f"\n{', '.join(blocked_joints)} は片側に動かせません。"
            "\n該当サーボがエンコーダ原点 (0/4095 の境目) に止まっており、またぐと"
            "\n位置が飛んでサーボが逆回りに全力で回ります。"
            "\nトルクOFF のまま手で中央寄りへ動かすか、中点を校正し直してください。"
        )

    step, rate = a.step, a.rate
    torque_on = True
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    msg = "トルクON"
    by_key = {k: (j, +1 if k == j.keys[0] else -1) for j in JOINTS for k in j.keys}
    rr = 0  # 動かしていない関節を 1 周ずつ見るための巡回位置
    t_start = time.time()
    last = time.time()
    last_draw = 0.0

    def resync() -> None:
        """現在位置を新しい基準にして、指令の溜まりを捨てる。"""
        for j in JOINTS:
            for sid in j.servos:
                j.start[sid] = bus.r2(sid, R_POS) or j.start[sid]
            j.goal = j.actual = j.hold_dir = 0
            j.blocked_dir = 0
            set_limits(j)
        apply(bus, JOINTS)

    dbg = open(a.debug_log, "w") if a.debug_log else None
    if dbg:
        dbg.write("t,joint,goal,actual,cmd,raw,start,hold,blocked\n")

    print(HELP)
    try:
        tty.setcbreak(fd)
        while True:
            now = time.time()
            dt, last = now - last, now
            quit_requested = False

            for k in drain_keys():
                if k == "q":
                    quit_requested = True
                    break
                elif k == "?":
                    print(HELP)
                elif k in ("[", "]"):
                    delta = 5 if k == "]" else -5
                    step = max(1, min(200, step + delta))
                    rate = max(20, min(2000, rate + delta * 20))
                    if a.speed <= 0:  # 追従できる速度を保つ
                        speed = max(300, int(rate * 1.5))
                        for sid in all_ids:
                            bus.w2(sid, R_GOAL_SPD, speed)
                    msg = f"ステップ {step} / 速度 {rate}"
                elif k == " ":
                    for j in JOINTS:
                        j.goal, j.hold_dir = j.actual, 0
                    apply(bus, JOINTS)
                    msg = "停止"
                elif k == "0":
                    for sid in all_ids:
                        bus.w1(sid, R_TORQUE, 0)
                    torque_on = False
                    for j in JOINTS:
                        j.hold_dir = 0
                    msg = "トルクOFF (方向キーで再投入)"
                elif k in by_key:
                    j, d = by_key[k]
                    if not torque_on:
                        resync()
                        for sid in all_ids:
                            bus.w1(sid, R_TORQUE, 1)
                        torque_on = True
                        msg = "トルクON"
                        continue
                    if j.blocked_dir and (d > 0) == (j.blocked_dir > 0):
                        msg = f"{j.name}: 負荷 {j.load} でこの向きは停止中"
                        continue
                    if j.hold_dir != d:  # 押し始めは 1 step 分だけ即座に動かす
                        j.goal = clamp_goal(j, j.goal + d * step, a.lead)
                        j.over = 0
                        j.stall_ref, j.stall_at = j.actual, now
                    j.hold_dir = d
                    j.hold_until = now + HOLD_GRACE
                    msg = f"{j.name} goal={j.goal:+d}"
            if quit_requested:
                break

            # 押しっぱなしのあいだだけ動かし続ける。離せば HOLD_GRACE 後に止まる。
            moving = []
            for j in JOINTS:
                if j.hold_dir and now < j.hold_until and torque_on:
                    j.goal = clamp_goal(j, j.goal + j.hold_dir * rate * dt, a.lead)
                    moving.append(j)
                elif j.hold_dir and now >= j.hold_until:
                    j.hold_dir = 0
            if moving:
                apply(bus, JOINTS)

            if torque_on:
                # 動かしている関節は毎周、それ以外は 1 個ずつ巡回して見る
                targets = moving or [JOINTS[rr % len(JOINTS)]]
                rr += 1
                for j in targets:
                    active = j in moving
                    measure(bus, j, with_load=active)
                    if dbg:
                        sid = next(iter(j.servos))
                        dbg.write(
                            f"{now - t_start:.3f},{j.name},{j.goal},{j.actual},"
                            f"{bus.last_cmd.get(sid)},{bus.last_raw.get(sid)},{j.start[sid]},"
                            f"{j.hold_dir},{j.blocked_dir}\n"
                        )
                    if not active:
                        j.over = 0
                        j.stall_ref, j.stall_at = j.actual, now
                        if j.blocked_dir and now - j.blocked_at > BLOCK_COOLDOWN:
                            j.blocked_dir = 0
                        continue

                    # 本当に動けないかどうかは、負荷の大きさより「目標が先行して
                    # いるのに実位置が動かない」で見るほうが確実。負荷は加速中や
                    # 姿勢によって普通に跳ねるので、連続超過のときだけ効かせる。
                    pushing = abs(j.goal - j.actual) > a.lead * 0.6
                    if pushing and abs(j.actual - j.stall_ref) < STALL_MOVE:
                        stalled = now - j.stall_at > a.stall_time
                    else:
                        j.stall_ref, j.stall_at = j.actual, now
                        stalled = False
                    j.over = j.over + 1 if abs(j.load) > a.load_stop else 0

                    if stalled or j.over >= LOAD_SAMPLES:
                        # 溜まった指令をその場で捨てる。捨てないと逆へ動かすのに
                        # 先行ぶんを打ち消すだけのキー入力が要る。
                        j.blocked_dir = 1 if j.goal >= j.actual else -1
                        j.goal, j.hold_dir = j.actual, 0
                        j.blocked_at = now
                        j.over = 0
                        apply(bus, JOINTS)
                        why = "動かない" if stalled else f"load={j.load}"
                        msg = f"★{j.name}: {why} ので停止"
                    elif j.blocked_dir and now - j.blocked_at > BLOCK_COOLDOWN:
                        j.blocked_dir = 0

            if now - last_draw > 0.1:
                last_draw = now
                state = "  ".join(
                    f"{j.name}:{j.goal:+5d}/{j.actual:+5d}"
                    + ("!" if j.blocked_dir else ">" if j.hold_dir else " ")
                    for j in JOINTS
                )
                sys.stdout.write(f"\r\033[K{state}  step={step:<3} | {msg}")
                sys.stdout.flush()
            time.sleep(0.005)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        for sid in all_ids:
            bus.w1(sid, R_TORQUE, 0)
        bus.close()
        if dbg:
            dbg.close()
        print("\n全サーボ トルクOFF。終了しました。")


if __name__ == "__main__":
    main()
