#!/usr/bin/env python3
"""Interactive zero/sign calibration for one servo bus config.

  ros2 run aha_servo servo_calibrate.py                    # config/head.yaml
  ros2 run aha_servo servo_calibrate.py --config my_bus.yaml --port /dev/ttyUSB1
  python3 overlay_ws/src/aha_servo/scripts/servo_calibrate.py   # no ROS, needs pyyaml

Torque goes off on every joint of the config (aha_servo/joint_map.py format), then:
  1. Hold the robot in the zero pose (`zero_pose` in the yaml) and press
     Enter: zero = present steps of every joint.
  2. For each joint, move it toward + (`positive` in the yaml) and press
     Enter: sign = direction of the step change.
The result is printed and, after a y, written back into the yaml (comments
kept). Torque stays off. Do not run it while servo_trajectory_bridge.py
holds the same port.
"""

import argparse
import math
import os
import select
import sys
import termios

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
SOURCE_CONFIG = os.path.join(PKG_DIR, "config", "head.yaml")

try:
    import aha_servo  # noqa: F401
except ModuleNotFoundError:
    if os.path.isfile(os.path.join(PKG_DIR, "aha_servo", "__init__.py")):
        # Run from the source tree (or a symlink install): use the module next to it.
        sys.path.append(PKG_DIR)
    else:
        # Shell sourced before aha_servo was built: find it via the ament index.
        from ament_index_python.packages import get_package_prefix

        sys.path.append(
            os.path.join(
                get_package_prefix("aha_servo"),
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
            )
        )

from aha_servo.joint_map import load_file, update_yaml_text, wrap_steps  # noqa: E402
from aha_servo.sts import POS_MAX, POS_MIN, StsBus, StsError  # noqa: E402

MIN_MOVE_DEG = 5.0
REFRESH_S = 0.2


def default_config():
    """Installed config/head.yaml, else the source tree's (no ROS on the host)."""
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        pass
    else:
        try:
            return os.path.join(
                get_package_share_directory("aha_servo"), "config", "head.yaml"
            )
        except PackageNotFoundError:
            pass
    if os.path.isfile(SOURCE_CONFIG):
        return SOURCE_CONFIG
    sys.exit(
        "aha_servo config/head.yaml not found (ROS not sourced?): pass --config <yaml>"
    )


def flush_input():
    """Drop keys typed ahead (e.g. an extra Enter) so they cannot answer the next prompt."""
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def confirm(question):
    """Ask until an explicit y or n; EOF / Ctrl-C count as n."""
    while True:
        flush_input()
        try:
            answer = input(f"{question} [y/n] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  please type y or n")


def read_steps(bus, joints):
    """{name: steps or None} for every joint."""
    steps = {}
    for j in joints:
        try:
            steps[j.name] = bus.read_state(j.id)[0]
        except StsError:
            steps[j.name] = None
    return steps


def wait_enter(bus, joints, prompt):
    """Show live positions until Enter; return the steps read after it."""
    flush_input()
    print(prompt)
    if sys.stdin.isatty():
        # Raw fd reads, so no typed line hides in sys.stdin's buffer from select.
        fd = sys.stdin.fileno()
        while True:
            steps = read_steps(bus, joints)
            live = "  ".join(
                f"{n}={s if s is not None else '??'}" for n, s in steps.items()
            )
            print(f"\r  [Enter] {live}\033[K", end="", flush=True)
            if select.select([fd], [], [], REFRESH_S)[0]:
                if not os.read(fd, 1024):
                    raise EOFError
                break
    elif not sys.stdin.readline():
        raise EOFError
    print()
    steps = read_steps(bus, joints)
    unread = [n for n, s in steps.items() if s is None]
    if unread:
        raise StsError(f"cannot read {', '.join(unread)}")
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", help="bus config yaml (default: aha_servo config/head.yaml)"
    )
    parser.add_argument("--port", help="override the port in the yaml")
    parser.add_argument("--baud", type=int, help="override the baud rate in the yaml")
    args = parser.parse_args()

    path = os.path.realpath(args.config or default_config())
    cfg = load_file(path)
    port = args.port or cfg.port
    baud = args.baud or cfg.baud
    joints = cfg.joints
    print(f"config: {path}")
    print(f"bus:    {port} @ {baud}, joints: {', '.join(cfg.joint_names)}")

    try:
        bus = StsBus(port, baud, timeout=cfg.timeout)
    except (OSError, ValueError) as e:
        sys.exit(f"cannot open {port}: {e}")
    try:
        missing = [f"{j.name} (ID{j.id})" for j in joints if not bus.ping(j.id)]
        if missing:
            sys.exit(
                f"no reply from {', '.join(missing)}: check power, Serial Forwarding and baud"
            )
        for j in joints:
            bus.set_torque(j.id, False)
        print("torque OFF: the joints can be moved by hand.\n")

        zero_pose = cfg.zero_pose or "the zero pose (every joint at 0)"
        zeros = wait_enter(bus, joints, f"1) Hold {zero_pose}, then press Enter.")

        signs = {}
        for k, j in enumerate(joints, start=2):
            direction = j.positive or "the + direction"
            min_move = round(MIN_MOVE_DEG / 360.0 * j.steps_per_rev)
            while True:
                steps = wait_enter(
                    bus,
                    joints,
                    f"{k}) Move {j.name} {direction} by about 20-30 deg "
                    "(others stay put), then press Enter.",
                )
                delta = wrap_steps(steps[j.name] - zeros[j.name], j.steps_per_rev)
                if abs(delta) >= min_move:
                    break
                print(f"   moved only {delta} steps (< {min_move}); move it further.")
            signs[j.name] = 1 if delta > 0 else -1
            print(f"   {j.name}: {delta:+d} steps -> sign {signs[j.name]:+d}")
    except (KeyboardInterrupt, EOFError):
        print("\naborted, nothing written (torque stays off)")
        sys.exit(130)
    except StsError as e:
        sys.exit(f"communication error: {e} (torque stays off)")
    finally:
        bus.close()

    print("\nresult:")
    for j in joints:
        j.zero, j.sign = zeros[j.name], signs[j.name]
        lo, hi = j.step_range()
        warn = (
            ""
            if POS_MIN <= lo and hi <= POS_MAX
            else (
                f"  WARNING: limits need steps {lo}..{hi}, outside {POS_MIN}..{POS_MAX}"
            )
        )
        print(
            f"  {j.name}: zero {j.zero}, sign {j.sign:+d}  "
            f"(limits [{math.degrees(j.min):+.0f}, {math.degrees(j.max):+.0f}] deg "
            f"= steps {lo}..{hi}){warn}"
        )

    if not confirm(f"\nwrite to {path}?"):
        print(f"not written. To apply, set these keys under ros__parameters in {path}:")
        for j in joints:
            print(f"  {j.name}.zero: {j.zero}\n  {j.name}.sign: {j.sign}")
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for j in joints:
        text = update_yaml_text(text, j.name, {"zero": j.zero, "sign": j.sign})
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    load_file(path)
    print("written. Restart servo_trajectory_bridge.py (or the launch) to use it.")


if __name__ == "__main__":
    main()
