#!/usr/bin/env python3
"""Ask the LLM where to put the held object and put it there.

  ros2 run aha_test_orchestration place_with_llm.py --label apple --variant C
  ros2 run aha_test_orchestration place_with_llm.py --label milk --variant A --dry-run

The robot must already hold the object (sim: aha_test_manipulation
sim_hold_object.py) in front of the furniture. Needs plane detection with the
furniture config, /aha/perception/objects, head_look_at.py and (without
--dry-run) the place server (aha_test_orchestration.place_runner). Prints the
prompt, the answer and the result; --log appends the full report as one JSON
line.
"""

import argparse
import json
import sys

import rclpy
from rclpy.parameter import Parameter

from aha_test_orchestration.llm import DEFAULT_MODEL, DEFAULT_URL, LlmClient
from aha_test_orchestration.place_decision import VARIANTS
from aha_test_orchestration.place_runner import PlaceLlmRunner, RunnerOptions
from aha_test_orchestration.scene import HeldObject


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="label of the held object")
    parser.add_argument(
        "--size", type=float, nargs=3, help="x y z (m), default: the object list's"
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="C")
    parser.add_argument("--furniture", default="shelf")
    parser.add_argument("--furniture-config", default="")
    parser.add_argument("--objects-config", default="")
    parser.add_argument("--arm", choices=["right", "left"], default="right")
    parser.add_argument("--grasp", choices=["side", "top"], default="side")
    parser.add_argument("--llm-url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-space-hint", action="store_true")
    parser.add_argument(
        "--back-off",
        action="store_true",
        help="observe from where every level is seen whole and come back "
        "(moves the base without an obstacle check)",
    )
    parser.add_argument("--dry-run", action="store_true", help="do not place")
    parser.add_argument("--log", help="append the report as a JSON line")
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = rclpy.create_node(
        "place_with_llm", parameter_overrides=[Parameter("use_sim_time", value=True)]
    )
    try:
        options = RunnerOptions(
            furniture=args.furniture,
            furniture_config=args.furniture_config,
            objects_config=args.objects_config,
            arm=args.arm,
            grasp=args.grasp,
            space_hint=not args.no_space_hint,
            back_off=args.back_off,
        )
        runner = PlaceLlmRunner(node, LlmClient(args.llm_url, args.model), options)
        size = args.size
        if size is None:
            entry = runner.object_list.get(args.label)
            if entry is None:
                parser.error(f"{args.label!r} is not in the object list; give --size")
            size = entry.size
        held = HeldObject(args.label, tuple(size))
        report = runner.run(held, args.variant, execute=not args.dry_run)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    for message in report.get("messages", [])[1:]:
        content = message.get("content") or json.dumps(message.get("tool_calls"))
        print(f"--- {message['role']}\n{content}")
    keys = ("observe_s", "llm_s", "prompt_tokens", "completion_tokens", "error")
    print({k: report[k] for k in keys if k in report})
    if "place" in report:
        print(report["place"])
    if args.log:
        with open(args.log, "a") as stream:
            stream.write(json.dumps(report) + "\n")
    if "error" in report:
        return 2
    return 0 if report.get("place", {}).get("success", args.dry_run) else 2


if __name__ == "__main__":
    sys.exit(main())
