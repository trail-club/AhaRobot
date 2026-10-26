#!/usr/bin/env python3
"""Place-with-LLM trials in Gazebo: compare the information variants A / B / C.

Runs in the development container with the overlay sourced, from the
repository root. With --launch, starts a headless sim (world rcjo2026, the
spawn of scenario.py, world_tf, the arena furniture config, plane detection,
objects:=true, MoveIt) and head_look_at.py on ROS_DOMAIN_ID / GZ_PARTITION of
this process and stops it at the end.

Setup: fold both arms and raise them to carry, spawn the props of
scenario.PROPS on the shelf. Per trial (noise x case x variant): put props
that moved back, set the detector's miss / mislabel rates and seed (the same
seed for the variants of a case), hold the case's object at carry
(sim_held_object), run aha_test_orchestration.place_runner (observe, ask the LLM,
PlaceObject) and after --settle read the Gazebo poses. Appends one row per
trial to results/trials.csv and the full report (prompts, answers, summary)
to results/reports.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from aha_msgs.msg import SurfaceArray
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

from ament_index_python.packages import get_package_share_directory

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation.sim_held_object import (
    GzSim,
    SimHolder,
    model_name,
    object_sdf,
    ycb_mesh,
)
from aha_test_manipulation.sim_trials import (
    fold_arms,
    launch_sim,
    open_csv,
    stop_sim,
    wait_controllers,
)
from aha_test_orchestration.llm import DEFAULT_MODEL, DEFAULT_URL, LlmCall, LlmClient
from aha_test_orchestration.place_runner import (
    PlaceLlmRunner,
    PlaceServerStuck,
    RunnerOptions,
)
from aha_test_orchestration.scene import HeldObject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenario import (  # noqa: E402
    CASES,
    FURNITURE,
    LEVEL_HEIGHTS,
    NEXT_TO,
    PROPS,
    SPAWN,
    YCB,
    level_of_height,
    on_boards,
)

HERE = Path(__file__).resolve().parent
DETECTOR = "/aha_perception_sim_object_detector"
NOISE = {"clean": (0.0, 0.0), "noisy": (0.10, 0.05)}  # miss, mislabel rate
PROP_MASS = 0.2  # kg
MOVED = 0.02  # m, a prop counts as knocked
ROBOT_MOVED = 0.02  # m / deg off the spawn pose: stop the run
ROBOT_TURNED = 2.0
RETURN_TOLERANCE = (0.01, 1.0)  # m / deg off the spawn pose left after a trial
RETURN_MAX_DISTANCE = 1.5  # m, e.g. from a back-off pose of a stopped run
RESET = 0.01  # m, props further from home are put back before a trial
MATCH = 0.06  # m, summary object to a prop
CONTROLLERS = ("right_arm_controller", "left_arm_controller", "lift_controller")
FIELDS = [
    "run",
    "noise",
    "case",
    "held",
    "variant",
    "expected_level",
    "detected",
    "missed",
    "mislabelled",
    "decision_level",
    "decision_target",
    "level_ok",
    "next_ok",
    "decision_ok",
    "llm_calls",
    "llm_error",
    "place_status",
    "place_success",
    "goals",
    "base_move",
    "compromise",
    "placed_level",
    "d_same",
    "d_other",
    "placed_ok",
    "knocked",
    "robot_moved",
    "observe_s",
    "stable",
    "stale_looks",
    "back_off",
    "llm_s",
    "place_s",
    "trial_s",
    "prompt_tokens",
    "completion_tokens",
]


def yaw_matrix(x, y, z, yaw=0.0):
    return geo.transform(geo.rot_z(yaw), [x, y, z])


class Displaced(RuntimeError):
    """The robot is no longer at the spawn pose: later trials are not valid."""


class FixedChoice:
    """Stands in for the LLM with --next-to: answers variant C's prompt with
    the level and id of the first object with that label (tests the robot
    side, e.g. a base move, independent of the LLM's choice)."""

    def __init__(self, label: str):
        self.label = label

    def chat(self, messages, tools=None, tool_choice=None):
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        arguments = {"reason": f"fixed: next to {self.label}"}
        for line in prompt.splitlines():
            level = re.match(r"- level (\d+) ", line)
            found = re.search(rf"O(\d+) {re.escape(self.label)}\b", line)
            if level and found:
                arguments.update(
                    furniture=FURNITURE,
                    level=int(level.group(1)),
                    next_to=f"O{found.group(1)}",
                )
                break
        call = {
            "id": "fixed",
            "type": "function",
            "function": {"name": "choose_place", "arguments": json.dumps(arguments)},
        }
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        return message, LlmCall(0.0)


def planar(T):
    """The x, y and yaw of a pose as a 4x4 matrix (z and tilt dropped)."""
    return yaw_matrix(T[0, 3], T[1, 3], 0.0, math.atan2(T[1, 0], T[0, 0]))


def robot_offset(world_robot, world_spawn) -> tuple[float, float]:
    """(horizontal distance m, yaw deg) of the robot from the spawn pose."""
    d = np.linalg.inv(world_spawn) @ world_robot
    return float(np.linalg.norm(d[:2, 3])), abs(
        math.degrees(math.atan2(d[1, 0], d[0, 0]))
    )


class Trials:
    def __init__(self, args):
        self.args = args
        self.node = rclpy.create_node(
            "place_llm_trials",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.gz = GzSim(timeout=300.0)
        self.holder = SimHolder(self.node, self.gz)
        client = (
            FixedChoice(args.next_to)
            if args.next_to
            else LlmClient(args.llm_url, args.model)
        )
        self.runner = PlaceLlmRunner(
            self.node,
            client,
            RunnerOptions(
                furniture=FURNITURE,
                move_base=args.move_base,
                next_to_distance=args.next_to_distance,
                back_off=args.back_off,
            ),
        )
        self.objects = self.runner.object_list
        self.set_params = self.node.create_client(
            SetParameters, f"{DETECTOR}/set_parameters"
        )
        self.surface_count = 0
        self.node.create_subscription(
            SurfaceArray, "/aha/perception/surfaces", self.on_surfaces, 10
        )
        self.home = {}

    def on_surfaces(self, _msg):
        self.surface_count += 1

    def spin(self, seconds):
        self.holder.spin_until(lambda: False, seconds)

    def wait_ready(self, timeout):
        ok = self.runner.place.wait_for_server(timeout_sec=timeout)
        ok = ok and self.holder.move_client.wait_for_server(timeout_sec=timeout)
        ok = ok and wait_controllers(self.node, CONTROLLERS, timeout)
        ok = ok and self.set_params.wait_for_service(timeout_sec=timeout)
        ok = ok and self.holder.spin_until(lambda: self.surface_count > 0, timeout)
        self.spin(10.0)  # spawners activate the controllers
        return ok

    def prepare_arms(self):
        fold_arms(self.node, self.holder)

    def geometry(self, label: str):
        """(shape, size, mesh or None) of a label's object: the YCB model with
        --ycb if the label has one, else the object list's box or cylinder."""
        ycb = YCB.get(label) if self.args.ycb else None
        if ycb is None:
            entry = self.objects.get(label)
            return entry.shape, tuple(entry.size), None
        models = Path(get_package_share_directory("tmc_wrs_gz_worlds")) / "models"
        return ycb.shape, ycb.size, ycb_mesh(models / ycb.model, ycb.size[2])

    def spawn_prop(self, prop):
        shape, size, mesh = self.geometry(prop.label)
        height = LEVEL_HEIGHTS[prop.level]
        yaw = YCB[prop.label].yaw if mesh is not None else 0.0
        self.gz.remove(prop.model)
        T = yaw_matrix(prop.x, prop.y, height + size[2] / 2 + 0.002, yaw)
        mass = mesh.mass if mesh is not None else PROP_MASS
        if not self.gz.spawn(
            object_sdf(prop.model, shape, size, mass, mesh=mesh), T, 20000
        ):
            raise RuntimeError(f"could not spawn {prop.model}")

    def reset_props(self) -> list[str]:
        """Spawn missing props and put moved ones back; returns their names."""
        poses = self.gz.model_poses()
        reset = []
        for prop in PROPS:
            T = poses.get(prop.model)
            home = self.home.get(prop.model)
            if (
                T is None
                or home is None
                or np.linalg.norm(T[:3, 3] - home[:3, 3]) > RESET
            ):
                self.spawn_prop(prop)
                reset.append(prop.model)
        if reset:
            time.sleep(2.0)
            poses = self.gz.model_poses()
            for prop in PROPS:
                if prop.model in reset:
                    self.home[prop.model] = poses[prop.model]
        return reset

    def configure_detector(self, miss, mislabel, seed):
        def value(v):
            if isinstance(v, int):
                return ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER, integer_value=v
                )
            return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v)

        request = SetParameters.Request()
        request.parameters = [
            ParameterMsg(name=n, value=value(v))
            for n, v in (
                ("miss_rate", miss),
                ("mislabel_rate", mislabel),
                ("seed", seed),
            )
        ]
        response = self.holder.wait_future(self.set_params.call_async(request), 10.0)
        if not all(r.successful for r in response.results):
            raise RuntimeError("could not set the detector parameters")

    def ground_truth(self, poses) -> list[dict]:
        out = []
        for prop in PROPS:
            T = poses.get(prop.model)
            if T is not None:
                out.append(
                    {
                        "model": prop.model,
                        "label": prop.label,
                        "category": self.objects.category(prop.label),
                        "level": prop.level,
                        "xyz": T[:3, 3],
                    }
                )
        return out

    def run_trial(self, noise, case, variant, seed) -> tuple[dict, dict]:
        start = time.monotonic()
        row = {
            "noise": noise,
            "case": case.name,
            "held": case.label,
            "variant": variant,
            "expected_level": case.level,
        }
        self.reset_props()
        self.configure_detector(*NOISE[noise], seed)
        shape, size, mesh = self.geometry(case.label)
        self.holder.hold(
            self.args.arm,
            shape,
            size,
            geo.SIDE,
            mass=mesh.mass if mesh is not None else 0.1,
            physical=self.args.physical_hold,
            mesh=mesh,
        )
        before = self.gz.model_poses()
        # The cases are laid out for the spawn pose.
        offset = robot_offset(before["aha_robot"], self.spawn_pose())
        if offset[0] > ROBOT_MOVED or offset[1] > ROBOT_TURNED:
            raise Displaced(
                f"robot {offset[0]:.3f} m / {offset[1]:.1f} deg off the spawn pose"
            )
        # The summary is in odom, which drifts from the spawn pose over base
        # moves: compare it with the ground truth through the current odom.
        world_odom = self.world_odom(before["aha_robot"])
        truth = self.ground_truth(before)
        report = self.runner.run(
            HeldObject(case.label, size, shape),
            variant,
            not self.args.dry_run,
        )
        self.spin(self.args.settle)
        after = self.gz.model_poses()
        report.update(noise=noise, case=case.name, seed=seed)
        row.update(
            {
                k: report.get(k, "")
                for k in (
                    "observe_s",
                    "stable",
                    "stale_looks",
                    "llm_s",
                    "llm_calls",
                    "llm_error",
                    "prompt_tokens",
                    "completion_tokens",
                )
            }
        )
        back_off = report.get("back_off")
        if back_off:
            back = back_off.get("return", {})
            row["back_off"] = (
                f"camera {back_off['camera_distance']} m, {back_off['looks']} looks, "
                f"back err {back.get('error_xy', '?')} m"
                if back_off["ok"]
                else f"failed: {back_off['message']}"
            )
        # Detections against the props (summary objects are in odom = spawn frame).
        # One to one, closest pairs first.
        summary = report.get("summary", {})
        pairs = []
        for obj in summary.get("objects", []):
            p = (world_odom @ np.array([*obj["position"], 1.0]))[:3]
            for k, t in enumerate(truth):
                d = np.linalg.norm(t["xyz"] - p)
                if d < MATCH:
                    pairs.append((d, k, obj, p))
        matched, used = {}, set()
        for _, k, obj, p in sorted(pairs, key=lambda pair: pair[0]):
            if obj["id"] not in matched and k not in used:
                matched[obj["id"]] = (truth[k], obj["label"], p)
                used.add(k)
        row["detected"] = len(matched)
        row["missed"] = " ".join(
            t["model"]
            for t in truth
            if t["model"] not in {m[0]["model"] for m in matched.values()}
        )
        row["mislabelled"] = " ".join(
            f"{t['model']}:{label}"
            for t, label, _ in matched.values()
            if label != t["label"]
        )

        decision = report.get("decision")
        if decision:
            row["decision_level"] = decision["level"]
            row["level_ok"] = decision["level"] == case.level
            target_xy = None
            if decision.get("next_to") is not None and decision["next_to"] in matched:
                t = matched[decision["next_to"]][0]
                row["decision_target"] = f"next to {t['model']}"
                next_ok = bool(case.next_to_category) and t["category"] == case.category
            elif decision.get("next_to") is not None:
                row["decision_target"] = f"next to unmatched O{decision['next_to']}"
                next_ok = False
            elif decision.get("point") is not None:
                px, py = decision["point"]
                target_xy = (world_odom @ np.array([px, py, 0.0, 1.0]))[:2]
                row["decision_target"] = f"point {px:.2f},{py:.2f}"
                next_ok = self.next_to_ok(case, truth, target_xy, case.level)
            else:
                height_only = variant.startswith("B")
                row["decision_target"] = "level only" if height_only else "empty"
                next_ok = None if height_only else not case.next_to_category
            row["next_ok"] = "" if next_ok is None else next_ok
            row["decision_ok"] = row["level_ok"] and next_ok is not False
        else:
            row["decision_ok"] = False

        place = report.get("place")
        if place:
            row["place_status"] = place["status"]
            row["place_success"] = place["success"]
            row["place_s"] = place.get("place_s", "")
            row["goals"] = ">".join(place.get("goals", []))
            row["compromise"] = place.get("compromise", "")
            move = place.get("base_move")
            if move:
                row["base_move"] = (
                    f"err {move['error_xy']:.3f} m {move['error_yaw_deg']:.1f} deg"
                    if move["ok"]
                    else f"failed: {move['message']}"
                )
            obj = after.get(model_name(self.args.arm))
            if obj is not None:
                bottom = obj[:3, 3] - obj[:3, 2] * size[2] / 2
                placed_level = level_of_height(bottom[2])
                inside = on_boards(bottom[0], bottom[1])
                row["placed_level"] = placed_level if inside else 0
                same, other = self.distances(case, truth, bottom[:2], placed_level)
                row["d_same"] = "" if same is None else round(same, 3)
                row["d_other"] = "" if other is None else round(other, 3)
                ok = place["success"] and inside and placed_level == case.level
                if case.next_to_category:
                    ok = ok and same is not None and same <= NEXT_TO
                    ok = ok and (other is None or same < other)
                row["placed_ok"] = bool(ok)
                report["placed"] = [round(float(v), 4) for v in bottom]
            else:
                row["placed_ok"] = False
        moved = [
            f"{p.model}:{np.linalg.norm(after[p.model][:3, 3] - before[p.model][:3, 3]):.3f}"
            for p in PROPS
            if p.model in before
            and p.model in after
            and np.linalg.norm(after[p.model][:3, 3] - before[p.model][:3, 3]) > MOVED
        ]
        row["knocked"] = " ".join(moved)
        # Base motion the odometry did not see (pushed, or wheel drift over
        # the robot's own moves).
        odom = self.runner.transform("odom", "base_footprint")
        expected = world_odom @ planar(odom) if odom is not None else world_odom
        robot = robot_offset(after["aha_robot"], expected)
        row["robot_moved"] = f"{robot[0]:.3f} m {robot[1]:.1f} deg"
        if self.args.move_base or self.args.back_off:
            self.return_to_spawn(after["aha_robot"])
        row["trial_s"] = round(time.monotonic() - start, 1)
        report["row"] = {k: v for k, v in row.items()}
        return row, report

    def spawn_pose(self):
        x, y, yaw = self.args.spawn
        return yaw_matrix(x, y, 0.0, yaw)

    def world_odom(self, world_robot):
        """World <- odom from the robot's ground truth and its odometry (the
        spawn pose without the odom TF)."""
        odom = self.runner.transform("odom", "base_footprint")
        if odom is None:
            return self.spawn_pose()
        return planar(world_robot) @ np.linalg.inv(planar(odom))

    def return_to_spawn(self, world_robot):
        """Drive back to the ground-truth spawn pose for the next trial (the
        target in the drifted odom)."""
        offset = robot_offset(world_robot, self.spawn_pose())
        if offset[0] <= RETURN_TOLERANCE[0] and offset[1] <= RETURN_TOLERANCE[1]:
            return
        target = np.linalg.inv(self.world_odom(world_robot)) @ self.spawn_pose()
        back = self.runner.base.move_to(
            target[0, 3],
            target[1, 3],
            math.atan2(target[1, 0], target[0, 0]),
            RETURN_MAX_DISTANCE,
        )
        if not back["ok"]:
            raise Displaced(f"could not move back: {back['message']}")

    @staticmethod
    def distances(case, truth, xy, level):
        same = [
            np.linalg.norm(t["xyz"][:2] - xy)
            for t in truth
            if t["level"] == level and t["category"] == case.category
        ]
        other = [
            np.linalg.norm(t["xyz"][:2] - xy)
            for t in truth
            if t["level"] == level and t["category"] != case.category
        ]
        return (min(same) if same else None), (min(other) if other else None)

    def next_to_ok(self, case, truth, xy, level) -> bool:
        same, other = self.distances(case, truth, xy, level)
        if not case.next_to_category:
            return same is None
        return same is not None and same <= NEXT_TO and (other is None or same < other)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    names = [c.name for c in CASES]
    parser.add_argument("--cases", nargs="+", choices=names, default=names)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["A", "B", "B2", "C"],
        default=["A", "B", "B2", "C"],
    )
    parser.add_argument("--noise", nargs="+", choices=list(NOISE), default=list(NOISE))
    parser.add_argument("--launch", action="store_true", help="start and stop the sim")
    parser.add_argument("--dry-run", action="store_true", help="decide only, no place")
    parser.add_argument("--arm", default="right")
    parser.add_argument("--settle", type=float, default=3.0, help="s before measuring")
    parser.add_argument("--llm-url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--run", default=time.strftime("%Y-%m-%d %H:%M"))
    parser.add_argument(
        "--move-base",
        action="store_true",
        help="move the base to the suggested pose when the spot is out of reach "
        "(and back to the spawn pose after the trial)",
    )
    parser.add_argument(
        "--ycb",
        action="store_true",
        help="YCB meshes for the props and held objects that have one "
        "(scenario.YCB); else boxes and cylinders of the object list",
    )
    parser.add_argument(
        "--physics-step",
        type=float,
        default=0.0,
        help="s, physics step set in the running sim before the trials (0: keep "
        "the world's, 0.004 in rcjo2026); 0.008 runs about twice as fast",
    )
    parser.add_argument(
        "--back-off",
        action="store_true",
        help="observe from where every level is seen whole "
        "(RunnerOptions.back_off) and come back",
    )
    parser.add_argument(
        "--next-to-distance",
        type=float,
        default=0.15,
        help="m, RunnerOptions.next_to_distance (0: any spot, group first)",
    )
    parser.add_argument(
        "--next-to",
        default="",
        help="no LLM: answer variant C with the first object of this label",
    )
    parser.add_argument(
        "--physical-hold",
        action="store_true",
        help="hold the object by finger friction only (no DetachableJoint weld)",
    )
    parser.add_argument(
        "--spawn",
        nargs=3,
        type=float,
        default=list(SPAWN),
        metavar=("X", "Y", "YAW"),
        help="robot spawn pose in the world (the launched or running sim's)",
    )
    args = parser.parse_args()
    if args.next_to and args.variants != ["C"]:
        parser.error("--next-to answers variant C only (add --variants C)")
    if args.launch and "GZ_PARTITION" not in os.environ:
        parser.error("set GZ_PARTITION (and ROS_DOMAIN_ID) for the launched sim")

    args.out.mkdir(parents=True, exist_ok=True)
    process = None
    if args.launch:
        log = Path(os.environ.get("TMPDIR", "/tmp")) / "place_llm_trials.log"
        print(f"launching the sim (log {log})", flush=True)
        process = launch_sim(
            log, "rcjo2026", args.spawn, furniture=True, extra="objects:=true"
        )
    rclpy.init()
    stream, writer = open_csv(args.out / "trials.csv", FIELDS)
    try:
        trials = Trials(args)
        if not trials.wait_ready(600.0):
            print("sim not ready", flush=True)
            return 1
        if args.physics_step > 0.0 and not trials.gz.set_physics_step(
            args.physics_step
        ):
            print(f"could not set the physics step {args.physics_step} s", flush=True)
            return 1
        trials.prepare_arms()
        if args.move_base or args.back_off:
            # E.g. left off the spawn pose by the odometry drift of a run before.
            trials.return_to_spawn(trials.gz.model_poses()["aha_robot"])
        trials.reset_props()
        with stream, (args.out / "reports.jsonl").open("a") as reports:
            for noise in args.noise:
                for k, case in enumerate(CASES):
                    if case.name not in args.cases:
                        continue
                    seed = 1000 + k if noise == "noisy" else 0
                    for variant in args.variants:
                        try:
                            row, report = trials.run_trial(noise, case, variant, seed)
                        except (Displaced, PlaceServerStuck) as error:
                            print(f"stopped: {error}; restart the sim", flush=True)
                            return 1
                        except RuntimeError as error:
                            row = {
                                "noise": noise,
                                "case": case.name,
                                "variant": variant,
                            }
                            row["llm_error"] = f"trial error: {error}"
                            report = {"row": row}
                        row["run"] = report["run"] = args.run
                        writer.writerow(row)
                        stream.flush()
                        reports.write(json.dumps(report, default=float) + "\n")
                        reports.flush()
                        print(json.dumps(row, default=str), flush=True)
    finally:
        if process is not None:
            stop_sim(process)
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # The gz-transport Python binding sometimes crashes during interpreter teardown.
    os._exit(code)
