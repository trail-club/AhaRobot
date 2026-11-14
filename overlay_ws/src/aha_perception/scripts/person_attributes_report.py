#!/usr/bin/env python3
"""Classify every person person_perception tracks and compare with a sim scene.

Each track is classified once (ClassifyPerson with the VLM) after it has been
visible for `--settle` s, so that gestures have several images. With
`--truth` (an aha_tasks config/sim/people scene), each result is matched to
the nearest scene person in map and scored: posture, gesture and the shirt
(a "<colour> t shirt" among the garments). Prints a table on exit (Ctrl-C or
`--duration`) and writes JSON with `--output`.
"""

import argparse
import json
import math
from pathlib import Path
import threading
import time

from aha_msgs.msg import PersonArray
from aha_msgs.srv import ClassifyPerson
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

MATCH_M = 0.8  # a result farther from every scene person is unmatched


def expected(label):
    """(posture, gesture) of a scene label."""
    if label in ("standing", "sitting", "lying"):
        return label, "none"
    return "standing", label


class Report(Node):
    def __init__(self, truth, settle, timeout):
        super().__init__("person_attributes_report")
        self.truth = truth
        self.settle = settle
        self.timeout = timeout
        self.lock = threading.Lock()
        self.first_seen = {}  # track id -> monotonic time
        self.results = {}  # track id -> dict
        self.busy = False
        self.client = self.create_client(
            ClassifyPerson,
            "/aha/perception/classify_person",
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_subscription(
            PersonArray, "/aha/perception/people", self.on_people, 10
        )

    def on_people(self, message):
        now = time.monotonic()
        with self.lock:
            if self.busy:
                return
            for person in message.people:
                if not person.visible:
                    self.first_seen.pop(person.id, None)
                    continue
                self.first_seen.setdefault(person.id, now)
                if (
                    person.id in self.results
                    or now - self.first_seen[person.id] < self.settle
                ):
                    continue
                self.busy = True
                threading.Thread(
                    target=self.classify,
                    args=(message.header.frame_id, person),
                    daemon=True,
                ).start()
                return

    def classify(self, frame, person):
        started = time.monotonic()
        result = {
            "id": person.id,
            "frame": frame,
            "position": [round(person.position.x, 2), round(person.position.y, 2)],
            "head_z": round(person.head.z, 2),
            "rule_posture": person.posture,
            "rule_gesture": person.gesture,
        }
        try:
            if not self.client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError("classify_person is not available")
            future = self.client.call_async(ClassifyPerson.Request(id=person.id))
            deadline = started + self.timeout
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not future.done():
                raise TimeoutError("no answer")
            reply = future.result()
            result.update(
                success=reply.success,
                posture=reply.posture,
                posture_source=reply.posture_source,
                gesture=reply.gesture,
                gesture_source=reply.gesture_source,
                garments=list(reply.garments),
                message=reply.message,
            )
        except (RuntimeError, TimeoutError) as exc:
            result.update(success=False, message=str(exc))
        result["seconds"] = round(time.monotonic() - started, 1)
        result.update(self.score(result))
        self.get_logger().info(json.dumps(result))
        with self.lock:
            self.results[person.id] = result
            self.busy = False

    def score(self, result):
        if not self.truth or result["frame"] != "map":
            return {}
        name, person = min(
            self.truth.items(),
            key=lambda item: math.dist(item[1]["position"][:2], result["position"]),
        )
        if math.dist(person["position"][:2], result["position"]) > MATCH_M:
            return {"truth": None}
        posture, gesture = expected(person["label"])
        shirt = f"{person['shirt']} t shirt"
        return {
            "truth": name,
            "truth_label": person["label"],
            "truth_shirt": person["shirt"],
            "posture_ok": result.get("posture") == posture,
            "gesture_ok": result.get("gesture") == gesture,
            "shirt_ok": shirt in result.get("garments", []),
        }

    def table(self):
        rows = sorted(
            self.results.values(), key=lambda r: (str(r.get("truth")), r["id"])
        )
        lines = [
            "| track | truth | posture | gesture | garments | s |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        def mark(ok):
            return "" if ok is None else (" ✓" if ok else " ✗")

        for r in rows:
            truth = (
                f"{r.get('truth')} ({r.get('truth_label')}, {r.get('truth_shirt')})"
                if r.get("truth")
                else "-"
            )
            lines.append(
                f"| {r['id']} | {truth} | {r.get('posture', '')} [{r.get('posture_source', '')}]"
                f"{mark(r.get('posture_ok'))} | {r.get('gesture', '')} "
                f"[{r.get('gesture_source', '')}]{mark(r.get('gesture_ok'))} | "
                f"{', '.join(r.get('garments', []))}{mark(r.get('shirt_ok'))} | {r['seconds']} |"
            )
        matched = [r for r in rows if r.get("truth")]
        if matched:
            for key in ("posture_ok", "gesture_ok", "shirt_ok"):
                lines.append(f"{key}: {sum(r[key] for r in matched)} / {len(matched)}")
            seen = {r["truth"] for r in matched}
            lines.append(f"scene people classified: {len(seen)} / {len(self.truth)}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path)
    parser.add_argument(
        "--duration", type=float, default=0.0, help="s; 0: until Ctrl-C"
    )
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", type=Path)
    args, ros_args = parser.parse_known_args()
    truth = None
    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))["people"]
    rclpy.init(args=ros_args)
    node = Report(truth, args.settle, args.timeout)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    deadline = time.monotonic() + args.duration if args.duration > 0 else math.inf
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with node.lock:
            text = node.table()
            results = list(node.results.values())
        print(text, flush=True)
        if args.output:
            args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
