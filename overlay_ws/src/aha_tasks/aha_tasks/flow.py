"""Validated, ROS-independent sequential task execution."""

import json
import math
from pathlib import Path


def load_flow(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("frame_id") != "map":
        raise ValueError("Task locations must use the map frame")
    locations = data.get("locations")
    if not isinstance(locations, dict) or not locations:
        raise ValueError("locations must be a nonempty object")
    for label, pose in locations.items():
        if not label or not isinstance(pose, list) or len(pose) != 3:
            raise ValueError(f"Invalid location: {label}")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in pose):
            raise ValueError(f"Location {label} requires finite x, y, yaw")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a nonempty array")
    ids = set()
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("id"), str):
            raise ValueError("Every step needs a string id")
        if not step["id"] or step["id"] in ids:
            raise ValueError("Step ids must be nonempty and unique")
        ids.add(step["id"])
        if set(step) == {"id", "navigate"}:
            if (
                not isinstance(step["navigate"], str)
                or step["navigate"] not in locations
            ):
                raise ValueError(f"Unknown navigation label in {step['id']}")
        elif set(step) == {"id", "skip"}:
            if not isinstance(step["skip"], str) or not step["skip"].strip():
                raise ValueError("Skipped steps need a reason")
        else:
            raise ValueError(f"Step {step['id']} needs exactly one of navigate or skip")
    return data


def execute_flow(flow, navigate, report):
    """Only a successful navigation permits the next step; never mask failures."""
    for step in flow["steps"]:
        if "skip" in step:
            report(f"SKIPPED {step['id']}: {step['skip']}")
            continue
        label = step["navigate"]
        report(f"NAVIGATING {step['id']}: {label}")
        if not navigate(flow["locations"][label]):
            report(f"FAILED {step['id']}: {label}")
            return False
        report(f"ARRIVED {step['id']}: {label}")
    report("COMPLETED navigation flow (interaction steps were skipped)")
    return True
