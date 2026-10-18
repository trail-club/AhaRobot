"""One multimodal Codex CLI decision per simulator action interval."""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from .api import Action, JOINT_LIMITS
from .images import image_array

MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "medium"
CAMERAS = ("head", "left_wrist", "right_wrist")
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "situation", "decision_note"],
    "properties": {
        "action": {
            "type": "object",
            "additionalProperties": False,
            "required": ["linear_velocity", "angular_velocity", "joint_positions"],
            "properties": {
                "linear_velocity": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                "angular_velocity": {"type": "number", "minimum": -0.5, "maximum": 0.5},
                "joint_positions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "position"],
                        "properties": {
                            "name": {"type": "string", "enum": list(JOINT_LIMITS)},
                            "position": {"type": "number"},
                        },
                    },
                },
            },
        },
        "situation": {"type": "string"},
        "decision_note": {"type": "string"},
    },
}


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def save_observation(observation, directory):
    if observation.ground_truth is not None:
        raise ValueError("Codex policy requires standard sensor observations")
    if not all(name in observation.cameras for name in CAMERAS):
        raise ValueError("Codex policy requires head and both wrist cameras")
    metadata = {
        "sim_time": observation.sim_time,
        "task_id": observation.task_id,
        "instruction": observation.instruction,
        "joint_positions": dict(observation.joint_positions),
        "joint_velocities": dict(observation.joint_velocities),
        "odometry_x_y_yaw": list(observation.odometry),
        "cameras": {},
    }
    paths = []
    for name in CAMERAS:
        camera = observation.cameras[name]
        path = directory / f"{name}.png"
        Image.fromarray(image_array(camera.image)).save(path)
        paths.append(path)
        info = camera.camera_info
        transform = camera.base_transform.transform
        data = {
            "image": path.name,
            "sim_time": stamp(camera.image),
            "frame": camera.image.header.frame_id,
            "width": info.width,
            "height": info.height,
            "distortion_model": info.distortion_model,
            **{key: list(getattr(info, key)) for key in ("k", "d", "r", "p")},
            "base_transform": {
                "parent": camera.base_transform.header.frame_id,
                "child": camera.base_transform.child_frame_id,
                "translation": [getattr(transform.translation, axis) for axis in "xyz"],
                "quaternion_xyzw": [
                    getattr(transform.rotation, axis) for axis in "xyzw"
                ],
            },
        }
        if camera.depth_image is not None:
            depth = image_array(camera.depth_image, depth=True)
            np.save(directory / f"{name}_depth.npy", depth)
            rows = np.linspace(
                0, depth.shape[0] - 1, min(12, depth.shape[0]), dtype=int
            )
            columns = np.linspace(
                0, depth.shape[1] - 1, min(16, depth.shape[1]), dtype=int
            )
            data["depth_metres"] = {
                "sim_time": stamp(camera.depth_image),
                "pixel_rows": rows.tolist(),
                "pixel_columns": columns.tolist(),
                "samples": [
                    [
                        round(float(v), 3) if math.isfinite(v) and v > 0 else None
                        for v in row
                    ]
                    for row in depth[np.ix_(rows, columns)]
                ],
            }
        metadata["cameras"][name] = data
    (directory / "observation.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    return metadata, paths


def parse_decision(value):
    def number(value):
        if type(value) not in (int, float):
            raise ValueError("Codex action values must be numbers")
        # JSON integers are valid numbers, but ROS float fields require float.
        return float(value)

    if set(value) != {"action", "situation", "decision_note"}:
        raise ValueError(
            "Codex response must contain action, situation, and decision_note"
        )
    for key in ("situation", "decision_note"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"Codex {key} must be a nonempty string")
    command = value["action"]
    if set(command) != {"linear_velocity", "angular_velocity", "joint_positions"}:
        raise ValueError("Invalid Codex action fields")
    joints = {}
    for joint in command["joint_positions"]:
        if set(joint) != {"name", "position"} or joint["name"] in joints:
            raise ValueError("Invalid or duplicate commanded joint")
        joints[joint["name"]] = number(joint["position"])
    action = Action(
        number(command["linear_velocity"]), number(command["angular_velocity"]), joints
    )
    action.validate()
    return action


class CodexPolicy:
    def reset(self, task_id, instruction):
        self.executable = shutil.which("codex")
        if self.executable is None:
            raise RuntimeError(
                "Install and authenticate Codex CLI before using --policy codex"
            )
        self.duration = float(os.environ.get("AHA_POLICY_STEP_DURATION", "0"))
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Codex requires stepped evaluation (--step-duration 2)")
        self.directory = Path(os.environ["ROS_LOG_DIR"]).parent / "codex_steps"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.task_id, self.instruction = task_id, instruction
        self.history = []
        self.index = 0

    def act(self, observation):
        self.index += 1
        directory = self.directory / f"{self.index:04d}"
        directory.mkdir()
        metadata, images = save_observation(observation, directory)
        prompt = (
            "You control AhaRobot in a simulated pick-and-place task. Choose exactly one "
            "robot action using the attached current camera images and sensor observation. "
            "Do not use tools or read other files. Return JSON matching the supplied schema. "
            "Include a brief situation description and a concise decision justification, "
            "not a detailed reasoning trace. The three images are head, left_wrist, right_wrist "
            "in that order. Head depth is sampled on the pixel grid in the observation (metres; "
            "null means invalid). Calibration and base_link-to-camera transforms are provided. "
            "Use these to estimate the apple and green destination table from vision. "
            "Simulation is paused during your decision. Your action is held for "
            f"{self.duration:g} simulation seconds, then you receive fresh observations. "
            "Base x is forward, y is left, positive yaw turns left. Velocities are m/s and rad/s. "
            "Joint commands are absolute positions (lift/fingers in metres, other joints in radians). "
            "Unspecified joints retain previous targets; joint trajectories take two simulation seconds. "
            "joint_r1 and joint_l1 are independent right/left vertical lifts (positive raises). "
            "joint_r2..r6 and joint_l2..l6 move arms. Right gripper opens with joint_r7r=0.06 "
            "and joint_r7l=-0.06, and closes toward zero; left gripper is analogous. "
            "Head pan/tilt move the head camera; use its transform rather than assuming its direction. "
            "The apple must first be grasped and lifted, then released on the green table. "
            "Continue checking the images and feedback after each action. "
            f"Joint position limits: {json.dumps(JOINT_LIMITS)}\n"
            f"Previous decisions (latest eight): {json.dumps(self.history[-8:])}\n"
            f"Current observation: {json.dumps(metadata, allow_nan=False)}\n"
        )
        (directory / "prompt.txt").write_text(prompt)
        schema = directory / "schema.json"
        schema.write_text(json.dumps(RESPONSE_SCHEMA))
        response = directory / "response.json"
        command = [
            self.executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--disable",
            "plugins",
            "--disable",
            "remote_plugin",
            "--model",
            MODEL,
            "-c",
            f'model_reasoning_effort="{REASONING_EFFORT}"',
            "-c",
            "features.shell_tool=false",
            "-c",
            'web_search="disabled"',
            "--cd",
            str(directory),
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(response),
        ]
        for path in images:
            command.extend(["--image", str(path)])
        command.append("-")
        started = time.monotonic()
        with (directory / "codex.log").open("w") as log:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Codex exited with {completed.returncode}; see {directory / 'codex.log'}"
            )
        decision = json.loads(response.read_text())
        action = parse_decision(decision)
        record = {
            "step": self.index,
            "sim_time": observation.sim_time,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "step_duration": self.duration,
            "inference_wall_seconds": time.monotonic() - started,
            "action": asdict(action),
            "situation": decision["situation"],
            "decision_note": decision["decision_note"],
        }
        self.history.append(record)
        with (self.directory / "decisions.jsonl").open("a") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        return action
