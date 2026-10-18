"""Record camera motion and policy decisions in a simulation-time H.264 video."""

from dataclasses import asdict
import json
import math
import textwrap

import cv2
import numpy as np

from .images import image_array

CAMERA_TOPICS = {
    "/camera/color/image_raw": "head",
    "/camera/left_wrist/image_raw": "left_wrist",
    "/camera/right_wrist/image_raw": "right_wrist",
}
CAMERAS = ("head", "left_wrist", "right_wrist")


class VideoRecorder:
    fps = 10
    size = (1920, 720)

    def __init__(self, directory, task_id, policy):
        self.directory = directory
        self.path = directory / "episode.mp4"
        self.task_id, self.policy = task_id, policy
        self.images = {}
        self.previous_images = None
        self.origin = None
        self.last_time = None
        self.frame_count = 0
        self.step = 0
        self.decision = {}
        self.command = None
        self.status = "initial observation"
        self.closed = False
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"avc1"), self.fps, self.size
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(
                "H.264 video encoder unavailable; cannot record episode.mp4"
            )
        self.index = (directory / "video_frames.jsonl").open("w")

    def image(self, topic, message):
        if topic in CAMERA_TOPICS:
            self.images[CAMERA_TOPICS[topic]] = message

    def start(self, sim_time):
        if not all(name in self.images for name in CAMERAS):
            raise RuntimeError("Video recording requires all three camera images")
        self.origin = sim_time
        self.capture(sim_time)

    def action(self, action, phase=None):
        self.step += 1
        self.command = asdict(action)
        self.status = phase or "executing action"
        path = self.directory / "codex_steps/decisions.jsonl"
        if path.exists():
            lines = path.read_text().splitlines()
            if lines:
                self.decision = json.loads(lines[-1])

    @staticmethod
    def text(frame, value, y, *, lines=1, color=(235, 235, 235)):
        # OpenCV's built-in font is ASCII; original notes remain in decisions.jsonl.
        value = str(value).replace("\u2019", "'").replace("\u2013", "-")
        value = value.encode("ascii", errors="replace").decode()
        wrapped = textwrap.wrap(value, width=160) or [""]
        for index, line in enumerate(wrapped[:lines]):
            if index == lines - 1 and len(wrapped) > lines:
                line = line[:157] + "..."
            cv2.putText(
                frame,
                line,
                (12, y + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                1,
                cv2.LINE_AA,
            )

    def compose(self, sim_time, images):
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        elapsed = sim_time - self.origin
        self.text(
            frame,
            f"{self.task_id} | policy: {self.policy} | elapsed {elapsed:6.2f}s | clock {sim_time:.3f}s | "
            "1x simulation time; inference pauses omitted",
            25,
        )
        for column, name in enumerate(CAMERAS):
            message = images[name]
            rgb = image_array(message)
            code = cv2.COLOR_GRAY2BGR if rgb.ndim == 2 else cv2.COLOR_RGB2BGR
            panel = cv2.resize(cv2.cvtColor(rgb, code), (640, 480))
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            cv2.rectangle(panel, (0, 0), (640, 29), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"{name} | image stamp {stamp:.3f}s",
                (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            frame[36:516, column * 640 : (column + 1) * 640] = panel
        self.text(
            frame, f"Decision {self.step} | {self.status}", 544, color=(80, 230, 255)
        )
        if self.command is not None:
            command = self.command
            joints = (
                ", ".join(
                    f"{name}={value:.3f}"
                    for name, value in command["joint_positions"].items()
                )
                or "hold joint targets"
            )
            self.text(
                frame,
                f"Action: vx={command['linear_velocity']:.3f} m/s, "
                f"wz={command['angular_velocity']:.3f} rad/s; {joints}",
                572,
                lines=2,
            )
        self.text(
            frame, "Situation: " + self.decision.get("situation", ""), 627, lines=2
        )
        self.text(
            frame, "Note: " + self.decision.get("decision_note", ""), 681, lines=2
        )
        return frame

    def write(self, sim_time, images=None):
        images = self.images if images is None else images
        self.writer.write(self.compose(sim_time, images))
        stamps = {
            name: image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
            for name, image in images.items()
        }
        self.index.write(
            json.dumps(
                {
                    "frame": self.frame_count,
                    "sim_time": sim_time,
                    "camera_stamps": stamps,
                    "decision": self.step,
                    "status": self.status,
                }
            )
            + "\n"
        )
        self.frame_count += 1
        self.last_time = sim_time

    def capture(self, sim_time):
        if self.origin is None or not math.isfinite(sim_time):
            return
        # Video sampling uses simulation time, so a slow model adds no frozen minutes.
        target = math.floor((sim_time - self.origin) * self.fps + 1e-6)
        images = dict(self.images)
        while self.frame_count <= target:
            # Hold the preceding image across a missed sampling instant.
            # Do not show a future image in an earlier video frame.
            sample = (
                self.previous_images
                if self.frame_count < target and self.previous_images is not None
                else images
            )
            self.write(self.origin + self.frame_count / self.fps, sample)
        self.previous_images = images
        self.index.flush()

    def close(self, sim_time=None, status="error"):
        if self.closed:
            return
        try:
            self.status = status
            if self.origin is not None:
                self.capture(sim_time if sim_time is not None else self.last_time)
                # Hold the terminal result for one playback second.
                for _ in range(self.fps):
                    self.write(sim_time if sim_time is not None else self.last_time)
        finally:
            self.writer.release()
            self.index.close()
            self.closed = True
        summary = {
            "path": str(self.path),
            "codec": "h264",
            "fps": self.fps,
            "width": self.size[0],
            "height": self.size[1],
            "frames": self.frame_count,
            "playback_seconds": self.frame_count / self.fps,
            "start_sim_time": self.origin,
            "end_sim_time": self.last_time,
            "status": status,
            "inference_pauses": "omitted",
            "terminal_hold_seconds": 1,
        }
        (self.directory / "video.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not self.frame_count:
            raise RuntimeError("No video frames recorded")
        return summary
