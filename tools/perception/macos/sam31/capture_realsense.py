#!/usr/bin/env python3
"""Save aligned RealSense RGB-D pairs for repeatable SAM 3.1 validation.

By default one pair is saved and the camera is released. With --serve the
camera stays open: preview JPEGs stream to stdout and a pair is saved for each
capture request read from stdin (see `serve`).
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.perception.macos.pointcloud.stream_realsense import (  # noqa: E402
    COLOR_FRAME,
    StreamError,
    _color_frame_rgb,
    _device_label,
    _load_realsense,
    depth_to_meters,
    make_camera_info_message,
    stamp_from_unix_ns,
)

# Preview depth colouring: red is near, blue is far, black is invalid.
PREVIEW_NEAR_M, PREVIEW_FAR_M = 0.3, 3.0
PREVIEW_JPEG_QUALITY = 80


def validate_stream_options(args) -> None:
    if min(args.width, args.height, args.fps, args.warmup_frames) < 1:
        raise StreamError("dimensions, fps and warmup-frames must be positive")


def start_pipeline(rs, args):
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps
    )
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    try:
        # Do not retain devices from a separate enumeration context while
        # starting the pipeline: RSUSB may try to claim the UVC interface
        # twice on macOS and fail with "failed to set power state".
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise StreamError(f"RealSense pipeline start failed: {exc}") from exc
    return pipeline, profile


def read_pair(rs, frames, profile, align):
    """Return aligned RGB, depth in meters and capture metadata."""
    aligned = align.process(frames)
    color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
    if not color or not depth:
        raise StreamError("aligned RGB-D pair is missing")
    rgb = _color_frame_rgb(color, "rgb8").copy()
    scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    depth_m = depth_to_meters(np.asanyarray(depth.get_data()), scale)
    if rgb.shape[:2] != depth_m.shape:
        raise StreamError("aligned depth dimensions differ from RGB")
    intrinsics = color.profile.as_video_stream_profile().get_intrinsics()
    stamp = stamp_from_unix_ns(time.time_ns())
    metadata = {
        "device": _device_label(rs, profile.get_device()),
        "camera_info": make_camera_info_message(
            intrinsics, stamp=stamp, frame_id=COLOR_FRAME
        ),
        "depth_scale": scale,
        "color_frame_number": int(color.get_frame_number()),
        "depth_frame_number": int(depth.get_frame_number()),
        "color_timestamp_ms": float(color.get_timestamp()),
        "depth_timestamp_ms": float(depth.get_timestamp()),
    }
    return rgb, depth_m, metadata


def save_pair(output: Path, rgb, depth_m, metadata) -> None:
    if output.exists():
        raise StreamError("output already exists; choose a new capture directory")
    created_dirs = []
    parent = output.resolve()
    while not parent.exists():
        created_dirs.append(parent)
        parent = parent.parent
    output.mkdir(parents=True)
    try:
        np.savez_compressed(
            output / "rgbd.npz",
            rgb=rgb,
            depth_m=depth_m,
            metadata=json.dumps(metadata, allow_nan=False),
        )
        Image.fromarray(rgb).save(output / "rgb.png")
        (output / "capture.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    finally:
        # Only USB access needs root. Return created artifacts, including
        # partial ones after a failed write, to the invoking user.
        if os.geteuid() == 0 and os.environ.get("SUDO_UID"):
            uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
            for path in [*output.iterdir(), *created_dirs]:
                os.chown(path, uid, gid)


def preview_jpeg(rgb: np.ndarray, depth_m: np.ndarray) -> bytes:
    """Colour image with a centre cross beside the colourised aligned depth."""
    height, width = depth_m.shape
    view = rgb.copy()
    view[height // 2, max(0, width // 2 - 15) : width // 2 + 16] = 255
    view[max(0, height // 2 - 15) : height // 2 + 16, width // 2] = 255
    valid = (
        np.isfinite(depth_m) & (depth_m >= PREVIEW_NEAR_M) & (depth_m <= PREVIEW_FAR_M)
    )
    t = np.clip(
        (np.nan_to_num(depth_m) - PREVIEW_NEAR_M) / (PREVIEW_FAR_M - PREVIEW_NEAR_M),
        0,
        1,
    )
    depth_rgb = np.stack(
        [(1 - t) * 255, (1 - np.abs(2 * t - 1)) * 200, t * 255], axis=-1
    ).astype(np.uint8)
    depth_rgb[~valid] = 0
    buffer = BytesIO()
    Image.fromarray(np.concatenate([view, depth_rgb], axis=1)).save(
        buffer, format="JPEG", quality=PREVIEW_JPEG_QUALITY
    )
    return buffer.getvalue()


def capture(args) -> int:
    validate_stream_options(args)
    rs = _load_realsense()
    if args.output.exists():
        raise StreamError("output already exists; choose a new capture directory")
    pipeline, profile = start_pipeline(rs, args)
    try:
        align = rs.align(rs.stream.color)
        for _ in range(args.warmup_frames):
            frames = pipeline.wait_for_frames(5000)
        rgb, depth_m, metadata = read_pair(rs, frames, profile, align)
        save_pair(args.output, rgb, depth_m, metadata)
        print(
            f"[capture] {metadata['device']}; {rgb.shape[1]}x{rgb.shape[0]}; "
            f"valid depth pixels={np.isfinite(depth_m).sum()}; saved: {args.output}"
        )
        return 0
    finally:
        pipeline.stop()


def write_message(stream, message: dict, payload: bytes = b"") -> None:
    """One JSON header line, followed by `bytes` raw payload bytes."""
    header = dict(message, bytes=len(payload))
    stream.write(json.dumps(header).encode() + b"\n" + payload)
    stream.flush()


def serve(args, requests=None, out=None) -> int:
    """Keep the camera open for an interactive session.

    stdout carries only protocol messages: `ready`, `preview` (JPEG payload),
    and `captured` / `error` answering each stdin line {"capture": "<dir>"}.
    Each capture saves the first frame that arrives after the request. EOF on
    stdin stops the camera.
    """
    validate_stream_options(args)
    out = out or sys.stdout.buffer
    if requests is None:
        requests = queue.Queue()

        def read_requests():
            for line in sys.stdin:
                requests.put(line)
            requests.put(None)

        threading.Thread(target=read_requests, daemon=True).start()
    rs = _load_realsense()
    pipeline, profile = start_pipeline(rs, args)
    try:
        align = rs.align(rs.stream.color)
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(5000)
        write_message(
            out, {"type": "ready", "device": _device_label(rs, profile.get_device())}
        )
        while True:
            try:
                request = requests.get_nowait()
            except queue.Empty:
                request = ""
            if request is None:
                return 0
            rgb, depth_m, metadata = read_pair(
                rs, pipeline.wait_for_frames(5000), profile, align
            )
            if request.strip():
                try:
                    output = Path(json.loads(request)["capture"])
                    save_pair(output, rgb, depth_m, metadata)
                    write_message(out, {"type": "captured", "output": str(output)})
                except (StreamError, OSError, ValueError, KeyError, TypeError) as exc:
                    write_message(out, {"type": "error", "message": str(exc)})
            write_message(out, {"type": "preview"}, preview_jpeg(rgb, depth_m))
    finally:
        pipeline.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument(
        "--serve", action="store_true", help="keep streaming; capture on stdin requests"
    )
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        return serve(args) if args.serve else capture(args)
    except KeyboardInterrupt:
        return 130
    except (StreamError, RuntimeError, OSError, ValueError) as exc:
        print(f"[capture] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Keep the rebuilt SDK in the project venv; no global linker changes.
    library = str(Path(sys.prefix) / "realsense-sdk/lib")
    if library not in os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep):
        env = os.environ.copy()
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, (library, env.get("DYLD_LIBRARY_PATH", "")))
        )
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)
    raise SystemExit(main())
