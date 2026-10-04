#!/usr/bin/env python3
"""Save one aligned RealSense RGB-D pair for repeatable SAM 3.1 validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
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


def capture(args) -> int:
    if min(args.width, args.height, args.fps, args.warmup_frames) < 1:
        raise StreamError("dimensions, fps and warmup-frames must be positive")
    rs = _load_realsense()
    if args.output.exists():
        raise StreamError("output already exists; choose a new capture directory")
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
    started = False
    try:
        try:
            # Do not retain devices from a separate enumeration context while
            # starting the pipeline: RSUSB may try to claim the UVC interface
            # twice on macOS and fail with "failed to set power state".
            profile = pipeline.start(config)
        except RuntimeError as exc:
            raise StreamError(f"RealSense pipeline start failed: {exc}") from exc
        started = True
        align = rs.align(rs.stream.color)
        for _ in range(args.warmup_frames):
            frames = pipeline.wait_for_frames(5000)
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
        created_dirs = []
        parent = args.output.resolve()
        while not parent.exists():
            created_dirs.append(parent)
            parent = parent.parent
        args.output.mkdir(parents=True)
        np.savez_compressed(
            args.output / "rgbd.npz",
            rgb=rgb,
            depth_m=depth_m,
            metadata=json.dumps(metadata, allow_nan=False),
        )
        Image.fromarray(rgb).save(args.output / "rgb.png")
        (args.output / "capture.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # Only USB access needs root. Return newly created artifacts to the
        # invoking user so later inference can write alongside the capture.
        if os.geteuid() == 0 and os.environ.get("SUDO_UID"):
            uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
            for path in [*args.output.iterdir(), *created_dirs]:
                os.chown(path, uid, gid)
        print(
            f"[capture] {metadata['device']}; {rgb.shape[1]}x{rgb.shape[0]}; "
            f"valid depth pixels={np.isfinite(depth_m).sum()}; saved: {args.output}"
        )
        return 0
    finally:
        if started:
            pipeline.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        return capture(args)
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
