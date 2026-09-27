#!/usr/bin/env python3
"""Publish a synthetic aligned RGB-D scene through the same conversion path."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from stream_realsense import (
    CAMERA_INFO_TOPIC,
    COLOR_TOPIC,
    DEPTH_TOPIC,
    RosbridgePublisher,
    make_frame_bundle,
    stamp_from_unix_ns,
)


def main() -> int:
    width, height = 160, 120
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    color = np.stack((x * 255 // width, y * 255 // height, np.full_like(x, 180)), axis=2)
    color = color.astype(np.uint8)
    depth = (800 + x * 3 + y * 2).astype(np.uint16)
    depth[(x - width // 2) ** 2 + (y - height // 2) ** 2 < 18 ** 2] = 0
    intrinsics = SimpleNamespace(
        width=width, height=height, fx=145.0, fy=145.0,
        ppx=width / 2, ppy=height / 2, model="none", coeffs=[0.0] * 5,
    )

    publisher = RosbridgePublisher("ws://127.0.0.1:9090", timeout=10)
    try:
        publisher.advertise()
        print("[demo] Publishing a synthetic RGB-D scene at 5 Hz", flush=True)
        while True:
            messages = make_frame_bundle(
                color, depth, intrinsics, 0.001, stamp_from_unix_ns(time.time_ns())
            )
            for topic, message in zip((COLOR_TOPIC, CAMERA_INFO_TOPIC, DEPTH_TOPIC), messages):
                publisher.publish(topic, message)
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 0
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
