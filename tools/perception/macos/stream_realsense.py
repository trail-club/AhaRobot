#!/usr/bin/env python3
"""Stream RealSense RGB-D frames to ROS 2 through rosbridge.

The Mac-only boundary ends at three standard camera topics:

* RGB image
* depth aligned to the RGB image
* RGB CameraInfo

PointCloud2 generation intentionally stays in the ROS 2 container, where
``depth_image_proc`` consumes these topics. No ROS Python package is required
on the Mac.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


COLOR_TOPIC = "/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"
DEPTH_TOPIC = "/camera/depth_registered/image_rect"

COLOR_FRAME = "camera_color_optical_frame"

ROS_IMAGE = "sensor_msgs/msg/Image"
ROS_CAMERA_INFO = "sensor_msgs/msg/CameraInfo"

SENSOR_QOS = {
    "history": "keep_last",
    "depth": 5,
    # depth_image_proc's Jazzy image_transport subscriptions use reliable
    # reliability by default. Keep the bridge publisher compatible without
    # adding a ROS-specific QoS override configuration to the Mac process.
    "reliability": "reliable",
    "durability": "volatile",
}


class StreamError(RuntimeError):
    """An expected setup, device, or transport failure."""


@dataclass(frozen=True)
class Stamp:
    sec: int
    nanosec: int

    def as_message(self) -> dict[str, int]:
        return {"sec": self.sec, "nanosec": self.nanosec}


def stamp_from_unix_ns(unix_ns: int) -> Stamp:
    """Convert a Unix timestamp to the ROS 2 builtin_interfaces/Time shape."""

    sec, nanosec = divmod(int(unix_ns), 1_000_000_000)
    return Stamp(sec=sec, nanosec=nanosec)


def _header(stamp: Stamp, frame_id: str) -> dict[str, Any]:
    return {"stamp": stamp.as_message(), "frame_id": frame_id}


def _encoded_bytes(data: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(data)
    return base64.b64encode(contiguous.tobytes(order="C")).decode("ascii")


def make_image_message(
    image: np.ndarray,
    *,
    encoding: str,
    stamp: Stamp,
    frame_id: str,
) -> dict[str, Any]:
    """Create a sensor_msgs/msg/Image dictionary with a base64 data field."""

    array = np.ascontiguousarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"image must have 2 or 3 dimensions, got {array.shape}")

    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    if encoding == "rgb8" and (array.dtype != np.uint8 or channels != 3):
        raise ValueError("rgb8 image must be a uint8 HxWx3 array")
    if encoding == "32FC1" and (array.dtype != np.float32 or channels != 1):
        raise ValueError("32FC1 image must be a float32 HxW array")

    return {
        "header": _header(stamp, frame_id),
        "height": int(height),
        "width": int(width),
        "encoding": encoding,
        "is_bigendian": 0,
        "step": int(width * array.dtype.itemsize * channels),
        "data": _encoded_bytes(array),
    }


def depth_to_meters(depth_z16: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert RealSense Z16 depth to 32FC1 metres.

    A zero Z16 sample is the RealSense invalid-depth value.  It is represented
    as NaN in the ROS image so consumers do not mistake it for a zero-metre
    measurement.
    """

    raw = np.asarray(depth_z16)
    if raw.ndim != 2:
        raise ValueError(f"depth must be a HxW array, got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"depth must contain integer Z16 samples, got {raw.dtype}")

    meters = np.full(raw.shape, np.nan, dtype=np.float32)
    valid = raw != 0
    meters[valid] = raw[valid].astype(np.float32) * np.float32(depth_scale)
    return meters


def _ros_distortion_model(model: Any) -> str:
    """Map a RealSense distortion enum to a sensor_msgs/CameraInfo model."""

    name = str(model).rsplit(".", 1)[-1].lower()
    if name in {"ftheta", "kannala_brandt4"}:
        return "equidistant"
    # RealSense Brown-Conrady variants use the same coefficient layout that
    # ROS represents as plumb_bob. The coefficients themselves are preserved.
    return "plumb_bob"


def make_camera_info_message(
    intrinsics: Any,
    *,
    stamp: Stamp,
    frame_id: str,
) -> dict[str, Any]:
    """Create a sensor_msgs/msg/CameraInfo dictionary from RS intrinsics."""

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    coefficients = [float(value) for value in intrinsics.coeffs]

    return {
        "header": _header(stamp, frame_id),
        "height": int(intrinsics.height),
        "width": int(intrinsics.width),
        "distortion_model": _ros_distortion_model(intrinsics.model),
        "d": coefficients,
        "k": [fx, 0.0, ppx, 0.0, fy, ppy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, ppx, 0.0, 0.0, fy, ppy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "binning_x": 0,
        "binning_y": 0,
        "roi": {
            "x_offset": 0,
            "y_offset": 0,
            "height": 0,
            "width": 0,
            "do_rectify": False,
        },
    }


def make_frame_bundle(
    color_rgb: np.ndarray,
    depth_z16: np.ndarray,
    intrinsics: Any,
    depth_scale: float,
    stamp: Stamp,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the three synchronized ROS messages from one aligned frame pair."""

    if color_rgb.shape[:2] != depth_z16.shape or color_rgb.shape[:2] != (
        intrinsics.height,
        intrinsics.width,
    ):
        raise StreamError("aligned depth, color, and camera intrinsics must have the same size")

    return (
        make_image_message(color_rgb, encoding="rgb8", stamp=stamp, frame_id=COLOR_FRAME),
        make_camera_info_message(intrinsics, stamp=stamp, frame_id=COLOR_FRAME),
        make_image_message(
            depth_to_meters(depth_z16, depth_scale),
            encoding="32FC1",
            stamp=stamp,
            frame_id=COLOR_FRAME,
        ),
    )


def advertise_operation(topic: str, message_type: str) -> dict[str, Any]:
    """Return an explicit rosbridge advertisement for a sensor topic."""

    return {
        "op": "advertise",
        "topic": topic,
        "type": message_type,
        "qos": dict(SENSOR_QOS),
    }


class RosbridgePublisher:
    """Small synchronous rosbridge client with no frame queue."""

    def __init__(self, url: str, timeout: float) -> None:
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - exercised by setup
            raise StreamError(
                "websocket-client is not installed; run "
                "python -m pip install -r tools/perception/macos/requirements.txt"
            ) from exc

        try:
            self._socket = websocket.create_connection(url, timeout=timeout)
        except Exception as exc:
            raise StreamError(
                f"could not connect to rosbridge at {url}: {exc}. "
                "Start rosbridge_server in the ROS container first."
            ) from exc

    def _send(self, operation: dict[str, Any]) -> None:
        try:
            self._socket.send(json.dumps(operation, separators=(",", ":")))
        except Exception as exc:
            raise StreamError(f"rosbridge connection failed while sending data: {exc}") from exc

    def advertise(self) -> None:
        self._send(advertise_operation(COLOR_TOPIC, ROS_IMAGE))
        self._send(advertise_operation(CAMERA_INFO_TOPIC, ROS_CAMERA_INFO))
        self._send(advertise_operation(DEPTH_TOPIC, ROS_IMAGE))

    def publish(self, topic: str, message: dict[str, Any]) -> None:
        self._send({"op": "publish", "topic": topic, "msg": message})

    def close(self) -> None:
        try:
            self._socket.close()
        except Exception:
            pass


def _load_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except (ImportError, OSError) as exc:  # pragma: no cover - exercised by setup
        raise StreamError(
            f"pyrealsense2 could not be loaded by {sys.executable}: {exc}. "
            "The RealSense SDK Python binding must match this Python version "
            "and CPU architecture."
        ) from exc
    return rs


def _device_label(rs: Any, device: Any) -> str:
    name = "RealSense"
    try:
        name = device.get_info(rs.camera_info.name)
    except Exception:
        pass
    try:
        serial = device.get_info(rs.camera_info.serial_number)
    except Exception:
        serial = "unknown serial"
    return f"{name} (serial {serial})"


def _list_devices(rs: Any) -> list[Any]:
    try:
        context = rs.context()
        devices = list(context.query_devices())
    except Exception as exc:
        raise StreamError(
            "RealSense USB access failed while enumerating devices: "
            f"{exc}. On macOS 12+, check the USB permission/admin requirement "
            "for libusb and close other RealSense applications."
        ) from exc
    if not devices:
        raise StreamError(
            "no RealSense device found. Check the USB 3 cable, macOS USB "
            "permission, and that another RealSense application is closed."
        )
    return devices


def _color_frame_rgb(frame: Any, color_format: str) -> np.ndarray:
    image = np.asanyarray(frame.get_data())
    if color_format == "bgr8":
        image = image[..., ::-1]
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise StreamError(f"unexpected color frame shape/format: {image.shape} {image.dtype}")
    return image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", default="ws://127.0.0.1:9090", help="rosbridge WebSocket URL")
    parser.add_argument("--serial", help="RealSense serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15, help="camera capture FPS")
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="maximum frame bundles sent per second (no unbounded queue)",
    )
    parser.add_argument(
        "--color-format",
        choices=("rgb8", "bgr8"),
        default="rgb8",
        help="format requested from the camera; output is always rgb8",
    )
    parser.add_argument("--ws-timeout", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, help="stop after sending this many bundles")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--list-devices", action="store_true", help="list connected cameras and exit")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.width < 1 or args.height < 1 or args.fps < 1:
        raise StreamError("width, height, and fps must be positive")
    if args.rate <= 0.0 or not math.isfinite(args.rate):
        raise StreamError("rate must be a finite number greater than zero")
    if args.max_frames is not None and args.max_frames < 1:
        raise StreamError("max-frames must be >= 1")
    if args.duration is not None and args.duration <= 0.0:
        raise StreamError("duration must be greater than zero")


def stream(args: argparse.Namespace) -> int:
    _validate_args(args)
    rs = _load_realsense()
    devices = _list_devices(rs)

    if args.list_devices:
        for device in devices:
            print(_device_label(rs, device))
        return 0

    if args.serial:
        matching = []
        for device in devices:
            try:
                if device.get_info(rs.camera_info.serial_number) == args.serial:
                    matching.append(device)
            except Exception:
                continue
        if not matching:
            available = ", ".join(_device_label(rs, device) for device in devices)
            raise StreamError(f"RealSense serial {args.serial!r} was not found; available: {available}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    requested_color_format = getattr(rs.format, args.color_format)
    config.enable_stream(rs.stream.color, args.width, args.height, requested_color_format, args.fps)

    publisher = RosbridgePublisher(args.ws_url, args.ws_timeout)
    align_to_color = rs.align(rs.stream.color)
    started = False
    sent = 0
    started_at = time.monotonic()
    last_report = started_at

    try:
        try:
            profile = pipeline.start(config)
            started = True
        except Exception as exc:
            raise StreamError(
                f"RealSense pipeline could not start with "
                f"{args.width}x{args.height}@{args.fps}: {exc}"
            ) from exc

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        actual_depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        actual_color = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_width = actual_depth.width()
        depth_height = actual_depth.height()
        color_width = actual_color.width()
        color_height = actual_color.height()
        print(
            f"[stream] {_device_label(rs, profile.get_device())}; "
            f"depth={depth_width}x{depth_height}, color={color_width}x{color_height}, "
            f"depth_scale={depth_scale:g}; aligned_depth={color_width}x{color_height}; "
            f"ws={args.ws_url}; rate={args.rate:g} Hz",
            flush=True,
        )

        publisher.advertise()
        interval = 1.0 / args.rate
        next_send = time.monotonic()

        while True:
            try:
                frames = pipeline.wait_for_frames()
            except Exception as exc:
                raise StreamError(f"RealSense frame acquisition failed: {exc}") from exc
            now = time.monotonic()
            if now < next_send:
                continue

            try:
                aligned_frames = align_to_color.process(frames)
            except Exception as exc:
                raise StreamError(f"RealSense depth-to-color alignment failed: {exc}") from exc

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # One acquisition timestamp is shared by the RGB-D bundle.
            stamp = stamp_from_unix_ns(time.time_ns())
            color_rgb = _color_frame_rgb(color_frame, args.color_format)
            depth_raw = np.asanyarray(depth_frame.get_data())
            messages = make_frame_bundle(
                color_rgb, depth_raw, actual_color.get_intrinsics(), depth_scale, stamp
            )

            # Synchronous sends intentionally provide back-pressure.  Frames
            # that arrive while the socket is busy are discarded by the loop;
            # they are never accumulated in a Python-side queue.
            for topic, message in zip(
                (COLOR_TOPIC, CAMERA_INFO_TOPIC, DEPTH_TOPIC), messages
            ):
                publisher.publish(topic, message)
            sent += 1

            current = time.monotonic()
            if current - last_report >= 2.0:
                elapsed = current - started_at
                print(
                    f"[stream] sent={sent}, elapsed={elapsed:.1f}s, "
                    f"send_hz={sent / elapsed:.2f}",
                    flush=True,
                )
                last_report = current

            if args.max_frames is not None and sent >= args.max_frames:
                break
            if args.duration is not None and current - started_at >= args.duration:
                break
            next_send = max(next_send + interval, time.monotonic())

        print(f"[stream] finished after {sent} frame bundle(s)", flush=True)
        return 0
    finally:
        publisher.close()
        if started:
            try:
                pipeline.stop()
            except Exception:
                pass


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return stream(args)
    except KeyboardInterrupt:
        print("\n[stream] stopped", file=sys.stderr)
        return 130
    except StreamError as exc:
        print(f"[stream] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
