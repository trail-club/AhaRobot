"""Read and write real depth captures for offline plane detection.

A scene is a directory with ``frames.npz`` (depth in millimeters as uint16,
0 = invalid; camera-to-target_frame transforms; stamps; intrinsics; head
joints), ``meta.json`` and optionally ``color.jpg`` (first frame). Only numpy
and the standard library are needed, so scenes can be read on the host.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
FRAMES_FILE = "frames.npz"
META_FILE = "meta.json"
COLOR_FILE = "color.jpg"
HEAD_JOINT_NAMES = ("joint_head_pan", "joint_head_tilt")
MAX_DEPTH_MM = np.iinfo(np.uint16).max


@dataclass
class RealScene:
    path: Path
    depth: np.ndarray  # (N, H, W) float32 meters, NaN = invalid
    camera_to_base: np.ndarray  # (N, 4, 4) depth optical frame -> target_frame
    stamps: np.ndarray  # (N,) seconds
    k: np.ndarray  # (9,) row-major camera matrix
    head_joints: np.ndarray  # (N, 2) pan, tilt in radians, NaN = unknown
    meta: dict[str, Any]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def n_frames(self) -> int:
        return len(self.depth)

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy)."""
        return float(self.k[0]), float(self.k[4]), float(self.k[2]), float(self.k[5])

    @property
    def base_height(self) -> float | None:
        """Height of the target_frame origin above the floor, if recorded."""
        value = self.meta.get("base_height")
        return None if value is None else float(value)

    def camera_to_floor(self) -> np.ndarray:
        """``camera_to_base`` raised by ``base_height`` so the floor is z = 0."""
        transforms = self.camera_to_base.copy()
        if self.base_height is not None:
            transforms[:, 2, 3] += self.base_height
        return transforms

    @property
    def color_path(self) -> Path | None:
        path = self.path / COLOR_FILE
        return path if path.is_file() else None


def scene_dir_name(scene: str, when: datetime.datetime) -> str:
    """``YYYYMMDD-HHMM_<scene>``."""
    if not scene or "/" in scene or scene.startswith("."):
        raise ValueError(f"invalid scene name {scene!r}")
    return f"{when:%Y%m%d-%H%M}_{scene}"


def depth_to_mm(depth: np.ndarray) -> np.ndarray:
    """Depth in meters (float) or millimeters (uint16) to uint16 mm, 0 = invalid."""
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth.copy()
    mm = np.asarray(depth, dtype=np.float64) * 1000.0
    valid = np.isfinite(mm) & (mm >= 0.5) & (mm <= MAX_DEPTH_MM)
    out = np.zeros(mm.shape, dtype=np.uint16)
    out[valid] = np.round(mm[valid]).astype(np.uint16)
    return out


def save_scene(
    scene_dir: str | Path,
    depth: np.ndarray,
    camera_to_base: np.ndarray,
    stamps: np.ndarray,
    k: np.ndarray,
    meta: dict[str, Any],
    head_joints: np.ndarray | None = None,
    head_joint_names: tuple[str, str] = HEAD_JOINT_NAMES,
    color_jpeg: bytes | None = None,
) -> Path:
    """Write a scene directory; refuses to overwrite an existing scene.

    ``depth`` is (N, H, W) in meters (NaN / <= 0 invalid) or uint16 mm.
    ``head_joints`` holds (pan, tilt) per frame in the order of
    ``head_joint_names``. ``frame_count``, ``width``, ``height``,
    ``format_version`` and ``head_joint_names`` are added to ``meta``.
    """
    scene_dir = Path(scene_dir)
    depth_mm = depth_to_mm(depth)
    if depth_mm.ndim != 3 or len(depth_mm) == 0:
        raise ValueError(f"depth must be (N, H, W) with N >= 1, got {depth_mm.shape}")
    n, height, width = depth_mm.shape
    camera_to_base = np.asarray(camera_to_base, dtype=np.float64)
    stamps = np.asarray(stamps, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64).reshape(-1)
    if head_joints is None:
        head_joints = np.full((n, 2), np.nan)
    head_joints = np.asarray(head_joints, dtype=np.float64)
    if camera_to_base.shape != (n, 4, 4):
        raise ValueError(f"camera_to_base must be ({n}, 4, 4)")
    if stamps.shape != (n,) or k.shape != (9,) or head_joints.shape != (n, 2):
        raise ValueError("stamps (N,), k (9,) and head_joints (N, 2) required")
    if (scene_dir / FRAMES_FILE).exists():
        raise FileExistsError(f"{scene_dir / FRAMES_FILE} already exists")

    meta = dict(meta)
    meta.update(
        format_version=FORMAT_VERSION,
        frame_count=n,
        width=width,
        height=height,
        depth_unit="mm",
        head_joint_names=list(head_joint_names),
    )
    meta["color_file"] = COLOR_FILE if color_jpeg is not None else None

    scene_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scene_dir / FRAMES_FILE,
        depth_mm=depth_mm,
        camera_to_base=camera_to_base,
        stamps=stamps,
        k=k,
        head_joints=head_joints,
    )
    if color_jpeg is not None:
        (scene_dir / COLOR_FILE).write_bytes(color_jpeg)
    (scene_dir / META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    return scene_dir


def load_scene(path: str | Path) -> RealScene:
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text())
    with np.load(path / FRAMES_FILE) as data:
        depth_mm = data["depth_mm"]
        depth = depth_mm.astype(np.float32) * np.float32(1e-3)
        depth[depth_mm == 0] = np.nan
        return RealScene(
            path=path,
            depth=depth,
            camera_to_base=data["camera_to_base"],
            stamps=data["stamps"],
            k=data["k"],
            head_joints=data["head_joints"],
            meta=meta,
        )


def list_scenes(root: str | Path) -> list[Path]:
    """Scene directories directly under ``root``, sorted by name."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if (p / FRAMES_FILE).is_file() and (p / META_FILE).is_file()
    )
