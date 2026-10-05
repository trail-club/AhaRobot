"""Conversions between ROS message payloads and the plane detectors.

Kept free of rclpy so it can be tested on the host: depth images to organized
clouds, TF to 4x4 matrices, plane colors and the packed XYZRGB point layout
used by sensor_msgs/PointCloud2.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import numpy as np

from aha_perception.planes.types import Plane, PlaneLabel

DEPTH_ENCODINGS = ("32FC1", "16UC1", "mono16")

LABEL_COLORS: dict[PlaneLabel, tuple[float, float, float, float]] = {
    PlaneLabel.FLOOR: (0.6, 0.6, 0.6, 1.0),
    PlaneLabel.TABLE: (0.1, 0.8, 0.2, 1.0),
    PlaneLabel.SHELF: (1.0, 0.65, 0.0, 1.0),
    PlaneLabel.WALL: (0.2, 0.45, 1.0, 1.0),
    PlaneLabel.OTHER: (0.7, 0.3, 0.9, 1.0),
}

# x, y, z, rgb as float32: the layout RViz's RGB8 color transformer reads.
POINT_DTYPE = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)]
)

# |normal . up| below this: the plane is reported as vertical.
VERTICAL_MAX_COS = 0.5


def depth_mm_to_m(depth: np.ndarray) -> np.ndarray:
    """16-bit millimeter depth to float32 meters (0 stays 0, i.e. invalid)."""
    return np.asarray(depth, dtype=np.float32) * np.float32(1e-3)


def depth_from_buffer(
    data: bytes,
    height: int,
    width: int,
    step: int,
    encoding: str,
    is_bigendian: bool = False,
) -> np.ndarray:
    """Decode a sensor_msgs/Image depth payload to (H, W) float32 meters."""
    if encoding == "32FC1":
        base = np.dtype(np.float32)
    elif encoding in ("16UC1", "mono16"):
        base = np.dtype(np.uint16)
    else:
        raise ValueError(f"unsupported depth encoding {encoding!r}")
    dtype = base.newbyteorder(">" if is_bigendian else "<")
    row_items = step // dtype.itemsize
    if step % dtype.itemsize or row_items < width:
        raise ValueError(f"step {step} does not fit width {width} of {encoding}")
    rows = np.frombuffer(data, dtype=dtype, count=height * row_items)
    image = rows.reshape(height, row_items)[:, :width]
    if base == np.uint16:
        return depth_mm_to_m(image)
    return image.astype(np.float32)


def depth_to_buffer(depth: np.ndarray, encoding: str) -> tuple[bytes, int]:
    """Depth in meters (non-finite = invalid) to a little-endian payload and step.

    32FC1 keeps the values (NaN for invalid); 16UC1 rounds to millimeters with
    0 for invalid.
    """
    if encoding == "32FC1":
        image = np.where(np.isfinite(depth), depth, np.nan).astype("<f4")
    elif encoding == "16UC1":
        mm = np.where(np.isfinite(depth), np.round(depth * 1e3), 0.0)
        image = np.clip(mm, 0, np.iinfo(np.uint16).max).astype("<u2")
    else:
        raise ValueError(f"unsupported depth encoding {encoding!r}")
    return image.tobytes(), image.shape[1] * image.itemsize


def depth_to_cloud(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int = 1
) -> np.ndarray:
    """Back-project a depth image (meters) to an organized optical-frame cloud.

    Keeps every ``stride``-th row and column; pixel coordinates stay those of
    the full image, so the intrinsics are not rescaled. Pixels with zero,
    negative or non-finite depth become NaN.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    depth = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
    rows, cols = depth.shape
    u = np.arange(cols, dtype=np.float32) * stride
    v = np.arange(rows, dtype=np.float32) * stride
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.float32(np.nan))
    cloud = np.empty((rows, cols, 3), dtype=np.float32)
    cloud[..., 0] = (u[None, :] - np.float32(cx)) / np.float32(fx) * z
    cloud[..., 1] = (v[:, None] - np.float32(cy)) / np.float32(fy) * z
    cloud[..., 2] = z
    return cloud


def transform_to_matrix(
    translation_xyz: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    """4x4 homogeneous matrix from a translation and a (x, y, z, w) quaternion."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return matrix


def label_color(label: PlaneLabel | str) -> tuple[float, float, float, float]:
    """RGBA in [0, 1] for a plane label."""
    return LABEL_COLORS[PlaneLabel(label)]


def plane_palette(index: int) -> tuple[float, float, float]:
    """Distinct RGB in [0, 1] for the ``index``-th plane (golden-ratio hues)."""
    hue = (0.13 + index * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.75, 0.95)


def is_vertical(plane: Plane, up: Sequence[float] = (0.0, 0.0, 1.0)) -> bool:
    return abs(float(np.dot(plane.normal, up))) < VERTICAL_MAX_COS


def plane_text(plane: Plane, up: Sequence[float] = (0.0, 0.0, 1.0)) -> str:
    """Marker text: height for horizontal planes, area for vertical ones."""
    label = PlaneLabel(plane.label).value
    if is_vertical(plane, up):
        return f"{label} {plane.area:.1f} m2"
    return f"{label} {float(np.dot(plane.centroid, up)):.2f} m"


def plane_points(
    planes: Sequence[Plane],
    cloud: np.ndarray,
    camera_to_base: np.ndarray,
    colors: Sequence[Sequence[float]],
    max_points_per_plane: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inlier points of every plane in the output frame with per-plane colors.

    ``cloud`` is the organized cloud passed to the detector, ``colors`` holds
    one RGB in [0, 1] per plane. With ``max_points_per_plane > 0`` larger
    planes are randomly subsampled. Returns (N, 3) float32 xyz and (N, 3)
    uint8 rgb.
    """
    if len(colors) != len(planes):
        raise ValueError("need one color per plane")
    flat = np.asarray(cloud).reshape(-1, 3)
    transform = np.asarray(camera_to_base, dtype=np.float64)
    rng = np.random.default_rng(seed)
    xyz_parts, rgb_parts = [], []
    for plane, color in zip(planes, colors):
        inliers = np.asarray(plane.inliers, dtype=np.int64)
        if max_points_per_plane > 0 and inliers.size > max_points_per_plane:
            inliers = rng.choice(inliers, max_points_per_plane, replace=False)
        points = flat[inliers].astype(np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        xyz_parts.append(points @ transform[:3, :3].T + transform[:3, 3])
        rgb = np.clip(np.round(np.asarray(color[:3]) * 255.0), 0, 255)
        rgb_parts.append(np.tile(rgb.astype(np.uint8), (len(points), 1)))
    if not xyz_parts:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    return (
        np.concatenate(xyz_parts).astype(np.float32),
        np.concatenate(rgb_parts),
    )


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 to float32 holding 0x00RRGGBB, as PCL / RViz expect."""
    rgb = np.asarray(rgb, dtype=np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    return packed.astype(np.uint32).view(np.float32)


def xyzrgb_records(xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Structured (N,) array in POINT_DTYPE; ``.tobytes()`` is PointCloud2 data."""
    records = np.empty(len(xyz), dtype=POINT_DTYPE)
    records["x"], records["y"], records["z"] = np.asarray(xyz, np.float32).T
    records["rgb"] = pack_rgb(rgb)
    return records
