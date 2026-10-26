"""Synthetic organized clouds for the plane detection tests."""

import numpy as np
import pytest

from aha_perception.planes import DepthNoiseModel, DetectionOptions

WIDTH, HEIGHT = 640, 480
FX = FY = 465.7
CX, CY = 319.5, 239.5


def camera_to_base(height=1.43, tilt_deg=35.0, pan_deg=0.0):
    """Head camera pose; positive tilt looks down. Optical z forward, y down."""
    optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    t, p = np.radians(tilt_deg), np.radians(pan_deg)
    pitch = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    yaw = np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = yaw @ pitch @ optical
    transform[:3, 3] = (0.0, 0.0, height)
    return transform


def render(horizontal=(), vertical=(), transform=None, noise_scale=0.5, seed=0):
    """Ray-cast axis-aligned rectangles into an organized cloud (camera frame).

    ``horizontal``: (z, xmin, xmax, ymin, ymax); ``vertical``: (x, ymin, ymax,
    zmin, zmax) for walls facing -x.
    """
    transform = camera_to_base() if transform is None else transform
    u, v = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    rays_cam = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u, float)], -1)
    rays = rays_cam @ transform[:3, :3].T
    origin = transform[:3, 3]
    depth = np.full((HEIGHT, WIDTH), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for z, x0, x1, y0, y1 in horizontal:
            t = (z - origin[2]) / rays[..., 2]
            hit = origin + t[..., None] * rays
            inside = (hit[..., 0] >= x0) & (hit[..., 0] <= x1)
            inside &= (hit[..., 1] >= y0) & (hit[..., 1] <= y1) & (t > 0)
            depth = np.where(inside & (t < depth), t, depth)
        for x, y0, y1, z0, z1 in vertical:
            t = (x - origin[0]) / rays[..., 0]
            hit = origin + t[..., None] * rays
            inside = (hit[..., 1] >= y0) & (hit[..., 1] <= y1)
            inside &= (hit[..., 2] >= z0) & (hit[..., 2] <= z1) & (t > 0)
            depth = np.where(inside & (t < depth), t, depth)
    rng = np.random.default_rng(seed)
    finite = np.isfinite(depth)
    sigma = DepthNoiseModel().sigma(np.where(finite, depth, 0.0))
    depth = depth + noise_scale * sigma * rng.standard_normal(depth.shape)
    depth[~finite] = np.nan
    return rays_cam * depth[..., None]


@pytest.fixture
def options():
    return DetectionOptions(camera_to_base=camera_to_base())


@pytest.fixture
def render_cloud():
    return render


@pytest.fixture
def camera_pose():
    return camera_to_base
