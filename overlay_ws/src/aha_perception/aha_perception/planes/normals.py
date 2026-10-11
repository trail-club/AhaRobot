"""Surface normal estimation for organized and unorganized clouds."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2r+1)x(2r+1) window using an integral image (zero padded)."""
    pad = [(radius + 1, radius), (radius + 1, radius)] + [(0, 0)] * (values.ndim - 2)
    integral = np.pad(values, pad).cumsum(axis=0).cumsum(axis=1)
    k = 2 * radius + 1
    return integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]


def _shift(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[i, j] = values[i + dy, j + dx], NaN outside the image."""
    out = np.full_like(values, np.nan)
    h, w = values.shape[:2]
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    ys_src = slice(max(0, dy), min(h, h + dy))
    xs_src = slice(max(0, dx), min(w, w + dx))
    out[ys, xs] = values[ys_src, xs_src]
    return out


def depth_discontinuities(
    depth: np.ndarray,
    inverse_sigma: float,
    k: float = 6.0,
    smoothing_radius: int = 2,
    spacing: int = 3,
) -> np.ndarray:
    """Mark pixels next to depth jumps.

    Inverse depth is linear in pixel coordinates on any plane, so its second
    difference vanishes on planes regardless of their slant, and its stereo
    noise is constant. Pixels whose box-smoothed second difference (taken
    ``spacing`` pixels apart) exceeds ``k * inverse_sigma`` are
    discontinuities. Holes are not discontinuities.
    """
    valid = np.isfinite(depth) & (depth > 0)
    inv = np.where(valid, 1.0 / np.where(valid, depth, 1.0), 0.0)
    count = box_sum(valid.astype(np.float64), smoothing_radius)
    window = (2 * smoothing_radius + 1) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(
            count >= window / 2, box_sum(inv, smoothing_radius) / count, np.nan
        )
    smooth[~valid] = np.nan
    tau = k * inverse_sigma
    s = spacing
    d2u = _shift(smooth, 0, s) + _shift(smooth, 0, -s) - 2.0 * smooth
    d2v = _shift(smooth, s, 0) + _shift(smooth, -s, 0) - 2.0 * smooth
    with np.errstate(invalid="ignore"):
        jump = (np.abs(d2u) > tau) | (np.abs(d2v) > tau)
    return valid & jump


def orient_towards(normals: np.ndarray, points: np.ndarray, viewpoint) -> np.ndarray:
    """Flip normals so that they face ``viewpoint``."""
    to_view = np.asarray(viewpoint, dtype=np.float64) - points
    flip = np.einsum("...i,...i->...", normals, to_view) < 0
    out = normals.copy()
    out[flip] *= -1.0
    return out


def organized_normals(
    points: np.ndarray,
    depth: np.ndarray,
    inverse_sigma: float,
    radius: int = 2,
    smoothing_radius: int = 2,
    discontinuity_k: float = 6.0,
    viewpoint=None,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Average-3D-gradient normals of an organized (H, W, 3) cloud.

    Points are box-smoothed with integral images, then the normal is the cross
    product of central differences ``radius`` pixels apart. Pixels whose
    support crosses a depth discontinuity are masked. Returns (normals, mask);
    normals are NaN where the mask is False.
    """
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth)
    filled = np.where(valid[..., None], points, 0.0)
    count = box_sum(valid.astype(np.float64), smoothing_radius)
    window = (2 * smoothing_radius + 1) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = box_sum(filled, smoothing_radius) / count[..., None]
    smooth[(count < min_valid_fraction * window) | ~valid] = np.nan

    du = _shift(smooth, 0, radius) - _shift(smooth, 0, -radius)
    dv = _shift(smooth, radius, 0) - _shift(smooth, -radius, 0)
    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = normals / norm[..., None]

    disc = depth_discontinuities(depth, inverse_sigma, discontinuity_k)
    reach = radius + smoothing_radius
    near_disc = box_sum(disc.astype(np.float64), reach) > 0
    mask = valid & np.isfinite(normals).all(axis=-1) & (norm > 0) & ~near_disc
    normals[~mask] = np.nan
    if viewpoint is not None:
        normals[mask] = orient_towards(normals[mask], points[mask], viewpoint)
    return normals, mask


def knn_normals(
    points: np.ndarray, k: int = 16, viewpoint=None
) -> tuple[np.ndarray, np.ndarray]:
    """PCA normals of an unorganized (N, 3) cloud from k nearest neighbours.

    Returns (normals, curvature) where curvature = l0 / (l0 + l1 + l2).
    Without ``viewpoint`` the sign of a normal is arbitrary.
    """
    n = len(points)
    if n < 3:
        return np.full((n, 3), np.nan), np.full(n, np.nan)
    k = min(k, n)
    _, idx = cKDTree(points).query(points, k=k)
    neigh = points[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    x, y, z = centered[..., 0], centered[..., 1], centered[..., 2]
    cov = [
        np.einsum("nk,nk->n", u, v) / k
        for u, v in ((x, x), (x, y), (x, z), (y, y), (y, z), (z, z))
    ]
    smallest, normals, total = smallest_eigh3(*cov)
    with np.errstate(invalid="ignore", divide="ignore"):
        curvature = np.where(total > 0, np.maximum(smallest, 0.0) / total, 0.0)
    if viewpoint is not None:
        normals = orient_towards(normals, points, viewpoint)
    return normals, curvature


def smallest_eigh3(cxx, cxy, cxz, cyy, cyz, czz):
    """Smallest eigenvalue, its unit eigenvector and the trace of symmetric
    3x3 matrices given by their 6 components (arrays of shape (N,)).

    Closed form (trigonometric solution of the characteristic cubic) instead
    of a batched ``eigh``; the eigenvector is the longest cross product of
    two rows of ``C - lambda I``, (0, 0, 1) if all vanish.
    """
    q = (cxx + cyy + czz) / 3.0
    a, b, c = cxx - q, cyy - q, czz - q
    off_diagonal = cxy * cxy + cxz * cxz + cyz * cyz
    p = np.sqrt((a * a + b * b + c * c + 2.0 * off_diagonal) / 6.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(p > 0, 1.0 / p, 0.0)
        ba, bb, bc = a * inv, b * inv, c * inv
        bxy, bxz, byz = cxy * inv, cxz * inv, cyz * inv
        det = ba * (bb * bc - byz * byz) - bxy * (bxy * bc - byz * bxz)
        det += bxz * (bxy * byz - bb * bxz)
        phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
    smallest = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    r0 = np.stack([cxx - smallest, cxy, cxz], axis=-1)
    r1 = np.stack([cxy, cyy - smallest, cyz], axis=-1)
    r2 = np.stack([cxz, cyz, czz - smallest], axis=-1)
    crosses = np.stack([np.cross(r0, r1), np.cross(r0, r2), np.cross(r1, r2)])
    lengths = np.einsum("cni,cni->cn", crosses, crosses)
    best = np.argmax(lengths, axis=0)
    vec = np.take_along_axis(crosses, best[None, :, None], axis=0)[0]
    length = np.sqrt(np.take_along_axis(lengths, best[None], axis=0)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        vec = np.where(length[:, None] > 0, vec / length[:, None], [0.0, 0.0, 1.0])
    return smallest, vec, 3.0 * q
