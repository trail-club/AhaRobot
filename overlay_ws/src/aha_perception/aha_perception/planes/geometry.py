"""Plane fitting and small geometric helpers shared by the detectors."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Least-squares plane through (N, 3) points.

    Returns (unit normal, offset, centroid, ascending covariance eigenvalues).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    return normal, -float(normal @ centroid), centroid, eigvals


def plane_from_moments(count: float, total: np.ndarray, outer: np.ndarray):
    """Plane from accumulated sum and sum of outer products of points.

    Returns (normal, offset, centroid, eigvals).
    """
    centroid = total / count
    cov = outer / count - np.outer(centroid, centroid)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    return normal, -float(normal @ centroid), centroid, eigvals


def face_viewpoint(normal: np.ndarray, offset: float, viewpoint) -> tuple:
    """Orient (normal, offset) so that the viewpoint is on the positive side."""
    if float(normal @ np.asarray(viewpoint)) + offset < 0:
        return -normal, -offset
    return normal, offset


def plane_axes(normal: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal in-plane basis (2, 3); the second axis follows ``up`` if possible."""
    helper = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float)
    if abs(float(normal @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(normal @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(helper, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return np.stack([first, second])


_HULL_PREFILTER_MIN_POINTS = 64


def _outside_octagon(points: np.ndarray) -> np.ndarray:
    """Mask of points not strictly inside the polygon of the extreme points in
    8 directions (Akl-Toussaint); the others cannot be hull vertices."""
    x, y = points[:, 0], points[:, 1]
    s, d = x + y, x - y
    ids = [
        int(np.argmax(x)),
        int(np.argmax(s)),
        int(np.argmax(y)),
        int(np.argmin(d)),
        int(np.argmin(x)),
        int(np.argmin(s)),
        int(np.argmin(y)),
        int(np.argmax(d)),
    ]
    ids = [i for k, i in enumerate(ids) if i != ids[k - 1]]
    if len(ids) < 3:
        return np.ones(len(points), dtype=bool)
    corners = points[ids]
    edges = np.roll(corners, -1, axis=0) - corners
    # Margin against rounding: points on or near an edge stay candidates.
    eps = 1e-9 * (float(np.abs(points).max()) or 1.0) ** 2
    inside = np.ones(len(points), dtype=bool)
    for (ax, ay), (ex, ey) in zip(corners, edges):
        inside &= ex * (y - ay) - ey * (x - ax) > eps
    return ~inside


def convex_hull_2d(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Counter-clockwise hull vertices and area of (N, 2) points."""
    if len(points) < 3:
        return points.copy(), 0.0
    if len(points) >= _HULL_PREFILTER_MIN_POINTS:
        points = points[_outside_octagon(points)]
    try:
        hull = ConvexHull(points)
    except QhullError:
        return points[:0].copy(), 0.0
    return points[hull.vertices], float(hull.volume)


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_convex_polygons(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersection of two counter-clockwise convex polygons (Sutherland-Hodgman)."""
    output = [tuple(p) for p in subject]
    n = len(clip)
    for i in range(n):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % n]
        edge = b - a

        def inside(p, a=a, edge=edge):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0

        def intersect(p, q, a=a, edge=edge):
            p, q = np.asarray(p), np.asarray(q)
            d = q - p
            denom = edge[0] * d[1] - edge[1] * d[0]
            if denom == 0:
                return tuple(p)
            t = (edge[0] * (a[1] - p[1]) - edge[1] * (a[0] - p[0])) / denom
            return tuple(p + t * d)

        inputs, output = output, []
        prev = inputs[-1]
        for cur in inputs:
            if inside(cur):
                if not inside(prev):
                    output.append(intersect(prev, cur))
                output.append(cur)
            elif inside(prev):
                output.append(intersect(prev, cur))
            prev = cur
    return np.array(output, dtype=np.float64).reshape(-1, 2)


def ransac_plane(
    points: np.ndarray,
    threshold,
    iterations: int,
    rng: np.random.Generator,
    normal_ok: Callable[[np.ndarray], np.ndarray] | None = None,
    sample_pool: np.ndarray | None = None,
    score_points: int = 3000,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Vectorised RANSAC plane fit with one least-squares refinement.

    ``threshold`` is a scalar or a per-point array. ``normal_ok`` filters
    hypotheses by their normals, ``sample_pool`` restricts minimal samples to a
    subset of indices. Hypotheses are scored on a random subset of points.
    Returns (normal, offset, inlier mask) or None.
    """
    n = len(points)
    if n < 3:
        return None
    threshold = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (n,))
    pool = np.arange(n) if sample_pool is None else sample_pool
    if len(pool) < 3:
        return None
    samples = pool[rng.integers(0, len(pool), size=(iterations, 3))]
    p0, p1, p2 = (points[samples[:, i]] for i in range(3))
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1)
    good = norms > 1e-9
    normals = normals[good] / norms[good, None]
    p0 = p0[good]
    if normal_ok is not None:
        keep = normal_ok(normals)
        normals, p0 = normals[keep], p0[keep]
    if len(normals) == 0:
        return None
    offsets = -np.einsum("ij,ij->i", normals, p0)

    subset = np.arange(n)
    if n > score_points:
        subset = rng.choice(n, size=score_points, replace=False)
    dist = np.abs(points[subset] @ normals.T + offsets)
    scores = (dist < threshold[subset, None]).sum(axis=0)
    best = int(np.argmax(scores))
    normal, offset = normals[best], offsets[best]

    inliers = np.abs(points @ normal + offset) < threshold
    if inliers.sum() >= 3:
        refined, refined_offset, _, _ = fit_plane(points[inliers])
        if normal_ok is None or normal_ok(refined[None])[0]:
            refined_inliers = np.abs(points @ refined + refined_offset) < threshold
            if refined_inliers.sum() >= inliers.sum():
                normal, offset, inliers = refined, refined_offset, refined_inliers
    return normal, float(offset), inliers


def point_components(points: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean connected components with a fixed neighbour radius."""
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def angle_to_axis_cos(normals: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """|cos| of the angle between each normal and ``axis``."""
    return np.abs(normals @ axis)
