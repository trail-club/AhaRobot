"""Cell (node) grids over an organized cloud with additive point moments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import plane_from_moments
from .normals import depth_discontinuities
from .preprocess import Frame
from .types import DetectionOptions


def tile(values: np.ndarray, size: int, rows: int, cols: int) -> np.ndarray:
    """(rows*size, cols*size, ...) -> (rows, cols, size*size, ...)."""
    trailing = values.shape[2:]
    cropped = values[: rows * size, : cols * size]
    tiles = cropped.reshape(rows, size, cols, size, *trailing).swapaxes(1, 2)
    return tiles.reshape(rows, cols, size * size, *trailing)


@dataclass
class CellGrid:
    """Per-cell sums of points, outer products and depths, plus a PCA fit.

    ``mse`` is the smallest covariance eigenvalue (mean squared distance to the
    cell's own plane). Cells beyond the last full row/column are not covered.
    """

    size: int
    rows: int
    cols: int
    shape: tuple[int, int]
    count: np.ndarray
    total: np.ndarray
    outer: np.ndarray
    depth_total: np.ndarray
    edge_count: np.ndarray
    mean: np.ndarray
    mean_depth: np.ndarray
    eigvals: np.ndarray
    normals: np.ndarray

    @property
    def mse(self) -> np.ndarray:
        return self.eigvals[..., 0]

    def stats(self, i: int, j: int):
        return (
            self.count[i, j],
            self.total[i, j],
            self.outer[i, j],
            self.depth_total[i, j],
        )

    def mse_to_plane(self, i: int, j: int, normal: np.ndarray, offset: float) -> float:
        """Mean squared distance of the cell's points to a given plane."""
        n, d = normal, offset
        value = (n @ self.outer[i, j] @ n + 2.0 * d * (n @ self.total[i, j])) / max(
            self.count[i, j], 1.0
        )
        return float(value + d * d)

    def pixel_mask(self, cell_mask: np.ndarray) -> np.ndarray:
        """Expand a (rows, cols) cell mask to the frame's working resolution."""
        out = np.zeros(self.shape, dtype=bool)
        up = np.repeat(np.repeat(cell_mask, self.size, axis=0), self.size, axis=1)
        out[: up.shape[0], : up.shape[1]] = up
        return out


def cell_grid(
    frame: Frame, size: int, options: DetectionOptions, discontinuity_k: float = 6.0
) -> CellGrid:
    h, w = frame.valid.shape
    rows, cols = h // size, w // size
    valid = frame.valid
    cell_points = tile(np.where(valid[..., None], frame.points, 0.0), size, rows, cols)
    count = tile(valid, size, rows, cols).sum(axis=-1).astype(np.float64)
    depth_total = tile(np.where(valid, frame.depth, 0.0), size, rows, cols).sum(-1)
    total = cell_points.sum(axis=2)
    outer = np.einsum("abki,abkj->abij", cell_points, cell_points)
    edges = depth_discontinuities(
        frame.depth, options.noise.inverse_sigma, discontinuity_k
    )
    edge_count = tile(edges, size, rows, cols).sum(axis=-1)
    safe = np.maximum(count, 1.0)
    mean = total / safe[..., None]
    cov = outer / safe[..., None, None] - np.einsum("abi,abj->abij", mean, mean)
    cov[count < 3] = np.eye(3)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return CellGrid(
        size=size,
        rows=rows,
        cols=cols,
        shape=(h, w),
        count=count,
        total=total,
        outer=outer,
        depth_total=depth_total,
        edge_count=edge_count,
        mean=mean,
        mean_depth=depth_total / safe,
        eigvals=eigvals,
        normals=eigvecs[..., 0],
    )


class Moments:
    """Accumulated point moments of a set of cells (incremental PCA by sums)."""

    def __init__(self):
        self.count = 0.0
        self.total = np.zeros(3)
        self.outer = np.zeros((3, 3))
        self.depth_total = 0.0
        self.cells: list[tuple[int, int]] = []

    def add(self, cells, count, total, outer, depth_total):
        self.cells.extend(cells)
        self.count += float(count)
        self.total += total
        self.outer += outer
        self.depth_total += float(depth_total)

    def add_cell(self, grid: CellGrid, i: int, j: int):
        self.add([(i, j)], *grid.stats(i, j))

    def absorb(self, other: Moments):
        self.add(other.cells, other.count, other.total, other.outer, other.depth_total)

    def update(self):
        self.normal, self.offset, self.centroid, eigvals = plane_from_moments(
            self.count, self.total, self.outer
        )
        self.mse = float(eigvals[0])

    @property
    def mean_depth(self) -> float:
        return self.depth_total / self.count

    def mask(self, rows: int, cols: int) -> np.ndarray:
        mask = np.zeros((rows, cols), dtype=bool)
        if self.cells:
            idx = np.array(self.cells)
            mask[idx[:, 0], idx[:, 1]] = True
        return mask
