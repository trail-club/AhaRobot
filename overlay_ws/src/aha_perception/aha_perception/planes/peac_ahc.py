"""Candidate PEAC: agglomerative hierarchical clustering of image nodes.

Clean-room implementation after Feng, Taguchi and Kamat, "Fast plane
extraction in organized point clouds using agglomerative hierarchical
clustering" (ICRA 2014), written from the paper's description.

1. The organized cloud is divided into square nodes. Nodes with missing data,
   a depth discontinuity or a plane-fit MSE above ``T_MSE(z)`` are dropped.
2. Adjacent nodes form a graph. The node with the smallest MSE is popped from
   a min-heap and merged with the neighbour whose union has the smallest MSE,
   while that MSE is below ``T_MSE(z) = (sigma z^2 + eps)^2``. A node that
   cannot merge is extracted as a plane if it has ``min_nodes`` nodes.
   Plane fits use incremental sums of points and outer products.
3. Boundary nodes are eroded and the planes are grown back per pixel.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from .cells import Moments, cell_grid
from .preprocess import prepare_frame
from .refine import EIGHT, assign_pixels, keep_seeded_components
from .segmentation import planes_from_labels
from .types import DetectionOptions, Plane

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass
class PeacParams:
    node_size: int = 10
    max_missing_fraction: float = 0.05
    discontinuity_k: float = 6.0
    mse_k: float = 1.5
    mse_eps: float = 0.005
    min_nodes: int = 4
    refine_k: float = 3.0


def _t_mse(depth, params, options):
    sigma = params.mse_k * options.noise.inverse_sigma
    return np.square(sigma * np.square(depth) + params.mse_eps)


def ahc(grid, usable, params, options) -> list[Moments]:
    """Agglomerative clustering on the node graph; returns extracted planes.

    Cluster sums live in arrays indexed by cluster id so that the merge MSE
    against all neighbours is one batched eigenvalue call.
    """
    rows, cols = np.nonzero(usable)
    n = len(rows)
    capacity = 2 * n + 1
    count = np.zeros(capacity)
    total = np.zeros((capacity, 3))
    outer = np.zeros((capacity, 3, 3))
    depth = np.zeros(capacity)
    count[:n] = grid.count[rows, cols]
    total[:n] = grid.total[rows, cols]
    outer[:n] = grid.outer[rows, cols]
    depth[:n] = grid.depth_total[rows, cols]
    members: list[list | None] = [[(int(i), int(j))] for i, j in zip(rows, cols)]
    members += [None] * (capacity - n)
    alive = np.zeros(capacity, dtype=bool)
    alive[:n] = True

    node_id = -np.ones((grid.rows, grid.cols), dtype=np.int64)
    node_id[rows, cols] = np.arange(n)
    neighbours: dict[int, set[int]] = {cid: set() for cid in range(n)}
    for a, b in ((node_id[1:, :], node_id[:-1, :]), (node_id[:, 1:], node_id[:, :-1])):
        linked = (a >= 0) & (b >= 0)
        for x, y in zip(a[linked].tolist(), b[linked].tolist()):
            neighbours[x].add(y)
            neighbours[y].add(x)
    heap = [(float(m), cid) for cid, m in enumerate(grid.mse[rows, cols])]
    heapq.heapify(heap)

    planes = []
    next_id = n
    while heap:
        _, cid = heapq.heappop(heap)
        if not alive[cid]:
            continue
        nbrs = np.fromiter(neighbours[cid], dtype=np.int64)
        if nbrs.size:
            c = count[cid] + count[nbrs]
            mean = (total[cid] + total[nbrs]) / c[:, None]
            cov = (outer[cid] + outer[nbrs]) / c[:, None, None]
            cov -= np.einsum("ni,nj->nij", mean, mean)
            mses = np.linalg.eigvalsh(cov)[:, 0]
            k = int(np.argmin(mses))
            best, best_mse = int(nbrs[k]), float(mses[k])
            merged_depth = (depth[cid] + depth[best]) / c[k]
            if best_mse < _t_mse(merged_depth, params, options):
                new = next_id
                next_id += 1
                for array in (count, total, outer, depth):
                    array[new] = array[cid] + array[best]
                big, small = sorted((cid, best), key=lambda i: -len(members[i]))
                members[new] = members[big]
                members[new].extend(members[small])
                members[big] = members[small] = None
                alive[[cid, best]] = False
                alive[new] = True
                links = (neighbours.pop(cid) | neighbours.pop(best)) - {cid, best}
                for other in links:
                    neighbours[other] -= {cid, best}
                    neighbours[other].add(new)
                neighbours[new] = links
                heapq.heappush(heap, (best_mse, new))
                continue
        if len(members[cid]) >= params.min_nodes:
            plane = Moments()
            plane.add(members[cid], count[cid], total[cid], outer[cid], depth[cid])
            plane.update()
            planes.append(plane)
        alive[cid] = False
        for other in neighbours.pop(cid):
            neighbours[other].discard(cid)
    return planes


def detect(
    cloud: np.ndarray,
    options: DetectionOptions | None = None,
    params: PeacParams | None = None,
) -> list[Plane]:
    options = options or DetectionOptions()
    params = params or PeacParams()
    frame = prepare_frame(cloud, options, 1)
    grid = cell_grid(frame, params.node_size, options, params.discontinuity_k)
    if grid.rows == 0 or grid.cols == 0:
        return []
    area = params.node_size**2
    usable = (
        (grid.count >= (1.0 - params.max_missing_fraction) * area)
        & (grid.edge_count == 0)
        & (grid.mse < _t_mse(grid.mean_depth, params, options))
    )
    planes = ahc(grid, usable, params, options)

    candidates, seeds = [], []
    for plane in planes:
        mask = plane.mask(grid.rows, grid.cols)
        core = binary_erosion(mask, _FOUR, border_value=0)
        seeds.append(np.flatnonzero(grid.pixel_mask(core if core.any() else mask)))
        candidates.append(grid.pixel_mask(binary_dilation(mask, EIGHT)))
    models = [(p.normal, p.offset) for p in planes]
    labels = assign_pixels(frame, models, candidates, params.refine_k, options)
    labels = keep_seeded_components(labels, seeds)
    return planes_from_labels(frame, labels, options, params.refine_k)
