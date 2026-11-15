"""Optional Open3D reference detectors (used only when open3d is importable).

Both run on the same voxelised cloud as the RANSAC candidates and hand their
segments to the shared ``planes_from_labels`` so that refinement, filters and
classification match the candidates.
"""

from __future__ import annotations

import numpy as np

from aha_perception.planes.preprocess import prepare_frame, voxel_downsample
from aha_perception.planes.segmentation import planes_from_labels


def available() -> bool:
    try:
        import open3d  # noqa: F401
    except ImportError:
        return False
    return True


def _voxels(cloud, options, stride, voxel_size):
    frame = prepare_frame(cloud, options, stride)
    pixels = np.flatnonzero(frame.valid)
    voxels, inverse, _ = voxel_downsample(
        frame.points.reshape(-1, 3)[pixels], voxel_size
    )
    return frame, pixels, voxels, inverse


def _to_planes(frame, pixels, inverse, voxel_labels, options):
    labels = np.full(frame.valid.shape, -1, dtype=np.int32)
    labels.reshape(-1)[pixels] = voxel_labels[inverse]
    return planes_from_labels(frame, labels, options)


def segment_plane_loop(
    cloud,
    options,
    voxel_size=0.02,
    distance_threshold=0.02,
    iterations=500,
    min_points=100,
    max_rounds=12,
):
    """Repeated ``PointCloud.segment_plane`` with inlier removal."""
    import open3d as o3d

    o3d.utility.random.seed(0)
    frame, pixels, voxels, inverse = _voxels(cloud, options, 2, voxel_size)
    voxel_labels = np.full(len(voxels), -1, dtype=np.int32)
    remaining = np.arange(len(voxels))
    for label in range(max_rounds):
        if len(remaining) < min_points:
            break
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voxels[remaining]))
        _, inliers = pcd.segment_plane(distance_threshold, 3, iterations)
        inliers = np.asarray(inliers, dtype=np.int64)
        if len(inliers) < min_points:
            break
        voxel_labels[remaining[inliers]] = label
        remaining = np.delete(remaining, inliers)
    return _to_planes(frame, pixels, inverse, voxel_labels, options)


def planar_patches(cloud, options, voxel_size=0.02, knn=30):
    """``PointCloud.detect_planar_patches`` with points assigned by patch box."""
    import open3d as o3d

    frame, pixels, voxels, inverse = _voxels(cloud, options, 2, voxel_size)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voxels))
    search = o3d.geometry.KDTreeSearchParamKNN(knn)
    pcd.estimate_normals(search)
    pcd.orient_normals_towards_camera_location(frame.camera_origin)
    patches = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0.0,
        min_num_points=0,
        search_param=search,
    )
    voxel_labels = np.full(len(voxels), -1, dtype=np.int32)
    order = sorted(range(len(patches)), key=lambda i: -np.prod(patches[i].extent))
    for label, i in enumerate(order):
        idx = np.asarray(
            patches[i].get_point_indices_within_bounding_box(pcd.points), dtype=np.int64
        )
        idx = idx[voxel_labels[idx] < 0]
        voxel_labels[idx] = label
    return _to_planes(frame, pixels, inverse, voxel_labels, options)


DETECTORS = {
    "open3d_segment_plane": segment_plane_loop,
    "open3d_planar_patches": planar_patches,
}
