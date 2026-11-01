"""Plane detection metrics against per-pixel ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as connected_label
from scipy.optimize import linear_sum_assignment

from scenes import DONT_CARE, INVALID, NON_PLANAR, SceneFrame

SUPPORT_CLASSES = ("floor", "table", "shelf")
_INFEASIBLE = 1e6


@dataclass(frozen=True)
class MatchParams:
    max_angle_deg: float = 10.0
    max_offset: float = 0.05
    min_iou: float = 0.2
    ignore_fraction: float = 0.5
    member_fraction: float = 0.5
    member_cover: float = 0.05
    under_cover: float = 0.2
    horizontal_tol_deg: float = 10.0


def _overlaps(frame: SceneFrame, detections):
    """Pixel intersections (G, D), detection sizes, don't-care counts, GT sizes."""
    labels = frame.labels.reshape(-1)
    n_gt = len(frame.planes)
    gt_size = np.bincount(labels[labels >= 0], minlength=n_gt)[:n_gt]
    inter = np.zeros((n_gt, len(detections)))
    det_size = np.zeros(len(detections))
    dont_care = np.zeros(len(detections))
    offset = -min(INVALID, DONT_CARE, NON_PLANAR)
    for k, det in enumerate(detections):
        hits = np.bincount(labels[det.inliers] + offset, minlength=n_gt + offset)
        inter[:, k] = hits[offset : offset + n_gt]
        det_size[k] = len(det.inliers)
        dont_care[k] = hits[DONT_CARE + offset]
    return inter, det_size, dont_care, gt_size


def _holes(labels: np.ndarray, inliers: np.ndarray, gt_index: int) -> tuple[int, int]:
    """Holes of a detection mask that contain valid, unassigned pixels of its GT.

    Returns (number of such holes, number of those GT pixels inside them).
    Holes made only of invalid or other-surface pixels are not counted.
    """
    if len(inliers) == 0:
        return 0, 0
    width = labels.shape[1]
    rows, cols = np.divmod(inliers, width)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    mask = np.zeros((r1 - r0 + 2, c1 - c0 + 2), dtype=bool)
    mask[rows - r0 + 1, cols - c0 + 1] = True
    holes = binary_fill_holes(mask) & ~mask
    gt = np.pad(labels[r0:r1, c0:c1] == gt_index, 1)
    missing = holes & gt
    if not missing.any():
        return 0, 0
    components, _ = connected_label(holes)
    return int(np.unique(components[missing]).size), int(missing.sum())


def _height_error(det, gt) -> float:
    """Signed height of the detected plane minus GT at the GT centroid (m)."""
    n, d = det.normal, det.offset
    x, y = gt.centroid[:2]
    return float(-(n[0] * x + n[1] * y + d) / n[2] - gt.centroid[2])


def evaluate(frame: SceneFrame, detections, params: MatchParams | None = None) -> dict:
    """Match detections to GT planes and return summable per-frame statistics.

    ``matches`` (GT index, detection index) and ``ignored`` detection indices
    are included for visualisation.
    """
    params = params or MatchParams()
    gts = frame.planes
    inter, det_size, dont_care, gt_size = _overlaps(frame, detections)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = inter / (gt_size[:, None] + det_size[None, :] - inter)
        ignored = dont_care / np.maximum(det_size, 1) >= params.ignore_fraction
    active = np.flatnonzero(~ignored)

    up = np.array([0.0, 0.0, 1.0])
    cos_h = np.cos(np.radians(params.horizontal_tol_deg))
    sin_v = np.sin(np.radians(params.horizontal_tol_deg))
    gt_horizontal = np.array([abs(g.normal @ up) >= cos_h for g in gts], dtype=bool)
    gt_vertical = np.array([abs(g.normal @ up) <= sin_v for g in gts], dtype=bool)
    det_horizontal = np.array(
        [abs(d.normal @ up) >= cos_h for d in detections], dtype=bool
    )

    angle = np.zeros((len(gts), len(detections)))
    offset_err = np.zeros_like(angle)
    for g, gt in enumerate(gts):
        for k, det in enumerate(detections):
            cos = min(1.0, abs(float(gt.normal @ det.normal)))
            angle[g, k] = np.degrees(np.arccos(cos))
            offset_err[g, k] = abs(float(det.normal @ gt.centroid) + det.offset)

    matches = []
    if len(gts) and len(active):
        sub = np.ix_(np.arange(len(gts)), active)
        feasible = (
            (angle[sub] <= params.max_angle_deg)
            & (offset_err[sub] <= params.max_offset)
            & (iou[sub] >= params.min_iou)
        )
        cost = (
            angle[sub] / params.max_angle_deg
            + offset_err[sub] / params.max_offset
            + (1.0 - iou[sub])
        )
        cost = np.where(feasible, cost, _INFEASIBLE)
        rows, cols = linear_sum_assignment(cost)
        matches = [(g, active[c]) for g, c in zip(rows, cols) if feasible[g, c]]

    stats = {
        "n_gt": len(gts),
        "n_det": len(detections),
        "n_ignored": int(ignored.sum()),
        "tp": len(matches),
        "fp": len(active) - len(matches),
        "fn": len(gts) - len(matches),
        "iou_sum": 0.0,
        "angle_sum": 0.0,
        "offset_sum": 0.0,
        "cls_correct": 0,
        "support_n": 0,
        "support_abs_sum": 0.0,
        "h_gt": int(gt_horizontal.sum()),
        "h_tp": 0,
        "h_fp": 0,
        "h_iou_sum": 0.0,
        "h_abs_sum": 0.0,
        "v_gt": int(gt_vertical.sum()),
        "v_tp": 0,
    }
    stats.update(completeness_sum=0.0, holes=0, hole_px=0, matched_gt_px=0)
    matched_dets = set()
    for g, k in matches:
        gt, det = gts[g], detections[k]
        matched_dets.add(k)
        holes, hole_px = _holes(frame.labels, det.inliers, g)
        stats["completeness_sum"] += float(inter[g, k] / max(gt_size[g], 1))
        stats["holes"] += holes
        stats["hole_px"] += hole_px
        stats["matched_gt_px"] += int(gt_size[g])
        stats["iou_sum"] += float(iou[g, k])
        stats["angle_sum"] += float(angle[g, k])
        stats["offset_sum"] += float(offset_err[g, k])
        stats["cls_correct"] += int(det.label.value == gt.cls)
        if gt_horizontal[g]:
            height = abs(_height_error(det, gt))
            stats["h_tp"] += 1
            stats["h_iou_sum"] += float(iou[g, k])
            stats["h_abs_sum"] += height
            if gt.cls in SUPPORT_CLASSES:
                stats["support_n"] += 1
                stats["support_abs_sum"] += height
        if gt_vertical[g]:
            stats["v_tp"] += 1
    stats["h_fp"] = int(
        sum(1 for k in active if det_horizontal[k] and k not in matched_dets)
    )

    over = 0
    for g in range(len(gts)):
        with np.errstate(invalid="ignore", divide="ignore"):
            inside = inter[g, active] / np.maximum(det_size[active], 1)
        cover = inter[g, active] / max(gt_size[g], 1)
        members = (inside >= params.member_fraction) & (cover >= params.member_cover)
        over += int(members.sum() >= 2)
    under = 0
    for k in active:
        covered = inter[:, k] / np.maximum(gt_size, 1) >= params.under_cover
        under += int(covered.sum() >= 2)
    stats["over_seg"] = over
    stats["under_seg"] = under
    stats["matches"] = matches
    stats["ignored"] = [int(k) for k in np.flatnonzero(ignored)]
    return stats


def aggregate(rows: list[dict]) -> dict:
    """Micro-averaged metrics over per-frame statistics (with ``runtime_ms``)."""

    def total(key):
        return float(sum(r[key] for r in rows))

    tp, fp, fn = total("tp"), total("fp"), total("fn")
    h_tp, h_fp, h_gt = total("h_tp"), total("h_fp"), total("h_gt")
    runtimes = np.array([r["runtime_ms"] for r in rows], dtype=np.float64)

    def ratio(a, b):
        return a / b if b else float("nan")

    recall = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    h_recall = ratio(h_tp, h_gt)
    h_precision = ratio(h_tp, h_tp + h_fp)
    return {
        "frames": len(rows),
        "gt_planes": int(tp + fn),
        "pq": ratio(total("iou_sum"), tp + 0.5 * fp + 0.5 * fn),
        "recall": recall,
        "precision": precision,
        "f1": ratio(2 * precision * recall, precision + recall),
        "miou_matched": ratio(total("iou_sum"), tp),
        "completeness": ratio(total("completeness_sum"), tp),
        "completeness_all": ratio(total("completeness_sum"), tp + fn),
        "holes_per_plane": ratio(total("holes"), tp),
        "hole_px_share": ratio(total("hole_px"), total("matched_gt_px")),
        "angle_deg": ratio(total("angle_sum"), tp),
        "offset_cm": 100 * ratio(total("offset_sum"), tp),
        "support_height_cm": 100 * ratio(total("support_abs_sum"), total("support_n")),
        "class_acc": ratio(total("cls_correct"), tp),
        "over_seg": int(total("over_seg")),
        "under_seg": int(total("under_seg")),
        "h_pq": ratio(total("h_iou_sum"), h_tp + 0.5 * h_fp + 0.5 * (h_gt - h_tp)),
        "h_recall": h_recall,
        "h_precision": h_precision,
        "h_miou": ratio(total("h_iou_sum"), h_tp),
        "h_height_cm": 100 * ratio(total("h_abs_sum"), h_tp),
        "v_recall": ratio(total("v_tp"), total("v_gt")),
        "runtime_mean_ms": float(runtimes.mean()) if len(runtimes) else float("nan"),
        "runtime_p95_ms": float(np.percentile(runtimes, 95))
        if len(runtimes)
        else float("nan"),
    }
