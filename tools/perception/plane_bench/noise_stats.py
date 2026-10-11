"""Depth noise statistics of a static view, shared by real and synthetic frames.

``compute`` takes a stack of depth frames (N x H x W, meters, NaN invalid) of
a static camera and returns the temporal noise level, its spatial correlation,
the frame-frozen deviation from large planes and the invalid / flicker pixel
statistics. Noise is normalized by Z^2 (disparity-like units), so the stereo
model sigma(Z) = Z^2 * subpixel / (f * b) has a constant normalized level.

- temporal subpixel: per-pixel temporal std of pixels valid in every frame
  and at least EDGE_MARGIN px from depth edges and invalid pixels, as the
  median of sigma / Z^2 * f * b per DEPTH_BIN (bias-corrected for N frames);
  the median over the bins in FIT_RANGE.
- correlated fraction / correlation px / aspect: c * exp(-d^2 / 4s^2) fitted
  to the row and column autocorrelation (lags TEMPORAL_LAGS) of frame minus
  temporal mean in CORR_RANGE; px is the vertical s, aspect horizontal / vertical.
- static: residual of a plane fit on the temporal mean over large planes
  found by ``ransac_gravity`` (PLANE_MIN_PIXELS, beyond 0.75 m), minus the
  temporal contribution sigma^2 / N. Static level is its std in subpixel
  units, static / temporal the ratio; its correlation length comes from the
  autocorrelation with the temporal part removed.
- invalid / never-valid / flicker: invalid fraction over all frames, pixels
  invalid in every frame, pixels invalid in some frames (of pixels ever valid).
- flicker rate / decay px: flicker per distance d to never-valid pixels as a
  per-frame dropout probability, fitted by q(d) = rate * exp(-(d - 1) / decay);
  pixels beyond 6 px count as isolated dropout.
- gross outlier: fraction of samples farther than 8 MAD and 10 % of the depth
  from the per-pixel median.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as connected_label
from scipy.optimize import least_squares
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
_PERCEPTION = HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"
if str(_PERCEPTION) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION))

BASELINE = 0.05  # m
DEPTH_BIN = 0.25  # m, temporal std table
BIN_MIN_PIXELS = 500
FIT_RANGE = (0.75, 3.5)  # m; below 0.75 m the 1 mm depth quantization dominates
FIT_MIN_PIXELS = 2000
CORR_RANGE = (1.0, 3.5)  # m, depth range for the temporal correlation
TEMPORAL_LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24)
STATIC_LAGS = (1, 2, 4, 8, 12, 16, 24, 32, 48)
EDGE_STEP = 0.02  # relative depth step to a 4-neighbor that marks a depth edge
EDGE_MARGIN = 2  # px removed around depth edges and invalid pixels
PLANE_MAX_DEPTH = 3.5  # m
PLANE_MIN_PIXELS = 15000  # a 0.1 m2 desk top at 0.4 m already has ~150k px
STATIC_MIN_DEPTH = FIT_RANGE[0]
PLANE_METHOD = "ransac_gravity"
CLIP_K = 4.0
FLICKER_EDGES = (0.0, 1.5, 2.5, 3.5, 4.5, 6.0, np.inf)  # px to a never-valid pixel
INVALID_BIN = 0.5  # m
OUTLIER_REL = 0.1


@dataclass
class PlaneStat:
    pixels: int
    depth: float  # mean depth (m)
    static_ratio: float  # frozen deviation std / temporal std


@dataclass
class NoiseStats:
    n_frames: int
    focal: float
    # (bin start m, pixels, median temporal std mm, subpixel estimate)
    depth_bins: list = field(default_factory=list)
    subpixel_temporal: float = float("nan")
    acorr_rows: np.ndarray | None = None  # at TEMPORAL_LAGS
    acorr_cols: np.ndarray | None = None
    correlated_fraction: float = float("nan")
    correlation_px: float = float("nan")  # vertical Gaussian sigma
    correlation_aspect: float = float("nan")  # horizontal / vertical sigma
    baseline: float = BASELINE
    planes: list = field(default_factory=list)
    static_pixels: int = 0  # pixels of the planes pooled for the static stats
    static_ratio: float = float("nan")  # frozen deviation std / temporal std
    static_level: float = float("nan")  # frozen deviation std in subpixel units
    static_acorr: np.ndarray | None = None  # frozen part only, at STATIC_LAGS
    static_correlation_px: float = float("nan")
    invalid_fraction: float = float("nan")
    never_valid_fraction: float = float("nan")
    flicker_fraction: float = float("nan")
    # (lo px, hi px, share of ever-valid pixels, flicker fraction, mean distance)
    flicker_by_distance: list = field(default_factory=list)
    # (bin start m, pixels, per-frame invalid rate)
    invalid_by_depth: list = field(default_factory=list)
    flicker_rate: float = float("nan")  # per-frame dropout 1 px from never-valid
    flicker_px: float = float("nan")  # decay length of that dropout
    isolated_dropout: float = float("nan")  # per-frame dropout > 6 px away
    outlier_fraction: float = float("nan")  # gross per-frame depth errors

    @property
    def static_fraction(self) -> float:
        r2 = self.static_ratio**2
        return r2 / (1.0 + r2) if np.isfinite(r2) else float("nan")


def _std_bias(n: int) -> float:
    """Median of the sample std of n Gaussian samples, in units of sigma."""
    return float(np.sqrt(chi2.median(n - 1) / (n - 1)))


def smooth_mask(mean: np.ndarray) -> np.ndarray:
    """Valid pixels away from depth edges and invalid pixels."""
    valid = np.isfinite(mean)
    filled = np.where(valid, mean, 0.0)
    edge = ~valid
    for axis in (0, 1):
        step = np.abs(np.diff(filled, axis=axis))
        near = np.minimum(np.delete(filled, -1, axis), np.delete(filled, 0, axis))
        jump = step > EDGE_STEP * near
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        edge |= np.pad(jump, pad)
        pad[axis] = (1, 0)
        edge |= np.pad(jump, pad)
    return valid & ~binary_dilation(edge, iterations=EDGE_MARGIN)


def autocorrelation(field_, mask, lags) -> tuple[np.ndarray, np.ndarray]:
    """Normalized autocorrelation along rows and columns, pooled over frames.

    ``field_`` is (H, W) or (N, H, W) and zero-mean per pixel set; only pairs
    with both pixels in ``mask`` count.
    """
    if field_.ndim == 2:
        field_ = field_[None]
    out = np.full((2, len(lags)), np.nan)
    for i, lag in enumerate(lags):
        for axis in (0, 1):
            a = np.delete(field_, np.s_[-lag:], axis=axis + 1)
            b = np.delete(field_, np.s_[:lag], axis=axis + 1)
            ok = np.delete(mask, np.s_[-lag:], axis=axis) & np.delete(
                mask, np.s_[:lag], axis=axis
            )
            if ok.sum() < 100:
                continue
            aa, bb = a[:, ok], b[:, ok]
            denom = np.sqrt((aa**2).sum() * (bb**2).sum())
            # axis 0 shifts along columns (vertical), axis 1 along rows.
            out[1 - axis, i] = (aa * bb).sum() / denom if denom > 0 else np.nan
    return out[0], out[1]


def _gauss(lags, c, sigma):
    return c * np.exp(-np.square(lags) / (4.0 * sigma**2))


def fit_correlation(lags, rows, cols):
    """Fit c * exp(-d^2 / (4 s^2)) with separate row / column sigma.

    Returns (c, vertical sigma, horizontal / vertical sigma).
    """
    lags = np.asarray(lags, dtype=np.float64)
    ok_r, ok_c = np.isfinite(rows), np.isfinite(cols)
    if ok_r.sum() < 2 or ok_c.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    def residual(x):
        c, sy, sx = x
        return np.concatenate(
            [
                _gauss(lags[ok_r], c, sx) - rows[ok_r],
                _gauss(lags[ok_c], c, sy) - cols[ok_c],
            ]
        )

    fit = least_squares(
        residual, x0=(0.9, 3.0, 3.0), bounds=([0.0, 0.3, 0.3], [1.0, 80.0, 80.0])
    )
    c, sy, sx = fit.x
    return float(c), float(sy), float(sx / sy)


def fit_isotropic(lags, values):
    """Fit c * exp(-d^2 / (4 s^2)); returns (c, s)."""
    lags = np.asarray(lags, dtype=np.float64)
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    fit = least_squares(
        lambda x: _gauss(lags[ok], *x) - values[ok],
        x0=(0.9, 5.0),
        bounds=([0.0, 0.3], [1.5, 200.0]),
    )
    return float(fit.x[0]), float(fit.x[1])


def _rays(shape, intrinsics):
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[: shape[0], : shape[1]]
    return np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(shape)], -1)


def _clipped(values):
    """Values kept by iterative clipping at CLIP_K robust sigmas."""
    keep = np.ones(values.shape, dtype=bool)
    for _ in range(5):
        med = np.median(values[keep])
        sigma = 1.4826 * np.median(np.abs(values[keep] - med))
        new = np.abs(values - med) <= CLIP_K * max(sigma, 1e-12)
        if (new == keep).all():
            break
        keep = new
    return keep


def _large_planes(mean, intrinsics, camera_to_base):
    """Pixel masks of large planes detected on the temporal-mean depth."""
    from aha_perception.planes import DepthNoiseModel, DetectionOptions, detect_planes

    cloud = (_rays(mean.shape, intrinsics) * mean[..., None]).astype(np.float32)
    options = DetectionOptions(
        camera_to_base=camera_to_base,
        max_depth=PLANE_MAX_DEPTH,
        noise=DepthNoiseModel(focal=intrinsics[0]),
    )
    masks = []
    for plane in detect_planes(cloud, PLANE_METHOD, options):
        mask = np.zeros(mean.size, dtype=bool)
        mask[plane.inliers] = True
        mask = binary_fill_holes(binary_closing(mask.reshape(mean.shape), iterations=3))
        comp, count = connected_label(mask)
        if count == 0:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        mask = comp == 1 + int(np.argmax(sizes))
        if mask.sum() >= PLANE_MIN_PIXELS:
            masks.append(mask)
    return masks


def _plane_residual(mean, mask, rays):
    """Depth residual of the temporal mean from a clipped LSQ plane, / Z^2."""
    pts = rays[mask] * mean[mask][:, None]
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(5):
        centroid = pts[keep].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[keep] - centroid, full_matrices=False)
        normal = vt[2]
        offset = -float(normal @ centroid)
        plane_z = -offset / (rays[mask] @ normal)
        resid = (mean[mask] - plane_z) / mean[mask] ** 2
        new = _clipped(resid)
        if (new == keep).all():
            break
        keep = new
    return resid, keep


def compute(
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    camera_to_base: np.ndarray,
    baseline: float = BASELINE,
) -> NoiseStats:
    """Statistics of a static depth sequence (N, H, W) in meters, NaN invalid."""
    depth = np.asarray(depth, dtype=np.float64)
    n = len(depth)
    focal = float(intrinsics[0])
    stats = NoiseStats(n_frames=n, focal=focal, baseline=baseline)
    finite = np.isfinite(depth)
    all_valid = finite.all(axis=0)
    any_valid = finite.any(axis=0)
    stats.invalid_fraction = float(1.0 - finite.mean())
    stats.never_valid_fraction = float(1.0 - any_valid.mean())
    flicker = any_valid & ~all_valid
    stats.flicker_fraction = float(flicker.sum() / max(int(any_valid.sum()), 1))

    distance = distance_transform_edt(any_valid)
    for lo, hi in zip(FLICKER_EDGES[:-1], FLICKER_EDGES[1:]):
        sel = any_valid & (distance > lo) & (distance <= hi)
        if sel.sum() == 0:
            continue
        stats.flicker_by_distance.append(
            (
                lo,
                hi,
                float(sel.sum() / any_valid.sum()),
                float(flicker[sel].mean()),
                float(distance[sel].mean()),
            )
        )
    _fit_flicker(stats, n)

    counts = finite.sum(axis=0)
    ever_mean = np.where(finite, depth, 0.0).sum(axis=0) / np.maximum(counts, 1)
    rate = 1.0 - finite.mean(axis=0)
    bins = np.floor(ever_mean / INVALID_BIN)
    for b in np.unique(bins[any_valid]).astype(int):
        sel = any_valid & (bins == b)
        if sel.sum() >= BIN_MIN_PIXELS:
            stats.invalid_by_depth.append(
                (b * INVALID_BIN, int(sel.sum()), float(rate[sel].mean()))
            )

    if n < 2 or not all_valid.any():
        return stats
    stats.outlier_fraction = _outliers(depth, all_valid)
    mean = np.where(all_valid, depth.mean(axis=0), np.nan)
    std = np.where(all_valid, depth.std(axis=0, ddof=1), np.nan)
    smooth = smooth_mask(mean)
    bias = _std_bias(n)
    scale = focal * baseline
    estimates = []
    bins = np.floor(mean / DEPTH_BIN)
    for b in np.unique(bins[smooth]).astype(int):
        sel = smooth & (bins == b)
        if sel.sum() < BIN_MIN_PIXELS:
            continue
        sub = float(np.median(std[sel] / mean[sel] ** 2) * scale / bias)
        start = b * DEPTH_BIN
        stats.depth_bins.append(
            (start, int(sel.sum()), float(np.median(std[sel]) * 1e3), sub)
        )
        center = start + DEPTH_BIN / 2
        if FIT_RANGE[0] <= center <= FIT_RANGE[1] and sel.sum() >= FIT_MIN_PIXELS:
            estimates.append(sub)
    if estimates:
        stats.subpixel_temporal = float(np.median(estimates))

    corr_mask = smooth & (mean >= CORR_RANGE[0]) & (mean <= CORR_RANGE[1])
    resid = np.where(smooth, (depth - mean) / np.square(np.where(smooth, mean, 1)), 0)
    rows, cols = autocorrelation(resid, corr_mask, TEMPORAL_LAGS)
    stats.acorr_rows, stats.acorr_cols = rows, cols
    c, sy, aspect = fit_correlation(TEMPORAL_LAGS, rows, cols)
    stats.correlated_fraction, stats.correlation_px = c, sy
    stats.correlation_aspect = aspect

    _static(stats, depth, mean, std, smooth, intrinsics, camera_to_base)
    return stats


def _outliers(depth, all_valid) -> float:
    """Share of samples off the per-pixel median by > OUTLIER_REL * Z and 8 MADs.

    Uses the median so that one gross error does not mark a depth edge.
    """
    med = np.where(all_valid, np.median(np.where(all_valid, depth, 0), axis=0), np.nan)
    with np.errstate(invalid="ignore"):
        mask = smooth_mask(med) & (med >= 0.3) & (med <= PLANE_MAX_DEPTH)
    if not mask.any():
        return float("nan")
    dev = np.abs(depth[:, mask] - med[mask])
    mad = 1.4826 * np.median(dev, axis=0)
    return float((dev > np.maximum(8 * mad, OUTLIER_REL * med[mask])).mean())


def _fit_flicker(stats: NoiseStats, n: int) -> None:
    """Per-frame dropout q(d) = rate * exp(-(d - 1) / length) near never-valid."""
    near = [b for b in stats.flicker_by_distance if np.isfinite(b[1])]
    far = [b for b in stats.flicker_by_distance if not np.isfinite(b[1])]
    if far:
        stats.isolated_dropout = float(1.0 - (1.0 - far[0][3]) ** (1.0 / n))
    d = np.array([b[4] for b in near])
    q = np.array([1.0 - (1.0 - b[3]) ** (1.0 / n) for b in near])
    ok = q > 0
    if ok.sum() < 2:
        return
    slope, intercept = np.polyfit(d[ok] - 1.0, np.log(q[ok]), 1)
    stats.flicker_rate = float(np.exp(intercept))
    stats.flicker_px = float(-1.0 / slope) if slope < 0 else float("inf")


def _static(stats, depth, mean, std, smooth, intrinsics, camera_to_base) -> None:
    """Frozen deviation of the temporal mean from large planes.

    Planes nearer than STATIC_MIN_DEPTH are listed but not pooled: there the
    1 mm quantization dominates the temporal noise and surface flatness the
    deviation.
    """
    n = len(depth)
    rays = _rays(mean.shape, intrinsics)
    field_ = np.zeros(mean.shape)
    member = np.zeros(mean.shape, dtype=bool)
    claimed = np.zeros(mean.shape, dtype=bool)
    v_total = v_temporal = 0.0
    pixels = 0
    for mask in _large_planes(mean, intrinsics, camera_to_base):
        mask = mask & smooth & ~claimed
        if mask.sum() < PLANE_MIN_PIXELS:
            continue
        claimed |= mask
        resid, keep = _plane_residual(mean, mask, rays)
        idx = np.flatnonzero(mask)[keep]
        u = resid[keep] - resid[keep].mean()
        z = mean.ravel()[idx]
        vt = float(np.mean((std.ravel()[idx] / z**2) ** 2))
        vu = float(np.mean(u**2))
        static = max(vu - vt / n, 0.0)
        stats.planes.append(
            PlaneStat(len(idx), float(z.mean()), float(np.sqrt(static / vt)))
        )
        if z.mean() < STATIC_MIN_DEPTH:
            continue
        field_.ravel()[idx] = u
        member.ravel()[idx] = True
        v_total += vu * len(idx)
        v_temporal += vt * len(idx)
        pixels += len(idx)
    if pixels == 0:
        return
    v_u, v_t = v_total / pixels, v_temporal / pixels
    v_s = max(v_u - v_t / n, 0.0)
    stats.static_pixels = pixels
    stats.static_ratio = float(np.sqrt(v_s / v_t))
    stats.static_level = float(np.sqrt(v_s) * stats.focal * stats.baseline)
    if v_s <= 0:
        return
    rows, cols = autocorrelation(field_, member, STATIC_LAGS)
    measured = np.nanmean(np.vstack([rows, cols]), axis=0)
    lags = np.asarray(STATIC_LAGS, dtype=np.float64)
    c, sy, aspect = (
        stats.correlated_fraction,
        stats.correlation_px,
        stats.correlation_aspect,
    )
    temporal = 0.5 * (_gauss(lags, c, sy) + _gauss(lags, c, sy * aspect))
    stats.static_acorr = (measured * v_u - temporal * v_t / n) / v_s
    _, stats.static_correlation_px = fit_isotropic(lags, stats.static_acorr)


def summary(stats: NoiseStats) -> dict:
    """Scalar statistics for comparison tables."""
    lag = {d: i for i, d in enumerate(TEMPORAL_LAGS)}

    def acorr(d):
        if stats.acorr_rows is None:
            return float("nan"), float("nan")
        return float(stats.acorr_rows[lag[d]]), float(stats.acorr_cols[lag[d]])

    out = {
        "subpixel_temporal": stats.subpixel_temporal,
        "correlated_fraction": stats.correlated_fraction,
        "correlation_px": stats.correlation_px,
        "correlation_aspect": stats.correlation_aspect,
    }
    for d in (1, 4, 8, 16):
        out[f"acorr_row_{d}"], out[f"acorr_col_{d}"] = acorr(d)
    out.update(
        static_pixels=stats.static_pixels,
        static_ratio=stats.static_ratio,
        static_level=stats.static_level,
        static_fraction=stats.static_fraction,
        static_correlation_px=stats.static_correlation_px,
        invalid_fraction=stats.invalid_fraction,
        never_valid_fraction=stats.never_valid_fraction,
        flicker_fraction=stats.flicker_fraction,
        flicker_rate=stats.flicker_rate,
        flicker_px=stats.flicker_px,
        isolated_dropout=stats.isolated_dropout,
        outlier_fraction=stats.outlier_fraction,
    )
    return out
