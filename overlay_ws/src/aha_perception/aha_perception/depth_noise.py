"""D435i-like stereo depth corruption (numpy / scipy, no rclpy).

The model and its presets are fitted to real captures by
tools/perception/plane_bench/calibrate_noise.py. The plane benchmark uses it on
ray-cast scenes and sim_depth_noise.py on the Gazebo depth image.

Depth is optical z in meters; non-finite and non-positive pixels are invalid.
Components in ``FrozenNoise`` (holes, static field) stay fixed in image
coordinates across frames; the rest is drawn per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

SHADOW_MAX_PX = 24
SHADOW_MIN_STEP = 0.03
FLYING_MIN_STEP = 0.05


@dataclass(frozen=True)
class NoiseParams:
    """D435i-like corruption. ``scale`` multiplies the noise-dependent terms.

    ``subpixel`` sets the per-frame depth sigma. Across frames sharing a
    ``FrozenNoise`` the holes and ``static_fraction`` of the depth noise
    variance are fixed; the rest is drawn per frame.
    ``correlation_px`` is the vertical Gaussian sigma of the correlated part,
    ``correlation_aspect`` the horizontal / vertical ratio. Pixels at distance
    d (px) from an invalid pixel (shadow, hole, out of range) drop out per
    frame with probability ``flicker_rate * exp(-(d - 1) / flicker_px)``.
    ``quantization`` rounds the depth (m, 0 = off). The defaults keep the
    original single-frame model of the plane benchmark.

    ``subpixel`` and the correlation lengths are in pixels of the image they
    are applied to (the presets were calibrated at fx = 603). With
    ``focal = fx`` on another image, sigma(Z) in meters scales with fx while
    its ratio to the detector's ``DepthNoiseModel(focal=fx)`` is kept.
    """

    baseline: float = 0.05
    subpixel: float = 0.08
    focal: float = 465.7
    scale: float = 1.0
    correlated_fraction: float = 0.5
    correlation_px: float = 1.5
    dropout: float = 0.005
    holes: int = 3
    flying_prob: float = 0.3
    outlier_rate: float = 0.001
    shadow: bool = True
    min_range: float = 0.28
    max_range: float = 6.0
    correlation_aspect: float = 1.0
    static_fraction: float = 0.0
    static_correlation_px: float = 14.0
    flicker_rate: float = 0.0
    flicker_px: float = 2.0
    quantization: float = 0.0


# Fitted to the real D435i captures by calibrate_noise.py; see
# tools/perception/plane_bench/results/noise_calibration.md. ``focal`` is
# replaced by the image fx.
NOISE_PRESETS = {
    "default": NoiseParams(),
    "d435i_raw": NoiseParams(
        subpixel=0.201,
        correlated_fraction=0.987,
        correlation_px=7.1,
        dropout=0.0006,
        holes=11,
        outlier_rate=0.0,
        correlation_aspect=1.3,
        static_fraction=0.53,
        static_correlation_px=16.6,
        flicker_rate=0.091,
        flicker_px=1.9,
        quantization=0.001,
    ),
    "d435i_filtered": NoiseParams(
        subpixel=0.154,
        correlated_fraction=0.981,
        correlation_px=8.5,
        dropout=0.0002,
        holes=13,
        outlier_rate=0.0,
        correlation_aspect=1.3,
        static_fraction=0.82,
        static_correlation_px=16.2,
        flicker_rate=0.084,
        flicker_px=1.7,
        quantization=0.001,
    ),
}


def noise_preset(name: str, focal: float) -> NoiseParams:
    return replace(NOISE_PRESETS[name], focal=focal)


@dataclass
class FrozenNoise:
    """Noise components shared by every frame that reuses them."""

    holes: np.ndarray  # (H, W) bool
    static: np.ndarray | None  # (H, W) unit-variance field
    # Pixel distance to the nearest invalid pixel, for the ``invalid`` mask it
    # was computed from; recomputed when the mask changes.
    near_invalid: np.ndarray | None = None
    invalid: np.ndarray | None = None


@dataclass
class CorruptedDepth:
    depth: np.ndarray  # (H, W) float64, NaN = invalid
    frozen: FrozenNoise
    flying: np.ndarray  # (H, W) bool, interpolated across a depth edge
    outlier: np.ndarray  # (H, W) bool, scaled gross outlier


def _unit_field(rng, shape, sigma):
    field_ = gaussian_filter(rng.standard_normal(shape), sigma)
    return field_ / field_.std()


def _max_ahead(values, window):
    """``max(values[v, u + s] - s for s in 1..window)``, -inf past the edge."""
    block = np.full(values.shape, -np.inf, values.dtype)
    np.subtract(values[:, 1:], 1, out=block[:, :-1])
    blocks = {1: block}
    scratch = np.empty_like(block)
    size = 1
    while 2 * size <= window:
        np.subtract(block[:, size:], size, out=scratch[:, :-size])
        block = block.copy()
        np.maximum(block[:, :-size], scratch[:, :-size], out=block[:, :-size])
        size *= 2
        blocks[size] = block
    out = blocks[size].copy()
    covered = size
    for size in sorted(blocks, reverse=True):
        if covered + size <= window:
            np.subtract(blocks[size][:, covered:], covered, out=scratch[:, :-covered])
            np.maximum(out[:, :-covered], scratch[:, :-covered], out=out[:, :-covered])
            covered += size
    return out


def _shadow(z, fb):
    """Stereo occlusion: pixels left of a nearer surface within its disparity step.

    Same result as testing every shift s = 1..SHADOW_MAX_PX on the whole image:
    ``z - z_right > SHADOW_MIN_STEP`` and ``s <= fb * (1 / z_right - 1 / z)``.
    A float32 disparity bound with a margin picks the candidate pixels, which
    are then tested exactly with the same float64 operations.
    """
    with np.errstate(divide="ignore"):
        inv = 1.0 / z
    finite = np.isfinite(z)
    disparity = (fb * inv).astype(np.float32)
    bound = _max_ahead(disparity, SHADOW_MAX_PX)
    disparity -= 1e-4 * (1.0 + disparity)
    candidate = finite & (bound >= disparity)
    rows, cols = np.nonzero(candidate)
    pad = ((0, 0), (0, SHADOW_MAX_PX))
    z_pad = np.pad(z, pad, constant_values=np.inf)
    inv_pad = np.pad(inv, pad, constant_values=0.0)
    z0, inv0 = z[rows, cols], inv[rows, cols]
    hit = np.zeros(rows.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for s in range(1, SHADOW_MAX_PX + 1):
            width = fb * (inv_pad[rows, cols + s] - inv0)
            hit |= (z0 - z_pad[rows, cols + s] > SHADOW_MIN_STEP) & (s <= width)
    shadow = np.zeros(z.shape, dtype=bool)
    shadow[rows[hit], cols[hit]] = True
    return shadow


def _draw_holes(rng, shape, count):
    h, w = shape
    holes = np.zeros(shape, dtype=bool)
    for _ in range(count):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(4, 15, size=2)
        # Ellipse bounding box with a pixel of margin; outside it the test fails.
        v0, v1 = max(0, int(cy - ry) - 1), min(h, int(cy + ry) + 2)
        u0, u1 = max(0, int(cx - rx) - 1), min(w, int(cx + rx) + 2)
        v, u = np.mgrid[v0:v1, u0:u1]
        holes[v0:v1, u0:u1] |= ((v - cy) / ry) ** 2 + ((u - cx) / rx) ** 2 <= 1
    return holes


def _near_invalid(frozen: FrozenNoise | None, invalid):
    if (
        frozen is not None
        and frozen.near_invalid is not None
        and np.array_equal(frozen.invalid, invalid)
    ):
        return frozen.near_invalid
    return distance_transform_edt(~invalid)


def corrupt_depth(
    depth,
    noise: NoiseParams,
    rng: np.random.Generator,
    frozen: FrozenNoise | None = None,
) -> CorruptedDepth:
    """Corrupt a depth image (m); returns NaN for invalid pixels.

    With ``frozen`` None the frozen components are drawn from ``rng``;
    otherwise they are reused and must match the image size.
    """
    if frozen is not None and frozen.holes.shape != np.shape(depth):
        raise ValueError(
            f"frozen noise is {frozen.holes.shape}, depth is {np.shape(depth)}"
        )
    depth = np.asarray(depth, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        z = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)
    if noise.shadow:
        z[_shadow(z, noise.focal * noise.baseline)] = np.inf
    with np.errstate(invalid="ignore"):
        frozen_invalid = (
            ~np.isfinite(z) | (depth < noise.min_range) | (depth > noise.max_range)
        )

    # In-place operations below keep the arithmetic (and so the output bits) of
    # the plain expressions in the comments.
    finite = np.isfinite(z)
    neighbors = np.where(finite, z, -np.inf)
    far = np.full(z.shape, -np.inf)
    np.maximum(far[:, :-1], neighbors[:, 1:], out=far[:, :-1])
    np.maximum(far[:, 1:], neighbors[:, :-1], out=far[:, 1:])
    np.maximum(far[:-1], neighbors[1:], out=far[:-1])
    np.maximum(far[1:], neighbors[:-1], out=far[1:])
    far[~finite] = -np.inf
    fly = np.isfinite(far) & (far - z > FLYING_MIN_STEP)
    fly &= rng.random(z.shape) < min(1.0, noise.flying_prob * noise.scale)
    alpha = rng.uniform(0.2, 0.8, z.shape)
    # z = where(fly, z + alpha * (far - z), z)
    z[fly] += alpha[fly] * (far[fly] - z[fly])

    # sigma = square(where(isfinite(z), z, 0)) * (subpixel / (f * b))
    sigma = np.where(np.isfinite(z), z, 0.0)
    np.square(sigma, out=sigma)
    sigma *= noise.subpixel / (noise.focal * noise.baseline)
    mix = rng.standard_normal(z.shape)
    corr_sigma = noise.correlation_px
    if noise.correlation_aspect != 1.0:
        corr_sigma = (
            noise.correlation_px,
            noise.correlation_px * noise.correlation_aspect,
        )
    corr = gaussian_filter(rng.standard_normal(z.shape), corr_sigma)
    corr /= corr.std()
    # mix = sqrt(1 - cf) * iid + sqrt(cf) * corr
    mix *= np.sqrt(1 - noise.correlated_fraction)
    corr *= np.sqrt(noise.correlated_fraction)
    mix += corr
    static = None
    if noise.static_fraction > 0:
        if frozen is None or frozen.static is None:
            static = _unit_field(rng, z.shape, noise.static_correlation_px)
        else:
            static = frozen.static
        # mix = sqrt(1 - sf) * mix + sqrt(sf) * static
        mix *= np.sqrt(1 - noise.static_fraction)
        mix += np.sqrt(noise.static_fraction) * static
    # z = z + scale * sigma * mix
    sigma *= noise.scale
    sigma *= mix
    z += sigma

    drop = rng.random(z.shape) < noise.dropout * noise.scale
    holes = _draw_holes(rng, z.shape, noise.holes) if frozen is None else frozen.holes
    drop |= holes
    near_invalid = invalid = None
    if noise.flicker_rate > 0:
        invalid = frozen_invalid | holes
        near_invalid = _near_invalid(frozen, invalid)
        # p = rate * exp(-(near_invalid - 1) / flicker_px)
        p = near_invalid - 1.0
        np.negative(p, out=p)
        p /= noise.flicker_px
        np.exp(p, out=p)
        p *= min(1.0, noise.flicker_rate * noise.scale)
        drop |= rng.random(z.shape) < p
    z[drop] = np.inf

    outlier = np.isfinite(z) & (rng.random(z.shape) < noise.outlier_rate * noise.scale)
    # z = where(outlier, z * uniform(0.5, 1.5), z)
    z[outlier] *= rng.uniform(0.5, 1.5, z.shape)[outlier]
    if noise.quantization > 0:
        # z = round(z / q) * q
        z /= noise.quantization
        np.round(z, out=z)
        z *= noise.quantization

    with np.errstate(invalid="ignore"):
        z[~np.isfinite(z) | (z < noise.min_range) | (z > noise.max_range)] = np.nan
    frozen = FrozenNoise(holes, static, near_invalid, invalid)
    return CorruptedDepth(z, frozen, fly, outlier)


class DepthNoiseSimulator:
    """Corrupts a stream of depth images with sensor-fixed frozen components.

    The frozen components are drawn with the first frame and redrawn when the
    image size changes.
    """

    def __init__(self, preset: str, seed: int = 0):
        if preset not in NOISE_PRESETS:
            raise ValueError(
                f"unknown depth noise preset {preset!r}, "
                f"choose from {sorted(NOISE_PRESETS)}"
            )
        self.preset = preset
        self.rng = np.random.default_rng(seed)
        self.frozen: FrozenNoise | None = None

    def apply(self, depth: np.ndarray, focal: float) -> np.ndarray:
        """Noisy copy of ``depth`` (m) as float32, NaN for invalid pixels."""
        if self.frozen is not None and self.frozen.holes.shape != depth.shape:
            self.frozen = None
        result = corrupt_depth(
            depth, noise_preset(self.preset, focal), self.rng, self.frozen
        )
        self.frozen = result.frozen
        return result.depth.astype(np.float32)
