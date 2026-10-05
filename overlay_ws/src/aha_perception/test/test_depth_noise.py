from dataclasses import replace

import numpy as np
import pytest

from aha_perception.depth_noise import (
    NOISE_PRESETS,
    SHADOW_MAX_PX,
    SHADOW_MIN_STEP,
    DepthNoiseSimulator,
    NoiseParams,
    _max_ahead,
    _shadow,
    corrupt_depth,
    noise_preset,
)

FOCAL = 200.0
SHAPE = (240, 320)


def flat(depth=1.5, shape=SHAPE):
    return np.full(shape, depth, np.float32)


def quiet(**kwargs):
    """Only Gaussian depth noise; the other components are off."""
    noise = NoiseParams(
        focal=FOCAL,
        dropout=0.0,
        holes=0,
        flying_prob=0.0,
        outlier_rate=0.0,
        shadow=False,
    )
    return replace(noise, **kwargs)


def test_presets_and_unknown_name():
    assert {"default", "d435i_raw", "d435i_filtered"} <= set(NOISE_PRESETS)
    assert noise_preset("d435i_raw", 123.0).focal == 123.0
    with pytest.raises(ValueError):
        DepthNoiseSimulator("missing")


def test_invalid_input_stays_invalid():
    depth = flat()
    depth[10:20, 10:20] = np.nan
    depth[30, 30] = 0.0
    depth[40, 40] = np.inf
    depth[50, 50] = -np.inf
    out = DepthNoiseSimulator("d435i_filtered").apply(depth, FOCAL)
    assert out.shape == depth.shape and out.dtype == np.float32
    for pixel in ((15, 15), (30, 30), (40, 40), (50, 50)):
        assert np.isnan(out[pixel])
    valid = np.isfinite(out)
    assert valid.mean() > 0.6
    assert np.abs(out[valid] - 1.5).max() < 0.3


def test_frozen_components_repeat_across_frames():
    sim = DepthNoiseSimulator("d435i_filtered", seed=3)
    first = sim.apply(flat(), FOCAL)
    frozen = sim.frozen
    second = sim.apply(flat(), FOCAL)
    assert sim.frozen.holes is frozen.holes
    assert sim.frozen.static is frozen.static
    assert np.isnan(first[frozen.holes]).all() and np.isnan(second[frozen.holes]).all()
    assert not np.array_equal(first, second, equal_nan=True)

    # Without temporal noise the frames are identical.
    noise = quiet(static_fraction=1.0, quantization=0.0)
    rng = np.random.default_rng(0)
    a = corrupt_depth(flat(), noise, rng)
    b = corrupt_depth(flat(), noise, rng, a.frozen)
    np.testing.assert_array_equal(a.depth, b.depth)
    assert np.nanstd(a.depth) > 0.0


def test_same_seed_same_output():
    a = DepthNoiseSimulator("d435i_raw", seed=5).apply(flat(), FOCAL)
    b = DepthNoiseSimulator("d435i_raw", seed=5).apply(flat(), FOCAL)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("preset", ["d435i_filtered", "d435i_raw"])
@pytest.mark.parametrize("z", [1.0, 2.5])
def test_temporal_std_matches_preset_sigma(preset, z):
    sim = DepthNoiseSimulator(preset, seed=1)
    frames = np.stack([sim.apply(flat(z), FOCAL) for _ in range(16)])
    ok = np.isfinite(frames).all(axis=0)
    std = np.median(frames[:, ok].astype(np.float64).std(axis=0, ddof=1))
    noise = noise_preset(preset, FOCAL)
    sigma = z**2 * noise.subpixel / (noise.focal * noise.baseline)
    # Only the per-frame share of the variance changes between frames.
    expected = sigma * np.sqrt(1.0 - noise.static_fraction)
    assert std == pytest.approx(expected, rel=0.2)


def test_frozen_noise_redrawn_on_size_change():
    sim = DepthNoiseSimulator("d435i_filtered")
    sim.apply(flat(), FOCAL)
    assert sim.frozen.holes.shape == SHAPE
    out = sim.apply(flat(shape=(120, 160)), FOCAL)
    assert out.shape == (120, 160)
    assert sim.frozen.holes.shape == (120, 160)
    noise = noise_preset("d435i_filtered", FOCAL)
    with pytest.raises(ValueError):
        corrupt_depth(flat(), noise, sim.rng, sim.frozen)


def test_flicker_follows_moving_invalid_region():
    noise = quiet(flicker_rate=1.0, flicker_px=0.5, subpixel=0.01)
    rng = np.random.default_rng(0)
    depth = flat()
    depth[:, :100] = np.nan
    first = corrupt_depth(depth, noise, rng)
    moved = flat()
    moved[:, 200:] = np.nan
    second = corrupt_depth(moved, noise, rng, first.frozen)
    # Pixels next to the current invalid region drop out, the old edge is kept.
    assert np.isnan(second.depth[:, 199]).mean() > 0.8
    assert np.isfinite(second.depth[:, 100]).all()


def shift_left(values, s, fill):
    """``out[v, u] = values[v, u + s]``, ``fill`` past the right edge."""
    out = np.full_like(values, fill)
    out[:, :-s] = values[:, s:]
    return out


@pytest.mark.parametrize("window", [1, 3, 7, 24, 50])
def test_max_ahead_matches_brute_force(window):
    values = np.random.default_rng(1).uniform(0.0, 80.0, (20, 64))
    expected = np.full(values.shape, -np.inf)
    for s in range(1, window + 1):
        expected = np.maximum(expected, shift_left(values, s, -np.inf) - s)
    np.testing.assert_array_equal(_max_ahead(values, window), expected)


def test_shadow_matches_brute_force():
    rng = np.random.default_rng(2)
    z = rng.uniform(0.4, 3.0, (60, 120))
    # Blocks of nearer surfaces cast shadows to their left.
    for _ in range(20):
        v, u = rng.integers(0, 50), rng.integers(0, 100)
        z[v : v + 10, u : u + 20] *= rng.uniform(0.2, 0.9)
    z[rng.random(z.shape) < 0.02] = np.inf
    fb = 465.7 * 0.05
    expected = np.zeros(z.shape, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in range(1, SHADOW_MAX_PX + 1):
            right = shift_left(z, s, np.inf)
            width = fb * (1.0 / right - 1.0 / z)
            expected |= (z - right > SHADOW_MIN_STEP) & (s <= width)
    expected &= np.isfinite(z)
    assert expected.sum() > 100
    np.testing.assert_array_equal(_shadow(z, fb), expected)
