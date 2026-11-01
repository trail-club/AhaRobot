import importlib.util
import pathlib
import sys
import unittest
from dataclasses import replace

import numpy as np

BENCH_DIR = pathlib.Path(__file__).parents[2] / "tools/perception/plane_bench"
sys.path.insert(0, str(BENCH_DIR))

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
if HAVE_SCIPY:
    import noise_stats  # noqa: E402
    import scenes  # noqa: E402


def _scene(name):
    return next(s for s in scenes.scene_set() if s.name == name)


@unittest.skipUnless(HAVE_SCIPY, "plane_bench needs scipy")
class DefaultNoiseRegressionTests(unittest.TestCase):
    def test_default_frame_matches_original_model(self):
        # Values from the single-frame model before sequences were added.
        frame = scenes.generate(_scene("table_1.5m"), 1.0, 0)
        z = frame.cloud[..., 2]

        self.assertEqual(int(np.isnan(z).sum()), 66924)
        values, counts = np.unique(frame.labels, return_counts=True)
        self.assertEqual(
            dict(zip(values.tolist(), counts.tolist())),
            {-3: 101010, -2: 9655, -1: 1762, 0: 179414, 1: 15359},
        )
        self.assertAlmostEqual(
            float(np.nansum(z.astype(np.float64))), 623858.5575962067, delta=1e-3
        )
        for pixel, depth in (
            ((240, 320), 1.4551961421966553),
            ((400, 500), 1.7898485660552979),
            ((470, 10), 1.5267226696014404),
        ):
            self.assertAlmostEqual(float(z[pixel]), depth, places=6)
        self.assertEqual([p.surface for p in frame.planes], ["floor", "table"])

    def test_first_sequence_frame_equals_generate(self):
        scene = _scene("table_1.5m")
        for noise in (None, scenes.noise_preset("d435i_raw", 465.7)):
            frame = scenes.generate(scene, 1.0, 5, noise=noise)
            first = scenes.generate_sequence(scene, 2, noise, 5)[0]
            np.testing.assert_array_equal(frame.cloud, first.cloud)
            np.testing.assert_array_equal(frame.labels, first.labels)


@unittest.skipUnless(HAVE_SCIPY, "plane_bench needs scipy")
class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = scenes.Intrinsics(
            width=160, height=120, fx=116.4, fy=116.4, cx=79.5, cy=59.5
        )
        self.scene = _scene("table_1.5m")
        self.quiet = replace(
            scenes.NoiseParams(focal=116.4),
            correlated_fraction=0.0,
            dropout=0.0,
            flying_prob=0.0,
            outlier_rate=0.0,
            subpixel=0.02,
        )

    def _depth(self, noise, n=3, seed=1):
        frames = scenes.generate_sequence(self.scene, n, noise, seed, self.intrinsics)
        return frames, np.stack([f.cloud[..., 2] for f in frames])

    def test_frozen_components_repeat_and_temporal_noise_differs(self):
        _, frozen = self._depth(replace(self.quiet, static_fraction=1.0))
        np.testing.assert_array_equal(frozen[0], frozen[1])
        np.testing.assert_array_equal(frozen[0], frozen[2])

        frames, depth = self._depth(self.quiet)
        holes = np.isnan(depth)
        # Away from the max range, invalid pixels are the frozen shadow and holes.
        inside = ~(np.fmin.reduce(depth, axis=0) > 5.0)
        np.testing.assert_array_equal(holes[0] & inside, holes[1] & inside)
        valid = ~holes[0]
        self.assertFalse(np.array_equal(depth[0][valid], depth[1][valid]))
        for frame in frames:
            invalid = np.isnan(frame.cloud[..., 2])
            self.assertTrue((frame.labels[invalid] == scenes.INVALID).all())

    def test_static_fraction_splits_the_variance(self):
        noise = replace(self.quiet, static_fraction=0.5, holes=0, shadow=False)
        frames = scenes.generate_sequence(self.scene, 2, noise, 3, scenes.Intrinsics())
        z0, z1 = (f.cloud[..., 2].astype(np.float64) for f in frames)
        ok = np.isfinite(z0) & np.isfinite(z1) & (frames[0].labels >= 0)
        sigma = z0[ok] ** 2 * noise.subpixel / (noise.focal * noise.baseline)
        # Frame difference holds only the per-frame half: var = 2 * 0.5.
        self.assertAlmostEqual(float(np.var((z1[ok] - z0[ok]) / sigma)), 1.0, delta=0.1)

    def test_flicker_drops_pixels_next_to_frozen_invalid(self):
        from scipy.ndimage import distance_transform_edt

        noise = replace(self.quiet, flicker_rate=0.5, flicker_px=1.0)
        _, depth = self._depth(noise, n=6)
        finite = np.isfinite(depth)
        flicker = finite.any(axis=0) & ~finite.all(axis=0)
        never = ~finite.any(axis=0)
        distance = distance_transform_edt(~never)
        self.assertGreater(flicker[(distance > 0) & (distance <= 1.5)].mean(), 0.5)
        self.assertLess(flicker[distance > 6].mean(), 0.01)


@unittest.skipUnless(HAVE_SCIPY, "plane_bench needs scipy")
class NoiseStatsTests(unittest.TestCase):
    def _plane_stack(self, subpixel, correlation_px=0.0, n=10, seed=0):
        from scipy.ndimage import gaussian_filter

        h, w, f = 240, 320, 300.0
        rng = np.random.default_rng(seed)
        # Floor seen from 1.0 m height with the optical axis 20 deg down.
        v, u = np.mgrid[:h, :w]
        tilt = np.radians(20.0)
        ray_down = np.sin(tilt) + np.cos(tilt) * (v - h / 2) / f
        depth = np.where(ray_down > 0, 1.0 / np.maximum(ray_down, 1e-6), np.inf)
        depth = np.where(depth < 3.4, depth, np.nan)
        sigma = depth**2 * subpixel / (f * 0.05)
        frames = []
        for _ in range(n):
            noise = rng.standard_normal((h, w))
            if correlation_px:
                noise = gaussian_filter(noise, correlation_px)
                noise /= noise.std()
            frames.append(depth + sigma * noise)
        camera_to_base = np.eye(4)
        camera_to_base[:3, :3] = np.array(
            [
                [0.0, -np.sin(tilt), np.cos(tilt)],
                [-1.0, 0.0, 0.0],
                [0.0, -np.cos(tilt), -np.sin(tilt)],
            ]
        )
        camera_to_base[2, 3] = 1.0
        return np.stack(frames), (f, f, w / 2 - 0.5, h / 2), camera_to_base

    def test_recovers_temporal_subpixel_of_iid_noise(self):
        depth, k, transform = self._plane_stack(0.1)
        stats = noise_stats.compute(depth, k, transform)
        self.assertAlmostEqual(stats.subpixel_temporal, 0.1, delta=0.015)
        self.assertLess(stats.correlated_fraction, 0.1)
        self.assertEqual(stats.flicker_fraction, 0.0)

    def test_recovers_correlation_length(self):
        depth, k, transform = self._plane_stack(0.1, correlation_px=3.0)
        stats = noise_stats.compute(depth, k, transform)
        self.assertAlmostEqual(stats.subpixel_temporal, 0.1, delta=0.015)
        self.assertGreater(stats.correlated_fraction, 0.9)
        self.assertAlmostEqual(stats.correlation_px, 3.0, delta=0.6)
        self.assertAlmostEqual(stats.correlation_aspect, 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
