import datetime
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np


BENCH_DIR = pathlib.Path(__file__).parents[2] / "tools/perception/plane_bench"
sys.path.insert(0, str(BENCH_DIR))

import real_frames  # noqa: E402


def _scene_arrays(n=3, height=4, width=5):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.4, 6.0, (n, height, width)).astype(np.float32)
    depth[0, 0, 0] = np.nan
    depth[1, 1, 1] = 0.0
    depth[-1, 2, 2] = -1.0
    transforms = np.tile(np.eye(4), (n, 1, 1))
    transforms[:, 2, 3] = 1.4
    stamps = 100.0 + 0.5 * np.arange(n)
    k = np.array([603.0, 0.0, 320.0, 0.0, 602.0, 240.0, 0.0, 0.0, 1.0])
    joints = np.column_stack([np.zeros(n), np.full(n, 0.4)])
    return depth, transforms, stamps, k, joints


class RealFramesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_keeps_millimeter_depth_and_metadata(self):
        depth, transforms, stamps, k, joints = _scene_arrays()
        name = real_frames.scene_dir_name(
            "desk", datetime.datetime(2026, 10, 4, 17, 47)
        )
        self.assertEqual(name, "20261004-1747_desk")

        real_frames.save_scene(
            self.root / name,
            depth=depth,
            camera_to_base=transforms,
            stamps=stamps,
            k=k,
            meta={"scene": "desk", "filter_label": "raw", "base_height": 0.7},
            head_joints=joints,
            color_jpeg=b"\xff\xd8jpeg",
        )
        scene = real_frames.load_scene(self.root / name)

        invalid = ~(np.isfinite(depth) & (depth > 0))
        np.testing.assert_array_equal(np.isnan(scene.depth), invalid)
        np.testing.assert_allclose(scene.depth[~invalid], depth[~invalid], atol=5e-4)
        np.testing.assert_array_equal(scene.camera_to_base, transforms)
        np.testing.assert_array_equal(scene.stamps, stamps)
        np.testing.assert_array_equal(scene.head_joints, joints)
        self.assertEqual(scene.intrinsics, (603.0, 602.0, 320.0, 240.0))
        self.assertEqual(scene.n_frames, 3)
        self.assertEqual(scene.meta["frame_count"], 3)
        self.assertEqual((scene.meta["width"], scene.meta["height"]), (5, 4))
        self.assertEqual(scene.meta["filter_label"], "raw")
        self.assertEqual(scene.color_path, self.root / name / "color.jpg")
        np.testing.assert_allclose(scene.camera_to_floor()[:, 2, 3], 2.1)
        with np.load(self.root / name / "frames.npz") as data:
            self.assertEqual(data["depth_mm"].dtype, np.uint16)
        self.assertEqual(real_frames.list_scenes(self.root), [self.root / name])

    def test_unknown_base_height_and_missing_joints(self):
        depth, transforms, stamps, k, _ = _scene_arrays(n=2)
        path = real_frames.save_scene(
            self.root / "scene", depth, transforms, stamps, k, meta={}
        )

        scene = real_frames.load_scene(path)

        self.assertIsNone(scene.base_height)
        np.testing.assert_array_equal(scene.camera_to_floor(), transforms)
        self.assertTrue(np.isnan(scene.head_joints).all())
        self.assertIsNone(scene.color_path)
        meta = json.loads((path / "meta.json").read_text())
        self.assertIsNone(meta["color_file"])

    def test_refuses_to_overwrite_and_rejects_bad_shapes(self):
        depth, transforms, stamps, k, _ = _scene_arrays(n=2)
        real_frames.save_scene(self.root / "a", depth, transforms, stamps, k, {})

        with self.assertRaises(FileExistsError):
            real_frames.save_scene(self.root / "a", depth, transforms, stamps, k, {})
        with self.assertRaises(ValueError):
            real_frames.save_scene(
                self.root / "b", depth, transforms[:1], stamps, k, {}
            )
        self.assertEqual(real_frames.list_scenes(self.root / "missing"), [])


if __name__ == "__main__":
    unittest.main()
