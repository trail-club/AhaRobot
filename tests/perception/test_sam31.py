"""Check mask/depth contracts without downloading the model."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "overlay_ws/src/aha_perception")
)
from aha_perception.sam31 import (  # noqa: E402
    Sam31Segmenter,
    load_capture,
    make_result,
    safe_error,
    save_result,
)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((2, 3, 3), 100, dtype=np.uint8)
        self.depth = np.array([[1, np.nan, 2], [0, np.inf, 4]], dtype=np.float32)

    def test_overlap_goes_to_highest_score_and_depth_excludes_invalid(self):
        outputs = {
            "out_binary_masks": np.array(
                [
                    [[1, 1, 0], [1, 1, 0]],
                    [[0, 1, 1], [0, 0, 1]],
                ],
                dtype=bool,
            ),
            "out_probs": np.array([0.7, 0.9]),
        }
        labels, overlay, objects = make_result(self.rgb, self.depth, outputs, 0.5)
        np.testing.assert_array_equal(labels, [[2, 1, 1], [2, 2, 1]])
        self.assertEqual(objects[0]["median_depth_m"], 3.0)
        self.assertEqual(objects[0]["valid_depth_pixels"], 2)
        self.assertEqual(objects[1]["median_depth_m"], 1.0)
        self.assertEqual(objects[0]["bbox_xyxy"], [1, 0, 3, 2])
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertFalse(np.array_equal(overlay, self.rgb))

    def test_empty_detection_clears_labels_and_preserves_rgb(self):
        outputs = {
            "out_binary_masks": np.zeros((0, 2, 3), bool),
            "out_probs": np.zeros(0),
        }
        labels, overlay, objects = make_result(self.rgb, self.depth, outputs, 0.5)
        self.assertFalse(labels.any())
        np.testing.assert_array_equal(overlay, self.rgb)
        self.assertEqual(objects, [])

    def test_threshold_and_fully_occluded_masks_are_removed(self):
        outputs = {
            "out_binary_masks": np.ones((3, 2, 3), bool),
            "out_probs": [0.2, 0.8, 0.9],
        }
        labels, _, objects = make_result(self.rgb, self.depth, outputs, 0.5)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["pixels"], 6)
        self.assertTrue((labels == 1).all())

    def test_all_invalid_depth_is_null_in_saved_json(self):
        outputs = {"out_binary_masks": np.ones((1, 2, 3), bool), "out_probs": [0.9]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_result(
                output,
                self.rgb,
                np.full((2, 3), np.nan),
                outputs,
                prompt="bottle",
                threshold=0.5,
                metadata={},
                elapsed=0.1,
            )
            summary = json.loads((output / "result.json").read_text())
            self.assertIsNone(summary["objects"][0]["median_depth_m"])
            self.assertEqual(summary["objects"][0]["valid_depth_pixels"], 0)
            np.testing.assert_array_equal(
                np.asarray(Image.open(output / "labels.png")), 1
            )
            with np.load(output / "masks.npz", allow_pickle=False) as data:
                self.assertTrue(np.isnan(data["depth_m"]).all())

    def test_bad_alignment_or_mask_shape_is_rejected(self):
        outputs = {"out_binary_masks": np.ones((1, 3, 2), bool), "out_probs": [0.9]}
        with self.assertRaisesRegex(ValueError, "dimensions"):
            make_result(self.rgb, self.depth, outputs, 0.5)
        with self.assertRaisesRegex(ValueError, "aligned"):
            make_result(self.rgb, self.depth.T, outputs, 0.5)

    def test_token_redacted_in_diagnostics(self):
        with patch.dict("os.environ", {"HF_TOKEN": "test-private-token"}):
            message = safe_error(RuntimeError("test-private-token hf_exampleSecret"))
        self.assertEqual(message, "[redacted] [redacted]")


class CaptureTests(unittest.TestCase):
    def bundle(self, **overrides):
        data = {
            "rgb": np.zeros((2, 3, 3), np.uint8),
            "depth_m": np.full((2, 3), 1.5, np.float32),
            "metadata": json.dumps({"device": "test"}),
        }
        data.update(overrides)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        buffer.seek(0)
        return buffer

    def test_capture_can_be_loaded_from_transport_buffer(self):
        rgb, depth, metadata = load_capture(self.bundle())
        self.assertEqual(rgb.shape, (2, 3, 3))
        np.testing.assert_array_equal(depth, 1.5)
        self.assertEqual(metadata, {"device": "test"})

    def test_raw_millimeter_depth_and_empty_image_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "meters"):
            load_capture(self.bundle(depth_m=np.full((2, 3), 1500, np.uint16)))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            load_capture(self.bundle(rgb=np.zeros((0, 3, 3), np.uint8)))

    def test_metadata_must_be_json_object(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            load_capture(self.bundle(metadata="[]"))
        with self.assertRaisesRegex(ValueError, "scalar JSON string"):
            load_capture(self.bundle(metadata=np.array(["{}"])))

    def test_plain_numpy_array_is_not_a_capture(self):
        buffer = io.BytesIO()
        np.save(buffer, np.zeros((2, 3, 3), np.uint8))
        buffer.seek(0)
        with self.assertRaisesRegex(ValueError, ".npz archive"):
            load_capture(buffer)


class SessionTests(unittest.TestCase):
    def test_inference_failure_closes_session(self):
        class Predictor:
            def __init__(self):
                self.closed = False

            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "test"}
                if request["type"] == "add_prompt":
                    raise RuntimeError("inference failed")
                self.closed = request["session_id"] == "test"

        engine = object.__new__(Sam31Segmenter)
        engine.torch = SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            autocast=lambda **kwargs: contextlib.nullcontext(),
            bfloat16="bfloat16",
        )
        engine.predictor = Predictor()
        engine.prompt = "bottle"
        engine.threshold = 0.5
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            engine.predict(np.zeros((2, 3, 3), dtype=np.uint8))
        self.assertTrue(engine.predictor.closed)


if __name__ == "__main__":
    unittest.main()
