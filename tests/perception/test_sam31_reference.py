"""Reference capture checks without SSH, CUDA or USB hardware."""

import argparse
import importlib.util
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perception/macos/sam31"))
SPEC = importlib.util.spec_from_file_location(
    "sam31_reference", ROOT / "tools/perception/macos/sam31/reference.py"
)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)

CAMERA_INFO = {"k": [500.0, 0, 100, 0, 500.0, 75, 0, 0, 1]}


def obj(label, bbox, score=0.9):
    x0, y0, x1, y1 = bbox
    return {
        "label": label,
        "bbox_xyxy": bbox,
        "score": score,
        "pixels": (x1 - x0) * (y1 - y0),
    }


def write_shot(directory, objects, depth=0.5, shape=(150, 200)):
    """A capture and a SAM result whose labels match `objects`."""
    capture, result = directory / "capture", directory / "result"
    capture.mkdir(parents=True)
    result.mkdir()
    rgb = np.full((*shape, 3), 90, np.uint8)
    labels = np.zeros(shape, np.uint16)
    for o in objects:
        x0, y0, x1, y1 = o["bbox_xyxy"]
        labels[y0:y1, x0:x1] = o["label"]
    np.savez_compressed(
        capture / "rgbd.npz",
        rgb=rgb,
        depth_m=np.full(shape, depth, np.float32),
        metadata=json.dumps({"camera_info": CAMERA_INFO, "device": "Fake"}),
    )
    Image.fromarray(labels).save(result / "labels.png")
    return capture, result, {"objects": objects}


class PickTests(unittest.TestCase):
    def test_object_nearest_the_centre_is_the_reference(self):
        objects = [
            obj(1, [0, 0, 40, 40]),
            obj(2, [80, 40, 120, 110]),
            obj(3, [150, 50, 200, 100]),
        ]
        self.assertEqual(reference.pick_object(objects, 200, 150)["label"], 2)
        self.assertIsNone(reference.pick_object([obj(1, [0, 0, 40, 40])], 200, 150))


class SaveViewTests(unittest.TestCase):
    def test_saves_mask_crop_size_and_index_without_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            library = directory / "refs"
            capture, result, summary = write_shot(
                directory, [obj(1, [70, 20, 130, 130])]
            )
            record = reference.save_view(
                library, "pringles", "front", capture, result, summary
            )
            view = library / "pringles/front"
            self.assertEqual(record["warnings"], [])
            # 110 px tall at 0.5 m with fy=500 -> 11 cm.
            self.assertEqual(
                record["size"], {"width_cm": 6.0, "height_cm": 11.0, "depth_m": 0.5}
            )
            mask = np.asarray(Image.open(view / "mask.png"))
            self.assertEqual(int((mask > 0).sum()), 60 * 110)
            self.assertEqual(Image.open(view / "crop.png").size, (60, 110))
            for name in ("rgb.png", "preview.png", "rgbd.npz", "view.json"):
                self.assertTrue((view / name).is_file(), name)
            index = (library / "pringles/index.html").read_text()
            self.assertIn("まだ撮っていない向き: back, left, right", index)
            saved = json.loads((library / "pringles/reference.json").read_text())
            self.assertEqual([v["view"] for v in saved["views"]], ["front"])

    def test_warns_about_cut_off_small_and_extra_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            objects = [obj(1, [80, 60, 120, 90]), obj(2, [0, 0, 30, 150], 0.8)]
            capture, result, summary = write_shot(directory, objects)
            record = reference.save_view(
                directory / "refs", "pringles", "back", capture, result, summary
            )
            text = " ".join(record["warnings"])
            self.assertIn("小さく", text)
            self.assertIn("ほかの物が1個", text)
            capture, result, summary = write_shot(
                directory / "2", [obj(1, [60, 0, 140, 150])]
            )
            record = reference.save_view(
                directory / "refs", "pringles", "back", capture, result, summary
            )
            self.assertIn("端で切れて", " ".join(record["warnings"]))

    def test_no_central_object_keeps_the_previous_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            library = directory / "refs"
            capture, result, summary = write_shot(
                directory / "1", [obj(1, [70, 20, 130, 130])]
            )
            reference.save_view(library, "pringles", "front", capture, result, summary)
            capture, result, summary = write_shot(
                directory / "2", [obj(1, [0, 0, 30, 30])]
            )
            with self.assertRaisesRegex(ValueError, "中央に物が見つかりません"):
                reference.save_view(
                    library, "pringles", "front", capture, result, summary
                )
            self.assertTrue((library / "pringles/front/view.json").is_file())
            self.assertEqual(
                [
                    p.name
                    for p in (library / "pringles").iterdir()
                    if p.name.startswith(".")
                ],
                [],
            )


class SessionTests(unittest.TestCase):
    def test_enter_walks_the_required_views_and_failures_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            library = directory / "refs"
            shots = iter(
                [
                    [obj(1, [70, 20, 130, 130])],  # front
                    [obj(1, [0, 0, 30, 30])],  # back: nothing central -> retry
                    [obj(1, [70, 20, 130, 130])],  # back
                ]
            )

            def capture(destination):
                objects = next(shots)
                c, r, s = write_shot(destination.parent / "shot", objects)
                c.rename(destination)
                capture.summary, capture.result = s, r

            def predict(phrase, path):
                self.assertEqual(phrase, "object")
                return {"type": "result"}

            def unpack(response, output):
                capture.result.rename(output)
                return capture.summary

            camera = Mock(capture=Mock(side_effect=capture))
            worker = Mock(predict=Mock(side_effect=predict))
            reader = Mock(side_effect=["", "", "", EOFError()])
            args = argparse.Namespace(name="pringles", phrase="object")
            original = reference.interactive.unpack_result
            reference.interactive.unpack_result = unpack
            try:
                with redirect_stdout(StringIO()), unittest.mock.patch(
                    "sys.stderr", StringIO()
                ):
                    with self.assertRaises(EOFError):
                        reference.session(args, worker, camera, library, reader)
            finally:
                reference.interactive.unpack_result = original
            self.assertEqual(
                sorted(reference.load_views(library, "pringles")), ["back", "front"]
            )
            prompts = [call.args[0] for call in reader.call_args_list]
            self.assertIn("front:", prompts[0])
            self.assertIn("back:", prompts[1])
            self.assertIn("back:", prompts[2])
            self.assertIn("left:", prompts[3])
            camera.show_result.assert_called()
            self.assertFalse(
                (library / ".work").exists() and any((library / ".work").iterdir())
            )
            saved = reference.load_phrase(library, "pringles")
            self.assertEqual((saved["phrase"], saved["source"]), ("object", "user"))

    def test_without_phrase_the_vlm_chooses_it_once_and_it_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            library = directory / "refs"
            (library / "can").mkdir(parents=True)
            shots = iter([[obj(1, [70, 20, 130, 130])], [obj(1, [70, 20, 130, 130])]])

            def capture(destination):
                c, r, s = write_shot(destination.parent / "shot", next(shots))
                c.rename(destination)
                capture.summary, capture.result = s, r

            def unpack(response, output):
                capture.result.rename(output)
                return capture.summary

            kind = Mock(return_value="cylindrical can")
            worker = Mock(predict=Mock(return_value={"type": "result"}))
            camera = Mock(capture=Mock(side_effect=capture))
            args = argparse.Namespace(name="can", phrase=None)
            original = reference.interactive.unpack_result
            reference.interactive.unpack_result = unpack
            try:
                with redirect_stdout(StringIO()), unittest.mock.patch(
                    "sys.stderr", StringIO()
                ):
                    with self.assertRaises(EOFError):
                        reference.session(
                            args,
                            worker,
                            camera,
                            library,
                            Mock(side_effect=["", "", EOFError()]),
                            kind=kind,
                        )
            finally:
                reference.interactive.unpack_result = original
            kind.assert_called_once()
            self.assertEqual(
                [c.args[0] for c in worker.predict.call_args_list],
                ["cylindrical can", "cylindrical can"],
            )
            saved = reference.load_phrase(library, "can")
            self.assertEqual(
                (saved["phrase"], saved["source"]), ("cylindrical can", "vlm")
            )
            self.assertIn("cylindrical can", (library / "can/index.html").read_text())

    def test_without_phrase_or_vlm_the_shot_is_not_segmented(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory)
            (library / "can").mkdir()
            capture = Path(directory) / "rgbd.npz"
            args = argparse.Namespace(name="can", phrase=None)
            with self.assertRaisesRegex(RuntimeError, "--phrase"):
                reference.resolve_phrase(args, library, capture, None)


if __name__ == "__main__":
    unittest.main()
