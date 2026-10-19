"""VLM candidate selection (option A) without a VLM, CUDA or the model."""

import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "overlay_ws/src/aha_perception")
)
from aha_perception import sam31_vlm, sam31_worker  # noqa: E402
from aha_perception.sam31 import render_candidates  # noqa: E402


def box(top, left, bottom, right, shape=(20, 30)):
    mask = np.zeros(shape, bool)
    mask[top:bottom, left:right] = True
    return mask


class FakeClient:
    model = "fake-vlm"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def ask(self, prompt, *images):
        self.prompts.append((prompt, [image.shape for image in images]))
        return self.replies.pop(0)


class KindPhraseTests(unittest.TestCase):
    def test_kind_phrase_is_a_normalized_generic_phrase_from_the_centre(self):
        client = FakeClient('{"phrase": " Cylindrical  Can "}')
        rgb = np.zeros((100, 200, 3), np.uint8)
        self.assertEqual(sam31_vlm.kind_phrase(client, rgb), "cylindrical can")
        self.assertEqual(client.prompts[0][1], [(60, 120, 3)])
        with self.assertRaises(ValueError):
            sam31_vlm.kind_phrase(FakeClient('{"phrase": "キャンベル"}'), rgb)


class FakeSegmenter:
    """Return fixed masks per phrase and record the threshold used."""

    threshold = 0.5

    def __init__(self, by_phrase):
        self.by_phrase = by_phrase
        self.calls = []
        self.prompt = None

    def predict(self, rgb, threshold=None):
        self.calls.append((self.prompt, threshold))
        masks, scores = self.by_phrase.get(self.prompt, ([], []))
        return {
            "out_binary_masks": np.array(masks, bool).reshape(-1, *rgb.shape[:2]),
            "out_probs": np.array(scores, np.float32),
        }


class ParsingTests(unittest.TestCase):
    def test_json_object_inside_code_fence(self):
        self.assertEqual(
            sam31_vlm.parse_json_object('```json\n{"keep": [1]}\n```'), {"keep": [1]}
        )
        for text in ("no json", "[1, 2]", "{broken"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                sam31_vlm.parse_json_object(text)

    def test_english_prompt_is_tried_before_vlm_phrases(self):
        client = FakeClient(
            '{"phrases": ["Red Snack Can", "red  snack can", "cylinder", "tube", "x"]}'
        )
        phrases = sam31_vlm.propose_phrases(
            client, np.zeros((4, 4, 3), np.uint8), "Pringles"
        )
        self.assertEqual(phrases, ["pringles", "red snack can", "cylinder"])
        self.assertIn('"Pringles"', client.prompts[0][0])

    def test_japanese_prompt_uses_only_vlm_phrases(self):
        client = FakeClient('{"phrases": ["paper snack cup"]}')
        rgb = np.zeros((4, 4, 3), np.uint8)
        self.assertEqual(
            sam31_vlm.propose_phrases(client, rgb, "じゃがりこ"), ["paper snack cup"]
        )
        with self.assertRaises(ValueError):
            sam31_vlm.propose_phrases(FakeClient('{"phrases": []}'), rgb, "じゃがりこ")

    def test_selection_rejects_numbers_outside_the_candidates(self):
        image = np.zeros((4, 4, 3), np.uint8)
        self.assertEqual(
            sam31_vlm.select_candidates(
                FakeClient('{"keep": [2, 1, 2]}'), image, image, "x", 2
            ),
            [1, 2],
        )
        for reply in ('{"keep": [3]}', '{"keep": [true]}', '{"keep": "1"}', "{}"):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                sam31_vlm.select_candidates(FakeClient(reply), image, image, "x", 2)


class CandidateTests(unittest.TestCase):
    def test_duplicates_across_phrases_keep_the_higher_score(self):
        can, near_copy, cup = box(0, 0, 10, 10), box(0, 0, 10, 9), box(12, 12, 18, 20)
        masks, scores, sources, runs = sam31_vlm.merge_candidates(
            [
                ("pringles", {"out_binary_masks": [near_copy], "out_probs": [0.6]}),
                (
                    "snack can",
                    {"out_binary_masks": [can, cup], "out_probs": [0.8, 0.4]},
                ),
            ],
            20,
            30,
        )
        np.testing.assert_allclose(scores, [0.8, 0.4])
        self.assertEqual(sources, ["snack can", "snack can"])
        self.assertTrue((masks[0] == can).all())
        self.assertEqual(
            [[m["outcome"] for m in run["masks"]] for run in runs],
            [["duplicate of 1"], ["candidate 1", "candidate 2"]],
        )
        self.assertEqual(runs[1]["masks"][1]["bbox_xyxy"], [12, 12, 20, 18])

    def test_candidate_count_is_capped(self):
        many = [box(0, i * 3, 2, i * 3 + 2) for i in range(10)]
        masks, scores, _, runs = sam31_vlm.merge_candidates(
            [("x", {"out_binary_masks": many, "out_probs": np.linspace(0.9, 0.3, 10)})],
            20,
            30,
        )
        self.assertEqual(len(masks), sam31_vlm.MAX_CANDIDATES)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)
        self.assertEqual(runs[0]["masks"][-1]["outcome"], "over limit")

    def test_candidate_image_marks_each_mask(self):
        rgb = np.full((20, 30, 3), 100, np.uint8)
        image = render_candidates(
            rgb, np.array([box(0, 0, 10, 10), box(10, 15, 20, 30)])
        )
        self.assertEqual(image.shape, rgb.shape)
        self.assertFalse((image[:10, :10] == rgb[:10, :10]).all())
        self.assertFalse((image[10:, 15:] == rgb[10:, 15:]).all())


class PipelineTests(unittest.TestCase):
    def test_vlm_keeps_only_the_target_among_look_alikes(self):
        rgb = np.full((20, 30, 3), 100, np.uint8)
        segmenter = FakeSegmenter(
            {
                "pringles": ([box(0, 0, 10, 10), box(0, 15, 10, 25)], [0.84, 0.74]),
                "red snack can": ([box(0, 0, 10, 10)], [0.9]),
            }
        )
        client = FakeClient(
            '{"phrases": ["red snack can"]}', '{"keep": [1]}', '{"same": false}'
        )
        outputs, record, images = sam31_vlm.segment_with_vlm(
            segmenter, client, rgb, "pringles", 0.3
        )
        self.assertEqual(segmenter.calls, [("pringles", 0.3), ("red snack can", 0.3)])
        self.assertEqual(len(outputs["out_probs"]), 1)
        np.testing.assert_allclose(outputs["out_probs"], [0.9])
        self.assertEqual(record["keep"], [1])
        self.assertEqual(
            [(c["number"], c["phrase"], c["kept"]) for c in record["candidates"]],
            [(1, "red snack can", True), (2, "pringles", False)],
        )
        self.assertEqual(images["candidates.png"].shape, rgb.shape)
        self.assertEqual(
            [s["name"] for s in record["steps"]], ["phrases", "select", "match 2"]
        )
        self.assertEqual(record["selected"], [1])
        self.assertEqual(
            record["matches"],
            [
                {
                    "number": 2,
                    "reference": 1,
                    "same": False,
                    "reference_size": None,
                    "size": None,
                }
            ],
        )
        self.assertEqual(images["match_2.png"].shape[0], sam31_vlm.MATCH_HEIGHT_PX)
        self.assertEqual(record["steps"][1]["answer"], '{"keep": [1]}')
        self.assertEqual(
            [run["phrase"] for run in record["sam_runs"]], ["pringles", "red snack can"]
        )
        self.assertEqual(record["candidates"][0]["color"], "#88C26E")
        # The selection sees the photo first, then the numbered candidates.
        select_prompt, shapes = client.prompts[1]
        self.assertEqual(shapes, [rgb.shape, rgb.shape])
        self.assertIn("Check every candidate one by one", select_prompt)

    def test_rejected_look_alike_of_a_kept_candidate_is_added_by_matching(self):
        rgb = np.full((20, 30, 3), 100, np.uint8)
        segmenter = FakeSegmenter(
            {
                "pringles": (
                    [box(0, 0, 10, 8), box(0, 10, 10, 18), box(0, 20, 10, 28)],
                    [0.9, 0.8, 0.7],
                )
            }
        )
        client = FakeClient(
            '{"phrases": []}', '{"keep": [2]}', '{"same": true}', '{"same": false}'
        )
        outputs, record, images = sam31_vlm.segment_with_vlm(
            segmenter, client, rgb, "pringles", 0.3
        )
        self.assertEqual(record["selected"], [2])
        self.assertEqual(record["keep"], [1, 2])
        self.assertEqual(
            [(m["number"], m["reference"], m["same"]) for m in record["matches"]],
            [(1, 2, True), (3, 2, False)],
        )
        self.assertEqual(
            [c["via"] for c in record["candidates"]], ["match", "select", None]
        )
        np.testing.assert_allclose(outputs["out_probs"], [0.9, 0.8])
        self.assertEqual(
            sorted(images), ["candidates.png", "match_1.png", "match_3.png"]
        )
        # Each check sends one side-by-side image.
        self.assertEqual(len(client.prompts[2][1]), 1)
        self.assertIn('A is a "pringles"', client.prompts[2][0])

    def test_matching_can_be_disabled_and_needs_a_boolean_answer(self):
        rgb = np.full((20, 30, 3), 100, np.uint8)
        masks = ([box(0, 0, 10, 8), box(0, 10, 10, 18)], [0.9, 0.8])
        client = FakeClient('{"phrases": []}', '{"keep": [1]}')
        _, record, images = sam31_vlm.segment_with_vlm(
            FakeSegmenter({"pringles": masks}), client, rgb, "pringles", 0.3, False
        )
        self.assertEqual((record["keep"], record["matches"]), ([1], []))
        self.assertEqual(sorted(images), ["candidates.png"])
        client = FakeClient('{"phrases": []}', '{"keep": [1]}', '{"same": "yes"}')
        with self.assertRaises(ValueError):
            sam31_vlm.segment_with_vlm(
                FakeSegmenter({"pringles": masks}), client, rgb, "pringles", 0.3
            )

    def test_pair_image_keeps_the_size_ratio(self):
        shape = (200, 200)
        rgb = np.full((*shape, 3), 128, np.uint8)
        tall, short = box(10, 10, 190, 40, shape), box(130, 100, 190, 130, shape)

        def heights(pair):
            # Rows that contain photo pixels (not white padding) in each half.
            middle = pair.shape[1] // 2
            return [
                int((pair[:, :middle] != 255).any(axis=(1, 2)).sum()),
                int((pair[:, middle:] != 255).any(axis=(1, 2)).sum()),
            ]

        pair = sam31_vlm.pair_image(rgb, tall, short)
        self.assertEqual(pair.shape[0], sam31_vlm.MATCH_HEIGHT_PX)
        tall_px, short_px = heights(pair)
        self.assertEqual(tall_px, sam31_vlm.MATCH_HEIGHT_PX)
        self.assertAlmostEqual(short_px / tall_px, 72 / 192, delta=0.02)
        # Twice as far away: the short object is shown at twice its pixel size.
        _, short_far = heights(sam31_vlm.pair_image(rgb, tall, short, (1.0, 2.0)))
        self.assertAlmostEqual(short_far / tall_px, 144 / 192, delta=0.02)

    def test_physical_size_uses_median_depth_and_marks_cut_off(self):
        depth = np.full((20, 30), 0.5, np.float32)
        depth[0, 0] = np.nan
        size = sam31_vlm.physical_size(box(0, 0, 10, 6), depth, (600.0, 500.0))
        self.assertEqual(
            size,
            {"width_cm": 0.5, "height_cm": 1.0, "depth_m": 0.5, "cut_off": True},
        )
        self.assertFalse(
            sam31_vlm.physical_size(box(5, 5, 10, 10), depth, (600.0, 500.0))["cut_off"]
        )
        self.assertIsNone(sam31_vlm.physical_size(box(5, 5, 10, 10), None, None))
        self.assertIn("cut off", sam31_vlm.size_text({**size, "cut_off": True}))
        self.assertEqual(sam31_vlm.size_text(None), "of unknown size")

    def test_matching_tells_the_vlm_both_sizes(self):
        rgb = np.full((20, 30, 3), 100, np.uint8)
        depth = np.full((20, 30), 1.0, np.float32)
        masks = ([box(2, 2, 18, 8), box(12, 12, 18, 18)], [0.9, 0.8])
        client = FakeClient('{"phrases": []}', '{"keep": [1]}', '{"same": false}')
        _, record, _ = sam31_vlm.segment_with_vlm(
            FakeSegmenter({"pringles": masks}),
            client,
            rgb,
            "pringles",
            0.3,
            depth_m=depth,
            intrinsics=(100.0, 100.0),
        )
        match = record["matches"][0]
        self.assertEqual(match["reference_size"]["height_cm"], 16.0)
        self.assertEqual(match["size"]["height_cm"], 6.0)
        self.assertIn("A is about 16 cm tall", client.prompts[2][0])
        self.assertIn("B is about 6 cm tall", client.prompts[2][0])

    def test_no_candidates_skips_the_selection_call(self):
        client = FakeClient('{"phrases": ["paper snack cup"]}')
        outputs, record, _ = sam31_vlm.segment_with_vlm(
            FakeSegmenter({}),
            client,
            np.zeros((20, 30, 3), np.uint8),
            "じゃがりこ",
            0.3,
        )
        self.assertEqual(len(client.prompts), 1)
        self.assertEqual(outputs["out_binary_masks"].shape, (0, 20, 30))
        self.assertEqual(record["candidates"], [])

    def test_worker_returns_candidates_image_and_vlm_record(self):
        rgb = np.full((20, 30, 3), 100, np.uint8)
        data = BytesIO()
        np.savez_compressed(
            data,
            rgb=rgb,
            depth_m=np.ones((20, 30), np.float32),
            metadata=json.dumps({"frame": 1}),
        )
        request = {
            "prompt": "pringles",
            "rgbd": base64.b64encode(data.getvalue()).decode(),
        }
        segmenter = FakeSegmenter({"pringles": ([box(0, 0, 10, 10)], [0.4])})
        client = FakeClient('{"phrases": []}', '{"keep": [1]}')
        response = sam31_worker.segment_request(segmenter, request, client, 0.3)
        names = zipfile.ZipFile(
            BytesIO(base64.b64decode(response["archive"]))
        ).namelist()
        self.assertIn("candidates.png", names)
        summary = response["summary"]
        self.assertEqual(summary["threshold"], 0.3)
        self.assertEqual(len(summary["objects"]), 1)
        self.assertEqual(summary["vlm"]["model"], "fake-vlm")


class ClientTests(unittest.TestCase):
    def reply(self, content):
        response = BytesIO(json.dumps({"message": {"content": content}}).encode())
        response.__enter__ = lambda self=response: self
        response.__exit__ = lambda *args: None
        return response

    def test_request_sends_the_image_and_returns_the_answer(self):
        client = sam31_vlm.OllamaClient("http://dgx:11434/", "qwen3-vl:30b")
        with patch.object(
            sam31_vlm.urllib.request, "urlopen", return_value=self.reply('{"keep": []}')
        ) as urlopen:
            self.assertEqual(
                client.ask("pick", np.zeros((4, 4, 3), np.uint8)), '{"keep": []}'
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://dgx:11434/api/chat")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "qwen3-vl:30b")
        self.assertTrue(body["messages"][0]["images"][0])

    def test_request_sends_every_image_in_order(self):
        client = sam31_vlm.OllamaClient()
        photo = np.zeros((4, 4, 3), np.uint8)
        with patch.object(
            sam31_vlm.urllib.request, "urlopen", return_value=self.reply('{"keep": []}')
        ) as urlopen:
            client.ask("pick", photo, np.full((2, 2, 3), 255, np.uint8))
        images = json.loads(urlopen.call_args.args[0].data)["messages"][0]["images"]
        self.assertEqual(len(images), 2)
        decoded = sam31_vlm.Image.open(BytesIO(base64.b64decode(images[0])))
        self.assertEqual(decoded.size, (4, 4))

    def test_empty_answer_and_unreachable_server_are_errors(self):
        client = sam31_vlm.OllamaClient()
        image = np.zeros((4, 4, 3), np.uint8)
        with patch.object(
            sam31_vlm.urllib.request, "urlopen", return_value=self.reply("")
        ):
            with self.assertRaisesRegex(RuntimeError, "no answer"):
                client.ask("pick", image)
        with patch.object(
            sam31_vlm.urllib.request,
            "urlopen",
            side_effect=sam31_vlm.urllib.error.URLError("refused"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not reachable"):
                client.ask("pick", image)


if __name__ == "__main__":
    unittest.main()
