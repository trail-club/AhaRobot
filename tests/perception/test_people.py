"""Person perception core and DGX worker protocol, without ROS, CUDA or the DGX."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "overlay_ws/src/aha_perception")
)
from aha_perception import dgx, person_worker  # noqa: E402
from aha_perception.people import (  # noqa: E402
    MATCH_RETRY,
    Detection,
    Gallery,
    PersonTracker,
    decode_image,
    decode_mask,
    describe_crop,
    encode_image,
    encode_mask,
    match_prompt,
    measure,
    parse_features,
    parse_match,
    rotation_matrix,
)


def detection(x, crop=True):
    centre = np.array([x, 0.0, 0.0])
    image = np.full((4, 2, 3), 100, np.uint8) if crop else None
    return Detection(centre, centre + [0, 0, 0.7], 0.9, [0, 0, 1, 1], image)


class MaskTests(unittest.TestCase):
    def test_round_trip_keeps_shape_and_pixels(self):
        mask = np.zeros((7, 11), bool)
        mask[2:5, 3:9] = True
        mask[4, 3] = False
        encoded = encode_mask(mask)
        self.assertEqual(encoded["bbox"], [3, 2, 9, 5])
        np.testing.assert_array_equal(decode_mask(encoded, mask.shape), mask)

    def test_bbox_outside_the_image_is_rejected(self):
        encoded = encode_mask(np.ones((4, 4), bool))
        with self.assertRaises(ValueError):
            decode_mask(encoded, (3, 3))

    def test_jpeg_round_trip_is_close(self):
        rgb = np.zeros((16, 16, 3), np.uint8)
        rgb[:, 8:] = (200, 30, 30)
        self.assertLess(
            np.abs(decode_image(encode_image(rgb)).astype(int) - rgb).mean(), 8
        )


class MeasureTests(unittest.TestCase):
    def test_centre_and_head_ignore_background_leaking_into_the_mask(self):
        depth = np.full((100, 100), 2.0, np.float32)
        mask = np.zeros((100, 100), bool)
        mask[20:80, 40:60] = True
        depth[20:80, 40:42] = 6.0  # wall pixels inside the mask
        depth[50, 50] = np.nan
        result = measure(mask, depth, fx=100, fy=100, cx=50, cy=50)
        np.testing.assert_allclose(result.centre, [0.01, -0.01, 2.0], atol=0.02)
        self.assertAlmostEqual(result.head[2], 2.0)
        self.assertLess(result.head[1], -0.5)  # optical y is down

    def test_too_few_depth_points_is_none(self):
        depth = np.full((20, 20), np.nan, np.float32)
        mask = np.ones((20, 20), bool)
        self.assertIsNone(measure(mask, depth, 10, 10, 10, 10))

    def test_rotation_matrix_turns_about_z(self):
        half = np.sqrt(0.5)
        np.testing.assert_allclose(
            rotation_matrix(0, 0, half, half) @ [1, 0, 0], [0, 1, 0], atol=1e-9
        )


class TrackerTests(unittest.TestCase):
    def test_nearby_sightings_keep_ids_and_far_ones_start_new_tracks(self):
        tracker = PersonTracker(gate_m=0.5, forget_after=2.0, smoothing=0.0)
        first = tracker.update([detection(0.0), detection(2.0)], 0.0)
        self.assertEqual([t.id for t in first], [1, 2])
        moved = tracker.update([detection(2.2), detection(0.1)], 0.5)
        self.assertEqual({t.id: round(t.centre[0], 1) for t in moved}, {1: 0.1, 2: 2.2})
        jumped = tracker.update([detection(5.0)], 1.0)
        self.assertEqual([t.id for t in jumped], [3])
        self.assertEqual(set(tracker.tracks), {1, 2, 3})
        tracker.update([], 2.6)
        self.assertEqual(set(tracker.tracks), {3})


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PersonTracker(forget_after=1.0)
        self.gallery = Gallery()

    def sighting(self, stamp, *xs):
        tracks = self.tracker.update([detection(x) for x in xs], stamp)
        return {t.id: t for t in tracks}

    def test_unnamed_visible_tracks_are_asked_about_free_names_once(self):
        tracks = self.sighting(0.0, 0.0, 1.0)
        self.gallery.remember("guest1", tracks[1], self.tracker.tracks.values())
        # guest1 is held by track 1, so there is nothing to ask about track 2.
        self.assertIsNone(self.gallery.next_match(self.tracker))
        # Once the holder is lost, its name is offered for the new tracks.
        later = self.sighting(5.0, 0.0, 1.0)
        first = self.gallery.next_match(self.tracker)
        second = self.gallery.next_match(self.tracker)
        self.assertEqual([first[0], second[0]], sorted(later))
        self.assertIs(first[1], later[first[0]].crop)
        self.assertEqual([n for n, _ in first[2]], ["guest1"])
        # Each track waits MATCH_RETRY before it is asked again.
        self.assertIsNone(self.gallery.next_match(self.tracker))
        self.sighting(5.0 + MATCH_RETRY, 0.0, 1.0)
        self.assertIsNotNone(self.gallery.next_match(self.tracker))

    def test_no_question_without_free_names_or_visible_unnamed_tracks(self):
        self.assertIsNone(self.gallery.next_match(self.tracker))
        tracks = self.sighting(0.0, 0.0)
        self.gallery.remember("guest1", tracks[1], self.tracker.tracks.values())
        self.sighting(0.5, 3.0)  # a stranger while guest1 is out of view
        self.assertIsNone(self.gallery.next_match(self.tracker))

    def test_a_match_names_the_track_unless_taken_meanwhile(self):
        tracks = self.sighting(0.0, 0.0)
        self.gallery.remember("guest1", tracks[1], self.tracker.tracks.values())
        later = sorted(self.sighting(5.0, 0.0, 1.0))
        self.assertTrue(self.gallery.give("guest1", later[0], self.tracker))
        self.assertFalse(self.gallery.give("guest1", later[1], self.tracker))
        self.assertFalse(self.gallery.give("unknown", later[1], self.tracker))
        self.assertFalse(self.gallery.give("guest1", 999, self.tracker))
        self.assertEqual(self.tracker.tracks[later[0]].name, "guest1")

    def test_remembering_moves_a_name(self):
        tracks = self.sighting(0.0, 0.0, 1.0)
        self.gallery.remember("guest1", tracks[1], self.tracker.tracks.values())
        self.gallery.remember("guest1", tracks[2], self.tracker.tracks.values())
        self.assertEqual([tracks[1].name, tracks[2].name], ["", "guest1"])

    def test_track_without_an_image_cannot_be_remembered(self):
        (track,) = self.tracker.update([detection(0.0, crop=False)], 0.0)
        with self.assertRaises(ValueError):
            self.gallery.remember("guest1", track)


class DescribeTests(unittest.TestCase):
    def test_crop_darkens_only_the_background(self):
        rgb = np.full((50, 50, 3), 200, np.uint8)
        mask = np.zeros((50, 50), bool)
        mask[10:40, 20:30] = True
        crop = describe_crop(rgb, mask)
        self.assertEqual(crop.shape, (36, 16, 3))
        self.assertEqual(crop[18, 8].tolist(), [200, 200, 200])
        self.assertLess(crop[0, 0, 0], 100)

    def test_features_are_cleaned_and_deduplicated(self):
        reply = 'ok ```json\n{"features": ["Red T-shirt.", "glasses", "glasses", 3, "<b>x</b>"]}```'
        self.assertEqual(parse_features(reply), ["red t-shirt", "glasses"])
        with self.assertRaises(ValueError):
            parse_features('{"other": []}')

    def test_match_prompt_and_reply(self):
        self.assertIn("photo 2 shows a person now", match_prompt(1))
        self.assertIn("one of photos 1 to 3", match_prompt(3))
        with self.assertRaises(ValueError):
            match_prompt(4)
        self.assertEqual(parse_match('```json\n{"photo": 2}\n```', 3), 1)
        self.assertIsNone(parse_match('{"photo": null}', 3))
        for bad in ('{"photo": 4}', '{"photo": true}', '{"photo": "1"}'):
            with self.assertRaises(ValueError):
                parse_match(bad, 3)


class FakeSegmenter:
    def predict(self, rgb, threshold):
        masks = np.zeros((2, *rgb.shape[:2]), bool)
        masks[0, 4:28, 2:22] = True
        masks[1, 0:2, 0:2] = True  # too small to be a person
        return {"out_binary_masks": masks, "out_probs": np.array([0.8, 0.9])}


class FakeVlm:
    def ask(self, prompt, *images):
        if len(images) == 1:
            return '{"features": ["glasses"]}'
        return json.dumps({"photo": len(images) - 1})


class WorkerTests(unittest.TestCase):
    def serve(self, lines, vlm=FakeVlm()):
        output = io.StringIO()
        code = person_worker.serve(
            io.StringIO("".join(json.dumps(line) + "\n" for line in lines)),
            output,
            threshold=0.5,
            load_segmenter=lambda prompt, threshold: FakeSegmenter(),
            vlm=vlm,
        )
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_detect_describe_and_match(self):
        image = encode_image(np.zeros((32, 24, 3), np.uint8))
        code, responses = self.serve(
            [
                {"type": "detect", "id": 1, "image": image},
                {"type": "describe", "id": 2, "image": image},
                {"type": "unknown", "id": 3},
                {"type": "match", "id": 4, "image": image, "references": [image] * 2},
                {"type": "match", "id": 5, "image": image, "references": [image] * 4},
            ]
        )
        self.assertEqual(code, 0)
        by_id = {r.get("id"): r for r in responses}
        self.assertEqual(by_id[None]["type"], "ready")
        (person,) = by_id[1]["people"]
        self.assertEqual(person["bbox"], [2, 4, 22, 28])
        self.assertEqual(by_id[1]["shape"], [32, 24])
        self.assertEqual(by_id[2]["features"], ["glasses"])
        self.assertEqual(by_id[3]["type"], "error")
        self.assertEqual(by_id[4]["index"], 1)
        self.assertEqual(by_id[5]["type"], "error")  # over the VLM's image limit

    def test_describe_without_vlm_is_an_error_and_the_worker_continues(self):
        image = encode_image(np.zeros((32, 24, 3), np.uint8))
        _, responses = self.serve(
            [
                {"type": "describe", "id": 1, "image": image},
                {"type": "detect", "id": 2, "image": image},
            ],
            vlm=None,
        )
        by_id = {r.get("id"): r for r in responses}
        self.assertEqual(by_id[1]["type"], "error")
        self.assertEqual(by_id[2]["type"], "detections")

    def test_startup_failure_is_reported(self):
        output = io.StringIO()

        def broken(prompt, threshold):
            raise RuntimeError("no CUDA")

        code = person_worker.serve(
            io.StringIO(""),
            output,
            threshold=0.5,
            load_segmenter=broken,
            vlm=None,
        )
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["message"], "no CUDA")


class DgxWorkerTests(unittest.TestCase):
    def test_ssh_command_uses_the_given_connection(self):
        options = dgx.WorkerOptions(user="taro", identity_file="/k", port=2222)
        command = dgx.DgxWorker(options).ssh("true")
        self.assertEqual(command[0], "ssh")
        self.assertEqual(command[-3:], ["--", "10.99.0.1", "true"])
        for pair in (["-p", "2222"], ["-l", "taro"], ["-i", "/k"]):
            index = command.index(pair[0])
            self.assertEqual(command[index : index + 2], pair)

    def test_alias_keeps_the_ssh_config_user_and_port(self):
        command = dgx.DgxWorker(dgx.WorkerOptions(host="dgx-spark", port=0)).ssh("true")
        self.assertNotIn("-p", command)
        self.assertNotIn("-l", command)

    def test_requests_through_a_local_worker(self):
        script = (
            "import sys, json\n"
            "print(json.dumps({'type': 'ready', 'prompt': 'person'}), flush=True)\n"
            "for line in sys.stdin:\n"
            "    r = json.loads(line)\n"
            "    t = 'error' if r['type'] == 'bad' else 'ok'\n"
            "    print(json.dumps({'type': t, 'id': r['id'], 'message': 'bad'}), flush=True)\n"
        )
        worker = dgx.DgxWorker(
            dgx.WorkerOptions(), command=[sys.executable, "-c", script]
        )
        try:
            self.assertEqual(worker.start(10)["prompt"], "person")
            self.assertEqual(worker.request({"type": "detect"}, 5)["type"], "ok")
            with self.assertRaises(dgx.RequestError):
                worker.request({"type": "bad"}, 5)
            self.assertIsNone(worker.failure)
        finally:
            self.assertTrue(worker.close())

    def test_large_request_while_the_worker_writes_large_responses(self):
        # While a request larger than a pipe buffer is being sent, the worker
        # answers an earlier request and then writes another large response
        # (a detection) before it reads further.
        script = (
            "import sys, json, time\n"
            "print(json.dumps({'type': 'ready'}), flush=True)\n"
            "for line in sys.stdin:\n"
            "    r = json.loads(line)\n"
            "    if r['type'] == 'first':\n"
            "        time.sleep(0.5)\n"
            "        print(json.dumps({'type': 'ok', 'id': r['id']}), flush=True)\n"
            "        print(json.dumps({'type': 'ok', 'id': 0, 'data': 'x' * 400000}),"
            " flush=True)\n"
            "    else:\n"
            "        print(json.dumps({'type': 'ok', 'id': r['id']}), flush=True)\n"
        )
        worker = dgx.DgxWorker(
            dgx.WorkerOptions(), command=[sys.executable, "-c", script]
        )
        pool = ThreadPoolExecutor(2)
        try:
            worker.start(10)
            first = pool.submit(worker.request, {"type": "first"}, 10)
            time.sleep(0.1)
            second = pool.submit(
                worker.request, {"type": "big", "image": "y" * 400000}, 10
            )
            self.assertEqual(first.result(15)["type"], "ok")
            self.assertEqual(second.result(15)["type"], "ok")
        finally:
            pool.shutdown(wait=False)  # a blocked write ends with close()
            self.assertTrue(worker.close())

    def test_worker_that_exits_is_a_failure_not_a_hang(self):
        worker = dgx.DgxWorker(
            dgx.WorkerOptions(), command=[sys.executable, "-c", "print('banner')"]
        )
        try:
            with self.assertRaises(dgx.WorkerError) as raised:
                worker.start(10)
            self.assertIn("banner", str(raised.exception))
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
