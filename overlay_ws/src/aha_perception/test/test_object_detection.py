"""Object detection core and DGX worker protocol, without ROS, CUDA or the DGX."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import sys
import threading
import time

import numpy as np
import pytest

from aha_perception import object_worker
from aha_perception.dgx import WorkerOptions, object_spec
from aha_perception.object_detection import (
    SAM_SECONDS,
    SHEET_COLUMNS,
    SHEET_TILES,
    TILE_PX,
    VLM_PARALLEL,
    ObjectDetection,
    ObjectTracker,
    annotate,
    crop_sheet,
    label_prompt,
    measure_object,
    merge_detections,
    near_any,
    parse_labels,
    pool_candidates,
    size_fits,
    touches_border,
    vlm_timeout,
)
from aha_perception.people import decode_mask, encode_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import camera_to_base  # noqa: E402
from fake_object_worker import ColourSegmenter, ColourVlm  # noqa: E402

W, H = 640, 480
FX = FY = 465.7
CX, CY = 319.5, 239.5


def rect(y0, y1, x0, x1, shape=(H, W)):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def outputs(*pairs):
    masks = np.array([m for m, _ in pairs], dtype=bool).reshape(-1, H, W)
    return {
        "out_binary_masks": masks,
        "out_probs": np.array([s for _, s in pairs], np.float32),
    }


# --- Candidates -----------------------------------------------------------------


def test_pool_drops_duplicates_small_and_large_masks():
    can = rect(100, 160, 100, 140)
    same_can = rect(101, 160, 100, 140)
    tiny = rect(10, 15, 10, 15)
    wall = rect(0, 400, 0, 640)
    cup = rect(200, 260, 300, 350)
    pooled = pool_candidates(
        [
            ("can", outputs((can, 0.6), (tiny, 0.9), (wall, 0.95))),
            ("bottle", outputs((same_can, 0.8), (cup, 0.5))),
        ],
        (H, W),
    )
    assert [(c.phrase, round(c.score, 2)) for c in pooled] == [
        ("bottle", 0.8),
        ("bottle", 0.5),
    ]


def test_pool_keeps_the_best_candidates_up_to_the_limit():
    masks = [(rect(10, 40, 30 * i, 30 * i + 20), 0.5 + 0.01 * i) for i in range(10)]
    pooled = pool_candidates([("box", outputs(*masks))], (H, W), max_candidates=3)
    assert [round(c.score, 2) for c in pooled] == [0.59, 0.58, 0.57]


def test_pool_drops_masks_cut_off_by_the_image_edge():
    inside = rect(100, 160, 100, 140)
    cut = rect(0, 40, 300, 340)
    pooled = pool_candidates(
        [("can", outputs((inside, 0.6), (cut, 0.9)))], (H, W), border_px=2
    )
    assert [round(c.score, 1) for c in pooled] == [0.6]
    assert touches_border(rect(100, 140, 638, 640), 2)
    assert not touches_border(rect(0, 40, 300, 340), 0)


def test_pool_rejects_mismatched_outputs():
    with pytest.raises(ValueError):
        pool_candidates(
            [("can", {"out_binary_masks": np.zeros((1, 4, 4)), "out_probs": [0.5]})],
            (H, W),
        )


def test_crop_sheet_numbers_tiles_left_to_right():
    rgb = np.full((H, W, 3), 128, np.uint8)
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    masks = []
    for i, colour in enumerate(colours):
        mask = rect(50 + 60 * i, 90 + 60 * i, 40 + 100 * i, 80 + 100 * i)
        rgb[mask] = colour
        masks.append(mask)
    sheet = crop_sheet(rgb, masks)
    assert sheet.shape == (2 * TILE_PX, SHEET_COLUMNS * TILE_PX, 3)
    for i, colour in enumerate(colours):
        y = (i // SHEET_COLUMNS) * TILE_PX + TILE_PX // 2
        x = (i % SHEET_COLUMNS) * TILE_PX + TILE_PX // 2
        assert tuple(sheet[y, x]) == colour


def test_label_prompt_lists_labels_sizes_and_count():
    prompt = label_prompt(
        [
            {"label": "mug", "category": "tableware", "size": [0.12, 0.09, 0.08]},
            {"label": "tuna can", "category": "food"},
            {"label": "apple"},
        ],
        3,
    )
    assert "- mug (tableware, about 12 x 9 x 8 cm)" in prompt
    assert "- tuna can (food)\n" in prompt and "- apple\n" in prompt
    assert "with every number from 1 to 3" in prompt


def test_parse_labels_maps_spellings_and_none():
    labels = ["red bull", "mug", "apple"]
    reply = (
        "Here:\n```json\n"
        '{"labels": {"1": "Red_Bull", "2": "none", "3": "teapot", "4": "MUG"}}\n```'
    )
    assert parse_labels(reply, 5, labels) == ["red bull", None, None, "mug", None]
    assert parse_labels('{"labels": ["apple", "red-bull"]}', 2, labels) == [
        "apple",
        "red bull",
    ]
    described = {
        "labels": {
            "1": {"seen": "red can", "label": "red bull"},
            "2": {"seen": "a shadow", "label": "none"},
            "3": {"seen": "mug"},
        }
    }
    assert parse_labels(json.dumps(described), 3, labels) == ["red bull", None, None]
    with pytest.raises(ValueError):
        parse_labels("no json here", 1, labels)
    with pytest.raises(ValueError):
        parse_labels('{"objects": {}}', 1, labels)


# --- Geometry -------------------------------------------------------------------


def ray_grid(transform):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays_cam = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u, float)], -1)
    return rays_cam @ transform[:3, :3].T, transform[:3, 3]


def render_box(transform, lows, highs):
    """Depth (optical z) of an axis-aligned box in the base frame, NaN elsewhere."""
    rays, origin = ray_grid(transform)
    lows, highs = np.asarray(lows), np.asarray(highs)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lows - origin) / rays
        t1 = (highs - origin) / rays
    near = np.nanmax(np.minimum(t0, t1), axis=-1)
    far = np.nanmin(np.maximum(t0, t1), axis=-1)
    hit = (near <= far) & (near > 0)
    return np.where(hit, near, np.nan)


def render_cylinder(transform, centre, radius, height):
    """Depth of an upright cylinder standing on z = 0 (side and top)."""
    rays, origin = ray_grid(transform)
    ox, oy = origin[0] - centre[0], origin[1] - centre[1]
    dx, dy, dz = rays[..., 0], rays[..., 1], rays[..., 2]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        side = (-b - np.sqrt(disc)) / (2 * a)
        z_side = origin[2] + side * dz
        side = np.where((disc >= 0) & (z_side >= 0) & (z_side <= height), side, np.inf)
        top = (height - origin[2]) / dz
        hx, hy = ox + top * dx, oy + top * dy
        top = np.where((hx * hx + hy * hy <= radius * radius) & (top > 0), top, np.inf)
    depth = np.minimum(side, top)
    return np.where(np.isfinite(depth), depth, np.nan)


def measure(depth, transform, **kwargs):
    mask = np.isfinite(depth)
    return measure_object(
        mask,
        depth,
        FX,
        FY,
        CX,
        CY,
        rotation=transform[:3, :3],
        translation=transform[:3, 3],
        **kwargs,
    )


@pytest.mark.parametrize("pan", [0.0, 25.0])
def test_measure_cube_centre_behind_the_visible_faces(pan):
    transform = camera_to_base(height=1.3, tilt_deg=40.0, pan_deg=pan)
    yaw = np.radians(pan)
    centre = np.array([1.0 * np.cos(yaw), 1.0 * np.sin(yaw), 0.05])
    half = np.array([0.04, 0.04, 0.05])
    measured = measure(render_box(transform, centre - half, centre + half), transform)
    assert measured is not None
    np.testing.assert_allclose(measured.centre, centre, atol=0.012)
    # The axis-aligned box of a footprint measured along the viewing
    # direction: larger than the cube when the view is oblique.
    assert 0.075 < measured.size[0] < 0.15 and 0.075 < measured.size[1] < 0.15
    assert measured.size[2] == pytest.approx(0.1, abs=0.01)


def test_measure_cylinder_centre_from_its_front_half():
    transform = camera_to_base(height=1.3, tilt_deg=35.0)
    centre = np.array([1.2, -0.2, 0.0])
    depth = render_cylinder(transform, centre, 0.033, 0.12)
    measured = measure(depth, transform)
    np.testing.assert_allclose(measured.centre[:2], centre[:2], atol=0.01)
    assert measured.centre[2] == pytest.approx(0.06, abs=0.01)
    assert measured.size[2] == pytest.approx(0.12, abs=0.01)


def test_measure_thin_box_uses_the_known_footprint():
    # A cereal box seen face on: the visible face is 0.2 m wide, but it is
    # only 0.07 m deep, which the label's size tells.
    transform = camera_to_base(height=1.3, tilt_deg=30.0)
    centre = np.array([1.0, 0.0, 0.15])
    half = np.array([0.035, 0.1, 0.15])
    depth = render_box(transform, centre - half, centre + half)
    guessed = measure(depth, transform)
    known = measure(depth, transform, footprint=0.07)
    assert abs(known.centre[0] - centre[0]) < 0.01
    assert guessed.centre[0] - centre[0] > 0.04


def test_measure_ignores_holes_and_leaked_background():
    transform = camera_to_base(height=1.3, tilt_deg=40.0)
    centre = np.array([1.0, 0.0, 0.05])
    half = np.array([0.04, 0.04, 0.05])
    depth = render_box(transform, centre - half, centre + half)
    mask = np.isfinite(depth)
    ys, xs = np.nonzero(mask)
    leaked = depth.copy()
    leaked[ys[::7], xs[::7]] = np.nan  # depth holes
    border = mask & ~np.roll(mask, 3, axis=1)
    leaked[border] = 4.0  # wall behind the object at the silhouette
    measured = measure_object(
        mask,
        leaked,
        FX,
        FY,
        CX,
        CY,
        rotation=transform[:3, :3],
        translation=transform[:3, 3],
    )
    np.testing.assert_allclose(measured.centre, centre, atol=0.015)


def test_measure_needs_enough_depth():
    depth = np.full((H, W), np.nan, np.float32)
    mask = rect(100, 140, 100, 140)
    depth[100:103, 100:103] = 1.0
    assert measure_object(mask, depth, FX, FY, CX, CY) is None


def test_measure_in_the_camera_frame_is_the_visible_box():
    depth = np.full((H, W), np.nan, np.float32)
    mask = rect(200, 280, 300, 340)
    depth[mask] = 1.5
    measured = measure_object(mask, depth, FX, FY, CX, CY)
    assert measured.centre[2] == pytest.approx(1.5)
    assert measured.size[0] == pytest.approx(40 * 1.5 / FX, abs=0.01)


# --- Tracking ---------------------------------------------------------------------


def detection(label, x, y=0.0, score=0.8):
    return ObjectDetection(
        label, np.array([x, y, 0.75]), np.array([0.07, 0.07, 0.1]), score, [0, 0, 1, 1]
    )


def test_size_fits_rejects_only_much_larger_boxes():
    tab = (0.03, 0.03, 0.015)
    assert size_fits([0.03, 0.02, 0.015], tab)
    assert size_fits([0.09, 0.08, 0.02], tab)  # oblique view, depth noise
    assert not size_fits([0.12, 0.13, 0.14], tab)  # a cookie box
    assert size_fits([0.05, 0.05, 0.05], (0.2, 0.04, 0.02))  # partly hidden


def test_near_any():
    assert near_any([1.0, 0.0, 0.1], [np.array([1.05, 0.0, 0.1])], 0.12)
    assert not near_any([1.0, 0.0, 0.1], [np.array([1.5, 0.0, 0.1])], 0.12)
    assert not near_any([1.0, 0.0, 0.1], [], 0.12)


def test_merge_keeps_one_detection_per_object():
    merged = merge_detections(
        [
            detection("cereal", 1.0, score=0.5),
            detection("cereal", 1.03, score=0.9),
            detection("milk", 1.1),  # touching the cereal
            detection("cereal", 1.3),
        ]
    )
    assert sorted((d.label, d.score) for d in merged) == [
        ("cereal", 0.8),
        ("cereal", 0.9),
        ("milk", 0.8),
    ]


def box(label, x, y, z, size, score=0.8):
    return ObjectDetection(
        label, np.array([x, y, z]), np.array(size, float), score, [0, 0, 1, 1]
    )


def test_merge_joins_parts_and_one_object_named_twice():
    whole = box("cereal", 1.0, 0.0, 0.81, [0.06, 0.16, 0.22], score=0.7)
    # The upper half found by another phrase: centres 5.5 cm apart.
    part = box("cereal", 1.0, 0.0, 0.865, [0.06, 0.16, 0.11], score=0.9)
    assert merge_detections([whole, part]) == [part]
    # Another label for nearly the same box.
    named = box("chocolate jello", 1.005, 0.0, 0.8, [0.06, 0.15, 0.2], score=0.6)
    assert merge_detections([whole, named]) == [whole]
    # The tracker then keeps one object.
    tracker = ObjectTracker()
    for stamp in (0.0, 1.0):
        tracker.update(merge_detections([whole, part, named]), stamp)
    assert len(tracker.tracks) == 1


def test_merge_keeps_neighbours_and_objects_on_others():
    plate = box("plate", 1.0, 0.0, 0.7135, [0.26, 0.26, 0.027])
    fork = box("fork", 1.0, 0.03, 0.71, [0.027, 0.195, 0.015], score=0.9)
    apple = box("apple", 1.05, -0.05, 0.765, [0.075, 0.075, 0.075])
    bowl = box("bowl", 2.0, 0.0, 0.7275, [0.16, 0.16, 0.055])
    spoon = box("spoon", 2.0, 0.0, 0.72, [0.194, 0.04, 0.02], score=0.9)
    # Two touching cans of one label, seen at 45 degrees (larger boxes).
    cans = [box("cola", 3.0 + d, -d, 0.81, [0.093, 0.093, 0.12]) for d in (0.0, 0.047)]
    detections = [plate, fork, apple, bowl, spoon, *cans]
    assert len(merge_detections(detections)) == len(detections)


def test_tracker_keeps_ids_votes_labels_and_forgets():
    tracker = ObjectTracker(gate_m=0.12, forget_after=10.0)
    first = tracker.update([detection("mug", 1.0), detection("apple", 1.5)], 0.0)
    mug, apple = first
    assert (mug.label, apple.label) == ("mug", "apple")
    # Noisy sightings average; a different name close by votes on the same track.
    seen = tracker.update(
        [detection("mug", 1.04), detection("orange", 1.52, score=0.5)], 1.0
    )
    assert [t.id for t in seen] == [mug.id, apple.id]
    assert mug.centre[0] == pytest.approx(1.02)
    assert apple.label == "apple" and apple.agreement == pytest.approx(0.8 / 1.3)
    # Another name farther than half the gate is another object.
    seen = tracker.update([detection("orange", 1.58)], 2.0)
    assert seen[0].id not in (mug.id, apple.id)
    orange = seen[0]
    # Tracks not seen for forget_after are dropped.
    tracker.update([detection("mug", 1.02)], 9.0)
    tracker.update([detection("mug", 1.02)], 11.5)
    assert set(tracker.tracks) == {mug.id, orange.id}


def test_tracker_pairs_same_labels_first():
    tracker = ObjectTracker(gate_m=0.12)
    spoon, bowl = tracker.update(
        [detection("spoon", 1.0), detection("bowl", 1.05)], 0.0
    )
    seen = tracker.update([detection("bowl", 1.02), detection("spoon", 1.03)], 1.0)
    assert {t.id: t.label for t in seen} == {spoon.id: "spoon", bowl.id: "bowl"}


def test_annotate_keeps_the_image_size():
    rgb = np.zeros((H, W, 3), np.uint8)
    image = annotate(rgb, [(rect(10, 50, 10, 50), "mug (cup 0.80)", True)])
    assert image.shape == rgb.shape
    assert tuple(image[49, 30]) == (0, 255, 0)


# --- Worker protocol ----------------------------------------------------------------


def run_worker(requests, vlm=None):
    output = io.StringIO()
    code = object_worker.serve(
        io.StringIO("".join(json.dumps(r) + "\n" for r in requests)),
        output,
        load_segmenter=ColourSegmenter,
        vlm=vlm,
    )
    return code, [json.loads(line) for line in output.getvalue().splitlines()]


def scene():
    rgb = np.full((H, W, 3), 128, np.uint8)
    rgb[100:180, 100:160] = (255, 0, 0)
    rgb[100:180, 300:360] = (0, 255, 0)
    rgb[300:360, 400:470] = (0, 0, 255)
    return rgb


LABELS = [{"label": "apple", "category": "fruit"}, {"label": "cola"}]


def test_worker_names_candidates():
    request = {
        "type": "detect",
        "id": 7,
        "image": encode_image(scene()),
        "prompts": ["can", "fruit"],
        "labels": LABELS,
    }
    code, (ready, response) = run_worker([request], ColourVlm())
    assert code == 0 and ready == {"type": "ready", "vlm": True}
    assert response["type"] == "detections" and response["id"] == 7
    # Both phrases find the same three blobs.
    assert response["candidates"] == 3
    found = {
        o["label"]: decode_mask(o, tuple(response["shape"]))
        for o in response["objects"]
    }
    assert set(found) == {"apple", "cola", None}
    assert found["apple"][140, 130] and found["cola"][140, 330]
    assert len(response["answers"]) == 1


def test_worker_reports_bad_requests_and_keeps_running():
    good = {
        "type": "detect",
        "id": 3,
        "image": encode_image(scene()),
        "prompts": ["can"],
        "labels": LABELS,
    }
    code, responses = run_worker(
        [
            {"type": "detect", "id": 1, "image": encode_image(scene())},
            {"type": "describe", "id": 2},
            good,
        ],
        ColourVlm(),
    )
    assert [r["type"] for r in responses] == ["ready", "error", "error", "detections"]
    assert [r.get("id") for r in responses[1:]] == [1, 2, 3]


def test_worker_rejects_bad_label_sizes():
    request = {
        "type": "detect",
        "id": 4,
        "image": encode_image(scene()),
        "prompts": ["can"],
        "labels": [{"label": "apple", "size": [0.07, 0.07]}],
    }
    _, responses = run_worker([request], ColourVlm())
    assert responses[1]["type"] == "error" and "size" in responses[1]["message"]


def test_worker_without_vlm_fails_each_request():
    code, responses = run_worker(
        [{"type": "detect", "id": 1, "image": "", "prompts": ["can"], "labels": LABELS}]
    )
    assert responses[0] == {"type": "ready", "vlm": False}
    assert responses[1]["type"] == "error" and "VLM" in responses[1]["message"]


# --- VLM timeout ---------------------------------------------------------------


def test_vlm_timeout_leaves_the_sheets_within_the_request_timeout():
    # 24 candidates: two sheets asked together.
    assert vlm_timeout(120.0, 24) == pytest.approx(120.0 - SAM_SECONDS)
    # 36 candidates: three sheets, the third after the first two.
    assert vlm_timeout(120.0, 36) == pytest.approx((120.0 - SAM_SECONDS) / 2)
    with pytest.raises(ValueError, match="request_timeout"):
        vlm_timeout(SAM_SECONDS + 5.0, 24)


def test_object_spec_passes_the_vlm_timeout_to_the_worker():
    command = object_spec(WorkerOptions(), 45.0).command
    assert command[:2] == ("-m", "aha_perception.object_worker")
    assert object_worker.parse_args(list(command[2:])).vlm_timeout == 45.0


class FirstSheetFails:
    """The first sheet's request fails at once; the second answers later."""

    def __init__(self):
        self.lock = threading.Lock()
        self.answered = 0

    def ask(self, prompt, photo, sheet):
        count = int(prompt.split("with every number from 1 to ")[1].split(".")[0])
        if count == SHEET_TILES:
            raise RuntimeError("VLM is not reachable: timed out")
        time.sleep(0.3)
        with self.lock:
            self.answered += 1
        return json.dumps({"labels": {}})


def test_failed_sheet_waits_for_the_other_sheets():
    # The error goes back only after every VLM request of the image ended,
    # so none of them runs on into the next image.
    masks = np.array(
        [rect(10, 40, 10 + 40 * i, 40 + 40 * i) for i in range(SHEET_TILES + 1)]
    )
    vlm = FirstSheetFails()
    with ThreadPoolExecutor(VLM_PARALLEL) as pool:
        with pytest.raises(RuntimeError, match="timed out"):
            object_worker.name_candidates(vlm, scene(), masks, LABELS, pool)
        assert vlm.answered == 1
