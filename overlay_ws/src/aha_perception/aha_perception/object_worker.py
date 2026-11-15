"""DGX worker for object_perception: SAM 3.1 candidates named by the VLM.

Speaks JSON lines over the SSH stdio channel like person_worker. Requests
carry an `id` that the response repeats and are answered one at a time.
SAM 3.1 runs once per phrase; the pooled candidates go to the VLM in sheets
of SHEET_TILES (two requests at a time), each with the photo with the
candidates numbered and their close-ups. Images are JPEG base64.

    -> {"type": "detect", "id": 1, "image": <image>, "prompts": ["can", ...],
        "labels": [{"label": "mug", "category": "tableware",
                    "size": [0.12, 0.09, 0.08]}, ...],
        "threshold": 0.4, "max_candidates": 24, "border_px": 2}
    <- {"type": "detections", "id": 1, "shape": [h, w], "candidates": n,
        "objects": [{"score", "phrase", "label" (null: none), "bbox", "mask"}],
        "answers": [<VLM reply per sheet>], "seconds": {"decode", "sam", "vlm"}}
    <- {"type": "error", "id": n, "message": "..."}

All candidates are returned, also those the VLM did not name, so that the
robot can show what was rejected. With `border_px` > 0, masks cut off by the
image edge are dropped before the VLM.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

import numpy as np

from .object_detection import (
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_MAX_CANDIDATES,
    SHEET_TILES,
    crop_sheet,
    label_prompt,
    parse_labels,
    pool_candidates,
)
from .people import decode_image, encode_mask
from .sam31 import render_candidates, validate_threshold
from .sam31_vlm import DEFAULT_VLM_MODEL, DEFAULT_VLM_URL, VlmClient
from .worker_io import protocol_stream, safe_error, send_json

# Concurrent VLM requests per image; the Qwen3.8 server takes 4 in all.
VLM_PARALLEL = 2
MAX_PROMPTS = 24


def _strings(value, what):
    if not isinstance(value, list) or not all(
        isinstance(v, str) and v.strip() for v in value
    ):
        raise ValueError(f"{what} must be a list of nonempty strings")
    return [v.strip() for v in value]


def _labels(value):
    if not isinstance(value, list) or not value:
        raise ValueError("labels must be a nonempty list")
    labels = []
    for entry in value:
        if not isinstance(entry, dict) or not str(entry.get("label", "")).strip():
            raise ValueError("each label needs a nonempty label")
        size = entry.get("size")
        if size is not None and (
            not isinstance(size, list)
            or len(size) != 3
            or not all(isinstance(v, (int, float)) and v > 0 for v in size)
        ):
            raise ValueError("a label's size must be three positive numbers")
        labels.append(
            {
                "label": str(entry["label"]).strip(),
                "category": str(entry.get("category", "")).strip(),
                "size": size,
            }
        )
    return labels


def name_candidates(vlm, rgb, masks, labels, pool=None):
    """The VLM's label (or None) per mask, asked in sheets of SHEET_TILES,
    and the VLM's answer per sheet."""
    names = [label["label"] for label in labels]

    def ask(start):
        chunk = masks[start : start + SHEET_TILES]
        answer = vlm.ask(
            label_prompt(labels, len(chunk)),
            render_candidates(rgb, chunk),
            crop_sheet(rgb, chunk),
        )
        return parse_labels(answer, len(chunk), names), answer

    starts = range(0, len(masks), SHEET_TILES)
    if pool is None or len(starts) == 1:
        results = [ask(start) for start in starts]
    else:
        results = list(pool.map(ask, starts))
    return [label for chunk, _ in results for label in chunk], [a for _, a in results]


def detect(segmenter, vlm, request: dict, pool=None) -> dict:
    started = time.perf_counter()
    prompts = _strings(request.get("prompts"), "prompts")
    if len(prompts) > MAX_PROMPTS:
        raise ValueError(f"at most {MAX_PROMPTS} prompts")
    labels = _labels(request.get("labels"))
    threshold = float(request.get("threshold", DEFAULT_CANDIDATE_THRESHOLD))
    validate_threshold(threshold)
    max_candidates = int(request.get("max_candidates", DEFAULT_MAX_CANDIDATES))
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    border_px = int(request.get("border_px", 0))
    rgb = decode_image(request["image"])
    decoded = time.perf_counter()
    results = []
    for phrase in prompts:
        segmenter.prompt = phrase
        results.append((phrase, segmenter.predict(rgb, threshold)))
    candidates = pool_candidates(
        results, rgb.shape[:2], max_candidates=max_candidates, border_px=border_px
    )
    segmented = time.perf_counter()
    masks = np.array([c.mask for c in candidates], dtype=bool).reshape(
        -1, *rgb.shape[:2]
    )
    names, answers = (
        name_candidates(vlm, rgb, masks, labels, pool) if len(masks) else ([], [])
    )
    finished = time.perf_counter()
    return {
        "type": "detections",
        "id": request.get("id"),
        "shape": list(rgb.shape[:2]),
        "candidates": len(candidates),
        "objects": [
            {
                "score": candidate.score,
                "phrase": candidate.phrase,
                "label": name,
                **encode_mask(candidate.mask),
            }
            for candidate, name in zip(candidates, names)
        ],
        "answers": answers,
        "seconds": {
            "decode": decoded - started,
            "sam": segmented - decoded,
            "vlm": finished - segmented,
        },
    }


def serve(input_stream, output_stream, *, load_segmenter, vlm):
    def send(response):
        send_json(output_stream, response)

    def fail(request_id, exc):
        send({"type": "error", "id": request_id, "message": safe_error(exc)})

    try:
        try:
            with redirect_stdout(sys.stderr):
                segmenter = load_segmenter()
        except Exception as exc:
            fail(None, exc)
            return 1
        send({"type": "ready", "vlm": vlm is not None})
        with ThreadPoolExecutor(VLM_PARALLEL) as pool:
            for line in input_stream:
                request = None
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("a request must be a JSON object")
                    kind = request.get("type")
                    if kind != "detect":
                        raise ValueError(f"unknown request type {kind!r}")
                    if vlm is None:
                        raise ValueError("the worker was started without a VLM")
                    with redirect_stdout(sys.stderr):
                        response = detect(segmenter, vlm, request, pool)
                except BrokenPipeError:
                    raise
                except Exception as exc:
                    fail(request.get("id") if isinstance(request, dict) else None, exc)
                    continue
                send(response)
        return 0
    except BrokenPipeError:
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-url", default=DEFAULT_VLM_URL, help="empty disables")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    args = parser.parse_args(argv)

    def load_segmenter():
        from .sam31 import Sam31Segmenter

        # The prompt is set per phrase before each prediction.
        return Sam31Segmenter("object", DEFAULT_CANDIDATE_THRESHOLD, args.checkpoint)

    vlm = VlmClient(args.vlm_url, args.vlm_model) if args.vlm_url else None
    return serve(sys.stdin, protocol_stream(), load_segmenter=load_segmenter, vlm=vlm)


if __name__ == "__main__":
    raise SystemExit(main())
