"""DGX worker for person_perception: SAM 3.1 people and VLM questions about them.

Speaks JSON lines over the SSH stdio channel like sam31_worker. Requests carry
an `id` that the response repeats. Detection runs in the reading thread, one
image at a time; VLM requests take seconds, so they run one at a time in their
own thread and may answer out of order. Images are JPEG base64; the VLM gets
person crops with the background darkened.

    -> {"type": "detect", "id": 1, "image": <image>}
    <- {"type": "detections", "id": 1, "shape": [h, w], "seconds": {...},
        "people": [{"score", "bbox", "mask"}]}
    -> {"type": "describe", "id": 2, "image": <crop>}
    <- {"type": "description", "id": 2, "features": [...], "seconds": s}
    -> {"type": "match", "id": 3, "image": <crop>, "references": [<crop>, ...]}
    <- {"type": "match", "id": 3, "index": <into references> or null, "seconds": s}
    <- {"type": "error", "id": n, "message": "..."}
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

from .people import (
    DESCRIBE_PROMPT,
    decode_image,
    encode_mask,
    match_prompt,
    parse_features,
    parse_match,
)
from .sam31 import safe_error, validate_threshold
from .sam31_vlm import DEFAULT_VLM_MODEL, DEFAULT_VLM_URL, VlmClient
from .sam31_worker import protocol_stream


MIN_MASK_PIXELS = 400


def detect(segmenter, request: dict, threshold: float) -> dict:
    started = time.perf_counter()
    rgb = decode_image(request["image"])
    decoded = time.perf_counter()
    outputs = segmenter.predict(rgb, threshold)
    finished = time.perf_counter()
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    scores = np.asarray(outputs["out_probs"], dtype=np.float32)
    people = [
        {"score": float(scores[i]), **encode_mask(masks[i])}
        for i in np.argsort(-scores, kind="stable")
        if scores[i] >= threshold and masks[i].sum() >= MIN_MASK_PIXELS
    ]
    return {
        "type": "detections",
        "id": request.get("id"),
        "shape": list(rgb.shape[:2]),
        "people": people,
        "seconds": {"decode": decoded - started, "sam": finished - decoded},
    }


def describe(vlm, request: dict) -> dict:
    started = time.perf_counter()
    crop = decode_image(request["image"])
    features = parse_features(vlm.ask(DESCRIBE_PROMPT, crop))
    return {
        "type": "description",
        "id": request.get("id"),
        "features": features,
        "seconds": time.perf_counter() - started,
    }


def match(vlm, request: dict) -> dict:
    started = time.perf_counter()
    references = [decode_image(image) for image in request["references"]]
    prompt = match_prompt(len(references))
    answer = vlm.ask(prompt, *references, decode_image(request["image"]))
    return {
        "type": "match",
        "id": request.get("id"),
        "index": parse_match(answer, len(references)),
        "seconds": time.perf_counter() - started,
    }


VLM_REQUESTS = {"describe": describe, "match": match}


def serve(
    input_stream, output_stream, *, threshold, load_segmenter, vlm, prompt="person"
):
    lock = threading.Lock()

    def send(response):
        with lock:
            output_stream.write(json.dumps(response, allow_nan=False) + "\n")
            output_stream.flush()

    def fail(request_id, exc):
        send({"type": "error", "id": request_id, "message": safe_error(exc)})

    vlm_requests = queue.Queue()

    def ask_vlm():
        while (request := vlm_requests.get()) is not None:
            try:
                with redirect_stdout(sys.stderr):
                    send(VLM_REQUESTS[request["type"]](vlm, request))
            except BrokenPipeError:
                return
            except Exception as exc:
                fail(request.get("id"), exc)

    try:
        try:
            with redirect_stdout(sys.stderr):
                segmenter = load_segmenter(prompt, threshold)
        except Exception as exc:
            fail(None, exc)
            return 1
        thread = threading.Thread(target=ask_vlm, daemon=True)
        thread.start()
        send({"type": "ready", "prompt": prompt, "vlm": vlm is not None})
        for line in input_stream:
            request = None
            try:
                request = json.loads(line)
                kind = request.get("type")
                if kind in VLM_REQUESTS:
                    if vlm is None:
                        raise ValueError("the worker was started without a VLM")
                    vlm_requests.put(request)
                    continue
                if kind != "detect":
                    raise ValueError(f"unknown request type {kind!r}")
                with redirect_stdout(sys.stderr):
                    response = detect(segmenter, request, threshold)
            except Exception as exc:
                fail(request.get("id") if isinstance(request, dict) else None, exc)
                continue
            send(response)
        vlm_requests.put(None)
        thread.join(timeout=5)
        return 0
    except BrokenPipeError:
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--vlm-url", default=DEFAULT_VLM_URL, help="empty disables")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    args = parser.parse_args(argv)
    validate_threshold(args.threshold)

    def load_segmenter(prompt, threshold):
        from .sam31 import Sam31Segmenter

        return Sam31Segmenter(prompt, threshold, args.checkpoint)

    vlm = None
    if args.vlm_url:
        vlm = VlmClient(args.vlm_url, args.vlm_model)
    return serve(
        sys.stdin,
        protocol_stream(),
        threshold=args.threshold,
        load_segmenter=load_segmenter,
        vlm=vlm,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
