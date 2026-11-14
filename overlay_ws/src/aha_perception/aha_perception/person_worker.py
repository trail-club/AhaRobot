"""DGX worker for person_perception: SAM 3.1 people and VLM questions about them.

Speaks JSON lines over the SSH stdio channel like sam31_worker. Requests carry
an `id` that the response repeats. Detection runs in the reading thread, one
image at a time; VLM requests take seconds, so they run one at a time in their
own thread and may answer out of order. Images are JPEG base64; the VLM gets
person crops with the background darkened.

    -> {"type": "detect", "id": 1, "image": <image>}
    <- {"type": "detections", "id": 1, "shape": [h, w], "seconds": {...},
        "people": [{"score", "bbox", "mask", "keypoints": [[[u, v, s] x 17], ...]}]}
    -> {"type": "describe", "id": 2, "image": <crop>}
    <- {"type": "description", "id": 2, "features": [...], "seconds": s}
    -> {"type": "match", "id": 3, "image": <crop>, "references": [<crop>, ...]}
    <- {"type": "match", "id": 3, "index": <into references> or null, "seconds": s}
    -> {"type": "attributes", "id": 4, "images": [<crop>, ...], "interval": s}
    <- {"type": "attributes", "id": 4, "posture", "gesture", "garments": [...],
        "seconds": s}
    <- {"type": "error", "id": n, "message": "..."}

With a pose model, each person has ViTPose keypoints for every part that
people.split_sideways cuts the mask into (one part for most masks), because the
robot may keep the parts as separate people.
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
    bbox_of,
    decode_image,
    encode_mask,
    match_prompt,
    parse_features,
    parse_match,
    split_sideways,
)
from .person_attributes import attributes_prompt, parse_attributes
from .pose_keypoints import DEFAULT_POSE_MODEL
from .sam31 import validate_threshold
from .sam31_vlm import DEFAULT_VLM_MODEL, DEFAULT_VLM_URL, VlmClient
from .worker_io import protocol_stream, safe_error, send_json


MIN_MASK_PIXELS = 400
# Photos of one person per attributes request: the Qwen3.8 server takes 4.
MAX_ATTRIBUTE_IMAGES = 3
# Margins around a mask's box for ViTPose, as shares of the box height.
POSE_PAD_TOP, POSE_PAD_SIDE, POSE_PAD_BOTTOM = 0.25, 0.25, 0.05


def pose_box(box, shape):
    """The person's box with room for raised or stretched arms, which SAM can miss."""
    x0, y0, x1, y1 = box
    height, width = shape
    size = y1 - y0
    return [
        max(0, int(x0 - POSE_PAD_SIDE * size)),
        max(0, int(y0 - POSE_PAD_TOP * size)),
        min(width, int(x1 + POSE_PAD_SIDE * size)),
        min(height, int(y1 + POSE_PAD_BOTTOM * size)),
    ]


def detect(segmenter, request: dict, threshold: float, pose=None) -> dict:
    started = time.perf_counter()
    rgb = decode_image(request["image"])
    decoded = time.perf_counter()
    outputs = segmenter.predict(rgb, threshold)
    finished = time.perf_counter()
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    scores = np.asarray(outputs["out_probs"], dtype=np.float32)
    kept = [
        i
        for i in np.argsort(-scores, kind="stable")
        if scores[i] >= threshold and masks[i].sum() >= MIN_MASK_PIXELS
    ]
    people = [{"score": float(scores[i]), **encode_mask(masks[i])} for i in kept]
    seconds = {"decode": decoded - started, "sam": finished - decoded}
    if pose is not None:
        boxes, owners = [], []
        for person, i in zip(people, kept):
            person["keypoints"] = []
            for part in split_sideways(masks[i]):
                boxes.append(pose_box(bbox_of(part), rgb.shape[:2]))
                owners.append(person)
        estimated = pose.predict(rgb, boxes)
        for person, keypoints in zip(owners, estimated):
            person["keypoints"].append(
                [
                    [round(float(u), 1), round(float(v), 1), round(float(s), 3)]
                    for u, v, s in keypoints
                ]
            )
        seconds["pose"] = time.perf_counter() - finished
    return {
        "type": "detections",
        "id": request.get("id"),
        "shape": list(rgb.shape[:2]),
        "people": people,
        "seconds": seconds,
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


def attributes(vlm, request: dict) -> dict:
    started = time.perf_counter()
    images = [decode_image(image) for image in request["images"]]
    if not 1 <= len(images) <= MAX_ATTRIBUTE_IMAGES:
        raise ValueError(f"attributes need 1 to {MAX_ATTRIBUTE_IMAGES} images")
    prompt = attributes_prompt(len(images), float(request.get("interval", 0.0)))
    answer = parse_attributes(vlm.ask(prompt, *images))
    return {
        "type": "attributes",
        "id": request.get("id"),
        "posture": answer.posture,
        "gesture": answer.gesture,
        "garments": answer.garments,
        "seconds": time.perf_counter() - started,
    }


VLM_REQUESTS = {"describe": describe, "match": match, "attributes": attributes}


def serve(
    input_stream,
    output_stream,
    *,
    threshold,
    load_segmenter,
    vlm,
    prompt="person",
    load_pose=None,
):
    lock = threading.Lock()

    def send(response):
        with lock:
            send_json(output_stream, response)

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
                pose = load_pose() if load_pose is not None else None
        except Exception as exc:
            fail(None, exc)
            return 1
        thread = threading.Thread(target=ask_vlm, daemon=True)
        thread.start()
        send(
            {
                "type": "ready",
                "prompt": prompt,
                "vlm": vlm is not None,
                "keypoints": pose is not None,
            }
        )
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
                    response = detect(segmenter, request, threshold, pose)
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
    parser.add_argument(
        "--pose-model", default=DEFAULT_POSE_MODEL, help="empty disables keypoints"
    )
    args = parser.parse_args(argv)
    validate_threshold(args.threshold)

    def load_segmenter(prompt, threshold):
        from .sam31 import Sam31Segmenter

        return Sam31Segmenter(prompt, threshold, args.checkpoint)

    def load_pose():
        from .pose_keypoints import VitPose

        return VitPose(args.pose_model)

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
        load_pose=load_pose if args.pose_model else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
