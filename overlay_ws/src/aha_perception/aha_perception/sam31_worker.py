"""Persistent SAM 3.1 worker speaking JSON lines over an SSH stdio channel."""

from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

from .sam31 import (
    Sam31Segmenter,
    load_capture,
    safe_error,
    save_result,
    validate_options,
    validate_threshold,
)
from .sam31_vlm import (
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_URL,
    MAX_CANDIDATES,
    OllamaClient,
    focal_lengths,
    segment_with_vlm,
)


RESULT_FILES = ("rgb.png", "labels.png", "overlay.png", "masks.npz", "result.json")
# Only VLM mode writes the numbered candidates the VLM chose from and the
# side-by-side crops of each same-product check.
OPTIONAL_RESULT_FILES = ("candidates.png",) + tuple(
    f"match_{n}.png" for n in range(1, MAX_CANDIDATES + 1)
)


def segment_request(
    segmenter,
    request: dict,
    vlm=None,
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
    match=True,
) -> dict:
    prompt = request["prompt"]
    validate_options(prompt, segmenter.threshold)
    prompt = prompt.strip()
    payload = base64.b64decode(request["rgbd"], validate=True)
    rgb, depth_m, metadata = load_capture(BytesIO(payload))
    started = time.perf_counter()
    if vlm is None:
        segmenter.prompt = prompt
        outputs = segmenter.predict(rgb)
        threshold, record, images = segmenter.threshold, None, None
    else:
        outputs, record, images = segment_with_vlm(
            segmenter,
            vlm,
            rgb,
            prompt,
            candidate_threshold,
            match,
            depth_m=depth_m,
            intrinsics=focal_lengths(metadata.get("camera_info")),
        )
        threshold = candidate_threshold
    with tempfile.TemporaryDirectory(prefix="sam31-result-") as directory:
        output = Path(directory)
        summary = save_result(
            output,
            rgb,
            depth_m,
            outputs,
            prompt=prompt,
            threshold=threshold,
            metadata=metadata,
            elapsed=time.perf_counter() - started,
            vlm=record,
            images=images,
        )
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as result:
            for name in RESULT_FILES + OPTIONAL_RESULT_FILES:
                if (output / name).exists():
                    result.write(output / name, name)
    return {
        "type": "result",
        "summary": summary,
        "archive": base64.b64encode(archive.getvalue()).decode("ascii"),
    }


def serve(
    input_stream,
    output_stream,
    *,
    threshold,
    checkpoint,
    factory=Sam31Segmenter,
    vlm=None,
    candidate_threshold=DEFAULT_CANDIDATE_THRESHOLD,
    match=True,
):
    def send(response):
        output_stream.write(json.dumps(response, allow_nan=False) + "\n")
        output_stream.flush()

    # Upstream prints model construction and prediction diagnostics. Keep the
    # protocol channel separate so those messages cannot corrupt responses.
    try:
        try:
            with redirect_stdout(sys.stderr):
                segmenter = factory("bottle", threshold, checkpoint)
        except Exception as exc:
            send({"type": "error", "message": safe_error(exc)})
            return 1
        send({"type": "ready"})
        for line in input_stream:
            # One bad request or upstream assertion must not discard the loaded
            # model; predict() closes its session before the error is reported.
            try:
                with redirect_stdout(sys.stderr):
                    response = segment_request(
                        segmenter, json.loads(line), vlm, candidate_threshold, match
                    )
            except Exception as exc:
                response = {"type": "error", "message": safe_error(exc)}
            send(response)
        return 0
    except BrokenPipeError:
        # The client has gone; there is nobody left to report to.
        return 1


def protocol_stream():
    """Return the original stdout and point fd 1 at stderr.

    redirect_stdout only covers Python writes; native libraries can still write
    to fd 1 directly, which would corrupt the JSON lines read by the client.
    """
    sys.stdout.flush()
    stream = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    return stream


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--vlm", action="store_true", help="choose SAM candidates with a VLM"
    )
    parser.add_argument("--vlm-url", default=DEFAULT_VLM_URL)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument(
        "--candidate-threshold", type=float, default=DEFAULT_CANDIDATE_THRESHOLD
    )
    parser.add_argument(
        "--no-match",
        action="store_true",
        help="skip comparing rejected candidates with a kept one",
    )
    args = parser.parse_args(argv)
    validate_threshold(args.candidate_threshold)
    return serve(
        sys.stdin,
        protocol_stream(),
        threshold=args.threshold,
        checkpoint=args.checkpoint,
        vlm=OllamaClient(args.vlm_url, args.vlm_model) if args.vlm else None,
        candidate_threshold=args.candidate_threshold,
        match=not args.no_match,
    )


if __name__ == "__main__":
    raise SystemExit(main())
