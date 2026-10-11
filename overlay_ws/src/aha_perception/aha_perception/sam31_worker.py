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
)


RESULT_FILES = ("rgb.png", "labels.png", "overlay.png", "masks.npz", "result.json")


def segment_request(segmenter, request: dict) -> dict:
    prompt = request["prompt"]
    validate_options(prompt, segmenter.threshold)
    prompt = prompt.strip()
    payload = base64.b64decode(request["rgbd"], validate=True)
    rgb, depth_m, metadata = load_capture(BytesIO(payload))
    segmenter.prompt = prompt
    started = time.perf_counter()
    outputs = segmenter.predict(rgb)
    with tempfile.TemporaryDirectory(prefix="sam31-result-") as directory:
        output = Path(directory)
        summary = save_result(
            output,
            rgb,
            depth_m,
            outputs,
            prompt=prompt,
            threshold=segmenter.threshold,
            metadata=metadata,
            elapsed=time.perf_counter() - started,
        )
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as result:
            for name in RESULT_FILES:
                result.write(output / name, name)
    return {
        "type": "result",
        "summary": summary,
        "archive": base64.b64encode(archive.getvalue()).decode("ascii"),
    }


def serve(
    input_stream, output_stream, *, threshold, checkpoint, factory=Sam31Segmenter
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
                    response = segment_request(segmenter, json.loads(line))
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
    args = parser.parse_args(argv)
    return serve(
        sys.stdin,
        protocol_stream(),
        threshold=args.threshold,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
