"""SAM 3.1 single-frame segmentation using the official multiplex predictor.

CUDA dependencies are loaded only when constructing the predictor, so capture,
result processing and unit tests do not require a GPU or model download.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import BinaryIO
from zipfile import BadZipFile

import numpy as np
from PIL import Image


MODEL_ID = "facebook/sam3.1"


def safe_error(error: Exception) -> str:
    message = str(error)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", message)


def validate_options(prompt: str, threshold: float) -> None:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")


def validate_rgb(rgb: np.ndarray) -> None:
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) < 1
    ):
        raise ValueError("RGB must be a nonempty uint8 HxWx3 array")


def validate_rgbd(rgb: np.ndarray, depth_m: np.ndarray) -> None:
    validate_rgb(rgb)
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError("depth must be aligned to RGB with the same dimensions")
    if not np.issubdtype(depth_m.dtype, np.floating):
        raise ValueError("depth_m must contain floating-point depths in meters")


def load_capture(source: Path | str | BinaryIO):
    """Load the capture contract without model dependencies or pickled arrays."""
    try:
        capture = np.load(source, allow_pickle=False)
        if not isinstance(capture, np.lib.npyio.NpzFile):
            raise ValueError("capture must be a RGB-D .npz archive")
        with capture:
            rgb = capture["rgb"].copy()
            depth_m = capture["depth_m"].copy()
            metadata_json = capture["metadata"]
            if metadata_json.ndim != 0 or not isinstance(metadata_json.item(), str):
                raise ValueError("capture metadata must be a scalar JSON string")
            metadata = json.loads(metadata_json.item())
    except BadZipFile as exc:
        raise ValueError("capture is not a valid RGB-D archive") from exc
    validate_rgbd(rgb, depth_m)
    if not isinstance(metadata, dict):
        raise ValueError("capture metadata must be a JSON object")
    return rgb, depth_m, metadata


class Sam31Segmenter:
    """One independent session per RGB frame; object IDs are not track IDs."""

    def __init__(self, prompt: str, threshold: float, checkpoint: Path):
        validate_options(prompt, threshold)
        if not Path(checkpoint).is_file():
            raise ValueError("checkpoint must be an existing SAM 3.1 checkpoint file")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM 3.1's official predictor requires a CUDA GPU. "
                "Capture RGB-D on macOS and run inference in a GPU development container."
            )
        from sam3.model_builder import build_sam3_predictor

        self.prompt = prompt.strip()
        self.threshold = threshold
        self.torch = torch
        self.predictor = build_sam3_predictor(
            version="sam3.1",
            checkpoint_path=str(checkpoint),
            compile=False,
            warm_up=False,
            use_fa3=False,
            use_rope_real=False,
            async_loading_frames=False,
        )
        # The pinned release enters autocast globally in its constructor.
        # Restore the caller's state and enter it per prediction.
        self.predictor.bf16_context.__exit__(None, None, None)
        self.predictor.model.tracker.bf16_context.__exit__(None, None, None)

    def predict(self, rgb: np.ndarray) -> dict:
        validate_rgb(rgb)
        with (
            self.torch.inference_mode(),
            self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16),
        ):
            session = self.predictor.handle_request(
                {"type": "start_session", "resource_path": [Image.fromarray(rgb)]}
            )["session_id"]
            try:
                response = self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session,
                        "frame_index": 0,
                        "text": self.prompt,
                        "output_prob_thresh": self.threshold,
                    }
                )
                return response["outputs"]
            finally:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session}
                )


def make_result(rgb: np.ndarray, depth_m: np.ndarray, outputs: dict, threshold: float):
    """Resolve mask overlaps by score and summarize valid aligned Z-depth.

    Labels 1..N are frame-local; 0 is background. Summaries use the resolved
    labels, so their pixel counts and depth statistics agree with the PNG.
    """
    validate_options("result", threshold)
    validate_rgbd(rgb, depth_m)
    height, width = rgb.shape[:2]
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    scores = np.asarray(outputs["out_probs"], dtype=np.float32)
    if scores.ndim != 1 or masks.shape != (len(scores), height, width):
        raise ValueError("SAM 3.1 masks/scores have unexpected dimensions")
    if not np.isfinite(scores).all():
        raise ValueError("SAM 3.1 scores must be finite")
    order = np.argsort(-scores, kind="stable")
    labels = np.zeros((height, width), dtype=np.uint16)
    objects = []
    overlay = rgb.copy()
    for index in order:
        if scores[index] < threshold:
            continue
        visible = masks[index] & (labels == 0)
        ys, xs = np.nonzero(visible)
        if not len(xs):
            continue
        label = len(objects) + 1
        if label > np.iinfo(np.uint16).max:
            raise ValueError("too many instances for mono16 labels")
        labels[visible] = label
        samples = depth_m[visible]
        valid = samples[np.isfinite(samples) & (samples > 0)]
        objects.append(
            {
                "label": label,
                "score": float(scores[index]),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ],
                "pixels": int(len(xs)),
                "valid_depth_pixels": int(len(valid)),
                "median_depth_m": float(np.median(valid)) if len(valid) else None,
            }
        )
        color = np.array(
            [
                (label * 73) % 192 + 63,
                (label * 131) % 192 + 63,
                (label * 47) % 192 + 63,
            ],
            dtype=np.float32,
        )
        overlay[visible] = (0.55 * rgb[visible] + 0.45 * color).astype(np.uint8)
    return labels, overlay, objects


def save_result(
    output: Path,
    rgb,
    depth_m,
    outputs,
    *,
    prompt,
    threshold,
    metadata,
    elapsed,
):
    labels, overlay, objects = make_result(rgb, depth_m, outputs, threshold)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output / "rgb.png")
    Image.fromarray(labels).save(output / "labels.png")
    Image.fromarray(overlay).save(output / "overlay.png")
    np.savez_compressed(
        output / "masks.npz",
        masks=outputs["out_binary_masks"],
        scores=outputs["out_probs"],
        labels=labels,
        depth_m=depth_m,
    )
    summary = {
        "model": MODEL_ID,
        "model_revision": None,
        "prompt": prompt,
        "threshold": threshold,
        "inference_seconds": elapsed,
        "source": metadata,
        "objects": objects,
    }
    (output / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
