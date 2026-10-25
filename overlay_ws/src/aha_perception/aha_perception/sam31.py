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
from PIL import Image, ImageDraw, ImageFont


MODEL_ID = "facebook/sam3.1"
# Overlay style: diagonal hatch stripes plus a solid outline per object.
HATCH_PERIOD_PX = 10
HATCH_WIDTH_PX = 3
OUTLINE_PX = 2


def safe_error(error: Exception) -> str:
    message = str(error)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", message)


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")


def validate_options(prompt: str, threshold: float) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    validate_threshold(threshold)


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

    def predict(self, rgb: np.ndarray, threshold: float | None = None) -> dict:
        validate_rgb(rgb)
        if threshold is None:
            threshold = self.threshold
        validate_threshold(threshold)
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
                        "output_prob_thresh": threshold,
                    }
                )
                return response["outputs"]
            finally:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session}
                )


def erode(mask: np.ndarray) -> np.ndarray:
    """4-neighbour binary erosion; pixels on the image border are eroded."""
    padded = np.pad(mask, 1, constant_values=False)
    return (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def label_color(label: int) -> np.ndarray:
    """Distinct, deterministic RGB colour per label (also in result.json)."""
    return np.array(
        [(label * 73) % 192 + 63, (label * 131) % 192 + 63, (label * 47) % 192 + 63],
        dtype=np.float32,
    )


def color_hex(label: int) -> str:
    return "#" + "".join(f"{int(c):02X}" for c in label_color(label))


def _hatch_stripes(height: int, width: int) -> np.ndarray:
    ys_grid, xs_grid = np.indices((height, width))
    return (xs_grid + ys_grid) % HATCH_PERIOD_PX < HATCH_WIDTH_PX


def _paint(overlay, rgb, visible, color, stripes) -> None:
    """Hatch the mask interior and draw a solid outline in place."""
    interior = visible
    for _ in range(OUTLINE_PX):
        interior = erode(interior)
    tint = visible & ~stripes
    overlay[tint] = (0.8 * rgb[tint] + 0.2 * color).astype(np.uint8)
    overlay[(visible & stripes) | (visible & ~interior)] = color.astype(np.uint8)


def render_candidates(rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Draw each candidate with its own colour and a numbered tag (1..N).

    Masks are drawn in reverse so candidate 1 stays on top where they overlap.
    The image is what a VLM sees when choosing which candidates to keep.
    """
    validate_rgb(rgb)
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
        raise ValueError("candidate masks must be NxHxW matching the RGB image")
    overlay = rgb.copy()
    stripes = _hatch_stripes(*rgb.shape[:2])
    for index in reversed(range(len(masks))):
        if masks[index].any():
            _paint(overlay, rgb, masks[index], label_color(index + 1), stripes)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, rgb.shape[0] // 24))
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        text = str(index + 1)
        x, y = int(xs.min()) + 2, int(ys.min()) + 2
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0))
        draw.text(
            (x, y), text, font=font, fill=tuple(int(c) for c in label_color(index + 1))
        )
    return np.asarray(image)


def make_result(rgb: np.ndarray, depth_m: np.ndarray, outputs: dict, threshold: float):
    """Resolve mask overlaps by score and summarize valid aligned Z-depth.

    Labels 1..N are frame-local; 0 is background. Summaries use the resolved
    labels, so their pixel counts and depth statistics agree with the PNG.
    """
    validate_threshold(threshold)
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
    stripes = _hatch_stripes(height, width)
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
                "color": color_hex(label),
            }
        )
        _paint(overlay, rgb, visible, label_color(label), stripes)
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
    vlm=None,
    images=None,
):
    """Save the result files; VLM mode adds a `vlm` record and its named images."""
    labels, overlay, objects = make_result(rgb, depth_m, outputs, threshold)
    output.mkdir(parents=True, exist_ok=True)
    for name, image in (images or {}).items():
        Image.fromarray(image).save(output / name)
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
    if vlm is not None:
        summary["vlm"] = vlm
    (output / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
