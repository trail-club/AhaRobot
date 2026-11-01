"""SAM 3.1 image predictor for the experiment scripts (runs in the aharobot-sam31 image)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_predictor


def load_predictor():
    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path="/ckpt/sam3.1_multiplex.pt",
        compile=False,
        warm_up=False,
        use_fa3=False,
        use_rope_real=False,
        async_loading_frames=False,
    )
    # Same as the interactive worker: run in fp32 outside the explicit autocast below.
    predictor.bf16_context.__exit__(None, None, None)
    predictor.model.tracker.bf16_context.__exit__(None, None, None)
    return predictor


def segment(predictor, rgb, text=None, boxes=(), threshold=0.3):
    """Masks and scores for a text and/or boxes ([x, y, w, h] normalized to 0-1)."""
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        session = predictor.handle_request(
            {"type": "start_session", "resource_path": [Image.fromarray(rgb)]}
        )["session_id"]
        try:
            request = {
                "type": "add_prompt",
                "session_id": session,
                "frame_index": 0,
                "output_prob_thresh": threshold,
            }
            if boxes:
                request["bounding_boxes"] = [list(map(float, b)) for b in boxes]
                request["bounding_box_labels"] = [1] * len(boxes)
            if text:
                request["text"] = text
            out = predictor.handle_request(request)["outputs"]
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session})
    return np.asarray(out["out_binary_masks"], bool), np.asarray(
        out["out_probs"], float
    )


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def save_mask(path, mask):
    np.save(path, np.packbits(mask))


def work_dir():
    work = Path(sys.argv[1])
    (work / "out").mkdir(exist_ok=True)
    return work, json.loads((work / "jobs.json").read_text())
