#!/usr/bin/env python3
"""object_worker protocol without CUDA: pure red / green / blue blobs are candidates.

Every phrase finds the same blobs, so the worker's pooling must drop the
duplicates. The VLM reads the colour at the centre of each close-up tile:
red is an apple, green a cola and blue is none of the list.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_perception import object_worker  # noqa: E402
from aha_perception.object_detection import SHEET_COLUMNS, TILE_PX  # noqa: E402

COLOURS = {"apple": (255, 0, 0), "cola": (0, 255, 0), "none": (0, 0, 255)}


def sheet_count(prompt):
    """The number of candidates a label_prompt asks about."""
    return int(prompt.split("with every number from 1 to ")[1].split(".")[0])


class ColourSegmenter:
    prompt = ""

    def predict(self, rgb, threshold):
        # JPEG blurs the colours; a tolerance keeps the masks whole.
        masks = [
            (np.abs(rgb.astype(int) - colour).max(axis=2) < 60)
            for colour in COLOURS.values()
        ]
        masks = [m for m in masks if m.sum() > 200]
        return {
            "out_binary_masks": np.array(masks, dtype=bool).reshape(-1, *rgb.shape[:2]),
            "out_probs": np.full(len(masks), 0.9, dtype=np.float32),
        }


class ColourVlm:
    def ask(self, prompt, photo, sheet):
        labels = {}
        for index in range(sheet_count(prompt)):
            y = (index // SHEET_COLUMNS) * TILE_PX + TILE_PX // 2
            x = (index % SHEET_COLUMNS) * TILE_PX + TILE_PX // 2
            pixel = sheet[y, x].astype(int)
            name = min(COLOURS, key=lambda k: np.abs(pixel - COLOURS[k]).sum())
            labels[str(index + 1)] = name
        return "```json\n" + json.dumps({"labels": labels}) + "\n```"


if __name__ == "__main__":
    raise SystemExit(
        object_worker.serve(
            sys.stdin,
            object_worker.protocol_stream(),
            load_segmenter=ColourSegmenter,
            vlm=ColourVlm(),
        )
    )
