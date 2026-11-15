#!/usr/bin/env python3
"""person_worker protocol without CUDA: pure red / blue pixels are people.

The VLM names the mean colour of a crop and matches crops of the same mean
colour, so tests can tell the two people apart the same way the DGX does.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_perception import person_worker  # noqa: E402

COLOURS = {"red": (255, 0, 0), "blue": (0, 0, 255)}


class ColourSegmenter:
    def predict(self, rgb, threshold):
        # JPEG blurs the colours; a tolerance keeps the masks whole.
        masks = [
            (np.abs(rgb.astype(int) - colour).max(axis=2) < 60)
            for colour in COLOURS.values()
        ]
        masks = [m for m in masks if m.any()]
        return {
            "out_binary_masks": np.array(masks, dtype=bool).reshape(-1, *rgb.shape[:2]),
            "out_probs": np.full(len(masks), 0.9, dtype=np.float32),
        }


def redder(image):
    r, _, b = image.reshape(-1, 3).mean(axis=0)
    return r > b


class ColourVlm:
    def ask(self, prompt, *images):
        *references, image = images
        if not references:
            return json.dumps(
                {"features": ["red shirt" if redder(image) else "blue shirt"]}
            )
        same = [i for i, r in enumerate(references, 1) if redder(r) == redder(image)]
        return json.dumps({"photo": same[0] if same else None})


if __name__ == "__main__":
    raise SystemExit(
        person_worker.serve(
            sys.stdin,
            person_worker.protocol_stream(),
            threshold=0.5,
            load_segmenter=lambda prompt, threshold: ColourSegmenter(),
            vlm=ColourVlm(),
        )
    )
