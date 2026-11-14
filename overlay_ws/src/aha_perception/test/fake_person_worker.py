#!/usr/bin/env python3
"""person_worker protocol without CUDA: pure red / blue pixels are people.

The VLM names the mean colour of a crop and matches crops of the same mean
colour, so tests can tell the two people apart the same way the DGX does.
The pose model puts the 17 keypoints of an upright figure into each box; the
attribute question names a t shirt of the crop's colour.
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


class FigurePose:
    """Keypoints of an upright person facing the camera, filling the box."""

    # (x, y) as fractions of the box; the person's left is on the image right.
    LAYOUT = (
        (0.5, 0.05),  # nose
        (0.55, 0.03),
        (0.45, 0.03),
        (0.6, 0.05),
        (0.4, 0.05),
        (0.75, 0.2),  # left shoulder
        (0.25, 0.2),
        (0.8, 0.4),  # left elbow
        (0.2, 0.4),
        (0.8, 0.55),  # left wrist
        (0.2, 0.55),
        (0.65, 0.55),  # left hip
        (0.35, 0.55),
        (0.65, 0.75),  # left knee
        (0.35, 0.75),
        (0.65, 0.95),  # left ankle
        (0.35, 0.95),
    )

    def predict(self, rgb, boxes):
        return [
            np.array(
                [
                    [x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), 0.9]
                    for fx, fy in self.LAYOUT
                ]
            )
            for x0, y0, x1, y1 in boxes
        ]


class ColourVlm:
    def ask(self, prompt, *images):
        if "upper_garments" in prompt:
            colour = "red" if redder(images[-1]) else "blue"
            return json.dumps(
                {
                    "posture": "standing",
                    "gesture": "none",
                    "upper_garments": [{"type": "t-shirt", "colour": colour}],
                }
            )
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
            load_pose=FigurePose,
        )
    )
