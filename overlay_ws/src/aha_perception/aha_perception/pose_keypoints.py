"""ViTPose body keypoints for person boxes; runs in the DGX person worker.

Top-down: SAM 3.1 already found each person, so ViTPose only places the 17
COCO keypoints inside their boxes. torch and transformers are imported on
load, so the rest of aha_perception does not need them.
"""

from __future__ import annotations

import numpy as np

DEFAULT_POSE_MODEL = "usyd-community/vitpose-base-simple"  # Apache-2.0


class VitPose:
    def __init__(self, model: str = DEFAULT_POSE_MODEL, device: str | None = None):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = VitPoseForPoseEstimation.from_pretrained(model).to(self.device)
        self.model.eval()

    def predict(self, rgb: np.ndarray, boxes: list[list[int]]) -> list[np.ndarray]:
        """(17, 3) [u, v, score] per [x0, y0, x1, y1] box, in image pixels."""
        if not boxes:
            return []
        from PIL import Image

        coco = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in boxes]
        inputs = self.processor(
            Image.fromarray(rgb), boxes=[coco], return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_pose_estimation(outputs, boxes=[coco])[0]
        return [
            np.column_stack(
                [
                    r["keypoints"].float().cpu().numpy(),
                    r["scores"].float().cpu().numpy(),
                ]
            )
            for r in results
        ]
