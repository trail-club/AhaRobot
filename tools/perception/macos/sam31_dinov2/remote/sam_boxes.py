"""Option B: Qwen boxes (and an optional phrase) as SAM 3.1 prompts on the same photo.

jobs.json: [{"key", "file", "boxes": [[x0, y0, x1, y1] px], "variants": {name: text|null}}]
"""

import json
import time

import numpy as np
from PIL import Image

from sam_session import bbox, load_predictor, save_mask, segment, work_dir

work, jobs = work_dir()
predictor = load_predictor()
results = []
for job in jobs:
    rgb = np.asarray(Image.open(work / job["file"]).convert("RGB"))
    h, w = rgb.shape[:2]
    boxes = [
        [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]
        for x0, y0, x1, y1 in job["boxes"]
    ]
    for variant, text in job["variants"].items():
        start = time.time()
        masks, scores = segment(predictor, rgb, text, boxes)
        detections = []
        for mask, score in zip(masks, scores):
            if not mask.any():
                continue
            detections.append({"score": round(float(score), 3), "bbox": bbox(mask)})
            save_mask(
                work / "out" / f"{job['key']}_{variant}_{len(detections)}.npy", mask
            )
        results.append(
            {
                "key": job["key"],
                "variant": variant,
                "text": text,
                "seconds": round(time.time() - start, 2),
                "detections": detections,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
(work / "out" / "results.json").write_text(
    json.dumps(results, indent=1, ensure_ascii=False)
)
