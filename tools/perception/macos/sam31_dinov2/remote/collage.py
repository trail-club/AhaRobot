"""Reference collage: photo | reference views on one canvas, boxes on the references.

SAM 3.1 exemplar boxes only work inside the same image, so the references are stitched
next to the photo; masks found in the photo half are the detections.
jobs.json: {"targets": [{"name", "file"}], "references": {view: {"file", "box"}},
            "conditions": [[reference set, text|null], ...]}
"""

import json
import time

import numpy as np
from PIL import Image

from sam_session import bbox, load_predictor, segment, work_dir

work, jobs = work_dir()
predictor = load_predictor()


def padded_crop(rgb, box, pad=0.4):
    """Reference object with some context, and its box inside the crop."""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    h, w = rgb.shape[:2]
    a0, b0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    a1, b1 = min(w, int(x1 + px)), min(h, int(y1 + py))
    return rgb[b0:b1, a0:a1], (x0 - a0, y0 - b0, x1 - a0, y1 - b0)


def panel(crops, height):
    """Crops in one column (one crop) or a 2x2 grid, padded to `height` with white."""
    cols = 1 if len(crops) == 1 else 2
    cell_h = height if cols == 1 else height // 2
    scaled = []
    for crop, (x0, y0, x1, y1) in crops:
        s = cell_h / crop.shape[0]
        size = (max(1, round(crop.shape[1] * s)), cell_h)
        scaled.append(
            (
                np.asarray(Image.fromarray(crop).resize(size)),
                (x0 * s, y0 * s, x1 * s, y1 * s),
            )
        )
    cell_w = max(img.shape[1] for img, _ in scaled)
    canvas = np.full((height, cols * cell_w, 3), 255, np.uint8)
    boxes = []
    for i, (img, (x0, y0, x1, y1)) in enumerate(scaled):
        oy, ox = (i // cols) * cell_h, (i % cols) * cell_w
        canvas[oy : oy + cell_h, ox : ox + img.shape[1]] = img
        boxes.append((x0 + ox, y0 + oy, x1 + ox, y1 + oy))
    return canvas, boxes


refs = {
    view: padded_crop(np.load(work / r["file"])["rgb"], r["box"])
    for view, r in jobs["references"].items()
}
ref_sets = {
    "none": [],
    "front": [refs["front"]],
    "all": [refs[v] for v in ("front", "back", "left", "right") if v in refs],
}
results = []
for job in jobs["targets"]:
    target = np.load(work / job["file"])["rgb"]
    h, w = target.shape[:2]
    for ref_set, text in jobs["conditions"]:
        image, boxes = target, []
        if ref_sets[ref_set]:
            ref_image, ref_boxes = panel(ref_sets[ref_set], h)
            image = np.concatenate([target, ref_image], axis=1)
            cw = image.shape[1]
            boxes = [
                [(w + x0) / cw, y0 / h, (x1 - x0) / cw, (y1 - y0) / h]
                for x0, y0, x1, y1 in ref_boxes
            ]
        variant = f"{ref_set}+{text or 'notext'}"
        start = time.time()
        masks, scores = segment(predictor, image, text, boxes, threshold=0.05)
        detections = []
        for mask, score in zip(masks, scores):
            xs = np.nonzero(mask)[1]
            if len(xs) and xs.mean() < w:  # in the photo half
                detections.append(
                    {"score": round(float(score), 3), "bbox": bbox(mask[:, :w])}
                )
        if job is jobs["targets"][0]:
            Image.fromarray(image).save(
                work / "out" / f"canvas_{variant.replace(' ', '_')}.png"
            )
        results.append(
            {
                "target": job["name"],
                "variant": variant,
                "seconds": round(time.time() - start, 2),
                "detections": detections,
            }
        )
        print(json.dumps(results[-1]), flush=True)
(work / "out" / "results.json").write_text(json.dumps(results, indent=1))
