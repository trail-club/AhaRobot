"""SAM 3.1 candidates per phrase set (which words to give SAM 3.1), scored by DINOv2 large.

jobs.json: {"targets": [...], "references": {view: file}, "conditions": {name: [phrases]}}
Each phrase runs once per photo; a condition's candidates are the union of its phrases'
masks, near-duplicates (mask IoU >= 0.8) merged keeping the higher SAM score.
"""

import json

import numpy as np

from dino_common import MIN_PIXELS, embed, load_models, object_crop
from dino_common import reference_crops, timed
from sam_session import bbox, load_predictor, save_mask, segment, work_dir

work, jobs = work_dir()
predictor = load_predictor()
model = load_models(["large"])["large"]
views, refs = reference_crops(work, jobs["references"])
ref_features = embed(model, refs)["cls"]
phrases = sorted({p for ps in jobs["conditions"].values() for p in ps})


def merge(items):
    """items: [(mask, score, phrase)] -> deduplicated by mask IoU >= 0.8.

    sam31_vlm.merge_candidates uses > 0.8; this is kept as run for the recorded results.
    """
    kept = []
    for m, s, p in sorted(items, key=lambda x: -x[1]):
        if any((m & k[0]).sum() / max((m | k[0]).sum(), 1) >= 0.8 for k in kept):
            continue
        kept.append((m, s, p))
    return kept


def process(job, save=True):
    rgb = np.load(work / job["file"])["rgb"]
    per_phrase = {}
    for p in phrases:
        (masks, scores), sec = timed(lambda: segment(predictor, rgb, p))
        per_phrase[p] = (
            [(m, float(s), p) for m, s in zip(masks, scores) if m.sum() >= MIN_PIXELS],
            sec,
        )
    out = {}
    for name, ps in jobs["conditions"].items():
        cands = merge([it for p in ps for it in per_phrase[p][0]])
        sam_seconds = sum(per_phrase[p][1] for p in ps)
        crops = [object_crop(rgb, m) for m, _, _ in cands]
        sims, embed_seconds = (None, 0.0)
        if crops:
            feats, embed_seconds = timed(lambda: embed(model, crops)["cls"])
            sims = (feats @ ref_features.T).float().cpu().numpy()
        rows = []
        for i, (m, s, p) in enumerate(cands):
            rows.append(
                {
                    "number": i + 1,
                    "phrase": p,
                    "sam_score": round(s, 3),
                    "bbox": bbox(m),
                    "sims": dict(zip(views, np.round(sims[i], 4).tolist())),
                }
            )
            if save:
                save_mask(work / "out" / f"{job['name']}_{name}_{i + 1}.npy", m)
        out[name] = {
            "sam_seconds": round(sam_seconds, 3),
            "embed_seconds": round(embed_seconds, 4),
            "candidates": rows,
        }
    return {"target": job["name"], "shape": list(rgb.shape[:2]), "conditions": out}


process(jobs["targets"][0], save=False)  # warm-up
results = []
for job in jobs["targets"]:
    results.append(process(job))
    print(
        json.dumps(
            {
                "target": job["name"],
                **{
                    k: len(v["candidates"])
                    for k, v in results[-1]["conditions"].items()
                },
            }
        ),
        flush=True,
    )
(work / "out" / "results.json").write_text(
    json.dumps({"views": views, "results": results}, indent=1)
)
