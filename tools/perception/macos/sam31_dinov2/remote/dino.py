"""SAM 3.1 finds every candidate for a generic phrase; DINOv2 compares each with references.

jobs.json: {"targets": [{"name", "file"}], "references": {view: file with rgb+mask},
            "phrase": "can", "models": ["large"] (optional, default all)}
Writes out/results.json (similarities, timings), <target>_<n>.npy masks and crops.
"""

import json

import numpy as np
from PIL import Image

from dino_common import MIN_PIXELS, MODELS, embed, load_models, object_crop
from dino_common import reference_crops, timed
from sam_session import bbox, load_predictor, save_mask, segment, work_dir

work, jobs = work_dir()
predictor = load_predictor()
models = load_models(k for k in MODELS if k in jobs.get("models", MODELS))
views, ref_crops = reference_crops(work, jobs["references"])
ref_features = {k: embed(m, ref_crops) for k, m in models.items()}


def process(job, save=True):
    rgb = np.load(work / job["file"])["rgb"]
    (masks, scores), sam_seconds = timed(
        lambda: segment(predictor, rgb, jobs["phrase"])
    )
    keep = [i for i, m in enumerate(masks) if m.sum() >= MIN_PIXELS]
    crops = [object_crop(rgb, masks[i]) for i in keep]
    sims, embed_seconds = {}, {}
    for k, model in models.items():
        features, embed_seconds[k] = timed(
            lambda: embed(model, crops) if crops else None
        )
        if features is not None:
            sims[k] = {
                f: (features[f] @ ref_features[k][f].T).float().cpu().numpy()
                for f in ("cls", "patch")
            }
    candidates = []
    for n, i in enumerate(keep):
        candidates.append(
            {
                "number": n + 1,
                "sam_score": round(float(scores[i]), 3),
                "bbox": bbox(masks[i]),
                "sims": {
                    k: {f: dict(zip(views, np.round(s[f][n], 4).tolist())) for f in s}
                    for k, s in sims.items()
                },
            }
        )
        if save:
            save_mask(work / "out" / f"{job['name']}_{n + 1}.npy", masks[i])
            Image.fromarray(crops[n]).save(
                work / "out" / f"{job['name']}_{n + 1}_crop.png"
            )
    return {
        "target": job["name"],
        "shape": list(rgb.shape[:2]),
        "sam_seconds": round(sam_seconds, 3),
        "embed_seconds": {k: round(v, 4) for k, v in embed_seconds.items()},
        "candidates": candidates,
    }


process(jobs["targets"][0], save=False)  # warm-up: CUDA kernels and caches
results = []
for job in jobs["targets"]:
    results.append(process(job))
    print(
        json.dumps({k: results[-1][k] for k in ("target", "sam_seconds")}), flush=True
    )
(work / "out" / "results.json").write_text(
    json.dumps({"views": views, "results": results}, indent=1)
)
