"""Reference collage (SAM 3.1 only) on the 27 Campbell's photos.

Each photo gets the reference views stitched to its right with exemplar boxes on them,
combined with no text or a text prompt.

    python collage.py run --refs clean|stand-in

`clean` uses the views captured with `sam31/reference.sh --name campbell`; `stand-in`
crops the can out of two of the photos (front 1011-12, back 1011-07) as a first check.
"""

import argparse
import functools

import numpy as np
from PIL import Image

import common

TEXTS = (None, "can", "soup can", "campbell's tomato soup")
CONDITIONS = [
    (refs, text)
    for refs in ("none", "front", "all")
    for text in TEXTS
    if not (refs == "none" and text is None)
]
REFERENCE_NAME = "campbell"
STAND_IN = {"front": "1011-12", "back": "1011-07"}


def variant_name(refs, text):
    return f"{refs}+{text or 'notext'}"


def prepare(refs):
    work = common.WORK / "collage" / refs
    work.mkdir(parents=True, exist_ok=True)
    targets = []
    for capture in common.campbell27():
        rgb, _, _ = common.load_capture(capture)
        np.savez_compressed(work / f"{capture}.npz", rgb=rgb)
        targets.append({"name": capture, "file": f"{capture}.npz"})
    references = {}
    if refs == "clean":
        library = common.CACHE / "references" / REFERENCE_NAME
        for view_json in sorted(library.glob("*/view.json")):
            view = view_json.parent.name
            rgb = np.asarray(Image.open(view_json.parent / "rgb.png").convert("RGB"))
            np.savez_compressed(work / f"ref_{view}.npz", rgb=rgb)
            box = common.read_json(view_json)["bbox_xyxy"]
            references[view] = {"file": f"ref_{view}.npz", "box": box}
    else:
        for view, capture in STAND_IN.items():
            np.savez_compressed(
                work / f"ref_{view}.npz", rgb=common.load_capture(capture)[0]
            )
            box = common.campbell27()[capture][0]
            references[view] = {"file": f"ref_{view}.npz", "box": box}
    common.write_json(
        work / "jobs.json",
        {"targets": targets, "references": references, "conditions": CONDITIONS},
    )
    return work


def run(refs, user):
    work = prepare(refs)
    common.run_in_container(work, "collage.py", user=user)
    common.write_json(
        common.RESULTS / f"collage_{refs}.json",
        common.read_json(work / "out" / "results.json"),
    )


@functools.cache
def load(refs):
    """Condition -> (photo -> [(score, box)], photo -> seconds)."""
    out = {}
    for r in common.read_json(common.RESULTS / f"collage_{refs}.json"):
        items, times = out.setdefault(r["variant"], ({}, {}))
        items[r["target"]] = [(d["score"], d["bbox"]) for d in r["detections"]]
        times[r["target"]] = r["seconds"]
    return out


def scored(refs, variant):
    """Photo -> [(score, box)] of one condition."""
    return load(refs)[variant][0]


def top_ranked(items, truth):
    """The target outscores every other detection (photos with a target only)."""
    best_target = max((s for s, b in items if common.matches_any(b, truth)), default=0)
    best_other = max(
        (s for s, b in items if not common.matches_any(b, truth)), default=0
    )
    return best_target > best_other


def conditions_table(refs):
    """Per condition: best threshold, photos correct at it, top-ranked photos."""
    truth = common.campbell27()
    rows = []
    for refs_set, text in CONDITIONS:
        variant = variant_name(refs_set, text)
        items, times = load(refs)[variant]
        per_photo, threshold = common.threshold_scores(items, truth, times)
        with_target = [k for k in items if truth[k]]
        rows.append(
            {
                "variant": variant,
                "refs": refs_set,
                "text": text,
                "threshold": threshold,
                "photos": len(items),
                "correct": sum(v["final"]["correct"] for v in per_photo.values()),
                "top": sum(top_ranked(items[k], truth[k]) for k in with_target),
                "with_target": len(with_target),
                "separation": common.separation(items, truth),
                "mean_seconds": round(float(np.mean(list(times.values()))), 2),
            }
        )
    return rows


def scores(refs, variant):
    """Per-photo evaluation of one condition at its best threshold."""
    items, times = load(refs)[variant]
    return common.threshold_scores(items, common.campbell27(), times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--refs", choices=("clean", "stand-in"), default="clean")
    parser.add_argument("--user", default=common.DGX_USER)
    args = parser.parse_args()
    run(args.refs, args.user)


if __name__ == "__main__":
    main()
