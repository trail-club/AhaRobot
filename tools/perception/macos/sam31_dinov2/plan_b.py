"""Option B on the 19 runs: Qwen draws the target boxes, SAM 3.1 segments them.

B1 (boxes only): Qwen boxes -> SAM 3.1 box prompt -> keep masks inside a Qwen box.
B2 (+ match): one Qwen call returns boxes and a shape phrase -> SAM 3.1 with box + phrase
-> masks inside a box are kept, the other masks (look-alikes) go to option A's
size-aware same-product match. When the phrase just repeats the product name or is not
English, fallback (a) uses "container" and (b) re-asks Qwen with only the boxed crop.

    python plan_b.py qwen --step boxes|phrase
    python plan_b.py reask
    python plan_b.py sam --step boxes|phrase
    python plan_b.py match --variant a|b
"""

import argparse
import json
import re
import time

import numpy as np
from PIL import Image

import common

BOXES_PROMPT = """Find every instance of "{target}" in the image. It may be a product or brand name in any language; read packaging text and logos. The same product may appear more than once, and an instance may be turned so that its logo is not visible (back or side) — include it if it is clearly the same product. Do not include different products.
Give each instance's bounding box as [x1, y1, x2, y2] in coordinates normalized to 0-1000.
Reply with JSON only: {{"objects": [{{"bbox_2d": [x1, y1, x2, y2]}}]}}"""

PHRASE_PROMPT = """Find every instance of "{target}" in the image. It may be a product or brand name in any language; read packaging text and logos. Do not include different products.
Also give one short generic English noun phrase (1-3 words) for the target's container shape and material, not its contents or brand, so a segmentation model also finds it from the back or side (examples: "cylindrical can", "plastic bottle", "paper cup", "cardboard box"; for unpackaged items name the item, e.g. "banana").
Give each instance's bounding box as [x1, y1, x2, y2] in coordinates normalized to 0-1000.
Reply with JSON only: {{"phrase": "...", "objects": [{{"bbox_2d": [x1, y1, x2, y2]}}]}}"""

REASK_PROMPT = """What kind of object is this, by its shape and material? Answer with a short generic English noun phrase of 1-3 words, without any brand or product name and without its contents (examples: "cylindrical can", "plastic bottle", "paper cup", "cardboard box"; for unpackaged items name the item, e.g. "banana").
Reply with JSON only: {"phrase": "..."}"""

FALLBACK_PHRASE = "container"


def result_path(name):
    return common.RESULTS / f"plan_b_{name}.json"


def work_dir(step):
    return common.WORK / "plan_b" / step


def parse_boxes(answer, width, height):
    """Qwen answers in its native list format or as the requested object."""
    data = parse_json(answer)
    objects = data if isinstance(data, list) else data.get("objects", [])
    boxes = []
    for o in objects:
        x0, y0, x1, y1 = o["bbox_2d"]
        boxes.append(
            [round(x0 * width / 1000), round(y0 * height / 1000)]
            + [round(x1 * width / 1000), round(y1 * height / 1000)]
        )
    return boxes


def parse_json(text):
    """The first JSON object or list in a reply (models may add prose around it)."""
    start = min(i for i in (text.find("{"), text.find("[")) if i >= 0)
    end = max(text.rfind("}"), text.rfind("]")) + 1
    return json.loads(text[start:end])


def english(phrase):
    return bool(re.fullmatch(r"[a-z][a-z0-9 '&.-]*", " ".join(phrase.lower().split())))


def valid_phrase(phrase, prompt):
    """An English generic phrase that does not repeat the product name."""
    return english(phrase) and prompt.lower() not in phrase.lower()


def ask_qwen(step, user):
    template = BOXES_PROMPT if step == "boxes" else PHRASE_PROMPT
    out = {}
    client = common.vlm_client()
    for key, run in common.runs19().items():
        rgb, _, _ = common.load_capture(run["capture"])
        start = time.time()
        answer = client.ask(template.format(target=run["prompt"]), rgb)
        h, w = rgb.shape[:2]
        out[key] = {
            "answer": answer,
            "boxes": parse_boxes(answer, w, h),
            "seconds": round(time.time() - start, 1),
        }
        if step == "phrase":
            out[key]["phrase"] = str(parse_json(answer).get("phrase", ""))
        print(f"[plan_b] {key} {run['prompt']}: {answer.strip()[:120]}", flush=True)
    common.write_json(result_path(f"qwen_{step}"), out)


def reask(user):
    """Fallback (b): a shape phrase from the boxed crop alone, without the name."""
    qwen = common.read_json(result_path("qwen_phrase"))
    out = {}
    client = common.vlm_client()
    for key, run in common.runs19().items():
        if valid_phrase(qwen[key]["phrase"], run["prompt"]) or not qwen[key]["boxes"]:
            continue
        rgb, _, _ = common.load_capture(run["capture"])
        x0, y0, x1, y1 = qwen[key]["boxes"][0]
        start = time.time()
        answer = client.ask(REASK_PROMPT, np.ascontiguousarray(rgb[y0:y1, x0:x1]))
        phrase = " ".join(str(parse_json(answer).get("phrase", "")).lower().split())
        out[key] = {
            "phrase": phrase,
            "ok": valid_phrase(phrase, run["prompt"]),
            "seconds": round(time.time() - start, 1),
        }
        print(f"[plan_b] {key} {run['prompt']} -> {phrase}", flush=True)
    common.write_json(result_path("reask"), out)


def sam_phrases(key, run, qwen, again):
    """B2 SAM text per fallback variant (qwen/again: the phrase and reask results)."""
    phrase = qwen[key]["phrase"]
    if valid_phrase(phrase, run["prompt"]):
        phrase = " ".join(phrase.lower().split())
        return {"a": phrase, "b": phrase}
    # The crop answer is used even when it equals the name: for unpackaged items
    # ("banana") the name is also the generic word.
    again = again.get(key, {}).get("phrase", "")
    return {"a": FALLBACK_PHRASE, "b": again if english(again) else None}


def sam(step, user):
    work = work_dir(step)
    (work / "img").mkdir(parents=True, exist_ok=True)
    qwen = common.read_json(result_path(f"qwen_{step}"))
    again = common.read_json(result_path("reask")) if step == "phrase" else {}
    jobs = []
    for key, run in common.runs19().items():
        rgb, _, _ = common.load_capture(run["capture"])
        Image.fromarray(rgb).save(work / "img" / f"{key}.png")
        variants = (
            {"box": None} if step == "boxes" else sam_phrases(key, run, qwen, again)
        )
        jobs.append(
            {
                "key": key,
                "file": f"img/{key}.png",
                "boxes": qwen[key]["boxes"],
                "variants": variants,
            }
        )
    common.write_json(work / "jobs.json", jobs)
    common.run_in_container(work, "sam_boxes.py", user=user)
    common.write_json(
        result_path(f"sam_{step}"), common.read_json(work / "out" / "results.json")
    )


def load_mask(step, key, variant, number, shape):
    path = work_dir(step) / "out" / f"{key}_{variant}_{number}.npy"
    return common.load_mask(path, shape)


def sam_results(step, variant):
    """Key -> SAM run of one step ("boxes"/"phrase") and variant."""
    return {
        r["key"]: r
        for r in common.read_json(result_path(f"sam_{step}"))
        if r["variant"] == variant
    }


def match(variant, user):
    """Masks inside a Qwen box are kept; the others are checked against the top one."""
    from aha_perception import sam31_vlm

    qwen = common.read_json(result_path("qwen_phrase"))
    results = sam_results("phrase", variant)
    out = {}
    client = common.vlm_client()
    for key, run in common.runs19().items():
        rgb, depth, intrinsics = common.load_capture(run["capture"])
        dets = results[key]["detections"]
        masks = [
            load_mask("phrase", key, variant, n, depth.shape)
            for n in range(1, len(dets) + 1)
        ]
        kept = [
            i
            for i, d in enumerate(dets)
            if common.matches_any(d["bbox"], qwen[key]["boxes"])
        ]
        checks = []
        if kept:
            ref = max(kept, key=lambda i: dets[i]["score"])
            ref_size = sam31_vlm.physical_size(masks[ref], depth, intrinsics)
            for i in (i for i in range(len(dets)) if i not in kept):
                size = sam31_vlm.physical_size(masks[i], depth, intrinsics)
                depths = tuple(s and s["depth_m"] for s in (ref_size, size))
                pair = sam31_vlm.pair_image(rgb, masks[ref], masks[i], depths)
                start = time.time()
                same = sam31_vlm.match_candidate(
                    client, pair, run["prompt"], (ref_size, size)
                )
                Image.fromarray(pair).save(
                    work_dir("phrase") / "out" / f"{key}_{variant}_pair_{i + 1}.png"
                )
                checks.append(
                    {
                        "index": i,
                        "ref": ref,
                        "same": same,
                        "ref_size": ref_size,
                        "size": size,
                        "seconds": round(time.time() - start, 1),
                    }
                )
        final = kept + [c["index"] for c in checks if c["same"]]
        out[key] = {"kept": kept, "matches": checks, "final": final}
        print(f"[plan_b] {key}: kept {kept} final {final}", flush=True)
    common.write_json(result_path(f"match_{variant}"), out)


def scores(method):
    """Per-run evaluation of B1 ("boxes") or B2 with fallback a/b ("match_a"/"match_b")."""
    runs = common.runs19()
    out = {}
    if method == "boxes":
        qwen = common.read_json(result_path("qwen_boxes"))
        sam_runs = sam_results("boxes", "box")
        for key, run in runs.items():
            boxes = [
                d["bbox"]
                for d in sam_runs[key]["detections"]
                if common.matches_any(d["bbox"], qwen[key]["boxes"])
            ]
            seconds = qwen[key]["seconds"] + sam_runs[key]["seconds"]
            out[key] = {"boxes": boxes, "seconds": seconds}
    else:
        variant = method[-1]
        qwen = common.read_json(result_path("qwen_phrase"))
        again = common.read_json(result_path("reask"))
        matched = common.read_json(result_path(method))
        sam_runs = sam_results("phrase", variant)
        for key, run in runs.items():
            dets = sam_runs[key]["detections"]
            boxes = [dets[i]["bbox"] for i in matched[key]["final"]]
            seconds = (
                qwen[key]["seconds"]
                + sam_runs[key]["seconds"]
                + sum(c["seconds"] for c in matched[key]["matches"])
                + (again.get(key, {}).get("seconds", 0) if variant == "b" else 0)
            )
            out[key] = {"boxes": boxes, "seconds": round(seconds, 1)}
    for key, run in runs.items():
        out[key]["final"] = common.evaluate(out[key]["boxes"], run["target"])
        out[key]["truth"] = run["target"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qwen", "reask", "sam", "match"))
    parser.add_argument("--step", choices=("boxes", "phrase"), default="phrase")
    parser.add_argument("--variant", choices=("a", "b"), default="b")
    parser.add_argument("--user", default=common.DGX_USER)
    args = parser.parse_args()
    if args.command == "qwen":
        ask_qwen(args.step, args.user)
    elif args.command == "reask":
        reask(args.user)
    elif args.command == "sam":
        sam(args.step, args.user)
    else:
        match(args.variant, args.user)


if __name__ == "__main__":
    main()
