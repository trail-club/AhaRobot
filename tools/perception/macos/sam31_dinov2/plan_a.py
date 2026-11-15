"""Option A on saved captures: the `interactive.py --vlm` pipeline without the camera.

VLM phrases -> SAM 3.1 candidates -> VLM selection -> size-aware same-product match.

    python plan_a.py run --set runs19|campbell27 [--user NAME]
    python plan_a.py collect --set runs19|campbell27

`run` writes the usual result directories (with flow.html) to
`.cache-sam31/sam31_dinov2/plan_a/<set>/<id>/`; `collect` summarizes them into
`results/plan_a_<set>.json`.
"""

import argparse

import common

SETS = ("runs19", "campbell27")
CAMPBELL_PROMPT = "campbell"


def jobs(name):
    """(id, prompt, capture id) of a set."""
    if name == "runs19":
        return [(k, r["prompt"], r["capture"]) for k, r in common.runs19().items()]
    return [(k, CAMPBELL_PROMPT, k) for k in common.campbell27()]


def out_dir(name):
    return common.WORK / "plan_a" / name


def run(name, user):
    import interactive  # tools/perception/macos/sam31

    args = argparse.Namespace(
        host=common.DGX_HOST,
        port=common.DGX_PORT,
        user=user,
        identity_file=None,
        image=common.IMAGE,
        checkpoint=f"{common.CHECKPOINT_DIR}/sam3.1_multiplex.pt",
        threshold=0.5,
        timeout=900,
        vlm=True,
        vlm_url=common.DEFAULT_VLM_URL,
        vlm_model=common.DEFAULT_VLM_MODEL,
        candidate_threshold=0.3,
        no_vlm_match=False,
    )
    captures = common.captures()
    worker = interactive.RemoteWorker(args)
    out_dir(name).mkdir(parents=True, exist_ok=True)
    with (out_dir(name) / "dgx.log").open("w") as log:
        try:
            worker.start(stderr=log)
            for key, prompt, capture in jobs(name):
                response = worker.predict(
                    prompt, common.ROOT / captures[capture]["path"]
                )
                if response.get("type") != "result":
                    print(f"[plan_a] {key}: {response.get('message')}", flush=True)
                    continue
                summary = interactive.unpack_result(response, out_dir(name) / key)
                interactive.write_flow_report(out_dir(name) / key, summary)
                print(
                    f"[plan_a] {key} {prompt}: keep={summary['vlm']['keep']}",
                    flush=True,
                )
        finally:
            worker.close()
    collect(name)


def summarize(result):
    """The parts of a result.json the comparison needs."""
    import interactive  # tools/perception/macos/sam31

    vlm = result["vlm"]
    # Older runs keep candidate boxes only in the SAM runs.
    boxes = interactive.candidate_boxes(vlm)
    return {
        "prompt": result["prompt"],
        "phrases": vlm["phrases"],
        "candidates": [
            {
                **{k: c[k] for k in ("number", "phrase", "score", "kept")},
                "bbox_xyxy": boxes.get(c["number"]),
            }
            for c in vlm["candidates"]
        ],
        "selected": vlm["selected"],
        "matches": [
            {
                k: m.get(k)
                for k in ("number", "reference", "same", "reference_size", "size")
            }
            for m in vlm.get("matches", [])
        ],
        "keep": vlm["keep"],
        "objects": [
            o["bbox_xyxy"]
            for o in result["objects"]
            if o["pixels"] >= common.MIN_PIXELS
        ],
        "seconds": round(result["inference_seconds"], 1),
        "vlm_seconds": round(vlm["vlm_seconds"], 1),
    }


def collect(name):
    runs = {
        key: summarize(common.read_json(out_dir(name) / key / "result.json"))
        for key, _, _ in jobs(name)
        if (out_dir(name) / key / "result.json").exists()
    }
    common.write_json(common.RESULTS / f"plan_a_{name}.json", runs)


def boxes_of(run, numbers):
    boxes = {c["number"]: c["bbox_xyxy"] for c in run["candidates"]}
    return [boxes[n] for n in numbers]


def scores(name):
    """Per-photo evaluation of selection only and of selection + match."""
    runs = common.read_json(common.RESULTS / f"plan_a_{name}.json")
    truth = (
        {k: r["target"] for k, r in common.runs19().items()}
        if name == "runs19"
        else common.campbell27()
    )
    return {
        key: {
            "select": common.evaluate(boxes_of(run, run["selected"]), truth[key]),
            "final": common.evaluate(run["objects"], truth[key]),
            "seconds": run["seconds"],
            "boxes": run["objects"],
            "truth": truth[key],
        }
        for key, run in runs.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "collect"))
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--user", default=common.DGX_USER)
    args = parser.parse_args()
    if args.command == "run":
        run(args.set, args.user)
    else:
        collect(args.set)


if __name__ == "__main__":
    main()
