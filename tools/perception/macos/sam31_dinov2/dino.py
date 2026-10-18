"""SAM 3.1 + DINOv2: find a product by comparing candidates with its reference views.

SAM 3.1 returns every candidate for a generic phrase (e.g. "can"); each candidate is cut
out with its mask and compared with the reference views (front/back/left/right) by
DINOv2 feature similarity. A candidate is kept when its best similarity reaches the
threshold. No VLM is used.

    python dino.py detect --reference NAME <session dir|rgbd.npz>...

The generic phrase is the one saved when the references were captured (chosen by the
VLM once, or given with reference.sh --phrase); --phrase overrides it.
    python dino.py run      # the 27-photo evaluation (all models and features)
"""

import argparse
import functools
import html
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

import common

MODELS = ("small", "base", "large")
FEATURES = ("cls", "patch")
FEATURE_JA = {"cls": "全体の特徴", "patch": "部分の特徴の平均"}
VIEW_JA = {"front": "正面", "back": "背面", "left": "左側面", "right": "右側面"}
# Chosen on the 27 Campbell's photos (large, global feature).
MODEL, FEATURE, THRESHOLD = "large", "cls", 0.52
EVAL_REFERENCE, EVAL_PHRASE = "campbell", "can"


def work_dir():
    return common.WORK / "dino"


def prepare(work, captures, reference, phrase, models=MODELS):
    """jobs.json, photos and reference views (with masks) for remote/dino.py."""
    work.mkdir(parents=True, exist_ok=True)
    targets = []
    for name, path in captures.items():
        np.savez_compressed(work / f"{name}.npz", rgb=np.load(path)["rgb"])
        targets.append({"name": name, "file": f"{name}.npz"})
    library = common.CACHE / "references" / reference
    references = {}
    for view_dir in sorted(p.parent for p in library.glob("*/view.json")):
        rgb = np.asarray(Image.open(view_dir / "rgb.png").convert("RGB"))
        mask = np.asarray(Image.open(view_dir / "mask.png")) > 0
        np.savez_compressed(work / f"ref_{view_dir.name}.npz", rgb=rgb, mask=mask)
        references[view_dir.name] = f"ref_{view_dir.name}.npz"
    if not references:
        raise SystemExit(f"no reference views in {library} (run sam31/reference.sh)")
    common.write_json(
        work / "jobs.json",
        {
            "targets": targets,
            "references": references,
            "phrase": phrase,
            "models": list(models),
        },
    )


def run(user):
    """The evaluation on the 27 Campbell's photos; writes results/dino.json."""
    captures = common.capture_paths(common.campbell27())
    prepare(work_dir(), captures, EVAL_REFERENCE, EVAL_PHRASE)
    common.run_in_container(work_dir(), "dino.py", user=user)
    common.write_json(
        common.RESULTS / "dino.json",
        common.read_json(work_dir() / "out" / "results.json"),
    )


def reference_phrase(reference, user, library=None):
    """The phrase saved with the references; ask the VLM once if there is none."""
    import reference as reference_tool  # tools/perception/macos/sam31
    from aha_perception.sam31_vlm import kind_phrase

    library = library or common.CACHE / "references"

    def ask():
        views = [library / reference / v for v in VIEW_JA]
        view = next((v for v in views if v.is_dir()), None)
        if view is None:
            raise SystemExit(f"no reference views for {reference} (run reference.sh)")
        rgb = np.asarray(Image.open(view / "rgb.png").convert("RGB"))
        with common.vlm_client(user) as client:
            return kind_phrase(client, rgb)

    return reference_tool.saved_or_ask(library, reference, ask)


def best_similarity(candidate, model=MODEL, feature=FEATURE):
    return max(candidate["sims"][model][feature].values())


def write_flow(page_dir, key, rgb, result, views, out, phrase, threshold, **options):
    """One photo's detection steps as `<page_dir>/<key>.html`.

    `out` holds the remote outputs (masks, crops, reference crops). Options: model,
    feature, truth (ground-truth boxes, for the evaluation), note and back (link).
    """
    model = options.get("model", MODEL)
    feature = options.get("feature", FEATURE)
    truth = options.get("truth")
    page_dir.mkdir(parents=True, exist_ok=True)
    candidates = result["candidates"]
    boxes = [c["bbox"] for c in candidates]
    common.save_jpeg(rgb, page_dir / f"{key}_input.jpg")
    common.save_jpeg(
        common.draw_boxes(
            rgb,
            boxes,
            [common.ORANGE] * len(boxes),
            [str(c["number"]) for c in candidates],
        ),
        page_dir / f"{key}_candidates.jpg",
    )
    kept = [c for c in candidates if best_similarity(c, model, feature) >= threshold]
    shape = tuple(result["shape"])
    tinted = rgb
    for c in kept:
        tinted = common.tint(
            tinted,
            common.load_mask(out / f"{key}_{c['number']}.npy", shape),
            common.GREEN,
        )
    common.save_jpeg(
        common.draw_boxes(
            tinted, [c["bbox"] for c in kept], [common.GREEN] * len(kept)
        ),
        page_dir / f"{key}_result.jpg",
    )
    for c in candidates:
        shutil.copy(
            out / f"{key}_{c['number']}_crop.png", page_dir / f"{key}_{c['number']}.png"
        )
    for v in views:
        shutil.copy(out / f"ref_{v}.png", page_dir / f"ref_{v}.png")
    rows = []
    for c in candidates:
        sims = c["sims"][model][feature]
        best_view = max(sims, key=sims.get)
        best = sims[best_view]
        cells = "".join(
            f'<td style="{"background:#e6f6ea;color:#111" if v == best_view else ""}">'
            f'<img src="ref_{v}.png" style="width:110px"><br>{VIEW_JA.get(v, v)} '
            f"{sims[v]:.3f}</td>"
            for v in views
        )
        verdict = "採用" if best >= threshold else "不採用"
        check = ""
        if truth is not None:
            is_target = common.matches_any(c["bbox"], truth)
            ok = (best >= threshold) == is_target
            verdict = f'<b class="{"ok" if ok else "ng"}">{verdict}</b>'
            check = f"<br><small>正解: {'対象' if is_target else 'ほかの物'}</small>"
        rows.append(
            f'<tr><th>{c["number"]}<br><small>SAM {c["sam_score"]:.2f}</small></th>'
            f'<td><img src="{key}_{c["number"]}.png" style="width:110px"></td>{cells}'
            f"<td><b>{best:.3f}</b><br>しきい値 {threshold}<br>{verdict}{check}</td></tr>"
        )
    seconds = result["sam_seconds"] + result["embed_seconds"][model]
    summary = f"採用 {len(kept)} 個・{seconds:.2f} 秒"
    if truth is not None:
        e = common.evaluate([c["bbox"] for c in kept], truth)
        summary = common.outcome(e, seconds, digits=2)
    back = options.get("back")
    common.write_page(
        page_dir / f"{key}.html",
        f"{key}：SAM 3.1＋DINOv2 の検出フロー",
        (f'<p><a href="{back}">← 一覧</a></p>' if back else "")
        + (f"<p>{html.escape(options['note'])}</p>" if options.get("note") else "")
        + f"<p>DINOv2 {model}・{FEATURE_JA[feature]}・しきい値 {threshold}。"
        f"SAM 3.1 {result['sam_seconds']:.2f} 秒＋見比べ "
        f"{result['embed_seconds'][model]:.3f} 秒。結果: {summary}</p>"
        f'<h2>1. 入力</h2><img src="{key}_input.jpg">'
        f"<h2>2. SAM 3.1 に「{html.escape(phrase)}」と言って候補を全部出す"
        f"（{len(boxes)} 個）</h2>"
        f'<img src="{key}_candidates.jpg">'
        "<h2>3. 候補を切り出し、参照画像と見た目の似ている度合いを測る</h2>"
        "<p>1に近いほど似ている。一番似ている向きを緑にした。最高値がしきい値以上なら採用。</p>"
        "<table><tr><th>候補</th><th>切り出し</th>"
        + "".join(f"<th>参照：{VIEW_JA.get(v, v)}</th>" for v in views)
        + "<th>判定</th></tr>"
        + ("".join(rows) or "<tr><td>候補なし</td></tr>")
        + f'</table><h2>4. 結果</h2><img src="{key}_result.jpg">',
    )
    return kept


def find_captures(paths):
    """Name -> rgbd.npz for session/frame directories or capture files."""
    captures = {}
    for path in map(Path, paths):
        files = [path] if path.is_file() else sorted(path.glob("**/capture/rgbd.npz"))
        for f in files:
            # <session>/<frame>/capture/rgbd.npz -> <session>_<frame>
            parts = f.resolve().parts
            name = f"{parts[-4]}_{parts[-3]}" if len(parts) >= 4 else f.stem
            captures[name] = f.resolve()
    if not captures:
        raise SystemExit("no captures (capture/rgbd.npz) found")
    return captures


def detect(paths, reference, phrase, threshold, user):
    """Detect the product in saved captures; writes a report with each photo's steps."""
    captures = find_captures(paths)
    work = common.WORK / "detect" / time.strftime("%Y%m%d-%H%M%S")
    prepare(work, captures, reference, phrase, models=(MODEL,))
    common.run_in_container(work, "dino.py", user=user)
    data = common.read_json(work / "out" / "results.json")
    report = work / "report"
    rows = []
    for result in data["results"]:
        key = result["target"]
        rgb = np.load(captures[key])["rgb"]
        kept = write_flow(
            report,
            key,
            rgb,
            result,
            data["views"],
            work / "out",
            phrase,
            threshold,
            back="index.html",
        )
        boxes = [c["bbox"] for c in kept]
        print(f"[dino] {key}: {len(kept)} found {boxes}", flush=True)
        rows.append(
            f'<tr><th><a href="{key}.html">{key}</a></th>'
            f'<td><img src="{key}_result.jpg"></td><td>{len(kept)} 個</td>'
            f"<td>{result['sam_seconds'] + result['embed_seconds'][MODEL]:.2f} 秒</td></tr>"
        )
    common.write_page(
        report / "index.html",
        f"「{reference}」の検出結果",
        f"<p>SAM 3.1「{html.escape(phrase)}」＋DINOv2 {MODEL}、しきい値 {threshold}。"
        "写真の名前を押すと検出フローを開く。</p>"
        "<table><tr><th>写真</th><th>結果</th><th>見つけた数</th><th>時間</th></tr>"
        + "".join(rows)
        + "</table>",
    )
    common.write_json(report / "detections.json", data)
    print(f"[dino] report: {report / 'index.html'}", flush=True)
    return report


@functools.cache
def load():
    """Views and per-photo results of the 27-photo evaluation (results/dino.json)."""
    data = common.read_json(common.RESULTS / "dino.json")
    return data["views"], {r["target"]: r for r in data["results"]}


def scored(model, feature):
    _, results = load()
    return {
        k: [(best_similarity(c, model, feature), c["bbox"]) for c in r["candidates"]]
        for k, r in results.items()
    }


def scores(model, feature):
    """Per-photo evaluation of one model/feature at its best threshold."""
    _, results = load()
    seconds = {
        k: r["sam_seconds"] + r["embed_seconds"][model] for k, r in results.items()
    }
    return common.threshold_scores(scored(model, feature), common.campbell27(), seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    find = sub.add_parser("detect", help="find a product in saved captures")
    find.add_argument("captures", nargs="+", help="session/frame directory or rgbd.npz")
    find.add_argument("--reference", required=True, help="name used with reference.sh")
    find.add_argument(
        "--phrase", help="generic English word for the item's kind (default: saved)"
    )
    find.add_argument("--threshold", type=float, default=THRESHOLD)
    sub.add_parser("run", help="the 27-photo evaluation")
    for p in (find, sub.choices["run"]):
        p.add_argument("--user", default=common.DGX_USER, help="DGX user name")
    args = parser.parse_args()
    if args.command == "detect":
        phrase = args.phrase or reference_phrase(args.reference, args.user)
        print(f"[dino] SAM 3.1 の言葉: {phrase}", flush=True)
        detect(args.captures, args.reference, phrase, args.threshold, args.user)
    else:
        run(args.user)


if __name__ == "__main__":
    main()
