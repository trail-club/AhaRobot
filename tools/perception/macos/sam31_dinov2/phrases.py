"""Which words to give SAM 3.1 so that the user enters only the item's name.

The same 27 photos and DINOv2 large comparison as `dino.py`, with five phrase choices:
the kind chosen by a person ("can"), the proper noun itself, "object", the kind chosen
once by the VLM from a reference view, and seven common kind words together.

    python phrases.py run [--user NAME]
"""

import argparse
import functools
import html
import time

import numpy as np
from PIL import Image

import common
import dino

PROPER_NOUN = "campbell's tomato soup"
COMMON_KINDS = ["can", "bottle", "box", "cup", "jar", "bag", "container"]


def conditions(kind):
    """Condition -> (label, phrases given to SAM 3.1); `kind` is the VLM's phrase."""
    return {
        "can": (f"① 種類を人が決める（{dino.EVAL_PHRASE}）", [dino.EVAL_PHRASE]),
        "proper": (f"② 固有名詞をそのまま（{PROPER_NOUN}）", [PROPER_NOUN]),
        "object": ("③ object に固定", ["object"]),
        "auto": (f"④ 種類を Qwen が決める（{kind}）", [kind]),
        "multi": ("⑤ よくある種類7語をまとめて", COMMON_KINDS),
    }


def run(user):
    from aha_perception.sam31_vlm import kind_phrase

    view = common.CACHE / "references" / dino.EVAL_REFERENCE / "front" / "rgb.png"
    client = common.vlm_client()
    start = time.time()
    kind = kind_phrase(client, np.asarray(Image.open(view).convert("RGB")))
    seconds = round(time.time() - start, 1)
    work = common.WORK / "phrases"
    captures = common.capture_paths(common.campbell27())
    dino.prepare(
        work, captures, dino.EVAL_REFERENCE, dino.EVAL_PHRASE, models=(dino.MODEL,)
    )
    jobs = common.read_json(work / "jobs.json")
    jobs["conditions"] = {k: v[1] for k, v in conditions(kind).items()}
    common.write_json(work / "jobs.json", jobs)
    common.run_in_container(work, "dino_phrases.py", user=user)
    data = common.read_json(work / "out" / "results.json")
    data["kind"] = {"phrase": kind, "seconds": seconds, "view": "front"}
    common.write_json(common.RESULTS / "dino_phrases.json", data)


@functools.cache
def load():
    """The VLM's kind phrase and photo -> results (results/dino_phrases.json)."""
    data = common.read_json(common.RESULTS / "dino_phrases.json")
    return data["kind"], {r["target"]: r for r in data["results"]}


def best(candidate):
    return max(candidate["sims"].values())


@functools.cache
def table():
    """Per condition: candidate recall, accuracy (best and fixed threshold), time."""
    kind, results = load()
    truth = common.campbell27()
    targets = sum(len(v) for v in truth.values())
    rows = {}
    for name in conditions(kind["phrase"]):
        per_photo = {k: r["conditions"][name] for k, r in results.items()}
        items = {
            k: [(best(c), c["bbox"]) for c in v["candidates"]]
            for k, v in per_photo.items()
        }
        seconds = {
            k: v["sam_seconds"] + v["embed_seconds"] for k, v in per_photo.items()
        }
        at_best, threshold = common.threshold_scores(items, truth, seconds)
        at_fixed, _ = common.threshold_scores(items, truth, seconds, dino.THRESHOLD)
        counts = [len(v["candidates"]) for v in per_photo.values()]
        rows[name] = {
            "threshold": threshold,
            "best": common.summarize_method(at_best),
            "fixed": common.summarize_method(at_fixed),
            "in_candidates": sum(
                common.matches_any(g, [b for _, b in items[k]])
                for k in truth
                for g in truth[k]
            ),
            "targets": targets,
            "candidates": round(float(np.mean(counts)), 1),
        }
    return rows


def table_html():
    kind, _ = load()
    label = conditions(kind["phrase"])
    body = "".join(
        f"<tr><th>{html.escape(label[n][0])}</th>"
        f"<td>{r['in_candidates']} / {r['targets']}</td><td>{r['candidates']}</td>"
        f"<td><b>{r['best']['correct']} / 27</b>（{r['threshold']}）</td>"
        f"<td>{r['best']['found']} / {r['best']['wrong']} / {r['best']['missed']}</td>"
        f"<td>{r['fixed']['correct']} / 27</td><td>{r['best']['mean_seconds']:.2f} 秒</td></tr>"
        for n, r in table().items()
    )
    return (
        "<table><tr><th>SAM 3.1 に渡す言葉</th><th>キャンベル缶が候補に出た数</th>"
        "<th>候補の数（平均）</th><th>正解した写真（しきい値）</th><th>見つけた / 誤り / 見逃し</th>"
        f"<th>しきい値 {dino.THRESHOLD} のまま</th><th>1枚の時間</th></tr>{body}</table>"
    )


def condition_section(view, masks, key, rgb, truth, name, label, cond, threshold):
    """Images and candidate table of one condition; returns (cell, section) HTML."""
    cands = cond["candidates"]
    boxes = [c["bbox"] for c in cands]
    hits = [common.matches_any(b, truth) for b in boxes]
    kept = [c for c in cands if best(c) >= threshold]
    kept_boxes = [c["bbox"] for c in kept]
    common.save_jpeg(
        common.draw_boxes(
            rgb,
            boxes,
            [common.ORANGE] * len(boxes),
            [str(c["number"]) for c in cands],
            width=4,
        ),
        view / f"{key}_{name}_cands.jpg",
        420,
    )
    tinted = rgb
    for c in kept:
        mask = common.load_mask(
            masks / f"{key}_{name}_{c['number']}.npy", rgb.shape[:2]
        )
        color = common.GREEN if common.matches_any(c["bbox"], truth) else common.RED
        tinted = common.tint(tinted, mask, color)
    common.save_jpeg(
        common.result_image(
            tinted, kept_boxes, truth, [f"{best(c):.2f}" for c in kept]
        ),
        view / f"{key}_{name}_result.jpg",
        420,
    )
    e = common.evaluate(kept_boxes, truth)
    seconds = cond["sam_seconds"] + cond["embed_seconds"]
    listed = all(common.matches_any(g, boxes) for g in truth)
    note = "" if listed else "・<b class=ng>候補に出ず</b>"
    cell = (
        f'<td><img src="phrases/{key}_{name}_result.jpg"><br>{common.mark(e["correct"])}'
        f" 候補{len(cands)}個{note}・{seconds:.2f}秒</td>"
    )
    rows = "".join(
        f'<tr><th>{c["number"]}</th><td>{html.escape(c["phrase"])}</td>'
        f'<td>{c["sam_score"]:.2f}</td><td><b>{best(c):.3f}</b>'
        f'（{dino.VIEW_JA.get(max(c["sims"], key=c["sims"].get), "")}）</td>'
        f'<td>{"採用" if best(c) >= threshold else "不採用"}</td>'
        f'<td>{"キャンベル缶" if hit else "ほかの物"}</td></tr>'
        for c, hit in sorted(zip(cands, hits), key=lambda x: -best(x[0]))
    )
    section = (
        f"<h2>{html.escape(label)}</h2><p>SAM 3.1 {cond['sam_seconds']:.2f}秒＋"
        f"見比べ {cond['embed_seconds']:.3f}秒、しきい値 {threshold}。"
        f"{common.outcome(e)}{note}</p>"
        f'<p><img src="{key}_{name}_cands.jpg" style="width:420px"> '
        f'<img src="{key}_{name}_result.jpg" style="width:420px"></p>'
        "<table><tr><th>候補</th><th>どの言葉で出たか</th><th>SAM の点数</th>"
        "<th>参照との似ている度合い（最も似た向き）</th><th>判定</th><th>正解</th></tr>"
        + (rows or "<tr><td colspan=6>候補なし</td></tr>")
        + "</table>"
    )
    return cell, section


def report(out):
    """Per-photo detection steps for every phrase choice (out/phrases/)."""
    kind, results = load()
    truth = common.campbell27()
    rows = table()
    label = {k: v[0] for k, v in conditions(kind["phrase"]).items()}
    view = out / "phrases"
    masks = common.WORK / "phrases" / "out"
    lines = []
    for key in truth:
        rgb, _, _ = common.load_capture(key)
        parts = [
            condition_section(
                view,
                masks,
                key,
                rgb,
                truth[key],
                name,
                label[name],
                results[key]["conditions"][name],
                rows[name]["threshold"],
            )
            for name in label
        ]
        common.write_page(
            view / f"{key}.html",
            f"{key}：言葉ごとの検出フロー",
            f'<p><a href="../phrases.html">← 一覧</a>　'
            f"{html.escape(common.captures()[key]['scene'])}</p>"
            + "".join(section for _, section in parts),
        )
        lines.append(
            f'<tr><th><a href="phrases/{key}.html">{key}</a></th>'
            + "".join(cell for cell, _ in parts)
            + "</tr>"
        )
    common.write_page(
        out / "phrases.html",
        "固有名詞1つで使うには：SAM 3.1 に渡す言葉の比較",
        f"<p>④の言葉は参照画像（正面）を Qwen に1回見せて決めた（{kind['seconds']}秒）。"
        "しきい値は条件ごとに27枚で一番良い値。緑＝キャンベル缶、赤＝ほかの物、白＝見逃し。</p>"
        + table_html()
        + "<table><tr><th>写真</th>"
        + "".join(f"<th>{html.escape(v)}</th>" for v in label.values())
        + f"</tr>{''.join(lines)}</table>",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--user", default=common.DGX_USER)
    run(parser.parse_args().user)


if __name__ == "__main__":
    main()
