"""Build the comparison pages from `results/`.

    python report.py            # full reports in .cache-sam31/sam31_dinov2/reports/
    python report.py --summary  # also refresh index.html and img/ (tracked)

The full reports draw every photo and need the saved captures; the tracked summary keeps
only a few representative images.
"""

import argparse
import html
import os

from PIL import Image

import collage
import common
import dino
import phrases
import plan_a
import plan_b

REPORTS = common.WORK / "reports"
IMG = common.HERE / "img"
COLLAGE_BEST = "front+soup can"
DINO_BEST = (dino.MODEL, dino.FEATURE)
TILE = 224  # size of the DINOv2 crops


# --- shared pieces --------------------------------------------------------------------


def slug(variant):
    """File-name form of a condition name."""
    return variant.replace(" ", "_").replace("'", "")


def scored_image(rgb, items, truth):
    """Result image of [(score, box)] with every score as a label."""
    return common.result_image(
        rgb, [b for _, b in items], truth, [f"{s:.2f}" for s, _ in items]
    )


def tile_row(paths):
    """Square crops side by side."""
    row = Image.new("RGB", (TILE * len(paths), TILE))
    for i, path in enumerate(paths):
        row.paste(Image.open(path), (TILE * i, 0))
    return row


def summary_table(methods, with_threshold=False):
    """Totals per method; `methods` maps a label to per-photo results (or with threshold)."""
    rows = []
    for label, method in methods.items():
        method, threshold = method if with_threshold else (method, None)
        s = common.summarize_method(method)
        rows.append(
            f"<tr><th>{html.escape(label)}</th><td><b>{s['correct']} / {s['photos']}</b>"
            f"</td><td>{s['found']} / {s['wrong']} / {s['missed']}</td>"
            f"<td>{s['mean_seconds']:.2f} 秒</td><td>{s['max_seconds']:.2f} 秒</td>"
            + (f"<td>{threshold}</td>" if with_threshold else "")
            + "</tr>"
        )
    return (
        "<table><tr><th>方式</th><th>正解した写真</th><th>見つけた / 誤り / 見逃し</th>"
        "<th>1枚の時間（平均）</th><th>最長</th>"
        + ("<th>しきい値</th>" if with_threshold else "")
        + f"</tr>{''.join(rows)}</table>"
    )


def result_cell(src, v):
    return f'<td><img src="{src}"><br>{common.outcome(v["final"], v["seconds"])}</td>'


# --- runs19: options A and B ----------------------------------------------------------


def runs19_methods():
    a = plan_a.scores("runs19")
    return {
        "案A 選択のみ": {k: {**v, "final": v["select"]} for k, v in a.items()},
        "案A 選択＋照合": a,
        "案B1 Qwenの枠だけ": plan_b.scores("boxes"),
        "案B2 枠＋言葉＋照合（a: container）": plan_b.scores("match_a"),
        "案B2 枠＋言葉＋照合（b: 聞き直し）": plan_b.scores("match_b"),
    }


def plan_b_image(key, variant):
    """B2 overlay: red = Qwen boxes, green = kept, blue = added by match, grey = rejected."""
    rgb, depth, _ = common.load_capture(common.runs19()[key]["capture"])
    qwen = common.read_json(plan_b.result_path("qwen_phrase"))[key]
    dets = plan_b.sam_results("phrase", variant)[key]["detections"]
    matched = common.read_json(plan_b.result_path(f"match_{variant}"))[key]
    added = {c["index"] for c in matched["matches"] if c["same"]}
    colors = []
    for i in range(len(dets)):
        if i in matched["kept"] or i in added:
            color = common.GREEN if i in matched["kept"] else common.BLUE
            mask = plan_b.load_mask("phrase", key, variant, i + 1, depth.shape)
            rgb = common.tint(rgb, mask, color)
        else:
            color = common.GREY
        colors.append(color)
    boxes = [d["bbox"] for d in dets] + qwen["boxes"]
    colors += [common.RED] * len(qwen["boxes"])
    labels = [str(i + 1) for i in range(len(dets))] + [""] * len(qwen["boxes"])
    return common.draw_boxes(rgb, boxes, colors, labels, width=5)


def write_runs19(out, methods):
    view = out / "runs19"
    lines = []
    for key, run in common.runs19().items():
        rgb, _, _ = common.load_capture(run["capture"])
        cells = []
        for i, (label, m) in enumerate(methods.items()):
            v = m[key]
            if label.startswith("案B2"):
                image = plan_b_image(key, label[label.index("（") + 1])
            else:
                image = common.result_image(rgb, v["boxes"], v["truth"])
            common.save_jpeg(image, view / f"{key}_{i}.jpg", width=360)
            cells.append(result_cell(f"runs19/{key}_{i}.jpg", v))
        flow = common.WORK / "plan_a" / "runs19" / key / "flow.html"
        lines.append(
            f"<tr><th>{key}<br>「{html.escape(run['prompt'])}」<br><small>{run['capture']}"
            f'</small><br><a href="{os.path.relpath(flow, out)}">案Aの流れ</a></th>'
            f"{''.join(cells)}</tr>"
        )
    common.write_page(
        out / "runs19.html",
        "案A・案B：19回の実行（プリングルズなど）",
        "<p>緑＝対象を見つけた、赤＝別の物（誤り）、白＝見逃し。案B2の図は、赤の太枠＝Qwenの枠、"
        "緑＝枠の中のマスク（採用）、青＝照合で同じ商品（採用）、灰＝除外。</p>"
        + summary_table(methods)
        + f"<table><tr><th>実行</th>{''.join(f'<th>{html.escape(k)}</th>' for k in methods)}"
        f"</tr>{''.join(lines)}</table>",
    )


# --- campbell27: A, collage and DINOv2 ------------------------------------------------


def campbell_methods():
    """Label -> (per-photo results, threshold) on the 27 Campbell's photos."""
    methods = {
        "案A（SAM 3.1＋Qwen）": (plan_a.scores("campbell27"), "-"),
        f"参照を並べる（{COLLAGE_BEST}）": collage.scores("clean", COLLAGE_BEST),
    }
    for model in dino.MODELS:
        methods[f"SAM 3.1＋DINOv2 {model}"] = dino.scores(model, "cls")
    return methods


def write_collage(out):
    truth = common.campbell27()
    parts = []
    for refs, title in (
        ("clean", "撮り直した参照画像"),
        ("stand-in", "写真から切り出した仮の参照"),
    ):
        rows = "".join(
            f"<tr><th>{html.escape(r['variant'])}</th><td>{r['top']} / {r['with_target']}</td>"
            f"<td>{r['correct']} / {r['photos']}</td><td>{r['threshold']}</td>"
            f"<td>{r['separation'][1]:.2f} / {r['separation'][2]:.2f}</td>"
            f"<td>{r['mean_seconds']:.2f} 秒</td></tr>"
            for r in collage.conditions_table(refs)
        )
        parts.append(
            f"<h2>{title}</h2><table><tr><th>条件（参照＋言葉）</th>"
            "<th>キャンベル缶が1番の点数</th><th>正解した写真（最良のしきい値）</th>"
            "<th>しきい値</th><th>キャンベル缶の最低点 / ほかの最高点</th><th>時間</th></tr>"
            f"{rows}</table>"
        )
    variants = [collage.variant_name(*c) for c in collage.CONDITIONS]
    view = out / "collage"
    lines = []
    for key in truth:
        rgb, _, _ = common.load_capture(key)
        cells = []
        for variant in variants:
            image = scored_image(rgb, collage.scored("clean", variant)[key], truth[key])
            name = f"{key}_{slug(variant)}.jpg"
            common.save_jpeg(image, view / name, width=320)
            cells.append(f'<td><img src="collage/{name}"></td>')
        lines.append(f"<tr><th>{key}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th>{html.escape(v)}</th>" for v in variants)
    common.write_page(
        out / "collage.html",
        "参照を並べる方式（SAM 3.1だけ）",
        "<p>撮影画像の右に参照画像を並べ、参照に枠を付けて SAM 3.1 に渡した。"
        "数字は SAM 3.1 の点数（0.05以上をすべて表示）。緑＝キャンベル缶、赤＝ほかの物。</p>"
        + "".join(parts)
        + f"<h2>写真ごと（撮り直した参照）</h2><table><tr><th>写真</th>{head}</tr>"
        + "".join(lines)
        + "</table>",
    )


def dino_settings_table():
    """Every DINOv2 model and feature at its best threshold."""
    truth = common.campbell27()
    rows = []
    for model in dino.MODELS:
        for feature in dino.FEATURES:
            per_photo, threshold = dino.scores(model, feature)
            s = common.summarize_method(per_photo)
            _, lo, hi = common.separation(dino.scored(model, feature), truth)
            rows.append(
                f"<tr><th>DINOv2 {model}・{dino.FEATURE_JA[feature]}</th>"
                f"<td>{s['correct']} / 27</td>"
                f"<td>{s['found']} / {s['wrong']} / {s['missed']}</td>"
                f"<td>{threshold}</td><td>{lo:.2f} / {hi:.2f}</td></tr>"
            )
    return (
        "<table><tr><th>設定</th><th>正解した写真</th><th>見つけた / 誤り / 見逃し</th>"
        f"<th>しきい値</th><th>キャンベル缶の最低点 / ほかの最高点</th></tr>{''.join(rows)}"
        "</table>"
    )


def write_campbell(out, methods):
    truth = common.campbell27()
    captures = common.captures()
    views, results = dino.load()
    labels = list(methods)
    shown = [labels[0], labels[1], f"SAM 3.1＋DINOv2 {DINO_BEST[0]}"]
    threshold = methods[shown[2]][1]
    lines = []
    for key in truth:
        rgb, _, _ = common.load_capture(key)
        dino.write_flow(
            out / "dino",
            key,
            rgb,
            results[key],
            views,
            dino.work_dir() / "out",
            dino.EVAL_PHRASE,
            threshold,
            model=DINO_BEST[0],
            feature=DINO_BEST[1],
            truth=truth[key],
            note=captures[key]["scene"],
            back="../dino.html",
        )
        cells = []
        for i, label in enumerate(shown):
            v = methods[label][0][key]
            name = f"campbell27/{key}_{i}.jpg"
            image = common.result_image(rgb, v["boxes"], v["truth"])
            common.save_jpeg(image, out / name, width=320)
            cells.append(result_cell(name, v))
        flow_a = common.WORK / "plan_a" / "campbell27" / key / "flow.html"
        lines.append(
            f"<tr><th>{key}<br><small>{html.escape(captures[key]['scene'])}</small><br>"
            f'<a href="dino/{key}.html">DINOv2の流れ</a><br>'
            f'<a href="{os.path.relpath(flow_a, out)}">案Aの流れ</a></th>'
            f"{''.join(cells)}</tr>"
        )
    common.write_page(
        out / "dino.html",
        "キャンベル缶27枚：案A・参照を並べる方式・SAM 3.1＋DINOv2",
        "<p>しきい値は同じ写真で一番良い値を選んだので、しきい値を使う方式の正解数は甘め。"
        "時間は DGX での処理時間（写真の転送は含まない）。</p>"
        + summary_table(methods, with_threshold=True)
        + "<h2>DINOv2 の設定ごと</h2>"
        + dino_settings_table()
        + "<h2>写真ごと</h2><table><tr><th>写真</th>"
        + "".join(f"<th>{html.escape(s)}</th>" for s in shown)
        + f"</tr>{''.join(lines)}</table>",
    )


# --- tracked summary ------------------------------------------------------------------


def representative_images(campbell):
    """A few images for the tracked summary (small JPEGs)."""
    IMG.mkdir(exist_ok=True)
    a13 = common.WORK / "plan_a" / "runs19" / "13"
    common.save_jpeg(Image.open(a13 / "candidates.png"), IMG / "a_candidates.jpg", 420)
    # The candidate the match judged "same": the back-facing Pringles.
    same = next(
        m["number"]
        for m in common.read_json(a13 / "result.json")["vlm"]["matches"]
        if m["same"]
    )
    for suffix, name in (("_boxes", "a_match_boxes"), ("", "a_match_pair")):
        image = Image.open(a13 / f"match_{same}{suffix}.png")
        common.save_jpeg(image, IMG / f"{name}.jpg", 420)
    common.save_jpeg(plan_b_image("13", "b"), IMG / "b2_overlay.jpg", 420)

    key = "1018-05"
    rgb, _, _ = common.load_capture(key)
    items = collage.scored("clean", COLLAGE_BEST)[key]
    image = scored_image(rgb, items, common.campbell27()[key])
    common.save_jpeg(image, IMG / f"collage_{key}.jpg", 420)
    canvas = common.WORK / "collage" / "clean" / "out" / "canvas_all+soup_can.png"
    if canvas.exists():
        common.save_jpeg(Image.open(canvas), IMG / "collage_canvas.jpg", 560)

    views, results = dino.load()
    work = dino.work_dir() / "out"
    strip = tile_row([work / f"ref_{v}.png" for v in views])
    common.save_jpeg(strip, IMG / "references.jpg", 560)
    threshold = campbell[f"SAM 3.1＋DINOv2 {DINO_BEST[0]}"][1]
    candidates = results[key]["candidates"]
    sims = [dino.best_similarity(c, *DINO_BEST) for c in candidates]
    common.save_jpeg(
        common.draw_boxes(
            rgb,
            [c["bbox"] for c in candidates],
            [common.GREEN if s >= threshold else common.GREY for s in sims],
            [f"{c['number']}: {s:.2f}" for c, s in zip(candidates, sims)],
        ),
        IMG / f"dino_{key}.jpg",
        420,
    )
    crops = tile_row([work / f"{key}_{c['number']}_crop.png" for c in candidates])
    common.save_jpeg(crops, IMG / f"dino_{key}_crops.jpg", 420)


def write_summary(runs19, campbell):
    representative_images(campbell)
    template = (common.HERE / "summary_template.html").read_text(encoding="utf-8")
    body = (
        template.replace("{{runs19}}", summary_table(runs19))
        .replace("{{campbell27}}", summary_table(campbell, with_threshold=True))
        .replace("{{phrases}}", phrases.table_html())
    )
    common.write_page(
        common.HERE / "index.html", "SAM 3.1 で商品名の物を見つける：方式の比較", body
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    runs19, campbell = runs19_methods(), campbell_methods()
    write_runs19(REPORTS, runs19)
    write_collage(REPORTS)
    write_campbell(REPORTS, campbell)
    phrases.report(REPORTS)
    index = (
        '<ul><li><a href="runs19.html">案A・案B（19回）</a></li>'
        '<li><a href="collage.html">参照を並べる方式</a></li>'
        '<li><a href="dino.html">キャンベル缶27枚の比較（DINOv2 の検出フローつき）</a></li>'
        '<li><a href="phrases.html">SAM 3.1 に渡す言葉の比較（固有名詞1つで使うには）</a></li></ul>'
    )
    common.write_page(REPORTS / "index.html", "SAM 3.1 実験レポート（全写真）", index)
    if args.summary:
        write_summary(runs19, campbell)


if __name__ == "__main__":
    main()
