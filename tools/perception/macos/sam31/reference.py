#!/usr/bin/env python3
"""Capture clean reference images of one product, view by view.

Each capture is segmented on the DGX with a generic phrase; the object nearest
the image centre becomes the reference (mask, box, crop and size from depth).
Without --phrase, the DGX VLM looks at the first shot once and chooses that phrase.
The phrase is saved with the references (phrase.json) so detection needs only the
product name. References are saved under .cache-sam31/references/<name>/<view>/
with an index.html for checking them.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interactive  # noqa: E402

ROOT = interactive.ROOT
sys.path.insert(0, str(ROOT / "overlay_ws/src/aha_perception"))
from aha_perception.sam31 import erode  # noqa: E402
from aha_perception import sam31_vlm  # noqa: E402
from aha_perception.sam31_vlm import focal_lengths, physical_size  # noqa: E402

# Views every product needs; extra views (e.g. "top", "front-far") are allowed.
REQUIRED_VIEWS = ("front", "back", "left", "right")
VIEW_HINTS = {
    "front": "正面（ロゴ・商品名が見える面）",
    "back": "背面（裏の表示が見える面）",
    "left": "左側面",
    "right": "右側面",
}
# Quality limits for a reference shot.
MIN_HEIGHT_RATIO = 0.3  # object box height / image height
MAX_OTHER_SCORE = 0.5  # other objects found with at least this score
MIN_VALID_DEPTH = 0.5  # share of mask pixels with valid depth
# Objects whose box centre is outside the middle of the image are ignored.
CENTRE_MARGIN = 0.2


def pick_object(objects, width, height):
    """Return the object nearest the image centre, ignoring ones near the edges."""
    central = []
    for obj in objects:
        x0, y0, x1, y1 = obj["bbox_xyxy"]
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        if (
            CENTRE_MARGIN <= cx <= 1 - CENTRE_MARGIN
            and CENTRE_MARGIN <= cy <= 1 - CENTRE_MARGIN
        ):
            central.append(((cx - 0.5) ** 2 + (cy - 0.5) ** 2, obj))
    return min(central, key=lambda item: item[0])[1] if central else None


def reference_size(mask, depth_m, camera_info):
    """Size from depth; a cut-off object is reported as a warning instead."""
    size = physical_size(mask, depth_m, focal_lengths(camera_info))
    if size is not None:
        del size["cut_off"]
    return size


def check_quality(chosen, objects, mask, depth_m, width, height):
    """Japanese warnings for problems that make a poor reference."""
    warnings = []
    x0, y0, x1, y1 = chosen["bbox_xyxy"]
    if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
        warnings.append(
            "物が画面の端で切れています。全体が入るように置き直してください"
        )
    if (y1 - y0) / height < MIN_HEIGHT_RATIO:
        warnings.append("物が小さく写っています。カメラに近づけてください")
    others = [o for o in objects if o is not chosen and o["score"] >= MAX_OTHER_SCORE]
    if others:
        warnings.append(
            f"ほかの物が{len(others)}個写っています。対象の商品だけを置いてください"
        )
    samples = depth_m[mask]
    if len(samples) and np.mean(np.isfinite(samples) & (samples > 0)) < MIN_VALID_DEPTH:
        warnings.append(
            "距離を測れない部分が多いです。光の反射や距離（近すぎ）を確認してください"
        )
    return warnings


def draw_reference(rgb, mask, bbox, label):
    """Photo with the reference outlined by a thick box and a label."""
    color = (40, 220, 90)
    pixels = rgb.copy()
    pixels[mask & ~erode(erode(erode(mask)))] = color
    image = Image.fromarray(pixels)
    font = ImageFont.load_default(size=max(16, rgb.shape[0] // 20))
    interactive.draw_labeled_box(
        ImageDraw.Draw(image), bbox, label, color, max(4, rgb.shape[0] // 80), font
    )
    return image


def save_view(library, name, view, capture_dir, result_dir, summary):
    """Store one reference view; returns its record or raises ValueError."""
    with np.load(capture_dir / "rgbd.npz", allow_pickle=False) as capture:
        rgb, depth_m = capture["rgb"], capture["depth_m"]
        metadata = json.loads(capture["metadata"].item())
    height, width = rgb.shape[:2]
    chosen = pick_object(summary["objects"], width, height)
    if chosen is None:
        raise ValueError(
            "画面の中央に物が見つかりません。商品を画面の中央に置いてください"
        )
    labels = np.asarray(Image.open(result_dir / "labels.png"))
    mask = labels == chosen["label"]
    warnings = check_quality(chosen, summary["objects"], mask, depth_m, width, height)
    target = library / name / view
    staging = target.with_name(f".{view}.{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    try:
        x0, y0, x1, y1 = chosen["bbox_xyxy"]
        Image.fromarray(rgb).save(staging / "rgb.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(staging / "mask.png")
        Image.fromarray(rgb[y0:y1, x0:x1]).save(staging / "crop.png")
        draw_reference(rgb, mask, chosen["bbox_xyxy"], f"{name} / {view}").save(
            staging / "preview.png"
        )
        shutil.copy(capture_dir / "rgbd.npz", staging / "rgbd.npz")
        record = {
            "name": name,
            "view": view,
            "bbox_xyxy": chosen["bbox_xyxy"],
            "score": chosen["score"],
            "pixels": chosen["pixels"],
            "size": reference_size(mask, depth_m, metadata.get("camera_info")),
            "warnings": warnings,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "device": metadata.get("device"),
        }
        (staging / "view.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Replace an earlier shot of the same view only after a complete save.
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    write_index(library, name)
    return record


def load_phrase(library, name):
    """{"phrase", "source"} saved for a product, or None."""
    path = library / name / "phrase.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def save_phrase(library, name, phrase, source):
    record = {
        "phrase": phrase,
        "source": source,  # "vlm" (chosen by the VLM) or "user" (--phrase)
        "decided_at": datetime.now().isoformat(timespec="seconds"),
    }
    (library / name).mkdir(parents=True, exist_ok=True)
    (library / name / "phrase.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def saved_or_ask(library, name, ask):
    """The saved phrase, else `ask()` (the VLM) once and save its answer."""
    saved = load_phrase(library, name)
    if saved:
        return saved["phrase"]
    phrase = ask()
    save_phrase(library, name, phrase, "vlm")
    print(f"[reference] Qwen が種類の言葉を決めました: {phrase}", flush=True)
    return phrase


def resolve_phrase(args, library, capture, kind):
    """The SAM phrase: --phrase, else the saved one, else ask the VLM once."""
    if args.phrase:
        saved = load_phrase(library, args.name)
        if not saved or (saved["phrase"], saved["source"]) != (args.phrase, "user"):
            save_phrase(library, args.name, args.phrase, "user")
        return args.phrase
    if kind is None and not load_phrase(library, args.name):
        raise RuntimeError("種類の言葉がありません。--phrase で指定してください")
    return saved_or_ask(library, args.name, lambda: kind(np.load(capture)["rgb"]))


def load_views(library, name):
    views = {}
    for path in sorted((library / name).glob("*/view.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        views[record["view"]] = record
    return views


def write_index(library, name):
    """reference.json (all views) and index.html for checking the shots."""
    views = load_views(library, name)
    (library / name / "reference.json").write_text(
        json.dumps(
            {"name": name, "views": list(views.values())}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    missing = [v for v in REQUIRED_VIEWS if v not in views]
    phrase = load_phrase(library, name)
    phrase_line = (
        f"<p class='meta'>種類の言葉: {escape(phrase['phrase'])}"
        f"（{'Qwen が決定' if phrase['source'] == 'vlm' else '指定'}）</p>"
        if phrase
        else ""
    )
    cards = []
    for view, record in views.items():
        notes = "".join(f"<li>{escape(w)}</li>" for w in record["warnings"])
        cards.append(
            f"<section><h2>{escape(view)} <span class='meta'>{escape(VIEW_HINTS.get(view, ''))}</span></h2>"
            f"<img src='{escape(view)}/preview.png' alt='{escape(view)}の参照画像'>"
            f"<p class='meta'>{interactive.size_text_ja(record['size'])} ・ score {record['score']:.2f} ・ {escape(record['captured_at'])}</p>"
            + (
                f"<ul class='warn'>{notes}</ul>"
                if notes
                else "<p class='ok'>問題なし</p>"
            )
            + "</section>"
        )
    body = (
        f"<h1>参照画像: {escape(name)}</h1>"
        + phrase_line
        + (
            f"<p class='warn'>まだ撮っていない向き: {escape(', '.join(missing))}</p>"
            if missing
            else "<p class='ok'>必要な向き（正面・背面・左右）はすべて撮影済み</p>"
        )
        + f"<div class='grid'>{''.join(cards)}</div>"
    )
    path = library / name / "index.html"
    path.write_text(
        interactive.html_page("参照画像", REFERENCE_STYLE, body), encoding="utf-8"
    )
    return path


REFERENCE_STYLE = """
:root{--bg:#f6f6f4;--fg:#222;--card:#fff;--line:#ddd;--muted:#666}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ddd;--card:#1f1f1f;--line:#333;--muted:#999}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,"Hiragino Sans",sans-serif}
main{max-width:1100px;margin:auto}h1{font-size:20px}h2{font-size:16px;margin:0 0 6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
img{width:100%;border-radius:4px}.meta{color:var(--muted);font-weight:normal;font-size:13px}
.warn{color:#c62828}.ok{color:#0a7d32}
"""


def next_view(views):
    return next((v for v in REQUIRED_VIEWS if v not in views), "")


def session(args, worker, camera, library, read=input, kind=None):
    """Capture views until EOF; `kind(rgb) -> phrase` asks the VLM when needed."""
    print(
        f"[reference] 「{args.name}」の参照画像を撮ります。"
        "商品を1つだけ、画面の中央に置いてください。",
        flush=True,
    )
    while True:
        views = load_views(library, args.name)
        suggested = next_view(views)
        hint = (
            f"{suggested}: {VIEW_HINTS[suggested]}"
            if suggested
            else "必要な向きは撮影済み"
        )
        entered = read(
            f"向き [{hint}]（Enterで撮影、別の向きは名前を入力、Ctrl+Cで終了）: "
        ).strip()
        view = entered or suggested
        if not view:
            print("[reference] 追加で撮る向きの名前を入力してください（例: top）。")
            continue
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", view):
            print("[reference] 向きの名前は半角英数字で入力してください（例: front）。")
            continue
        work = library / ".work" / uuid.uuid4().hex[:8]
        camera.set_status(f"「{args.name}」{view} を撮影中…")
        try:
            camera.capture(work / "capture")
        except interactive.CaptureError as exc:
            print(f"[reference] 撮影に失敗しました: {exc}", file=sys.stderr)
            camera.set_status("撮影に失敗しました。Enterで撮り直せます")
            continue
        camera.set_status(f"「{args.name}」{view} から商品を切り出し中…")
        try:
            phrase = resolve_phrase(args, library, work / "capture/rgbd.npz", kind)
            response = worker.predict(phrase, work / "capture/rgbd.npz")
            summary = interactive.unpack_result(response, work / "result")
            record = save_view(
                library, args.name, view, work / "capture", work / "result", summary
            )
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            print(f"[reference] {view}: {exc}", file=sys.stderr)
            camera.set_status(f"{view}: 保存できませんでした。置き直してEnter")
            continue
        finally:
            shutil.rmtree(work, ignore_errors=True)
        preview = library / args.name / view / "preview.png"
        camera.show_result(preview, f"「{args.name}」{view}")
        message = f"[reference] {view} を保存しました: {preview.parent}"
        if record["warnings"]:
            message += "\n" + "\n".join(f"  注意: {w}" for w in record["warnings"])
            camera.set_status(
                f"{view}: 保存しました（注意あり）。撮り直す場合は {view} と入力"
            )
        else:
            camera.set_status(f"{view}: 保存しました。次の向きに置き換えてEnter")
        print(message, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="product key, e.g. pringles")
    parser.add_argument(
        "--phrase",
        help="generic English phrase SAM 3.1 uses to find the product in the shot;"
        " omit to let the DGX VLM choose it from the first shot",
    )
    parser.add_argument("--vlm-model", default=sam31_vlm.DEFAULT_VLM_MODEL)
    parser.add_argument(
        "--library", type=Path, default=ROOT / ".cache-sam31/references"
    )
    parser.add_argument("--host", default=os.environ.get("SAM31_SSH_HOST"))
    parser.add_argument("--user")
    parser.add_argument("--port", type=int)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument(
        "--checkpoint", default="/srv/shared/models/sam3.1/sam3.1_multiplex.pt"
    )
    parser.add_argument("--image", default="local/aharobot-sam31")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--preview-port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.name):
        parser.error("name must be lowercase letters, digits, '-' or '_'")
    if args.phrase is not None and not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9 '&.-]*", args.phrase
    ):
        parser.error("phrase must be a short English phrase")
    if args.user is not None and not re.fullmatch(r"[a-z_][a-z0-9_.-]*", args.user):
        parser.error("invalid Unix username")
    if args.identity_file is not None:
        args.identity_file = args.identity_file.expanduser().resolve()
    # The DGX worker runs plain SAM 3.1 at a low threshold to find the product.
    args.threshold, args.vlm, args.no_vlm_match = 0.3, False, False
    library = args.library.resolve()
    worker = camera = None
    status = 0
    log = None
    stack = contextlib.ExitStack()
    try:
        interactive.configure_ssh(args)
        worker = interactive.RemoteWorker(args)
        worker.run("true")
        interactive.prepare_camera("reference")
        (library / args.name).mkdir(parents=True, exist_ok=True)
        camera = interactive.CameraStream(args)
        print("[reference] カメラを起動しています...", flush=True)
        device = camera.start()
        print(f"[reference] {device}; ライブ映像: {camera.url}", flush=True)
        index = write_index(library, args.name)
        if platform.system() == "Darwin" and not args.no_open:
            subprocess.run(["open", camera.url], check=False)
        log = (library / args.name / "dgx.log").open("w", encoding="utf-8")
        worker.start(stderr=log)
        print(f"[reference] 準備完了。確認用ページ: {index}", flush=True)
        if platform.system() == "Darwin" and not args.no_open:
            subprocess.run(["open", str(index)], check=False)
        kind = None
        if not args.phrase and not load_phrase(library, args.name):
            url = stack.enter_context(interactive.vlm_tunnel(args))
            client = sam31_vlm.OllamaClient(url, args.vlm_model, timeout=600)
            kind = lambda rgb: sam31_vlm.kind_phrase(client, rgb)  # noqa: E731
        session(args, worker, camera, library, kind=kind)
    except (KeyboardInterrupt, EOFError):
        print("\n[reference] 終了します...", flush=True)
        status = 130
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[reference] {exc}", file=sys.stderr)
        status = 1
    finally:
        stack.close()
        if camera is not None:
            camera.close()
        if worker is not None and not worker.close():
            status = 1
        if log is not None:
            log.close()
        shutil.rmtree(library / ".work", ignore_errors=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
