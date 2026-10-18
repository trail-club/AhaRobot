"""Shared helpers for SAM 3.1 + DINOv2 detection and the method comparison.

Inputs are the saved captures under `.cache-sam31/` (not tracked); `data/` lists them with
ground truth and `results/` keeps each experiment's numeric output. Large outputs (masks,
crops, full HTML reports) go to `.cache-sam31/sam31_dinov2/`.
"""

import argparse
import contextlib
import functools
import html
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CACHE = ROOT / ".cache-sam31"
DATA = HERE / "data"
RESULTS = HERE / "results"
WORK = CACHE / "sam31_dinov2"
REMOTE = HERE / "remote"

sys.path.insert(0, str(ROOT / "overlay_ws/src/aha_perception"))
sys.path.insert(0, str(ROOT / "tools/perception/macos/sam31"))
import interactive  # noqa: E402
from aha_perception.sam31 import load_capture as load_rgbd  # noqa: E402
from aha_perception.sam31_vlm import OllamaClient, focal_lengths  # noqa: E402

DGX_HOST = os.environ.get("SAM31_SSH_HOST", interactive.DGX_ADDRESS)
DGX_PORT = int(os.environ.get("SAM31_SSH_PORT", interactive.DGX_SSH_PORT))
DGX_USER = os.environ.get("SAM31_SSH_USER")
IMAGE = "local/aharobot-sam31"
CHECKPOINT_DIR = "/srv/shared/models/sam3.1"
VLM_MODEL = "qwen3-vl:30b"
MATCH_IOU = 0.5
MIN_PIXELS = 50  # specks are ignored for every method (also remote/dino_common.py)
# Shipped with every remote script (the container sees only its work directory).
REMOTE_MODULES = ("sam_session.py", "dino_common.py")

GREEN, RED, BLUE, ORANGE, GREY, WHITE = (
    (40, 200, 70),
    (230, 40, 40),
    (40, 140, 255),
    (255, 170, 0),
    (150, 150, 150),
    (255, 255, 255),
)


# --- data ---------------------------------------------------------------------------


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


@functools.cache
def captures():
    """Capture id -> {"path": rgbd.npz relative to the repo, "scene": description}."""
    return read_json(DATA / "captures.json")


def capture_paths(keys):
    """Capture id -> absolute rgbd.npz path."""
    return {k: ROOT / captures()[k]["path"] for k in keys}


def load_capture(capture_id):
    """RGB, depth in metres and (fx, fy) of a saved capture."""
    rgb, depth, metadata = load_rgbd(ROOT / captures()[capture_id]["path"])
    return rgb, depth, focal_lengths(metadata["camera_info"])


@functools.cache
def runs19():
    """The 19 prompt runs (14 captures) used for options A and B."""
    return read_json(DATA / "runs19.json")


@functools.cache
def campbell27():
    """Campbell's can ground truth for the 27 captures used for the reference methods."""
    return read_json(DATA / "campbell27.json")


def load_mask(path, shape):
    """A mask saved by remote/sam_session.save_mask (packed bits)."""
    return (
        np.unpackbits(np.load(path))[: shape[0] * shape[1]].reshape(shape).astype(bool)
    )


# --- scoring --------------------------------------------------------------------------


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def matches_any(box, boxes):
    return any(iou(box, other) > MATCH_IOU for other in boxes)


def evaluate(boxes, truth):
    """Found / wrong / missed counts of detected boxes against ground-truth boxes."""
    hit = [matches_any(b, truth) for b in boxes]
    found = sum(matches_any(t, boxes) for t in truth)
    return {
        "found": found,
        "wrong": hit.count(False),
        "missed": len(truth) - found,
        "hit": hit,
        "correct": found == len(truth) and hit.count(False) == 0,
    }


def kept(items, threshold):
    """Boxes of [(score, box)] at or above a threshold."""
    return [b for s, b in items if s >= threshold]


def best_threshold(scored, truth):
    """Threshold (0.00-1.00) keeping the most photos correct.

    `scored` maps a photo to [(score, box)]. Ties keep the lowest threshold.
    """
    best = (0.0, -1)
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        correct = sum(
            evaluate(kept(items, t), truth[name])["correct"]
            for name, items in scored.items()
        )
        if correct > best[1]:
            best = (float(t), correct)
    return best[0]


def threshold_scores(scored, truth, seconds, threshold=None):
    """Per-photo evaluation at `threshold` (default: the best one) and the threshold."""
    threshold = best_threshold(scored, truth) if threshold is None else threshold
    out = {}
    for key, items in scored.items():
        boxes = kept(items, threshold)
        out[key] = {
            "boxes": boxes,
            "final": evaluate(boxes, truth[key]),
            "seconds": seconds[key],
            "truth": truth[key],
        }
    return out, threshold


def separation(scored, truth):
    """(margin, lowest target score, highest other score) over all photos."""
    target = [
        s for n, items in scored.items() for s, b in items if matches_any(b, truth[n])
    ]
    other = [
        s
        for n, items in scored.items()
        for s, b in items
        if not matches_any(b, truth[n])
    ]
    lo, hi = min(target, default=0.0), max(other, default=0.0)
    return round(lo - hi, 3), round(lo, 3), round(hi, 3)


def summarize(evaluations, seconds):
    """Totals over photos: correct photos, found/wrong/missed and time."""
    return {
        "photos": len(evaluations),
        "correct": sum(e["correct"] for e in evaluations),
        "found": sum(e["found"] for e in evaluations),
        "wrong": sum(e["wrong"] for e in evaluations),
        "missed": sum(e["missed"] for e in evaluations),
        "mean_seconds": round(float(np.mean(seconds)), 2),
        "max_seconds": round(float(np.max(seconds)), 2),
    }


def summarize_method(method):
    """`summarize` of a per-photo {"final", "seconds"} mapping."""
    return summarize(
        [v["final"] for v in method.values()], [v["seconds"] for v in method.values()]
    )


# --- DGX ------------------------------------------------------------------------------


def dgx(user=DGX_USER):
    """SSH destination in the form interactive.ssh_command expects."""
    return argparse.Namespace(
        host=DGX_HOST, port=DGX_PORT, user=user, identity_file=None
    )


def retry(run, attempts=60):
    """Repeat `run() -> bool` while the VPN/SSH link is down."""
    for _ in range(attempts):
        if run():
            return
        time.sleep(10)
    raise RuntimeError("DGX is not reachable over SSH")


def ssh(script, *, user=DGX_USER, **kwargs):
    """Run a remote shell script, retrying while the VPN/SSH link is down."""
    for _ in range(60):
        result = subprocess.run(interactive.ssh_command(dgx(user), script), **kwargs)
        if result.returncode != 255:  # 255 = ssh itself failed to connect
            return result
        time.sleep(10)
    raise RuntimeError("DGX is not reachable over SSH")


def piped(producer, consumer, env=None):
    """Run `producer | consumer`; True when both succeed."""
    with subprocess.Popen(producer, stdout=subprocess.PIPE, env=env) as first:
        done = subprocess.run(consumer, stdin=first.stdout)
    return done.returncode == 0 and first.returncode == 0


def run_in_container(work_dir, script, *, user=DGX_USER):
    """Run `remote/<script>` in the SAM 3.1 image on the DGX with `work_dir` as /work.

    The container runs detached so a dropped connection does not stop it; outputs written
    to /work/out are copied back to `work_dir/out`.
    """
    work_dir = Path(work_dir)
    for name in (script, *REMOTE_MODULES):
        (work_dir / name).write_bytes((REMOTE / name).read_bytes())
    remote = ssh(
        "mktemp -d /tmp/aharobot-exp-XXXXXX", user=user, capture_output=True, text=True
    ).stdout.strip()
    if not remote.startswith("/tmp/aharobot-exp-"):
        raise RuntimeError(f"unexpected remote directory {remote!r}")
    quoted = shlex.quote(remote)
    retry(
        lambda: piped(
            ["tar", "--no-xattrs", "-C", str(work_dir), "-cf", "-", "."],
            interactive.ssh_command(
                dgx(user), f"tar -xf - -C {quoted} && chmod -R a+rwX {quoted}"
            ),
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
    )
    name = "aharobot-exp-" + remote.rsplit("-", 1)[1]
    ssh(
        f"docker run -d --name {name} --gpus all --network host --shm-size=2g"
        f" --user $(id -u):$(id -g) -e HOME=/tmp -v {quoted}:/work"
        f" -v {CHECKPOINT_DIR}:/ckpt:ro -v $HOME/.cache/aharobot-sam31:/cache"
        " -e TRITON_CACHE_DIR=/cache/triton -e HF_HOME=/cache/huggingface"
        f" --entrypoint sh {IMAGE} -c"
        f" 'python -u /work/{script} /work > /work/run.out 2> /work/err.log'",
        user=user,
        check=True,
    )
    print(f"[exp] {script} running on the DGX ({remote})", flush=True)
    while True:
        state = ssh(
            f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {name}",
            user=user,
            capture_output=True,
            text=True,
        ).stdout.split()
        if state and state[0] != "running":
            break
        time.sleep(15)
    if state[1:] != ["0"]:
        err = ssh(
            f"tail -30 {quoted}/err.log", user=user, capture_output=True, text=True
        )
        raise RuntimeError(f"{script} failed on the DGX:\n{err.stdout}")
    out = work_dir / "out"
    out.mkdir(exist_ok=True)
    retry(
        lambda: piped(
            interactive.ssh_command(dgx(user), f"tar -C {quoted}/out -cf - ."),
            ["tar", "-C", str(out), "-xf", "-"],
        )
    )
    ssh(f"docker rm {name} >/dev/null; rm -rf {quoted}", user=user)


@contextlib.contextmanager
def vlm_client(user=DGX_USER):
    """OllamaClient for the DGX VLM through an SSH tunnel."""
    with interactive.vlm_tunnel(dgx(user)) as url:
        yield OllamaClient(url, VLM_MODEL, timeout=600)


# --- images and pages -----------------------------------------------------------------


def _font(size=20):
    """A Japanese-capable font (labels such as 見逃し); PIL's default has no kana."""
    for path in ("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",):
        with contextlib.suppress(OSError):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


FONT = _font()


def draw_boxes(rgb, boxes, colors, labels=None, width=6):
    """RGB with thick boxes; optional labels in filled tags at the top-left corner."""
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for i, (box, color) in enumerate(zip(boxes, colors)):
        draw.rectangle(box, outline=color, width=width)
        if labels and labels[i]:
            x, y = box[0] + 4, box[1] + 4
            w = draw.textlength(labels[i], font=FONT)
            draw.rectangle((x - 2, y - 2, x + w + 4, y + 24), fill=color)
            draw.text((x + 1, y), labels[i], fill=(0, 0, 0), font=FONT)
    return image


def result_image(rgb, boxes, truth, labels=None):
    """Green = target found, red = wrong object, white = missed target."""
    missed = [t for t in truth if not matches_any(t, boxes)]
    return draw_boxes(
        rgb,
        [*boxes, *missed],
        [GREEN if matches_any(b, truth) else RED for b in boxes]
        + [WHITE] * len(missed),
        (labels or [""] * len(boxes)) + ["見逃し"] * len(missed),
        width=7,
    )


def tint(rgb, mask, color, alpha=0.45):
    out = rgb.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def save_jpeg(image, path, width=480, quality=80):
    """Downscaled JPEG; reports stay small enough to browse and to track."""
    image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=quality)


STYLE = """
:root{--bg:#f6f6f4;--fg:#222;--card:#fff;--line:#ddd;--muted:#666;--ok:#178a35;--ng:#c62828}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161616;--fg:#ddd;
--card:#1f1f1f;--line:#333;--muted:#999;--ok:#5fd27c;--ng:#ff7b7b}}
body{font:14px -apple-system,"Hiragino Sans",sans-serif;margin:0 auto;padding:16px;
max-width:1400px;background:var(--bg);color:var(--fg)}
table{border-collapse:collapse;background:var(--card);margin:8px 0 24px}
td,th{border:1px solid var(--line);padding:4px 8px;vertical-align:top;text-align:left}
img{max-width:100%;border-radius:4px}td img{width:240px}.ok{color:var(--ok)}
.ng{color:var(--ng)}small,.muted{color:var(--muted)}a{color:inherit}.nowrap{white-space:nowrap}
"""


def page(title, body):
    return (
        f'<!doctype html><html lang="ja"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style>"
        f"<h1>{html.escape(title)}</h1>{body}</html>"
    )


def mark(ok):
    return '<b class="ok">○</b>' if ok else '<b class="ng">×</b>'


def outcome(e, seconds=None, digits=1):
    """○/× and found/wrong/missed counts of an `evaluate` result, with the time."""
    text = f'{mark(e["correct"])} 見つけた{e["found"]}/誤り{e["wrong"]}/見逃し{e["missed"]}'
    return text if seconds is None else f"{text}・{seconds:.{digits}f}秒"


def write_page(path, title, body):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page(title, body), encoding="utf-8")
    print(f"[exp] wrote {path}", flush=True)
