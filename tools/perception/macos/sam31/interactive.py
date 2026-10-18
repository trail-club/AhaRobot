#!/usr/bin/env python3
"""Capture RealSense frames on Enter and segment them in a persistent DGX worker."""

from __future__ import annotations

import argparse
import base64
import contextlib
from datetime import datetime
from html import escape
import http.server
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import queue
import re
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[4]
MACOS = ROOT / "tools/perception/macos"
RESULT_FILES = {"rgb.png", "labels.png", "overlay.png", "masks.npz", "result.json"}
# VLM mode only: numbered candidates and side-by-side crops per same-product
# check (up to sam31_vlm.MAX_CANDIDATES).
OPTIONAL_RESULT_FILES = {"candidates.png", *(f"match_{n}.png" for n in range(1, 9))}
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
]
DGX_ADDRESS = "10.99.0.1"
DGX_SSH_PORT = 2222
CAMERA_PASSWORD_PROMPT = "[sam31] このホストのパスワード（RealSense撮影用）: "


def configure_ssh(args, read=None):
    """Prompt for the roster username when no explicit SSH destination is given."""
    if args.host is None:
        args.host = DGX_ADDRESS
        if args.port is None:
            args.port = DGX_SSH_PORT
        if args.user is None:
            read = read or input
            print("[sam31] Cloudflare One (WARP) をConnectedにしてください。")
            while True:
                username = read(
                    "DGXのUnixユーザー名（directory-accessのroster.yml）: "
                ).strip()
                if re.fullmatch(r"[a-z_][a-z0-9_.-]*", username):
                    args.user = username
                    break
                print(
                    "[sam31] 登録済みのUnixユーザー名を入力してください（例: taro.yamada）。"
                )


def ssh_command(args, script):
    return [*ssh_options(args), "--", args.host, script]


def ssh_options(args):
    """`ssh` and its options, without the host."""
    command = ["ssh", *SSH_OPTIONS]
    if getattr(args, "port", None) is not None:
        command += ["-p", str(args.port)]
    if getattr(args, "user", None):
        command += ["-l", args.user]
    if getattr(args, "identity_file", None):
        command += ["-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"]
    return command


@contextlib.contextmanager
def vlm_tunnel(args):
    """Local URL forwarded to the DGX VLM (Ollama listens on loopback there)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tunnel = subprocess.Popen(
        [
            *ssh_options(args),
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"{port}:127.0.0.1:11434",
            "--",
            args.host,
        ]
    )
    try:
        for _ in range(100):
            if tunnel.poll() is not None:
                raise RuntimeError("DGXのVLMへのSSHトンネルを開始できません")
            with contextlib.suppress(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("DGXのVLMへのSSHトンネルが10秒以内につながりません")
        yield f"http://127.0.0.1:{port}"
    finally:
        tunnel.terminate()
        tunnel.wait()


def source_archive(destination):
    """Send only model code and locked build inputs; exclude credentials/artifacts."""
    upstream = ROOT / "upstream/sam3"
    tracked = (
        subprocess.run(
            ["git", "-C", str(upstream), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    if "pyproject.toml" not in tracked:
        raise RuntimeError("initialize upstream/sam3 before running this test")
    files = [upstream / name for name in tracked if name]
    files += list((ROOT / "overlay_ws/src/aha_perception/aha_perception").glob("*.py"))
    files += [
        ROOT / "tools/perception/macos/sam31" / name
        for name in ("Dockerfile", "pyproject.toml", "uv.lock")
    ]
    with tarfile.open(fileobj=destination, mode="w") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    destination.seek(0)


def unpack_result(response, output):
    """Only accept the expected result files, never arbitrary archive paths."""
    if response.get("type") != "result":
        raise RuntimeError(response.get("message", "unexpected worker response"))
    payload = base64.b64decode(response["archive"], validate=True)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
        unique = set(names)
        allowed = RESULT_FILES | OPTIONAL_RESULT_FILES
        if len(names) != len(unique) or not RESULT_FILES <= unique <= allowed:
            raise ValueError("unexpected result archive contents")
        output.mkdir()
        for name in names:
            (output / name).write_bytes(archive.read(name))
    return response["summary"]


FLOW_STYLE = """
body{margin:0;padding:16px;background:#f6f6f4;color:#222;font:14px/1.55 -apple-system,"Hiragino Sans",sans-serif}
main{max-width:1100px;margin:auto}
section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:12px 0}
h1{font-size:20px;margin:4px 0}h2{font-size:16px;margin:0 0 8px}h3{font-size:14px;margin:12px 0 4px}
.meta{color:#666}.arrow{text-align:center;color:#999;font-size:20px;margin:-6px 0}
img{max-width:100%;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:6px 0}
th,td{border-bottom:1px solid #eee;padding:4px 8px;text-align:left;vertical-align:top}
th{color:#666;font-weight:600}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:4px}
code,pre{font:12px ui-monospace,Menlo,monospace;background:#f3f3f3;border-radius:4px}
code{padding:1px 4px}pre{padding:8px;white-space:pre-wrap;max-height:320px;overflow:auto}
.keep{color:#0a7d32;font-weight:600}.drop{color:#999}
.pair{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end}.pair figure{margin:0}
.pair img{max-height:260px;max-width:100%}figcaption{color:#666;font-size:12px}
@media (prefers-color-scheme:dark){body{background:#161616;color:#ddd}
section{background:#1f1f1f;border-color:#333}th,td{border-color:#2c2c2c}
code,pre{background:#2a2a2a}.meta,th{color:#999}}
"""


def html_page(title, style, body):
    return (
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{style}</style><main>{body}</main></html>\n"
    )


def size_text_ja(size):
    """Box size from depth (see sam31_vlm.physical_size) for the HTML reports."""
    if not size:
        return "大きさ不明"
    text = f"高さ約{size['height_cm']:.0f} cm・幅約{size['width_cm']:.0f} cm（距離 {size['depth_m']:.2f} m）"
    return text + ("・画面端で切れている" if size.get("cut_off") else "")


def draw_labeled_box(draw, bbox, label, color, line_width, font):
    """Thick box with the label above it, or inside it at the image's top edge."""
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=line_width)
    left, top, right, bottom = draw.textbbox((x0, y0), label, font=font)
    dy = bottom - top + 8 if y0 >= bottom - top + 8 else -line_width
    draw.rectangle((left, top - dy, right + 8, bottom - dy + 4), fill=color)
    draw.text((x0 + 4, y0 - dy + 2), label, font=font, fill="#000000")


def candidate_boxes(vlm):
    """Candidate number -> bbox; older records only have it in the SAM runs."""
    boxes = {c["number"]: c["bbox_xyxy"] for c in vlm["candidates"] if "bbox_xyxy" in c}
    for run in vlm.get("sam_runs", []):
        for mask in run["masks"]:
            kind, _, number = mask.get("outcome", "").rpartition(" ")
            if kind == "candidate":
                boxes.setdefault(int(number), mask["bbox_xyxy"])
    return boxes


def draw_match_boxes(result_dir, match, boxes, colors):
    """Draw A (kept) and B (compared) as thick labelled boxes on the photo."""
    from PIL import Image, ImageDraw, ImageFont

    result_dir = Path(result_dir)
    pairs = [("A", match["reference"]), ("B", match["number"])]
    if not (result_dir / "rgb.png").exists() or any(n not in boxes for _, n in pairs):
        return None
    image = Image.open(result_dir / "rgb.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(4, image.height // 80)
    font = ImageFont.load_default(size=max(16, image.height // 20))
    for tag, number in pairs:
        color = colors.get(number, "#FFFFFF")
        draw_labeled_box(
            draw, boxes[number], f"{tag} #{number}", color, line_width, font
        )
    path = result_dir / f"match_{match['number']}_boxes.png"
    image.save(path)
    return path.name


def _outcome_text(outcome):
    kind, _, number = outcome.rpartition(" ")
    return {
        "candidate": f"→ 候補{number}",
        "duplicate of": f"候補{number}と重複（除外）",
    }.get(kind, "候補数の上限で除外" if outcome == "over limit" else outcome)


def _depth_text(depth):
    return "-" if depth is None else f"{depth:.2f} m"


def _swatch(color):
    return f'<span class="sw" style="background:{escape(color)}"></span>'


def _vlm_exchange_html(step):
    if not step:
        return "<p class='meta'>（実行なし）</p>"
    thinking = step.get("thinking", "").replace("<think>", "").strip()
    return (
        f"<p class='meta'>{step['seconds']:.1f} s</p>"
        f"<p>回答: <code>{escape(step.get('answer', '').strip())}</code></p>"
        + (
            f"<details><summary>推論過程</summary><pre>{escape(thinking)}</pre></details>"
            if thinking
            else ""
        )
    )


def _sam_runs_html(vlm):
    runs = []
    for run in vlm.get("sam_runs", []):
        rows = (
            "".join(
                f"<tr><td>{m['score']:.3f}</td><td>{m['bbox_xyxy']}</td><td>{m['pixels']}</td>"
                f"<td>{escape(_outcome_text(m.get('outcome', '')))}</td></tr>"
                for m in run["masks"]
            )
            or "<tr><td colspan=4 class='drop'>検出なし</td></tr>"
        )
        runs.append(
            f"<h3><code>{escape(run['phrase'])}</code> "
            f"<span class='meta'>{run.get('seconds', 0):.2f} s</span></h3>"
            "<table><tr><th>score</th><th>bbox (x1,y1,x2,y2)</th><th>画素数</th><th>扱い</th></tr>"
            f"{rows}</table>"
        )
    return "".join(runs) or "<p class=meta>記録なし</p>"


def _candidates_html(vlm):
    verdicts = {"select": "採用（選択）", "match": "採用（照合）"}
    rows = "".join(
        f"<tr><td>{_swatch(c['color'])}#{c['number']}</td><td><code>{escape(c['phrase'])}</code></td>"
        f"<td>{c['score']:.3f}</td><td class='{'keep' if c['kept'] else 'drop'}'>"
        f"{verdicts.get(c.get('via'), '採用') if c['kept'] else '除外'}</td></tr>"
        for c in vlm["candidates"]
    )
    return (
        "<img src='candidates.png' alt='番号付きの候補'>"
        "<table><tr><th>番号</th><th>元の句</th><th>score</th><th>VLMの判定</th></tr>"
        f"{rows}</table>"
    )


def _matches_html(result_dir, vlm, steps):
    if "matches" not in vlm:
        return "<p class='meta'>（この結果には照合の記録がない）</p>"
    if not vlm["matches"]:
        return "<p class='meta'>照合する候補なし（採用なし、または全候補を採用）</p>"
    boxes = candidate_boxes(vlm)
    colors = {c["number"]: c["color"] for c in vlm["candidates"]}
    parts = [
        "<p class='meta'>選択で除外した候補を、採用した候補と並べて同じ商品か確認する。"
        "ロゴが見えない向きの同じ商品を拾うため。切り出しは実寸の比率で並べ、"
        "深度から求めた大きさもVLMに伝える。</p>"
    ]
    for m in vlm["matches"]:
        a, b = m["reference"], m["number"]
        figures = ""
        box_image = draw_match_boxes(result_dir, m, boxes, colors)
        if box_image:
            figures += (
                f"<figure><img src='{box_image}' alt='写真上の候補{a}と候補{b}の枠'>"
                "<figcaption>写真上の位置（SAM 3.1 が検出した枠）</figcaption></figure>"
            )
        if (result_dir / f"match_{b}.png").exists():
            figures += (
                f"<figure><img src='match_{b}.png' alt='候補{a}と候補{b}の切り出し'>"
                "<figcaption>VLMに見せた切り出し（左がA、右がB）</figcaption></figure>"
            )
        parts.append(
            f"<h3>候補{a}（A） と 候補{b}（B） "
            f"<span class='{'keep' if m['same'] else 'drop'}'>"
            f"{'同じ商品 → 採用' if m['same'] else '別の商品 → 除外'}</span></h3>"
            + (
                f"<p class='meta'>A: {size_text_ja(m['reference_size'])} / B: {size_text_ja(m['size'])}</p>"
                if "size" in m
                else ""
            )
            + f"<div class='pair'>{figures}</div>"
            + _vlm_exchange_html(steps.get(f"match {b}"))
        )
    return "".join(parts)


def write_flow_report(result_dir, summary):
    """Write flow.html: prompt → VLM phrases → SAM per phrase → candidates → VLM pick → match → result."""
    result_dir = Path(result_dir)
    vlm = summary["vlm"]
    steps = {step.get("name"): step for step in vlm.get("steps", [])}
    objects = "".join(
        f"<tr><td>{_swatch(o.get('color', '#888'))}#{o['label']}</td><td>{o['score']:.3f}</td>"
        f"<td>{_depth_text(o['median_depth_m'])}</td>"
        f"<td>{o['bbox_xyxy']}</td></tr>"
        for o in summary["objects"]
    )
    has_candidates = (result_dir / "candidates.png").exists() and vlm["candidates"]
    sections = [
        f"<h1>「{escape(summary['prompt'])}」の検出フロー</h1>"
        f"<p class='meta'>合計 {summary['inference_seconds']:.1f} s（VLM {vlm['vlm_seconds']:.1f} s）"
        f" ・ VLM {escape(vlm['model'])} ・ 候補の閾値 {vlm['candidate_threshold']}</p>",
        "<section><h2>1. VLMによる言い換え</h2><img src='rgb.png' alt='入力画像' style='max-width:480px'>"
        f"<p>SAMに渡す句: {' / '.join(f'<code>{escape(p)}</code>' for p in vlm['phrases'])}</p>"
        f"{_vlm_exchange_html(steps.get('phrases'))}</section>",
        f"<section><h2>2. 句ごとのSAM 3.1検出</h2>{_sam_runs_html(vlm)}</section>",
        "<section><h2>3. 候補（重複除去後）</h2>"
        + (
            _candidates_html(vlm)
            if has_candidates
            else "<p class='meta'>候補なし（VLMの選択は実行しない）</p>"
        )
        + "</section>",
        f"<section><h2>4. VLMによる選択</h2><p>採用: {vlm.get('selected', vlm['keep']) or 'なし'}</p>"
        f"{_vlm_exchange_html(steps.get('select'))}</section>",
        f"<section><h2>5. 照合（同じ商品か）</h2>{_matches_html(result_dir, vlm, steps)}"
        f"<p>最終的な採用: {vlm['keep'] or 'なし'}</p></section>",
        "<section><h2>6. 結果</h2><img src='overlay.png' alt='最終結果'>"
        "<table><tr><th>番号</th><th>score</th><th>深度の中央値</th><th>bbox</th></tr>"
        f"{objects or '<tr><td colspan=4 class=drop>検出なし</td></tr>'}</table></section>",
    ]
    path = result_dir / "flow.html"
    path.write_text(
        html_page(
            "検出フロー", FLOW_STYLE, "<div class='arrow'>↓</div>".join(sections)
        ),
        encoding="utf-8",
    )
    return path


class RemoteWorker:
    def __init__(self, args):
        self.args = args
        self.name = "aharobot-sam31-" + uuid.uuid4().hex
        self.remote_dir = None
        self.process = None
        self.responses = queue.Queue()
        self.container_requested = False

    def ssh(self, script):
        return ssh_command(self.args, script)

    def run(self, script, **kwargs):
        return subprocess.run(self.ssh(script), check=True, **kwargs)

    def start(self, *, stderr=None):
        self.remote_dir = self.run(
            "mktemp -d /tmp/aharobot-sam31-XXXXXXXX", capture_output=True, text=True
        ).stdout.strip()
        if (
            not self.remote_dir.startswith("/tmp/aharobot-sam31-")
            or "\n" in self.remote_dir
        ):
            raise RuntimeError("unexpected remote temporary directory")
        remote = shlex.quote(self.remote_dir)
        print(
            "[sam31] Syncing code and building the DGX image (cached layers are reused)...",
            flush=True,
        )
        with tempfile.TemporaryFile() as archive:
            source_archive(archive)
            self.run(f"tar -xf - -C {remote}", stdin=archive)
        # The tag is shared by DGX users; run the ID built from this session's
        # sources so a concurrent build cannot swap the image under us.
        self.run(
            f"cd {remote} && docker build --iidfile .image-id "
            f"-t {shlex.quote(self.args.image)} "
            "-f tools/perception/macos/sam31/Dockerfile ."
        )
        image_id = self.run(
            f"cat {remote}/.image-id", capture_output=True, text=True
        ).stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise RuntimeError("unexpected Docker image ID")
        checkpoint = shlex.quote(self.args.checkpoint)
        self.run(
            f"test -r {checkpoint} && mkdir -p "
            '"$HOME/.cache/aharobot-sam31/torch/kernels" "$HOME/.cache/aharobot-sam31/triton"'
        )
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            self.name,
            "--gpus",
            "all",
            "--shm-size=2g",
            "-v",
            f"{self.remote_dir}:/app:ro",
            "-v",
            f"{self.args.checkpoint}:/models/sam3.1_multiplex.pt:ro",
        ]
        if self.args.vlm:
            # Ollama listens on the DGX loopback only.
            command += ["--network", "host"]
        command += [
            image_id,
            "--checkpoint",
            "/models/sam3.1_multiplex.pt",
            "--threshold",
            str(self.args.threshold),
        ]
        if self.args.vlm:
            command += [
                "--vlm",
                "--vlm-url",
                self.args.vlm_url,
                "--vlm-model",
                self.args.vlm_model,
                "--candidate-threshold",
                str(self.args.candidate_threshold),
            ]
            if self.args.no_vlm_match:
                command.append("--no-match")
        # Use the SSH user's UID for cache ownership, and never transfer HF tokens.
        script = (
            shlex.join(command[:2])
            + ' --user "$(id -u):$(id -g)" -v "$HOME/.cache/aharobot-sam31:/cache" '
            + shlex.join(command[2:])
        )
        self.container_requested = True
        self.process = subprocess.Popen(
            self.ssh(script),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            stderr=stderr,
            start_new_session=True,
        )
        threading.Thread(target=self._read_responses, daemon=True).start()
        print("[sam31] Loading SAM 3.1 on DGX...", flush=True)
        response = self.receive()
        if response.get("type") != "ready":
            raise RuntimeError(response.get("message", "worker did not become ready"))

    def _read_responses(self):
        try:
            for line in self.process.stdout:
                try:
                    response = json.loads(line)
                except ValueError:
                    response = None
                if not isinstance(response, dict):
                    # e.g. a shell startup file printing on the SSH channel.
                    raise RuntimeError(
                        f"unexpected output from DGX worker: {line.strip()[:200]!r}"
                    )
                self.responses.put(response)
        except (OSError, RuntimeError) as exc:
            self.responses.put(exc)
        finally:
            self.responses.put(
                RuntimeError("DGX worker disconnected; check SSH/Docker logs")
            )

    def receive(self):
        try:
            response = self.responses.get(timeout=self.args.timeout)
        except queue.Empty as exc:
            raise TimeoutError("DGX worker response timed out") from exc
        if isinstance(response, Exception):
            raise response
        return response

    def predict(self, prompt, capture):
        request = {
            "prompt": prompt,
            "rgbd": base64.b64encode(capture.read_bytes()).decode("ascii"),
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        # Ignore a repeated Ctrl+C while releasing the remote GPU container.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        success = True
        try:
            if self.process is not None:
                try:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                except OSError as exc:
                    print(f"[sam31] Local SSH cleanup: {exc}", file=sys.stderr)
                finally:
                    # Interrupted writes may leave buffered data that raises
                    # BrokenPipeError on close. Still attempt remote cleanup.
                    for stream in (self.process.stdin, self.process.stdout):
                        try:
                            stream.close()
                        except OSError:
                            pass
            if self.container_requested:
                try:
                    script = (
                        "docker info >/dev/null 2>&1 && "
                        f"if docker container inspect {self.name} >/dev/null 2>&1; then "
                        f"docker rm -f {self.name}; fi"
                    )
                    self.run(script, timeout=30)
                    print("[sam31] DGX SAM 3.1 container stopped.", flush=True)
                except (OSError, subprocess.SubprocessError) as exc:
                    print(
                        f"[sam31] Container cleanup failed: {exc}\nRun: {shlex.join(self.ssh('docker rm -f ' + self.name))}",
                        file=sys.stderr,
                    )
                    success = False
            if self.remote_dir:
                try:
                    self.run("rm -rf -- " + shlex.quote(self.remote_dir), timeout=30)
                except (OSError, subprocess.SubprocessError) as exc:
                    print(
                        f"[sam31] Temporary directory cleanup failed: {exc}",
                        file=sys.stderr,
                    )
                    success = False
        finally:
            signal.signal(signal.SIGINT, previous)
        return success


class CaptureError(RuntimeError):
    pass


PREVIEW_PAGE = """<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>カメラプレビュー</title><style>
:root{--bg:#f6f6f4;--fg:#1c1c1c;--muted:#666}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e6e6e6;--muted:#999}}
body{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 sans-serif}
#status{font-size:18px;font-weight:600;margin:4px 0}p{color:var(--muted);margin:4px 0}
img{width:100%;max-width:1280px;display:block;background:#000}
</style><div id="status">接続中…</div>
<p>左: カラー（中央に十字） / 右: 位置合わせ済み深度（赤=近い 0.3 m 〜 青=遠い 3 m、黒=測定不能）。
ターミナルでプロンプトを入力してEnterで撮影する。</p>
<img src="/stream" alt="RealSenseのライブ映像">
<h2 id="result-title" hidden>直前の検出結果</h2>
<img id="result" alt="直前の検出結果（マスク）" hidden style="max-width:640px">
<script>
let shown=0;
async function poll(){try{const r=await fetch('/status');const s=await r.json();
document.getElementById('status').textContent=s.status;
if(s.result&&s.result!==shown){shown=s.result;const img=document.getElementById('result');
img.src='/result.png?v='+shown;img.hidden=false;document.getElementById('result-title').hidden=false;
document.getElementById('result-title').textContent='直前の検出結果: '+s.result_label}}
catch(e){document.getElementById('status').textContent='停止しました'}}
poll();setInterval(poll,1000);
</script></html>
""".encode()


class CameraStream:
    """Keep one camera process open for the session and serve its preview.

    Only the camera process runs as root; it has no network access. The
    loopback preview server runs here, in the user's process.
    """

    def __init__(self, args):
        self.args = args
        self.process = None
        self.server = None
        self.frame = b""
        self.status = "起動中"
        # Latest result overlay: (version, label, PNG path).
        self.result = (0, "", None)
        self.updated = threading.Condition()
        self.replies = queue.Queue()

    def command(self):
        return camera_command(self.args, "--serve")

    def start(self, timeout=60):
        self.process = subprocess.Popen(
            self.command(), cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        threading.Thread(target=self._read, daemon=True).start()
        reply = self._reply(timeout)
        if reply.get("type") != "ready":
            raise CaptureError(f"camera did not start: {reply}")
        self.server = self._serve()
        self.set_status("待機中（プロンプトを入力してEnterで撮影）")
        return reply.get("device", "RealSense")

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/"

    def set_status(self, text):
        self.status = text

    def show_result(self, overlay, label):
        self.result = (self.result[0] + 1, label, Path(overlay))

    def _read(self):
        stream = self.process.stdout
        try:
            while line := stream.readline():
                message = json.loads(line)
                payload = stream.read(int(message.get("bytes", 0)))
                if message.get("type") == "preview":
                    with self.updated:
                        self.frame = payload
                        self.updated.notify_all()
                else:
                    self.replies.put(message)
        except (OSError, ValueError, TypeError):
            pass
        finally:
            self.replies.put({"type": "exit"})
            with self.updated:
                self.updated.notify_all()

    def _reply(self, timeout):
        try:
            reply = self.replies.get(timeout=timeout)
        except queue.Empty:
            raise CaptureError("camera did not respond") from None
        if reply.get("type") == "exit":
            self.replies.put(reply)
            raise CaptureError("camera process stopped; see the messages above")
        return reply

    def capture(self, destination):
        request = json.dumps({"capture": str(destination)}).encode() + b"\n"
        try:
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except OSError as exc:
            raise CaptureError(f"camera process stopped: {exc}") from exc
        reply = self._reply(timeout=15)
        if reply.get("type") != "captured":
            raise CaptureError(reply.get("message", f"unexpected reply {reply}"))

    def _serve(self):
        camera = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, content_type, body):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                version, label, overlay = camera.result
                if self.path == "/status":
                    body = json.dumps(
                        {
                            "status": camera.status,
                            "result": version,
                            "result_label": label,
                        },
                        ensure_ascii=False,
                    )
                    self.send("application/json; charset=utf-8", body.encode())
                elif self.path.startswith("/result.png") and overlay is not None:
                    try:
                        self.send("image/png", overlay.read_bytes())
                    except OSError:
                        self.send_error(404)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        # Send the current frame at once, then each new one.
                        frame = camera.frame
                        while camera.process.poll() is None:
                            if frame:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                    + frame
                                    + b"\r\n"
                                )
                                self.wfile.flush()
                            with camera.updated:
                                camera.updated.wait(timeout=5)
                                frame = camera.frame
                    except OSError:
                        pass
                else:
                    self.send("text/html; charset=utf-8", PREVIEW_PAGE)

        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self.args.preview_port), Handler
        )
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()


def camera_command(args, *mode):
    """capture_realsense.py with the camera options; root (sudo -n) on macOS."""
    command = [sys.executable, str(MACOS / "sam31/capture_realsense.py"), *mode]
    for name in ("width", "height", "fps", "warmup_frames"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.serial:
        command += ["--serial", args.serial]
    if platform.system() == "Darwin":
        command = ["sudo", "-n", *command]
    return command


def capture_frame(args, destination):
    if platform.system() == "Darwin":
        # Refresh authentication per frame so an expired sudo timestamp does not
        # interrupt a long interactive session. The controller/SSH remain user-owned.
        subprocess.run(["sudo", "-p", CAMERA_PASSWORD_PROMPT, "-v"], check=True)
    subprocess.run(
        camera_command(args, "--output", str(destination)), check=True, cwd=ROOT
    )


def prepare_camera(tag):
    """Make sure the RealSense binding is built and cache sudo for the camera."""
    subprocess.run([sys.executable, str(MACOS / "bootstrap_realsense.py")], check=True)
    if platform.system() == "Darwin":
        print(
            f"[{tag}] RealSense撮影のため、このMacのパスワードを入力してください。",
            flush=True,
        )
        subprocess.run(["sudo", "-p", CAMERA_PASSWORD_PROMPT, "-v"], check=True)


def interact(args, worker, session, read=input, camera=None):
    def status(text):
        if camera is not None:
            camera.set_status(text)

    prompt = ""
    count = 0
    while True:
        entered = read(
            f"Prompt{' [' + prompt + ']' if prompt else ''} (English; Enter to capture; Ctrl+C to stop): "
        ).strip()
        if entered:
            prompt = entered
        if not prompt:
            print("[sam31] Enter a prompt such as bottle, cup or person.")
            continue
        count += 1
        frame = session / f"{count:04d}"
        frame.mkdir()
        status(f"「{prompt}」を撮影中…")
        try:
            if camera is None:
                capture_frame(args, frame / "capture")
            else:
                camera.capture(frame / "capture")
        except (subprocess.CalledProcessError, CaptureError) as exc:
            detail = f": {exc}" if isinstance(exc, CaptureError) else ""
            print(
                f"[sam31] Capture failed{detail}; retry with Enter or change the prompt.",
                file=sys.stderr,
            )
            status("撮影に失敗しました。Enterで再撮影できます")
            continue
        status(f"「{prompt}」を推論中…（撮影済み: {frame.name}）")
        response = worker.predict(prompt, frame / "capture/rgbd.npz")
        if response.get("type") == "error":
            print("[sam31] Inference failed: " + response["message"], file=sys.stderr)
            status("推論に失敗しました。Enterで再撮影できます")
            continue
        # A damaged result loses only this frame, like capture/inference errors.
        try:
            summary = unpack_result(response, frame / "result")
            view = frame / "result/overlay.png"
            message = f"[sam31] {len(summary['objects'])} objects; {summary['inference_seconds']:.3f}s; {view}"
            if "vlm" in summary:
                vlm = summary["vlm"]
                view = write_flow_report(frame / "result", summary)
                message += (
                    f"\n[sam31] VLM phrases: {', '.join(vlm['phrases'])}; "
                    f"kept {vlm['keep']} of {len(vlm['candidates'])} candidates; "
                    f"VLM {vlm['vlm_seconds']:.1f}s; flow: {view}"
                )
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile, zlib.error) as exc:
            print(
                f"[sam31] Result transfer failed: {exc}; retry with Enter or change the prompt.",
                file=sys.stderr,
            )
            status("結果の受信に失敗しました。Enterで再撮影できます")
            continue
        print(message, flush=True)
        if camera is not None:
            camera.show_result(
                frame / "result/overlay.png",
                f"「{prompt}」{len(summary['objects'])}件（{frame.name}）",
            )
        status(
            f"完了: 「{prompt}」{len(summary['objects'])}件（{frame.name}）。"
            "次のプロンプトを入力するかEnterで再撮影"
        )
        if platform.system() == "Darwin" and not args.no_open:
            subprocess.run(["open", str(view)], check=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("SAM31_SSH_HOST"),
        help="SSH host/alias; omit to prompt for a user on 10.99.0.1:2222",
    )
    parser.add_argument("--user", help="directory-access Unix username; omit to prompt")
    parser.add_argument("--port", type=int, help="SSH port (direct connection: 2222)")
    parser.add_argument(
        "--identity-file", type=Path, help="private key path, if not a default SSH key"
    )
    parser.add_argument(
        "--checkpoint",
        default="/srv/shared/models/sam3.1/sam3.1_multiplex.pt",
        help="absolute checkpoint path on DGX",
    )
    parser.add_argument("--image", default="local/aharobot-sam31")
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache-sam31/results/interactive"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="worker startup/inference timeout in seconds",
    )
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="open the camera per frame instead of keeping it open with a live preview",
    )
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="let the DGX VLM pick SAM candidates (product and brand names)",
    )
    parser.add_argument("--vlm-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vlm-model", default="qwen3-vl:30b")
    parser.add_argument("--candidate-threshold", type=float, default=0.3)
    parser.add_argument(
        "--no-vlm-match",
        action="store_true",
        help="skip comparing rejected candidates side by side with a kept one",
    )
    args = parser.parse_args(argv)
    if args.host is not None and (not args.host or args.host.startswith("-")):
        parser.error("invalid SSH host")
    if args.user is not None and not re.fullmatch(r"[a-z_][a-z0-9_.-]*", args.user):
        parser.error("invalid Unix username")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("SSH port must be 1..65535")
    if args.identity_file is not None:
        args.identity_file = args.identity_file.expanduser().resolve()
        if not args.identity_file.is_file():
            parser.error("identity-file must be an existing private key file")
    if not args.checkpoint.startswith("/") or ":" in args.checkpoint:
        parser.error("checkpoint must be an absolute remote path without ':'")
    if not re.fullmatch(r"http://[A-Za-z0-9.:\[\]-]+/?", args.vlm_url):
        parser.error("vlm-url must look like http://host:port")
    if not re.fullmatch(r"[A-Za-z0-9._:/-]+", args.vlm_model):
        parser.error("invalid vlm-model")
    if not 0 <= args.candidate_threshold <= 1:
        parser.error("candidate-threshold must be 0..1")
    if not 0 <= args.threshold <= 1 or not 0 < args.timeout < float("inf"):
        parser.error("threshold must be 0..1 and timeout must be positive and finite")
    if min(args.width, args.height, args.fps, args.warmup_frames) < 1:
        parser.error("camera dimensions, fps and warmup-frames must be positive")
    if not 0 <= args.preview_port <= 65535:
        parser.error("preview-port must be 0..65535")
    worker = None
    camera = None
    status = 0
    log = None
    try:
        configure_ssh(args)
        worker = RemoteWorker(args)
        print(
            f"[sam31] Checking SSH access to {args.user + '@' if args.user else ''}{args.host}...",
            flush=True,
        )
        worker.run("true")
        prepare_camera("sam31")
        session = args.output.resolve() / (
            datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        )
        session.mkdir(parents=True)
        if not args.no_preview:
            # Open the camera first so framing can start while the model loads.
            camera = CameraStream(args)
            print("[sam31] Starting the camera...", flush=True)
            device = camera.start()
            print(f"[sam31] {device}; live preview: {camera.url}", flush=True)
            if platform.system() == "Darwin" and not args.no_open:
                subprocess.run(["open", camera.url], check=False)
        log = (session / "dgx.log").open("w", encoding="utf-8")
        print(f"[sam31] DGX model diagnostics: {session / 'dgx.log'}", flush=True)
        worker.start(stderr=log)
        print(f"[sam31] Ready. Results: {session}", flush=True)
        interact(args, worker, session, camera=camera)
    except (KeyboardInterrupt, EOFError):
        print("\n[sam31] Stopping...", flush=True)
        status = 130
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"[sam31] {exc}", file=sys.stderr)
        status = 1
    finally:
        if camera is not None:
            camera.close()
        if worker is not None and not worker.close():
            status = 1
        if log is not None:
            log.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
