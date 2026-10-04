#!/usr/bin/env python3
"""Capture RealSense frames on Enter and segment them in a persistent DGX worker."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import queue
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[4]
MACOS = ROOT / "tools/perception/macos"
RESULT_FILES = {"rgb.png", "labels.png", "overlay.png", "masks.npz", "result.json"}
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
    command = ["ssh", *SSH_OPTIONS]
    if getattr(args, "port", None) is not None:
        command += ["-p", str(args.port)]
    if getattr(args, "user", None):
        command += ["-l", args.user]
    if getattr(args, "identity_file", None):
        command += ["-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"]
    return [*command, "--", args.host, script]


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
        if set(archive.namelist()) != RESULT_FILES or len(archive.namelist()) != len(
            RESULT_FILES
        ):
            raise ValueError("unexpected result archive contents")
        output.mkdir()
        for name in RESULT_FILES:
            (output / name).write_bytes(archive.read(name))
    return response["summary"]


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
        self.run(
            f"cd {remote} && docker build -t {shlex.quote(self.args.image)} "
            "-f tools/perception/macos/sam31/Dockerfile ."
        )
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
            self.args.image,
            "--checkpoint",
            "/models/sam3.1_multiplex.pt",
            "--threshold",
            str(self.args.threshold),
        ]
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
                self.responses.put(json.loads(line))
        except (OSError, ValueError) as exc:
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


def capture_frame(args, destination):
    command = [
        sys.executable,
        str(MACOS / "sam31/capture_realsense.py"),
        "--output",
        str(destination),
    ]
    for name in ("width", "height", "fps", "warmup_frames"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.serial:
        command += ["--serial", args.serial]
    if platform.system() == "Darwin":
        # Refresh authentication per frame so an expired sudo timestamp does not
        # interrupt a long interactive session. The controller/SSH remain user-owned.
        subprocess.run(["sudo", "-p", CAMERA_PASSWORD_PROMPT, "-v"], check=True)
        command = ["sudo", "-n", *command]
    subprocess.run(command, check=True, cwd=ROOT)


def interact(args, worker, session, read=input):
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
        try:
            capture_frame(args, frame / "capture")
        except subprocess.CalledProcessError:
            print(
                "[sam31] Capture failed; retry with Enter or change the prompt.",
                file=sys.stderr,
            )
            continue
        response = worker.predict(prompt, frame / "capture/rgbd.npz")
        if response.get("type") == "error":
            print("[sam31] Inference failed: " + response["message"], file=sys.stderr)
            continue
        summary = unpack_result(response, frame / "result")
        print(
            f"[sam31] {len(summary['objects'])} objects; {summary['inference_seconds']:.3f}s; {frame / 'result/overlay.png'}",
            flush=True,
        )
        if platform.system() == "Darwin" and not args.no_open:
            subprocess.run(["open", str(frame / "result/overlay.png")], check=False)


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
    if not 0 <= args.threshold <= 1 or not 0 < args.timeout < float("inf"):
        parser.error("threshold must be 0..1 and timeout must be positive and finite")
    if min(args.width, args.height, args.fps, args.warmup_frames) < 1:
        parser.error("camera dimensions, fps and warmup-frames must be positive")
    worker = None
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
        subprocess.run(
            [sys.executable, str(MACOS / "bootstrap_realsense.py")], check=True
        )
        if platform.system() == "Darwin":
            print(
                "[sam31] RealSense撮影のため、このMacのパスワードを入力してください。",
                flush=True,
            )
            subprocess.run(["sudo", "-p", CAMERA_PASSWORD_PROMPT, "-v"], check=True)
        session = args.output.resolve() / (
            datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        )
        session.mkdir(parents=True)
        log = (session / "dgx.log").open("w", encoding="utf-8")
        print(f"[sam31] DGX model diagnostics: {session / 'dgx.log'}", flush=True)
        worker.start(stderr=log)
        print(f"[sam31] Ready. Results: {session}", flush=True)
        interact(args, worker, session)
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
        if worker is not None and not worker.close():
            status = 1
        if log is not None:
            log.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
