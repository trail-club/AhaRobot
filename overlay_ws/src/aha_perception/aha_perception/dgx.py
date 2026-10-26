"""Start person_worker on the DGX Spark over SSH and exchange requests with it.

Follows tools/perception/macos/sam31 (interactive.py): send the model code and
locked build inputs, build the shared Dockerfile, run one container with the
read-only SAM 3.1 weights, and remove the container and sources on close.
The container uses the DGX host network to reach the Qwen3.8 server there.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile
import threading
import uuid

from .sam31_vlm import DEFAULT_VLM_MODEL, DEFAULT_VLM_URL


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
DOCKERFILE = "tools/perception/macos/sam31/Dockerfile"
BUILD_INPUTS = [DOCKERFILE] + [
    f"tools/perception/macos/sam31/{name}" for name in ("pyproject.toml", "uv.lock")
]


def repository_root() -> Path:
    """The checkout this package was built from (requires --symlink-install)."""
    root = Path(__file__).resolve().parents[4]
    if not (root / DOCKERFILE).is_file():
        raise RuntimeError(
            f"{root} is not the AhaRobot checkout; build aha_perception with "
            "--symlink-install or set the repository_root parameter"
        )
    return root


def source_archive(root: Path, destination) -> None:
    """Model code and locked build inputs only; never credentials or caches."""
    upstream = root / "upstream/sam3"
    listed = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(upstream), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    tracked = [name for name in listed.split("\0") if name]
    if "pyproject.toml" not in tracked:
        raise RuntimeError("initialize the upstream/sam3 submodule first")
    files = [upstream / name for name in tracked]
    files += sorted(
        (root / "overlay_ws/src/aha_perception/aha_perception").glob("*.py")
    )
    files += [root / name for name in BUILD_INPUTS]
    with tarfile.open(fileobj=destination, mode="w") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    destination.seek(0)


@dataclass
class WorkerOptions:
    host: str = "10.99.0.1"
    port: int = 2222  # 0: from the SSH config
    user: str = ""  # empty: from the SSH config
    identity_file: str = ""
    image: str = "local/aharobot-sam31"
    checkpoint: str = "/srv/shared/models/sam3.1/sam3.1_multiplex.pt"
    threshold: float = 0.5
    prompt: str = "person"
    vlm_url: str = DEFAULT_VLM_URL  # seen from the DGX; empty disables
    vlm_model: str = DEFAULT_VLM_MODEL


class WorkerError(RuntimeError):
    pass


class RequestError(WorkerError):
    """The worker failed one request and keeps running."""


class DgxWorker:
    """One worker process; `request` may be called from several threads."""

    def __init__(self, options: WorkerOptions, *, root=None, log=print, command=None):
        self.options = options
        self.root = root
        self.log = log
        self.command = command  # local stand-in for tests: argv of a worker
        self.name = "aharobot-people-" + uuid.uuid4().hex
        self.remote_dir = None
        self.process = None
        self.container_requested = False
        self.ids = itertools.count(1)
        self.lock = threading.Lock()  # pending
        # Separate: a large write blocks until the worker reads it, and the
        # worker may first be writing a response that _read hands over.
        self.write_lock = threading.Lock()
        self.pending: dict[int, dict] = {}
        self.ready = threading.Event()
        self.failure = None
        self.info = {}

    # -- SSH ---------------------------------------------------------------

    def ssh(self, script: str) -> list[str]:
        command = ["ssh", *SSH_OPTIONS]
        if self.options.port:
            command += ["-p", str(self.options.port)]
        if self.options.user:
            command += ["-l", self.options.user]
        if self.options.identity_file:
            command += ["-i", self.options.identity_file, "-o", "IdentitiesOnly=yes"]
        return command + ["--", self.options.host, script]

    def run(self, script: str, **kwargs):
        return subprocess.run(self.ssh(script), check=True, **kwargs)

    def remote_command(self) -> list[str]:
        o = self.options
        self.remote_dir = self.run(
            "mktemp -d /tmp/aharobot-people-XXXXXXXX", capture_output=True, text=True
        ).stdout.strip()
        if not re.fullmatch(r"/tmp/aharobot-people-[A-Za-z0-9]+", self.remote_dir):
            raise WorkerError("unexpected remote temporary directory")
        remote = shlex.quote(self.remote_dir)
        self.log("Syncing code and building the DGX image (cached layers are reused)")
        with tempfile.TemporaryFile() as archive:
            source_archive(self.root or repository_root(), archive)
            self.run(f"tar -xf - -C {remote}", stdin=archive)
        # The tag is shared by DGX users; run the ID built from these sources.
        self.run(
            f"cd {remote} && docker build -q --iidfile .image-id "
            f"-t {shlex.quote(o.image)} -f {DOCKERFILE} . >/dev/null",
        )
        image_id = self.run(
            f"cat {remote}/.image-id", capture_output=True, text=True
        ).stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise WorkerError("unexpected Docker image ID")
        self.run(
            f"test -r {shlex.quote(o.checkpoint)} && mkdir -p "
            '"$HOME/.cache/aharobot-sam31/torch/kernels" '
            '"$HOME/.cache/aharobot-sam31/triton" '
            '"$HOME/.cache/aharobot-sam31/huggingface"'
        )
        args = [
            "--name",
            self.name,
            "--gpus",
            "all",
            "--shm-size=2g",
            # The Qwen3.8 server listens on the DGX's WARP address only.
            "--network",
            "host",
            "-e",
            "HF_HOME=/cache/huggingface",
            "-v",
            f"{self.remote_dir}:/app:ro",
            "-v",
            f"{o.checkpoint}:/models/sam3.1_multiplex.pt:ro",
            "--entrypoint",
            "python",
            image_id,
            "-u",
            "-m",
            "aha_perception.person_worker",
            "--checkpoint",
            "/models/sam3.1_multiplex.pt",
            "--threshold",
            str(o.threshold),
            "--prompt",
            o.prompt,
            "--vlm-url",
            o.vlm_url,
            "--vlm-model",
            o.vlm_model,
        ]
        # The SSH user's UID owns the caches; HF tokens are never transferred.
        script = (
            'docker run --rm -i --user "$(id -u):$(id -g)" '
            '-v "$HOME/.cache/aharobot-sam31:/cache" ' + shlex.join(args)
        )
        self.container_requested = True
        return self.ssh(script)

    # -- Protocol ----------------------------------------------------------

    def start(self, timeout: float, stderr=None) -> dict:
        command = self.command or self.remote_command()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        threading.Thread(target=self._read, daemon=True).start()
        self.log("Loading SAM 3.1 on the DGX")
        if not self.ready.wait(timeout):
            raise TimeoutError("DGX person worker did not become ready")
        if self.failure is not None:
            raise WorkerError(self.failure)
        return self.info

    def _read(self):
        message = "DGX person worker disconnected; check the SSH/Docker log"
        try:
            for line in self.process.stdout:
                try:
                    response = json.loads(line)
                except ValueError:
                    response = None
                if not isinstance(response, dict):
                    message = (
                        f"unexpected output from the DGX worker: {line.strip()[:200]!r}"
                    )
                    break
                if response.get("type") == "ready":
                    self.info = response
                    self.ready.set()
                    continue
                with self.lock:
                    slot = self.pending.get(response.get("id"))
                if slot is not None:
                    slot["response"] = response
                    slot["event"].set()
                elif response.get("type") == "error" and not self.ready.is_set():
                    message = response.get("message", "worker failed to start")
                    break
        except (OSError, ValueError) as exc:
            message = f"DGX person worker channel failed: {exc}"
        self.failure = message
        self.ready.set()
        with self.lock:
            slots = list(self.pending.values())
        for slot in slots:
            slot["event"].set()

    def request(self, payload: dict, timeout: float) -> dict:
        if self.failure is not None:
            raise WorkerError(self.failure)
        slot = {"event": threading.Event(), "response": None}
        with self.lock:
            request_id = next(self.ids)
            self.pending[request_id] = slot
        try:
            try:
                with self.write_lock:
                    self.process.stdin.write(
                        json.dumps({**payload, "id": request_id}) + "\n"
                    )
                    self.process.stdin.flush()
            except OSError as exc:
                raise WorkerError(f"cannot send to the DGX worker: {exc}") from exc
            if not slot["event"].wait(timeout):
                raise TimeoutError(f"DGX worker did not answer within {timeout:.0f} s")
        finally:
            with self.lock:
                self.pending.pop(request_id, None)
        response = slot["response"]
        if response is None:
            raise WorkerError(self.failure or "DGX worker stopped")
        if response.get("type") == "error":
            raise RequestError(response.get("message", "DGX worker error"))
        return response

    def close(self) -> bool:
        success = True
        if self.process is not None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except OSError as exc:
                self.log(f"Local SSH cleanup: {exc}")
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        if self.container_requested:
            try:
                self.run(
                    "docker info >/dev/null 2>&1 && "
                    f"if docker container inspect {self.name} >/dev/null 2>&1; then "
                    f"docker rm -f {self.name} >/dev/null; fi",
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.log(
                    f"DGX container cleanup failed: {exc}; run "
                    f"{shlex.join(self.ssh('docker rm -f ' + self.name))}"
                )
                success = False
        if self.remote_dir:
            try:
                self.run("rm -rf -- " + shlex.quote(self.remote_dir), timeout=30)
            except (OSError, subprocess.SubprocessError) as exc:
                self.log(f"DGX temporary directory cleanup failed: {exc}")
                success = False
        return success
