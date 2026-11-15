"""Start a worker on a GPU host over SSH and exchange requests with it.

Follows tools/perception/macos/sam31 (interactive.py): send the model code and
locked build inputs, build the Dockerfile, run one container, and remove the
container and sources on close. A ContainerSpec says what to send and run; the
default is person_worker on the DGX Spark, with the read-only SAM 3.1 weights
and the DGX host network to reach the Qwen3.8 server there.
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
from typing import Callable
import uuid

from .pose_keypoints import DEFAULT_POSE_MODEL
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
PERSON_DOCKERFILE = "tools/perception/macos/sam31/Dockerfile"
PERSON_BUILD_INPUTS = [PERSON_DOCKERFILE] + [
    f"tools/perception/macos/sam31/{name}" for name in ("pyproject.toml", "uv.lock")
]


def repository_root(dockerfile: str) -> Path:
    """The checkout this package was built from (requires --symlink-install)."""
    root = Path(__file__).resolve().parents[4]
    if not (root / dockerfile).is_file():
        raise RuntimeError(
            f"{root} is not the AhaRobot checkout; build aha_perception with "
            "--symlink-install or set the repository_root parameter"
        )
    return root


def source_archive(files: list[Path], root: Path, destination) -> None:
    """Exactly the listed files; never credentials or caches."""
    with tarfile.open(fileobj=destination, mode="w") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    destination.seek(0)


CACHE_DIRS = ("torch/kernels", "triton", "huggingface")


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
    pose_model: str = DEFAULT_POSE_MODEL  # Hugging Face ViTPose; empty disables


@dataclass(frozen=True)
class ContainerSpec:
    title: str  # in messages, e.g. "DGX person worker"
    name: str  # aharobot-<name>: the container and temporary directory
    dockerfile: str
    image: str
    sources: Callable[[Path], list[Path]]  # files to send, under the repository root
    prepare: str  # shell run on the host before the container starts
    cache: str  # under $HOME/.cache on the host; mounted at /cache
    docker_args: tuple[str, ...]  # for docker run, before the image
    command: tuple[str, ...]  # python arguments, after the image


def person_sources(root: Path) -> list[Path]:
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
    return files + [root / name for name in PERSON_BUILD_INPUTS]


def person_spec(options: WorkerOptions) -> ContainerSpec:
    return ContainerSpec(
        title="DGX person worker",
        name="people",
        dockerfile=PERSON_DOCKERFILE,
        image=options.image,
        sources=person_sources,
        prepare=(f"test -r {shlex.quote(options.checkpoint)}"),
        cache="aharobot-sam31",
        docker_args=(
            # The Qwen3.8 server listens on the DGX's WARP address only.
            "--network",
            "host",
            "-e",
            "HF_HOME=/cache/huggingface",
            "-v",
            f"{options.checkpoint}:/models/sam3.1_multiplex.pt:ro",
        ),
        command=(
            "-m",
            "aha_perception.person_worker",
            "--checkpoint",
            "/models/sam3.1_multiplex.pt",
            "--threshold",
            str(options.threshold),
            "--prompt",
            options.prompt,
            "--vlm-url",
            options.vlm_url,
            "--vlm-model",
            options.vlm_model,
            "--pose-model",
            options.pose_model,
        ),
    )


class WorkerError(RuntimeError):
    pass


class RequestError(WorkerError):
    """The worker failed one request and keeps running."""


class DgxWorker:
    """One worker process; `request` may be called from several threads."""

    def __init__(
        self, options: WorkerOptions, *, spec=None, root=None, log=print, command=None
    ):
        self.options = options
        self.spec = spec or person_spec(options)
        self.root = root
        self.log = log
        self.command = command  # local stand-in for tests: argv of a worker
        self.name = f"aharobot-{self.spec.name}-{uuid.uuid4().hex}"
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
        # Without this, ssh would swallow what the user types during startup.
        kwargs.setdefault("stdin", subprocess.DEVNULL)
        return subprocess.run(self.ssh(script), check=True, **kwargs)

    def remote_command(self) -> list[str]:
        spec = self.spec
        self.remote_dir = self.run(
            f"mktemp -d /tmp/aharobot-{spec.name}-XXXXXXXX",
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not re.fullmatch(
            rf"/tmp/aharobot-{spec.name}-[A-Za-z0-9]+", self.remote_dir
        ):
            raise WorkerError("unexpected remote temporary directory")
        remote = shlex.quote(self.remote_dir)
        self.log("Syncing code and building the image (cached layers are reused)")
        root = self.root or repository_root(spec.dockerfile)
        with tempfile.TemporaryFile() as archive:
            source_archive(spec.sources(root), root, archive)
            self.run(f"tar -xf - -C {remote}", stdin=archive)
        # The tag is shared by host users; run the ID built from these sources.
        self.run(
            f"cd {remote} && docker build -q --iidfile .image-id "
            f"-t {shlex.quote(spec.image)} -f {spec.dockerfile} . >/dev/null",
        )
        image_id = self.run(
            f"cat {remote}/.image-id", capture_output=True, text=True
        ).stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise WorkerError("unexpected Docker image ID")
        cache = " ".join(f'"$HOME/.cache/{spec.cache}/{name}"' for name in CACHE_DIRS)
        self.run(" && ".join(filter(None, (spec.prepare, f"mkdir -p {cache}"))))
        args = [
            "--name",
            self.name,
            "--gpus",
            "all",
            "--shm-size=2g",
            *spec.docker_args,
            "-v",
            f"{self.remote_dir}:/app:ro",
            "--entrypoint",
            "python",
            image_id,
            "-u",
            *spec.command,
        ]
        # The SSH user's UID owns the caches; HF tokens are never transferred.
        script = (
            'docker run --rm -i --user "$(id -u):$(id -g)" '
            f'-v "$HOME/.cache/{spec.cache}:/cache" ' + shlex.join(args)
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
        self.log(f"Waiting for the {self.spec.title}")
        if not self.ready.wait(timeout):
            raise TimeoutError(f"{self.spec.title} did not become ready")
        if self.failure is not None:
            raise WorkerError(self.failure)
        return self.info

    def _read(self):
        title = self.spec.title
        message = f"{title} disconnected; check the SSH/Docker log"
        try:
            for line in self.process.stdout:
                try:
                    response = json.loads(line)
                except ValueError:
                    response = None
                if not isinstance(response, dict):
                    message = (
                        f"unexpected output from the {title}: {line.strip()[:200]!r}"
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
            message = f"{title} channel failed: {exc}"
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
                raise WorkerError(
                    f"cannot send to the {self.spec.title}: {exc}"
                ) from exc
            if not slot["event"].wait(timeout):
                raise TimeoutError(
                    f"{self.spec.title} did not answer within {timeout:.0f} s"
                )
        finally:
            with self.lock:
                self.pending.pop(request_id, None)
        response = slot["response"]
        if response is None:
            raise WorkerError(self.failure or f"{self.spec.title} stopped")
        if response.get("type") == "error":
            raise RequestError(response.get("message", f"{self.spec.title} error"))
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
                    f"Container cleanup failed: {exc}; run "
                    f"{shlex.join(self.ssh('docker rm -f ' + self.name))}"
                )
                success = False
        if self.remote_dir:
            try:
                self.run("rm -rf -- " + shlex.quote(self.remote_dir), timeout=30)
            except (OSError, subprocess.SubprocessError) as exc:
                self.log(f"Temporary directory cleanup failed: {exc}")
                success = False
        return success
