"""ASR / TTS on a GPU host: the worker container and engines that call it.

RemoteEngine and RemoteTts have the interfaces of ASREngine and KokoroTts, so the
nodes and the Mac tool use them like the in-process models. Speech goes over SSH
to the chosen host. No ROS dependency.
"""

from __future__ import annotations

from pathlib import Path

from ..dgx import ContainerSpec, DgxWorker, WorkerOptions
from .asr_engines import ASREngine
from .audio_io import decode_pcm, encode_pcm
from .model_registry import Registry

SSH_PORT = WorkerOptions.port
# target: SSH address over Cloudflare One (WARP)
TARGETS = {"dgx": WorkerOptions.host, "trailpc": "10.99.0.2"}
LOCAL = "local"  # models in this process, no worker

PACKAGE = "overlay_ws/src/aha_perception"
MODELS_FILE = f"{PACKAGE}/config/audio/models.yaml"
WORKER_DIR = "tools/perception/audio_worker"
DOCKERFILE = f"{WORKER_DIR}/Dockerfile"
CACHE = "aharobot-audio"


def audio_sources(root: Path) -> list[Path]:
    package = root / PACKAGE / "aha_perception"
    files = [package / "__init__.py", package / "worker_io.py"]
    files += sorted((package / "audio").glob("*.py"))
    files += [root / MODELS_FILE, root / DOCKERFILE]
    return files + [root / WORKER_DIR / name for name in ("pyproject.toml", "uv.lock")]


def audio_spec(target: str, roles: list[str], device: str) -> ContainerSpec:
    return ContainerSpec(
        title=f"{target} audio worker",
        name="audio",
        dockerfile=DOCKERFILE,
        image="local/aharobot-audio-worker",
        sources=audio_sources,
        prepare="",
        cache=CACHE,
        docker_args=(),
        command=(
            "-m",
            "aha_perception.audio.worker",
            "--models-file",
            f"/app/{MODELS_FILE}",
            "--roles",
            ",".join(roles),
            "--device",
            device,
        ),
    )


def make_worker(
    target: str,
    roles: list[str],
    *,
    device: str = "cuda:0",
    host: str = "",
    port: int = SSH_PORT,
    user: str = "",
    identity_file: str = "",
    log=print,
    command=None,
) -> DgxWorker:
    """An unstarted worker. `host` overrides the target's address (e.g. an SSH alias)."""
    if target not in TARGETS:
        raise ValueError(f'unknown target "{target}" (known: {", ".join(TARGETS)})')
    options = WorkerOptions(
        host=host or TARGETS[target], port=port, user=user, identity_file=identity_file
    )
    return DgxWorker(
        options, spec=audio_spec(target, roles, device), log=log, command=command
    )


class RemoteEngine(ASREngine):
    def __init__(
        self, entry, worker: DgxWorker, supports_hotwords: bool, timeout: float
    ):
        super().__init__(entry)
        self.worker = worker
        self.supports_hotwords = supports_hotwords
        self.timeout = timeout

    def transcribe(self, audio, language, hotwords):
        return self.worker.request(
            {
                "type": "transcribe",
                "key": self.entry.key,
                "audio": encode_pcm(audio),
                "language": language,
                "hotwords": hotwords,
            },
            self.timeout,
        )["text"]


def load_remote_engine(
    registry: Registry,
    key: str,
    worker: DgxWorker,
    target: str,
    load_timeout: float,
    timeout: float,
    logger,
) -> RemoteEngine:
    entry = registry.entry("asr", key)
    logger.info(f"loading ASR {entry.tag} on {target} ...")
    reply = worker.request({"type": "load", "key": key}, load_timeout)
    record = reply["record"]
    registry.loaded[key] = {**record, "device": f'{target}:{record["device"]}'}
    logger.info(f'loaded ASR {entry.tag} in {record["load_seconds"]:.1f}s')
    return RemoteEngine(entry, worker, reply["supports_hotwords"], timeout)


class RemoteTts:
    def __init__(
        self, worker: DgxWorker, registry: Registry, target: str, timeout: float
    ):
        info = worker.info["tts"]
        record = info["record"]
        registry.loaded["kokoro"] = {**record, "device": f'{target}:{record["device"]}'}
        self.worker = worker
        self.timeout = timeout
        self.voices = dict.fromkeys(info["voices"])
        self.default_voice = info["default_voice"]
        self.sample_rate = int(info["sample_rate"])
        self.tag = info["tag"]

    def synthesize(self, text: str, voice: str):
        """Yield the audio of `text`: one chunk, since a request carries one sentence."""
        reply = self.worker.request(
            {"type": "synthesize", "text": text, "voice": voice}, self.timeout
        )
        audio = decode_pcm(reply["audio"])
        if len(audio):
            yield audio
