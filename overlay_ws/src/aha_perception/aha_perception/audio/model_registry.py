"""Model registry: config/audio/models.yaml を読み、台帳どおりの版を取得・記録する."""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_registry_path() -> str:
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(
        get_package_share_directory("aha_perception"), "config", "audio", "models.yaml"
    )


@dataclass
class ModelEntry:
    key: str
    kind: str  # "asr" | "vad" | "turn" | "tts"
    spec: dict[str, Any]

    @property
    def repo(self) -> str | None:
        return self.spec.get("repo")

    @property
    def revision(self) -> str | None:
        return self.spec.get("revision")

    @property
    def tag(self) -> str:
        """ "<repo>@<revision[:7]>" or "<package>==<installed version>" for pip-bundled models."""
        if self.repo:
            return f'{self.repo}@{(self.revision or "main")[:7]}'
        package = self.spec["package"]
        return f"{package}=={importlib.metadata.version(package)}"

    def fetch(self, allow_patterns: list[str] | None = None) -> Path:
        """Download (or reuse the HF cache of) exactly the pinned revision."""
        from huggingface_hub import snapshot_download

        if not self.repo:
            raise ValueError(f"{self.key} has no repo to fetch")
        if not self.revision:
            raise ValueError(f"{self.key}: revision must be pinned in models.yaml")
        if allow_patterns is None and self.spec.get("file"):
            allow_patterns = [self.spec["file"]]
        return Path(
            snapshot_download(
                repo_id=self.repo, revision=self.revision, allow_patterns=allow_patterns
            )
        )

    def fetch_file(self) -> Path:
        """Fetch the single file named by `file` in models.yaml and return its path."""
        return self.fetch() / self.spec["file"]


@dataclass
class Registry:
    path: str
    raw: dict[str, Any]
    loaded: dict[str, dict[str, Any]] = field(default_factory=dict)
    packages: dict[str, str] | None = (
        None  # 別ホストの推論ワーカーの版 (None: このプロセス)
    )

    @classmethod
    def load(cls, path: str | None = None) -> "Registry":
        import yaml

        path = path or default_registry_path()
        with open(path, encoding="utf-8") as f:
            return cls(path=path, raw=yaml.safe_load(f))

    def entry(self, kind: str, key: str) -> ModelEntry:
        try:
            return ModelEntry(key=key, kind=kind, spec=self.raw[kind][key])
        except KeyError as e:
            raise KeyError(f"{kind}.{key} is not in {self.path}") from e

    def language(self, lang: str) -> dict[str, Any]:
        try:
            return self.raw["languages"][lang]
        except KeyError as e:
            raise KeyError(
                f'language "{lang}" is not in {self.path} (known: {", ".join(self.languages)})'
            ) from e

    def language_code(self, lang: str) -> str:
        """The language itself ("ja" for "ja-qwen"), for SpeechTranscript.language."""
        return self.language(lang).get("code", lang)

    @property
    def languages(self) -> list[str]:
        return list(self.raw.get("languages", {}))

    @contextlib.contextmanager
    def loading(self, entry: ModelEntry, device: str):
        """Time the load in the `with` body and record it. The yielded dict adds extra fields."""
        extra: dict[str, Any] = {}
        t0 = time.monotonic()
        yield extra
        self.loaded[entry.key] = {
            "kind": entry.kind,
            "tag": entry.tag,
            "repo": entry.repo,
            "revision": entry.revision,
            "file": entry.spec.get("file"),
            "license": entry.spec.get("license"),
            "device": device,
            "load_seconds": round(time.monotonic() - t0, 2),
            **extra,
        }

    def runtime_record(self, node_name: str) -> str:
        """JSON published on /aha/perception/audio/models so the running set is always visible."""
        return json.dumps(
            {
                "node": node_name,
                "registry": self.path,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "models": self.loaded,
                "packages": self.packages or package_versions(),
            },
            ensure_ascii=False,
        )


@functools.cache
def package_versions() -> dict[str, str]:
    out = {}
    for pkg in (
        "torch",
        "nemo_toolkit",
        "qwen-asr",
        "transformers",
        "kokoro",
        "misaki",
        "silero-vad",
        "onnxruntime",
    ):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out
