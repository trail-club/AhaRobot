"""ASR / TTS inference worker on a GPU host, speaking JSON lines over SSH stdio.

Requests are handled one at a time and each response repeats the `id`. Audio is
base64 of little-endian float32 mono samples. Nothing is written to disk.

    <- {"type": "ready", "device": ..., "packages": {...},
        "tts": {"voices": [...], "default_voice", "sample_rate", "tag", "record"}}
    -> {"type": "load", "id": 1, "key": "parakeet_en"}
    <- {"type": "loaded", "id": 1, "supports_hotwords": false, "record": {...}}
    -> {"type": "transcribe", "id": 2, "key": ..., "audio": <16 kHz>,
        "language": "en", "hotwords": []}
    <- {"type": "transcript", "id": 2, "text": "...", "seconds": s}
    -> {"type": "synthesize", "id": 3, "text": "...", "voice": "af_heart"}
    <- {"type": "audio", "id": 3, "audio": <sample_rate>, "sample_rate": 24000,
        "seconds": s}
    <- {"type": "error", "id": n, "message": "..."}
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

from ..worker_io import protocol_stream, safe_error, send_json
from .asr_engines import load_engine
from .audio_io import decode_pcm, encode_pcm
from .model_registry import Registry, package_versions
from .tts_engine import KokoroTts


class Inference:
    """The models of one worker; ASR models load on request, TTS at start."""

    def __init__(
        self,
        registry: Registry,
        device: str,
        roles: list[str],
        load_asr=load_engine,
        load_tts=KokoroTts,
    ):
        self.registry = registry
        self.device = device
        self.load_asr = load_asr
        self.asr_allowed = "asr" in roles
        self.engines = {}
        self.tts = load_tts(registry, device) if "tts" in roles else None

    def ready(self) -> dict:
        response = {
            "type": "ready",
            "device": self.device,
            "packages": package_versions(),
        }
        if self.tts is not None:
            response["tts"] = {
                "voices": list(self.tts.voices),
                "default_voice": self.tts.default_voice,
                "sample_rate": self.tts.sample_rate,
                "tag": self.tts.tag,
                "record": self.registry.loaded["kokoro"],
            }
        return response

    def _engine(self, key: str):
        if not self.asr_allowed:
            raise ValueError("this worker was started without ASR")
        if key not in self.engines:
            self.engines[key] = self.load_asr(
                self.registry, key, self.device, logging.getLogger("audio_worker")
            )
        return self.engines[key]

    def load(self, request: dict) -> dict:
        engine = self._engine(request["key"])
        return {
            "type": "loaded",
            "supports_hotwords": engine.supports_hotwords,
            "record": self.registry.loaded[request["key"]],
        }

    def transcribe(self, request: dict) -> dict:
        engine = self._engine(request["key"])
        started = time.perf_counter()
        text = engine.transcribe(
            decode_pcm(request["audio"]), request["language"], request["hotwords"]
        )
        return {
            "type": "transcript",
            "text": text,
            "seconds": time.perf_counter() - started,
        }

    def synthesize(self, request: dict) -> dict:
        if self.tts is None:
            raise ValueError("this worker was started without TTS")
        started = time.perf_counter()
        chunks = list(self.tts.synthesize(request["text"], request["voice"]))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return {
            "type": "audio",
            "audio": encode_pcm(audio),
            "sample_rate": self.tts.sample_rate,
            "seconds": time.perf_counter() - started,
        }

    def handle(self, request: dict) -> dict:
        handlers = {
            "load": self.load,
            "transcribe": self.transcribe,
            "synthesize": self.synthesize,
        }
        kind = request.get("type")
        if kind not in handlers:
            raise ValueError(f"unknown request type {kind!r}")
        return handlers[kind](request)


def serve(input_stream, output_stream, load_inference) -> int:
    def send(response):
        send_json(output_stream, response)

    try:
        try:
            with redirect_stdout(sys.stderr):
                inference = load_inference()
        except Exception as exc:
            send({"type": "error", "message": safe_error(exc)})
            return 1
        send(inference.ready())
        for line in input_stream:
            # One bad request must not discard the loaded models.
            request = None
            try:
                request = json.loads(line)
                with redirect_stdout(sys.stderr):
                    response = {**inference.handle(request), "id": request.get("id")}
            except Exception as exc:
                request_id = request.get("id") if isinstance(request, dict) else None
                response = {
                    "type": "error",
                    "id": request_id,
                    "message": safe_error(exc),
                }
            send(response)
        return 0
    except BrokenPipeError:
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-file", type=Path, required=True)
    parser.add_argument("--roles", default="asr,tts", help="comma-separated: asr, tts")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    return serve(
        sys.stdin,
        protocol_stream(),
        lambda: Inference(
            Registry.load(str(args.models_file)), args.device, args.roles.split(",")
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
