"""ASR engines. 16 kHz mono float32 in, text out.

Engine は models.yaml の `engine` 名で選ぶ。新しいモデルを足すときは
ASREngine を継承したクラスを書いて ENGINES に登録する。
"""

from __future__ import annotations

import copy

import numpy as np

from .audio_io import SAMPLE_RATE
from .model_registry import ModelEntry, Registry


class ASREngine:
    #: True なら hotwords を実際に認識へ反映できる
    supports_hotwords = False

    def __init__(self, entry: ModelEntry):
        self.entry = entry

    @property
    def tag(self) -> str:
        return self.entry.tag

    def transcribe(self, audio: np.ndarray, language: str, hotwords: list[str]) -> str:
        """`language` は models.yaml の languages のキー ("en" | "ja" | "ja-qwen")."""
        raise NotImplementedError

    def warmup(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), "en", [])


def without_cuda_graphs(model) -> None:
    """Decode without NeMo's CUDA graphs.

    NeMo compiles their kernels with the image's NVRTC, and a driver older than the
    image's CUDA rejects the result. Set in the config, so that a later
    change_decoding_strategy (Haas's boosting makes one) keeps it.
    """
    from omegaconf import open_dict

    decoding = copy.deepcopy(model.cfg.decoding)
    with open_dict(decoding):
        decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding, verbose=False)


class NemoEngine(ASREngine):
    """NVIDIA Parakeet (FastConformer-TDT) via NeMo."""

    # TODO(speech): NeMo GPU-PB (boosting_tree) で hotwords を反映し supports_hotwords を True にする。

    def __init__(self, entry: ModelEntry, device: str):
        super().__init__(entry)
        import torch
        from nemo.collections.asr.models import ASRModel

        self.model = ASRModel.restore_from(
            str(entry.fetch_file()), map_location=torch.device(device)
        )
        self.model.eval()
        without_cuda_graphs(self.model)

    def transcribe(self, audio, language, hotwords):
        import torch

        with torch.inference_mode():
            out = self.model.transcribe([audio], batch_size=1, verbose=False)
        # NeMo returns list[Hypothesis] (or list[str] on older versions); TDT may nest.
        hyp = out[0]
        if isinstance(hyp, list):
            hyp = hyp[0]
        return (getattr(hyp, "text", hyp) or "").strip()


class Qwen3AsrEngine(ASREngine):
    """Qwen3-ASR via the qwen-asr package (transformers backend)."""

    supports_hotwords = True

    def __init__(self, entry: ModelEntry, device: str):
        super().__init__(entry)
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            str(entry.fetch()),
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio, language, hotwords):
        # Qwen3-ASR は system prompt (context) に書いた語彙へ寄せて認識する。
        context = ""
        if hotwords:
            context = "Words that may appear: " + ", ".join(hotwords)
        results = self.model.transcribe(
            audio=(audio, SAMPLE_RATE),
            context=context,
            language=self.entry.spec["language_names"].get(language),
        )
        return (results[0].text or "").strip()


class HaasEngine(ASREngine):
    """Haas-ja via the haas-asr package (Linux + CUDA only: its decoder needs Triton)."""

    supports_hotwords = True

    def __init__(self, entry: ModelEntry, device: str):
        super().__init__(entry)
        import haas_asr

        self.haas_asr = haas_asr
        self.model = haas_asr.load_model(entry.fetch_file(), device=device)
        without_cuda_graphs(self.model)

    def transcribe(self, audio, language, hotwords):
        # A hotword may give its readings: "Haas=ハース".
        vocabulary = self.haas_asr.Vocabulary.parse(hotwords) if hotwords else None
        return self.haas_asr.transcribe(self.model, [audio], vocabulary=vocabulary)[0]


ENGINES: dict[str, type[ASREngine]] = {
    "nemo": NemoEngine,
    "qwen3_asr": Qwen3AsrEngine,
    "haas": HaasEngine,
}


def load_engine(registry: Registry, key: str, device: str, logger) -> ASREngine:
    entry = registry.entry("asr", key)
    cls = ENGINES[entry.spec["engine"]]
    logger.info(f"loading ASR {entry.tag} ({cls.__name__}) on {device} ...")
    with registry.loading(entry, device) as extra:
        engine = cls(entry, device)
        engine.warmup()
        extra["hotwords"] = cls.supports_hotwords
    logger.info(
        f'loaded ASR {entry.tag} in {registry.loaded[key]["load_seconds"]:.1f}s'
    )
    return engine
