"""ASR engines. 16 kHz mono float32 in, text out.

Engine は models.yaml の `engine` 名で選ぶ。新しいモデルを足すときは
ASREngine を継承したクラスを書いて ENGINES に登録する。
"""

from __future__ import annotations

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
        """`language` は models.yaml の languages のキー ("en" | "ja")."""
        raise NotImplementedError

    def warmup(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), 'en', [])


class NemoEngine(ASREngine):
    """NVIDIA Parakeet (FastConformer-TDT) via NeMo."""

    # TODO(speech): NeMo GPU-PB (boosting_tree) で hotwords を反映し supports_hotwords を True にする。

    def __init__(self, entry: ModelEntry, device: str):
        super().__init__(entry)
        import torch
        from nemo.collections.asr.models import ASRModel

        self.model = ASRModel.restore_from(str(entry.fetch_file()), map_location=torch.device(device))
        self.model.eval()

    def transcribe(self, audio, language, hotwords):
        import torch

        with torch.inference_mode():
            out = self.model.transcribe([audio], batch_size=1, verbose=False)
        # NeMo returns list[Hypothesis] (or list[str] on older versions); TDT may nest.
        hyp = out[0]
        if isinstance(hyp, list):
            hyp = hyp[0]
        return (getattr(hyp, 'text', hyp) or '').strip()


class Qwen3AsrEngine(ASREngine):
    """Qwen3-ASR via the qwen-asr package (transformers backend)."""

    supports_hotwords = True

    def __init__(self, entry: ModelEntry, device: str):
        super().__init__(entry)
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            str(entry.fetch()),
            dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio, language, hotwords):
        # Qwen3-ASR は system prompt (context) に書いた語彙へ寄せて認識する。
        context = ''
        if hotwords:
            context = 'Words that may appear: ' + ', '.join(hotwords)
        results = self.model.transcribe(
            audio=(audio, SAMPLE_RATE), context=context,
            language=self.entry.spec['language_names'].get(language))
        return (results[0].text or '').strip()


ENGINES: dict[str, type[ASREngine]] = {
    'nemo': NemoEngine,
    'qwen3_asr': Qwen3AsrEngine,
}


def load_engine(registry: Registry, key: str, device: str, logger) -> ASREngine:
    entry = registry.entry('asr', key)
    cls = ENGINES[entry.spec['engine']]
    logger.info(f'loading ASR {entry.tag} ({cls.__name__}) on {device} ...')
    with registry.loading(entry, device) as extra:
        engine = cls(entry, device)
        engine.warmup()
        extra['hotwords'] = cls.supports_hotwords
    logger.info(f'loaded ASR {entry.tag} in {registry.loaded[key]["load_seconds"]:.1f}s')
    return engine
