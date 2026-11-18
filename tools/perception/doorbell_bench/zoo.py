"""Pretrained models compared by the benchmark, behind one interface per kind.

  tagger  predict(batch of windows) -> (B, classes) probabilities
  frame   predict(batch of 10 s windows) -> (B, frames, classes) probabilities
  clap    predict(batch of windows) -> (B, prompts) cosine similarities

Weights come from the pinned revisions below and are cached under $HF_HOME /
$TORCH_HOME. Code that is not on PyPI (PretrainedSED, EfficientAT) is in
$THIRD_PARTY; both ship top-level `models` / `helpers` packages, so load only one
model per process.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", "/opt/third_party"))
CACHE = Path(os.environ.get("HOME", "/cache"))

HF = {
    "sslam": ("ta012/SSLAM_AS2M_Finetuned", "4500c5d989436b57b0621418fa49a54ca2ed8dff"),
    "eat_large": (
        "worstchan/EAT-large_epoch20_finetune_AS2M",
        "53b1251196e3dabb991dfa7629fa89394c7f8335",
    ),
    "ced_base": ("mispeech/ced-base", "db3e14a8db4c21b56b165261c39649741a900e7f"),
    "ced_mini": ("mispeech/ced-mini", "26c3ebcae85d4330f4fc26763f029539a3afcda0"),
    "clap_fused": (
        "laion/clap-htsat-fused",
        "365dea6ef167def6676140ed93bbc43f84dabb28",
    ),
    "clap_larger": (
        "laion/larger_clap_general",
        "ada0c23a36c4e8582805bb38fec3905903f18b41",
    ),
}
# BEATs iter3+ AS2M fine-tuned (cpt2). Microsoft's own link needs a SAS token; this
# mirror's file matches the sha256 of three other independent mirrors.
BEATS = (
    "AudioLLMs/SALMONN_7B",
    "b0f7c8581ff96dac823fb8e02c7303b33665e409",
    "beats_path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c",
)
EFFICIENTAT = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
YAMNET = (
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/"
    "audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite",
    "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de",
)

# CLAP prompts. A target's score sums the softmax over every prompt for its own
# prompts: "chime"/"knock" in the strict and door sets, plus the "_broad" ones in
# the tonal set (common.CLASS_SETS). The "other" prompts only compete.
PROMPTS = {
    "chime": (
        "a doorbell ringing",
        "the ding-dong chime of a doorbell",
        "an electronic doorbell chime",
        "a door buzzer",
    ),
    "chime_broad": (
        "an intercom chime",
        "a chime",
        "a bell ringing",
        "electronic beeps",
    ),
    "knock": ("knocking on a door", "someone knocking on a wooden door"),
    "knock_broad": ("tapping on wood",),
    "other": (
        "people talking",
        "a crowd of people chattering",
        "music playing",
        "a telephone ringing",
        "an alarm beeping",
        "a church bell",
        "a bicycle bell",
        "a microwave oven beeping",
        "wind chimes",
        "a piano",
        "a clock ticking",
        "footsteps",
        "a door closing",
        "dishes clattering",
        "applause",
        "background noise",
        "silence",
    ),
}
PROMPT_LIST = [p for group in PROMPTS.values() for p in group]


def audioset_names() -> list[str]:
    """The 527 AudioSet display names in the standard index order."""
    path = THIRD_PARTY / "EfficientAT" / "metadata" / "class_labels_indices.csv"
    with open(path) as f:
        return [row["display_name"] for row in csv.DictReader(f)]


def audioset_mid_to_name() -> dict[str, str]:
    path = THIRD_PARTY / "EfficientAT" / "metadata" / "class_labels_indices.csv"
    with open(path) as f:
        return {row["mid"]: row["display_name"] for row in csv.DictReader(f)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, name: str, digest: str | None = None) -> Path:
    path = CACHE / "weights" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path) + ".part", progress=False)
        os.replace(str(path) + ".part", path)
    if digest is not None and sha256(path) != digest:
        raise RuntimeError(f"sha256 mismatch for {path}")
    return path


@contextlib.contextmanager
def third_party(repo: str):
    """Import from a repo that reads/writes relative paths: run in a writable
    copy of its top level (symlinks) with its own `resources/` dir."""
    src = THIRD_PARTY / repo
    work = CACHE / "work" / repo
    work.mkdir(parents=True, exist_ok=True)
    (work / "resources").mkdir(exist_ok=True)
    for entry in src.iterdir():
        link = work / entry.name
        if entry.name != "resources" and not link.exists():
            link.symlink_to(entry)
    old = os.getcwd()
    os.chdir(work)
    sys.path.insert(0, str(work))
    try:
        yield work
    finally:
        os.chdir(old)


def kaldi_fbank(wav: torch.Tensor) -> torch.Tensor:
    """EAT/SSLAM features of one 16 kHz waveform: (frames, 128), one per 10 ms."""
    import torchaudio

    return torchaudio.compliance.kaldi.fbank(
        wav[None],
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )


class Model:
    kind = "tagger"
    rate = 16000
    labels: list[str]

    def __init__(self, device: str):
        self.device = device

    def predict(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_stream(
        self, x: torch.Tensor, seconds: float, hop: float, batch: int, ks: range
    ) -> torch.Tensor:
        """Outputs for windows k in `ks`: `seconds` long, ending at (k + 1) * hop (zeros before x)."""
        h, w = round(hop * self.rate), round(seconds * self.rate)
        padded = torch.cat([x.new_zeros(w), x])
        wins = padded[h:].unfold(0, w, h)[ks.start : ks.stop]
        return torch.cat(
            [self.predict(wins[b : b + batch]) for b in range(0, len(wins), batch)]
        )


class EatLike(Model):
    """EAT and SSLAM (same code path); input padded to 10 s as in their usage.

    `predict_stream` computes the fbank of the whole stream once and slices it per
    window, which skips the per-window mean removal of the reference code (kaldi
    removes each frame's DC offset anyway)."""

    def __init__(self, device, key):
        super().__init__(device)
        from transformers import AutoModel

        repo, rev = HF[key]
        self.model = (
            AutoModel.from_pretrained(repo, revision=rev, trust_remote_code=True)
            .eval()
            .to(device)
        )
        self.labels = audioset_names()

    @torch.inference_mode()
    def _forward(self, mel):
        mel = torch.nn.functional.pad(mel, (0, 0, 0, max(0, 1024 - mel.shape[1])))[
            :, :1024
        ]
        mel = ((mel + 4.268) / (4.569 * 2)).unsqueeze(1)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
        ):
            logits = self.model(mel)
        return torch.sigmoid(logits.float())

    @torch.inference_mode()
    def predict(self, wav):
        return self._forward(torch.stack([kaldi_fbank(w - w.mean()) for w in wav]))

    @torch.inference_mode()
    def predict_stream(self, x, seconds, hop, batch, ks):
        h, w = round(hop * 16000), round(seconds * 16000)
        mel = kaldi_fbank(torch.cat([x.new_zeros(w), x]))
        per_hop, frames = h // 160, (w - 400) // 160 + 1
        starts = (torch.arange(ks.start, ks.stop, device=x.device) + 1) * per_hop
        out = []
        for b in range(0, len(starts), batch):
            idx = starts[b : b + batch, None] + torch.arange(frames, device=x.device)
            out.append(self._forward(mel[idx]))
        return torch.cat(out)


class Ced(Model):
    def __init__(self, device, key):
        super().__init__(device)
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        repo, rev = HF[key]
        self.fe = AutoFeatureExtractor.from_pretrained(
            repo, revision=rev, trust_remote_code=True
        )
        self.model = (
            AutoModelForAudioClassification.from_pretrained(
                repo, revision=rev, trust_remote_code=True
            )
            .eval()
            .to(device)
        )
        cfg = self.model.config.id2label
        self.labels = [cfg[i] for i in range(len(cfg))]

    @torch.inference_mode()
    def predict(self, wav):
        inputs = self.fe(wav.cpu(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(
            **inputs
        ).logits.float()  # CED's head already applies the sigmoid


class Beats(Model):
    def __init__(self, device):
        super().__init__(device)
        from huggingface_hub import hf_hub_download

        repo, rev, name, digest = BEATS
        path = Path(hf_hub_download(repo, name, revision=rev))
        if sha256(path) != digest:
            raise RuntimeError("BEATs checkpoint sha256 mismatch")
        with third_party("PretrainedSED"):
            from models.beats.BEATs import BEATs, BEATsConfig
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model = BEATs(BEATsConfig(ckpt["cfg"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(device)
        mid = audioset_mid_to_name()
        self.labels = [
            mid[ckpt["label_dict"][i]] for i in range(len(ckpt["label_dict"]))
        ]

    @torch.inference_mode()
    def predict(self, wav):
        pad = torch.zeros(wav.shape, dtype=torch.bool, device=wav.device)
        probs, _ = self.model.extract_features(wav, padding_mask=pad)
        return probs.float()


class EfficientAT(Model):
    rate = 32000

    def __init__(self, device, name, file, width):
        super().__init__(device)
        with third_party("EfficientAT"):
            from models.mn.model import get_model
            from models.preprocess import AugmentMelSTFT

            self.model = get_model(width_mult=width, pretrained_name=None)
        state = torch.load(
            fetch(EFFICIENTAT + file, file), map_location="cpu", weights_only=False
        )
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.mel = (
            AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
            .eval()
            .to(device)
        )
        self.labels = audioset_names()
        self.name = name

    @torch.inference_mode()
    def predict(self, wav):
        with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            spec = self.mel(wav)
            logits, _ = self.model(spec.unsqueeze(1))
        return torch.sigmoid(logits.float()).reshape(
            len(wav), -1
        )  # MN squeezes a batch of 1


class Yamnet(Model):
    """TFLite classifier on 0.975 s patches (CPU). `predict` scores the last patch of
    each window; longer windows take the max over patches offline (score.py)."""

    patch = 15600

    def __init__(self, device):
        super().__init__("cpu")
        import zipfile

        from ai_edge_litert.interpreter import Interpreter

        path = fetch(*YAMNET[:1], "yamnet_classification.tflite", YAMNET[1])
        self.labels = (
            zipfile.ZipFile(path).read("yamnet_label_list.txt").decode().splitlines()
        )
        self.interp = Interpreter(model_path=str(path), num_threads=1)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]

    def predict(self, wav):
        rows = []
        for w in wav.cpu().numpy():
            self.interp.set_tensor(self.inp, w[-self.patch :].astype(np.float32))
            self.interp.invoke()
            rows.append(self.interp.get_tensor(self.out)[0].copy())
        return torch.from_numpy(np.stack(rows))


class FrameSed(Model):
    """PretrainedSED AudioSet-strong checkpoints: 10 s in, 250 frames (40 ms) out."""

    kind = "frame"
    seconds = 10.0
    frame_hop = 0.04

    def __init__(self, device, name):
        super().__init__(device)
        with third_party("PretrainedSED"):
            from data_util import audioset_classes
            from models.prediction_wrapper import PredictionsWrapper

            if name == "BEATs":
                from models.beats.BEATs_wrapper import BEATsWrapper

                self.model = PredictionsWrapper(
                    BEATsWrapper(), checkpoint="BEATs_strong_1"
                )
            elif name == "ATST-F":
                from models.atstframe.ATSTF_wrapper import ATSTWrapper

                self.model = PredictionsWrapper(
                    ATSTWrapper(), checkpoint="ATST-F_strong_1"
                )
            elif name.startswith("frame_mn"):
                from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
                from models.frame_mn.utils import NAME_TO_WIDTH

                inner = FrameMNWrapper(NAME_TO_WIDTH(name))
                dim = inner.state_dict()["frame_mn.features.16.1.bias"].shape[0]
                self.model = PredictionsWrapper(
                    inner, checkpoint=f"{name}_strong_1", embed_dim=dim
                )
            else:
                raise ValueError(name)
            self.labels = list(audioset_classes.as_strong_train_classes)
        self.model.eval().to(device)

    @torch.inference_mode()
    def predict(self, wav):
        mel = self.model.mel_forward(wav)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
        ):
            strong, _ = self.model(mel)
        return torch.sigmoid(strong.float()).transpose(1, 2)  # (B, frames, classes)


class Clap(Model):
    kind = "clap"
    rate = 48000

    def __init__(self, device, key):
        super().__init__(device)
        from transformers import ClapModel, ClapProcessor

        repo, rev = HF[key]
        self.processor = ClapProcessor.from_pretrained(repo, revision=rev)
        self.model = ClapModel.from_pretrained(repo, revision=rev).eval().to(device)
        self.labels = PROMPT_LIST
        with torch.inference_mode():
            text = self.processor(text=PROMPT_LIST, return_tensors="pt", padding=True)
            emb = self.model.get_text_features(
                **{k: v.to(device) for k, v in text.items()}
            )
            self.text = torch.nn.functional.normalize(emb.float(), dim=-1)
        self.logit_scale = float(self.model.logit_scale_a.exp())
        fe = self.processor.feature_extractor
        assert fe.truncation in ("fusion", "rand_trunc") and fe.padding == "repeatpad"
        self.fusion = fe.truncation == "fusion"
        self.max_samples = fe.nb_max_samples
        filters = fe.mel_filters if self.fusion else fe.mel_filters_slaney
        self.mel_fb = torch.from_numpy(filters).float().to(device)  # (513, 64)
        self.window = torch.hann_window(
            fe.fft_window_size, periodic=True, device=device
        )
        self.fft, self.hop_length = fe.fft_window_size, fe.hop_length

    def features(self, wav: torch.Tensor) -> dict:
        """ClapFeatureExtractor for windows shorter than 10 s, on the GPU: "repeatpad",
        then (fusion models) the htk mel stacked 4 times or (rand_trunc) one Slaney
        mel. Unlike the extractor, no window is randomly flagged `is_longer` when
        none of a batch is."""
        n = wav.shape[1]
        if n < self.max_samples:
            wav = wav.repeat(1, self.max_samples // n)
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - wav.shape[1]))
        else:
            wav = wav[:, -self.max_samples :]
        spec = torch.stft(
            wav,
            self.fft,
            self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mel = (spec.abs() ** 2).transpose(1, 2) @ self.mel_fb
        logmel = 10 * torch.log10(mel.clamp(min=1e-10))
        return {
            "input_features": logmel[:, None]
            .expand(-1, 4 if self.fusion else 1, -1, -1)
            .contiguous(),
            "is_longer": torch.zeros(len(wav), 1, dtype=torch.bool, device=wav.device),
        }

    @torch.inference_mode()
    def predict(self, wav):
        emb = self.model.get_audio_features(**self.features(wav))
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        return emb @ self.text.T


REGISTRY = {
    "yamnet": lambda d: Yamnet(d),
    "effat_mn40": lambda d: EfficientAT(d, "effat_mn40", "mn40_as_ext_mAP_487.pt", 4.0),
    "ced_mini": lambda d: Ced(d, "ced_mini"),
    "ced_base": lambda d: Ced(d, "ced_base"),
    "beats": lambda d: Beats(d),
    "eat_large": lambda d: EatLike(d, "eat_large"),
    "sslam": lambda d: EatLike(d, "sslam"),
    "sed_beats": lambda d: FrameSed(d, "BEATs"),
    "sed_atstf": lambda d: FrameSed(d, "ATST-F"),
    "sed_mn10": lambda d: FrameSed(d, "frame_mn10"),
    "clap_fused": lambda d: Clap(d, "clap_fused"),
    "clap_larger": lambda d: Clap(d, "clap_larger"),
}


def load(name: str, device: str = "cuda") -> Model:
    return REGISTRY[name](device)
