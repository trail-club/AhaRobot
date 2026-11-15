"""Run one model over every stream as a streaming detector would.

Window k of a stream ends at (k + 1) * HOP s and is `--window` s long (zeros
before the stream start), so its score uses only audio up to that time. Positive
streams are scored from POS_BEFORE s before the onset to POS_AFTER s after it;
`first` holds each stream's first window index.

  tagger  scores/<model>/w<W>.npz: probabilities of every class (float16)
  frame   scores/<model>/frames.npz: 10 s windows; per class the max over the
          frames (40 ms) of the last 0.2 s, of the last 1 s, and of the 0.2 s
          ending 0.4 s before the window end (FRAME_VIEWS)
  clap    scores/<model>/w<W>.npz: cosine similarity to every prompt
  yamnet  scores/yamnet/patch.npz: the 0.975 s patch ending at each hop
          (longer windows are the max over patches, see evaluate.py)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import zoo
from common import (
    HOP,
    STORE_RATE,
    data_root,
    load_audio,
    read_jsonl,
    resample,
    results_root,
)

FRAME_VIEWS = {
    "last0.2": (-5, None),
    "last1": (-25, None),
    "delay0.4": (-15, -10),
}  # frame slices
POS_BEFORE = 2.0  # s of a positive stream scored before the onset
POS_AFTER = 4.6  # s scored after it (the hit window plus margin)


class Streams(IterableDataset):
    """Yields (index, stream resampled to the model rate); workers split the streams."""

    def __init__(self, rows, rate):
        self.rows, self.rate = rows, rate

    def __iter__(self):
        info = get_worker_info()
        wid, n = (info.id, info.num_workers) if info else (0, 1)
        root = data_root()
        for i in range(wid, len(self.rows), n):
            row = self.rows[i]
            x = load_audio(root / "streams" / row["split"] / f"{row['id']}.npy")
            yield i, resample(x, STORE_RATE, self.rate)


def window_range(row: dict, count: int) -> range:
    """Windows to score: all of a negative stream; from POS_BEFORE s before the onset
    to POS_AFTER s after it in a positive one (earlier windows cannot change a hit)."""
    if row["kind"] != "pos":
        return range(count)
    lo = max(0, int(np.floor((row["onset"] - POS_BEFORE) / HOP)))
    return range(lo, min(count, int(np.ceil((row["onset"] + POS_AFTER) / HOP))))


def run(model, rows, args, first):
    n_rows = len(rows)
    out = [None] * n_rows
    loader = DataLoader(
        Streams(rows, model.rate),
        batch_size=None,
        num_workers=args.workers,
        prefetch_factor=2,
    )
    seconds = {"frame": 10.0}.get(model.kind, args.window)
    if isinstance(model, zoo.Yamnet):
        seconds = model.patch / model.rate
    done = 0
    for i, x in loader:
        ks = window_range(rows[i], len(x) // round(HOP * model.rate))
        first[i] = ks.start
        probs = model.predict_stream(
            torch.as_tensor(x).to(model.device), seconds, HOP, args.batch, ks
        )
        if model.kind == "frame":
            out[i] = {
                f"frames_{name}": probs[:, lo:hi]
                .amax(dim=1)
                .cpu()
                .numpy()
                .astype(np.float16)
                for name, (lo, hi) in FRAME_VIEWS.items()
            }
        elif model.kind == "clap":
            out[i] = {"sims": probs.cpu().numpy().astype(np.float16)}
        else:
            out[i] = {"probs": probs.cpu().numpy().astype(np.float16)}
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{n_rows} streams", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=sorted(zoo.REGISTRY))
    ap.add_argument("--window", type=float, default=2.0, help="s (taggers and CLAP)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=0, help="first N streams only (smoke test)"
    )
    ap.add_argument(
        "--skip-hard",
        action="store_true",
        help="leave out the hard_chime / hard_knock negatives (slow models)",
    )
    args = ap.parse_args()

    rows = read_jsonl(data_root() / "streams.jsonl")
    if args.skip_hard:
        rows = [r for r in rows if r["kind"] == "pos" or r["group"] == "ambient"]
    if args.limit:
        rows = rows[: args.limit]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = zoo.load(args.model, "cuda")
    t0 = time.time()
    first = np.zeros(len(rows), int)
    out = run(model, rows, args, first)
    elapsed = time.time() - t0

    if model.kind == "frame":
        name = "frames"
    elif isinstance(model, zoo.Yamnet):
        name = "patch"
    else:
        name = f"w{args.window:g}"
    path = results_root() / "scores" / args.model / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = out[0].keys()
    counts = np.array([len(next(iter(o.values()))) for o in out])
    arrays = {k: np.concatenate([o[k] for o in out]) for k in keys}
    np.savez(
        path,
        ids=np.array([r["id"] for r in rows]),
        counts=counts,
        first=first,
        labels=np.array(model.labels),
        logit_scale=getattr(model, "logit_scale", 0.0),
        elapsed=elapsed,
        **arrays,
    )
    hours = sum(r["duration"] for r in rows) / 3600
    print(
        f"{args.model} {name}: {counts.sum()} windows, {hours:.1f} h audio in {elapsed:.0f} s -> {path}"
    )


if __name__ == "__main__":
    main()
