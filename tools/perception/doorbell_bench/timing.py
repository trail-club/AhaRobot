"""GPU inference time of one window (batch 1), as the streaming detector runs it.

Prints and appends to <results>/timing.jsonl: median and p90 ms over `--runs`
calls after `--warmup` calls. Run with the GPU otherwise idle. YAMNet (TFLite)
always runs on the CPU.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import zoo
from common import results_root


def measure(model, seconds: float, runs: int, warmup: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    wav = torch.from_numpy(
        rng.normal(0, 0.05, int(seconds * model.rate)).astype(np.float32)
    )[None]
    wav = wav.to(model.device)
    times = []
    for i in range(warmup + runs):
        if model.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(wav)
        if model.device.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.quantile(times, 0.9))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=sorted(zoo.REGISTRY))
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    model = zoo.load(args.model, "cuda")
    seconds = 10.0 if model.kind == "frame" else args.window
    if isinstance(model, zoo.Yamnet):
        seconds = model.patch / model.rate
    median, p90 = measure(model, seconds, args.runs, args.warmup)
    net = getattr(model, "model", None)
    params = (
        sum(p.numel() for p in net.parameters())
        if isinstance(net, torch.nn.Module)
        else None
    )
    row = {
        "model": args.model,
        "device": model.device,
        "window": seconds,
        "median_ms": round(median, 1),
        "p90_ms": round(p90, 1),
        "params_m": round(params / 1e6, 1) if params else None,
    }
    print(json.dumps(row))
    with open(results_root() / "timing.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
