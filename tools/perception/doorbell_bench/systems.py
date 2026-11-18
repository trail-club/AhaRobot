"""Score files to per-stream chime / knock scores, by system spec.

Each plain spec ends with a class set of common.CLASS_SETS (strict / door / tonal):

  <model>:w<W>:<set>        tagger: max over the set's classes, W s window
  yamnet:w<W>:<set>         the same over the 0.975 s patches inside the last W s
  <sed>:<view>:<set>        frame model; view last0.2 / last1 (max over the frames
                            of the last 0.2 / 1 s) or delay0.4 (the 0.2 s ending
                            0.4 s before the hop)
  <clap>:w<W>:<set>         softmax over all prompts (CLAP logit scale), summed over
                            the set's prompts (zoo.PROMPTS)
  ens(<spec>,<spec>,...)    mean of the members' scores
  gate(<clap spec>,<t>,<spec>)  <spec>, but 0 where the CLAP score is below t
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import zoo
from common import CLASS_SETS, HOP, TARGETS, data_root, read_jsonl, results_root

SED_VIEWS = ("last0.2", "last1", "delay0.4")
SET_NAMES = sorted(
    {n for s in CLASS_SETS.values() for names in s.values() for n in names}
)


@lru_cache(maxsize=None)
def streams() -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(data_root() / "streams.jsonl")}


@lru_cache(maxsize=None)
def clip_index() -> dict[tuple[str, str], dict]:
    return {
        (c["split"], c["fname"]): c for c in read_jsonl(data_root() / "clips.jsonl")
    }


def score_path(model: str, name: str) -> Path:
    return results_root() / "scores" / model / f"{name}.npz"


@lru_cache(maxsize=None)
def load_npz(model: str, name: str, key: str):
    """(ids, bounds, first, column names, float32 columns) of one array in a score
    file, keeping only the classes some class set uses (all prompts for CLAP)."""
    z = np.load(score_path(model, name))
    labels = [str(x) for x in z["labels"]]
    keep = (
        range(len(labels))
        if key == "sims"
        else [labels.index(n) for n in SET_NAMES if n in labels]
    )
    cols = z[key][:, list(keep)].astype(np.float32)
    first = z["first"] if "first" in z.files else np.zeros(len(z["counts"]), int)
    bounds = np.concatenate([[0], np.cumsum(z["counts"])])
    extra = {"logit_scale": float(z["logit_scale"])}
    return (
        [str(s) for s in z["ids"]],
        bounds,
        first,
        [labels[i] for i in keep],
        cols,
        extra,
    )


def _split(arr, ids, bounds, first):
    """Per-stream arrays over every hop of the stream (unscored hops are 0)."""
    out = {}
    for i, sid in enumerate(ids):
        part = arr[bounds[i] : bounds[i + 1]]
        full = max(first[i] + len(part), int(streams()[sid]["duration"] / HOP + 1e-6))
        s = np.zeros(full, np.float32)
        s[first[i] : first[i] + len(part)] = part
        out[sid] = s
    return out


def _args(spec: str) -> list[str]:
    """Top-level comma-separated arguments of 'name(a,b(c,d),e)'."""
    inner, depth, cur, out = spec[spec.index("(") + 1 : -1], 0, "", []
    for ch in inner:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def _shift_max(s: np.ndarray, patches: int) -> np.ndarray:
    return np.max(
        [np.concatenate([np.zeros(j), s[: len(s) - j]]) for j in range(patches)], axis=0
    )


@lru_cache(maxsize=None)
def system(spec: str) -> dict[str, dict[str, np.ndarray]]:
    """{stream id: {"chime": scores, "knock": scores}} of a system spec."""
    if spec.startswith("ens("):
        members = [system(m) for m in _args(spec)]
        out = {}
        for sid in set.intersection(*(set(m) for m in members)):
            n = min(len(m[sid]["chime"]) for m in members)
            out[sid] = {
                t: np.mean([m[sid][t][:n] for m in members], axis=0) for t in TARGETS
            }
        return out
    if spec.startswith("gate("):
        clap_spec, tau, base_spec = _args(spec)
        clap, base = system(clap_spec), system(base_spec)
        out = {}
        for sid in set(base) & set(clap):
            n = min(len(base[sid]["chime"]), len(clap[sid]["chime"]))
            out[sid] = {
                t: np.where(clap[sid][t][:n] >= float(tau), base[sid][t][:n], 0.0)
                for t in TARGETS
            }
        return out
    model, variant, cset = spec.split(":")
    sets = CLASS_SETS[cset]
    if model.startswith("clap"):
        ids, bounds, first, labels, sims, extra = load_npz(model, variant, "sims")
        logits = sims * extra["logit_scale"]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        cols = {}
        for t in TARGETS:
            prompts = zoo.PROMPTS[t] + (
                zoo.PROMPTS[f"{t}_broad"] if cset == "tonal" else ()
            )
            cols[t] = p[:, [labels.index(q) for q in prompts]].sum(axis=1)
    else:
        if model == "yamnet":
            ids, bounds, first, labels, arr, _ = load_npz("yamnet", "patch", "probs")
        elif model.startswith("sed_"):
            ids, bounds, first, labels, arr, _ = load_npz(
                model, "frames", f"frames_{variant}"
            )
        else:
            ids, bounds, first, labels, arr, _ = load_npz(model, variant, "probs")
        cols = {
            t: arr[:, [labels.index(n) for n in sets[t] if n in labels]].max(axis=1)
            for t in TARGETS
        }
    per_target = {t: _split(cols[t], ids, bounds, first) for t in TARGETS}
    out = {sid: {t: per_target[t][sid] for t in TARGETS} for sid in ids}
    if model == "yamnet":
        patches = int(round((float(variant[1:]) - 0.975) / HOP)) + 1
        out = {
            sid: {t: _shift_max(s[t], patches) for t in TARGETS}
            for sid, s in out.items()
        }
    return out


def top_classes(spec: str, sid: str, k: int, top: int = 3) -> list[tuple[str, float]]:
    """Top classes at hop k of a stream from the first tagger in the spec."""
    for token in spec.replace("(", ",").replace(")", ",").split(","):
        parts = token.split(":")
        if (
            len(parts) == 3
            and parts[1].startswith("w")
            and not parts[0].startswith(("clap", "yamnet"))
        ):
            z = np.load(score_path(parts[0], parts[1]))
            ids = [str(s) for s in z["ids"]]
            j = ids.index(sid)
            bounds = np.concatenate([[0], np.cumsum(z["counts"])])
            first = z["first"] if "first" in z.files else np.zeros(len(ids), int)
            i = bounds[j] + k - first[j]
            if not bounds[j] <= i < bounds[j + 1]:
                return []
            p = z["probs"][i].astype(np.float32)
            return [
                (str(z["labels"][c]), round(float(p[c]), 2))
                for c in np.argsort(p)[::-1][:top]
            ]
    return []


def available() -> list[str]:
    """Every plain spec that has a score file."""
    specs = []
    for d in sorted(p for p in (results_root() / "scores").iterdir() if p.is_dir()):
        if d.name == "yamnet":
            variants = ["w1", "w2"]
        elif d.name.startswith("sed_"):
            variants = list(SED_VIEWS)
        else:
            variants = [f.stem for f in sorted(d.glob("w*.npz"))]
        specs += [f"{d.name}:{v}:{c}" for v in variants for c in CLASS_SETS]
    return specs
