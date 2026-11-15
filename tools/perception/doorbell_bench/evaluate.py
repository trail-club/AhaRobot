"""Compare the detection systems on the scored streams and write the report.

A system is a spec of systems.py (a model, its window or frame view and class set,
or an ensemble / CLAP gate of such specs).

Post-processing: moving average over `avg` hops, then the score must stay at or
above the threshold for `n` consecutive hops. A detection starts when that becomes
true; one that starts less than REFRACTORY s after the previous start is the same
alarm.

Thresholds and post-processing come from the dev split. The main operating point
maximizes the gain, P(hit) * exp(-FA/h on ambient sound * WAIT / 3600): the chance
that the robot hears the bell after listening WAIT s without a false alarm (the
bell only rings while the robot waits at the start). The fixed budgets take the
lowest threshold within that many ambient alarms per hour on dev. Report numbers
are on the eval split unless named otherwise.

  hit       a detection between the onset and HIT s after it
  latency   time of that detection minus the onset (window end, without compute)
  FA/h      alarms per hour of negative streams (ambient / hard_chime / hard_knock)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from common import HOP, TARGETS, results_root
from systems import available, clip_index, streams, system, top_classes

HIT = 4.0  # s after the onset in which a detection counts
REFRACTORY = 5.0  # s
BUDGETS = (1.0, 5.0)  # alarms per hour on ambient negatives
WAIT = 60.0  # s of listening before the bell rings, for the "wait" operating point
POSTPROC = [(a, n) for a in (1, 2, 3) for n in (1, 2, 3)]
NEG_GROUPS = ("ambient", "hard_chime", "hard_knock")
AMBIENT_SUBGROUPS = ("speech", "music", "household")
CONDITIONS = ("clean", "snr10", "snr0", "room")
GATES = (0.05, 0.2)  # CLAP score thresholds tried by --auto


# ----------------------------------------------------------------------------- metrics
def postprocess(s: np.ndarray, avg: int, n: int) -> np.ndarray:
    """q[k] >= tau  <=>  the moving average stayed >= tau for the last n hops."""
    if avg > 1:
        c = np.cumsum(np.concatenate([np.zeros(1), s]))
        lo = np.maximum(np.arange(len(s)) + 1 - avg, 0)
        s = (c[1:] - c[lo]) / (np.arange(len(s)) + 1 - lo)
    if n > 1:
        s = np.min(
            [np.concatenate([np.full(j, -np.inf), s[: len(s) - j]]) for j in range(n)],
            axis=0,
        )
    return s


def alarms(q: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Alarms for each threshold: detection starts, merged within REFRACTORY."""
    above = q[None, :] >= taus[:, None]
    rising = above & ~np.concatenate(
        [np.zeros((len(taus), 1), bool), above[:, :-1]], axis=1
    )
    gap = int(round(REFRACTORY / HOP))
    c = np.concatenate(
        [np.zeros((len(taus), 1), int), np.cumsum(rising, axis=1)], axis=1
    )
    k = np.arange(rising.shape[1])
    recent = c[:, k] - c[:, np.maximum(k - gap, 0)]  # starts in the previous `gap` hops
    return (rising & (recent == 0)).sum(axis=1)


def latencies(q: np.ndarray, onset: float, taus: np.ndarray) -> np.ndarray:
    """Latency of the first detection start within HIT after the onset (nan = miss).

    Like the alarms, a detection is the start of a run above the threshold: a score
    that was already above before the onset does not count again."""
    t = (np.arange(len(q)) + 1) * HOP
    above = q[None, :] >= taus[:, None]
    rising = above & ~np.concatenate(
        [np.zeros((len(taus), 1), bool), above[:, :-1]], axis=1
    )
    sel = (t >= onset - 1e-6) & (t <= onset + HIT + 1e-6)
    rising = rising[:, sel]
    first = np.where(rising.any(axis=1), rising.argmax(axis=1), -1)
    return np.where(first >= 0, t[sel][np.maximum(first, 0)] - onset, np.nan)


@lru_cache(maxsize=None)
def evaluation(spec: str, target: str, avg: int, n: int) -> "Evaluation":
    return Evaluation(spec, target, avg, n)


class Evaluation:
    """Detections of one system, target and post-processing over all thresholds."""

    def __init__(self, spec: str, target: str, avg: int, n: int):
        self.spec, self.target, self.avg, self.n = spec, target, avg, n
        rows = streams()
        self.q = {
            sid: postprocess(s[target], avg, n) for sid, s in system(spec).items()
        }
        v = np.unique(np.round(np.concatenate(list(self.q.values())), 4))
        v = v[np.isfinite(v)]
        if len(v) > 300:
            v = np.unique(np.quantile(v, np.linspace(0, 1, 300)))
        self.taus = np.concatenate([v, [np.inf]])
        self.neg = defaultdict(lambda: np.zeros(len(self.taus)))
        self.hours = defaultdict(float)
        self.pos = defaultdict(list)  # split -> [(row, latency per tau)]
        for sid, q in self.q.items():
            row = rows[sid]
            if row["kind"] == "neg":
                count = alarms(q, self.taus)
                for key in (
                    (row["split"], row["group"]),
                    (row["split"], row["group"], row["subgroup"]),
                ):
                    self.neg[key] = self.neg[key] + count
                    self.hours[key] += row["duration"] / 3600
            elif row["target"] == target:
                self.pos[row["split"]].append(
                    (row, latencies(q, row["onset"], self.taus))
                )

    def fa(self, split: str, *group: str) -> np.ndarray:
        """Alarms per hour; nan for negatives the system was not run on."""
        hours = self.hours[(split, *group)]
        if hours == 0:
            return np.full(len(self.taus), np.nan)
        return self.neg[(split, *group)] / hours

    def recalls(self, split: str) -> np.ndarray:
        return np.mean([np.isfinite(lt) for _, lt in self.pos[split]], axis=0)

    def gain(self, split: str) -> np.ndarray:
        """P(hit) * P(no alarm on ambient sound during WAIT s), per threshold."""
        return self.recalls(split) * np.exp(-self.fa(split, "ambient") * WAIT / 3600)

    def index(self, point) -> int:
        """Threshold index of an operating point: "wait" (the highest dev gain)
        or a number (the lowest threshold within that many dev alarms per hour)."""
        if point == "wait":
            return int(np.argmax(self.gain("dev")))
        return int(np.flatnonzero(self.fa("dev", "ambient") <= point)[0])

    def objective(self, point) -> float:
        i = self.index(point)
        return float(self.gain("dev")[i] if point == "wait" else self.recalls("dev")[i])

    def recall(self, split: str, i: int, where=None) -> float:
        lat = [lt[i] for row, lt in self.pos[split] if where is None or where(row)]
        return float(np.mean(np.isfinite(lat))) if lat else float("nan")

    def summary(self, point, split: str = "eval") -> dict:
        i = self.index(point)
        lat = np.array([lt[i] for _, lt in self.pos[split]])
        hit = lat[np.isfinite(lat)]
        out = {
            "system": self.spec,
            "target": self.target,
            "avg": self.avg,
            "n": self.n,
            "point": point,
            "tau": float(self.taus[i]),
            "gain": float(self.gain(split)[i]),
            "dev_gain": float(self.gain("dev")[i]),
            "recall": self.recall(split, i),
            "dev_recall": self.recall("dev", i),
            "positives": len(lat),
            "latency_median": float(np.median(hit)) if len(hit) else float("nan"),
            "latency_p90": float(np.quantile(hit, 0.9)) if len(hit) else float("nan"),
            "neg_hours": {g: round(self.hours[(split, g)], 2) for g in NEG_GROUPS},
        }
        for g in NEG_GROUPS:
            out[f"fa_{g}"] = float(self.fa(split, g)[i])
            out[f"dev_fa_{g}"] = float(self.fa("dev", g)[i])
        for sub in AMBIENT_SUBGROUPS:
            out[f"fa_ambient_{sub}"] = float(self.fa(split, "ambient", sub)[i])
        for c in CONDITIONS:
            out[f"recall_{c}"] = self.recall(
                split, i, lambda r, c=c: r["condition"] == c
            )
        for s in ("fsd50k", "synth"):
            out[f"recall_{s}"] = self.recall(split, i, lambda r, s=s: r["source"] == s)
        groups = sorted({row["group"] for row, _ in self.pos[split]})
        out["recall_by_group"] = {
            g: [
                self.recall(split, i, lambda r, g=g: r["group"] == g),
                sum(r["group"] == g for r, _ in self.pos[split]),
            ]
            for g in groups
        }
        return out


def best(spec: str, target: str, point) -> Evaluation:
    """Post-processing with the best dev objective at the operating point."""
    return max(
        (evaluation(spec, target, avg, n) for avg, n in POSTPROC),
        key=lambda ev: (round(ev.objective(point), 4), -ev.avg - ev.n),
    )


def evaluate_spec(spec: str) -> dict:
    out = {}
    for target in TARGETS:
        r = best(spec, target, "wait").summary("wait")
        r["recall_at"] = {
            str(b): best(spec, target, b).summary(b)["recall"] for b in BUDGETS
        }
        out[target] = r
    return out


# ----------------------------------------------------------------------------- report
def fmt(x, pct=False, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:.0f}" if pct else f"{x:.{digits}f}"


def table(rows: list[dict], target: str) -> str:
    out = [
        "| system | avg, n | τ | gain | recall | clean | snr10 | snr0 | room "
        "| real | synth | latency med / p90 s | FA/h ambient (speech / music / household) "
        "| FA/h hard chime | FA/h hard knock | recall at FA/h ≤ 1 / 5 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        at = r["recall_at"]
        amb = " / ".join(fmt(r[f"fa_ambient_{s}"], digits=1) for s in AMBIENT_SUBGROUPS)
        synth = fmt(r["recall_synth"], True) if target == "chime" else "–"
        out.append(
            f"| {r['system']} | {r['avg']}, {r['n']} | {fmt(r['tau'], digits=3)} "
            f"| **{fmt(r['gain'], True)}** | {fmt(r['recall'], True)} "
            + "".join(f"| {fmt(r[f'recall_{c}'], True)} " for c in CONDITIONS)
            + f"| {fmt(r['recall_fsd50k'], True)} | {synth} "
            f"| {fmt(r['latency_median'], digits=1)} / {fmt(r['latency_p90'], digits=1)} "
            f"| {fmt(r['fa_ambient'], digits=1)} ({amb}) "
            f"| {fmt(r['fa_hard_chime'], digits=1)} | {fmt(r['fa_hard_knock'], digits=1)} "
            f"| {fmt(at['1.0'], True)} / {fmt(at['5.0'], True)} |"
        )
    return "\n".join(out) + "\n"


def examples(spec: str, target: str, limit: int = 40) -> str:
    """Misses and false alarms of a system on eval at the "wait" operating point."""
    ev = best(spec, target, "wait")
    i = ev.index("wait")
    tau = ev.taus[i]
    lines = [
        f"## {spec} / {target} (avg {ev.avg}, n {ev.n}, τ {tau:.3f})",
        "",
        "### Misses",
        "",
    ]
    for row, lt in ev.pos["eval"]:
        if not np.isfinite(lt[i]):
            q = ev.q[row["id"]]
            k0 = int(round(row["onset"] / HOP)) - 1
            peak = float(np.max(q[k0 : k0 + int(HIT / HOP) + 1]))
            clip = clip_index().get((row["split"], row["clips"][0]), {})
            top = top_classes(spec, row["id"], min(k0 + 5, len(q) - 1))
            lines.append(
                f"- `{row['id']}` {row['group']} / {row['condition']}: peak {peak:.2f}"
                f"{' — ' + clip.get('title', '')[:50] if clip else ''}; top at +1 s: {top}"
            )
    lines += ["", "### False alarms (eval negatives)", ""]
    count = 0
    for sid, q in ev.q.items():
        row = streams()[sid]
        if row["kind"] != "neg" or row["split"] != "eval":
            continue
        above = q >= tau
        starts = np.flatnonzero(above & ~np.concatenate([[False], above[:-1]]))
        last = -1e9
        for k in starts:
            t = (k + 1) * HOP
            if t - last < REFRACTORY:
                continue
            last = t
            clip, start = None, 0.0
            for fname in row["clips"]:
                c = clip_index()[(row["split"], fname)]
                if start + c["duration"] >= t:
                    clip = c
                    break
                start += c["duration"]
            labels = ",".join(clip["labels"][:6]) if clip else ""
            lines.append(
                f"- `{sid}` {t:.1f} s ({row['group']}): FSD50K {clip['fname'] if clip else '?'} "
                f"“{clip['title'][:40] if clip else ''}” [{labels}]; top: {top_classes(spec, sid, k)}"
            )
            count += 1
            if count >= limit:
                return "\n".join(lines) + "\n"
    return "\n".join(lines) + "\n"


def combinations(per_spec: dict, top: int = 4) -> list[str]:
    """B and D of the plan from the single systems, chosen by dev gain per target:
    ens(...) of the best 2..`top` systems of different models (CLAP excluded, its
    scores are on another scale), and gate(best CLAP, t, x) of the best single and
    ensemble at CLAP thresholds GATES."""
    out = []
    for target in TARGETS:
        ranked = sorted(per_spec, key=lambda s: -per_spec[s][target]["dev_gain"])
        best, models = [], set()
        for spec in ranked:
            model = spec.split(":")[0]
            if model.startswith("clap") or model in models:
                continue
            best.append(spec)
            models.add(model)
        ensembles = [
            f"ens({','.join(best[:k])})" for k in range(2, min(top, len(best)) + 1)
        ]
        out += ensembles
        claps = [s for s in ranked if s.startswith("clap")]
        if claps and best:
            for base in [best[0], *ensembles[:1]]:
                out += [f"gate({claps[0]},{t},{base})" for t in GATES]
    return list(dict.fromkeys(out))


def describe(spec: str) -> str:
    """Operating points of one system (gain, then the budgets) as a markdown table,
    with the recall per sound type and condition at the gain point."""
    lines = [
        f"## {spec}",
        "",
        "| target | point | avg, n | τ | recall | ambient FA/h (speech / music / household) "
        "| hard chime / knock FA/h | latency med / p90 s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_type = []
    for target in TARGETS:
        for point in ("wait", *BUDGETS):
            ev = best(spec, target, point)
            r = ev.summary(point)
            amb = " / ".join(
                fmt(r[f"fa_ambient_{s}"], digits=1) for s in AMBIENT_SUBGROUPS
            )
            lines.append(
                f"| {target} | {'gain' if point == 'wait' else f'≤ {point:g}/h'} "
                f"| {ev.avg}, {ev.n} | {fmt(r['tau'], digits=4)} | {fmt(r['recall'], True)} "
                f"| {fmt(r['fa_ambient'], digits=1)} ({amb}) "
                f"| {fmt(r['fa_hard_chime'], digits=1)} / {fmt(r['fa_hard_knock'], digits=1)} "
                f"| {fmt(r['latency_median'], digits=1)} / {fmt(r['latency_p90'], digits=1)} |"
            )
            if point == "wait":
                groups = ", ".join(
                    f"{g} {fmt(v[0], True)}" for g, v in r["recall_by_group"].items()
                )
                conds = ", ".join(
                    f"{c} {fmt(r[f'recall_{c}'], True)}" for c in CONDITIONS
                )
                by_type.append(f"- {target}: {groups}; {conds}")
    return "\n".join(lines + ["", *by_type]) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--systems", nargs="*", help="specs (default: every score file)")
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="added specs, e.g. 'ens(sslam:w2,ced_base:w2)'",
    )
    ap.add_argument(
        "--examples",
        nargs="*",
        default=[],
        help="specs to list misses and false alarms for",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="also evaluate ensembles and CLAP gates of the best systems "
        "(combinations()) and list the errors of the best system per target",
    )
    ap.add_argument(
        "--describe",
        metavar="SPEC",
        help="only print the operating points and per-type recall of one system",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/report")
    args = ap.parse_args()
    if args.describe:
        print(describe(args.describe))
        return
    out = args.out or results_root() / "report"
    out.mkdir(parents=True, exist_ok=True)

    specs = (args.systems if args.systems else available()) + args.extra
    with ProcessPoolExecutor(args.jobs) as ex:
        per_spec = dict(zip(specs, ex.map(evaluate_spec, specs)))
        if args.auto:
            added = [c for c in combinations(per_spec) if c not in per_spec]
            per_spec.update(zip(added, ex.map(evaluate_spec, added)))
            specs += added
    results = {}
    for target in TARGETS:
        rows = sorted(
            (per_spec[s][target] for s in specs),
            key=lambda r: -np.nan_to_num(r["dev_gain"]),
        )
        results[target] = rows
        (out / f"table_{target}.md").write_text(table(rows, target))
        print(f"\n{target}:\n" + table(rows, target))
    (out / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    shown = list(args.examples)
    if args.auto:
        shown += [results[t][0]["system"] for t in TARGETS]
    for spec in dict.fromkeys(shown):
        name = "".join(ch if ch.isalnum() else "_" for ch in spec)
        (out / f"examples_{name}.md").write_text(
            "\n".join(examples(spec, t) for t in TARGETS)
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
