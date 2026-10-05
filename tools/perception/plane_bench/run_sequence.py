"""Run plane detectors on synthetic static frame sequences with ground truth.

For each scene x noise preset x method x max depth, a sequence from
``scenes.generate_sequence`` is detected frame by frame. Reports GT accuracy
per frame (mean and std over frames), plane counts of the raw and selected
output, frame-to-frame stability as in ``run_real.py`` and the GT detection
persistence of horizontal planes. See README.md.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"))
sys.path.insert(0, str(HERE))

from aha_perception.planes import (  # noqa: E402
    DETECTORS,
    DepthNoiseModel,
    DetectionOptions,
    detect_planes,
    select_output_planes,
)
from metrics import MatchParams, aggregate, evaluate  # noqa: E402
from run_bench import _cpu_name, _table, _write_csv  # noqa: E402
from run_real import Run, summarize  # noqa: E402
from scenes import (  # noqa: E402
    NOISE_PRESETS,
    Intrinsics,
    frame_seed,
    generate_sequence,
    noise_preset,
    scene_set,
)

DEFAULT_METHODS = ["ransac_gravity_refined", "ransac_gravity"]
DEFAULT_PRESETS = ["default", "d435i_raw", "d435i_filtered"]
MIN_DEPTH = 0.3
TRACK_DISTANCE = 0.15  # m, GT centroid distance that links a plane across frames
ACCURACY = [
    ("PQ", "pq"),
    ("F1", "f1"),
    ("recall", "recall"),
    ("precision", "precision"),
    ("h recall", "h_recall"),
    ("h height cm", "h_height_cm"),
]


def gt_persistence(frames, frame_stats):
    """Per horizontal GT plane, the share of its frames in which it is matched.

    GT planes are linked across frames by surface name and centroid distance.
    Returns (mean share over planes, share of planes matched in every frame).
    """
    min_cos = np.cos(np.radians(MatchParams().horizontal_tol_deg))
    tracks = []  # [surface, centroid, present, detected, last frame]
    for t, (frame, stats) in enumerate(zip(frames, frame_stats)):
        matched = {g for g, _ in stats["matches"]}
        for g, gt in enumerate(frame.planes):
            if abs(gt.normal[2]) < min_cos:
                continue
            candidates = [
                (float(np.linalg.norm(track[1] - gt.centroid)), i)
                for i, track in enumerate(tracks)
                if track[0] == gt.surface and track[4] != t
            ]
            candidates = [c for c in candidates if c[0] <= TRACK_DISTANCE]
            if candidates:
                track = tracks[min(candidates)[1]]
            else:
                track = [gt.surface, gt.centroid, 0, 0, -1]
                tracks.append(track)
            track[1], track[4] = gt.centroid, t
            track[2] += 1
            track[3] += int(g in matched)
    if not tracks:
        return float("nan"), float("nan")
    shares = np.array([track[3] / track[2] for track in tracks])
    return float(shares.mean()), float((shares == 1.0).mean())


def make_sequence(scene, preset, max_depth, n_frames, intrinsics=None):
    intrinsics = intrinsics or Intrinsics()
    return generate_sequence(
        scene,
        n_frames,
        noise_preset(preset, intrinsics.fx),
        frame_seed(f"{scene.name}:{preset}", 1.0),
        intrinsics,
        eval_range=(MIN_DEPTH, max_depth),
    )


def run_sequence(frames, method, max_depth):
    run = Run(frames[0], method, max_depth, [], [], [], [])
    frame_stats = []
    for frame in frames:
        options = DetectionOptions(
            camera_to_base=frame.camera_to_base,
            min_depth=MIN_DEPTH,
            max_depth=max_depth,
            noise=DepthNoiseModel(focal=frame.intrinsics.fx),
        )
        start = time.perf_counter()
        planes = detect_planes(frame.cloud, method, options)
        ms = (time.perf_counter() - start) * 1e3
        run.planes.append(planes)
        run.selected.append(select_output_planes(planes, options))
        run.ms.append(ms)
        z = frame.cloud[..., 2].reshape(-1)
        run.valid.append(np.isfinite(z) & (z >= MIN_DEPTH) & (z <= max_depth))
        stats = evaluate(frame, planes)
        stats["runtime_ms"] = ms
        frame_stats.append(stats)
    return run, frame_stats


def summarize_sequence(frames, run, frame_stats) -> dict:
    per_frame = [aggregate([s]) for s in frame_stats]
    row = {}
    for _, key in ACCURACY:
        values = np.array([a[key] for a in per_frame], dtype=np.float64)
        finite = values[np.isfinite(values)]
        row[f"{key}_mean"] = float(finite.mean()) if len(finite) else float("nan")
        row[f"{key}_std"] = float(finite.std()) if len(finite) else float("nan")
    for output in ("all", "selected"):
        counts = np.array([len(p) for p in run.output(output)])
        row[f"planes_{output}"] = float(counts.mean())
        row[f"planes_{output}_min"] = int(counts.min())
        row[f"planes_{output}_max"] = int(counts.max())
    # Stability only looks at horizontal planes, which both outputs keep.
    s = summarize(run, "all")
    row.update(h_matched=s["matched"], h_std_mm=s["h_std_mm"], h_iou=s["iou"])
    row["persistence"], row["persistent_share"] = gt_persistence(frames, frame_stats)
    row["gt_h_planes"] = float(np.mean([s["h_gt"] for s in frame_stats]))
    row["detect_ms"] = float(np.mean(run.ms))
    return row


def _fmt(value, digits=2):
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def _pm(row, key, digits=2):
    return f"{_fmt(row[key + '_mean'], digits)} ± {_fmt(row[key + '_std'], digits)}"


def _planes(row, output):
    return (
        f"{row[f'planes_{output}']:.1f} "
        f"({row[f'planes_{output}_min']}-{row[f'planes_{output}_max']})"
    )


def _nanmean(values):
    values = np.array(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def build_report(args, rows, elapsed):
    lines = [
        "# Plane detection on synthetic sequences",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`run_sequence.py` in {elapsed / 60:.1f} min",
        f"- CPU: {_cpu_name()}, single thread; Python {platform.python_version()}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"- {args.frames} frames per sequence of a static camera; frozen noise "
        "(static field, holes, shadow) shared, the rest per frame. Presets from "
        "`scenes.NOISE_PRESETS` with focal = fx; detection noise model focal = fx",
        f"- Depth range {MIN_DEPTH:g} m to max depth; GT evaluation range the same",
        "- Accuracy uses the raw `detect_planes` output; `selected` = "
        "`select_output_planes`. Metric definitions: see README.md",
        "",
        "## By preset and method (mean over scenes)",
        "",
    ]
    keys = sorted(
        {(r["preset"], r["method"], r["max_depth"]) for r in rows},
        key=lambda k: (k[2], args.presets.index(k[0]), args.methods.index(k[1])),
    )
    table = []
    for preset, method, max_depth in keys:
        group = [
            r
            for r in rows
            if (r["preset"], r["method"], r["max_depth"]) == (preset, method, max_depth)
        ]

        def mean(key, group=group):
            return _nanmean([r[key] for r in group])

        table.append(
            [
                preset,
                method,
                f"{max_depth:g}",
                _fmt(mean("pq_mean"), 3),
                _fmt(mean("pq_std"), 3),
                _fmt(mean("f1_mean"), 3),
                _fmt(mean("h_recall_mean"), 3),
                _fmt(mean("h_height_cm_mean")),
                _fmt(mean("persistence"), 3),
                _fmt(mean("persistent_share"), 3),
                _fmt(mean("planes_all"), 1),
                _fmt(mean("planes_selected"), 1),
                _fmt(mean("h_matched"), 3),
                _fmt(mean("h_std_mm"), 1),
                _fmt(mean("h_iou"), 3),
                _fmt(mean("detect_ms"), 0),
            ]
        )
    header = ["preset", "method", "max depth m", "PQ", "PQ std", "F1", "h recall"]
    header += ["h height cm", "persistence", "persistent", "planes all"]
    header += ["planes selected", "h matched", "h std mm", "h IoU", "detect ms"]
    lines += [
        _table(header, table),
        "",
        "`PQ std` = std of the per-frame PQ over the frames of a sequence. "
        "`persistence` = share of frames in which a horizontal GT plane is matched, "
        "averaged over planes; `persistent` = share of horizontal GT planes matched "
        "in every frame they are present. `h matched` / `h std` / `h IoU` as in "
        "`run_real.py` (no GT).",
        "",
        "## Per scene",
        "",
    ]
    table = []
    for r in sorted(
        rows,
        key=lambda r: (
            r["scene_index"],
            r["max_depth"],
            args.presets.index(r["preset"]),
            args.methods.index(r["method"]),
        ),
    ):
        table.append(
            [
                r["scene"],
                r["preset"],
                r["method"],
                f"{r['max_depth']:g}",
                _pm(r, "pq", 3),
                _pm(r, "f1", 3),
                _pm(r, "recall", 3),
                _pm(r, "precision", 3),
                _pm(r, "h_height_cm"),
                _fmt(r["persistence"], 2),
                _fmt(r["persistent_share"], 2),
                _planes(r, "all"),
                _planes(r, "selected"),
                _fmt(r["h_matched"]),
                _fmt(r["h_std_mm"], 1),
                _fmt(r["h_iou"]),
            ]
        )
    header = ["scene", "preset", "method", "max depth m", "PQ", "F1", "recall"]
    header += ["precision", "h height cm", "persistence", "persistent"]
    header += ["planes all (min-max)", "planes selected (min-max)", "h matched"]
    header += ["h std mm", "h IoU"]
    lines += [
        _table(header, table),
        "",
        "Accuracy cells are mean ± std over frames.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--max-depth", type=float, nargs="+", default=[3.0])
    parser.add_argument("--out", type=Path, default=HERE / "results" / "sequence.md")
    args = parser.parse_args()
    for name, known, valid in (
        ("methods", args.methods, DETECTORS),
        ("presets", args.presets, NOISE_PRESETS),
    ):
        unknown = [v for v in known if v not in valid]
        if unknown:
            parser.error(f"unknown {name} {unknown}, choose from {sorted(valid)}")
    if args.frames < 2:
        parser.error("--frames must be >= 2")
    scenes = [s for s in scene_set() if not args.scenes or s.name in args.scenes]
    if not scenes:
        parser.error("no matching scenes")
    started = time.perf_counter()
    rows = []
    for index, scene in enumerate(scenes):
        for preset in args.presets:
            for max_depth in args.max_depth:
                frames = make_sequence(scene, preset, max_depth, args.frames)
                for method in args.methods:
                    run, stats = run_sequence(frames, method, max_depth)
                    row = {
                        "scene": scene.name,
                        "scene_index": index,
                        "preset": preset,
                        "method": method,
                        "max_depth": max_depth,
                    }
                    row.update(summarize_sequence(frames, run, stats))
                    rows.append(row)
                    print(
                        f"[{time.perf_counter() - started:6.1f} s] {scene.name} "
                        f"{preset} {method} max {max_depth:g} m: "
                        f"PQ {row['pq_mean']:.3f}, persistence "
                        f"{_fmt(row['persistence'])}",
                        flush=True,
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args, rows, time.perf_counter() - started)
    args.out.write_text(report + "\n")
    _write_csv(
        args.out.with_suffix(".csv"),
        [{k: v for k, v in r.items() if k != "scene_index"} for r in rows],
    )
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
