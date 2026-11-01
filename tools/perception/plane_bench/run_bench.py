"""Run every plane detection candidate on the synthetic scene set.

Writes a markdown report and CSV files (and optional PNG figures) to the
output directory. See README.md for usage and metric definitions.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import itertools  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"))
sys.path.insert(0, str(HERE))

import open3d_baseline  # noqa: E402
from aha_perception.planes import DETECTORS, DetectionOptions  # noqa: E402
from metrics import aggregate, evaluate  # noqa: E402
from scenes import (  # noqa: E402
    DONT_CARE,
    INVALID,
    NOISE_LEVELS,
    NON_PLANAR,
    frame_seed,
    generate,
    iter_frames,
    scene_set,
)

SWEEP_NOISE = 1.0


def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


SWEEPS = {
    "ransac_sequential": _grid(
        distance_threshold=[0.01, 0.02, 0.03, 0.04], voxel_size=[0.015, 0.02, 0.03]
    ),
    "ransac_gravity": _grid(
        threshold_k=[2.0, 3.0, 4.0], group_tol_deg=[10.0, 15.0, 20.0]
    ),
    "organized_region_growing": _grid(
        cell_size=[10, 16, 20], planarity_k=[2.0, 3.0, 5.0]
    ),
    "normal_clustering": [
        {"cluster_angle_deg": a, "normal_radius": r, "smoothing_radius": r}
        for a, r in itertools.product([6.0, 10.0, 15.0], [2, 3, 4])
    ],
    "ransac_gravity_refined": _grid(
        refine_k=[2.0, 3.0, 4.0], normals=["knn", "organized"]
    ),
    "ransac_gravity_cape_hybrid": _grid(
        cell_size=[10, 16, 20], merge_rms_k=[2.0, 3.0, 4.0]
    ),
    "peac_ahc": _grid(node_size=[8, 10, 16], mse_k=[1.0, 1.5, 2.5]),
    "cape": _grid(cell_size=[16, 20, 24], mse_k=[1.0, 1.5, 2.5]),
}

# Detector variants reported as separate rows (label -> (detector, params)).
EXTRA_VARIANTS = {
    "ransac_gravity[organized_normals]": ("ransac_gravity", {"normals": "organized"}),
}

# Two label figures per scene: original candidates and the second round.
FIGURE_SETS = {
    "a": [
        "ransac_sequential",
        "ransac_gravity",
        "organized_region_growing",
        "normal_clustering",
    ],
    "b": ["ransac_gravity_refined", "ransac_gravity_cape_hybrid", "peac_ahc", "cape"],
}

GROUPS = [
    ("all", lambda r: True),
    ("table", lambda r: "table" in r["tags"]),
    ("shelf", lambda r: "shelf" in r["tags"]),
    ("coplanar split", lambda r: "coplanar" in r["tags"]),
    ("walls", lambda r: "walls" in r["tags"]),
    ("far distance", lambda r: "far" in r["tags"]),
    ("floor only", lambda r: "floor" in r["tags"]),
    ("clutter", lambda r: "clutter" in r["tags"]),
    ("noise x0.5", lambda r: r["noise"] == 0.5),
    ("noise x1", lambda r: r["noise"] == 1.0),
    ("high noise x2", lambda r: r["noise"] == 2.0),
]

SKIP_CSV = {"matches", "ignored"}


def _setting_name(setting: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in setting.items())


def _run(fn, frame, options):
    wall, cpu = time.perf_counter(), time.process_time()
    planes = fn(frame.cloud, options)
    cpu = (time.process_time() - cpu) * 1000
    wall = (time.perf_counter() - wall) * 1000
    stats = evaluate(frame, planes)
    stats.update(runtime_ms=wall, cpu_ms=cpu)
    return planes, stats


def _candidate(name, setting=None):
    detector = DETECTORS[name]
    params = detector.params_type(**(setting or {}))
    return lambda cloud, options: detector.detect(cloud, options, params)


def _row(frame, method, variant, setting, stats):
    row = {
        "method": method,
        "variant": variant,
        "setting": setting,
        "scene": frame.name,
        "tags": "|".join(frame.tags),
        "noise": frame.noise_scale,
    }
    row.update({k: v for k, v in stats.items() if k not in SKIP_CSV})
    return row


def run_pass(frames_args, configs, log):
    """Run (method, variant, setting, fn, walls, sweep_only) configs on all frames."""
    rows = []
    for frame in iter_frames(**frames_args):
        start = time.perf_counter()
        for method, variant, setting, fn, walls, sweep_only in configs:
            if sweep_only and frame.noise_scale != SWEEP_NOISE:
                continue
            options = DetectionOptions(
                camera_to_base=frame.camera_to_base, detect_vertical=walls
            )
            _, stats = _run(fn, frame, options)
            rows.append(_row(frame, method, variant, setting, stats))
        log(f"{frame.name} x{frame.noise_scale}: {time.perf_counter() - start:.1f} s")
    return rows


def _select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def _agg(rows):
    out = aggregate(rows)
    cpu = np.array([r["cpu_ms"] for r in rows], dtype=np.float64)
    out["cpu_mean_ms"] = float(cpu.mean()) if len(cpu) else float("nan")
    out["cpu_p95_ms"] = float(np.percentile(cpu, 95)) if len(cpu) else float("nan")
    return out


def _fmt(value, digits=3):
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in r) + " |"
        for r in rows
    ]
    return "\n".join(lines)


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: _fmt(v, 4) if isinstance(v, float) else v for k, v in row.items()}
            )


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def make_figures(out_dir: Path, names, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    palette = plt.get_cmap("tab10").colors
    special = {
        INVALID: (1, 1, 1),
        DONT_CARE: (0.82, 0.82, 0.82),
        NON_PLANAR: (0.45, 0.45, 0.45),
    }
    false_positive = (0.9, 0.0, 0.7)

    def colorize(index_image, colors):
        rgb = np.ones(index_image.shape + (3,))
        for key, color in colors.items():
            rgb[index_image == key] = color
        return rgb

    for name, (key, methods) in itertools.product(names, FIGURE_SETS.items()):
        scene = next(s for s in scene_set() if s.name == name)
        frame = generate(scene, SWEEP_NOISE, frame_seed(name, SWEEP_NOISE))
        options = DetectionOptions(camera_to_base=frame.camera_to_base)
        gt_colors = {
            **special,
            **{g: palette[g % 10] for g in range(len(frame.planes))},
        }
        panels = [("depth", None), ("ground truth", colorize(frame.labels, gt_colors))]
        for method in methods:
            planes, stats = _run(_candidate(method), frame, options)
            image = np.full(frame.labels.size, -1)
            colors = {-1: (1, 1, 1)}
            matched = {k: g for g, k in stats["matches"]}
            for k, plane in enumerate(planes):
                image[plane.inliers] = k
                if k in matched:
                    colors[k] = palette[matched[k] % 10]
                elif k in stats["ignored"]:
                    colors[k] = special[DONT_CARE]
                else:
                    colors[k] = false_positive
            title = f"{method}\nTP {stats['tp']} FP {stats['fp']} FN {stats['fn']}"
            panels.append((title, colorize(image.reshape(frame.labels.shape), colors)))
        cols = 3
        fig, axes = plt.subplots(2, cols, figsize=(12, 6.4))
        for ax, (title, image) in zip(axes.ravel(), panels):
            if image is None:
                depth = frame.cloud[..., 2]
                ax.imshow(depth, cmap=ListedColormap(plt.get_cmap("viridis").colors))
            else:
                ax.imshow(image, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(
            f"{name} (noise x{SWEEP_NOISE}): same color = matched GT plane, "
            "magenta = false positive, light gray = don't care / ignored",
            fontsize=9,
        )
        fig.tight_layout()
        path = out_dir / f"labels_{name}_{key}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        log(f"wrote {path}")


def build_report(args, rows, sweep_rows, best, methods, elapsed):
    lines = [
        "# Plane detection benchmark",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"by `run_bench.py` in {elapsed / 60:.1f} min",
        f"- CPU: {_cpu_name()}, single thread (OMP/BLAS threads = "
        f"{os.environ.get('OMP_NUM_THREADS')}); Python {platform.python_version()}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"- Frames: {len({r['scene'] for r in rows})} scenes x noise "
        f"{tuple(args.noise)}; "
        f"sweeps on noise x{SWEEP_NOISE} frames",
        "- Runtime is per frame (640x480 input) including preprocessing; `cpu` is "
        "process CPU time, `wall` is elapsed time",
        "- Metric definitions: see README.md",
        "",
    ]

    default_rows = [r for r in rows if r["variant"] in ("default", "baseline")]
    lines += ["## Aggregate (default parameters, walls on)", ""]
    columns = [
        ("PQ", "pq"),
        ("recall", "recall"),
        ("precision", "precision"),
        ("F1", "f1"),
        ("mIoU", "miou_matched"),
        ("compl.", "completeness"),
        ("holes/plane", "holes_per_plane"),
        ("hole px", "hole_px_share"),
        ("angle deg", "angle_deg"),
        ("support h cm", "support_height_cm"),
        ("class acc", "class_acc"),
        ("over", "over_seg"),
        ("under", "under_seg"),
        ("cpu mean ms", "cpu_mean_ms"),
        ("cpu p95 ms", "cpu_p95_ms"),
        ("wall mean ms", "runtime_mean_ms"),
    ]
    table = []
    for method in methods:
        a = _agg(_select(default_rows, method=method))
        table.append([method, *(a[key] for _, key in columns)])
    lines += [
        _table(["method", *(label for label, _ in columns)], table),
        "",
        "`compl.` = mean fraction of a matched GT plane's pixels in its detection; "
        "`holes/plane` = enclosed holes per matched detection that contain valid "
        "GT-plane pixels; `hole px` = share of matched GT pixels inside such holes.",
        "",
    ]

    variants = ["ransac_gravity", *EXTRA_VARIANTS, "ransac_gravity_refined"]
    variants.append("ransac_gravity_cape_hybrid")
    lines += ["## ransac_gravity and its refinements (default parameters)", ""]
    table = []
    for method in variants:
        for group in ("all", "noise x1", "high noise x2"):
            keep = dict(GROUPS)[group]
            a = _agg([r for r in _select(default_rows, method=method) if keep(r)])
            table.append(
                [
                    method,
                    group,
                    a["pq"],
                    a["precision"],
                    a["miou_matched"],
                    a["completeness"],
                    a["holes_per_plane"],
                    a["hole_px_share"],
                    a["angle_deg"],
                    a["cpu_mean_ms"],
                ]
            )
    header = ["method", "frames", "PQ", "precision", "mIoU", "compl.", "holes/plane"]
    header += ["hole px", "angle deg", "cpu mean ms"]
    lines += [_table(header, table), ""]

    lines += ["## By scene type (default parameters, walls on)", ""]
    for metric, label in (
        ("pq", "PQ"),
        ("f1", "F1"),
        ("completeness", "completeness of matched planes"),
        ("holes_per_plane", "holes per matched plane"),
        ("miou_matched", "mIoU of matched planes"),
        ("support_height_cm", "support-plane height error (cm)"),
        ("over_seg", "over-segmented GT planes"),
        ("under_seg", "under-segmenting detections"),
    ):
        table = []
        for group, keep in GROUPS:
            reference = _select(default_rows, method=methods[0])
            cells = [group, sum(1 for r in reference if keep(r))]
            for method in methods:
                selected = [r for r in _select(default_rows, method=method) if keep(r)]
                cells.append(_agg(selected)[metric] if selected else float("nan"))
            table.append(cells)
        lines += [f"### {label}", "", _table(["group", "frames", *methods], table), ""]

    lines += [
        "## Wall detection on vs off (horizontal planes only)",
        "",
        "`h_*` metrics consider only horizontal GT planes and horizontal "
        "detections. `v_recall` is the recall of vertical GT planes.",
        "",
    ]
    header = [
        "method",
        "scenes",
        "walls",
        "h_PQ",
        "h_recall",
        "h_precision",
        "h_mIoU",
        "h_height cm",
        "v_recall",
        "cpu mean ms",
    ]
    table = []
    for method in DETECTORS:
        for scope, keep in (
            ("all", lambda r: True),
            ("walls", lambda r: "walls" in r["tags"]),
        ):
            for variant, walls in (("default", "on"), ("walls_off", "off")):
                selected = [
                    r for r in _select(rows, method=method, variant=variant) if keep(r)
                ]
                a = _agg(selected)
                table.append(
                    [
                        method,
                        scope,
                        walls,
                        a["h_pq"],
                        a["h_recall"],
                        a["h_precision"],
                        a["h_miou"],
                        a["h_height_cm"],
                        a["v_recall"],
                        a["cpu_mean_ms"],
                    ]
                )
    lines += [_table(header, table), ""]

    if sweep_rows:
        lines += [
            "## Parameter sweep",
            "",
            f"Each candidate on the noise x{SWEEP_NOISE} frames (walls on), "
            "ranked by PQ. The best setting is then run on all frames (`tuned`).",
            "",
        ]
        for method in DETECTORS:
            settings = sorted(
                {r["setting"] for r in _select(sweep_rows, method=method)}
            )
            table = []
            for setting in settings:
                a = _agg(_select(sweep_rows, method=method, setting=setting))
                mark = " (best)" if setting == best[method] else ""
                table.append(
                    [
                        setting + mark,
                        a["pq"],
                        a["f1"],
                        a["miou_matched"],
                        a["completeness"],
                        a["support_height_cm"],
                        a["cpu_mean_ms"],
                    ]
                )
            table.sort(key=lambda r: -r[1])
            header = ["setting", "PQ", "F1", "mIoU", "compl.", "support h cm", "cpu ms"]
            lines += [
                f"### {method}",
                "",
                _table(header, table),
                "",
            ]

        lines += ["### Default vs tuned on all frames", ""]
        table = []
        for method in DETECTORS:
            for variant in ("default", "tuned"):
                a = _agg(_select(rows, method=method, variant=variant))
                setting = "" if variant == "default" else best[method]
                table.append(
                    [
                        method,
                        variant,
                        setting,
                        a["pq"],
                        a["recall"],
                        a["precision"],
                        a["support_height_cm"],
                        a["cpu_mean_ms"],
                    ]
                )
        lines += [
            _table(
                [
                    "method",
                    "variant",
                    "setting",
                    "PQ",
                    "recall",
                    "precision",
                    "support h cm",
                    "cpu ms",
                ],
                table,
            ),
            "",
        ]

    if args.figures:
        lines += ["## Figures", ""]
        for name, (key, methods) in itertools.product(
            args.figures, FIGURE_SETS.items()
        ):
            lines += [f"![{name} {key}](labels_{name}_{key}.png)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--noise", type=float, nargs="+", default=list(NOISE_LEVELS))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--no-open3d", action="store_true")
    parser.add_argument(
        "--figures",
        nargs="*",
        default=["table_1.5m", "shelf_4board", "two_tables"],
        help="scene names to visualise (needs matplotlib); pass none to skip",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - started:7.1f} s] {msg}", flush=True)

    frames_args = {"noise_levels": tuple(args.noise), "names": args.scenes}
    configs = []
    for name in DETECTORS:
        configs.append((name, "default", "", _candidate(name), True, False))
        configs.append((name, "walls_off", "", _candidate(name), False, False))
    methods = list(DETECTORS)
    for label, (name, setting) in EXTRA_VARIANTS.items():
        configs.append((label, "default", "", _candidate(name, setting), True, False))
        methods.append(label)
    if not args.no_open3d and open3d_baseline.available():
        for name, fn in open3d_baseline.DETECTORS.items():
            configs.append((name, "baseline", "", fn, True, False))
            methods.append(name)
    if not args.no_sweep and SWEEP_NOISE in args.noise:
        for name, settings in SWEEPS.items():
            for setting in settings:
                configs.append(
                    (
                        name,
                        "sweep",
                        _setting_name(setting),
                        _candidate(name, setting),
                        True,
                        True,
                    )
                )
    rows = run_pass(frames_args, configs, log)

    sweep_rows = [r for r in rows if r["variant"] == "sweep"]
    rows = [r for r in rows if r["variant"] != "sweep"]
    best = {}
    if sweep_rows:
        tuned_configs = []
        for name, settings in SWEEPS.items():
            scores = {
                _setting_name(s): _agg(
                    _select(sweep_rows, method=name, setting=_setting_name(s))
                )["pq"]
                for s in settings
            }
            best[name] = max(scores, key=scores.get)
            setting = settings[list(scores).index(best[name])]
            defaults = dataclasses.asdict(DETECTORS[name].params_type())
            if all(defaults[k] == v for k, v in setting.items()):
                rows += [
                    dict(r, variant="tuned", setting=best[name])
                    for r in _select(rows, method=name, variant="default")
                ]
            else:
                tuned_configs.append(
                    (name, "tuned", best[name], _candidate(name, setting), True, False)
                )
        if tuned_configs:
            log("tuned pass")
            rows += run_pass(frames_args, tuned_configs, log)

    if args.figures:
        make_figures(args.out, args.figures, log)

    summary = []
    for method in methods:
        for variant in sorted({r["variant"] for r in _select(rows, method=method)}):
            for group, keep in GROUPS:
                selected = [
                    r for r in _select(rows, method=method, variant=variant) if keep(r)
                ]
                if selected:
                    summary.append(
                        {
                            "method": method,
                            "variant": variant,
                            "group": group,
                            **_agg(selected),
                        }
                    )
    _write_csv(args.out / "per_frame.csv", rows)
    _write_csv(args.out / "summary.csv", summary)
    sweep_summary = []
    for name in SWEEPS:
        for setting in sorted({r["setting"] for r in _select(sweep_rows, method=name)}):
            sweep_summary.append(
                {
                    "method": name,
                    "setting": setting,
                    **_agg(_select(sweep_rows, method=name, setting=setting)),
                }
            )
    _write_csv(args.out / "sweep.csv", sweep_summary)
    report = build_report(
        args, rows, sweep_rows, best, methods, time.perf_counter() - started
    )
    (args.out / "report.md").write_text(report + "\n")
    log(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
