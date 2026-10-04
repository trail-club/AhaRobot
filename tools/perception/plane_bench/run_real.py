"""Run plane detectors on captured real frames (no ground truth).

Reports per scene x method x max depth the plane count, frame-to-frame
stability of horizontal planes, coverage, small fragments and runtime, plus
the per-pixel temporal depth noise of each scene. See README.md.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"))
sys.path.insert(0, str(HERE))

from aha_perception.plane_detection_io import depth_to_cloud, plane_palette  # noqa: E402
from aha_perception.planes import (  # noqa: E402
    DETECTORS,
    DepthNoiseModel,
    DetectionOptions,
    detect_planes,
)
from real_frames import list_scenes, load_scene  # noqa: E402
from run_bench import _cpu_name, _table  # noqa: E402

DEFAULT_METHODS = ["ransac_gravity_refined", "ransac_gravity"]
SMALL_AREA = 0.1  # m^2
MATCH_HEIGHT = 0.03  # m
MATCH_XY = 0.2  # m, centroid distance
HORIZONTAL_TOL_DEG = 10.0
NOISE_BIN = 0.5  # m
NOISE_MAX_DEPTH = 8.0  # m
NOISE_MIN_PIXELS = 100


@dataclass
class Run:
    scene: object
    method: str
    max_depth: float
    planes: list  # per frame
    valid: list  # per frame (H*W,) bool, depth within range
    ms: list  # per frame


def _options(scene, index, min_depth, max_depth):
    fx = scene.intrinsics[0]
    return DetectionOptions(
        camera_to_base=scene.camera_to_floor()[index],
        min_depth=min_depth,
        max_depth=max_depth,
        noise=DepthNoiseModel(focal=fx),
    )


def run_detection(scene, method, min_depth, max_depth):
    fx, fy, cx, cy = scene.intrinsics
    run = Run(scene, method, max_depth, [], [], [])
    for i, depth in enumerate(scene.depth):
        cloud = depth_to_cloud(depth, fx, fy, cx, cy)
        options = _options(scene, i, min_depth, max_depth)
        start = time.perf_counter()
        planes = detect_planes(cloud, method, options)
        run.ms.append((time.perf_counter() - start) * 1e3)
        run.planes.append(planes)
        z = depth.reshape(-1)
        run.valid.append(np.isfinite(z) & (z >= min_depth) & (z <= max_depth))
    return run


def _horizontal(planes):
    min_cos = np.cos(np.radians(HORIZONTAL_TOL_DEG))
    return [p for p in planes if abs(p.normal[2]) >= min_cos]


def _iou(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0


def match_horizontal(prev, curr):
    """(i, j) pairs of horizontal planes matched by height and centroid xy."""
    if not prev or not curr:
        return []
    cost = np.full((len(prev), len(curr)), 1e6)
    for i, a in enumerate(prev):
        for j, b in enumerate(curr):
            dh = abs(a.centroid[2] - b.centroid[2])
            dxy = float(np.linalg.norm(a.centroid[:2] - b.centroid[:2]))
            if dh <= MATCH_HEIGHT and dxy <= MATCH_XY:
                cost[i, j] = dh / MATCH_HEIGHT + dxy / MATCH_XY
    rows, cols = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e6]


def stability(frames_planes):
    """Matched fraction, mean per-track height std (mm) and mean IoU."""
    horizontal = [_horizontal(p) for p in frames_planes]
    matched = total = 0
    ious = []
    tracks = []  # heights per track
    track_of = {j: None for j in range(len(horizontal[0]))} if horizontal else {}
    for t in range(1, len(horizontal)):
        prev, curr = horizontal[t - 1], horizontal[t]
        pairs = match_horizontal(prev, curr)
        matched += 2 * len(pairs)
        total += len(prev) + len(curr)
        next_track = {}
        for i, j in pairs:
            ious.append(_iou(prev[i].inliers, curr[j].inliers))
            k = track_of.get(i)
            if k is None:
                k = len(tracks)
                tracks.append([prev[i].centroid[2]])
            tracks[k].append(curr[j].centroid[2])
            next_track[j] = k
        track_of = next_track
    stds = [np.std(h) * 1e3 for h in tracks if len(h) >= 2]
    nan = float("nan")
    return (
        matched / total if total else nan,
        float(np.mean(stds)) if stds else nan,
        float(np.mean(ious)) if ious else nan,
    )


def summarize(run):
    counts = np.array([len(p) for p in run.planes])
    small = [sum(p.area < SMALL_AREA for p in planes) for planes in run.planes]
    assigned = []
    for planes, valid in zip(run.planes, run.valid):
        mask = np.zeros(valid.shape, dtype=bool)
        for p in planes:
            mask[p.inliers] = True
        n_valid = int(valid.sum())
        assigned.append((mask & valid).sum() / n_valid if n_valid else np.nan)
    matched, h_std, iou = stability(run.planes)
    return {
        "planes": f"{counts.mean():.1f} ({counts.min()}-{counts.max()})",
        "small": float(np.mean(small)),
        "assigned": float(np.nanmean(assigned)),
        "matched": matched,
        "h_std_mm": h_std,
        "iou": iou,
        "ms": float(np.mean(run.ms)),
    }


def temporal_noise(scene):
    """Per depth bin: (bin start, pixels, median temporal std mm, model sigma mm)."""
    depth = scene.depth
    finite = np.isfinite(depth)
    all_valid = finite.all(axis=0)
    rows = []
    if scene.n_frames < 2 or not all_valid.any():
        return rows
    samples = depth[:, all_valid].astype(np.float64)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1) * 1e3
    model = DepthNoiseModel(focal=scene.intrinsics[0])
    bins = np.floor(mean / NOISE_BIN).astype(int)
    for b in range(int(NOISE_MAX_DEPTH / NOISE_BIN)):
        sel = bins == b
        if sel.sum() < NOISE_MIN_PIXELS:
            continue
        center = (b + 0.5) * NOISE_BIN
        median = float(np.median(std[sel]))
        sigma = float(model.sigma(center) * 1e3)
        rows.append((b * NOISE_BIN, int(sel.sum()), median, sigma))
    return rows


def scene_stats(scene):
    finite = np.isfinite(scene.depth)
    any_valid = finite.any(axis=0)
    flicker = (any_valid & ~finite.all(axis=0)).sum() / max(int(any_valid.sum()), 1)
    t = scene.camera_to_base
    moved_mm = np.linalg.norm(t[:, :3, 3] - t[0, :3, 3], axis=1).max() * 1e3
    cos = (np.einsum("nij,ij->n", t[:, :3, :3], t[0, :3, :3]) - 1.0) / 2.0
    moved_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max()
    return {
        "valid": float(finite.mean()),
        "flicker": float(flicker),
        "moved": f"{moved_mm:.0f} mm / {moved_deg:.2f} deg",
    }


def make_figure(path, scene, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = scene.depth.shape[1:]
    panels = []
    if scene.color_path is not None:
        panels.append(("color", plt.imread(scene.color_path), {}))
    vmax = max(run.max_depth for run in runs)
    depth_style = {"cmap": "viridis", "vmin": 0.0, "vmax": vmax}
    panels.append((f"depth (0-{vmax:g} m)", scene.depth[0], depth_style))
    for run in runs:
        image = np.ones((shape[0] * shape[1], 3))
        image[run.valid[0]] = 0.8
        for i, plane in enumerate(run.planes[0]):
            image[plane.inliers] = plane_palette(i)
        small = sum(p.area < SMALL_AREA for p in run.planes[0])
        title = (
            f"{run.method} max {run.max_depth:g} m\n"
            f"{len(run.planes[0])} planes, {small} < {SMALL_AREA} m2"
        )
        panels.append((title, image.reshape(shape + (3,)), {}))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4))
    for ax, (title, image, kwargs) in zip(np.atleast_1d(axes), panels):
        ax.imshow(image, interpolation="nearest", **kwargs)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{scene.name} frame 0: one color per plane, light gray = unassigned in range, "
        "white = invalid or out of range",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _fmt(value, digits=2):
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def build_report(args, scenes, runs, figures):
    lines = [
        "# Plane detection on real frames",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} "
        "by `run_real.py`",
        f"- CPU: {_cpu_name()}, single thread; Python {platform.python_version()}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"- Depth range {args.min_depth:g} m to max depth, walls on, noise model "
        "focal = fx of each scene. Floor at z = 0 only when `base height` is known",
        "- Metric definitions: see README.md",
        "",
        "## Scenes",
        "",
    ]
    table = []
    for scene in scenes:
        stats = scene_stats(scene)
        meta = scene.meta
        base = scene.base_height
        table.append(
            [
                scene.name,
                str(scene.n_frames),
                meta.get("filter_label") or "-",
                "-" if base is None else f"{base:g} m",
                f"{scene.intrinsics[0]:.1f}",
                stats["moved"],
                _fmt(stats["valid"]),
                _fmt(stats["flicker"], 3),
                meta.get("note") or "",
            ]
        )
    header = ["scene", "frames", "filter", "base height", "fx", "camera motion"]
    header += ["valid px", "flicker px", "note"]
    lines += [_table(header, table), ""]
    lines += [
        "`camera motion` = max change of camera_to_base from frame 0; "
        "`flicker px` = share of pixels valid in some but not all frames.",
        "",
        "## Detection",
        "",
    ]
    table = []
    for run in runs:
        s = summarize(run)
        table.append(
            [
                run.scene.name,
                run.method,
                f"{run.max_depth:g}",
                s["planes"],
                _fmt(s["small"], 1),
                _fmt(s["assigned"]),
                _fmt(s["matched"]),
                _fmt(s["h_std_mm"], 1),
                _fmt(s["iou"]),
                _fmt(s["ms"], 0),
            ]
        )
    header = ["scene", "method", "max depth m", "planes/frame (min-max)"]
    header += ["small/frame", "assigned px", "h matched", "h std mm", "h IoU"]
    header += ["detect ms"]
    lines += [_table(header, table), ""]
    lines += [
        f"`small` = planes under {SMALL_AREA} m2; `assigned px` = share of valid "
        "pixels in range assigned to a plane; `h matched` = share of horizontal "
        f"planes matched in the next frame (height within {MATCH_HEIGHT * 100:g} cm, "
        f"centroid xy within {MATCH_XY * 100:g} cm); `h std` = mean height std of "
        "tracks of matched planes; `h IoU` = inlier pixel IoU of matched pairs.",
        "",
        "## Temporal depth noise",
        "",
        "Median per-pixel std over frames (pixels valid in every frame), by mean "
        "depth, against the model sigma(Z) with the scene fx.",
        "",
    ]
    table = []
    for scene in scenes:
        for start, count, median, model in temporal_noise(scene):
            table.append(
                [
                    scene.name,
                    f"{start:.1f}-{start + NOISE_BIN:.1f}",
                    str(count),
                    _fmt(median, 1),
                    _fmt(model, 1),
                ]
            )
    header = ["scene", "depth m", "pixels", "median std mm", "model sigma mm"]
    lines += [_table(header, table), ""]
    if figures:
        lines += ["## Figures", ""]
        lines += [f"![{name}]({name})" for name in figures]
        lines += [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", type=Path, help="scene directories")
    parser.add_argument("--root", type=Path, default=HERE / "real_frames")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--max-depth", type=float, nargs="+", default=[3.0])
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "real.md")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    unknown = [m for m in args.methods if m not in DETECTORS]
    if unknown:
        parser.error(f"unknown methods {unknown}, choose from {sorted(DETECTORS)}")
    paths = args.scenes or list_scenes(args.root)
    if not paths:
        parser.error(f"no scenes given and none under {args.root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scenes, runs, figures = [], [], []
    for path in paths:
        scene = load_scene(path)
        scenes.append(scene)
        for method in args.methods:
            method_runs = []
            for max_depth in args.max_depth:
                run = run_detection(scene, method, args.min_depth, max_depth)
                method_runs.append(run)
                print(
                    f"{scene.name} {method} max {max_depth:g} m: "
                    f"{np.mean([len(p) for p in run.planes]):.1f} planes/frame, "
                    f"{np.mean(run.ms):.0f} ms",
                    flush=True,
                )
            runs += method_runs
            if not args.no_figures:
                name = f"real_{scene.name}_{method}.png"
                make_figure(args.out.parent / name, scene, method_runs)
                figures.append(name)
    args.out.write_text(build_report(args, scenes, runs, figures) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
