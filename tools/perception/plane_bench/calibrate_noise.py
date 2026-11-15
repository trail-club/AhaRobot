"""Fit the synthetic depth noise presets to the real static captures.

Computes ``noise_stats`` on the real scenes (``d435i_raw`` from raw captures,
``d435i_filtered`` from filtered ones), derives ``NoiseParams``, checks them by
running the same statistics on synthetic static sequences, and writes
``results/noise_calibration.md``. The printed presets are copied into
``NOISE_PRESETS`` of ``aha_perception.depth_noise`` by hand. See README.md.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import fields, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_stats  # noqa: E402
from real_frames import list_scenes, load_scene  # noqa: E402
from run_bench import _table  # noqa: E402
from scenes import (  # noqa: E402
    NOISE_PRESETS,
    CameraPose,
    Intrinsics,
    NoiseParams,
    Scene,
    box_object,
    floor,
    generate_sequence,
    table,
    wall_x,
)

PRESET_FILTERS = {"d435i_raw": "raw", "d435i_filtered": "filtered"}
QUANTIZATION = 0.001  # m, depth images are integer millimeters
REFINE_ITERATIONS = 3
SYNTHETIC_FRAMES = 10
SYNTHETIC_SEED = 20261005
# Multiplicative refinement: preset field <- noise_stats field.
_SCALED = {
    "correlation_px": "correlation_px",
    "correlation_aspect": "correlation_aspect",
    "static_correlation_px": "static_correlation_px",
    "flicker_rate": "flicker_rate",
    "flicker_px": "flicker_px",
    "dropout": "isolated_dropout",
    "outlier_rate": "outlier_fraction",
}
OUTLIER_DETECTED = 0.8  # share of z * U(0.5, 1.5) outliers beyond 10 % of Z
MEAN_HOLE_PX = np.pi * 9.5**2  # ellipse radii ~ U(4, 15) px
ROUND = {
    "subpixel": 3,
    "correlated_fraction": 3,
    "correlation_px": 1,
    "correlation_aspect": 2,
    "static_fraction": 2,
    "static_correlation_px": 1,
    "flicker_rate": 3,
    "flicker_px": 1,
    "dropout": 4,
    "outlier_rate": 5,
}


def calibration_scenes() -> list[Scene]:
    """Static views resembling the captures: a near desk top and a far wall."""
    desk = [floor(), wall_x(2.6, -3.0, 3.0, "wall")]
    desk += table("desk", (0.45, 0.0), size=(0.6, 2.4), height=1.0)
    desk += [
        box_object(0.5, 0.25, 1.0, (0.12, 0.10, 0.15), 20),
        box_object(0.6, -0.3, 1.0, (0.20, 0.15, 0.06), -10),
        box_object(1.6, 0.6, 0.0, (0.4, 0.4, 0.5), 15),
        box_object(1.8, -0.8, 0.0, (0.5, 0.4, 0.8)),
    ]
    wall = [floor(), wall_x(2.7, -3.0, 3.0, "wall")]
    wall += table("table", (1.4, -0.2), size=(1.2, 0.8))
    wall += [box_object(1.3, -0.3, 0.72, (0.15, 0.10, 0.20), 30)]
    wall += [box_object(2.2, 0.9, 0.0, (0.4, 0.4, 1.0))]
    return [
        Scene(
            "calib_desk",
            (),
            CameraPose(tilt_deg=45),
            {"floor": "floor", "wall": "wall", "desk": "table"},
            desk,
        ),
        Scene(
            "calib_wall",
            (),
            CameraPose(tilt_deg=15),
            {"floor": "floor", "wall": "wall", "table": "table"},
            wall,
        ),
    ]


def _preset_of(scene) -> str | None:
    label = (scene.meta.get("filter_label") or "").split()[0:1]
    for preset, prefix in PRESET_FILTERS.items():
        if label and label[0] == prefix:
            return preset
    return None


def _mean(stats, key):
    values = [getattr(s, key) for s in stats]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _pooled_static(stats):
    """Pixel-weighted static level (rms) and correlation length."""
    usable = [s for s in stats if s.static_pixels and np.isfinite(s.static_level)]
    weights = np.array([s.static_pixels for s in usable], dtype=np.float64)
    level = np.sqrt(np.average([s.static_level**2 for s in usable], weights=weights))
    lengths = [(s.static_correlation_px, w) for s, w in zip(usable, weights)]
    lengths = [(v, w) for v, w in lengths if np.isfinite(v)]
    length = np.average([v for v, _ in lengths], weights=[w for _, w in lengths])
    return float(level), float(length)


def initial_params(stats, static_level, static_px, focal) -> NoiseParams:
    """NoiseParams from the real statistics of one preset."""
    sub_t = _mean(stats, "subpixel_temporal")
    total = float(np.hypot(sub_t, static_level))
    return NoiseParams(
        subpixel=total,
        focal=focal,
        correlated_fraction=_mean(stats, "correlated_fraction"),
        correlation_px=_mean(stats, "correlation_px"),
        correlation_aspect=_mean(stats, "correlation_aspect"),
        static_fraction=(static_level / total) ** 2,
        static_correlation_px=static_px,
        flicker_rate=_mean(stats, "flicker_rate"),
        flicker_px=_mean(stats, "flicker_px"),
        dropout=_mean(stats, "isolated_dropout"),
        outlier_rate=_mean(stats, "outlier_fraction") / OUTLIER_DETECTED,
        quantization=QUANTIZATION,
    )


def synthetic_stats(params: NoiseParams, intrinsics: Intrinsics) -> list:
    out = []
    for i, scene in enumerate(calibration_scenes()):
        frames = generate_sequence(
            scene, SYNTHETIC_FRAMES, params, SYNTHETIC_SEED + i, intrinsics
        )
        depth = np.stack([f.cloud[..., 2] for f in frames])
        k = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        out.append(noise_stats.compute(depth, k, frames[0].camera_to_base))
    return out


def refine(
    params: NoiseParams, real: list, real_static, syn: list, intrinsics: Intrinsics
) -> NoiseParams:
    """One fixed-point step towards the real statistics."""
    syn_static, _ = _pooled_static(syn)
    sub_t = params.subpixel * np.sqrt(1 - params.static_fraction)
    level = params.subpixel * np.sqrt(params.static_fraction)
    sub_t *= _mean(real, "subpixel_temporal") / _mean(syn, "subpixel_temporal")
    level *= real_static[0] / syn_static
    total = float(np.hypot(sub_t, level))
    updates = {
        "subpixel": total,
        "static_fraction": (level / total) ** 2,
        "correlated_fraction": float(
            np.clip(
                params.correlated_fraction
                + _mean(real, "correlated_fraction")
                - _mean(syn, "correlated_fraction"),
                0.0,
                0.999,
            )
        ),
    }
    missing = _mean(real, "never_valid_fraction") - _mean(syn, "never_valid_fraction")
    pixels = intrinsics.width * intrinsics.height
    updates["holes"] = max(0, params.holes + round(missing * pixels / MEAN_HOLE_PX))
    for name, key in _SCALED.items():
        if key == "static_correlation_px":
            target, current = real_static[1], _pooled_static(syn)[1]
        else:
            target, current = _mean(real, key), _mean(syn, key)
        if np.isfinite(target) and np.isfinite(current) and current > 0:
            updates[name] = getattr(params, name) * target / current
    return replace(params, **updates)


def rounded(params: NoiseParams) -> NoiseParams:
    return replace(
        params, **{k: round(getattr(params, k), d) for k, d in ROUND.items()}
    )


def preset_source(name: str, params: NoiseParams) -> str:
    default = NoiseParams()
    items = [
        f"{f.name}={getattr(params, f.name)!r}"
        for f in fields(NoiseParams)
        if f.name not in ("focal", "scale")
        and getattr(params, f.name) != getattr(default, f.name)
    ]
    return f'    "{name}": NoiseParams(' + ", ".join(items) + "),"


# --- report -----------------------------------------------------------------


def _f(value, digits=3):
    if isinstance(value, (int, np.integer)):
        return str(value)
    if value is None or not np.isfinite(value):
        return "-"
    if float(value).is_integer() and abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{digits}g}" if abs(value) < 1e-2 else f"{value:.{digits}f}"


SUMMARY_ROWS = [
    ("temporal subpixel (fit 0.75-3.5 m)", "subpixel_temporal"),
    ("correlated fraction", "correlated_fraction"),
    ("correlation sigma px (vertical)", "correlation_px"),
    ("correlation aspect (h / v)", "correlation_aspect"),
    ("acorr row lag 1", "acorr_row_1"),
    ("acorr col lag 1", "acorr_col_1"),
    ("acorr row lag 8", "acorr_row_8"),
    ("acorr col lag 8", "acorr_col_8"),
    ("acorr row lag 16", "acorr_row_16"),
    ("acorr col lag 16", "acorr_col_16"),
    ("static plane px", "static_pixels"),
    ("static / temporal std", "static_ratio"),
    ("static level (subpixel units)", "static_level"),
    ("static correlation sigma px", "static_correlation_px"),
    ("invalid fraction", "invalid_fraction"),
    ("never-valid fraction", "never_valid_fraction"),
    ("flicker fraction", "flicker_fraction"),
    ("flicker rate at 1 px", "flicker_rate"),
    ("flicker decay px", "flicker_px"),
    ("isolated per-frame dropout", "isolated_dropout"),
    ("gross outlier fraction", "outlier_fraction"),
]


def _summary_table(real, syn):
    rs = [noise_stats.summary(s) for s in real]
    ss = [noise_stats.summary(s) for s in syn]
    rows = []
    for label, key in SUMMARY_ROWS:
        rv = [r[key] for r in rs]
        sv = [s[key] for s in ss]
        rm = float(np.nanmean(rv)) if np.isfinite(rv).any() else float("nan")
        sm = float(np.nanmean(sv)) if np.isfinite(sv).any() else float("nan")
        ratio = sm / rm if np.isfinite(rm) and rm != 0 else float("nan")
        rows.append(
            [
                label,
                _f(rm),
                " / ".join(_f(v) for v in rv),
                _f(sm),
                " / ".join(_f(v) for v in sv),
                _f(ratio, 2),
            ]
        )
    header = ["statistic", "real", "real scenes", "synthetic", "synthetic scenes"]
    return _table(header + ["syn / real"], rows)


def _bins(stats_list, attr, key_index, value_index, weight_index):
    """Weighted mean of a per-bin value over scenes, keyed by bin start."""
    acc = {}
    for stats in stats_list:
        for row in getattr(stats, attr):
            key = row[key_index]
            w = row[weight_index]
            s, sw = acc.get(key, (0.0, 0.0))
            acc[key] = (s + w * row[value_index], sw + w)
    return {k: s / w for k, (s, w) in acc.items() if w > 0}


def _temporal_table(real, syn, focal):
    r = _bins(real, "depth_bins", 0, 2, 1)
    s = _bins(syn, "depth_bins", 0, 2, 1)
    rows = []
    for start in sorted(set(r) | set(s)):
        center = start + noise_stats.DEPTH_BIN / 2
        model = center**2 * 0.08 / (focal * noise_stats.BASELINE) * 1e3
        rows.append(
            [
                f"{start:.2f}-{start + noise_stats.DEPTH_BIN:.2f}",
                _f(r.get(start, float("nan")), 3),
                _f(s.get(start, float("nan")), 3),
                _f(model, 3),
            ]
        )
    header = ["depth m", "real median std mm", "synthetic median std mm"]
    return _table(header + ["default model sigma mm"], rows)


def _acorr_table(real, syn):
    def mean(stats_list, attr):
        values = [getattr(s, attr) for s in stats_list if getattr(s, attr) is not None]
        return np.nanmean(values, axis=0) if values else None

    rr, rc = mean(real, "acorr_rows"), mean(real, "acorr_cols")
    sr, sc = mean(syn, "acorr_rows"), mean(syn, "acorr_cols")
    rows = [
        [str(lag), _f(rr[i]), _f(rc[i]), _f(sr[i]), _f(sc[i])]
        for i, lag in enumerate(noise_stats.TEMPORAL_LAGS)
    ]
    header = ["lag px", "real row", "real col", "synthetic row", "synthetic col"]
    return _table(header, rows)


def _static_acorr_table(real, syn):
    def mean(stats_list):
        values = [s.static_acorr for s in stats_list if s.static_acorr is not None]
        return np.nanmean(values, axis=0) if values else None

    r, s = mean(real), mean(syn)
    rows = [
        [
            str(lag),
            "-" if r is None else _f(r[i]),
            "-" if s is None else _f(s[i]),
        ]
        for i, lag in enumerate(noise_stats.STATIC_LAGS)
    ]
    return _table(["lag px", "real", "synthetic"], rows)


def _flicker_table(real, syn):
    r = _bins(real, "flicker_by_distance", 0, 3, 2)
    s = _bins(syn, "flicker_by_distance", 0, 3, 2)
    edges = dict(zip(noise_stats.FLICKER_EDGES[:-1], noise_stats.FLICKER_EDGES[1:]))
    rows = [
        [
            f"{lo:g}-{edges[lo]:g}",
            _f(r.get(lo, float("nan"))),
            _f(s.get(lo, float("nan"))),
        ]
        for lo in sorted(set(r) | set(s))
    ]
    return _table(["px to never-valid", "real flicker", "synthetic flicker"], rows)


def _invalid_table(real, syn):
    r = _bins(real, "invalid_by_depth", 0, 2, 1)
    s = _bins(syn, "invalid_by_depth", 0, 2, 1)
    rows = [
        [
            f"{start:.1f}-{start + noise_stats.INVALID_BIN:.1f}",
            _f(r.get(start, float("nan"))),
            _f(s.get(start, float("nan"))),
        ]
        for start in sorted(set(r) | set(s))
    ]
    return _table(["depth m", "real invalid rate", "synthetic invalid rate"], rows)


def _planes_table(groups):
    rows = []
    for source, names, stats_list in groups:
        for name, stats in zip(names, stats_list):
            for plane in stats.planes:
                pooled = plane.depth >= noise_stats.STATIC_MIN_DEPTH
                rows.append(
                    [
                        source,
                        name,
                        _f(plane.depth, 2),
                        str(plane.pixels),
                        _f(plane.static_ratio, 2),
                        "yes" if pooled else "no (near)",
                    ]
                )
    header = ["source", "scene", "depth m", "pixels", "static / temporal", "pooled"]
    return _table(header, rows)


NOTES = """\
## Limitations

- The temporal noise level depends on the surface. The per-bin subpixel of the
  real captures varies about 3x between depth bins (material and texture); the
  preset uses the median of the 0.75-3.5 m bins. Glossy (whiteboard), dark or
  mesh surfaces that fail or flicker in the real frames are not modeled; the
  synthetic noise is uniform over all surfaces.
- The frozen (static) deviation is measured on only a few planes (the desk
  surface at about 0.9 m in the desk captures, the wall and desk in scene2).
  The same desk surface gives the same static level in the raw and filtered
  captures (the post-processing reduces only the temporal part), so the static
  level is pooled over all captures and shared by both presets. Its
  correlation is fitted as one Gaussian; the real wall shows a longer tail.
- Planes nearer than 0.75 m are listed but not pooled: there the 1 mm depth
  quantization dominates the temporal noise and surface flatness the deviation.
- Real flicker sits next to never-valid pixels and depth edges; the synthetic
  flicker is a band around the frozen invalid pixels (shadow, holes, out of
  range) plus an isolated per-frame dropout. Flying pixels at depth edges stay
  valid in both. The share of never-valid pixels depends on the scene.
- The captures reach only about 3.5 m, so the presets are not calibrated
  beyond that range. Frames are 0.5 s apart; the presets assume independent
  temporal noise between frames.
- `subpixel` and the correlation lengths are in pixels of the image they are
  applied to (calibrated at the capture fx). With the bench intrinsics
  (fx = 465.7) and `focal = fx`, sigma(Z) in meters is larger by
  603 / 465.7 while its ratio to `DepthNoiseModel(focal=fx)` is kept.
"""


def build_report(groups_by_preset, presets, real_static, focal, elapsed):
    lines = [
        "# Depth noise calibration",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`calibrate_noise.py` in {elapsed:.0f} s",
        "- Real: static captures in `real_frames/`, 10 frames 0.5 s apart. "
        "Synthetic: `calib_desk` (desk top at 0.5-0.7 m, wall at 2.6 m) and "
        f"`calib_wall` (wall at 2.7 m, table) with {SYNTHETIC_FRAMES} frames "
        f"each, fx = {focal:.1f}, generated with the fitted preset",
        "- The same `noise_stats.compute` runs on both; definitions in README.md",
        "",
        "## Presets",
        "",
    ]
    names = list(presets)
    rows = []
    for f in fields(NoiseParams):
        if f.name in ("focal", "scale"):
            continue
        values = [getattr(NoiseParams(), f.name)]
        values += [getattr(presets[n], f.name) for n in names]
        if len({repr(v) for v in values}) == 1:
            continue
        rows.append([f.name, *(repr(v) for v in values)])
    stale = [n for n in names if NOISE_PRESETS.get(n) != presets[n]]
    lines += [
        _table(["field", "default", *names], rows),
        "",
        "Unlisted fields keep the defaults. `focal` is set to the image fx. "
        + (
            "`aha_perception.depth_noise.NOISE_PRESETS` holds these values."
            if not stale
            else "`aha_perception.depth_noise.NOISE_PRESETS` differs for "
            f"{', '.join(stale)}."
        ),
        "",
    ]
    for preset, (real_names, real, syn) in groups_by_preset.items():
        lines += [
            f"## {preset}",
            "",
            f"Real scenes: {', '.join(real_names)}. The static level and its "
            f"correlation are fitted to the values pooled over all captures "
            f"({_f(real_static[0])} subpixel units, sigma {_f(real_static[1], 3)} px).",
            "",
            _summary_table(real, syn),
            "",
            "### Temporal std by depth",
            "",
            _temporal_table(real, syn, focal),
            "",
            "### Spatial autocorrelation of the temporal noise",
            "",
            f"Pixels at {noise_stats.CORR_RANGE[0]:g}-"
            f"{noise_stats.CORR_RANGE[1]:g} m away from depth edges.",
            "",
            _acorr_table(real, syn),
            "",
            "### Autocorrelation of the static deviation",
            "",
            _static_acorr_table(real, syn),
            "",
            "### Flicker by distance to never-valid pixels",
            "",
            _flicker_table(real, syn),
            "",
            "### Per-frame invalid rate of ever-valid pixels by depth",
            "",
            _invalid_table(real, syn),
            "",
        ]
    groups = []
    for preset, (real_names, real, syn) in groups_by_preset.items():
        groups.append(("real", real_names, real))
        groups.append(
            (f"synthetic {preset}", [s.name for s in calibration_scenes()], syn)
        )
    lines += ["## Large planes", "", _planes_table(groups), "", NOTES]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "real_frames")
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "noise_calibration.md"
    )
    args = parser.parse_args()
    started = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - started:6.1f} s] {msg}", flush=True)

    real = {preset: ([], []) for preset in PRESET_FILTERS}
    k = None
    for path in list_scenes(args.root):
        scene = load_scene(path)
        preset = _preset_of(scene)
        if preset is None:
            continue
        stats = noise_stats.compute(
            scene.depth, scene.intrinsics, scene.camera_to_floor()[0]
        )
        real[preset][0].append(scene.name)
        real[preset][1].append(stats)
        k = scene.intrinsics
        log(f"real {scene.name} -> {preset}")
    if k is None:
        parser.error(f"no raw / filtered scenes under {args.root}")
    intrinsics = Intrinsics(fx=k[0], fy=k[1], cx=k[2], cy=k[3])
    all_real = [s for _, stats in real.values() for s in stats]
    real_static = _pooled_static(all_real)

    presets, groups = {}, {}
    for preset, (names, stats) in real.items():
        if not stats:
            continue
        params = initial_params(stats, *real_static, intrinsics.fx)
        for i in range(REFINE_ITERATIONS):
            syn = synthetic_stats(params, intrinsics)
            params = refine(params, stats, real_static, syn, intrinsics)
            log(f"{preset} refinement {i + 1}")
        params = replace(rounded(params), focal=NoiseParams().focal)
        syn = synthetic_stats(replace(params, focal=intrinsics.fx), intrinsics)
        presets[preset] = params
        groups[preset] = (names, stats, syn)
        log(f"{preset} final check")

    report = build_report(
        groups, presets, real_static, intrinsics.fx, time.perf_counter() - started
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    log(f"wrote {args.out}")
    for name, params in presets.items():
        state = "matches" if NOISE_PRESETS.get(name) == params else "differs from"
        print(
            f"{name} {state} aha_perception.depth_noise.NOISE_PRESETS:\n"
            f"{preset_source(name, params)}"
        )


if __name__ == "__main__":
    main()
