"""Evaluate surface tracking (aha_perception.surfaces) on moving-camera sequences.

For each scene of ``scenes.tracking_scene_set()`` x noise preset, a sequence
from ``scenes.generate_motion_sequence`` follows ``trajectory``: static, a
head pan sweep (+-40 deg), the base turning away (180 deg) for ``--away``
seconds and back, then a base move forward with a tilt change. Frames are
DT apart except while turned away. Each frame is detected like the node
(``NODE_CONFIG``, ``select_output_planes``) and fed to a ``SurfaceTracker``
with the known camera -> base and base -> odom transforms and the camera view
(decimated depth). Output planes are matched to GT planes with
``metrics.evaluate``; a GT surface (by name) gets the surface ids of its
matched planes. The static real frames (``real_frames/``) are tracked too
(no GT). See README.md for the metrics.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import datetime  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"))
sys.path.insert(0, str(HERE))

from aha_perception.plane_detection_io import depth_to_cloud  # noqa: E402
from aha_perception.planes import (  # noqa: E402
    DetectionOptions,
    detect_planes,
    select_output_planes,
)
from aha_perception.surfaces import (  # noqa: E402
    CameraView,
    SurfaceTracker,
    TrackerOptions,
)
from configs import parse_config  # noqa: E402
from metrics import evaluate  # noqa: E402
from real_frames import list_scenes, load_scene  # noqa: E402
from run_bench import _cpu_name, _table, _write_csv  # noqa: E402
from run_sequence import MIN_DEPTH, node_frame  # noqa: E402
from scenes import (  # noqa: E402
    NOISE_PRESETS,
    Intrinsics,
    MotionPose,
    frame_seed,
    generate_motion_sequence,
    noise_preset,
    tracking_scene_set,
)

METHOD = "ransac_gravity_refined"
NODE_CONFIG = "sub=0.12,merge=on,node_stride=2"
MAX_DEPTH = 3.0
DT = 0.2  # s between detections (node max_rate 5 Hz)
TRACKED_CLASSES = ("floor", "table", "shelf", "wall")
MIN_GAP = 2  # frames without the GT surface that count as out of view
DEFAULT_PRESETS = ["d435i_filtered", "d435i_raw"]


def trajectory(tilt: float, away_s: float, away_frames: int = 6):
    """(MotionPose, seconds since the previous frame) per frame."""
    steps = [(MotionPose(tilt_deg=tilt), DT) for _ in range(5)]
    pans = [10, 20, 30, 40, 30, 20, 10, 0, -10, -20, -30, -40, -30, -20, -10, 0]
    steps += [(MotionPose(pan_deg=p, tilt_deg=tilt), DT) for p in pans]
    steps += [(MotionPose(yaw_deg=y, tilt_deg=tilt), DT) for y in (45, 90, 135, 180)]
    steps += [
        (MotionPose(yaw_deg=180, tilt_deg=tilt), away_s / away_frames)
        for _ in range(away_frames)
    ]
    steps += [(MotionPose(yaw_deg=y, tilt_deg=tilt), DT) for y in (135, 90, 45, 0)]
    for k in [*range(1, 7), *range(5, -1, -1)]:
        pose = MotionPose(x=0.05 * k, yaw_deg=2.0 * k, tilt_deg=tilt + 2.0 * k)
        steps.append((pose, DT))
    return steps


def node_config():
    return parse_config(NODE_CONFIG, [METHOD], node_stride=True)


def detect(frame, config):
    """Selected output planes of the node and the detection options."""
    options = DetectionOptions(
        camera_to_base=frame.camera_to_base,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        noise=config.noise(frame.intrinsics.fx),
    )
    planes = detect_planes(frame.cloud, METHOD, options, config.detector_params(METHOD))
    return select_output_planes(planes, options), options


def camera_view(camera_to_tracking, k, width, height, depth, stride):
    fx, fy, cx, cy = k
    return CameraView(
        camera_to_tracking=camera_to_tracking,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        depth=depth,
        stride=stride,
    )


def run_sequence(frames, stamps, config, tracker_options):
    """Per-frame records of detection, GT matching and tracking."""
    tracker = SurfaceTracker(tracker_options)
    records = []
    for frame, stamp in zip(frames, stamps):
        frame = node_frame(frame, config)
        intr = frame.intrinsics
        start = time.perf_counter()
        selected, options = detect(frame, config)
        detected = time.perf_counter()
        view = camera_view(
            frame.base_to_odom @ frame.camera_to_base,
            (intr.fx, intr.fy, intr.cx, intr.cy),
            intr.width,
            intr.height,
            frame.cloud[..., 2],
            config.node_stride,
        )
        ids = tracker.update(selected, frame.base_to_odom, stamp, view, options.noise)
        tracked = time.perf_counter()
        stats = evaluate(frame, selected)
        surfaces = {s.id: s for s in tracker.surfaces()}
        gt_of_det = {k: frame.planes[g] for g, k in stats["matches"]}
        present, gt_ids = {}, {}
        for gt in frame.planes:
            if gt.cls in TRACKED_CLASSES:
                present[gt.surface] = gt.centroid[2]
                gt_ids.setdefault(gt.surface, set())
        track_status = {}
        for k, surface_id in enumerate(ids):
            gt = gt_of_det.get(k)
            if gt is not None and gt.surface in gt_ids and surface_id:
                gt_ids[gt.surface].add(surface_id)
            if not surface_id:
                continue
            status = "gt" if gt is not None else "dc" if k in stats["ignored"] else "fp"
            entry = track_status.setdefault(surface_id, set())
            entry.add(status)
            if gt is not None:
                entry.add(f"surface:{gt.surface}")
        records.append(
            {
                "stamp": stamp,
                "present": present,
                "detected": {frame.planes[g].surface for g, _ in stats["matches"]},
                "ids": gt_ids,
                "tracks": track_status,
                "levels": {i: s.level for i, s in surfaces.items()},
                "surfaces": len(surfaces),
                "detect_ms": (detected - start) * 1e3,
                "track_ms": (tracked - detected) * 1e3,
            }
        )
    return records


def surface_rows(records, classes):
    """Per GT surface: presence, ID coverage, latency, switches, returns."""
    rows = []
    names = sorted({name for r in records for name in r["present"]})
    heights = {}
    for r in records:
        heights.update(r["present"])
    shelf_names = sorted(
        (n for n in names if classes.get(n) == "shelf"), key=lambda n: heights[n]
    )
    for name in names:
        present = [t for t, r in enumerate(records) if name in r["present"]]
        with_id = [t for t in present if records[t]["ids"].get(name)]
        sequence = []
        for t in with_id:
            ids = records[t]["ids"][name]
            keep = sequence and sequence[-1] in ids
            sequence.append(sequence[-1] if keep else min(ids))
        switches = sum(a != b for a, b in zip(sequence, sequence[1:]))
        returns = retained = 0
        max_away = 0.0
        for a, b in zip(present, present[1:]):
            if b - a - 1 < MIN_GAP:
                continue
            before = [s for t, s in zip(with_id, sequence) if t <= a]
            after = [s for t, s in zip(with_id, sequence) if t >= b]
            if not before:
                continue
            returns += 1
            if after and after[0] == before[-1]:
                retained += 1
                away = records[b]["stamp"] - records[a]["stamp"]
                max_away = max(max_away, away)
        level_ok = level_n = 0
        if name in shelf_names:
            for t, surface_id in zip(with_id, sequence):
                so_far = {n for r in records[: t + 1] for n in r["present"]}
                seen = sorted(so_far & set(shelf_names), key=lambda n: heights[n])
                level_n += 1
                level_ok += int(
                    records[t]["levels"].get(surface_id) == seen.index(name) + 1
                )
        first = present[0]
        latency = with_id[0] - first if with_id else None
        rows.append(
            {
                "surface": name,
                "class": classes.get(name, "?"),
                "present": len(present),
                "detected": sum(name in records[t]["detected"] for t in present),
                "with_id": len(with_id),
                "coverage": len(with_id) / len(present),
                "latency_frames": latency,
                "latency_s": (
                    records[with_id[0]]["stamp"] - records[first]["stamp"]
                    if with_id
                    else None
                ),
                "ids": " ".join(str(i) for i in dict.fromkeys(sequence)),
                "distinct_ids": len(set(sequence)),
                "switches": switches,
                "returns": returns,
                "retained": retained,
                "max_away_s": max_away,
                "level_ok": level_ok,
                "level_n": level_n,
            }
        )
    return rows


def track_summary(records):
    """Confirmed tracks: total, false (mostly unmatched), don't care, shared."""
    statuses: dict[int, Counter] = {}
    surfaces: dict[int, Counter] = {}
    for r in records:
        for surface_id, entry in r["tracks"].items():
            kind = "gt" if "gt" in entry else "fp" if "fp" in entry else "dc"
            statuses.setdefault(surface_id, Counter())[kind] += 1
            for item in entry:
                if item.startswith("surface:"):
                    surfaces.setdefault(surface_id, Counter())[item[8:]] += 1
    false = [i for i, c in statuses.items() if c["fp"] > 0.5 * sum(c.values())]
    dont_care = [
        i
        for i, c in statuses.items()
        if i not in false and c["dc"] > 0.5 * sum(c.values())
    ]
    shared = [i for i, c in surfaces.items() if sum(n >= 2 for n in c.values()) >= 2]
    return {
        "tracks": len(statuses),
        "false_tracks": len(false),
        "dont_care_tracks": len(dont_care),
        "shared_ids": len(shared),
    }


def run_job(job):
    index, name, preset, away_s = job
    scene = next(s for s in tracking_scene_set() if s.name == name)
    config = node_config()
    steps = trajectory(scene.pose.tilt_deg, away_s)
    frames = generate_motion_sequence(
        scene,
        [pose for pose, _ in steps],
        noise_preset(preset, Intrinsics().fx),
        frame_seed(f"{scene.name}:{preset}:motion", 1.0),
        eval_range=(MIN_DEPTH, MAX_DEPTH),
    )
    stamps = np.cumsum([dt for _, dt in steps]) - steps[0][1]
    records = run_sequence(frames, stamps, config, TrackerOptions())
    rows = surface_rows(records, scene.surfaces)
    for row in rows:
        row.update(scene=name, preset=preset)
    summary = {"scene": name, "scene_index": index, "preset": preset}
    summary.update(track_summary(records))
    summary["frames"] = len(records)
    summary["detect_ms"] = float(np.mean([r["detect_ms"] for r in records]))
    summary["track_ms"] = float(np.mean([r["track_ms"] for r in records]))
    summary["track_ms_max"] = float(np.max([r["track_ms"] for r in records]))
    return summary, rows


def run_real(path, config):
    """Track the static real frames (no GT): ID stability per scene."""
    scene = load_scene(path)
    fx, fy, cx, cy = scene.intrinsics
    height, width = scene.depth.shape[1:]
    tracker = SurfaceTracker()
    transforms = scene.camera_to_floor()
    per_frame = []
    first_frame = {}
    for t in range(scene.n_frames):
        cloud = depth_to_cloud(scene.depth[t], fx, fy, cx, cy, config.node_stride)
        options = DetectionOptions(
            camera_to_base=transforms[t],
            min_depth=MIN_DEPTH,
            max_depth=MAX_DEPTH,
            noise=config.noise(fx),
        )
        planes = detect_planes(cloud, METHOD, options, config.detector_params(METHOD))
        selected = select_output_planes(planes, options)
        view = camera_view(
            transforms[t],
            (fx, fy, cx, cy),
            width,
            height,
            cloud[..., 2],
            config.node_stride,
        )
        ids = tracker.update(selected, np.eye(4), scene.stamps[t], view, options.noise)
        for surface_id in ids:
            if surface_id:
                first_frame.setdefault(surface_id, t)
        per_frame.append(ids)
    surfaces = tracker.surfaces()
    late = [i for i, t in first_frame.items() if t > 1]
    after = per_frame[1:]
    with_id = sum(sum(1 for i in ids if i) for ids in after)
    total = sum(len(ids) for ids in after)
    visible = []
    for surface_id, t0 in first_frame.items():
        frames = per_frame[t0:]
        visible.append(np.mean([surface_id in ids for ids in frames]))
    return {
        "scene": scene.name,
        "frames": scene.n_frames,
        "planes": float(np.mean([len(ids) for ids in per_frame])),
        "with_id": with_id / total if total else float("nan"),
        "ids": len(first_frame),
        "late_ids": len(late),
        "alive": len(surfaces),
        "visible": float(np.mean(visible)) if visible else float("nan"),
        "surfaces": ", ".join(
            f"S{s.id} {s.label} {s.height:.2f} m ({s.observations})" for s in surfaces
        ),
    }


def _fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def build_report(args, summaries, rows, real_rows, elapsed):
    options = TrackerOptions()
    lines = [
        "# Surface tracking on moving-camera sequences",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`run_tracking.py` in {elapsed / 60:.1f} min",
        f"- CPU: {_cpu_name()}, single thread per process; Python "
        f"{platform.python_version()}, numpy {np.__version__}, scipy "
        f"{scipy.__version__}",
        f"- Detection: `{METHOD}` [{node_config().label}], `select_output_planes`, "
        f"depth {MIN_DEPTH:g}-{MAX_DEPTH:g} m; tracker `TrackerOptions()` "
        f"(confirm {options.confirm_hits}, max misses {options.max_misses}, "
        f"max unseen {options.max_unseen_s:g} s), tracking frame = scene frame "
        "(exact odometry)",
        f"- Trajectory: 5 static, pan sweep +-40 deg (16), turn away 180 deg (4), "
        f"{args.away:g} s away (6 frames), back (4), forward 0.3 m with yaw +12 "
        f"deg and tilt +12 deg and back (12); {DT:g} s between frames otherwise",
        "- Per GT surface (classes floor / table / shelf / wall); definitions in "
        "README.md",
        "",
        "## By scene and preset",
        "",
    ]

    def key(r):
        return (r["scene_index"], args.presets.index(r["preset"]))

    def key_of(r):
        return (r["scene"], r["preset"])

    table = []
    for s in sorted(summaries, key=key):
        group = [r for r in rows if (r["scene"], r["preset"]) == key_of(s)]
        with_id = [r for r in group if r["with_id"]]
        returns = sum(r["returns"] for r in group)
        retained = sum(r["retained"] for r in group)
        level_n = sum(r["level_n"] for r in group)
        table.append(
            [
                s["scene"],
                s["preset"],
                str(len(group)),
                f"{len(with_id)}/{len(group)}",
                _fmt(float(np.mean([r["coverage"] for r in group]))),
                _fmt(float(np.mean([r["latency_frames"] for r in with_id])), 1)
                if with_id
                else "-",
                str(sum(r["switches"] for r in group)),
                _fmt(float(np.mean([r["distinct_ids"] for r in with_id])))
                if with_id
                else "-",
                str(s["shared_ids"]),
                str(s["tracks"]),
                str(s["false_tracks"]),
                str(s["dont_care_tracks"]),
                f"{retained}/{returns}",
                _fmt(max((r["max_away_s"] for r in group), default=0.0), 1),
                _fmt(sum(r["level_ok"] for r in group) / level_n) if level_n else "-",
                _fmt(s["detect_ms"], 0),
                _fmt(s["track_ms"], 1),
                _fmt(s["track_ms_max"], 1),
            ]
        )
    header = ["scene", "preset", "GT surfaces", "with ID", "coverage"]
    header += ["latency frames", "switches", "IDs/surface", "shared IDs", "tracks"]
    header += ["false tracks", "don't care tracks", "returns kept", "max away s"]
    header += ["level acc", "detect ms", "track ms", "track ms max"]
    lines += [_table(header, table), "", "## Per surface", ""]

    table = []
    for r in sorted(
        rows, key=lambda r: (r["scene"], args.presets.index(r["preset"]), r["surface"])
    ):
        table.append(
            [
                r["scene"],
                r["preset"],
                r["surface"],
                r["class"],
                str(r["present"]),
                str(r["detected"]),
                str(r["with_id"]),
                _fmt(r["latency_frames"]),
                r["ids"] or "-",
                str(r["switches"]),
                f"{r['retained']}/{r['returns']}",
                f"{r['level_ok']}/{r['level_n']}" if r["level_n"] else "-",
            ]
        )
    header = ["scene", "preset", "surface", "class", "present", "detected"]
    header += ["with ID", "latency", "IDs", "switches", "returns kept", "level ok"]
    lines += [_table(header, table), ""]

    if real_rows:
        table = [
            [
                r["scene"],
                str(r["frames"]),
                _fmt(r["planes"], 1),
                _fmt(r["with_id"]),
                str(r["ids"]),
                str(r["late_ids"]),
                str(r["alive"]),
                _fmt(r["visible"]),
                r["surfaces"],
            ]
            for r in real_rows
        ]
        header = ["scene", "frames", "planes/frame", "planes with ID", "IDs"]
        header += ["late IDs", "alive", "visible", "surfaces at the end (obs.)"]
        lines += [
            "## Real static frames",
            "",
            "Same detection and tracker; tracking frame = `base_footprint` "
            "(static robot), stamps of the capture.",
            "",
            _table(header, table),
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    parser.add_argument(
        "--away", type=float, default=60.0, help="seconds turned away (default 60)"
    )
    parser.add_argument(
        "--real",
        nargs="*",
        type=Path,
        default=None,
        help="real scene dirs (default: all in real_frames/; none with --real)",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "tracking_report.md"
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    unknown = [p for p in args.presets if p not in NOISE_PRESETS]
    if unknown:
        parser.error(f"unknown presets {unknown}, choose from {sorted(NOISE_PRESETS)}")
    scenes = [
        s for s in tracking_scene_set() if not args.scenes or s.name in args.scenes
    ]
    if not scenes:
        parser.error("no matching scenes")
    jobs = [
        (index, scene.name, preset, args.away)
        for index, scene in enumerate(scenes)
        for preset in args.presets
    ]
    started = time.perf_counter()
    summaries, rows = [], []
    with contextlib.ExitStack() as stack:
        if args.jobs > 1:
            pool = stack.enter_context(ProcessPoolExecutor(args.jobs))
            results = pool.map(run_job, jobs)
        else:
            results = map(run_job, jobs)
        for summary, job_rows in results:
            summaries.append(summary)
            rows += job_rows
            print(
                f"[{time.perf_counter() - started:6.1f} s] {summary['scene']} "
                f"{summary['preset']}: {summary['tracks']} tracks, switches "
                f"{sum(r['switches'] for r in job_rows)}",
                flush=True,
            )
    real_dirs = list_scenes(HERE / "real_frames") if args.real is None else args.real
    real_rows = [run_real(path, node_config()) for path in real_dirs]
    for r in real_rows:
        print(f"real {r['scene']}: {r['ids']} IDs, {r['late_ids']} late", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - started
    report = build_report(args, summaries, rows, real_rows, elapsed)
    args.out.write_text(report + "\n")
    _write_csv(args.out.with_suffix(".csv"), rows)
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
