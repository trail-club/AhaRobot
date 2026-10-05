"""Place study on shelves: per-board scores of the node output on synthetic sequences.

For each scene of ``scenes.shelf_scene_set`` x noise preset x method x detector
configuration, a static sequence is detected frame by frame. The node output
(``select_output_planes``) is scored per shelf board, identified by its surface
name (all GT pieces of a board form one target), and every horizontal output
plane is checked against a noise-free face identity image (board tops,
undersides and edges, floor, table, object tops). See README.md.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"))
sys.path.insert(0, str(HERE))

from aha_perception.planes import (  # noqa: E402
    DETECTORS,
    DetectionOptions,
    detect_planes,
    select_output_planes,
)
from aha_perception.planes.classify import is_horizontal  # noqa: E402
from configs import parse_config  # noqa: E402
from run_bench import _cpu_name, _table, _write_csv  # noqa: E402
from run_sequence import MIN_DEPTH, make_sequence  # noqa: E402
from scenes import (  # noqa: E402
    _FACES,
    NOISE_PRESETS,
    Cylinder,
    Quad,
    raycast_faces,
    shelf_scene_set,
)

DEFAULT_CONFIGS = ["sub=0.08", "sub=0.12,merge=on"]
DEFAULT_LABELS = ["current", "node"]
DEFAULT_PRESETS = ["d435i_filtered", "d435i_raw", "default"]
DEFAULT_FIGURES = ["shelf_s25_d1.5", "shelf_s35_d1.0", "shelf_s40_d1.0_table"]
SUPPORT_CLASSES = ("floor", "table", "shelf")
MIN_COVERAGE = 0.3  # share of the board's GT pixels inside the plane
MIN_PURITY = 0.5  # share of the plane's inliers on that board (top and edges)
FRAGMENT_COVERAGE = 0.05  # pure planes covering this share count as fragments
MIX_SHARE = 0.1  # share of a plane's inliers on a second surface (merge error)
MIX_MIN_PX = 100
FALSE_SUPPORT_SHARE = 0.5  # below this share on support surfaces: false place plane
HIDDEN_SHARE = 0.3  # share of a plane's inliers on a hidden board's faces
LATER_FRAME = 5
KEEP_SUFFIX = " +keep"


@dataclass
class SceneInfo:
    """Noise-free face identity of a scene and its boards."""

    role: np.ndarray  # (H*W,) role index per pixel
    names: list  # role names
    kinds: np.ndarray  # role kind: support / underside / edge / object / other / none
    board_of: np.ndarray  # board index of a role (top, underside, edge) or -1
    surface_of: np.ndarray  # support surface index of a role (top, edge) or -1
    surfaces: list  # support surface names (floor, boards, table)
    boards: list  # [(surface name, top height)] from low to high
    board_surface: list  # support surface index of each board
    top_px: np.ndarray  # visible top pixels per board within the depth range

    def surface_counts(self, inliers) -> np.ndarray:
        """Inlier counts per support surface (top and edges of the same body)."""
        index = self.surface_of[self.role[inliers]] + 1
        return np.bincount(index, minlength=len(self.surfaces) + 1)[1:]


def _face_role(scene, primitive, face):
    """(role name, kind) of one face of a primitive."""
    if isinstance(primitive, Cylinder):
        return ("object top", "object") if face == 1 else ("object side", "other")
    if isinstance(primitive, Quad):
        tag = primitive.tag
        if isinstance(tag, str):
            kind = "support" if scene.surfaces[tag] in SUPPORT_CLASSES else "other"
            return tag, kind
        return "other", "other"
    name = _FACES[face]
    top = primitive.faces.get("+z")
    if isinstance(top, str) and scene.surfaces[top] in SUPPORT_CLASSES:
        if name == "+z":
            return top, "support"
        if name == "-z":
            return f"{top} underside", "underside"
        return f"{top} edge", "edge"
    tag = primitive.faces.get(name)
    if isinstance(tag, str):
        kind = "support" if scene.surfaces[tag] in SUPPORT_CLASSES else "other"
        return tag, kind
    if name == "+z":
        return "object top", "object"
    if name == "-z":
        return "object underside", "underside"
    return "other", "other"


def scene_info(scene, intrinsics, max_depth) -> SceneInfo:
    depth, index, faces = raycast_faces(scene, intrinsics)
    in_range = np.isfinite(depth) & (depth >= MIN_DEPTH) & (depth <= max_depth)
    boards = []
    for primitive in scene.primitives:
        top = getattr(primitive, "faces", {}).get("+z")
        if isinstance(top, str) and scene.surfaces.get(top) == "shelf":
            boards.append((top, primitive.center[2] + primitive.size[2] / 2))
    boards.sort(key=lambda b: b[1])
    board_index = {name: b for b, (name, _) in enumerate(boards)}

    surfaces = [name for name, cls in scene.surfaces.items() if cls in SUPPORT_CLASSES]
    names, kinds, board_of, surface_of = ["none"], ["none"], [-1], [-1]
    role = np.zeros(depth.shape, dtype=np.int64)
    for k, primitive in enumerate(scene.primitives):
        for face in np.unique(faces[index == k]):
            if face < 0:
                continue
            name, kind = _face_role(scene, primitive, int(face))
            if name not in names:
                body = name.split(" ")[0]
                names.append(name)
                kinds.append(kind)
                board_of.append(board_index.get(body, -1))
                on_surface = kind in ("support", "edge") and body in surfaces
                surface_of.append(surfaces.index(body) if on_surface else -1)
            role[(index == k) & (faces == face)] = names.index(name)
    surface_of = np.array(surface_of)
    board_surface = [surfaces.index(name) for name, _ in boards]
    top_px = np.array(
        [
            int(((role == r) & in_range).sum())
            for r in [names.index(n) if n in names else -1 for n, _ in boards]
        ]
    )
    return SceneInfo(
        role=role.reshape(-1),
        names=names,
        kinds=np.array(kinds),
        board_of=np.array(board_of),
        surface_of=surface_of,
        surfaces=surfaces,
        boards=boards,
        board_surface=board_surface,
        top_px=top_px,
    )


def tag_value(scene, key) -> float:
    for tag in scene.tags:
        if tag.startswith(f"{key}="):
            return float(tag.split("=", 1)[1])
    return float("nan")


def plane_height_at(plane, xy) -> float:
    n, d = plane.normal, plane.offset
    return float(-(n[0] * xy[0] + n[1] * xy[1] + d) / n[2])


def evaluate_frame(frame, outputs, info, options):
    """Per-board and per-plane place statistics of one frame's output planes."""
    labels = frame.labels.reshape(-1)
    horizontal = [p for p in outputs if is_horizontal(p.normal, options)]
    n_roles = len(info.names)
    counts = [np.bincount(info.role[p.inliers], minlength=n_roles) for p in horizontal]
    surface_counts = [info.surface_counts(p.inliers) for p in horizontal]
    pieces = defaultdict(list)
    for g, gt in enumerate(frame.planes):
        pieces[gt.surface].append(g)

    board_recs = []
    for b, (name, height) in enumerate(info.boards):
        gids = pieces.get(name, [])
        mask = np.isin(labels, gids)
        n_gt = int(mask.sum())
        if info.top_px[b] == 0:
            status = "hidden"
        elif n_gt == 0:
            status = "sliver"
        else:
            status = "target"
        rec = {
            "board": b,
            "name": name,
            "height": height,
            "status": status,
            "gt_px": n_gt,
            "detected": False,
            "dh": float("nan"),
            "coverage": float("nan"),
            "fragments": 0,
            "label": "",
        }
        if n_gt:
            weights = np.array([frame.planes[g].pixels for g in gids], dtype=float)
            centroid = (
                np.array([frame.planes[g].centroid for g in gids]) * weights[:, None]
            ).sum(axis=0) / weights.sum()
            best, best_cover = None, 0.0
            for k, plane in enumerate(horizontal):
                purity = surface_counts[k][info.board_surface[b]] / len(plane.inliers)
                cover = mask[plane.inliers].sum() / n_gt
                if purity < MIN_PURITY:
                    continue
                rec["fragments"] += int(cover >= FRAGMENT_COVERAGE)
                if cover > best_cover:
                    best, best_cover = plane, cover
            if best is not None and best_cover >= MIN_COVERAGE:
                rec.update(
                    detected=True,
                    dh=plane_height_at(best, centroid[:2]) - height,
                    coverage=float(best_cover),
                    label=best.label.value,
                )
        board_recs.append(rec)

    hidden = [b for b, r in enumerate(board_recs) if r["status"] == "hidden"]
    plane_recs = []
    on_surface = info.surface_of >= 0
    for plane, count, surface_count in zip(horizontal, counts, surface_counts):
        n = len(plane.inliers)
        share = count / n
        mixed = np.flatnonzero(
            (surface_count >= MIX_SHARE * n) & (surface_count >= MIX_MIN_PX)
        )
        on_board = [s for s in mixed if s in info.board_surface]
        merge = ""
        if len(mixed) >= 2 and on_board:
            merge = "board-board" if len(on_board) == len(mixed) else "board-support"
        elif len(mixed) >= 2:
            merge = "support-support"
        object_mix = bool(on_board) and share[info.kinds == "object"].sum() >= MIX_SHARE
        false_kind = ""
        if surface_count.sum() < FALSE_SUPPORT_SHARE * n:
            other = np.where(on_surface, 0, count)
            false_kind = str(info.kinds[int(np.argmax(other))])
        on_hidden = [
            info.boards[b][0]
            for b in hidden
            if share[info.board_of == b].sum() >= HIDDEN_SHARE
        ]
        dominant = int(np.argmax(surface_count)) if surface_count.any() else -1
        plane_recs.append(
            {
                "height": float(plane.centroid[2]),
                "label": plane.label.value,
                "pixels": n,
                "surface": info.surfaces[dominant]
                if dominant >= 0 and not false_kind
                else "",
                "merge": merge,
                "merged": "+".join(info.surfaces[s] for s in mixed) if merge else "",
                "object_mix": object_mix,
                "false_kind": false_kind,
                "hidden": "+".join(on_hidden),
            }
        )
    return board_recs, plane_recs


def detect_sequence(frames, method, max_depth, config):
    params = config.detector_params(method)
    out = []
    for frame in frames:
        options = DetectionOptions(
            camera_to_base=frame.camera_to_base,
            min_depth=MIN_DEPTH,
            max_depth=max_depth,
            noise=config.noise(frame.intrinsics.fx),
        )
        start = time.perf_counter()
        planes = detect_planes(frame.cloud, method, options, params)
        ms = (time.perf_counter() - start) * 1e3
        out.append((planes, select_output_planes(planes, options), options, ms))
    return out


def run_job(job):
    """Board and plane records of one (scene, preset) sequence for all runs."""
    name, preset, max_depth, n_frames, methods, configs, labels, keep = job
    scene = next(s for s in shelf_scene_set() if s.name == name)
    frames = make_sequence(scene, preset, max_depth, n_frames)
    info = scene_info(scene, frames[0].intrinsics, max_depth)
    keys = {
        "scene": name,
        "preset": preset,
        "distance": tag_value(scene, "distance"),
        "spacing": tag_value(scene, "spacing"),
    }
    boards, planes, runs = [], [], []
    for method in methods:
        for config, label in zip(configs, labels):
            detections = detect_sequence(frames, method, max_depth, config)
            # The keep-downward variant only changes the selection.
            for variant in [False, True] if keep else [False]:
                run = dict(
                    keys, method=method, config=f"{label}{KEEP_SUFFIX * variant}"
                )
                counts = []
                for t, (frame, (all_planes, selected, options, _)) in enumerate(
                    zip(frames, detections)
                ):
                    if variant:
                        options = keep_downward_options(options)
                        selected = select_output_planes(all_planes, options)
                    counts.append(len(selected))
                    board_recs, plane_recs = evaluate_frame(
                        frame, selected, info, options
                    )
                    boards += [dict(run, frame=t, **r) for r in board_recs]
                    planes += [dict(run, frame=t, **r) for r in plane_recs]
                runs.append(
                    dict(
                        run,
                        frames=len(frames),
                        planes_selected=float(np.mean(counts)),
                        detect_ms=float(np.mean([d[3] for d in detections])),
                    )
                )
    return boards, planes, runs


def keep_downward_options(options):
    classify = dataclasses.replace(options.classify, drop_downward=False)
    return dataclasses.replace(options, classify=classify)


def _mean(values):
    values = np.array(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def summarize(boards, planes, n_frames) -> dict:
    """Place metrics over board-frame and plane records of ``n_frames`` frames."""
    targets = [r for r in boards if r["status"] == "target"]
    detected = [r for r in targets if r["detected"]]
    by_board = defaultdict(list)
    for r in detected:
        by_board[(r["scene"], r["name"])].append(r["dh"])
    stds = [np.std(v) for v in by_board.values() if len(v) >= 2]
    n_det = len(detected)

    def label_share(label):
        return sum(r["label"] == label for r in detected) / n_det if n_det else np.nan

    hidden_frames = {
        (r["scene"], r["frame"], name)
        for r in planes
        if r["hidden"]
        for name in r["hidden"].split("+")
    }
    return {
        "targets": len(targets),
        "det_rate": n_det / len(targets) if targets else float("nan"),
        "dh_abs_cm": 100 * _mean(abs(r["dh"]) for r in detected),
        "dh_max_cm": 100 * max((abs(r["dh"]) for r in detected), default=np.nan),
        "dh_std_mm": 1000 * _mean(stds),
        "coverage": _mean(r["coverage"] for r in detected),
        "split": _mean(r["fragments"] >= 2 for r in targets),
        "label_shelf": label_share("shelf"),
        "label_table": label_share("table"),
        "label_other": label_share("other"),
        "merge_bb": sum(r["merge"] == "board-board" for r in planes) / n_frames,
        "merge_bs": sum(r["merge"] == "board-support" for r in planes) / n_frames,
        "object_mix": sum(r["object_mix"] for r in planes) / n_frames,
        "false": sum(bool(r["false_kind"]) for r in planes) / n_frames,
        "false_underside": sum(r["false_kind"] == "underside" for r in planes)
        / n_frames,
        "hidden_det": len(hidden_frames),
        "h_planes": len(planes) / n_frames,
    }


def _fmt(value, digits=2):
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


SUMMARY_COLUMNS = [
    ("targets", "targets", 0),
    ("detected", "det_rate", 2),
    ("abs dh cm", "dh_abs_cm", 2),
    ("max dh cm", "dh_max_cm", 1),
    ("dh std mm", "dh_std_mm", 1),
    ("coverage", "coverage", 2),
    ("split", "split", 2),
    ("shelf", "label_shelf", 2),
    ("table", "label_table", 2),
    ("other", "label_other", 2),
    ("merge b-b /fr", "merge_bb", 2),
    ("merge b-s /fr", "merge_bs", 2),
    ("board+obj /fr", "object_mix", 2),
    ("false /fr", "false", 2),
    ("underside /fr", "false_underside", 2),
    ("hidden det", "hidden_det", 0),
    ("h planes /fr", "h_planes", 1),
]


def _summary_cells(s):
    return [_fmt(s[key], digits) for _, key, digits in SUMMARY_COLUMNS]


def _group_table(boards, planes, runs, group_keys, args):
    """One row per group of ``group_keys`` x preset x method x config."""
    board_groups, plane_groups, frames = defaultdict(list), defaultdict(list), {}
    for r in boards:
        board_groups[tuple(r[k] for k in group_keys)].append(r)
    for r in planes:
        plane_groups[tuple(r[k] for k in group_keys)].append(r)
    for r in runs:
        key = tuple(r[k] for k in group_keys)
        frames[key] = frames.get(key, 0) + r["frames"]
    rows = []
    for key in sorted(frames, key=lambda k: _sort_key(k, group_keys, args)):
        summary = summarize(board_groups[key], plane_groups[key], frames[key])
        rows.append(
            [f"{v:g}" if isinstance(v, float) else str(v) for v in key]
            + _summary_cells(summary)
        )
    header = [k.replace("_", " ") for k in group_keys]
    header += [name for name, _, _ in SUMMARY_COLUMNS]
    return _table(header, rows)


def _sort_key(key, group_keys, args):
    out = []
    for name, value in zip(group_keys, key):
        if name == "preset":
            out.append(args.presets.index(value))
        elif name == "method":
            out.append(args.methods.index(value))
        elif name == "config":
            out.append(args.labels.index(value))
        elif name == "scene":
            out.append(args.scene_names.index(value))
        else:
            out.append(value)
    return tuple(out)


def board_table(boards, args, preset):
    """Per scene x board (preset only): status and per-config place metrics.

    ``+keep`` variants are left out: keeping downward planes never changes a
    board (top faces point up).
    """
    configs = [label for label in args.labels if not label.endswith(KEEP_SUFFIX)]
    rows = []
    groups = defaultdict(list)
    for r in boards:
        if r["preset"] == preset:
            groups[(r["scene"], r["board"], r["method"], r["config"])].append(r)
    keys = sorted(
        {(k[0], k[1], k[2]) for k in groups},
        key=lambda k: (args.scene_names.index(k[0]), k[1], args.methods.index(k[2])),
    )
    for scene, board, method in keys:
        first = groups[(scene, board, method, configs[0])]
        target = [r for r in first if r["status"] == "target"]
        statuses = sorted({r["status"] for r in first})
        row = [
            scene,
            first[0]["name"].split("_")[-1],
            f"{first[0]['height']:.2f}",
            "/".join(statuses),
            _fmt(_mean(r["gt_px"] for r in target), 0) if target else "0",
        ]
        if len(args.methods) > 1:
            row.insert(1, method)
        for config in configs:
            tgt = [
                r
                for r in groups[(scene, board, method, config)]
                if r["status"] == "target"
            ]
            det = [r for r in tgt if r["detected"]]
            labels = sorted({r["label"] for r in det})
            row += [
                f"{len(det)}/{len(tgt)}",
                _fmt(100 * _mean(abs(r["dh"]) for r in det)),
                _fmt(1000 * float(np.std([r["dh"] for r in det])), 1)
                if len(det) >= 2
                else "-",
                _fmt(_mean(r["coverage"] for r in det)),
                "/".join(labels) or "-",
            ]
        rows.append(row)
    header = ["scene", "board", "height m", "status", "GT px"]
    if len(args.methods) > 1:
        header.insert(1, "method")
    for config in configs:
        header += [
            f"{config} det",
            f"{config} abs dh cm",
            f"{config} dh std mm",
            f"{config} cov",
            f"{config} labels",
        ]
    return _table(header, rows)


def event_table(planes, args):
    """Merge errors, false place planes and hidden-board detections, counted."""
    events = defaultdict(int)
    for r in planes:
        kinds = []
        if r["merge"]:
            kinds.append(f"merge {r['merged']}")
        if r["false_kind"]:
            kinds.append(f"false ({r['false_kind']})")
        if r["hidden"]:
            kinds.append(f"on hidden {r['hidden']}")
        for kind in kinds:
            key = (r["scene"], r["preset"], r["method"], r["config"], kind)
            events[key] += 1
    rows = [
        list(key) + [str(n)]
        for key, n in sorted(
            events.items(),
            key=lambda item: (
                args.scene_names.index(item[0][0]),
                args.presets.index(item[0][1]),
                args.methods.index(item[0][2]),
                args.labels.index(item[0][3]),
                item[0][4],
            ),
        )
    ]
    header = ["scene", "preset", "method", "config", "event", "planes"]
    return _table(header, rows) if rows else "None."


def build_report(args, boards, planes, runs, elapsed):
    config_text = "; ".join(
        f"`{label}` = `{text}`" for label, text in zip(args.base_labels, args.configs)
    )
    if args.keep_downward:
        config_text += (
            f"; `<config>{KEEP_SUFFIX}` = same detections, selected with "
            "`ClassifyOptions(drop_downward=False)`"
        )
    lines = [
        "# Shelf place study on synthetic sequences",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`run_shelf.py` in {elapsed / 60:.1f} min",
        f"- CPU: {_cpu_name()}, single thread per process; Python "
        f"{platform.python_version()}, numpy {np.__version__}, scipy "
        f"{scipy.__version__}",
        f"- Scenes `scenes.shelf_scene_set()`, {args.frames} frames per sequence, "
        f"depth {MIN_DEPTH:g}-{args.max_depth:g} m, presets with focal = fx",
        f"- Configurations: {config_text}",
        "- Scored on the node output (`select_output_planes`), horizontal planes "
        "only. Definitions: see README.md (Shelf study)",
        "",
        "## By preset, method and config (all scenes)",
        "",
        _group_table(boards, planes, runs, ["preset", "method", "config"], args),
        "",
        "## By distance to the shelf front (m)",
        "",
        _group_table(
            boards, planes, runs, ["preset", "distance", "method", "config"], args
        ),
        "",
        "## By board spacing (m)",
        "",
        _group_table(
            boards, planes, runs, ["preset", "spacing", "method", "config"], args
        ),
        "",
        "## Per scene",
        "",
        _group_table(
            boards, planes, runs, ["scene", "preset", "method", "config"], args
        ),
        "",
    ]
    for preset in args.presets:
        lines += [
            f"## Per board ({preset})",
            "",
            "`status` over the frames: target = GT top pixels, sliver = top "
            "visible but below the GT size, hidden = no top pixel in range. "
            "`det` = detected / target frames.",
            "",
            board_table(boards, args, preset),
            "",
        ]
    lines += [
        "## Events (merge errors, false place planes, hidden boards)",
        "",
        event_table(planes, args),
    ]
    return "\n".join(lines)


def _board_color(b):
    import matplotlib.pyplot as plt

    return plt.get_cmap("tab10").colors[(b + 1) % 10]


def make_figure(path, scene, preset, configs, labels, method, max_depth):
    """Depth error, GT boards, frame 0 of the first and last config, frame 5 last."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = make_sequence(scene, preset, max_depth, LATER_FRAME + 1)
    info = scene_info(scene, frames[0].intrinsics, max_depth)
    depth_true, _, _ = raycast_faces(scene, frames[0].intrinsics)
    shape = frames[0].labels.shape
    width = shape[1]
    floor_color, table_color = (0.55, 0.65, 0.8), (0.4, 0.75, 0.4)
    panels = [
        (configs[0], labels[0], 0),
        (configs[-1], labels[-1], 0),
        (configs[-1], labels[-1], LATER_FRAME),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6))
    error = (frames[0].cloud[..., 2] - depth_true) * 1e3
    image = axes[0].imshow(error, cmap="RdBu", vmin=-30, vmax=30)
    fig.colorbar(image, ax=axes[0], fraction=0.035)
    axes[0].set_title(f"{preset}\ndepth error mm, frame 0", fontsize=10)

    gt = np.ones(shape + (3,))
    gt[frames[0].labels == -2] = 0.85
    gt[frames[0].labels == -1] = 0.45
    for g, plane in enumerate(frames[0].planes):
        color = (0.75, 0.75, 0.75)
        if plane.surface in dict(info.boards):
            color = _board_color([n for n, _ in info.boards].index(plane.surface))
        elif plane.cls == "floor":
            color = floor_color
        elif plane.cls == "table":
            color = table_color
        gt[frames[0].labels == g] = color
    axes[1].imshow(gt, interpolation="nearest")
    visible = ", ".join(
        f"{h:.2f}" + ("" if px else " (hidden)")
        for (_, h), px in zip(info.boards, info.top_px)
    )
    axes[1].set_title(f"GT boards: {visible}", fontsize=9)

    for ax, (config, label, t) in zip(axes[2:], panels):
        frame = frames[t]
        options = DetectionOptions(
            camera_to_base=frame.camera_to_base,
            min_depth=MIN_DEPTH,
            max_depth=max_depth,
            noise=config.noise(frame.intrinsics.fx),
        )
        planes = detect_planes(
            frame.cloud, method, options, config.detector_params(method)
        )
        selected = select_output_planes(planes, options)
        board_recs, plane_recs = evaluate_frame(frame, selected, info, options)
        horizontal = [p for p in selected if is_horizontal(p.normal, options)]
        image = np.ones((shape[0] * shape[1], 3))
        z = frame.cloud[..., 2].reshape(-1)
        image[np.isfinite(z) & (z >= MIN_DEPTH) & (z <= max_depth)] = 0.85
        for plane in selected:
            if not is_horizontal(plane.normal, options):
                image[plane.inliers] = (0.7, 0.7, 0.95)
        board_names = [name for name, _ in info.boards]
        for plane, rec in zip(horizontal, plane_recs):
            color = (0.6, 0.6, 0.6)
            if rec["surface"] in board_names:
                color = _board_color(board_names.index(rec["surface"]))
            elif rec["surface"] == "floor":
                color = floor_color
            elif rec["surface"]:
                color = table_color
            if rec["false_kind"]:
                color = (0.9, 0.0, 0.7)
            if rec["merge"]:
                color = (1.0, 0.0, 0.0)
            image[plane.inliers] = color
            rows, cols = np.divmod(plane.inliers, width)
            text = f"{rec['label']} {rec['height']:.2f}"
            if rec["merge"]:
                text += " MERGE"
            ax.text(
                np.median(cols),
                np.median(rows),
                text,
                fontsize=7,
                ha="center",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.6, "pad": 1, "lw": 0},
            )
        targets = [r for r in board_recs if r["status"] == "target"]
        n_det = sum(r["detected"] for r in targets)
        ax.imshow(image.reshape(shape + (3,)), interpolation="nearest")
        ax.set_title(
            f"{label}, frame {t}: {len(selected)} planes\n"
            f"boards detected {n_det}/{len(targets)}",
            fontsize=9,
        )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{scene.name}: {method}, max {max_depth:g} m, node output. Board color = "
        "dominant board top, red = merge error, magenta = false place plane, "
        "blue-gray = floor, lilac = wall; text = label and height (m)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=60)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    parser.add_argument("--methods", nargs="+", default=["ransac_gravity_refined"])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="detector configurations, e.g. sub=0.08 sub=0.12,merge=on",
    )
    parser.add_argument("--labels", nargs="*", default=None, help="config labels")
    parser.add_argument(
        "--keep-downward",
        action="store_true",
        help=f"also score each config with ClassifyOptions(drop_downward=False) "
        f"(config label suffix '{KEEP_SUFFIX}')",
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--figures",
        nargs="*",
        default=DEFAULT_FIGURES,
        help="scenes for shelf_<scene>.png (first preset); none with no names",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "shelf_report.md"
    )
    args = parser.parse_args()
    for name, known, valid in (
        ("methods", args.methods, DETECTORS),
        ("presets", args.presets, NOISE_PRESETS),
    ):
        unknown = [v for v in known if v not in valid]
        if unknown:
            parser.error(f"unknown {name} {unknown}, choose from {sorted(valid)}")
    try:
        configs = [parse_config(text, args.methods) for text in args.configs]
    except ValueError as error:
        parser.error(str(error))
    if args.labels is None:
        default = args.configs == DEFAULT_CONFIGS
        args.labels = DEFAULT_LABELS if default else [c.label for c in configs]
    if len(args.labels) != len(configs):
        parser.error("--labels needs one label per config")
    scenes = [s for s in shelf_scene_set() if not args.scenes or s.name in args.scenes]
    if not scenes:
        parser.error("no matching scenes")
    args.scene_names = [s.name for s in scenes]
    base_labels = args.base_labels = list(args.labels)
    if args.keep_downward:
        args.labels = [
            label + suffix for label in base_labels for suffix in ("", KEEP_SUFFIX)
        ]
    jobs = [
        (
            scene.name,
            preset,
            args.max_depth,
            args.frames,
            args.methods,
            configs,
            base_labels,
            args.keep_downward,
        )
        for scene in scenes
        for preset in args.presets
    ]
    started = time.perf_counter()
    boards, planes, runs = [], [], []
    with contextlib.ExitStack() as stack:
        if args.jobs > 1:
            pool = stack.enter_context(ProcessPoolExecutor(args.jobs))
            results = pool.map(run_job, jobs)
        else:
            results = map(run_job, jobs)
        for job, (b, p, r) in zip(jobs, results):
            boards += b
            planes += p
            runs += r
            for run in r:
                s = summarize(
                    [x for x in b if x["config"] == run["config"]],
                    [x for x in p if x["config"] == run["config"]],
                    run["frames"],
                )
                print(
                    f"[{time.perf_counter() - started:6.1f} s] {job[0]} {job[1]} "
                    f"{run['method']} [{run['config']}]: detected "
                    f"{_fmt(s['det_rate'])}, abs dh {_fmt(s['dh_abs_cm'])} cm",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report(args, boards, planes, runs, time.perf_counter() - started) + "\n"
    )
    _write_csv(args.out.with_suffix(".csv"), boards)
    _write_csv(args.out.with_name(args.out.stem + "_planes.csv"), planes)
    print(f"wrote {args.out} and its CSV files", flush=True)
    for name in args.figures:
        scene = next((s for s in shelf_scene_set() if s.name == name), None)
        if scene is None:
            parser.error(f"unknown figure scene {name}")
        path = args.out.parent / f"{name}.png"
        make_figure(
            path,
            scene,
            args.presets[0],
            configs,
            base_labels,
            args.methods[0],
            args.max_depth,
        )
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
