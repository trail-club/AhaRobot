"""Place candidates on synthetic sequences: validity, availability, area, stability.

Scenes with table tops and shelf boards (``scene_set``, ``shelf_scene_set``,
``place_scene_set``) are detected frame by frame with the node configuration
(``ransac_gravity_refined``, merge on, subpixel 0.12, ``select_output_planes``)
and ``aha_perception.place.find_place_candidates`` runs on the output for
several object sizes and ``PlaceOptions`` variants. Ground truth is analytic:
support rectangles (the +z faces of table / shelf boxes), footprints (xy) of
every primitive that reaches into the band of the object height above the
surface (including upper boards and side panels), and the visibility of the
surface from a noise-free ray cast within the depth range. GT placeable
centers keep the footprint radius r = hypot(x, y) / 2 from the rectangle edge
and the footprints, without margins, on a GT_RES grid. Object box tops of at
least OBJECT_MIN_TOP height and OBJECT_MIN_SIZE side are kept only to
classify candidates. See README.md (place study) for the metrics.
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
from scipy.ndimage import distance_transform_edt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"))
sys.path.insert(0, str(HERE))

from aha_perception.place import (  # noqa: E402
    PlaceOptions,
    base_cloud,
    find_place_candidates,
    footprint_radius,
)
from aha_perception.planes import (  # noqa: E402
    DetectionOptions,
    detect_planes,
    select_output_planes,
)
from configs import parse_config  # noqa: E402
from run_bench import _cpu_name, _table, _write_csv  # noqa: E402
from run_sequence import MIN_DEPTH, make_sequence  # noqa: E402
from scenes import (  # noqa: E402
    _FACES,
    NOISE_PRESETS,
    Box,
    Cylinder,
    place_scene_set,
    raycast_faces,
    scene_set,
    shelf_scene_set,
)

METHOD = "ransac_gravity_refined"
NODE_CONFIG = "sub=0.12,merge=on"
DEFAULT_SIZES = ["0.06x0.06x0.12", "0.10x0.10x0.20", "0.20x0.15x0.10"]
DEFAULT_PRESETS = ["d435i_filtered", "d435i_raw", "default"]
DEFAULT_FIGURES = [
    "place_table_d1.0:1",
    "place_crowded_d1.0:0",
    "place_shelf_low_d1.0:1",
]
SUPPORT_CLASSES = ("table", "shelf")
GT_RES = 0.005  # m, GT grid on each surface
GT_PAD = 0.15  # m, GT grid extent beyond the surface rectangle
HEIGHT_TOL = 0.04  # m, candidate / plane height to a GT surface
AVAIL_SLACK = 0.04  # m, GT free slack beyond the radius for "available"
BAND_EPS = 1e-3  # m, primitives must reach this much above the surface
OBJECT_MIN_TOP = 0.05  # m, object box tops scored as incidental surfaces
OBJECT_MIN_SIZE = 0.05  # m, smallest side of such a top
SAME_TOP = 0.02  # m, top candidate counts as unchanged between frames
EVAL_MAX_CANDIDATES = 50
TOP_FACE = _FACES.index("+z")


def parse_size(text: str) -> tuple:
    values = tuple(float(v) for v in text.lower().split("x"))
    if len(values) != 3 or min(values) <= 0:
        raise ValueError(f"size must be XxYxZ in m, got {text}")
    return values


def parse_variant(text: str) -> PlaceOptions:
    """``PlaceOptions`` from ``KEY=VALUE,...``; ``default`` or empty = defaults."""
    fields = {f.name: f.default for f in dataclasses.fields(PlaceOptions)}
    values = {}
    for item in filter(None, text.split(",")):
        if item == "default":
            continue
        key, sep, value = item.partition("=")
        if not sep or key not in fields:
            raise ValueError(f"unknown PlaceOptions item {item}")
        values[key] = type(fields[key])(value)
    return PlaceOptions(**values)


def scene_groups() -> dict:
    bench = [
        s
        for s in scene_set()
        if any(cls in SUPPORT_CLASSES for cls in s.surfaces.values())
    ]
    return {"bench": bench, "shelf": shelf_scene_set(), "place": place_scene_set()}


def find_scene(name):
    for group, scenes in scene_groups().items():
        for scene in scenes:
            if scene.name == name:
                return group, scene
    raise KeyError(name)


# Signed distances in the xy plane (negative inside).


def _rect_sdf(xy, center, half, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    d = xy - center
    local = np.stack(
        [c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1
    )
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    return outside + np.minimum(q.max(axis=-1), 0.0)


def _segment_distance(xy, a, b):
    ab = b - a
    t = np.clip(((xy - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(xy - (a + t[..., None] * ab), axis=-1)


def _polygon_sdf(xy, polygon):
    edges = [
        _segment_distance(xy, a, b)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
    ]
    dist = np.min(edges, axis=0)
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area) < 1e-8:
        return dist
    inside = np.ones(xy.shape[:-1], dtype=bool)
    dx, dy = xy[..., 0], xy[..., 1]
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        cross = (b[0] - a[0]) * (dy - a[1]) - (b[1] - a[1]) * (dx - a[0])
        inside &= np.sign(area) * cross >= 0
    return np.where(inside, -dist, dist)


@dataclass
class Footprint:
    """Vertical extent and xy signed distance of one primitive."""

    z0: float
    z1: float
    sdf: object


def footprint(primitive) -> Footprint:
    if isinstance(primitive, Box):
        center = np.asarray(primitive.center, dtype=np.float64)
        half = np.asarray(primitive.size, dtype=np.float64) / 2
        yaw = np.radians(primitive.yaw_deg)
        return Footprint(
            center[2] - half[2],
            center[2] + half[2],
            lambda xy: _rect_sdf(xy, center[:2], half[:2], yaw),
        )
    if isinstance(primitive, Cylinder):
        center = np.asarray(primitive.center_xy, dtype=np.float64)
        return Footprint(
            primitive.z0,
            primitive.z1,
            lambda xy: np.linalg.norm(xy - center, axis=-1) - primitive.radius,
        )
    o, u, v = (
        np.asarray(a, dtype=np.float64)
        for a in (primitive.origin, primitive.edge_u, primitive.edge_v)
    )
    corners = np.array([o, o + u, o + u + v, o + v])
    polygon = corners[:, :2]
    return Footprint(
        float(corners[:, 2].min()),
        float(corners[:, 2].max()),
        lambda xy: _polygon_sdf(xy, polygon),
    )


@dataclass
class Surface:
    """A box top and its GT grid: a named support (table top, shelf board) or
    the top of an object box, which only serves to score candidates on it."""

    name: str
    cls: str
    named: bool
    primitive: int
    center: np.ndarray
    half: np.ndarray
    yaw: float
    height: float
    local: np.ndarray  # (n, m, 2) grid in the rectangle frame
    xy: np.ndarray  # (n, m, 2) grid in the base frame
    edge: np.ndarray  # (n, m) distance to the rectangle edge, positive inside
    visible: np.ndarray  # (n, m) top face seen in the noise-free ray cast
    visible_dist: np.ndarray  # (n, m) distance to the nearest unseen cell

    def edge_at(self, xy) -> np.ndarray:
        return -_rect_sdf(xy, self.center, self.half, self.yaw)


def scene_surfaces(scene, intrinsics, max_depth) -> list[Surface]:
    depth, index, faces = raycast_faces(scene, intrinsics)
    in_range = np.isfinite(depth) & (depth >= MIN_DEPTH) & (depth <= max_depth)
    transform = scene.pose.camera_to_base()
    rotation, origin = transform[:3, :3], transform[:3, 3]
    out = []
    for k, primitive in enumerate(scene.primitives):
        if not isinstance(primitive, Box):
            continue
        name = primitive.faces.get("+z")
        named = isinstance(name, str) and scene.surfaces.get(name) in SUPPORT_CLASSES
        height = float(primitive.center[2] + primitive.size[2] / 2)
        if not named:
            if height < OBJECT_MIN_TOP or min(primitive.size[:2]) < OBJECT_MIN_SIZE:
                continue
            name = f"object{k}"
        center = np.asarray(primitive.center[:2], dtype=np.float64)
        half = np.asarray(primitive.size[:2], dtype=np.float64) / 2
        yaw = float(np.radians(primitive.yaw_deg))
        lx = np.arange(-half[0] - GT_PAD, half[0] + GT_PAD + 1e-9, GT_RES)
        ly = np.arange(-half[1] - GT_PAD, half[1] + GT_PAD + 1e-9, GT_RES)
        local = np.stack(np.meshgrid(lx, ly, indexing="ij"), -1)
        c, s = np.cos(yaw), np.sin(yaw)
        xy = center + np.stack(
            [
                c * local[..., 0] - s * local[..., 1],
                s * local[..., 0] + c * local[..., 1],
            ],
            -1,
        )
        edge = -_rect_sdf(xy, center, half, yaw)
        points = np.concatenate([xy, np.full(xy.shape[:-1] + (1,), height)], -1)
        cam = (points - origin) @ rotation
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(intrinsics.fx * cam[..., 0] / cam[..., 2] + intrinsics.cx)
            v = np.round(intrinsics.fy * cam[..., 1] / cam[..., 2] + intrinsics.cy)
        ok = (cam[..., 2] > 0) & (u >= 0) & (u < intrinsics.width)
        ok &= (v >= 0) & (v < intrinsics.height) & (edge > 0)
        ui = np.where(ok, u, 0).astype(int)
        vi = np.where(ok, v, 0).astype(int)
        visible = ok & (index[vi, ui] == k) & (faces[vi, ui] == TOP_FACE)
        visible &= in_range[vi, ui]
        out.append(
            Surface(
                name=name,
                cls=scene.surfaces[name] if named else "object",
                named=named,
                primitive=k,
                center=center,
                half=half,
                yaw=yaw,
                height=height,
                local=local,
                xy=xy,
                edge=edge,
                visible=visible,
                visible_dist=distance_transform_edt(visible) * GT_RES,
            )
        )
    return out


@dataclass
class SurfaceGT:
    """GT for one surface and object size."""

    surface: Surface
    obstacles: list
    obstacle: np.ndarray  # (n, m) signed distance to the nearest obstacle
    place: np.ndarray  # GT placeable centers (no margins)
    visible_place: np.ndarray  # placeable and the whole disc visible
    available: bool  # a visible center with AVAIL_SLACK of free slack

    def obstacle_at(self, xy) -> np.ndarray:
        dist = np.full(np.shape(xy)[:-1], np.inf)
        for fp in self.obstacles:
            dist = np.minimum(dist, fp.sdf(xy))
        return dist


def surface_gt(surface, scene, size) -> SurfaceGT:
    radius = footprint_radius(size)
    top = surface.height + size[2]
    obstacles = []
    for k, primitive in enumerate(scene.primitives):
        if k == surface.primitive:
            continue
        fp = footprint(primitive)
        if fp.z1 > surface.height + BAND_EPS and fp.z0 < top:
            obstacles.append(fp)
    gt = SurfaceGT(surface, obstacles, None, None, None, False)
    gt.obstacle = gt.obstacle_at(surface.xy)
    clear = np.minimum(surface.edge, gt.obstacle)
    gt.place = clear >= radius
    gt.visible_place = gt.place & (surface.visible_dist >= radius)
    slack = radius + AVAIL_SLACK
    gt.available = bool(((clear >= slack) & (surface.visible_dist >= slack)).any())
    return gt


def score_candidate(candidate, gts, radius) -> dict:
    """GT surface, footprint violations and clearance error of one candidate."""
    xy = np.asarray(candidate.position[:2], dtype=np.float64)
    best, best_edge = None, -np.inf
    for g, gt in enumerate(gts):
        if abs(candidate.position[2] - gt.surface.height) >= HEIGHT_TOL:
            continue
        edge = float(gt.surface.edge_at(xy))
        if edge > best_edge:
            best, best_edge = g, edge
    rec = {
        "surface": "",
        "off_surface": best is None,
        "on_object": best is not None and not gts[best].surface.named,
        "overhang": float("nan"),
        "collision": float("nan"),
        "violation": float("nan"),
        "valid": False,
        "clear_err": float("nan"),
        "dz": float("nan"),
    }
    if best is None:
        return rec
    gt = gts[best]
    obstacle = float(gt.obstacle_at(xy))
    overhang = max(radius - best_edge, 0.0)
    collision = max(radius - obstacle, 0.0)
    rec.update(
        surface=gt.surface.name,
        surface_index=best,
        overhang=overhang,
        collision=collision,
        violation=max(overhang, collision),
        valid=overhang <= 0.0 and collision <= 0.0,
        clear_err=candidate.clearance - min(best_edge, obstacle),
        dz=float(candidate.position[2] - gt.surface.height),
    )
    return rec


def predicted_place(result, surface) -> np.ndarray:
    """Union of the placeable rasters of planes at the surface height."""
    points = np.concatenate(
        [surface.xy, np.full(surface.xy.shape[:-1] + (1,), surface.height)], -1
    ).reshape(-1, 3)
    pred = np.zeros(len(points), dtype=bool)
    for grid in result.grids:
        if abs(float(grid.origin[2]) - surface.height) < HEIGHT_TOL:
            pred |= grid.sample(grid.placeable, points)
    return pred.reshape(surface.xy.shape[:-1])


def evaluate_frame(result, gts, radius):
    cand_recs = [
        dict(
            score_candidate(c, gts, radius),
            rank=r,
            label=c.label,
            clearance=c.clearance,
        )
        for r, c in enumerate(result.candidates)
    ]
    surf_recs = []
    cell = GT_RES * GT_RES
    for g, gt in enumerate(gts):
        if not gt.surface.named:
            continue
        on = [c for c in cand_recs if c.get("surface_index") == g]
        pred = predicted_place(result, gt.surface)
        truth = gt.visible_place
        surf_recs.append(
            {
                "surface": gt.surface.name,
                "class": gt.surface.cls,
                "height": gt.surface.height,
                "visible": bool(gt.surface.visible.any()),
                "available": gt.available,
                "has_any": bool(on),
                "has_valid": any(c["valid"] for c in on),
                "gt_area": float(gt.place.sum() * cell),
                "gt_vis_area": float(truth.sum() * cell),
                "pred_area": float(pred.sum() * cell),
                "inter_area": float((pred & truth).sum() * cell),
                "union_area": float((pred | truth).sum() * cell),
                "false_area": float((pred & ~gt.place).sum() * cell),
            }
        )
    return cand_recs, surf_recs


def node_options(frame, max_depth, config):
    return DetectionOptions(
        camera_to_base=frame.camera_to_base,
        min_depth=MIN_DEPTH,
        max_depth=max_depth,
        noise=config.noise(frame.intrinsics.fx),
    )


def node_config(node_stride: int):
    return parse_config(
        f"{NODE_CONFIG},node_stride={node_stride}", [METHOD], node_stride=True
    )


def detect(frame, max_depth, config):
    options = node_options(frame, max_depth, config)
    cloud = config.decimate(frame.cloud)
    planes = detect_planes(cloud, METHOD, options, config.detector_params(METHOD))
    points, depth = base_cloud(cloud, frame.camera_to_base)
    return select_output_planes(planes, options), points, depth, options.noise


def run_job(job):
    """Frame, candidate and surface records of one (scene, preset) sequence."""
    name, preset, max_depth, n_frames, sizes, variants, labels, node_stride = job
    group, scene = find_scene(name)
    config = node_config(node_stride)
    variants = [v.for_input_stride(node_stride) for v in variants]
    frames = make_sequence(scene, preset, max_depth, n_frames)
    surfaces = scene_surfaces(scene, frames[0].intrinsics, max_depth)
    gts = {size: [surface_gt(s, scene, size) for s in surfaces] for size in sizes}
    detections = [detect(frame, max_depth, config) for frame in frames]
    frame_recs, cand_recs, surf_recs = [], [], []
    for size in sizes:
        radius = footprint_radius(size)
        for variant, label in zip(variants, labels):
            keys = {
                "scene": name,
                "group": group,
                "preset": preset,
                "size": "x".join(f"{v:g}" for v in size),
                "variant": label,
            }
            for t, (planes, points, depth, noise) in enumerate(detections):
                start = time.perf_counter()
                result = find_place_candidates(
                    planes,
                    points,
                    depth,
                    size,
                    noise,
                    variant,
                    max_candidates=EVAL_MAX_CANDIDATES,
                )
                ms = (time.perf_counter() - start) * 1e3
                cands, surfs = evaluate_frame(result, gts[size], radius)
                top = result.candidates[0].position if result.candidates else None
                frame_recs.append(
                    dict(
                        keys,
                        frame=t,
                        ms=ms,
                        candidates=len(cands),
                        top_valid=cands[0]["valid"] if cands else float("nan"),
                        top_x=float(top[0]) if top is not None else float("nan"),
                        top_y=float(top[1]) if top is not None else float("nan"),
                        top_z=float(top[2]) if top is not None else float("nan"),
                    )
                )
                cand_recs += [dict(keys, frame=t, **c) for c in cands]
                surf_recs += [dict(keys, frame=t, **s) for s in surfs]
    return frame_recs, cand_recs, surf_recs


def _mean(values):
    values = np.array([float(v) for v in values], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def _ratio(num, den):
    return num / den if den else float("nan")


def summarize(frames, cands, surfs) -> dict:
    invalid = [c for c in cands if not c["valid"]]
    available = [s for s in surfs if s["available"]]
    scored = [s for s in surfs if s["union_area"] > 0]
    tops = defaultdict(list)
    for f in sorted(frames, key=lambda f: f["frame"]):
        key = (f["scene"], f["preset"], f["size"], f["variant"])
        tops[key].append((f["top_x"], f["top_y"], f["top_z"]))
    steps = []
    for positions in tops.values():
        p = np.array(positions)
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        steps += list(d[np.isfinite(d)])
    steps = np.array(steps)
    return {
        "candidates": len(cands),
        "per_frame": _ratio(len(cands), len(frames)),
        "validity": _ratio(len(cands) - len(invalid), len(cands)),
        "top_validity": _mean(f["top_valid"] for f in frames),
        "off_surface": _ratio(sum(c["off_surface"] for c in cands), len(cands)),
        "on_object": _ratio(sum(c["on_object"] for c in cands), len(cands)),
        "violation_cm": 100 * _mean(c["violation"] for c in invalid),
        "violation_max_cm": 100
        * max(
            (c["violation"] for c in invalid if np.isfinite(c["violation"])),
            default=np.nan,
        ),
        "available": len(available),
        "availability": _mean(s["has_valid"] for s in available),
        "no_place": _mean(not s["has_any"] for s in available),
        "iou": _mean(s["inter_area"] / s["union_area"] for s in scored),
        "area_ratio": _ratio(
            sum(s["pred_area"] for s in surfs), sum(s["gt_vis_area"] for s in surfs)
        ),
        "false_area": _ratio(
            sum(s["false_area"] for s in surfs), sum(s["pred_area"] for s in surfs)
        ),
        "clear_err_cm": 100 * _mean(c["clear_err"] for c in cands if c["valid"]),
        "jitter_cm": 100 * float(steps.mean()) if len(steps) else float("nan"),
        "top_same": float((steps < SAME_TOP).mean()) if len(steps) else float("nan"),
        "ms": _mean(f["ms"] for f in frames),
        "ms_max": max((f["ms"] for f in frames), default=float("nan")),
    }


SEQUENCE_KEYS = ["scene", "group", "preset", "size", "variant"]


def grouped_summaries(frames, cands, surfs, keys) -> dict:
    """``summarize`` of the records grouped by the values of ``keys``."""
    groups = [defaultdict(list) for _ in range(3)]
    for records, grouped in zip((frames, cands, surfs), groups):
        for r in records:
            grouped[tuple(r[k] for k in keys)].append(r)
    return {
        key: summarize(rows, groups[1][key], groups[2][key])
        for key, rows in groups[0].items()
    }


def sequence_rows(frames, cands, surfs) -> list[dict]:
    """One summary row per scene x preset x size x variant."""
    summaries = grouped_summaries(frames, cands, surfs, SEQUENCE_KEYS)
    return [dict(zip(SEQUENCE_KEYS, key), **s) for key, s in summaries.items()]


def _fmt(value, digits=2):
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


SUMMARY_COLUMNS = [
    ("cand /fr", "per_frame", 1),
    ("validity", "validity", 3),
    ("top1 valid", "top_validity", 3),
    ("off surf", "off_surface", 3),
    ("on obj", "on_object", 3),
    ("viol cm", "violation_cm", 1),
    ("max viol cm", "violation_max_cm", 1),
    ("avail n", "available", 0),
    ("availability", "availability", 2),
    ("no place", "no_place", 2),
    ("IoU", "iou", 2),
    ("pred/GT", "area_ratio", 2),
    ("false area", "false_area", 3),
    ("clear err cm", "clear_err_cm", 1),
    ("top jitter cm", "jitter_cm", 1),
    ("top same", "top_same", 2),
    ("ms", "ms", 1),
    ("max ms", "ms_max", 0),
]


def _order_key(order, keys, values):
    return tuple(order(name, value) for name, value in zip(keys, values))


def _group_table(frames, cands, surfs, keys, order):
    summaries = grouped_summaries(frames, cands, surfs, keys)
    rows = []
    for key in sorted(summaries, key=lambda k: _order_key(order, keys, k)):
        s = summaries[key]
        rows.append(list(key) + [_fmt(s[k], d) for _, k, d in SUMMARY_COLUMNS])
    return _table(keys + [name for name, _, _ in SUMMARY_COLUMNS], rows)


def failure_table(cands, order):
    """Invalid candidates by scene, preset, size and variant."""
    groups = defaultdict(list)
    for c in cands:
        if not c["valid"]:
            groups[(c["scene"], c["preset"], c["size"], c["variant"])].append(c)
    rows = []
    keys = ["scene", "preset", "size", "variant"]
    for key in sorted(groups, key=lambda k: _order_key(order, keys, k)):
        items = groups[key]
        rows.append(
            list(key)
            + [
                str(len(items)),
                str(sum(c["off_surface"] for c in items)),
                str(sum(c["collision"] > 0 for c in items)),
                str(sum(c["overhang"] > 0 for c in items)),
                _fmt(100 * _mean(c["violation"] for c in items), 1),
                _fmt(_mean(c["rank"] + 1 for c in items), 1),
            ]
        )
    header = keys + ["invalid", "off surface", "collision", "overhang", "viol cm"]
    header.append("mean rank")
    return _table(header, rows) if rows else "None."


def build_report(args, frames, cands, surfs, elapsed):
    def order(name, value):
        lists = {
            "preset": args.presets,
            "size": args.size_names,
            "variant": args.labels,
            "scene": args.scene_names,
            "group": ["bench", "shelf", "place"],
        }
        return lists[name].index(value) if name in lists else value

    variants = "; ".join(
        f"`{label}` = `{text}`" for label, text in zip(args.labels, args.variants)
    )
    first = args.labels[0]
    main = [r for r in frames if r["variant"] == first]
    main_c = [r for r in cands if r["variant"] == first]
    main_s = [r for r in surfs if r["variant"] == first]
    defaults = ", ".join(
        f"{f.name}={getattr(PlaceOptions(), f.name)}"
        for f in dataclasses.fields(PlaceOptions)
    )
    lines = [
        "# Place candidates on synthetic sequences",
        "",
        f"- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`run_place.py` in {elapsed / 60:.1f} min",
        f"- CPU: {_cpu_name()}, single thread per process; Python "
        f"{platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}",
        f"- {len(args.scene_names)} scenes, {args.frames} frames per sequence, depth "
        f"{MIN_DEPTH:g}-{args.max_depth:g} m, detector `{METHOD}` `{NODE_CONFIG}` "
        f"(node output, node stride {args.node_stride}; variants through "
        "`PlaceOptions.for_input_stride`), presets with focal = fx",
        f"- Object sizes (x x y x z m): {', '.join(args.size_names)}; up to "
        f"{EVAL_MAX_CANDIDATES} candidates per call",
        f"- Variants: {variants}",
        f"- PlaceOptions defaults: {defaults}",
        f"- GT: grid {GT_RES * 1000:g} mm, surface height tolerance {HEIGHT_TOL * 100:g} cm, "
        f"available = a visible center with {AVAIL_SLACK * 100:g} cm slack. "
        "Definitions: README.md (place study)",
        "",
        "## By variant, preset and size (all scenes)",
        "",
        _group_table(frames, cands, surfs, ["variant", "preset", "size"], order),
        "",
        f"## By scene group (`{first}`)",
        "",
        _group_table(main, main_c, main_s, ["group", "preset", "size"], order),
        "",
    ]
    for preset in args.presets:
        lines += [
            f"## Per scene ({preset}, `{first}`)",
            "",
            _group_table(
                [r for r in main if r["preset"] == preset],
                [r for r in main_c if r["preset"] == preset],
                [r for r in main_s if r["preset"] == preset],
                ["scene", "size"],
                order,
            ),
            "",
        ]
    lines += ["## Invalid candidates", "", failure_table(cands, order)]
    return "\n".join(lines)


def _project(points, frame):
    transform = frame.camera_to_base
    cam = (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]
    k = frame.intrinsics
    return np.stack(
        [k.fx * cam[:, 0] / cam[:, 2] + k.cx, k.fy * cam[:, 1] / cam[:, 2] + k.cy], -1
    )


def make_figure(path, scene, preset, size, options, label, max_depth, node_stride=1):
    """Depth error, candidates in the image and top views of the supports."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = node_config(node_stride)
    frame = make_sequence(scene, preset, max_depth, 1)[0]
    planes, points, depth, noise = detect(frame, max_depth, config)
    result = find_place_candidates(
        planes,
        points,
        depth,
        size,
        noise,
        options.for_input_stride(node_stride),
        max_candidates=EVAL_MAX_CANDIDATES,
    )
    radius = result.radius
    surfaces = scene_surfaces(scene, frame.intrinsics, max_depth)
    gts = [surface_gt(s, scene, size) for s in surfaces]
    cands = [score_candidate(c, gts, radius) for c in result.candidates]
    shown = sorted(
        [g for g in gts if g.surface.named and g.surface.visible.any()],
        key=lambda g: -g.surface.visible.sum(),
    )[:3]
    shown.sort(key=lambda g: g.surface.height)

    fig, axes = plt.subplots(
        1, 2 + len(shown), figsize=(6 * (2 + len(shown)), 5.2), squeeze=False
    )
    axes = axes[0]
    depth_true, _, _ = raycast_faces(scene, frame.intrinsics)
    error = (frame.cloud[..., 2] - depth_true) * 1e3
    image = axes[0].imshow(error, cmap="RdBu", vmin=-30, vmax=30)
    fig.colorbar(image, ax=axes[0], fraction=0.035)
    axes[0].set_title(f"{preset}: depth error mm, frame 0", fontsize=10)

    axes[1].imshow(frame.cloud[..., 2], cmap="gray_r", vmin=0.3, vmax=max_depth)
    circle = np.linspace(0, 2 * np.pi, 48)
    for rank, (c, rec) in enumerate(zip(result.candidates, cands)):
        ring = c.position + radius * np.stack(
            [np.cos(circle), np.sin(circle), np.zeros_like(circle)], -1
        )
        uv = _project(ring, frame)
        color = "tab:green" if rec["valid"] else "red"
        axes[1].plot(uv[:, 0], uv[:, 1], color=color, lw=1.5 if rank else 2.5)
        center = _project(c.position[None], frame)[0]
        axes[1].text(
            center[0], center[1], str(rank + 1), color=color, fontsize=7, ha="center"
        )
    valid = sum(rec["valid"] for rec in cands)
    axes[1].set_title(
        f"candidates (r = {radius * 100:.1f} cm): {valid}/{len(cands)} valid",
        fontsize=10,
    )
    axes[1].set_xlim(0, frame.intrinsics.width)
    axes[1].set_ylim(frame.intrinsics.height, 0)

    for ax, gt in zip(axes[2:], shown):
        s = gt.surface
        pred = predicted_place(result, s)
        rgb = np.ones(s.edge.shape + (3,))
        inside = s.edge > 0
        rgb[inside] = (0.93, 0.9, 0.82)
        rgb[inside & ~s.visible] = (0.8, 0.8, 0.8)
        rgb[gt.place & ~gt.visible_place] = (0.82, 0.95, 0.82)
        rgb[gt.visible_place] = (0.55, 0.85, 0.55)
        rgb[gt.obstacle < 0] = (0.35, 0.35, 0.35)
        rgb[pred & gt.place] = (0.2, 0.4, 0.9)
        rgb[pred & ~gt.place] = (0.95, 0.1, 0.1)
        lx, ly = s.local[:, 0, 0], s.local[0, :, 1]
        ax.imshow(
            rgb.transpose(1, 0, 2),
            origin="lower",
            extent=(lx[0], lx[-1], ly[0], ly[-1]),
            interpolation="nearest",
        )
        c, sn = np.cos(s.yaw), np.sin(s.yaw)
        for rank, (cand, rec) in enumerate(zip(result.candidates, cands)):
            if rec["surface"] != s.name:
                continue
            d = cand.position[:2] - s.center
            local = (c * d[0] + sn * d[1], -sn * d[0] + c * d[1])
            color = "tab:orange" if rec["valid"] else "red"
            ax.add_patch(plt.Circle(local, radius, fill=False, color=color, lw=1.5))
            ax.text(*local, str(rank + 1), fontsize=7, ha="center", va="center")
        truth = gt.visible_place.sum()
        iou = (pred & gt.visible_place).sum() / max((pred | gt.visible_place).sum(), 1)
        ax.set_title(
            f"{s.name} {s.height:.2f} m: GT visible place {truth * GT_RES**2 * 1e4:.0f} cm2, "
            f"IoU {iou:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("x on the surface (m)")
        ax.set_ylabel("y (m)")
    for ax in axes[:2]:
        ax.axis("off")
    fig.suptitle(
        f"{scene.name}: object {' x '.join(f'{v:g}' for v in size)} m, `{label}`. "
        "Top views: dark gray = GT obstacle, light gray = unseen, green = GT placeable "
        "(pale: not fully visible), blue = predicted placeable, red = predicted but not GT",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=60)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--groups", nargs="+", default=["bench", "shelf", "place"], help="scene groups"
    )
    parser.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, help="XxYxZ (m)")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["default"],
        help="PlaceOptions overrides, e.g. default obstacle_k=2.5,edge_margin=0.03",
    )
    parser.add_argument("--labels", nargs="*", default=None, help="variant labels")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument(
        "--node-stride",
        type=int,
        default=1,
        help="node parameter stride (every N-th row and column of the image)",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--figures",
        nargs="*",
        default=DEFAULT_FIGURES,
        help="<scene>[:<size index>] for place_<scene>.png (place_ added unless "
        "the name has it; first preset and variant); none with no names",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "place_report.md"
    )
    args = parser.parse_args()
    unknown = [p for p in args.presets if p not in NOISE_PRESETS]
    if unknown:
        parser.error(f"unknown presets {unknown}, choose from {sorted(NOISE_PRESETS)}")
    if args.node_stride < 1:
        parser.error("--node-stride must be >= 1")
    try:
        sizes = [parse_size(text) for text in args.sizes]
        options = [parse_variant(text) for text in args.variants]
    except ValueError as error:
        parser.error(str(error))
    args.size_names = ["x".join(f"{v:g}" for v in size) for size in sizes]
    if args.labels is None:
        args.labels = list(args.variants)
    if len(args.labels) != len(options):
        parser.error("--labels needs one label per variant")
    groups = scene_groups()
    scenes = [
        s
        for g in args.groups
        for s in groups[g]
        if not args.scenes or s.name in args.scenes
    ]
    if not scenes:
        parser.error("no matching scenes")
    args.scene_names = [s.name for s in scenes]
    jobs = [
        (
            scene.name,
            preset,
            args.max_depth,
            args.frames,
            sizes,
            options,
            args.labels,
            args.node_stride,
        )
        for scene in scenes
        for preset in args.presets
    ]
    started = time.perf_counter()
    frames, cands, surfs = [], [], []
    with contextlib.ExitStack() as stack:
        if args.jobs > 1:
            pool = stack.enter_context(ProcessPoolExecutor(args.jobs))
            results = pool.map(run_job, jobs)
        else:
            results = map(run_job, jobs)
        for job, (f, c, s) in zip(jobs, results):
            frames += f
            cands += c
            surfs += s
            summary = summarize(f, c, s)
            print(
                f"[{time.perf_counter() - started:6.1f} s] {job[0]} {job[1]}: validity "
                f"{_fmt(summary['validity'], 3)}, availability "
                f"{_fmt(summary['availability'])}, IoU {_fmt(summary['iou'])}",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report(args, frames, cands, surfs, time.perf_counter() - started) + "\n"
    )
    _write_csv(args.out.with_suffix(".csv"), sequence_rows(frames, cands, surfs))
    invalid = [c for c in cands if not c["valid"]]
    _write_csv(args.out.with_name(args.out.stem + "_invalid.csv"), invalid)
    print(f"wrote {args.out} and its CSV files", flush=True)
    for item in args.figures:
        name, _, index = item.partition(":")
        try:
            _, scene = find_scene(name)
            size = sizes[int(index or 0)]
        except (KeyError, ValueError, IndexError):
            parser.error(f"unknown figure scene or size index {item}")
        stem = name if name.startswith("place_") else f"place_{name}"
        path = args.out.parent / f"{stem}.png"
        make_figure(
            path,
            scene,
            args.presets[0],
            size,
            options[0],
            args.labels[0],
            args.max_depth,
            args.node_stride,
        )
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
