"""Label figures comparing detector configurations on sequences and real frames.

Synthetic scenes: one row per noise preset x configuration with the frame 0
depth error, the GT and the detections (frame 0 all / selected, frame 5 all),
colored by the matched GT plane. Real scenes: one row per configuration with
the color image and the same detection panels. See README.md.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "overlay_ws" / "src" / "aha_perception"))
sys.path.insert(0, str(HERE))

from aha_perception.plane_detection_io import depth_to_cloud, plane_palette  # noqa: E402
from aha_perception.planes import (  # noqa: E402
    DetectionOptions,
    detect_planes,
    select_output_planes,
)
from configs import parse_config  # noqa: E402
from metrics import evaluate  # noqa: E402
from real_frames import load_scene  # noqa: E402
from run_real import SMALL_AREA, _options  # noqa: E402
from run_sequence import MIN_DEPTH, make_sequence  # noqa: E402
from scenes import DONT_CARE, INVALID, NON_PLANAR, Intrinsics, raycast, scene_set  # noqa: E402

LATER_FRAME = 5
ERROR_RANGE_MM = 30.0
GRAY = (0.85, 0.85, 0.85)
FALSE_POSITIVE = (0.9, 0.0, 0.7)
SPECIAL = {INVALID: (1.0, 1.0, 1.0), DONT_CARE: GRAY, NON_PLANAR: (0.45, 0.45, 0.45)}


def _gt_color(index):
    return plt.get_cmap("tab10").colors[index % 10]


def _paint(shape, planes, colors, valid=None):
    image = np.ones((shape[0] * shape[1], 3))
    if valid is not None:
        image[valid.reshape(-1)] = 0.8
    for plane in planes:
        image[plane.inliers] = colors[id(plane)]
    return image.reshape(shape + (3,))


def _title(name, planes):
    small = sum(p.area < SMALL_AREA for p in planes)
    return f"{name}: {len(planes)} planes, {small} < {SMALL_AREA:g} m2"


def synthetic_figure(path, scene, presets, configs, labels, method, max_depth):
    intrinsics = Intrinsics()
    depth_true, _ = raycast(scene, intrinsics)
    n_rows = len(presets) * len(configs)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 3.7 * n_rows))
    axes = np.atleast_2d(axes)
    for row, preset in enumerate(presets):
        frames = make_sequence(scene, preset, max_depth, LATER_FRAME + 1)
        for c, (config, label) in enumerate(zip(configs, labels)):
            ax = axes[row * len(configs) + c]
            params = config.detector_params(method)
            shape = frames[0].labels.shape
            panels = []
            for t in (0, LATER_FRAME):
                frame = frames[t]
                options = DetectionOptions(
                    camera_to_base=frame.camera_to_base,
                    min_depth=MIN_DEPTH,
                    max_depth=max_depth,
                    noise=config.noise(intrinsics.fx),
                )
                planes = detect_planes(frame.cloud, method, options, params)
                stats = evaluate(frame, planes)
                matched = {k: g for g, k in stats["matches"]}
                colors = {}
                for k, plane in enumerate(planes):
                    if k in matched:
                        colors[id(plane)] = _gt_color(matched[k])
                    elif k in stats["ignored"]:
                        colors[id(plane)] = GRAY
                    else:
                        colors[id(plane)] = FALSE_POSITIVE
                panels.append((planes, select_output_planes(planes, options), colors))
            (all0, sel0, colors0), (all5, _, colors5) = panels
            error = (frames[0].cloud[..., 2] - depth_true) * 1e3
            image = ax[0].imshow(
                error, cmap="RdBu", vmin=-ERROR_RANGE_MM, vmax=ERROR_RANGE_MM
            )
            fig.colorbar(image, ax=ax[0], fraction=0.035)
            ax[0].set_title(f"{preset}\n{label}\ndepth error mm, frame 0", fontsize=10)
            gt = np.ones(shape + (3,))
            for key, color in SPECIAL.items():
                gt[frames[0].labels == key] = color
            for g in range(len(frames[0].planes)):
                gt[frames[0].labels == g] = _gt_color(g)
            ax[1].imshow(gt, interpolation="nearest")
            ax[1].set_title(f"GT: {len(frames[0].planes)} planes")
            for a, (name, planes, colors) in zip(
                ax[2:],
                (
                    ("frame 0 all", all0, colors0),
                    ("frame 0 selected", sel0, colors0),
                    (f"frame {LATER_FRAME} all", all5, colors5),
                ),
            ):
                a.imshow(_paint(shape, planes, colors), interpolation="nearest")
                a.set_title(_title(name, planes), fontsize=9)
            for a in ax:
                a.axis("off")
    fig.suptitle(
        f"{scene.name}: {method}, max {max_depth:g} m. Detections in the color of "
        "the matched GT plane, magenta = false positive, gray = ignored",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=60)
    plt.close(fig)


def real_figure(path, scene, configs, labels, method, min_depth, max_depth):
    fx, fy, cx, cy = scene.intrinsics
    shape = scene.depth.shape[1:]
    fig, axes = plt.subplots(len(configs), 4, figsize=(16, 3.7 * len(configs)))
    axes = np.atleast_2d(axes)
    for ax, config, label in zip(axes, configs, labels):
        params = config.detector_params(method)
        panels = []
        for t in (0, min(LATER_FRAME, scene.n_frames - 1)):
            depth = scene.depth[t]
            options = _options(scene, t, min_depth, max_depth, config)
            cloud = depth_to_cloud(depth, fx, fy, cx, cy)
            planes = detect_planes(cloud, method, options, params)
            z = depth.reshape(-1)
            valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
            colors = {id(p): plane_palette(k) for k, p in enumerate(planes)}
            selected = select_output_planes(planes, options)
            panels.append((planes, selected, colors, valid))
        (all0, sel0, colors0, valid0), (all5, _, colors5, valid5) = panels
        if scene.color_path is not None:
            ax[0].imshow(plt.imread(scene.color_path))
        else:
            ax[0].imshow(scene.depth[0], cmap="viridis", vmin=0.0, vmax=max_depth)
        ax[0].set_title(f"{label}\nframe 0", fontsize=10)
        for a, (name, planes, colors, valid) in zip(
            ax[1:],
            (
                ("frame 0 all", all0, colors0, valid0),
                ("frame 0 selected", sel0, colors0, valid0),
                (f"frame {LATER_FRAME} all", all5, colors5, valid5),
            ),
        ):
            a.imshow(_paint(shape, planes, colors, valid), interpolation="nearest")
            a.set_title(_title(name, planes), fontsize=9)
        for a in ax:
            a.axis("off")
    fig.suptitle(
        f"{scene.name}: {method}, max {max_depth:g} m. One color per plane, light "
        "gray = unassigned in range",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=60)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["sub=0.08"],
        help="one row each, e.g. sub=0.08 sub=0.16,merge=on",
    )
    parser.add_argument("--labels", nargs="*", default=None, help="row labels")
    parser.add_argument("--scenes", nargs="*", default=["table_1.5m", "clutter_room"])
    parser.add_argument("--presets", nargs="+", default=["d435i_filtered", "d435i_raw"])
    parser.add_argument("--real", nargs="*", type=Path, default=[], help="scene dirs")
    parser.add_argument("--method", default="ransac_gravity_refined")
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--prefix", default="tuning")
    args = parser.parse_args()
    try:
        configs = [parse_config(text, [args.method]) for text in args.configs]
    except ValueError as error:
        parser.error(str(error))
    labels = args.labels or [c.label for c in configs]
    if len(labels) != len(configs):
        parser.error("--labels needs one label per config")
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = {s.name: s for s in scene_set()}
    for name in args.scenes:
        if name not in scenes:
            parser.error(f"unknown scene {name}")
        path = args.out / f"{args.prefix}_{name}.png"
        scene = scenes[name]
        synthetic_figure(
            path, scene, args.presets, configs, labels, args.method, args.max_depth
        )
        print(f"wrote {path}", flush=True)
    for directory in args.real:
        scene = load_scene(directory)
        path = args.out / f"{args.prefix}_real_{scene.name}.png"
        real_figure(
            path, scene, configs, labels, args.method, args.min_depth, args.max_depth
        )
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
