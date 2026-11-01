#!/usr/bin/env python3
"""Draw the evidence of a stress trial (stress.py --evidence) as a PNG.

Left: top view in base_footprint at the goal start with the detected planes,
the place candidates and obstacle boxes of the last place markers, the
planning scene boxes, the spawned models before (grey) and after (black when
moved), the place pose and where the held object ended. Right: the color
image at the goal start. Needs only numpy, matplotlib and OpenCV.

    python3 plot_evidence.py <evidence>/<scenario>_<trial> [-o out.png]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

LABEL_COLORS = {
    "floor": "0.6",
    "table": "tab:green",
    "shelf": "tab:orange",
    "wall": "tab:blue",
    "other": "tab:purple",
}
# Outer footprints (x0, x1, y0, y1) of model:// furniture in the model frame.
MODEL_FOOTPRINTS = {
    "model://shelf": (-0.2, 0.22, -0.32, 0.32),
    "model://dining_table": (-0.6, 0.6, -0.9, 0.9),
    "model://wrc_long_table": (-0.6, 0.6, -0.2, 0.2),
    "model://wrc_tall_table": (-0.2, 0.2, -0.2, 0.2),
}
BASE_OUTLINE = (-0.58, 0.07, -0.28, 0.28)  # base_link mesh bounds


def rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def footprint(prop) -> np.ndarray | None:
    kind, size = prop["kind"], prop["size"]
    if kind in ("table", "wall", "box", "cylinder"):
        return rect(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2)
    if kind in MODEL_FOOTPRINTS:
        return rect(*MODEL_FOOTPRINTS[kind])
    return None


def to_base(robot0, T, points):
    M = np.linalg.inv(robot0) @ T
    return points @ M[:2, :2].T + M[:2, 3]


def yaw_rect(x, y, length, width, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = rect(-length / 2, length / 2, -width / 2, width / 2)
    return corners @ np.array([[c, s], [-s, c]]) + [x, y]


def draw(evidence: Path, out: Path):
    data = json.loads((evidence / "result.json").read_text())
    row = data["row"]
    robot0 = np.array(data["robot0"])
    fig, (ax, ax_img) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    ax.add_patch(Polygon(rect(*BASE_OUTLINE), fill=False, ec="0.3", ls="--"))
    for plane in data["planes"]:
        hull = np.array(plane["hull"])
        if len(hull) < 3 or plane["label"] == "floor":
            continue
        color = LABEL_COLORS.get(plane["label"], "k")
        ax.add_patch(Polygon(hull[:, :2], fill=True, fc=color, alpha=0.15, ec=color))
        c = hull[:, :2].mean(axis=0)
        ax.text(*c, f"{plane['label']} {plane['height']:.2f}", color=color, fontsize=7)

    for obj in data.get("scene") or []:
        if not obj["id"].startswith("aha_place_obstacle_"):
            continue
        T = np.array(obj["pose"])
        sx, sy = obj["size"][:2]
        corners = rect(-sx / 2, sx / 2, -sy / 2, sy / 2) @ T[:2, :2].T + T[:2, 3]
        ax.add_patch(Polygon(corners, fill=False, ec="tab:blue", ls=":", lw=1))

    markers = data["markers"]
    k = data.get("markers_at_trigger") or len(markers)
    summary = next((m for m in reversed(markers[:k]) if m["candidates"]), None)
    if summary:
        for ox, oy, _, length, width, _, yaw in summary["obstacles"]:
            ax.add_patch(
                Polygon(yaw_rect(ox, oy, length, width, yaw), fc="red", alpha=0.3)
            )
        for rank, x, y, _, clearance in summary["candidates"]:
            ax.add_patch(Circle((x, y), clearance, fill=False, ec="tab:green", lw=0.6))
            ax.plot(x, y, "o", ms=3, color="green" if rank == 0 else "tab:olive")
            ax.text(x, y, str(rank), fontsize=6)

    held = f"aha_held_object_{row.get('arm', 'right')}"
    moved = {item.split(":")[0] for item in (row.get("prop_moves") or "").split()}
    for name, prop in data["props"].items():
        shape = footprint(prop)
        if shape is None:
            continue
        if name in data["before"]:
            pts = to_base(robot0, np.array(data["before"][name]), shape)
            ax.add_patch(Polygon(pts, fc="0.75", alpha=0.5, ec="0.5", lw=0.5))
        if name in data["after"] and name in moved:
            pts = to_base(robot0, np.array(data["after"][name]), shape)
            ax.add_patch(Polygon(pts, fill=False, ec="k", lw=1.2))
            ax.text(*pts.mean(axis=0), name, fontsize=6)
    if held in data["after"]:
        size = [float(v) for v in row["size"].split("x")]
        shape = rect(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2)
        pts = to_base(robot0, np.array(data["after"][held]), shape)
        ax.add_patch(Polygon(pts, fc="tab:orange", alpha=0.8, ec="k", lw=0.8))
    if data.get("place_pose"):
        x, y, _, _ = data["place_pose"]
        ax.plot(x, y, "x", color="magenta", ms=10, mew=2)

    legend = [
        Polygon(rect(0, 1, 0, 1), fc="0.75", alpha=0.5, ec="0.5", label="model before"),
        Polygon(rect(0, 1, 0, 1), fill=False, ec="k", label="model after (moved)"),
        Polygon(rect(0, 1, 0, 1), fc="red", alpha=0.3, label="perceived obstacle"),
        Polygon(
            rect(0, 1, 0, 1),
            fill=False,
            ec="tab:blue",
            ls=":",
            label="planning scene box",
        ),
        Polygon(rect(0, 1, 0, 1), fc="tab:orange", label="held object after"),
        Circle((0, 0), 1, fill=False, ec="tab:green", label="candidate (clearance)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=6)
    ax.set_aspect("equal")
    ax.autoscale_view()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(min(xlim[0], -0.2), max(xlim[1], 1.0))
    ax.set_ylim(min(ylim[0], -0.8), max(ylim[1], 0.6))
    ax.set_xlabel("x (m, base_footprint)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)
    ax.set_title(
        f"{row['scenario']} #{row['trial']}: {row.get('status', '')} / {row.get('outcome', '')}"
        f" / ended on {row.get('ended_on', '')}",
        fontsize=9,
    )
    note = row.get("note") or ""
    message = (row.get("message") or "")[:110]
    ax.text(
        0.01,
        0.01,
        f"{note}\n{message}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )

    color = evidence / "color.png"
    if color.exists():
        ax_img.imshow(cv2.cvtColor(cv2.imread(str(color)), cv2.COLOR_BGR2RGB))
    ax_img.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence", type=Path, nargs="+")
    parser.add_argument("-o", "--out", type=Path, help="PNG (one evidence dir only)")
    args = parser.parse_args()
    for evidence in args.evidence:
        draw(evidence, args.out or evidence / "plot.png")


if __name__ == "__main__":
    main()
