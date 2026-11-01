"""Scenarios of stress.py: spawned models, trials and perturbations.

Poses are in the scenario frame: base_footprint at the scenario start (x
ahead, y left, z up from the floor). ROS free apart from the shared object
sizes of run_trials.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from run_trials import BOX_L, BOX_S, CYL

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation.sim_held_object import object_sdf

# rcjo2026 default spawn: open floor ahead (+x) of the robot.
SPAWN = (-2.0, 1.5, 0.0)
FILTERED = "d435i_filtered"


@dataclass(frozen=True)
class Prop:
    """A model spawned for a scenario or a trial.

    kind: "table" (top slab and four legs; size = depth along x, width along
    y, height), "wall" (static box; size x, y, z), "box" / "cylinder" (size
    x, y, z standing on z) or a model:// URI (origin on the floor).
    """

    name: str
    kind: str
    size: tuple = ()
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # deg
    static: bool = True
    mass: float = 0.2


def table(name, x, y, depth, width, height, yaw=0.0, static=True, mass=5.0):
    return Prop(name, "table", (depth, width, height), x, y, 0.0, yaw, static, mass)


def wall(name, x, y, size, yaw=0.0):
    return Prop(name, "wall", tuple(size), x, y, 0.0, yaw)


def box(name, x, y, z, size, yaw=0.0, mass=0.2):
    return Prop(name, "box", tuple(size), x, y, z, yaw, False, mass)


def cyl(name, x, y, z, diameter, height, mass=0.2):
    return Prop(
        name, "cylinder", (diameter, diameter, height), x, y, z, 0.0, False, mass
    )


def model(name, uri, x, y, yaw=0.0, static=True):
    return Prop(name, uri, (), x, y, 0.0, yaw, static)


@dataclass(frozen=True)
class Perturb:
    """Change while the goal runs, at the first feedback of ``phase``.

    drive: base forward by ``value`` m; turn: base yaw by ``value`` deg;
    head: look at ``value`` (base_footprint); spawn: ``prop`` with x, y
    relative to the first candidate of the last place markers (base_footprint
    axes), standing on its surface; cancel: cancel the goal; second_goal:
    send the same goal again.
    """

    kind: str
    phase: str = "planning"
    value: object = None
    prop: Prop | None = None


@dataclass(frozen=True)
class Trial:
    shape: str
    size: tuple
    grasp: str = geo.SIDE
    approach: str = "auto"
    arm: str = "right"
    near: tuple | None = None  # scenario frame
    restrict: float | None = None
    labels: tuple = ()
    target: str | None = None  # intended surface; None: the scenario's
    props: tuple = ()  # spawned before the trial, removed after it
    keep: bool = False  # leave the placed object for the next trials
    perturb: Perturb | None = None
    # base_footprint; None: the scenario's. The head while holding only: the
    # place server aims it again (aim_head) before perceiving.
    look_at: tuple | None = None
    mass: float = 0.1
    note: str = ""


@dataclass(frozen=True)
class Scenario:
    name: str
    props: tuple
    look_at: tuple  # base_footprint
    trials: tuple
    target: str = "table"  # surface name (prop, or prop@height for a model)
    noise: str = FILTERED
    world: str = "rcjo2026"
    spawn: tuple = SPAWN


# --- models ---------------------------------------------------------------------------

# Support tops of model:// furniture in the model frame: (z, x0, x1, y0, y1).
MODEL_SURFACES = {
    "model://shelf": [
        (z, -0.2, 0.2, -0.3, 0.3) for z in (0.02, 0.31, 0.61, 0.91, 1.20)
    ],
    "model://dining_table": [(0.80, -0.6, 0.6, -0.9, 0.9)],
    "model://wrc_long_table": [(0.40, -0.6, 0.6, -0.2, 0.2)],
    "model://wrc_tall_table": [(0.60, -0.2, 0.2, -0.2, 0.2)],
}


def prop_surfaces(prop: Prop) -> list[tuple]:
    """(name, z, x0, x1, y0, y1) of the support tops in the prop's model frame."""
    if prop.kind == "table":
        d, w, h = prop.size
        return [(prop.name, h, -d / 2, d / 2, -w / 2, w / 2)]
    if prop.kind in ("box", "cylinder"):
        x, y, z = prop.size
        return [(prop.name, z / 2, -x / 2, x / 2, -y / 2, y / 2)]
    tops = MODEL_SURFACES.get(prop.kind, [])
    if len(tops) == 1:
        return [(prop.name, *tops[0])]
    return [(f"{prop.name}@{t[0]:.2f}", *t) for t in tops]


def _box_parts_sdf(name, parts, static, mass, color="0.75 0.7 0.6 1"):
    material = (
        f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>"
    )
    links = "".join(
        f"<{tag} name='{part}{suffix}'><pose>{x} {y} {z} 0 0 0</pose><geometry><box>"
        f"<size>{a} {b} {c}</size></box></geometry>{extra}</{tag}>"
        for part, (a, b, c), (x, y, z) in parts
        for tag, suffix, extra in (("collision", "", ""), ("visual", "_v", material))
    )
    return (
        f"<sdf version='1.9'><model name='{name}'><static>{str(static).lower()}</static>"
        f"<link name='link'><inertial><mass>{mass}</mass><inertia><ixx>1</ixx><iyy>1</iyy>"
        f"<izz>1</izz></inertia></inertial>{links}</link></model></sdf>"
    )


def prop_sdf(prop: Prop) -> str:
    if prop.kind == "table":
        d, w, h = prop.size
        top, leg = 0.03, 0.04
        parts = [("top", (d, w, top), (0, 0, h - top / 2))]
        for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            center = (sx * (d / 2 - leg), sy * (w / 2 - leg), (h - top) / 2)
            parts.append((f"leg{k}", (leg, leg, h - top), center))
        return _box_parts_sdf(prop.name, parts, prop.static, prop.mass)
    if prop.kind == "wall":
        x, y, z = prop.size
        parts = [("wall", (x, y, z), (0, 0, z / 2))]
        return _box_parts_sdf(prop.name, parts, True, 1.0, "0.9 0.9 0.9 1")
    if prop.kind in ("box", "cylinder"):
        return object_sdf(prop.name, prop.kind, prop.size, prop.mass)
    return (
        f"<sdf version='1.9'><include><uri>{prop.kind}</uri><name>{prop.name}</name>"
        f"<static>{str(prop.static).lower()}</static></include></sdf>"
    )


def prop_pose(prop: Prop, frame: np.ndarray) -> np.ndarray:
    """World pose of the model origin; ``frame`` is the scenario frame in the world."""
    z = prop.z + prop.size[2] / 2 + 0.002 if prop.kind in ("box", "cylinder") else 0.0
    return frame @ geo.transform(geo.rot_z(math.radians(prop.yaw)), [prop.x, prop.y, z])


# --- scenarios ------------------------------------------------------------------------

S, T, A = geo.SIDE, geo.TOP, "auto"
# Table 0.60 x 1.20 x 0.75 m, front edge 0.30 m ahead, from y -0.70 to 0.50.
T75 = table("table", 0.60, -0.10, 0.60, 1.20, 0.75)
LOOK = (0.55, -0.15, 0.75)
GAP_AT = (0.45, -0.25, 0.75)
# Shelf with the front 0.40 m ahead, open side to the robot.
SHELF = model("shelf", "model://shelf", 0.60, -0.05)


def obj(spec, **kw) -> Trial:
    return Trial(*spec, **kw)


def gap_row(gap, shape="box", size=(0.06, 0.06, 0.10), x=0.45, y=-0.25, spacing=0.12):
    """Objects in a row along y on T75 with one gap of ``gap`` (edge to edge) at y."""
    half = gap / 2 + size[1] / 2
    ys = [y - half - k * spacing for k in range(5)]
    ys += [y + half + k * spacing for k in range(5)]
    return tuple(
        Prop(f"row_{k}", shape, size, x, v, 0.75, 0.0, False, 0.2)
        for k, v in enumerate(ys)
        if -0.66 < v < 0.46
    )


def scattered(count=14, seed=3):
    """Small objects of mixed heights on the right half of T75 (reproducible)."""
    rng = np.random.default_rng(seed)
    props, centers = [], []
    while len(props) < count:
        x, y = rng.uniform(0.37, 0.85), rng.uniform(-0.64, 0.10)
        if any(math.hypot(x - a, y - b) < 0.10 for a, b in centers):
            continue
        centers.append((x, y))
        h = float(rng.choice([0.04, 0.06, 0.10, 0.15, 0.20]))
        name = f"small_{len(props)}"
        if rng.uniform() < 0.4:
            props.append(cyl(name, x, y, 0.75, 0.05, h))
        else:
            w = float(rng.uniform(0.04, 0.07))
            props.append(box(name, x, y, 0.75, (w, w, h), float(rng.uniform(0, 90))))
    return tuple(props)


def shadowed(x, y, distance, size=(0.06, 0.06, 0.10), name="hidden"):
    """Box on T75 ``distance`` behind (x, y) on the line from the camera (x 0.11, y 0.03)."""
    dx, dy = x - 0.11, y - 0.03
    norm = math.hypot(dx, dy)
    return box(name, x + dx / norm * distance, y + dy / norm * distance, 0.75, size)


def yawed_table(yaw):
    """T75 turned by ``yaw`` deg about its center, nearest corner 0.24 m ahead."""
    a = math.radians(abs(yaw))
    cx = 0.24 + 0.30 * math.cos(a) + 0.60 * math.sin(a)
    return table("table", cx, -0.10, 0.60, 1.20, 0.75, yaw)


def _filtered_scenarios():
    tall = ("box", (0.10, 0.10, 0.25))
    book = (0.20, 0.15)
    hidden_low = box("hidden", 0.32, -0.50, 0.45, (0.06, 0.06, 0.15))
    long_table = model("table", "model://wrc_long_table", 0.50, -0.15, 90)
    tall_table = model("table", "model://wrc_tall_table", 0.50, -0.20)
    deep = table("table", 0.80, -0.10, 1.00, 1.20, 0.75)
    close = table("table", 0.40, -0.10, 0.60, 1.20, 0.75)
    shallow = table("table", 0.525, -0.10, 0.45, 1.20, 0.75)
    flat_box = box("flat_box", 0.45, -0.25, 0.75, (0.30, 0.30, 0.08))
    dining = model("dining_table", "model://dining_table", 1.20, 0.0, 90, static=False)
    loose_shelf = model("shelf", "model://shelf", 0.60, -0.05, static=False)
    board = (
        box("on_board_0", 0.50, -0.10, 0.61, (0.06, 0.06, 0.12)),
        cyl("on_board_1", 0.55, -0.28, 0.61, 0.06, 0.15),
    )
    return [
        Scenario(
            "baseline",
            (T75,),
            LOOK,
            (
                obj(BOX_S, approach=A),
                obj(BOX_L, approach=S),
                obj(CYL, grasp=T, approach=T),
                obj(BOX_S, arm="left"),
            ),
        ),
        Scenario(
            "gap",
            (T75,),
            LOOK,
            (
                obj(BOX_S, near=GAP_AT, props=gap_row(0.22), note="gap 0.22"),
                obj(BOX_S, near=GAP_AT, props=gap_row(0.18), note="gap 0.18"),
                obj(BOX_S, near=GAP_AT, props=gap_row(0.15), note="gap 0.15"),
                obj(BOX_L, near=GAP_AT, props=gap_row(0.22), note="gap 0.22"),
                obj(
                    BOX_S,
                    grasp=T,
                    approach=T,
                    near=GAP_AT,
                    props=gap_row(0.22),
                    note="gap 0.22",
                ),
                obj(
                    BOX_S,
                    near=GAP_AT,
                    props=gap_row(0.22, "cylinder", (0.07, 0.07, 0.28)),
                    note="gap 0.22 between 0.28 m bottles",
                ),
            ),
        ),
        Scenario(
            "scattered",
            (T75, *scattered()),
            LOOK,
            (
                obj(BOX_S),
                obj(BOX_L),
                obj(CYL, grasp=T, approach=T),
                obj(BOX_S, arm="left"),
            ),
        ),
        Scenario(
            "occluded",
            (
                T75,
                cyl("bottle_0", 0.50, -0.18, 0.75, 0.07, 0.30),
                cyl("bottle_1", 0.52, -0.40, 0.75, 0.07, 0.30),
                shadowed(0.50, -0.18, 0.10, name="hidden_0"),
                shadowed(0.52, -0.40, 0.10, name="hidden_1"),
            ),
            LOOK,
            (
                obj(BOX_S, near=(0.62, -0.30, 0.75), note="between the hidden boxes"),
                obj(BOX_S, near=(0.66, -0.52, 0.75), note="right of hidden_1"),
                obj(BOX_S, grasp=T, approach=T, near=(0.62, -0.30, 0.75)),
                obj(BOX_S, near=(0.70, -0.10, 0.75), note="behind hidden_0"),
            ),
        ),
        Scenario(
            "self_occlusion",
            (table("table", 0.55, -0.20, 0.60, 1.00, 0.45),),
            (0.55, -0.20, 0.45),
            (
                obj(BOX_L, props=(hidden_low,), note="box in front of the carry hand"),
                obj(BOX_L, grasp=T, approach=T, props=(hidden_low,)),
                obj(
                    tall,
                    props=(box("hidden", 0.30, -0.40, 0.45, (0.06, 0.06, 0.15)),),
                    note="large held object",
                ),
            ),
        ),
        Scenario(
            "edge",
            (
                T75,
                box("over_front", 0.31, -0.25, 0.75, (0.08, 0.08, 0.12)),
                box("over_side", 0.55, -0.70, 0.75, (0.08, 0.08, 0.12)),
            ),
            LOOK,
            (
                obj(BOX_S),
                obj(BOX_S, near=(0.33, -0.45, 0.75), note="near the front edge"),
                obj(BOX_S, near=(0.60, -0.68, 0.75), note="near the side edge"),
                obj(BOX_S, grasp=T, approach=T),
            ),
        ),
        Scenario(
            "view_cut",
            (T75,),
            LOOK,
            (
                obj(
                    BOX_S, look_at=(1.4, -0.10, 0.75), note="front edge below the image"
                ),
                obj(BOX_S, grasp=T, approach=T, look_at=(1.4, -0.10, 0.75)),
                obj(
                    BOX_S,
                    look_at=(0.60, 0.90, 0.75),
                    note="right part outside the image",
                ),
                obj(
                    BOX_S,
                    look_at=(0.70, -0.10, 1.25),
                    note="table at the bottom border",
                ),
            ),
        ),
        Scenario(
            "far",
            (),
            (1.2, -0.1, 0.75),
            (
                obj(
                    BOX_S,
                    props=(table("table", 1.20, -0.10, 0.60, 1.20, 0.75),),
                    note="front edge 0.9 m",
                ),
                obj(
                    BOX_S,
                    props=(table("table", 1.80, -0.10, 0.60, 1.20, 0.75),),
                    look_at=(1.8, -0.1, 0.75),
                    note="front edge 1.5 m",
                ),
            ),
        ),
        Scenario(
            "yaw",
            (),
            LOOK,
            tuple(
                obj(
                    spec,
                    props=(yawed_table(yaw),),
                    look_at=(0.68, y, 0.75),
                    note=f"yaw {yaw}",
                    **kw,
                )
                for spec, yaw, y, kw in (
                    (BOX_S, 30, -0.15, {}),
                    (BOX_S, -30, -0.15, {}),
                    (BOX_S, 45, -0.15, {}),
                    (BOX_S, -45, -0.15, {}),
                    (BOX_S, -45, -0.15, {"grasp": T, "approach": T}),
                    (BOX_S, 30, 0.0, {"arm": "left"}),
                )
            ),
        ),
        Scenario(
            "shelf",
            (SHELF,),
            (0.55, -0.10, 0.61),
            (
                obj(
                    BOX_S,
                    restrict=0.31,
                    target="shelf@0.31",
                    look_at=(0.50, -0.10, 0.31),
                ),
                obj(
                    CYL, restrict=0.31, target="shelf@0.31", look_at=(0.50, -0.10, 0.31)
                ),
                obj(
                    BOX_S,
                    restrict=1.20,
                    target="shelf@1.20",
                    look_at=(0.60, -0.10, 1.20),
                ),
                obj(
                    BOX_S,
                    grasp=T,
                    approach=T,
                    restrict=1.20,
                    target="shelf@1.20",
                    look_at=(0.60, -0.10, 1.20),
                ),
                obj(BOX_S, restrict=0.61, props=board, note="objects on the board"),
                obj(
                    ("cylinder", (0.065, 0.065, 0.24)),
                    restrict=0.61,
                    note="0.24 m under a 0.28 m gap",
                ),
                obj(
                    ("box", (0.06, 0.06, 0.26)),
                    restrict=0.61,
                    note="0.26 m under a 0.28 m gap",
                ),
                obj(
                    BOX_L,
                    restrict=0.91,
                    target="shelf@0.91",
                    look_at=(0.55, -0.10, 0.91),
                ),
            ),
            target="shelf@0.61",
        ),
        Scenario(
            "surfaces",
            (),
            (0.55, -0.15, 0.5),
            (
                obj(
                    BOX_S,
                    props=(long_table,),
                    look_at=(0.5, -0.15, 0.40),
                    note="long_table 0.40",
                ),
                obj(
                    BOX_S,
                    restrict=0.40,
                    props=(long_table,),
                    look_at=(0.5, -0.15, 0.40),
                    note="long_table 0.40",
                ),
                obj(
                    BOX_S,
                    props=(table("table", 0.55, -0.15, 0.50, 0.80, 0.35),),
                    look_at=(0.55, -0.15, 0.35),
                    note="coffee table 0.35",
                ),
                obj(
                    BOX_S,
                    props=(tall_table,),
                    look_at=(0.5, -0.2, 0.60),
                    note="tall_table 0.4 x 0.4",
                ),
                obj(
                    BOX_L,
                    props=(tall_table,),
                    look_at=(0.5, -0.2, 0.60),
                    note="tall_table 0.4 x 0.4",
                ),
                obj(
                    BOX_S,
                    props=(table("table", 0.45, -0.25, 0.25, 0.25, 0.70),),
                    look_at=(0.45, -0.25, 0.70),
                    note="0.25 x 0.25 top",
                ),
                obj(
                    BOX_S,
                    props=(table("table", 0.45, -0.25, 0.20, 0.20, 0.50),),
                    look_at=(0.45, -0.25, 0.50),
                    note="0.20 x 0.20 stool",
                ),
                obj(
                    BOX_S,
                    props=(table("table", 0.60, -0.10, 0.60, 1.20, 1.00),),
                    look_at=(0.6, -0.1, 1.0),
                    note="counter 1.00",
                ),
                obj(
                    BOX_S,
                    props=(table("table", 0.60, -0.10, 0.60, 1.20, 1.05),),
                    look_at=(0.6, -0.1, 1.05),
                    note="counter 1.05",
                ),
            ),
        ),
        # Table 0.45 deep against a wall (face 0.75 m ahead, within the side reach).
        Scenario(
            "wall",
            (shallow, wall("wall", 0.80, -0.10, (0.10, 2.0, 1.6))),
            (0.55, -0.15, 0.80),
            (
                obj(BOX_S, near=(0.70, -0.25, 0.75), note="near the wall"),
                obj(BOX_L, near=(0.70, -0.25, 0.75), note="near the wall"),
                obj(
                    BOX_S,
                    near=(0.70, -0.22, 0.75),
                    props=(box("at_wall", 0.715, -0.10, 0.75, (0.06, 0.06, 0.12)),),
                    note="object touching the wall",
                ),
                obj(
                    BOX_S,
                    near=(0.68, -0.25, 0.75),
                    props=(box("at_wall", 0.725, -0.25, 0.75, (0.04, 0.12, 0.04)),),
                    note="low object touching the wall at the near point",
                ),
            ),
        ),
        Scenario(
            "wall_low",
            (shallow, wall("wall", 0.80, -0.10, (0.10, 2.0, 0.85))),
            (0.55, -0.15, 0.80),
            (
                obj(BOX_S, near=(0.70, -0.25, 0.75), note="0.85 m wall"),
                obj(BOX_L, near=(0.70, -0.25, 0.75), note="0.85 m wall"),
            ),
        ),
        Scenario(
            "reach",
            (),
            LOOK,
            (
                obj(
                    BOX_S,
                    near=(0.95, -0.25, 0.75),
                    props=(deep,),
                    look_at=(0.75, -0.15, 0.75),
                    note="near 0.95 m",
                ),
                obj(
                    BOX_S,
                    grasp=T,
                    approach=T,
                    near=(0.70, -0.20, 0.75),
                    props=(deep,),
                    look_at=(0.75, -0.15, 0.75),
                    note="top, near 0.70 m",
                ),
                obj(
                    BOX_S,
                    props=(close,),
                    look_at=(0.45, -0.15, 0.75),
                    note="front edge 0.10",
                ),
                obj(
                    BOX_S,
                    grasp=T,
                    approach=T,
                    props=(close,),
                    look_at=(0.45, -0.15, 0.75),
                    note="front edge 0.10",
                ),
                obj(
                    BOX_S,
                    arm="left",
                    near=(0.50, -0.30, 0.75),
                    props=(
                        T75,
                        box("block_0", 0.45, -0.02, 0.75, (0.06, 0.06, 0.20)),
                        box("block_1", 0.58, -0.08, 0.75, (0.06, 0.06, 0.20)),
                    ),
                    note="left arm to the right past 0.20 m boxes",
                ),
                obj(
                    BOX_S,
                    near=(0.50, 0.30, 0.75),
                    props=(
                        T75,
                        box("block_0", 0.45, 0.08, 0.75, (0.06, 0.06, 0.20)),
                        box("block_1", 0.58, 0.02, 0.75, (0.06, 0.06, 0.20)),
                    ),
                    note="right arm to the left past 0.20 m boxes",
                ),
                obj(
                    BOX_S,
                    near=(0.85, -0.25, 0.75),
                    props=(T75,),
                    note="near 0.85 m, free spots in reach",
                ),
            ),
        ),
        Scenario(
            "objects",
            (T75,),
            LOOK,
            (
                obj(("box", (0.20, 0.10, 0.10)), note="long along the approach"),
                obj(tall, note="tall"),
                obj(("cylinder", (0.09, 0.09, 0.25)), note="bottle"),
                obj(("box", (0.08, 0.14, 0.10)), note="wider than the gripper (0.12)"),
                obj(BOX_S, mass=1.0, note="1 kg"),
                obj(tall, grasp=T, approach=T, note="tall, top grasp"),
            ),
        ),
        Scenario(
            "repeat",
            (T75,),
            LOOK,
            (
                obj(BOX_S, keep=True),
                obj(BOX_L, keep=True),
                obj(CYL, keep=True),
                obj(BOX_S, keep=True),
                obj(BOX_S, arm="left", keep=True),
                obj(BOX_L, arm="left", keep=True),
                obj(BOX_S, grasp=T, approach=T, keep=True),
            ),
        ),
        Scenario(
            "perturb",
            (
                T75,
                box("clutter_0", 0.58, -0.12, 0.75, (0.06, 0.06, 0.12)),
                box("clutter_1", 0.40, -0.45, 0.75, (0.06, 0.06, 0.12)),
            ),
            LOOK,
            (
                obj(BOX_S, perturb=Perturb("drive", value=0.05)),
                obj(BOX_S, perturb=Perturb("drive", value=-0.08)),
                obj(BOX_S, perturb=Perturb("turn", value=5.0)),
                obj(BOX_S, perturb=Perturb("head", value=(0.6, -0.1, 1.5))),
                obj(
                    BOX_S,
                    perturb=Perturb(
                        "spawn", prop=box("intruder", 0.0, 0.0, 0.0, (0.06, 0.06, 0.15))
                    ),
                ),
                obj(
                    BOX_S,
                    perturb=Perturb(
                        "spawn",
                        prop=box("intruder", -0.12, 0.0, 0.0, (0.06, 0.06, 0.15)),
                    ),
                ),
                obj(BOX_S, perturb=Perturb("cancel", phase="approaching")),
                obj(BOX_S, perturb=Perturb("cancel", phase="planning")),
                obj(BOX_S, perturb=Perturb("second_goal")),
                obj(
                    BOX_S,
                    perturb=Perturb("drive", phase="approaching", value=0.05),
                    note="base moved during the descent",
                ),
                obj(
                    BOX_S,
                    perturb=Perturb("drive", value=-0.60),
                    note="base keeps moving (12 s)",
                ),
            ),
        ),
        Scenario(
            "flat",
            (T75,),
            LOOK,
            (
                obj(
                    BOX_S,
                    props=(box("book", 0.45, -0.25, 0.75, (*book, 0.010)),),
                    note="1.0 cm book",
                ),
                obj(
                    BOX_S,
                    props=(box("book", 0.45, -0.25, 0.75, (*book, 0.018)),),
                    note="1.8 cm book",
                ),
                obj(BOX_S, props=(flat_box,), note="0.30 x 0.30 x 0.08 box"),
                obj(
                    BOX_S,
                    restrict=0.75,
                    props=(flat_box,),
                    note="0.30 x 0.30 x 0.08 box",
                ),
            ),
        ),
        Scenario(
            "dynamic",
            (),
            (0.55, -0.15, 0.80),
            (
                obj(BOX_S, props=(dining,)),
                obj(CYL, grasp=T, approach=T, props=(dining,)),
                obj(
                    BOX_L,
                    near=(0.36, -0.30, 0.80),
                    props=(dining,),
                    note="near the front edge",
                ),
                obj(
                    BOX_S,
                    restrict=0.61,
                    target="shelf@0.61",
                    props=(loose_shelf,),
                    look_at=(0.55, -0.1, 0.61),
                ),
                obj(
                    BOX_S,
                    restrict=0.31,
                    target="shelf@0.31",
                    props=(loose_shelf,),
                    look_at=(0.5, -0.1, 0.31),
                ),
            ),
            target="dining_table",
        ),
        # Objects right under the hand in carry (TCP base (0.22, -0.30, 0.92)).
        Scenario(
            "under_hand",
            (close,),
            (0.45, -0.15, 0.75),
            (
                obj(
                    BOX_S,
                    props=(box("under", 0.20, -0.30, 0.75, (0.06, 0.06, 0.08)),),
                    note="0.08 m box under the held object",
                ),
                obj(
                    BOX_S,
                    props=(box("under", 0.24, -0.40, 0.75, (0.06, 0.06, 0.08)),),
                    note="0.08 m box beside the held object",
                ),
                obj(
                    BOX_S,
                    grasp=T,
                    approach=T,
                    props=(box("under", 0.30, -0.25, 0.75, (0.06, 0.06, 0.10)),),
                    note="0.10 m box under the top-grasped object",
                ),
            ),
        ),
        # T75 as a free 10 kg body: an arm touching the table moves it.
        Scenario(
            "loose_table",
            (table("table", 0.60, -0.10, 0.60, 1.20, 0.75, static=False, mass=10.0),),
            LOOK,
            (
                obj(BOX_S),
                obj(BOX_L, approach=S),
                obj(CYL, grasp=T, approach=T),
                obj(BOX_S, arm="left"),
                obj(BOX_S, near=(0.36, -0.30, 0.75), note="near the front edge"),
            ),
        ),
    ]


def _noise_scenarios(noise, suffix):
    """A subset of the trials with another depth noise."""
    board = (
        box("on_board_0", 0.50, -0.10, 0.61, (0.06, 0.06, 0.12)),
        cyl("on_board_1", 0.55, -0.28, 0.61, 0.06, 0.15),
    )
    tall_table = model("table", "model://wrc_tall_table", 0.50, -0.20)
    return [
        Scenario(
            f"baseline{suffix}",
            (T75,),
            LOOK,
            (obj(BOX_S), obj(BOX_L, grasp=T, approach=T)),
            noise=noise,
        ),
        Scenario(
            f"gap{suffix}",
            (T75,),
            LOOK,
            (obj(BOX_S, near=GAP_AT, props=gap_row(0.22), note="gap 0.22"),),
            noise=noise,
        ),
        Scenario(
            f"scattered{suffix}",
            (T75, *scattered()),
            LOOK,
            (obj(BOX_S), obj(BOX_L)),
            noise=noise,
        ),
        Scenario(
            f"shelf{suffix}",
            (SHELF,),
            (0.55, -0.10, 0.61),
            (obj(BOX_S, restrict=0.61, props=board, note="objects on the board"),),
            target="shelf@0.61",
            noise=noise,
        ),
        Scenario(
            f"yaw{suffix}",
            (),
            LOOK,
            (
                obj(
                    BOX_S,
                    props=(yawed_table(30),),
                    look_at=(0.68, -0.15, 0.75),
                    note="yaw 30",
                ),
            ),
            noise=noise,
        ),
        Scenario(
            f"surfaces{suffix}",
            (),
            (0.5, -0.2, 0.60),
            (obj(BOX_S, props=(tall_table,), note="tall_table 0.4 x 0.4"),),
            noise=noise,
        ),
    ]


SCENARIOS = {
    s.name: s
    for s in (
        *_filtered_scenarios(),
        *_noise_scenarios("d435i_raw", "_raw"),
        *_noise_scenarios("none", "_none"),
    )
}
