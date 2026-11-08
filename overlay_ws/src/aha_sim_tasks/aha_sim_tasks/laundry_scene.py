"""Doing Laundry scene: garment sampling and the laundry-room world.

The room follows the RoboCup@Home 2026 laundry area (rulebook 2027, 5.4): the
robot waits outside the arena door, the washer holds 2-4 T-shirts behind a
closed door, the basket next to it 6-8, and one bunched shirt lies on the
folding table. Furniture models and textures come from laundry_assets.py.
"""

from copy import deepcopy
import math
from pathlib import Path
import random

from .laundry_assets import BASKET, DESIGNS, TABLE, WASHER, floor_mesh

WORLD_NAME = "laundry_room"
SIZE_LABELS = {"S": "small", "M": "medium", "L": "large", "XL": "extra large"}


def package_share():
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("aha_sim_tasks"))


def assets_dir():
    return package_share() / "laundry"


def is_laundry(scene):
    return scene.get("scene_type") == "laundry"


def _weighted(rng, weights):
    names = sorted(weights)
    return rng.choices(names, weights=[weights[name] for name in names])[0]


def validate_laundry(scene):
    garments = scene["garments"]
    for key in ("washer", "basket"):
        low, high = garments[key]
        if not 0 <= low <= high:
            raise ValueError(f"garments.{key} must be an increasing [min, max] range")
    if garments["table"] not in (0, 1):
        raise ValueError("garments.table must be 0 or 1")
    if scene["washer"]["finish"] not in ("graphite", "white"):
        raise ValueError("washer.finish must be graphite or white")
    if scene["basket"]["color"] not in ("white", "blue"):
        raise ValueError("basket.color must be white or blue")
    if scene["arena"]["floor"] not in ("oak", "walnut"):
        raise ValueError("arena.floor must be oak or walnut")
    for name in garments["designs"]:
        if not (assets_dir() / "textures" / f"shirt_{name}.jpg").is_file():
            raise ValueError(f"Unknown T-shirt design: {name}")


def sample_laundry_scene(base, *, seed=0, randomize=False):
    """Choose the laundry and, with randomize, vary layout and lighting.

    The garments always follow the seed. Without randomize the counts are the
    middle of each range and the furniture and lighting keep their base values.
    """
    scene = deepcopy(base)
    rng = random.Random(seed)
    scene.update(seed=seed, randomized=randomize)
    spec = scene["garments"]
    counts = {}
    for origin in ("washer", "basket"):
        low, high = spec[origin]
        counts[origin] = rng.randint(low, high) if randomize else (low + high) // 2
    counts["table"] = spec["table"]
    garments = []
    for origin in ("washer", "basket", "table"):
        for _ in range(counts[origin]):
            design = _weighted(rng, spec["designs"])
            size = _weighted(rng, spec["sizes"])
            garments.append(
                {
                    "id": f"shirt_{len(garments):02d}",
                    "origin": origin,
                    "design": design,
                    "size": size,
                    "label": f"{design.replace('_', ' ')} T-shirt ({SIZE_LABELS[size]})",
                }
            )
    scene["laundry_items"] = garments
    if randomize:
        variation = scene["randomization"]
        spawn = scene["robot_spawn"]
        for key in ("x", "y"):
            spawn[key] += rng.uniform(-variation["robot_xy"], variation["robot_xy"])
        spawn["yaw"] += rng.uniform(-variation["robot_yaw"], variation["robot_yaw"])
        basket = scene["basket"]
        basket["position"] = [
            value + rng.uniform(-variation["basket_xy"], variation["basket_xy"])
            for value in basket["position"]
        ]
        basket["yaw"] += rng.uniform(-variation["basket_yaw"], variation["basket_yaw"])
        basket["color"] = rng.choice(variation["basket_color"])
        scene["washer"]["finish"] = rng.choice(variation["washer_finish"])
        scene["arena"]["floor"] = rng.choice(variation["floor"])
        lighting = scene["lighting"]
        lighting["intensity"] = rng.uniform(*variation["light_intensity"])
        lighting["ambient"] = rng.uniform(*variation["ambient"])
        lighting["direction"] = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), -1.0]
    return resolve_laundry_scene(scene)


def resolve_laundry_scene(scene):
    """Add asset paths and the configuration read by the Gazebo system."""
    scene = deepcopy(scene)
    validate_laundry(scene)
    if "laundry_items" not in scene:
        return sample_laundry_scene(scene, seed=scene.get("seed", 0))
    assets = assets_dir()
    garments = []
    for item in scene["laundry_items"]:
        garments.append(
            {
                **item,
                "color": [c / 255 for c in DESIGNS[item["design"]]["color"]],
                "pattern": str(assets / "patterns" / f"tshirt_{item['size']}.json"),
                "albedo": str(assets / "textures" / f"shirt_{item['design']}.jpg"),
                "normal": str(assets / "textures" / "shirt_knit_normal.png"),
            }
        )
    w = WASHER
    lip_x = w["depth"] / 2 - 0.04
    b = BASKET
    scene["laundry"] = {
        "garments": garments,
        "cloth": scene["cloth"],
        "grippers": scene["grippers"],
        "placement": scene.get("placement", {}),
        "regions": {
            "washer": {
                "model": "washing_machine",
                "link": "body",
                "door_model": "washing_machine_door",
                "door_link": "door",
                "door_joint": "hinge",
                "drum_center": [lip_x - w["drum_depth"] / 2, 0.0, w["door_center_z"]],
                "drum_axis": [1.0, 0.0, 0.0],
                "drum_radius": w["drum_radius"],
                "drum_depth": w["drum_depth"],
            },
            "basket": {
                "model": "laundry_basket",
                "link": "link",
                # Heaped laundry up to 12 cm above the rim still counts as inside.
                "interior_center": [0.0, 0.0, (b["floor"] + b["height"] + 0.12) / 2],
                "interior_half": [
                    b["top"][0] / 2 - 0.006,
                    b["top"][1] / 2 - 0.006,
                    (b["height"] + 0.12 - b["floor"]) / 2,
                ],
            },
            "table": {
                "model": "folding_table",
                "link": "link",
                "top_center": [0.0, 0.0, TABLE["height"]],
                "top_half": [TABLE["size"][0] / 2, TABLE["size"][1] / 2],
            },
        },
        "arena_door": {
            "model": "arena_door",
            "joint": "hinge",
            "open_angle": 1.5,
            "speed": 0.8,
        },
        "ignore_models": ["venue"],
    }
    scene["resolved"] = True
    return scene


def load_laundry_scene(scene):
    return scene if scene.get("resolved") else resolve_laundry_scene(scene)


def table_frame(scene):
    """Table top centre (x, y, z) and yaw in the world."""
    x, y = scene["table"]["position"]
    return (x, y, TABLE["height"]), scene["table"]["yaw"]


# ------------------------------------------------------------------- world
def _pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    return f"<pose>{x:.5f} {y:.5f} {z:.5f} {roll:.5f} {pitch:.5f} {yaw:.5f}</pose>"


def _material(
    albedo=None,
    normal=None,
    color=(1, 1, 1),
    roughness=0.6,
    metalness=0.0,
    emissive=None,
):
    r, g, b = color
    maps = ""
    if albedo:
        maps += f"<albedo_map>{albedo}</albedo_map>"
    if normal:
        maps += f"<normal_map>{normal}</normal_map>"
    glow = (
        f"<emissive>{emissive[0]} {emissive[1]} {emissive[2]} 1</emissive>"
        if emissive
        else ""
    )
    return (
        f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>"
        f"<specular>0.2 0.2 0.2 1</specular>{glow}<pbr><metal>{maps}"
        f"<roughness>{roughness}</roughness><metalness>{metalness}</metalness>"
        "</metal></pbr></material>"
    )


def _mesh_visual(name, path, material, pose="", extra=""):
    return (
        f'<visual name="{name}">{pose}<geometry><mesh><uri>{path}</uri></mesh></geometry>'
        f"{material}{extra}</visual>"
    )


def _box_visual(name, size, pose, material):
    return (
        f'<visual name="{name}">{pose}<geometry><box><size>{size[0]:.5f} {size[1]:.5f} '
        f"{size[2]:.5f}</size></box></geometry>{material}</visual>"
    )


def _box_collision(name, size, pose, friction=0.6):
    return (
        f'<collision name="{name}">{pose}<geometry><box><size>{size[0]:.5f} {size[1]:.5f} '
        f"{size[2]:.5f}</size></box></geometry><surface><friction><ode><mu>{friction}</mu>"
        f"<mu2>{friction}</mu2></ode></friction></surface></collision>"
    )


def _cylinder_collision(name, radius, length, pose):
    return (
        f'<collision name="{name}">{pose}<geometry><cylinder><radius>{radius}</radius>'
        f"<length>{length}</length></cylinder></geometry></collision>"
    )


def _ring(
    prefix, center_x, center_z, radius, thickness, length, count=16, axis_offset=0.0
):
    """Boxes approximating a tube about the x axis (inner radius `radius`)."""
    parts = []
    mid = radius + thickness / 2
    width = 2 * (radius + thickness) * math.tan(math.pi / count) * 1.08
    for k in range(count):
        a = 2 * math.pi * k / count
        parts.append(
            _box_collision(
                f"{prefix}_{k}",
                (length, width, thickness),
                _pose(
                    center_x + axis_offset,
                    mid * math.cos(a),
                    center_z + mid * math.sin(a),
                    roll=a + math.pi / 2,
                ),
                0.35,
            )
        )
    return "".join(parts)


def _washer(scene, assets):
    w = WASHER
    finish = scene["washer"]["finish"]
    tex = assets / "textures"
    meshes = assets / "meshes"
    x, y = scene["washer"]["position"]
    d2, w2, top = w["depth"] / 2, w["width"] / 2, w["height"]
    zc, ro = w["door_center_z"], w["opening_radius"]
    lip_x = d2 - 0.04
    back_x = lip_x - w["drum_depth"]
    paint = (
        dict(roughness=0.32, metalness=0.3)
        if finish == "graphite"
        else dict(roughness=0.3, metalness=0.05)
    )
    body = _material(tex / f"washer_body_{finish}.jpg", **paint)
    front = _material(tex / f"washer_front_{finish}.jpg", **paint)
    visuals = "".join(
        [
            _mesh_visual("cabinet", meshes / "washer_body.obj", body),
            _mesh_visual("front", meshes / "washer_front.obj", front),
            _mesh_visual(
                "panel",
                meshes / "washer_panel.obj",
                _material(tex / "washer_panel.jpg", roughness=0.15, metalness=0.2),
            ),
            _mesh_visual(
                "gasket",
                meshes / "washer_gasket.obj",
                _material(tex / "rubber_gray.jpg", roughness=0.85),
            ),
            _mesh_visual(
                "drum",
                meshes / "washer_drum.obj",
                _material(
                    tex / "washer_drum.jpg",
                    tex / "washer_drum_normal.png",
                    roughness=0.35,
                    metalness=0.3,
                ),
            ),
            _mesh_visual(
                "feet",
                meshes / "washer_feet.obj",
                _material(color=(0.1, 0.1, 0.1), roughness=0.8),
            ),
        ]
    )
    s = w["foot"]
    hole = ro
    collisions = "".join(
        [
            _box_collision(
                "top", (w["depth"], w["width"], 0.02), _pose(0, 0, top - 0.01)
            ),
            _box_collision(
                "base", (w["depth"], w["width"], 0.06), _pose(0, 0, s + 0.03)
            ),
            _box_collision(
                "left", (w["depth"], 0.02, top - s), _pose(0, -w2 + 0.01, (top + s) / 2)
            ),
            _box_collision(
                "right", (w["depth"], 0.02, top - s), _pose(0, w2 - 0.01, (top + s) / 2)
            ),
            _box_collision(
                "back", (0.02, w["width"], top - s), _pose(-d2 + 0.01, 0, (top + s) / 2)
            ),
            _box_collision(
                "panel",
                (0.02, w["width"], top - w["panel_bottom"]),
                _pose(d2 - 0.01, 0, (top + w["panel_bottom"]) / 2),
            ),
            _box_collision(
                "front_top",
                (0.02, w["width"], w["panel_bottom"] - zc - hole),
                _pose(d2 - 0.01, 0, (w["panel_bottom"] + zc + hole) / 2),
            ),
            _box_collision(
                "front_bottom",
                (0.02, w["width"], zc - hole - s - 0.06),
                _pose(d2 - 0.01, 0, (zc - hole + s + 0.06) / 2),
            ),
            _box_collision(
                "front_left",
                (0.02, w2 - hole, 2 * hole),
                _pose(d2 - 0.01, -(w2 + hole) / 2, zc),
            ),
            _box_collision(
                "front_right",
                (0.02, w2 - hole, 2 * hole),
                _pose(d2 - 0.01, (w2 + hole) / 2, zc),
            ),
        ]
    )
    for k, (sy, sz) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        d = (hole + 0.04) / math.sqrt(2)
        collisions += _box_collision(
            f"front_corner_{k}",
            (0.02, 0.22, 0.08),
            _pose(d2 - 0.01, sy * d, zc + sz * d, roll=-sy * sz * math.pi / 4),
        )
    collisions += _ring(
        "gasket", (lip_x + d2) / 2, zc, w["lip_radius"] + 0.012, 0.03, 0.04
    )
    collisions += _ring(
        "lip",
        lip_x + 0.005,
        zc,
        w["lip_radius"],
        w["drum_radius"] - w["lip_radius"],
        0.01,
    )
    collisions += _ring(
        "drum", (lip_x + back_x) / 2, zc, w["drum_radius"], 0.02, w["drum_depth"]
    )
    collisions += _box_collision(
        "drum_back", (0.02, 0.52, 0.52), _pose(back_x - 0.01, 0, zc)
    )
    for k in range(3):
        a = math.radians(90 + 120 * k)
        r = w["drum_radius"] - 0.016
        collisions += _box_collision(
            f"lifter_{k}",
            (w["drum_depth"] - 0.02, 0.05, 0.035),
            _pose((lip_x + back_x) / 2, r * math.cos(a), zc + r * math.sin(a), roll=a),
            0.35,
        )
    rd = w["door_radius"]
    door_visuals = "".join(
        [
            _mesh_visual(
                "frame",
                meshes / "washer_door_frame.obj",
                _material(tex / "door_plastic.jpg", roughness=0.4, metalness=0.1),
            ),
            _mesh_visual(
                "trim",
                meshes / "washer_door_trim.obj",
                _material(color=(0.86, 0.86, 0.88), roughness=0.2, metalness=0.45),
            ),
            _mesh_visual(
                "glass",
                meshes / "washer_door_glass.obj",
                _material(color=(0.1, 0.11, 0.13), roughness=0.05, metalness=0.3),
                extra="<transparency>0.55</transparency>",
            ),
            _mesh_visual(
                "handle",
                meshes / "washer_door_handle.obj",
                _material(tex / "door_plastic.jpg", roughness=0.35),
            ),
        ]
    )
    door_collisions = "".join(
        [
            _cylinder_collision(
                "door", rd, 0.05, _pose(0.025, rd + 0.01, 0, pitch=math.pi / 2)
            ),
            # Shorter than the glass so that it clears the drum lip when swinging.
            _cylinder_collision(
                "bowl", 0.165, 0.065, _pose(0.0025, rd + 0.01, 0, pitch=math.pi / 2)
            ),
            _box_collision(
                "handle", (0.03, 0.035, 0.12), _pose(0.05, 2 * rd - 0.02, 0)
            ),
        ]
    )
    yaw = scene["washer"]["yaw"]
    # The cabinet is static (cheap for the physics engine); the door is a
    # separate model hinged to the world.
    return f"""
    <model name="washing_machine">
      <static>true</static>
      {_pose(x, y, 0, yaw=yaw)}
      <link name="body">{visuals}{collisions}</link>
    </model>
    <model name="washing_machine_door">
      {_pose(x, y, 0, yaw=yaw)}
      <link name="door">
        {_pose(d2 + 0.01, -(rd + 0.01), zc)}
        <inertial>{_pose(0.03, rd + 0.01, 0)}<mass>2.5</mass><inertia><ixx>0.06</ixx>
          <iyy>0.04</iyy><izz>0.04</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        {door_visuals}{door_collisions}
      </link>
      <joint name="hinge" type="revolute">
        <parent>world</parent><child>door</child>
        <axis><xyz>0 0 -1</xyz><limit><lower>0</lower><upper>1.9</upper><effort>50</effort>
          <velocity>3</velocity></limit><dynamics><damping>0.6</damping>
          <friction>0.4</friction></dynamics></axis>
      </joint>
    </model>"""


def _basket(scene, assets):
    b = BASKET
    color = scene["basket"]["color"]
    x, y = scene["basket"]["position"]
    (tx, ty), (bx, by), height = b["top"], b["bottom"], b["height"]
    tilt_x = math.atan2((tx - bx) / 2, height)
    tilt_y = math.atan2((ty - by) / 2, height)
    material = _material(assets / "textures" / f"plastic_{color}.jpg", roughness=0.45)
    t = b["wall"]
    my = (ty + by) / 4
    slant_y = height / math.cos(tilt_y)
    hw, hh = b["handle"]
    collisions = _box_collision(
        "bottom", (bx, by, b["floor"]), _pose(0, 0, b["floor"] / 2), 0.5
    )
    for sign in (1, -1):
        # Long walls (along x).
        collisions += _box_collision(
            f"side_{sign}",
            ((tx + bx) / 2, t, slant_y),
            _pose(0, sign * my, height / 2, roll=-sign * tilt_y),
            0.5,
        )
        # Short walls carry the hand holes below the rim band.
        lower = height - b["rim"] - hh
        collisions += _box_collision(
            f"end_{sign}_low",
            (t, (ty + by) / 2, lower),
            _pose(
                sign * (bx / 2 + (tx - bx) / 2 * lower / 2 / height),
                0,
                lower / 2,
                pitch=sign * tilt_x,
            ),
            0.5,
        )
        rim_x = sign * (tx / 2 - (tx - bx) / 2 * b["rim"] / 2 / height)
        collisions += _box_collision(
            f"end_{sign}_rim",
            (t * 2, ty * 0.98, b["rim"]),
            _pose(rim_x, 0, height - b["rim"] / 2),
            0.6,
        )
        for side in (1, -1):
            part = (ty - hw) / 2
            collisions += _box_collision(
                f"end_{sign}_{side}",
                (t, part, hh),
                _pose(
                    sign * (tx / 2 - (tx - bx) / 2 * (b["rim"] + hh / 2) / height),
                    side * (hw / 2 + part / 2),
                    height - b["rim"] - hh / 2,
                ),
                0.5,
            )
    return f"""
    <model name="laundry_basket">
      {_pose(x, y, 0.001, yaw=scene['basket']['yaw'])}
      <link name="link">
        <inertial>{_pose(0, 0, 0.09)}<mass>0.7</mass><inertia><ixx>0.012</ixx><iyy>0.02</iyy>
          <izz>0.025</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        {_mesh_visual("basket", assets / "meshes" / "basket.obj", material)}
        {collisions}
      </link>
    </model>"""


def _table(scene, assets):
    t = TABLE
    x, y = scene["table"]["position"]
    (lx, ly), height, top, leg = t["size"], t["height"], t["top"], t["leg"]
    collisions = _box_collision(
        "top", (lx, ly, top), _pose(0, 0, height - top / 2), 0.6
    )
    inset = 0.03
    for sx in (-1, 1):
        for sy in (-1, 1):
            collisions += _box_collision(
                f"leg_{sx}_{sy}",
                (leg, leg, height - top),
                _pose(
                    sx * (lx / 2 - inset - leg / 2),
                    sy * (ly / 2 - inset - leg / 2),
                    (height - top) / 2,
                ),
            )
    meshes, tex = assets / "meshes", assets / "textures"
    return f"""
    <model name="folding_table">
      <static>true</static>
      {_pose(x, y, 0, yaw=scene['table']['yaw'])}
      <link name="link">
        {_mesh_visual("top", meshes / "table_top.obj", _material(tex / "melamine_white.jpg", roughness=0.35))}
        {_mesh_visual("frame", meshes / "table_frame.obj", _material(tex / "powder_white.jpg", roughness=0.5, metalness=0.2))}
        {collisions}
      </link>
    </model>"""


def _arena(scene, assets, directory):
    a = scene["arena"]
    (x0, y0), (x1, y1) = a["interior"]
    h, t = a["wall_height"], a["wall_thickness"]
    door = a["door_width"]
    tex = assets / "textures"
    paint = _material(tex / "wall_paint.jpg", roughness=0.85)
    parts = []

    def wall(name, cx, cy, sx, sy, height=h):
        size = (sx, sy, height)
        pose = _pose(cx, cy, height / 2)
        parts.append(
            _box_visual(name, size, pose, paint) + _box_collision(name, size, pose, 0.4)
        )

    wall("north", (x0 + x1) / 2, y1 + t / 2, x1 - x0 + 2 * t, t)
    wall("south", (x0 + x1) / 2, y0 - t / 2, x1 - x0 + 2 * t, t)
    wall("east", x1 + t / 2, (y0 + y1) / 2, t, y1 - y0)
    side = (y1 - y0 - door) / 2
    wall("west_a", x0 - t / 2, y0 + side / 2, t, side)
    wall("west_b", x0 - t / 2, y1 - side / 2, t, side)
    # Entrance frame with a header sign, as at the Incheon arena entrance.
    post = 0.06
    for name, cy in (("post_a", -door / 2 - post / 2), ("post_b", door / 2 + post / 2)):
        wall(name, x0 - t / 2, cy, t + 0.02, post, a["door_height"] + 0.1)
    parts.append(
        _box_visual(
            "header",
            (t + 0.02, door + 2 * post, 0.1),
            _pose(x0 - t / 2, 0, a["door_height"] + 0.05),
            paint,
        )
    )
    floor_tex = tex / f"floor_{a['floor']}.jpg"
    floor_normal = tex / f"floor_{a['floor']}_normal.png"
    floor_mesh(
        directory, ((x0 - t, y0 - t), (x1 + t, y1 + t)), "arena_floor", 2.4, z=0.002
    )
    parts.append(
        _mesh_visual(
            "floor",
            directory / "arena_floor.obj",
            _material(floor_tex, floor_normal, roughness=0.45),
        )
    )
    return f"""
    <model name="arena">
      <static>true</static>
      <link name="link">{''.join(parts)}</link>
    </model>
    <model name="arena_door">
      {_pose(x0 - t / 2, door / 2, 0)}
      <link name="frame">
        <inertial><mass>5</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz><ixy>0</ixy>
          <ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      </link>
      <link name="panel">
        <!-- The pivot is on the room-side face, so the panel swings clear of the frame post. -->
        {_pose(t / 2, -0.015, 0)}
        <inertial>{_pose(-0.0175, -door / 2, a['door_height'] / 2)}<mass>8</mass><inertia>
          <ixx>2.7</ixx><iyy>3.2</iyy><izz>0.6</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia></inertial>
        {_box_visual("panel", (0.035, door - 0.03, a['door_height'] - 0.02),
                     _pose(-0.0175, -(door - 0.03) / 2, a['door_height'] / 2), paint)}
        {_box_visual("knob", (0.08, 0.03, 0.03), _pose(-0.0175, -door + 0.1, 1.0),
                     _material(color=(0.75, 0.75, 0.78), roughness=0.3, metalness=0.4))}
        {_box_collision("panel", (0.035, door - 0.03, a['door_height'] - 0.02),
                        _pose(-0.0175, -(door - 0.03) / 2, a['door_height'] / 2))}
      </link>
      <joint name="anchor" type="fixed"><parent>world</parent><child>frame</child></joint>
      <joint name="hinge" type="revolute">
        <parent>frame</parent><child>panel</child>
        <axis><xyz>0 0 1</xyz><limit><lower>0</lower><upper>1.6</upper><effort>1000</effort>
          <velocity>2</velocity></limit>
          <dynamics><damping>2.0</damping></dynamics></axis>
      </joint>
    </model>"""


def _venue(assets):
    """The competition hall around the arena, seen above the low walls."""
    tex = assets / "textures"
    wall = _material(tex / "venue_wall.jpg", roughness=0.9)
    parts = [
        _mesh_visual(
            "floor",
            assets / "meshes" / "venue_floor.obj",
            _material(tex / "venue_floor.jpg", roughness=0.8),
        )
    ]
    for name, pose, size in (
        ("hall_n", _pose(2, 14, 4), (40, 0.3, 8)),
        ("hall_s", _pose(2, -14, 4), (40, 0.3, 8)),
        ("hall_e", _pose(20, 0, 4), (0.3, 28, 8)),
        ("hall_w", _pose(-16, 0, 4), (0.3, 28, 8)),
    ):
        parts.append(_box_visual(name, size, pose, wall))
    parts.append(
        _box_visual(
            "ceiling",
            (40, 28, 0.2),
            _pose(2, 0, 8.1),
            _material(color=(0.22, 0.23, 0.25), roughness=0.9),
        )
    )
    for i in range(-3, 4):
        for j in range(-2, 3):
            parts.append(
                _box_visual(
                    f"lamp_{i}_{j}",
                    (1.2, 0.3, 0.05),
                    _pose(2 + 5 * i, 5 * j, 7.9),
                    _material(color=(1, 1, 1), emissive=(1, 1, 0.95)),
                )
            )
    # Neighbouring arenas: white partitions beyond ours.
    paint = _material(tex / "wall_paint.jpg", roughness=0.85)
    for name, pose, size in (
        ("other_a", _pose(2, 4.5, 0.6), (6, 0.05, 1.2)),
        ("other_b", _pose(7.5, 0, 0.6), (0.05, 7, 1.2)),
        ("other_c", _pose(2, -4.5, 0.6), (6, 0.05, 1.2)),
        ("other_d", _pose(-4, 2.5, 1.0), (0.05, 5, 2.0)),
    ):
        parts.append(_box_visual(name, size, pose, paint))
    return f"""
    <model name="venue">
      <static>true</static>
      <link name="link">{''.join(parts)}</link>
    </model>"""


def build_world(scene, scene_path, cameras=True):
    """SDF of the laundry room for a resolved scene written at scene_path."""
    assets = assets_dir()
    light = scene["lighting"]
    i = light["intensity"]
    ambient = light["ambient"]
    dx, dy, dz = light["direction"]
    (x0, y0), (x1, y1) = scene["arena"]["interior"]
    lamps = "".join(
        f"""<light type="point" name="lamp_{k}">{_pose(x, y, 3.2)}
          <diffuse>{0.22 * i} {0.22 * i} {0.21 * i} 1</diffuse><specular>0.05 0.05 0.05 1</specular>
          <attenuation><range>9</range><constant>0.4</constant><linear>0.05</linear>
          <quadratic>0.01</quadratic></attenuation><cast_shadows>false</cast_shadows></light>"""
        for k, (x, y) in enumerate(
            ((x0 + 1, y0 + 1), (x0 + 1, y1 - 1), (x1 - 1, y0 + 1), (x1 - 1, y1 - 1))
        )
    )
    sensors = (
        '<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">'
        "<render_engine>ogre2</render_engine></plugin>"
        if cameras
        else ""
    )
    return f"""<?xml version="1.0"?>
<sdf version="1.10">
  <world name="{WORLD_NAME}">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    {sensors}
    <plugin filename="aha_laundry" name="aha_laundry::LaundrySystem">
      <scene_config>{scene_path}</scene_config>
    </plugin>
    <scene><ambient>{ambient} {ambient} {ambient} 1</ambient>
      <background>0.62 0.64 0.67 1</background><shadows>true</shadows></scene>
    <light type="directional" name="sun">{_pose(2, 0, 9)}
      <diffuse>{i} {i} {0.97 * i} 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <direction>{dx} {dy} {dz}</direction><cast_shadows>true</cast_shadows></light>
    {lamps}
    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane>
      </geometry><surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>
      </collision></link></model>
    {_venue(assets)}
    {_arena(scene, assets, Path(scene_path).parent)}
    {_washer(scene, assets)}
    {_table(scene, assets)}
    {_basket(scene, assets)}
  </world>
</sdf>
"""
