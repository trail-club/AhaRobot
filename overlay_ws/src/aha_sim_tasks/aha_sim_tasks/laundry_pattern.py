"""Flat sewing pattern of a short-sleeve T-shirt, shared by meshing and textures.

Pattern coordinates are metres in the plane of a shirt laid flat, front up:
x to the wearer's left-right, y from the hem (0) to the shoulder line.
"""

import math

# Adult size M laid flat (chest and length follow common retail size charts).
BASE = {
    "body_half_width": 0.26,
    "length": 0.72,
    "shoulder_half_width": 0.23,
    "shoulder_drop": 0.045,
    "underarm_height": 0.47,
    "sleeve_top": 0.21,
    "sleeve_under": 0.11,
    "sleeve_angle": 0.5,  # rad below horizontal
    "neck_half_width": 0.095,
    "front_neck_drop": 0.085,
    "back_neck_drop": 0.025,
}
# Uniform scalings keep normalized outlines identical, so one texture fits all.
SIZES = {"S": 0.92, "M": 1.0, "L": 1.08, "XL": 1.16}
# Texture atlas: front panel in the left half, back panel (seen from behind)
# mirrored into the right half.
NECK_SAMPLES = 10


def outline(neck_drop, scale=1.0):
    """Counter-clockwise outline and per-edge seam flags."""
    p = {key: value * scale for key, value in BASE.items() if key != "sleeve_angle"}
    angle = BASE["sleeve_angle"]
    direction = (math.cos(angle), -math.sin(angle))
    length = p["length"]
    underarm = (p["body_half_width"], p["underarm_height"])
    shoulder = (p["shoulder_half_width"], length - p["shoulder_drop"])
    cuff_top = (
        shoulder[0] + p["sleeve_top"] * direction[0],
        shoulder[1] + p["sleeve_top"] * direction[1],
    )
    cuff_under = (
        underarm[0] + p["sleeve_under"] * direction[0],
        underarm[1] + p["sleeve_under"] * direction[1],
    )
    neck = (p["neck_half_width"], length)
    drop = neck_drop * scale

    def mirror(point):
        return (-point[0], point[1])

    vertices, seams = [], []

    def add(point, seam_after):
        vertices.append(point)
        seams.append(seam_after)

    add((-underarm[0], 0.0), False)  # hem
    add((underarm[0], 0.0), True)  # right side seam
    add(underarm, True)  # under-sleeve seam
    add(cuff_under, False)  # cuff opening
    add(cuff_top, True)  # top-sleeve seam
    add(shoulder, True)  # shoulder seam
    for i in range(NECK_SAMPLES + 1):
        phi = math.pi / 2 - math.pi * i / NECK_SAMPLES
        add(
            (neck[0] * math.sin(phi), length - drop * math.cos(phi)),
            i == NECK_SAMPLES,
        )
    add(mirror(shoulder), True)
    add(mirror(cuff_top), False)
    add(mirror(cuff_under), True)
    add(mirror(underarm), True)
    return vertices, seams


def extent(scale=1.0):
    vertices, _ = outline(BASE["front_neck_drop"], scale)
    half_width = max(abs(x) for x, _ in vertices)
    return half_width, BASE["length"] * scale


def uv_mapping(scale, back):
    half_width, length = extent(scale)
    sign = -1.0 if back else 1.0
    return {
        "offset": [0.75 if back else 0.25, 1.0],
        "scale": [sign / (4 * half_width), -1.0 / length],
    }


def pattern(size="M", mass=None):
    scale = SIZES[size]
    panels = {}
    for name, drop, back in (
        ("front", BASE["front_neck_drop"], False),
        ("back", BASE["back_neck_drop"], True),
    ):
        vertices, seams = outline(drop, scale)
        panels[name] = {
            "outline": [list(v) for v in vertices],
            "seam": seams,
            "uv": uv_mapping(scale, back),
        }
    # Cotton jersey of about 160 g/m2; a size M shirt weighs about 0.17 kg.
    return {
        "name": f"tshirt_{size}",
        "size": size,
        "mass": mass if mass is not None else round(0.17 * scale**2, 4),
        **panels,
    }


def normalized_point(x, y, back=False):
    """Atlas uv of a size-independent pattern point."""
    mapping = uv_mapping(1.0, back)
    return (
        mapping["offset"][0] + mapping["scale"][0] * x,
        mapping["offset"][1] + mapping["scale"][1] * y,
    )
