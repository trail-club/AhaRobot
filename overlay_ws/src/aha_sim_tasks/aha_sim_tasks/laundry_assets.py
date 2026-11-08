"""Generate the laundry-room meshes and textures procedurally.

Everything is derived from code and a fixed seed, so the build is
reproducible and no third-party asset enters the repository. Dimensions follow
the RoboCup@Home 2026 Incheon laundry area: a graphite front-loading washer,
a white desk, a white lattice basket and adult T-shirts.

    python3 laundry_assets.py --output <directory>
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from . import laundry_pattern
except ImportError:  # executed as a build script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import laundry_pattern

SEED = 2026
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/open-sans")

# Washer geometry in its model frame: floor centre, +x out of the door.
WASHER = {
    "depth": 0.565,
    "width": 0.595,
    "height": 0.845,
    "foot": 0.02,
    "panel_bottom": 0.735,  # control panel band above
    "door_center_z": 0.40,
    "opening_radius": 0.19,
    "door_radius": 0.255,
    "drum_radius": 0.235,
    "drum_depth": 0.40,
    "lip_radius": 0.165,
}
BASKET = {
    "top": (0.56, 0.39),
    "bottom": (0.48, 0.31),
    "height": 0.27,
    "wall": 0.006,
    "floor": 0.012,
    "rim": 0.035,
    "handle": (0.13, 0.035),  # cut-out width and height below the rim band
}
TABLE = {"size": (1.2, 0.6), "height": 0.75, "top": 0.025, "leg": 0.03}

# T-shirt colourways. Charcoal matches the shirts used in Incheon 2026.
DESIGNS = {
    "charcoal": {"color": (66, 68, 74), "heather": 0.10},
    "navy": {"color": (34, 42, 70)},
    "black": {"color": (30, 30, 32)},
    "white": {"color": (234, 234, 230)},
    "heather_gray": {"color": (158, 158, 160), "heather": 0.28},
    "red": {"color": (176, 32, 40)},
    "royal_blue": {"color": (38, 76, 166)},
    "lime": {
        "color": (136, 192, 44),
        "print": {"lines": ["ROBOT", "LEAGUE"], "color": (28, 40, 28), "size": 0.06},
    },
    "sky_blue": {
        "color": (126, 174, 212),
        "print": {
            "lines": ["@HOME"],
            "color": (244, 244, 244),
            "size": 0.035,
            "offset": (0.08, 0.0),
        },
    },
    "yellow": {"color": (236, 198, 44)},
    "stripes": {
        "color": (238, 238, 234),
        "stripes": {"color": (32, 46, 92), "period": 0.045, "width": 0.016},
    },
    "olive": {"color": (96, 102, 62), "graphic": "mountains"},
    "maroon": {
        "color": (108, 30, 44),
        "print": {
            "lines": ["TEAM", "07"],
            "color": (236, 220, 200),
            "size": 0.08,
            "back": True,
        },
    },
    "orange": {"color": (230, 112, 34)},
}


# --------------------------------------------------------------------- noise
def periodic_noise(height, width, rng, beta=2.0, stretch=(1.0, 1.0)):
    """Tileable fractal noise in [0, 1] from filtered white noise."""
    spectrum = np.fft.fft2(rng.standard_normal((height, width)))
    fy = np.fft.fftfreq(height)[:, None] * height * stretch[1]
    fx = np.fft.fftfreq(width)[None, :] * width * stretch[0]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    field = np.real(np.fft.ifft2(spectrum / radius ** (beta / 2)))
    field -= field.min()
    return field / max(field.max(), 1e-9)


def save(image, path, quality=92):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jpg":
        image.convert("RGB").save(path, quality=quality)
    else:
        image.save(path)


def to_image(rgb):
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def normal_map(height, strength):
    """Tangent-space normal map from a height field (rows grow downwards)."""
    dy, dx = np.gradient(height)
    normal = np.dstack((-dx * strength, dy * strength, np.ones_like(height)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    return to_image((normal * 0.5 + 0.5) * 255)


def font(size):
    for directory in FONT_DIRS:
        for name in ("DejaVuSans-Bold.ttf", "OpenSans-Bold.ttf"):
            path = Path(directory) / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


# -------------------------------------------------------------------- meshes
class Mesh:
    def __init__(self):
        self.v, self.vt, self.vn, self.f = [], [], [], []

    def vertex(self, p, uv, n):
        self.v.append(tuple(p))
        self.vt.append(tuple(uv))
        self.vn.append(tuple(n))
        return len(self.v) - 1

    def quad(self, p, uv, n):
        i = [self.vertex(p[k], uv[k], n) for k in range(4)]
        self.f += [(i[0], i[1], i[2]), (i[0], i[2], i[3])]

    def triangles(self, positions, uvs, normals, faces):
        base = len(self.v)
        for p, uv, n in zip(positions, uvs, normals):
            self.vertex(p, uv, n)
        self.f += [(base + a, base + b, base + c) for a, b, c in faces]

    def merge(self, other):
        base = len(self.v)
        self.v += other.v
        self.vt += other.vt
        self.vn += other.vn
        self.f += [(a + base, b + base, c + base) for a, b, c in other.f]
        return self

    def transform(self, rotation=None, translation=(0, 0, 0)):
        r = np.eye(3) if rotation is None else np.asarray(rotation)
        self.v = [tuple(r @ np.asarray(p) + translation) for p in self.v]
        self.vn = [tuple(r @ np.asarray(n)) for n in self.vn]
        return self

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {path.name}: generated by aha_sim_tasks/laundry_assets.py"]
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in self.v]
        lines += [f"vt {u:.5f} {1 - t:.5f}" for u, t in self.vt]
        lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in self.vn]
        lines += [
            f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}"
            for a, b, c in self.f
        ]
        path.write_text("\n".join(lines) + "\n")


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def box(size, center=(0, 0, 0), uv_scale=1.0):
    """Axis-aligned box; texture coordinates in metres times uv_scale."""
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = center
    mesh = Mesh()
    faces = [
        (
            (1, 0, 0),
            [(sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz), (sx, -sy, sz)],
            (1, 2),
        ),
        (
            (-1, 0, 0),
            [(-sx, sy, -sz), (-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz)],
            (1, 2),
        ),
        (
            (0, 1, 0),
            [(sx, sy, -sz), (-sx, sy, -sz), (-sx, sy, sz), (sx, sy, sz)],
            (0, 2),
        ),
        (
            (0, -1, 0),
            [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)],
            (0, 2),
        ),
        (
            (0, 0, 1),
            [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)],
            (0, 1),
        ),
        (
            (0, 0, -1),
            [(-sx, sy, -sz), (sx, sy, -sz), (sx, -sy, -sz), (-sx, -sy, -sz)],
            (0, 1),
        ),
    ]
    for normal, corners, (a, b) in faces:
        p = [(x + cx, y + cy, z + cz) for x, y, z in corners]
        uv = [
            ((c[a] + size[a] / 2) * uv_scale, (c[b] + size[b] / 2) * uv_scale)
            for c in corners
        ]
        mesh.quad(p, uv, normal)
    return mesh


def lathe(
    profile, segments=64, inward=False, u_repeat=1.0, v_scale=1.0, sweep=2 * math.pi
):
    """Surface of revolution about +x. profile: [(x, radius)]."""
    mesh = Mesh()
    lengths = [0.0]
    for a, b in zip(profile, profile[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    rings = []
    for k in range(segments + 1):
        angle = sweep * k / segments
        c, s = math.cos(angle), math.sin(angle)
        ring = []
        for j, (x, r) in enumerate(profile):
            # Profile tangent for the normal.
            a = profile[max(j - 1, 0)]
            b = profile[min(j + 1, len(profile) - 1)]
            tx, tr = b[0] - a[0], b[1] - a[1]
            nx, nr = -tr, tx  # rotate the tangent by +90 degrees
            norm = math.hypot(nx, nr) or 1.0
            nx, nr = nx / norm, nr / norm
            if inward:
                nx, nr = -nx, -nr
            ring.append(
                mesh.vertex(
                    (x, r * c, r * s),
                    (u_repeat * k / segments, lengths[j] * v_scale),
                    (nx, nr * c, nr * s),
                )
            )
        rings.append(ring)
    for k in range(segments):
        for j in range(len(profile) - 1):
            a, b, c, d = (
                rings[k][j],
                rings[k + 1][j],
                rings[k + 1][j + 1],
                rings[k][j + 1],
            )
            if inward:
                mesh.f += [(a, c, b), (a, d, c)]
            else:
                mesh.f += [(a, b, c), (a, c, d)]
    return mesh


def panel_with_hole(width, height, hole_center, radius, segments=64):
    """Rectangle in the y-z plane (normal +x) with a circular hole.

    Rays from the hole centre pair each rectangle point with a circle point;
    the rectangle corners are always among them, so the panel has no gaps.
    """
    hy, hz = hole_center
    angles = {2 * math.pi * k / segments for k in range(segments)}
    for cy, cz in (
        (width / 2, height / 2),
        (-width / 2, height / 2),
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
    ):
        angles.add(math.atan2(cz - hz, cy - hy) % (2 * math.pi))
    angles = sorted(angles)
    ring = []
    for angle in angles:
        c, s = math.cos(angle), math.sin(angle)
        ty = (
            ((width / 2 if c > 0 else -width / 2) - hy) / c
            if abs(c) > 1e-12
            else math.inf
        )
        tz = (
            ((height / 2 if s > 0 else -height / 2) - hz) / s
            if abs(s) > 1e-12
            else math.inf
        )
        t = min(ty, tz)
        ring.append(((hy + c * t, hz + s * t), (hy + c * radius, hz + s * radius)))
    mesh = Mesh()
    for k in range(len(ring)):
        (o0, i0), (o1, i1) = ring[k], ring[(k + 1) % len(ring)]
        p = [(0, *o0), (0, *o1), (0, *i1), (0, *i0)]
        uv = [(q[1] + width / 2, q[2] + height / 2) for q in p]
        mesh.quad(p, uv, (1, 0, 0))
    return mesh


def rounded_rect(width, depth, radius, segments=6):
    """Counter-clockwise rounded rectangle in the x-y plane."""
    points = []
    for cx, cy, start in (
        (depth / 2 - radius, width / 2 - radius, 0),
        (-depth / 2 + radius, width / 2 - radius, 90),
        (-depth / 2 + radius, -width / 2 + radius, 180),
        (depth / 2 - radius, -width / 2 + radius, 270),
    ):
        for k in range(segments + 1):
            a = math.radians(start + 90 * k / segments)
            points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return points


def prism(outline, z0, z1, uv_scale=1.0, skip=None, caps=True):
    """Extrude a convex outline; edges whose midpoint satisfies skip are left open."""
    mesh = Mesh()
    perimeter = 0.0
    n = len(outline)
    for k in range(n):
        a, b = outline[k], outline[(k + 1) % n]
        length = math.dist(a, b)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if length > 1e-9 and not (skip and skip(mid)):
            nx, ny = b[1] - a[1], -(b[0] - a[0])
            norm = math.hypot(nx, ny)
            p = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
            uv = [
                (perimeter * uv_scale, z0 * uv_scale),
                ((perimeter + length) * uv_scale, z0 * uv_scale),
                ((perimeter + length) * uv_scale, z1 * uv_scale),
                (perimeter * uv_scale, z1 * uv_scale),
            ]
            mesh.quad(p, uv, (nx / norm, ny / norm, 0))
        perimeter += length
    if caps:
        cx = sum(p[0] for p in outline) / n
        cy = sum(p[1] for p in outline) / n
        for z, normal in ((z1, 1), (z0, -1)):
            positions = [(cx, cy, z)] + [(x, y, z) for x, y in outline]
            uvs = [(x * uv_scale, y * uv_scale) for x, y, _ in positions]
            faces = [
                (0, k + 1, (k + 1) % n + 1)
                if normal > 0
                else (0, (k + 1) % n + 1, k + 1)
                for k in range(n)
            ]
            mesh.triangles(positions, uvs, [(0, 0, normal)] * len(positions), faces)
    return mesh


# ------------------------------------------------------------------ textures
def floor_texture(directory, rng, name, base, spread):
    """Laminate planks 0.2 m wide over a tileable 2.4 m square."""
    size, metres = 2048, 2.4
    px = size / metres
    grain = periodic_noise(size, size, rng, beta=2.6, stretch=(0.06, 1.0))
    fine = periodic_noise(size, size, rng, beta=1.4, stretch=(0.15, 1.0))
    knots = periodic_noise(size, size, rng, beta=3.2)
    image = np.zeros((size, size, 3))
    height = np.ones((size, size))
    rows = int(round(metres / 0.2))
    row_px = size / rows
    for row in range(rows):
        y0, y1 = int(round(row * row_px)), int(round((row + 1) * row_px))
        x = rng.uniform(0, metres)
        start = x
        cuts = []
        while x < start + metres - 0.3:
            cuts.append(x % metres)
            x += rng.uniform(0.7, 1.35)
        cuts = sorted(cuts) or [0.0]
        for k, cut in enumerate(cuts):
            end = cuts[(k + 1) % len(cuts)] + (metres if k + 1 == len(cuts) else 0)
            tone = np.array(base) * (1 + rng.uniform(-spread, spread))
            shift = rng.integers(0, size)
            xs = np.arange(int(cut * px), int(end * px)) % size
            band = np.roll(grain, shift, axis=1)[y0:y1][:, xs]
            detail = np.roll(fine, shift // 2, axis=1)[y0:y1][:, xs]
            spots = np.roll(knots, shift, axis=0)[y0:y1][:, xs]
            shade = (
                0.86
                + 0.22 * band
                + 0.08 * detail
                - 0.10 * np.clip(spots - 0.75, 0, 1) * 4
            )
            image[y0:y1][:, xs] = tone[None, None, :] * shade[..., None]
            # Bevelled joints between planks.
            height[y0:y1, int(cut * px) % size] = 0.0
        height[y0, :] = 0.0
    joints = height < 0.5
    image[joints] *= 0.55
    save(to_image(image), directory / f"{name}.jpg")
    blurred = Image.fromarray((height * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.2)
    )
    save(
        normal_map(np.asarray(blurred, float) / 255, 2.0),
        directory / f"{name}_normal.png",
    )


def plain_texture(
    directory, rng, name, color, amount, size=512, beta=1.8, stretch=(1, 1)
):
    noise = periodic_noise(size, size, rng, beta=beta, stretch=stretch)
    rgb = np.array(color)[None, None, :] * (1 - amount / 2 + amount * noise[..., None])
    save(to_image(rgb), directory / f"{name}.jpg")


def drum_texture(directory, rng):
    """Perforated, brushed stainless steel: a 0.15 m tile."""
    size = 512
    px = size / 0.15
    brushed = periodic_noise(size, size, rng, beta=1.8, stretch=(0.02, 1.0))
    rgb = np.full((size, size, 3), 186.0) * (0.9 + 0.2 * brushed[..., None])
    yy, xx = np.mgrid[0:size, 0:size] / px
    pitch = 0.015
    height = np.ones((size, size))
    for row in range(int(0.15 / (pitch * 0.866)) + 1):
        cy = row * pitch * 0.866
        offset = (row % 2) * pitch / 2
        for col in range(int(0.15 / pitch) + 1):
            cx = col * pitch + offset
            d = np.hypot(
                (xx - cx + 0.075) % 0.15 - 0.075, (yy - cy + 0.075) % 0.15 - 0.075
            )
            hole = d < 0.0022
            rgb[hole] = (24, 24, 26)
            height = np.minimum(height, np.clip((d - 0.0022) / 0.003, 0, 1))
    save(to_image(rgb), directory / "washer_drum.jpg")
    save(normal_map(height, 3.0), directory / "washer_drum_normal.png")


def control_panel_texture(directory, rng, body):
    """Graphite control band: program dial labels, display and buttons."""
    w, h = 2048, 384  # 0.595 m x 0.11 m
    image = Image.new("RGB", (w, h), (24, 25, 27))
    draw = ImageDraw.Draw(image)
    px = w / WASHER["width"]
    # Detergent drawer on the left (viewer's left is -y, i.e. image left).
    draw.rounded_rectangle(
        (int(0.012 * px), 22, int(0.17 * px), h - 22),
        radius=14,
        fill=tuple(int(c * 1.05) for c in body),
        outline=(70, 70, 74),
        width=3,
    )
    draw.rounded_rectangle(
        (int(0.03 * px), h - 90, int(0.15 * px), h - 52), radius=10, fill=(30, 30, 33)
    )
    # Program dial with a chrome ring.
    cx, cy, r = int(0.30 * px), h // 2, 120
    draw.ellipse((cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8), fill=(170, 172, 176))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(40, 41, 44))
    draw.ellipse(
        (cx - r + 30, cy - r + 30, cx + r - 30, cy + r - 30), fill=(60, 61, 64)
    )
    draw.line((cx, cy - r + 34, cx, cy - r + 70), fill=(230, 230, 230), width=8)
    labels = [
        "Cotton",
        "Eco",
        "Synthetic",
        "Mix",
        "Quick 15'",
        "Wool",
        "Spin",
        "Rinse",
        "Baby",
        "Bedding",
        "Delicate",
        "Drum Clean",
    ]
    small = font(26)
    for k, label in enumerate(labels):
        column = cx - r - 260 if k < 6 else cx + r + 40
        draw.text((column, 30 + 52 * (k % 6)), label, fill=(200, 200, 204), font=small)
    # Display and buttons on the right.
    dx = int(0.475 * px)
    draw.rounded_rectangle((dx - 150, 70, dx + 150, 190), radius=12, fill=(8, 8, 9))
    draw.text((dx - 105, 82), "1:20", fill=(238, 238, 238), font=font(86))
    for k, label in enumerate(("Temp", "Spin", "Delay", "Start")):
        bx = int(0.445 * px) + k * 130
        draw.ellipse(
            (bx - 30, 250, bx + 30, 310),
            outline=(150, 150, 154),
            width=4,
            fill=(34, 34, 37),
        )
        draw.text((bx - 34, 316), label, fill=(190, 190, 194), font=font(22))
    draw.text(
        (int(0.18 * px), h - 64), "8 kg  INVERTER", fill=(150, 150, 154), font=font(28)
    )
    save(image, directory / "washer_panel.jpg")


def body_texture(directory, rng, name, color):
    """Metallic-flake appliance paint with an energy label and a sticker."""
    size = 1024  # covers a 1 m square of panel
    flake = rng.random((size, size))
    tone = periodic_noise(size, size, rng, beta=2.2)
    rgb = np.array(color)[None, None, :] * (
        0.93 + 0.08 * tone[..., None] + 0.05 * flake[..., None]
    )
    save(to_image(rgb), directory / f"{name}.jpg")


def front_texture(directory, rng, name, color):
    """Front panel below the controls (0.595 x 0.715 m) with labels."""
    w, h = 1024, 1230
    px = w / WASHER["width"]
    flake = rng.random((h, w))
    tone = periodic_noise(h, w, rng, beta=2.2)
    rgb = np.array(color)[None, None, :] * (
        0.93 + 0.08 * tone[..., None] + 0.05 * flake[..., None]
    )
    image = to_image(rgb)
    draw = ImageDraw.Draw(image)
    # Energy label, top right (viewer's right is +y, image right).
    x0, y0 = int(0.47 * px), int(0.03 * px)
    draw.rectangle(
        (x0, y0, x0 + int(0.09 * px), y0 + int(0.15 * px)), fill=(244, 244, 240)
    )
    bars = [
        (0, 150, 64),
        (80, 180, 50),
        (190, 210, 40),
        (250, 200, 30),
        (240, 130, 30),
        (220, 50, 40),
    ]
    for k, color_bar in enumerate(bars):
        top = y0 + 18 + k * 30
        draw.rectangle((x0 + 10, top, x0 + 40 + 12 * k, top + 20), fill=color_bar)
    draw.text((x0 + 12, y0 + int(0.12 * px)), "A+++", fill=(20, 20, 20), font=font(22))
    # Blue venue sticker, bottom left.
    draw.rounded_rectangle(
        (int(0.03 * px), int(0.62 * px), int(0.11 * px), int(0.66 * px)),
        radius=6,
        fill=(40, 120, 200),
    )
    save(image, directory / f"{name}.jpg")


def shirt_textures(directory, rng):
    """Front/back atlas per colourway plus a shared knit normal map."""
    w, h = 2048, 1024
    half_width, length = laundry_pattern.extent(1.0)
    vv, uu = np.mgrid[0:h, 0:w]
    u = (uu + 0.5) / w
    v = (vv + 0.5) / h
    back = u >= 0.5
    # Pattern coordinates of every texel, front and back (mirrored).
    scale_u = 1 / (4 * half_width)
    px_x = np.where(back, (0.75 - u) / scale_u, (u - 0.25) / scale_u)
    px_y = (1 - v) * length
    knit = periodic_noise(h, w, rng, beta=1.2, stretch=(0.5, 1.0))
    wales = 0.5 + 0.5 * np.sin(2 * math.pi * px_x / 0.0016)
    height = 0.6 * knit + 0.4 * wales
    seams = np.zeros((h, w))

    def to_pixel(x, y, rear):
        uu_, vv_ = laundry_pattern.normalized_point(x, y, rear)
        return uu_ * w, vv_ * h

    stitch = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(stitch)
    for rear in (False, True):
        drop = laundry_pattern.BASE["back_neck_drop" if rear else "front_neck_drop"]
        outline, seam_flags = laundry_pattern.outline(drop)
        n = len(outline)
        for k in range(n):
            a, b = outline[k], outline[(k + 1) % n]
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm  # inward for a counter-clockwise outline
            if seam_flags[k]:
                offsets = (0.004,)
            elif ay < 0.01 and by < 0.01:
                offsets = (0.018, 0.024)  # double-needle hem
            else:
                offsets = (0.014, 0.020)  # cuffs and neck band
            for d in offsets:
                p0 = to_pixel(ax + nx * d, ay + ny * d, rear)
                p1 = to_pixel(bx + nx * d, by + ny * d, rear)
                draw.line((p0, p1), fill=255, width=2)
    seams = np.asarray(stitch, dtype=float) / 255
    # Neck rib band: a 2 cm strip along the neckline.
    neck_band = np.zeros((h, w), bool)
    for rear in (False, True):
        drop = laundry_pattern.BASE["back_neck_drop" if rear else "front_neck_drop"]
        a = laundry_pattern.BASE["neck_half_width"]
        top = laundry_pattern.BASE["length"]
        side = back if rear else ~back
        ex = px_x / a
        ey = (top - px_y) / drop
        r = np.sqrt(ex * ex + ey * ey)
        outer = np.sqrt((px_x / (a + 0.02)) ** 2 + ((top - px_y) / (drop + 0.02)) ** 2)
        neck_band |= side & (r >= 1.0) & (outer <= 1.0) & (px_y <= top)
    rib = 0.5 + 0.5 * np.sin(2 * math.pi * px_x / 0.003)
    height = np.where(neck_band, 0.5 + 0.5 * rib, height) - 0.6 * seams
    save(normal_map(height, 6.0), directory / "shirt_knit_normal.png")

    for name, design in DESIGNS.items():
        base = np.array(design["color"], float)
        heather = design.get("heather", 0.05)
        fibers = periodic_noise(h, w, rng, beta=0.6, stretch=(0.4, 1.0))
        tone = periodic_noise(h, w, rng, beta=2.4)
        rgb = base[None, None, :] * (1 - heather / 2 + heather * fibers[..., None])
        rgb *= (0.95 + 0.06 * tone + 0.04 * (knit - 0.5))[..., None]
        if "stripes" in design:
            s = design["stripes"]
            band = ((px_y % s["period"]) < s["width"]) & (px_y < length - 0.06)
            rgb[band] = np.array(s["color"]) * (0.95 + 0.08 * fibers[band, None])
        rgb[neck_band] *= 0.9
        rgb *= (1 - 0.18 * seams)[..., None]
        image = to_image(rgb)
        if "print" in design or "graphic" in design:
            draw = ImageDraw.Draw(image)
            spec = design.get("print", {})
            rear = spec.get("back", False)
            cx, cy = to_pixel(spec.get("offset", (0.0, 0.0))[0], length * 0.66, rear)
            if "lines" in spec:
                size = int(spec["size"] / length * h)
                face = font(size)
                for k, line in enumerate(spec["lines"]):
                    box = draw.textbbox((0, 0), line, font=face)
                    tw = box[2] - box[0]
                    y = cy + (k - len(spec["lines"]) / 2) * size * 1.15
                    draw.text((cx - tw / 2, y), line, fill=spec["color"], font=face)
            if design.get("graphic") == "mountains":
                scale = h / length
                pts = [
                    (-0.11, 0.0),
                    (-0.04, 0.09),
                    (0.0, 0.05),
                    (0.05, 0.12),
                    (0.12, 0.0),
                ]
                poly = [(cx + x * scale, cy - y * scale + 40) for x, y in pts]
                draw.polygon(poly, fill=(226, 220, 196))
                draw.ellipse(
                    (
                        cx + 0.05 * scale,
                        cy - 0.17 * scale,
                        cx + 0.09 * scale,
                        cy - 0.13 * scale,
                    ),
                    fill=(236, 170, 60),
                )
            # Screen prints crack slightly with the fabric.
            image = Image.fromarray(
                (np.asarray(image, float) * (0.97 + 0.06 * knit[..., None]))
                .clip(0, 255)
                .astype(np.uint8)
            )
        save(image, directory / f"shirt_{name}.jpg")


def generate_textures(directory):
    rng = np.random.default_rng(SEED)
    floor_texture(directory, rng, "floor_oak", (214, 186, 150), 0.07)
    floor_texture(directory, rng, "floor_walnut", (150, 108, 76), 0.09)
    plain_texture(directory, rng, "wall_paint", (243, 243, 240), 0.03)
    plain_texture(directory, rng, "venue_floor", (118, 120, 124), 0.18, beta=1.2)
    plain_texture(directory, rng, "venue_wall", (176, 178, 182), 0.10, beta=2.0)
    plain_texture(directory, rng, "melamine_white", (238, 238, 235), 0.025)
    plain_texture(directory, rng, "powder_white", (230, 230, 228), 0.04, beta=1.0)
    plain_texture(directory, rng, "plastic_white", (236, 236, 232), 0.05, beta=1.4)
    plain_texture(directory, rng, "plastic_blue", (44, 102, 168), 0.06, beta=1.4)
    plain_texture(directory, rng, "rubber_gray", (52, 52, 55), 0.12, beta=1.2)
    plain_texture(directory, rng, "door_plastic", (34, 35, 38), 0.06, beta=1.6)
    drum_texture(directory, rng)
    for name, color in (("graphite", (62, 63, 67)), ("white", (232, 232, 230))):
        body_texture(directory, rng, f"washer_body_{name}", color)
        front_texture(directory, rng, f"washer_front_{name}", color)
    control_panel_texture(directory, rng, (62, 63, 67))
    shirt_textures(directory, rng)


# -------------------------------------------------------------------- models
def washer_meshes(directory):
    w = WASHER
    d2, w2, top = w["depth"] / 2, w["width"] / 2, w["height"]
    zc, ro = w["door_center_z"], w["opening_radius"]
    # Cabinet without its front, rounded vertical edges.
    outline = rounded_rect(w["width"], w["depth"], 0.02)
    body = prism(outline, w["foot"], top, uv_scale=1.0, skip=lambda m: m[0] > d2 - 1e-6)
    body.write(directory / "washer_body.obj")
    # Front below the controls, with the porthole.
    lower_h = w["panel_bottom"] - w["foot"]
    front = panel_with_hole(
        w["width"] - 0.04, lower_h, (0.0, zc - w["foot"] - lower_h / 2), ro, segments=72
    )
    # Texture spans the full front for the decals.
    for k, (x, y, z) in enumerate(front.v):
        front.vt[k] = ((y + w2) / w["width"], (z + lower_h / 2) / lower_h)
    front.transform(translation=(d2, 0, w["foot"] + lower_h / 2))
    sides = Mesh()
    for y0 in (-w2, w2 - 0.02):
        strip = box(
            (0.002, 0.02, lower_h), (d2 - 0.001, y0 + 0.01, w["foot"] + lower_h / 2)
        )
        sides.merge(strip)
    front.merge(sides)
    front.write(directory / "washer_front.obj")
    panel = Mesh()
    panel.quad(
        [
            (d2 + 0.001, -w2, w["panel_bottom"]),
            (d2 + 0.001, w2, w["panel_bottom"]),
            (d2 + 0.001, w2, top),
            (d2 + 0.001, -w2, top),
        ],
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        (1, 0, 0),
    )
    panel.write(directory / "washer_panel.obj")
    # Rubber gasket from the cabinet opening into the drum lip.
    lip_x = d2 - 0.04
    gasket = lathe(
        [
            (d2 + 0.004, ro + 0.012),
            (d2 + 0.004, ro),
            (d2 - 0.012, ro - 0.012),
            (lip_x + 0.006, w["lip_radius"] + 0.006),
            (lip_x, w["lip_radius"]),
        ],
        segments=72,
        inward=True,
        u_repeat=6,
        v_scale=4,
    )
    gasket.transform(translation=(0, 0, zc))
    gasket.write(directory / "washer_gasket.obj")
    # Drum: perforated cylinder, front lip, back plate and three lifters.
    back_x = lip_x - w["drum_depth"]
    drum = lathe(
        [
            (lip_x, w["lip_radius"]),
            (lip_x, w["drum_radius"]),
            (back_x, w["drum_radius"]),
            (back_x, 0.0),
        ],
        segments=72,
        inward=True,
        u_repeat=10,
        v_scale=1 / 0.15,
    )
    for k in range(3):
        lifter = box((w["drum_depth"] - 0.02, 0.05, 0.035), uv_scale=1 / 0.15)
        angle = math.radians(90 + 120 * k)
        r = w["drum_radius"] - 0.016
        lifter.transform(
            np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(angle), -math.sin(angle)],
                    [0, math.sin(angle), math.cos(angle)],
                ]
            )
        )
        lifter.v = [
            (x + (lip_x + back_x) / 2, y + r * math.cos(angle), z + r * math.sin(angle))
            for x, y, z in lifter.v
        ]
        drum.merge(lifter)
    drum.transform(translation=(0, 0, zc))
    drum.write(directory / "washer_drum.obj")
    feet = Mesh()
    for x in (-d2 + 0.05, d2 - 0.05):
        for y in (-w2 + 0.05, w2 - 0.05):
            feet.merge(box((0.04, 0.04, w["foot"]), (x, y, w["foot"] / 2)))
    feet.write(directory / "washer_feet.obj")
    # Door, in the door link frame: hinge on the z axis, closed door along +y.
    rd = w["door_radius"]
    center = (0.03, rd + 0.01, 0.0)
    frame = lathe(
        [
            (0.0, rd - 0.004),
            (0.02, rd),
            (0.045, rd - 0.01),
            (0.05, rd - 0.05),
            (0.03, 0.17),
            (0.0, 0.165),
        ],
        segments=72,
        u_repeat=4,
        v_scale=4,
    )
    frame.transform(translation=(center[0] - 0.03, center[1], 0))
    frame.write(directory / "washer_door_frame.obj")
    trim = lathe(
        [
            (0.046, rd - 0.012),
            (0.052, rd - 0.02),
            (0.05, rd - 0.045),
            (0.044, rd - 0.05),
        ],
        segments=72,
    )
    trim.transform(translation=(center[0] - 0.03, center[1], 0))
    trim.write(directory / "washer_door_trim.obj")
    glass_profile = [(0.035, 0.172)]
    for k in range(1, 9):
        a = math.radians(90 * k / 8)
        glass_profile.append((0.035 - 0.085 * math.sin(a), 0.172 * math.cos(a) + 1e-4))
    glass = lathe(glass_profile, segments=64)
    glass.transform(translation=(center[0] - 0.03, center[1], 0))
    glass.write(directory / "washer_door_glass.obj")
    handle = box((0.03, 0.035, 0.12), (center[0] + 0.02, center[1] + rd - 0.03, 0.0))
    handle.write(directory / "washer_door_handle.obj")


def basket_mesh(directory):
    """Lattice laundry basket, origin at the floor centre."""
    b = BASKET
    (tx, ty), (bx, by), height = b["top"], b["bottom"], b["height"]
    rim = b["rim"]
    mesh = Mesh()
    mesh.merge(box((bx, by, b["floor"]), (0, 0, b["floor"] / 2), uv_scale=2))

    def wall(length_bottom, length_top, offset_bottom, offset_top, axis, sign, handle):
        """One tapered wall made of bars around rectangular holes."""
        part = Mesh()
        thickness = b["wall"]
        # Wall plane: s along the wall, t up the slope.
        slope = math.hypot(offset_top - offset_bottom, height)

        def point(s, t, inner=0.0):
            frac = t / slope
            half = (length_bottom + (length_top - length_bottom) * frac) / 2
            off = offset_bottom + (offset_top - offset_bottom) * frac - inner
            z = t * height / slope
            x = s * half
            if axis == "x":
                return (sign * off, x, z)
            return (x, sign * off, z)

        def bar(s0, s1, t0, t1):
            for inner, flip in ((0.0, 1), (thickness, -1)):
                p = [
                    point(s0, t0, inner),
                    point(s1, t0, inner),
                    point(s1, t1, inner),
                    point(s0, t1, inner),
                ]
                n = np.cross(np.subtract(p[1], p[0]), np.subtract(p[3], p[0]))
                n = n / (np.linalg.norm(n) or 1)
                if (sign * (n[0] if axis == "x" else n[1])) * flip < 0:
                    p = p[::-1]
                    n = -n
                uv = [
                    (s0 * 2, t0 * 6),
                    (s1 * 2, t0 * 6),
                    (s1 * 2, t1 * 6),
                    (s0 * 2, t1 * 6),
                ]
                part.quad(p, uv, tuple(n))

        holes_s = 9 if length_top > 0.45 else 6
        bars = 0.012 / (length_top / 2)
        t_rim = slope - rim * slope / height
        rows = [(0.035, 0.075), (0.09, 0.13), (0.145, 0.185)]
        bar(-1, 1, 0, 0.025)  # bottom band
        bar(-1, 1, t_rim, slope)  # rim band
        edges = np.linspace(-1, 1, holes_s + 1)
        for k, (t0, t1) in enumerate(rows):
            bar(-1, 1, t1, rows[k + 1][0] if k + 1 < len(rows) else t_rim - 0.015)
            for e in edges:
                bar(max(-1, e - bars / 2), min(1, e + bars / 2), t0, t1)
            bar(-1, 1, rows[k - 1][1] if k else 0.025, t0)
        # Upper band, with a hand hole on the short walls.
        t_top = t_rim - 0.015
        if handle:
            hw = b["handle"][0] / length_top
            bar(-1, -hw, t_top, t_rim)
            bar(hw, 1, t_top, t_rim)
            bar(-1, 1, rows[-1][1], t_top - b["handle"][1] + 0.015)
        else:
            bar(-1, 1, rows[-1][1], t_rim)
        return part

    mesh.merge(wall(by, ty, bx / 2, tx / 2, "x", 1, True))
    mesh.merge(wall(by, ty, bx / 2, tx / 2, "x", -1, True))
    mesh.merge(wall(bx, tx, by / 2, ty / 2, "y", 1, False))
    mesh.merge(wall(bx, tx, by / 2, ty / 2, "y", -1, False))
    # Rolled rim lip.
    outline = rounded_rect(ty + 0.02, tx + 0.02, 0.03)
    inner = rounded_rect(ty - 0.005, tx - 0.005, 0.02)
    lip = Mesh()
    n = len(outline)
    for k in range(n):
        a0, a1 = outline[k], outline[(k + 1) % n]
        b0, b1 = inner[k], inner[(k + 1) % n]
        p = [(*b0, height), (*b1, height), (*a1, height), (*a0, height)]
        lip.quad(p, [(0, 0), (1, 0), (1, 1), (0, 1)], (0, 0, 1))
    mesh.merge(lip)
    mesh.merge(prism(outline, height - 0.018, height, caps=False))
    mesh.write(directory / "basket.obj")


def table_meshes(directory):
    t = TABLE
    (lx, ly), height, top, leg = t["size"], t["height"], t["top"], t["leg"]
    box((lx, ly, top), (0, 0, height - top / 2), uv_scale=1.0).write(
        directory / "table_top.obj"
    )
    frame = Mesh()
    inset = 0.03
    for x in (-lx / 2 + inset + leg / 2, lx / 2 - inset - leg / 2):
        for y in (-ly / 2 + inset + leg / 2, ly / 2 - inset - leg / 2):
            frame.merge(box((leg, leg, height - top), (x, y, (height - top) / 2)))
        # Lower stretcher on each short side, as on the Incheon desk.
        frame.merge(box((leg * 0.8, ly - 2 * inset, 0.025), (x, 0, 0.16)))
    for y in (-ly / 2 + inset + leg / 2, ly / 2 - inset - leg / 2):
        frame.merge(box((lx - 2 * inset, 0.02, 0.05), (0, y, height - top - 0.025)))
    for x in (-lx / 2 + inset + leg / 2, lx / 2 - inset - leg / 2):
        frame.merge(box((0.02, ly - 2 * inset, 0.05), (x, 0, height - top - 0.025)))
    frame.write(directory / "table_frame.obj")


def floor_mesh(directory, size, name, tile, z=0.0):
    """Floor surface with UVs in metres / tile (repeat the texture)."""
    mesh = Mesh()
    (x0, y0), (x1, y1) = size
    mesh.quad(
        [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],
        [
            (x0 / tile, y0 / tile),
            (x1 / tile, y0 / tile),
            (x1 / tile, y1 / tile),
            (x0 / tile, y1 / tile),
        ],
        (0, 0, 1),
    )
    mesh.write(directory / f"{name}.obj")


def generate_meshes(directory):
    floor_mesh(directory, ((-16, -14), (20, 14)), "venue_floor", 3.0)
    washer_meshes(directory)
    basket_mesh(directory)
    table_meshes(directory)


def generate_patterns(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for size in laundry_pattern.SIZES:
        (directory / f"tshirt_{size}.json").write_text(
            json.dumps(laundry_pattern.pattern(size), indent=1) + "\n"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=["patterns", "meshes", "textures"])
    args = parser.parse_args(argv)
    if args.only in (None, "patterns"):
        generate_patterns(args.output / "patterns")
    if args.only in (None, "meshes"):
        generate_meshes(args.output / "meshes")
    if args.only in (None, "textures"):
        generate_textures(args.output / "textures")
    (args.output / "catalog.json").write_text(
        json.dumps(
            {
                "designs": sorted(DESIGNS),
                "sizes": list(laundry_pattern.SIZES),
                "washer": WASHER,
                "basket": BASKET,
                "table": TABLE,
            },
            indent=1,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
