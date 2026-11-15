"""Pictures of perceived objects for the robot's screen, without ROS.

PPC rule 16 (Communicating Perception): the robot shows one object at a time,
and the surrounding scene must remain visible. A picture is the whole head
camera image with a frame around the object and its label, never a crop.

The frame is the object's box (centre `position`, extents `size`, axes of the
frame the position is given in, z up) projected into the image with the
camera's intrinsics (CameraInfo k, the image is not rectified: distortion is
ignored) and the transform from that frame into the camera's optical frame
(z forward, x right, y down).
"""

import io
import itertools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MIN_DEPTH = 0.05  # m; box corners closer to the camera plane are not projected
PAD = 4  # px of margin around the projected box
FRAME_COLOR = (0, 230, 118)  # bright green, with black edges on any background
EDGE_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)
JPEG_QUALITY = 90
# TrueType fonts tried for the label, before Pillow's built-in one.
FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "Arial Bold.ttf",
)
CHANNELS = {"rgb8": (3, (0, 1, 2)), "bgr8": (3, (2, 1, 0))}
CHANNELS.update({"rgba8": (4, (0, 1, 2)), "bgra8": (4, (2, 1, 0))})


def rotation_matrix(x, y, z, w):
    """3x3 rotation of the unit quaternion (x, y, z, w)."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"not a rotation: {(x, y, z, w)}")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def box_corners(center, size):
    """The 8 corners (8x3) of the box with `center` and extents `size`."""
    center, half = np.asarray(center, float), np.asarray(size, float) / 2.0
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    return center + signs * half


def project(points, k):
    """Pixels (Nx2) of camera-frame points in front of the camera, or None.

    `k` is the row-major 3x3 intrinsic matrix (CameraInfo k). Points closer
    than MIN_DEPTH to the camera plane, or behind it, are left out.
    """
    points = np.asarray(points, float)
    front = points[points[:, 2] > MIN_DEPTH]
    if len(front) == 0:
        return None
    fx, _, cx, _, fy, cy = np.asarray(k, float)[:6]
    u = fx * front[:, 0] / front[:, 2] + cx
    v = fy * front[:, 1] / front[:, 2] + cy
    return np.stack([u, v], axis=1)


def image_box(center, size, rotation, translation, k, width, height, pad=PAD):
    """((left, top, right, bottom), cut) of the object in the image, or (None, None).

    `rotation` (x, y, z, w) and `translation` take a point from the object's
    frame into the camera's optical frame. The box is the bounding rectangle
    of the projected corners, `pad` px larger, clipped to the image; `cut`
    tells that the clipping took some of it away (the object is partly out
    of view). None when the object is behind the camera or out of the image.
    """
    corners = box_corners(center, size) @ rotation_matrix(*rotation).T
    pixels = project(corners + np.asarray(translation, float), k)
    if pixels is None:
        return None, None
    left, top = pixels.min(axis=0) - pad
    right, bottom = pixels.max(axis=0) + pad
    clipped = (max(left, 0.0), max(top, 0.0), min(right, width), min(bottom, height))
    if clipped[2] - clipped[0] < 1.0 or clipped[3] - clipped[1] < 1.0:
        return None, None
    cut = clipped != (left, top, right, bottom)
    return tuple(int(round(value)) for value in clipped), cut


def image_array(data, width, height, step, encoding):
    """HxWx3 uint8 RGB array of a sensor_msgs/Image (8-bit RGB / BGR, with alpha)."""
    if encoding not in CHANNELS:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    channels, order = CHANNELS[encoding]
    rows = np.frombuffer(bytes(data), dtype=np.uint8)
    if step < width * channels or len(rows) < step * height:
        raise ValueError(f"image data too short for {width}x{height} {encoding}")
    rows = rows[: step * height].reshape(height, step)[:, : width * channels]
    return np.ascontiguousarray(rows.reshape(height, width, channels)[:, :, order])


def font(size):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_object(rgb, box, label):
    """PIL image of `rgb` (HxWx3) with a frame around `box` and a `label` tag.

    The frame is drawn just outside `box` so it does not hide the object; the
    tag sits above the frame, else below it, else inside its top. Everything
    else of the image is kept.
    """
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line = max(3, round(min(width, height) / 120))
    left, top, right, bottom = box

    def rectangle(grow, color):
        draw.rectangle(
            (left - grow, top - grow, right - 1 + grow, bottom - 1 + grow),
            outline=color,
        )

    rectangle(0, EDGE_COLOR)
    for grow in range(1, line + 1):
        rectangle(grow, FRAME_COLOR)
    rectangle(line + 1, EDGE_COLOR)

    text_font = font(max(14, round(height / 18)))
    x0, y0, x1, y1 = draw.textbbox((0, 0), label, font=text_font)
    margin = max(2, line)
    tag_w, tag_h = x1 - x0 + 2 * margin, y1 - y0 + 2 * margin
    outer = line + 1
    if top - outer - tag_h >= 0:
        tag_top = top - outer - tag_h
    elif bottom + outer + tag_h <= height:
        tag_top = bottom + outer
    else:
        tag_top = max(top, 0)
    tag_left = min(max(left - outer, 0), max(width - tag_w, 0))
    draw.rectangle(
        (tag_left, tag_top, tag_left + tag_w - 1, tag_top + tag_h - 1),
        fill=FRAME_COLOR,
        outline=EDGE_COLOR,
    )
    draw.text(
        (tag_left + margin - x0, tag_top + margin - y0),
        label,
        fill=TEXT_COLOR,
        font=text_font,
    )
    return image


def encode_jpeg(image, quality=JPEG_QUALITY):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def place_words(surface):
    """'on the dining table' for a surface label such as dining_table; '' for none."""
    if not surface:
        return ""
    return "on the " + " ".join(surface.replace("_", " ").split())


def object_sentence(label, surface, trash=False, shown=True):
    """What the robot says about one perceived object.

    Shown: "This is the mug on the dining table." Not shown (no image):
    "I recognized the mug on the dining table." A trash item adds "It is trash."
    """
    place = place_words(surface)
    words = f"the {label} {place}".strip() if place else f"the {label}"
    text = f"This is {words}." if shown else f"I recognized {words}."
    return f"{text} It is trash." if trash else text
