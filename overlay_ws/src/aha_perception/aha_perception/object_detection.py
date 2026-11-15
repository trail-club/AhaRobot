"""Objects from SAM 3.1 masks named by the VLM: candidates, 3D boxes and tracks.

ROS-independent so that object_perception.py, the DGX object_worker and the
unit tests share it. SAM 3.1 finds candidate masks for a few generic phrases
("can", "bottle", ...); the VLM sees the candidates numbered on the photo and
as close-ups and names each one with a label of the object list, or none.
Positions and sizes come from the aligned depth on the robot side, so depth
never leaves the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .people import bbox_of
from .sam31 import erode


# --- Candidates (worker side) ---------------------------------------------

# Generic SAM 3.1 phrases for the RoboCup@Home objects; the VLM names them.
# Specific words find more than brand names do (docs/context/perception.md);
# "item", "food" and "small box" find what the shape words miss.
DEFAULT_PROMPTS = (
    "item",
    "food",
    "small box",
    "box",
    "can",
    "bottle",
    "fruit",
    "cup",
    "bowl",
    "plate",
    "fork",
    "spoon",
    "knife",
    "toy",
)
DEFAULT_CANDIDATE_THRESHOLD = 0.4
MIN_MASK_PIXELS = 120
# Larger masks are furniture, walls or the floor, not graspable objects.
MAX_MASK_SHARE = 0.2
# Masks of different phrases overlapping this much are the same candidate.
DUPLICATE_IOU = 0.7
DEFAULT_MAX_CANDIDATES = 24
# Close-ups per VLM request (one sheet): the numbers stay legible.
SHEET_TILES = 12
SHEET_COLUMNS = 4
TILE_PX = 192
TILE_PAD = 0.25  # context around the mask's box, as a share of its size
OUTLINE_RGB = (0, 255, 0)
# Brightness kept outside the mask in a close-up: neighbours stay visible as
# context but do not take the VLM's attention.
TILE_BACKGROUND = 0.25
NONE = "none"


@dataclass
class Candidate:
    mask: np.ndarray  # bool HxW
    score: float
    phrase: str


def touches_border(mask: np.ndarray, border_px: int) -> bool:
    """Whether the mask comes within `border_px` of the image edge (cut off)."""
    if border_px <= 0:
        return False
    x0, y0, x1, y1 = bbox_of(mask)
    height, width = mask.shape
    return (
        x0 < border_px
        or y0 < border_px
        or x1 > width - border_px
        or y1 > height - border_px
    )


def pool_candidates(
    results,
    shape,
    *,
    max_candidates=DEFAULT_MAX_CANDIDATES,
    min_pixels=MIN_MASK_PIXELS,
    max_share=MAX_MASK_SHARE,
    border_px=0,
):
    """Masks of all phrases, best score first, without duplicates.

    `results` holds (phrase, SAM outputs) pairs. Masks smaller than
    `min_pixels` or larger than `max_share` of the image are dropped, and with
    `border_px` > 0 those cut off by the image edge, whose position and size
    the depth cannot give.
    """
    height, width = shape
    pooled = []
    for phrase, outputs in results:
        masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
        scores = np.asarray(outputs["out_probs"], dtype=np.float32)
        if scores.ndim != 1 or masks.shape != (len(scores), height, width):
            raise ValueError("SAM 3.1 masks/scores have unexpected dimensions")
        for score, mask in zip(scores, masks):
            pixels = int(mask.sum())
            if (
                min_pixels <= pixels <= max_share * height * width
                and not touches_border(mask, border_px)
            ):
                pooled.append(Candidate(mask, float(score), phrase))
    pooled.sort(key=lambda c: -c.score)
    kept = []
    for candidate in pooled:
        if len(kept) == max_candidates:
            break
        if any(
            (candidate.mask & other.mask).sum() / (candidate.mask | other.mask).sum()
            > DUPLICATE_IOU
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _font(size):
    return ImageFont.load_default(size=size)


def _outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~erode(erode(mask))


def tile_box(mask: np.ndarray, shape) -> list[int]:
    """The mask's box with context around it, inside the image."""
    x0, y0, x1, y1 = bbox_of(mask)
    pad = TILE_PAD * max(x1 - x0, y1 - y0, 16)
    height, width = shape
    return [
        max(0, int(x0 - pad)),
        max(0, int(y0 - pad)),
        min(width, int(math.ceil(x1 + pad))),
        min(height, int(math.ceil(y1 + pad))),
    ]


def crop_sheet(rgb: np.ndarray, masks) -> np.ndarray:
    """Close-ups of the masks in numbered tiles (1..N), the mask outlined.

    Each tile shows the mask's box with some context, darkened outside the
    mask and scaled to fit TILE_PX, so that the VLM can read small objects that
    are a few pixels in the photo.
    """
    count = len(masks)
    columns = min(SHEET_COLUMNS, max(count, 1))
    rows = max(1, math.ceil(count / columns))
    sheet = Image.new("RGB", (columns * TILE_PX, rows * TILE_PX), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = _font(TILE_PX // 7)
    for index, mask in enumerate(masks):
        x0, y0, x1, y1 = tile_box(mask, rgb.shape[:2])
        inside = mask[y0:y1, x0:x1]
        crop = rgb[y0:y1, x0:x1].astype(np.float32)
        crop[~inside] *= TILE_BACKGROUND
        crop = crop.astype(np.uint8)
        crop[_outline(inside)] = OUTLINE_RGB
        image = Image.fromarray(crop)
        scale = TILE_PX / max(image.width, image.height)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        )
        left = (index % columns) * TILE_PX + (TILE_PX - image.width) // 2
        top = (index // columns) * TILE_PX + (TILE_PX - image.height) // 2
        sheet.paste(image, (left, top))
        tag_x, tag_y = (index % columns) * TILE_PX + 2, (index // columns) * TILE_PX
        text = str(index + 1)
        box = draw.textbbox((tag_x, tag_y), text, font=font)
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 2), fill=(0, 0, 0))
        draw.text((tag_x, tag_y), text, font=font, fill=(255, 255, 0))
    for column in range(1, columns):
        draw.line(
            [(column * TILE_PX, 0), (column * TILE_PX, rows * TILE_PX)],
            fill=(128, 128, 128),
        )
    for row in range(1, rows):
        draw.line(
            [(0, row * TILE_PX), (columns * TILE_PX, row * TILE_PX)],
            fill=(128, 128, 128),
        )
    return np.asarray(sheet)


LABEL_PROMPT = """You help a household robot recognize objects. Image 1 is the photo from its camera with {count} candidate regions, each drawn with its own colour and a number tag (1-{count}) at its top-left corner. Image 2 shows the same candidates as close-ups: tile N is candidate N, outlined in green, with everything outside it darkened.
Name what each candidate is with one label from this list (category and typical size in brackets):
{labels}
Rules:
- First look at the bright part of the close-up and note in a few words its shape, colours and any readable text or logo; then choose the label that matches them.
- Text printed on an object usually names it; prefer the label that the text names.
- A whole object that resembles a listed one gets the closest label, also another brand of the same kind of item or a fruit of another colour.
- Use "none" only for what is clearly not one of these objects: part of the furniture, the floor or a wall, part of the robot, a shadow or a reflection, a fragment of an object, or several objects together.
- The hatching in image 1 only shows where each candidate is; read details in image 2.
Reply with JSON only: {{"labels": {{"1": {{"seen": "<few words>", "label": "<label or none>"}}, ...}}}} with every number from 1 to {count}."""


def label_line(entry) -> str:
    """`- tomato soup (food, about 7 x 7 x 10 cm)`: category and size if given."""
    notes = [entry["category"]] if entry.get("category") else []
    size = entry.get("size")
    if size:
        notes.append("about " + " x ".join(f"{100 * v:.0f}" for v in size) + " cm")
    return f"- {entry['label']}" + (f" ({', '.join(notes)})" if notes else "")


def label_prompt(labels, count: int) -> str:
    """The VLM question for `count` candidates; `labels` holds dicts with
    `label` and optionally `category` and `size` (x, y, z in m)."""
    lines = "\n".join(label_line(entry) for entry in labels)
    return LABEL_PROMPT.format(count=count, labels=lines)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", str(text)).lower().split())


def parse_labels(text: str, count: int, labels) -> list[str | None]:
    """Each candidate's label from the VLM reply; None for none or unknown words.

    An answer is `{"seen": ..., "label": ...}` or the label alone. Accepts the
    labels' spelling with "_" or "-" for spaces and any case.
    Missing numbers are None. Raises ValueError when there is no JSON object.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"VLM reply has no JSON object: {text[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM reply is not valid JSON: {text[:200]!r}") from exc
    answers = data.get("labels") if isinstance(data, dict) else None
    if isinstance(answers, list):
        answers = {str(i + 1): value for i, value in enumerate(answers)}
    if not isinstance(answers, dict):
        raise ValueError("VLM reply must contain a labels object")
    known = {_normalize(label): label for label in labels}
    result = []
    for number in range(1, count + 1):
        answer = answers.get(str(number), answers.get(number))
        if isinstance(answer, dict):
            answer = answer.get("label")
        result.append(
            known.get(_normalize(answer)) if isinstance(answer, str) else None
        )
    return result


# --- Geometry (robot side) ------------------------------------------------

MIN_DEPTH_POINTS = 30
# Depth samples farther than this from the mask's median depth are background
# or foreground that leaked into the mask (m).
DEPTH_BAND_M = 0.3
# Share of the extreme points ignored at each end of an extent.
EXTENT_PERCENTILE = 2.0
MIN_SIZE_M = 0.01


@dataclass
class ObjectMeasurement:
    """An object's box in the target frame (z up), metres."""

    centre: np.ndarray
    size: np.ndarray  # axis-aligned extents x, y, z
    points: int  # depth samples used


def camera_points(mask, depth_m, fx, fy, cx, cy):
    """Depth samples of a mask as optical-frame points and the pixels eroded
    from each side, or (None, 0) if too few samples.

    The mask is eroded first when it stays large enough, because depth at an
    object's silhouette mixes the object and the background.
    """
    if mask.shape != depth_m.shape:
        raise ValueError("mask and depth must have the same shape")
    eroded = 0
    inner = erode(mask)
    if inner.sum() >= MIN_DEPTH_POINTS:
        mask, eroded = inner, 1
    ys, xs = np.nonzero(mask)
    z = depth_m[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if valid.sum() < MIN_DEPTH_POINTS:
        return None, 0
    ys, xs, z = ys[valid], xs[valid], z[valid]
    near = np.abs(z - np.median(z)) < DEPTH_BAND_M
    if near.sum() < MIN_DEPTH_POINTS:
        return None, 0
    ys, xs, z = ys[near], xs[near], z[near]
    return np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z]), eroded


def _extent(values):
    low, high = np.percentile(values, [EXTENT_PERCENTILE, 100 - EXTENT_PERCENTILE])
    return float(low), float(high)


def measure_object(
    mask,
    depth_m,
    fx,
    fy,
    cx,
    cy,
    rotation=None,
    translation=None,
    footprint=None,
) -> ObjectMeasurement | None:
    """The 3D box of the object in a mask, from the aligned depth.

    `rotation` / `translation` map the camera optical frame to the target
    frame, whose z axis must point up (map / odom / base_footprint). The camera
    sees only the near side of an object, so the hidden depth along the
    horizontal viewing direction is taken as the visible width, or as
    `footprint` (the smaller footprint side of the object's known size) when
    that is smaller; the centre is half that depth behind the nearest
    point. The size is the axis-aligned box of that footprint in the target
    frame. Without a rotation, the box of the visible points in the optical
    frame is returned (no hidden depth).
    """
    points, eroded = camera_points(mask, depth_m, fx, fy, cx, cy)
    if points is None:
        return None
    # The width the erosion took off both sides, at the object's distance.
    margin = 2 * eroded * float(np.median(points[:, 2])) / fx
    if rotation is None:
        lows, highs = zip(*(_extent(points[:, k]) for k in range(3)))
        lows, highs = np.array(lows), np.array(highs)
        size = highs - lows + np.array([margin, margin * fx / fy, 0.0])
        return ObjectMeasurement(
            (lows + highs) / 2, np.maximum(size, MIN_SIZE_M), len(points)
        )
    translation = np.zeros(3) if translation is None else np.asarray(translation)
    world = points @ np.asarray(rotation).T + translation
    relative = world[:, :2] - translation[:2]
    direction = np.median(relative, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:  # straight below the camera
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
    across = np.array([-direction[1], direction[0]])
    near, far = _extent(relative @ direction)
    left, right = _extent(relative @ across)
    width = max(right - left + margin, MIN_SIZE_M)
    hidden = width if footprint is None else min(width, footprint)
    depth = max(far - near, hidden, MIN_SIZE_M)
    along = near + depth / 2
    centre_xy = translation[:2] + along * direction + (left + right) / 2 * across
    bottom, top = _extent(world[:, 2])
    cos, sin = abs(direction[0]), abs(direction[1])
    size = np.array(
        [
            cos * depth + sin * width,
            sin * depth + cos * width,
            max(top - bottom + margin, MIN_SIZE_M),
        ]
    )
    centre = np.array([centre_xy[0], centre_xy[1], (bottom + top) / 2])
    return ObjectMeasurement(centre, size, len(points))


# Measured boxes larger than the label's size by this factor and margin are
# another object that the VLM named wrongly (the close-ups have no scale).
SIZE_FACTOR = 1.5
SIZE_MARGIN_M = 0.05


def size_fits(measured, listed) -> bool:
    """Whether a measured box can be an object of the listed size.

    Only too large is rejected: occlusion and the image edge make a measured
    box smaller, and an oblique view enlarges an axis-aligned box less than
    SIZE_FACTOR.
    """
    return float(np.max(measured)) <= SIZE_FACTOR * max(listed) + SIZE_MARGIN_M


def near_any(point, others, distance_m) -> bool:
    """Whether `point` is within `distance_m` of one of `others`."""
    return any(np.linalg.norm(np.asarray(point) - o) < distance_m for o in others)


# --- Tracking -----------------------------------------------------------------


@dataclass
class ObjectDetection:
    """One labelled object of one image, in the tracker's frame."""

    label: str
    centre: np.ndarray
    size: np.ndarray
    score: float
    bbox: list[int]


def merge_detections(detections, distance_m=0.05):
    """One detection per object: same label with centres closer than
    `distance_m` (a part found by another phrase) keeps the best score."""
    kept = []
    for detection in sorted(detections, key=lambda d: -d.score):
        if any(
            d.label == detection.label
            and np.linalg.norm(d.centre - detection.centre) < distance_m
            for d in kept
        ):
            continue
        kept.append(detection)
    return kept


# Sightings averaged into a track's position before it follows like a filter.
TRACK_AVERAGE = 5


@dataclass
class ObjectTrack:
    id: int
    centre: np.ndarray
    size: np.ndarray
    score: float
    bbox: list[int]
    last_seen: float
    hits: int = 0
    votes: dict = field(default_factory=dict)  # label -> summed score

    @property
    def label(self) -> str:
        return max(self.votes.items(), key=lambda item: (item[1], item[0]))[0]

    @property
    def agreement(self) -> float:
        """Share of the votes for the track's label."""
        total = sum(self.votes.values())
        return self.votes[self.label] / total if total > 0 else 0.0

    def add(self, detection: ObjectDetection, stamp: float) -> None:
        self.hits += 1
        weight = 1.0 / min(self.hits, TRACK_AVERAGE)
        self.centre = self.centre + weight * (detection.centre - self.centre)
        self.size = self.size + weight * (detection.size - self.size)
        self.score = detection.score
        self.bbox = list(detection.bbox)
        self.last_seen = stamp
        self.votes[detection.label] = (
            self.votes.get(detection.label, 0.0) + detection.score
        )


class ObjectTracker:
    """Nearest-neighbour association of labelled 3D detections to tracks.

    Objects stand still, so a track averages its sightings. A detection joins
    the nearest track within `gate_m` with the same label, or with another
    label within half the gate (the VLM named the same object differently);
    the label is the vote of the sightings, weighted by score. The frame
    should be fixed in the world (map / odom).
    """

    def __init__(self, gate_m=0.08, forget_after=120.0):
        self.gate_m = gate_m
        self.forget_after = forget_after
        self.tracks: dict[int, ObjectTrack] = {}
        self.next_id = 1

    def update(self, detections, stamp: float) -> list[ObjectTrack]:
        """Associate one image's detections; returns the tracks seen in it."""
        for track_id in [
            t.id
            for t in self.tracks.values()
            if stamp - t.last_seen > self.forget_after
        ]:
            del self.tracks[track_id]
        pairs = []
        for track in self.tracks.values():
            for index, detection in enumerate(detections):
                distance = float(np.linalg.norm(track.centre - detection.centre))
                same = detection.label == track.label
                if distance <= (self.gate_m if same else self.gate_m / 2):
                    pairs.append((0 if same else 1, distance, track.id, index))
        pairs.sort()
        used_tracks, used_detections, seen = set(), set(), []
        for _, _, track_id, index in pairs:
            if track_id in used_tracks or index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(index)
            track = self.tracks[track_id]
            track.add(detections[index], stamp)
            seen.append(track)
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            track = ObjectTrack(
                self.next_id,
                np.array(detection.centre, dtype=float),
                np.array(detection.size, dtype=float),
                detection.score,
                list(detection.bbox),
                stamp,
            )
            track.add(detection, stamp)
            self.next_id += 1
            self.tracks[track.id] = track
            seen.append(track)
        return sorted(seen, key=lambda t: t.id)

    def clear(self) -> None:
        self.tracks.clear()


# --- Debug image ----------------------------------------------------------------


def annotate(rgb: np.ndarray, items) -> np.ndarray:
    """Outline each (mask, text, labelled) item; unlabelled ones in grey."""
    image = rgb.copy()
    texts = []
    for mask, text, labelled in items:
        colour = (0, 255, 0) if labelled else (150, 150, 150)
        image[_outline(mask)] = colour
        x0, y0, _, _ = bbox_of(mask)
        texts.append((x0, y0, text, colour))
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    font = _font(max(12, rgb.shape[0] // 32))
    for x, y, text, colour in texts:
        box = draw.textbbox((x, y), text, font=font)
        top = max(0, y - (box[3] - box[1]) - 4)
        box = draw.textbbox((x, top), text, font=font)
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0))
        draw.text((x, top), text, font=font, fill=colour)
    return np.asarray(canvas)
