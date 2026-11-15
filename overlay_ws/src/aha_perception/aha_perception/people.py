"""People from SAM 3.1 masks: 3D position, tracking and naming by the VLM.

ROS-independent so that the node, the DGX worker and the unit tests share it.
Masks cross the SSH channel cropped to their bounding box and bit-packed.
Positions are computed on the robot side from the aligned depth image, so
depth never leaves the robot.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
import json
import math
import re

import numpy as np
from PIL import Image


# Depth samples farther than this from the mask's median depth are background
# or foreground that leaked into the mask (m).
DEPTH_BAND_M = 0.4
# Fraction of the person's image height, from the top, averaged for the head.
HEAD_FRACTION = 0.15
MIN_DEPTH_POINTS = 50
# Background brightness kept in the crop sent to the VLM.
DESCRIBE_BACKGROUND = 0.35
DESCRIBE_PAD = 0.1
# SAM 3.1 sometimes returns people side by side as one mask. A mask is split at
# this many empty image columns; parts below the share of its area are dropped.
SPLIT_GAP = 5
SPLIT_MIN_SHARE = 0.05
# Parts of a split mask closer than this on the floor plane (m) are one person
# behind something thin, such as a door frame: less than a person's width.
SAME_PERSON_M = 0.4
# Remembered people per match request: with the person asked about, the
# Qwen3.8 server's limit of 4 images.
MAX_REFERENCES = 3
# Image time (s) before a track the VLM did not recognize is asked again.
MATCH_RETRY = 5.0
# Keypoints kept per track (s of image time), for gestures that are motion.
KEYPOINT_HISTORY = 3.0
# Recent crops per track for the VLM's attribute question, at least this far
# apart (s of image time), so that it can see a hand move.
RECENT_CROPS = 3
RECENT_CROP_INTERVAL = 0.4

DESCRIBE_PROMPT = """The photo shows one person; the background is darkened. List the visible features that would let someone pick this person out of a group: clothing with its colour (for example "red t-shirt", "blue jeans", "black jacket"), glasses, hat, bag, hair (for example "long hair", "short black hair", "bald"), beard.
Only list what you can clearly see in this photo. Do not guess hidden parts, and do not mention age, gender, ethnicity or emotions.
Reply with JSON only: {"features": ["...", "..."]} with at most 6 short English phrases, most distinctive first."""

MATCH_PROMPT = """{earlier} a person met earlier; the last photo, photo {last}, shows a person now. Backgrounds are darkened.
Is the person in the last photo the same person as in {choices}? Compare clothing and its colours, hair, glasses, hat and bag; pose, distance and lighting may differ. Different people can wear similar clothes, so choose a photo only when the features clearly agree.
The last photo is not a choice. Reply with JSON only: {{"photo": {numbers}}}, or {{"photo": null}} if no photo clearly shows the same person."""


# --- Transport -------------------------------------------------------------


def encode_image(rgb: np.ndarray, quality: int = 90) -> str:
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(data: str) -> np.ndarray:
    image = Image.open(BytesIO(base64.b64decode(data, validate=True)))
    return np.asarray(image.convert("RGB"))


def bbox_of(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def split_sideways(mask: np.ndarray) -> list[np.ndarray]:
    """The mask's parts separated by empty columns, left to right."""
    columns = np.flatnonzero(mask.any(axis=0))
    if columns.size == 0:
        return []
    parts = []
    for run in np.split(columns, np.flatnonzero(np.diff(columns) > SPLIT_GAP) + 1):
        part = np.zeros_like(mask)
        part[:, run[0] : run[-1] + 1] = mask[:, run[0] : run[-1] + 1]
        parts.append(part)
    area = mask.sum()
    return [part for part in parts if part.sum() >= SPLIT_MIN_SHARE * area]


def encode_mask(mask: np.ndarray) -> dict:
    x0, y0, x1, y1 = bbox_of(mask)
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=bool)
    return {
        "bbox": [x0, y0, x1, y1],
        "mask": base64.b64encode(np.packbits(crop).tobytes()).decode("ascii"),
    }


def decode_mask(encoded: dict, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = (int(v) for v in encoded["bbox"])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"mask bbox {encoded['bbox']} is outside {width}x{height}")
    bits = np.frombuffer(base64.b64decode(encoded["mask"], validate=True), np.uint8)
    count = (x1 - x0) * (y1 - y0)
    if len(bits) != (count + 7) // 8:
        raise ValueError("mask size does not match its bbox")
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = np.unpackbits(bits, count=count).reshape(y1 - y0, x1 - x0)
    return mask


# --- Geometry ----------------------------------------------------------------


@dataclass
class Measurement:
    """Points in the camera optical frame (x right, y down, z forward), metres."""

    centre: np.ndarray
    head: np.ndarray


def measure(mask, depth_m, fx, fy, cx, cy) -> Measurement | None:
    """Body centre and head point from the depth inside a mask.

    Returns None when too few valid depth pixels remain, e.g. a person cut off
    by the image edge or beyond the depth range.
    """
    if mask.shape != depth_m.shape:
        raise ValueError("mask and depth must have the same shape")
    ys, xs = np.nonzero(mask)
    z = depth_m[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if valid.sum() < MIN_DEPTH_POINTS:
        return None
    ys, xs, z = ys[valid], xs[valid], z[valid]
    near = np.abs(z - np.median(z)) < DEPTH_BAND_M
    if near.sum() < MIN_DEPTH_POINTS:
        return None
    ys, xs, z = ys[near], xs[near], z[near]
    points = np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z])
    top, bottom = ys.min(), ys.max()
    upper = ys <= top + max(1, HEAD_FRACTION * (bottom - top))
    return Measurement(
        centre=np.median(points, axis=0), head=np.median(points[upper], axis=0)
    )


def measure_people(
    mask, depth_m, fx, fy, cx, cy, *, rotation=None
) -> list[tuple[np.ndarray, Measurement]]:
    """Each person in one SAM mask with their own mask and position, left to right.

    The mask's parts stay separate people only when depth puts them a person's
    width apart. Height is ignored: an occluder can leave different heights visible.
    rotation maps the optical frame to a frame with z up. Without it, optical x/z
    approximate the floor plane of a level camera.
    """
    people = []
    for part in split_sideways(mask):
        measured = measure(part, depth_m, fx, fy, cx, cy)
        if measured is None:
            continue
        if people:
            offset = people[-1][1].centre - measured.centre
            horizontal = (
                (rotation @ offset)[:2] if rotation is not None else offset[[0, 2]]
            )
            if np.linalg.norm(horizontal) < SAME_PERSON_M:
                joined = people[-1][0] | part
                measured = measure(joined, depth_m, fx, fy, cx, cy)
                if measured is not None:
                    people[-1] = (joined, measured)
                continue
        people.append((part, measured))
    return people


def rotation_matrix(x, y, z, w) -> np.ndarray:
    """Rotation of a unit quaternion (ROS order x, y, z, w)."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# --- Tracking and names ------------------------------------------------------


@dataclass
class Detection:
    """One person in the tracker's frame."""

    centre: np.ndarray
    head: np.ndarray
    score: float
    bbox: list[int]
    crop: np.ndarray | None = None  # RGB for the VLM, background darkened
    # (17, 3) COCO body keypoints in the tracker's frame and their scores
    # (0: not seen); None without keypoints.
    keypoints: np.ndarray | None = None
    keypoint_scores: np.ndarray | None = None
    camera: np.ndarray | None = None  # camera position in the tracker's frame


@dataclass
class Track:
    id: int
    centre: np.ndarray
    head: np.ndarray
    score: float
    bbox: list[int]
    last_seen: float
    crop: np.ndarray | None = None
    crop_area: int = 0
    crop_stamp: float = -math.inf
    name: str = ""  # kept until the track is lost
    match_after: float = -math.inf  # image time of the next VLM match
    keypoints: np.ndarray | None = None  # of the last sighting
    keypoint_scores: np.ndarray | None = None
    # (stamp, keypoints, scores, camera) of recent sightings with keypoints,
    # oldest first.
    history: deque = field(default_factory=deque)
    # (stamp, crop) of recent sightings, oldest first.
    recent_crops: deque = field(default_factory=lambda: deque(maxlen=RECENT_CROPS))

    @classmethod
    def start(cls, track_id: int, detection: Detection, stamp: float) -> Track:
        track = cls(track_id, detection.centre, detection.head, 0.0, [], stamp)
        track.add(detection, stamp, smoothing=0.0)
        return track

    def add(self, detection: Detection, stamp: float, smoothing: float) -> None:
        self.centre = smoothing * self.centre + (1 - smoothing) * detection.centre
        self.head = smoothing * self.head + (1 - smoothing) * detection.head
        self.score = detection.score
        self.bbox = list(detection.bbox)
        self.last_seen = stamp
        if detection.crop is not None:
            # Keep the largest recent view: more pixels for the VLM.
            area = int(
                (detection.bbox[2] - detection.bbox[0])
                * (detection.bbox[3] - detection.bbox[1])
            )
            if area >= 0.7 * self.crop_area or stamp - self.crop_stamp > 5.0:
                self.crop, self.crop_area, self.crop_stamp = detection.crop, area, stamp
            if (
                not self.recent_crops
                or stamp - self.recent_crops[-1][0] >= RECENT_CROP_INTERVAL
            ):
                self.recent_crops.append((stamp, detection.crop))
        self.keypoints = detection.keypoints
        self.keypoint_scores = detection.keypoint_scores
        if detection.keypoints is not None:
            self.history.append(
                (
                    stamp,
                    detection.keypoints,
                    detection.keypoint_scores,
                    detection.camera,
                )
            )
        while self.history and stamp - self.history[0][0] > KEYPOINT_HISTORY:
            self.history.popleft()


class PersonTracker:
    """Nearest-neighbour association of 3D detections to tracks.

    The frame should be fixed in the world (map / odom) so that a still person
    keeps the same position while the robot moves.
    """

    def __init__(self, gate_m=0.8, forget_after=3.0, smoothing=0.5):
        self.gate_m = gate_m
        self.forget_after = forget_after
        self.smoothing = smoothing
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.stamp = -math.inf  # of the last update; tracks seen then are visible

    def update(self, detections: list[Detection], stamp: float) -> list[Track]:
        """Associate one image's detections; returns the tracks seen in it."""
        self.stamp = stamp
        for track_id in [
            t.id
            for t in self.tracks.values()
            if stamp - t.last_seen > self.forget_after
        ]:
            del self.tracks[track_id]
        pairs = sorted(
            (float(np.linalg.norm(track.centre - d.centre)), track.id, index)
            for track in self.tracks.values()
            for index, d in enumerate(detections)
        )
        used_tracks, used_detections, seen = set(), set(), []
        for distance, track_id, index in pairs:
            if distance > self.gate_m:
                break
            if track_id in used_tracks or index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(index)
            track = self.tracks[track_id]
            track.add(detections[index], stamp, self.smoothing)
            seen.append(track)
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            track = Track.start(self.next_id, detection, stamp)
            self.next_id += 1
            self.tracks[track.id] = track
            seen.append(track)
        return sorted(seen, key=lambda t: t.id)

    def clear(self) -> None:
        self.tracks.clear()


class Gallery:
    """Remembered people by name, as crops; each name is on at most one track.

    remember() names a track directly. Other tracks get a name from the VLM,
    which compares a visible unnamed track with the remembered people whose
    names no track holds, so guests who swapped seats are told apart by
    appearance rather than by where they sit.
    """

    def __init__(self):
        self.crops: dict[str, np.ndarray] = {}

    def remember(self, name: str, track: Track, tracks=()) -> None:
        """Store the track's crop as `name`; other tracks lose that name."""
        if track.crop is None:
            raise ValueError(f"person {track.id} has no image yet")
        self.crops[name] = track.crop
        for other in tracks:
            if other.name == name:
                other.name = ""
        track.name = name

    def next_match(self, tracker: PersonTracker):
        """(track ID, crop, [(name, crop)]) to ask the VLM about, or None.

        The track is the visible unnamed one asked least recently; it is not
        asked again for MATCH_RETRY s of image time.
        """
        tracks = list(tracker.tracks.values())
        held = {t.name for t in tracks if t.name}
        references = [(n, c) for n, c in self.crops.items() if n not in held]
        waiting = [
            t
            for t in tracks
            if not t.name
            and t.crop is not None
            and t.last_seen == tracker.stamp
            and t.match_after <= tracker.stamp
        ]
        if not references or not waiting:
            return None
        track = min(waiting, key=lambda t: (t.match_after, t.id))
        track.match_after = tracker.stamp + MATCH_RETRY
        return track.id, track.crop, references

    def give(self, name: str, track_id: int, tracker: PersonTracker) -> bool:
        """Name a track after a match, unless it or the name was taken since."""
        track = tracker.tracks.get(track_id)
        if track is None or track.name or name not in self.crops:
            return False
        if any(t.name == name for t in tracker.tracks.values()):
            return False
        track.name = name
        return True


# --- VLM ---------------------------------------------------------------------


def describe_crop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The person's bounding box with the background darkened, for the VLM."""
    x0, y0, x1, y1 = bbox_of(mask)
    pad = int(DESCRIBE_PAD * max(x1 - x0, y1 - y0))
    height, width = mask.shape
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    inside = mask[y0:y1, x0:x1, None]
    return np.where(inside, crop, crop * DESCRIBE_BACKGROUND).astype(np.uint8)


def json_object(text: str) -> dict:
    """The JSON object in a VLM reply, which may wrap it in other text."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"VLM reply has no JSON object: {text[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM reply is not valid JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"VLM reply is not a JSON object: {text[:200]!r}")
    return data


def parse_features(text: str, limit: int = 6) -> list[str]:
    """Feature phrases from the VLM's JSON reply, cleaned and deduplicated."""
    raw = json_object(text).get("features")
    if not isinstance(raw, list):
        raise ValueError("VLM reply must contain a features list")
    features = []
    for item in raw:
        if not isinstance(item, str):
            continue
        phrase = " ".join(item.lower().split()).strip(" .")
        if re.fullmatch(r"[a-z][a-z0-9 '/-]{0,40}", phrase) and phrase not in features:
            features.append(phrase)
    return features[:limit]


def match_prompt(count: int) -> str:
    """Prompt for `count` remembered people followed by the person asked about."""
    if not 1 <= count <= MAX_REFERENCES:
        raise ValueError(f"a match needs 1 to {MAX_REFERENCES} remembered people")
    if count == 1:
        earlier, choices, numbers = "Photo 1 shows", "photo 1", "1"
    else:
        earlier = f"Photos 1 to {count} each show"
        choices = f"one of photos 1 to {count}"
        numbers = f"<a number from 1 to {count}>"
    return MATCH_PROMPT.format(
        earlier=earlier, last=count + 1, choices=choices, numbers=numbers
    )


def parse_match(text: str, count: int) -> int | None:
    """Index of the remembered person the VLM chose, or None."""
    photo = json_object(text).get("photo")
    if photo is None:
        return None
    if isinstance(photo, bool) or not isinstance(photo, int) or not 1 <= photo <= count:
        raise ValueError(f"VLM chose an invalid photo: {photo!r}")
    return photo - 1
