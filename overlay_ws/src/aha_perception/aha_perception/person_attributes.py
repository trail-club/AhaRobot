"""Posture, gesture and clothing of a person: body keypoints, depth and the VLM.

ROS-independent so that the node, the DGX worker and the unit tests share it.
Keypoints are the 17 COCO body points from ViTPose on the DGX. They are lifted
to 3D with the aligned depth on the robot and transformed to the tracker's
frame, which must have z up with the floor at `floor_z` for the rules below.

The rules decide only clear cases. Waving needs several images, because it is
motion; left and right come from the keypoint labels, which name the person's
own sides. The VLM answers when the rules cannot (missing keypoints, unclear
geometry) and names the upper-body clothing, which the rules do not see.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

import numpy as np

from .people import json_object

KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
INDEX = {name: i for i, name in enumerate(KEYPOINTS)}
SIDES = ("left", "right")


def swap_side(name):
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


POSTURES = ("standing", "sitting", "lying")
GESTURES = (
    "waving",
    "raising_left_arm",
    "raising_right_arm",
    "raising_both_arms",
    "pointing_left",
    "pointing_right",
    "none",
)
# Types and colours of the GPSR command generator (RoboCupAtHome/CommandGenerator),
# with colours it does not use so that the VLM need not force a wrong one.
GARMENT_TYPES = ("t shirt", "shirt", "blouse", "sweater", "coat", "jacket")
COLOURS = (
    "blue",
    "yellow",
    "black",
    "white",
    "red",
    "orange",
    "gray",
    "green",
    "purple",
    "pink",
    "brown",
    "beige",
)
OTHER_GARMENT = "top"  # an upper-body garment of no listed type
GARMENT_SYNONYMS = {"tshirt": "t shirt", "tee": "t shirt", "tee shirt": "t shirt"}

# ViTPose heatmap score below which a keypoint counts as not seen.
MIN_SCORE = 0.35
# Pixel radius around a keypoint whose depth is used, and the depth band (m)
# around the body's median depth that still belongs to the person.
KEYPOINT_RADIUS_PX = 4
KEYPOINT_DEPTH_BAND_M = 0.6
# Keypoints this close to the image edge are guesses for a cut-off limb.
BORDER_PX = 8

# Posture (m and degrees; heights above the floor).
LYING_TORSO_TILT = 60.0  # torso from vertical
UPRIGHT_TORSO_TILT = 40.0
SITTING_THIGH_TILT = 55.0  # thigh from vertical
STANDING_THIGH_TILT = 30.0
STANDING_HEAD_Z = 1.45  # top of the head when knees are hidden
SITTING_HEAD_Z = 1.25
LYING_HEAD_Z = 0.8
SITTING_HIP_Z = 0.7  # hips when knees are hidden
STANDING_HIP_Z = 0.78

# Gesture (m).
RAISED_ABOVE_SHOULDER = 0.15  # wrist above the shoulder
POINTING_REACH = 0.42  # wrist from the shoulder, horizontally
POINTING_LEVEL = 0.25  # wrist within this of the shoulder height
POINTING_LATERAL = 0.25  # sideways part of the reach, from the shoulders' middle
WAVE_SWING = 0.1  # range of the wrist sideways around the elbow
WAVE_HAND_BELOW_SHOULDER = 0.1  # a waving wrist is at least this high
WAVE_FOREARM_RISE = 0.08  # and this far above the elbow (a level arm points)
WAVE_WINDOW = 3.0  # s of image time for the motion
MIN_MOTION_SAMPLES = 4  # images in the window to tell waving from a still hand
MIN_MOTION_SPAN = 1.5  # s between the first and the last of them
VOTE_SHARE = 0.6  # of the window's images that must agree on a still gesture
VOTE_WINDOW = 1.5  # s of image time for the vote: views before the head turned fade out
# Mirrored labels: the shoulders must be this far apart across the camera's
# view (m), and the face seen (all of nose and eyes at least FACE_SEEN).
MIRROR_MIN_ACROSS = 0.15
FACE_SEEN = 0.5


# --- 3D keypoints --------------------------------------------------------------


def lift_keypoints(keypoints, mask, depth_m, fx, fy, cx, cy):
    """(17, 3) keypoints [u, v, score] -> (17, 3) optical-frame points and scores.

    A keypoint takes the median depth of the person's mask pixels around it,
    else of any depth near the body around it (an arm can leave the mask), else
    the body's median depth. Scores below MIN_SCORE, and keypoints within
    BORDER_PX of the image edge, become 0.
    """
    keypoints = np.asarray(keypoints, dtype=np.float64)
    points = np.zeros((len(KEYPOINTS), 3))
    scores = np.where(keypoints[:, 2] >= MIN_SCORE, keypoints[:, 2], 0.0)
    height, width = depth_m.shape
    inside = (
        (keypoints[:, 0] >= BORDER_PX)
        & (keypoints[:, 0] < width - BORDER_PX)
        & (keypoints[:, 1] >= BORDER_PX)
        & (keypoints[:, 1] < height - BORDER_PX)
    )
    scores = np.where(inside, scores, 0.0)
    body = depth_m[mask]
    body = body[np.isfinite(body) & (body > 0)]
    if body.size == 0:
        return points, np.zeros(len(KEYPOINTS))
    body_z = float(np.median(body))
    r = KEYPOINT_RADIUS_PX
    for i, (u, v, _) in enumerate(keypoints):
        if scores[i] == 0:
            continue
        x, y = int(round(u)), int(round(v))
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            scores[i] = 0.0
            continue
        window = depth_m[y0:y1, x0:x1]
        near = np.isfinite(window) & (np.abs(window - body_z) < KEYPOINT_DEPTH_BAND_M)
        inside = near & mask[y0:y1, x0:x1]
        if inside.any():
            z = float(np.median(window[inside]))
        elif near.any():
            z = float(np.median(window[near]))
        else:
            z = body_z
        points[i] = ((u - cx) * z / fx, (v - cy) * z / fy, z)
    return points, scores


def keypoints_in_mask(candidates, mask):
    """The candidate (17, 3) keypoints with the most seen points inside the mask."""
    best, best_count = None, 0
    height, width = mask.shape
    for keypoints in candidates:
        keypoints = np.asarray(keypoints, dtype=np.float64)
        count = 0
        for u, v, score in keypoints:
            x, y = int(round(u)), int(round(v))
            if score >= MIN_SCORE and 0 <= x < width and 0 <= y < height and mask[y, x]:
                count += 1
        if count > best_count:
            best, best_count = keypoints, count
    return best


# --- Rules -----------------------------------------------------------------------


@dataclass
class Sample:
    """Keypoints of one sighting in a z-up frame, and where the camera was."""

    stamp: float
    points: np.ndarray  # (17, 3)
    scores: np.ndarray  # (17,)
    camera: np.ndarray | None = None  # (3,) camera position in the same frame

    def get(self, name):
        i = INDEX[name]
        return self.points[i] if self.scores[i] > 0 else None

    def middle(self, *names):
        seen = [p for p in (self.get(n) for n in names) if p is not None]
        return np.mean(seen, axis=0) if seen else None

    def camera_right(self):
        """Horizontal unit vector to the camera's right as seen at the person, or None."""
        body = self.middle("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if self.camera is None or body is None:
            return None
        ray = (body - self.camera)[:2]
        norm = float(np.linalg.norm(ray))
        if norm < 1e-6:
            return None
        # ray x up, with up = +z
        return np.array([ray[1], -ray[0]]) / norm

    def unmirrored(self):
        """This sample with left and right swapped when the labels are a mirror image.

        A person facing the camera (eyes and nose seen) has their left side on
        the camera's right. ViTPose can label a whole body mirrored; those
        labels are swapped back. Without a clearly seen face nothing is changed:
        the face may be turned away or only out of the image. Side views, where
        the shoulders are nearly in line with the camera, are kept as well.
        """
        right = self.camera_right()
        left_shoulder, right_shoulder = (
            self.get("left_shoulder"),
            self.get("right_shoulder"),
        )
        if right is None or left_shoulder is None or right_shoulder is None:
            return self
        across = float(np.dot((left_shoulder - right_shoulder)[:2], right))
        if abs(across) < MIRROR_MIN_ACROSS:
            return self
        face = [self.scores[INDEX[n]] for n in ("nose", "left_eye", "right_eye")]
        if min(face) < FACE_SEEN or across > 0:
            return self
        order = [INDEX[swap_side(name)] for name in KEYPOINTS]
        return Sample(self.stamp, self.points[order], self.scores[order], self.camera)


@dataclass
class Label:
    value: str
    confident: bool
    reason: str = ""


def tilt(vector):
    """Angle of a vector from vertical (degrees)."""
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return None
    return math.degrees(math.acos(min(1.0, abs(float(vector[2])) / length)))


def classify_posture(sample: Sample | None, head_z: float | None) -> Label | None:
    """Standing / sitting / lying from keypoints and the head height above the floor."""
    shoulders = hips = None
    torso = None
    if sample is not None:
        shoulders = sample.middle("left_shoulder", "right_shoulder")
        hips = sample.middle("left_hip", "right_hip")
        if shoulders is not None and hips is not None:
            torso = tilt(shoulders - hips)
    if torso is not None and torso > LYING_TORSO_TILT:
        return Label("lying", True, f"torso {torso:.0f} deg from vertical")
    if torso is not None and torso < UPRIGHT_TORSO_TILT:
        thighs = []
        for side in SIDES:
            hip, knee = sample.get(f"{side}_hip"), sample.get(f"{side}_knee")
            if hip is not None and knee is not None:
                angle = tilt(knee - hip)
                if angle is not None:
                    thighs.append(angle)
        if thighs:
            thigh = float(np.mean(thighs))
            if thigh > SITTING_THIGH_TILT:
                return Label("sitting", True, f"thighs {thigh:.0f} deg from vertical")
            if thigh < STANDING_THIGH_TILT:
                return Label("standing", True, f"thighs {thigh:.0f} deg from vertical")
        hip_z = float(hips[2])
        if hip_z < SITTING_HIP_Z:
            return Label("sitting", True, f"upright, hips at {hip_z:.2f} m")
        if hip_z > STANDING_HIP_Z:
            return Label("standing", True, f"upright, hips at {hip_z:.2f} m")
    if head_z is None:
        return None
    if head_z >= STANDING_HEAD_Z and (torso is None or torso < UPRIGHT_TORSO_TILT):
        return Label("standing", torso is not None, f"head at {head_z:.2f} m")
    if head_z <= LYING_HEAD_Z:
        # Even sitting on the floor puts the top of the head higher.
        return Label(
            "lying",
            torso is None or torso > UPRIGHT_TORSO_TILT,
            f"head at {head_z:.2f} m",
        )
    if head_z <= SITTING_HEAD_Z and torso is not None and torso < UPRIGHT_TORSO_TILT:
        return Label("sitting", False, f"upright, head at {head_z:.2f} m")
    return None


def left_axis(sample: Sample):
    """Horizontal unit vector to the person's left, from the shoulders (or hips)."""
    for pair in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")):
        left, right = sample.get(pair[0]), sample.get(pair[1])
        if left is not None and right is not None:
            axis = (left - right)[:2]
            norm = float(np.linalg.norm(axis))
            if norm > 0.05:
                return axis / norm
    return None


def arm_pose(sample: Sample, side: str):
    """'raised', 'pointing_left' / 'pointing_right', 'down' or None (not seen)."""
    shoulder, wrist = sample.get(f"{side}_shoulder"), sample.get(f"{side}_wrist")
    if shoulder is None or wrist is None:
        return None
    if wrist[2] - shoulder[2] > RAISED_ABOVE_SHOULDER:
        return "raised"
    reach = float(np.linalg.norm((wrist - shoulder)[:2]))
    axis = left_axis(sample)
    if reach > POINTING_REACH and abs(wrist[2] - shoulder[2]) < POINTING_LEVEL:
        if axis is None:
            return None
        middle = sample.middle("left_shoulder", "right_shoulder")
        lateral = float(np.dot((wrist - middle)[:2], axis))
        if lateral > POINTING_LATERAL:
            return "pointing_left"
        if lateral < -POINTING_LATERAL:
            return "pointing_right"
    return "down"


def hand_up(sample: Sample, side: str) -> bool:
    """The forearm points up with the wrist about shoulder height or higher."""
    shoulder = sample.get(f"{side}_shoulder")
    elbow, wrist = sample.get(f"{side}_elbow"), sample.get(f"{side}_wrist")
    if shoulder is None or elbow is None or wrist is None:
        return False
    return (
        wrist[2] > elbow[2] + WAVE_FOREARM_RISE
        and wrist[2] > shoulder[2] - WAVE_HAND_BELOW_SHOULDER
    )


def wave_swing(samples, side):
    """Range (m) of one raised wrist sideways around its elbow over the samples.

    Sideways is across the camera's view when the camera is known. Samples
    should be unmirrored first, so that one hand keeps its label.
    """
    offsets = []
    for sample in samples:
        if not hand_up(sample, side):
            continue
        elbow, wrist = sample.get(f"{side}_elbow"), sample.get(f"{side}_wrist")
        axis = sample.camera_right()
        if axis is None:
            axis = left_axis(sample)
        if axis is None:
            continue
        offsets.append(float(np.dot((wrist - elbow)[:2], axis)))
    return (max(offsets) - min(offsets)) if len(offsets) >= 2 else 0.0


def upright(sample: Sample) -> bool:
    """False when the torso is seen tilted; arms are only judged on upright bodies."""
    shoulders = sample.middle("left_shoulder", "right_shoulder")
    hips = sample.middle("left_hip", "right_hip")
    if shoulders is None or hips is None:
        return True
    angle = tilt(shoulders - hips)
    return angle is None or angle <= UPRIGHT_TORSO_TILT


def still_gesture(sample: Sample):
    """The gesture of one image without motion, or None if no arm is seen."""
    arms = {side: arm_pose(sample, side) for side in SIDES}
    if all(pose is None for pose in arms.values()):
        return None
    raised = [side for side in SIDES if arms[side] == "raised"]
    if len(raised) == 2:
        return "raising_both_arms"
    if raised:
        return f"raising_{raised[0]}_arm"
    pointing = [p for p in arms.values() if p and p.startswith("pointing")]
    if pointing:
        return pointing[0]
    return "none" if None not in arms.values() else "none?"


def classify_gesture(history) -> Label | None:
    """Gesture from recent samples (oldest first); the last is the current view.

    A still gesture is the majority over the window, so that one image with
    swapped left / right labels does not flip it.
    """
    if not history:
        return None
    history = [sample.unmirrored() for sample in history]
    latest = history[-1]
    if not upright(latest):
        # A lying person's arms lie level beside the body and would look like pointing.
        return None
    window = [
        s for s in history if latest.stamp - s.stamp <= WAVE_WINDOW and upright(s)
    ]
    for side in SIDES:
        swing = wave_swing(window, side)
        if swing > WAVE_SWING:
            return Label("waving", True, f"{side} wrist swings {swing:.2f} m")
    recent = [s for s in window if latest.stamp - s.stamp <= VOTE_WINDOW]
    votes = [g for g in (still_gesture(s) for s in recent) if g is not None]
    if still_gesture(latest) is None or not votes:
        return None
    # "none?" (an arm not seen) only counts for "none".
    counted = Counter(g.rstrip("?") for g in votes)
    value, count = counted.most_common(1)[0]
    share = count / len(votes)
    agreed = share >= VOTE_SHARE
    span = window[-1].stamp - window[0].stamp
    if value.startswith("raising"):
        swing = max(wave_swing(window, side) for side in SIDES)
        moving = len(window) >= MIN_MOTION_SAMPLES and span >= MIN_MOTION_SPAN
        # A hand held still for long enough is not waving.
        return Label(
            value,
            moving and agreed,
            f"wrist above the shoulder in {count}/{len(votes)} images, swing "
            f"{swing:.2f} m over {span:.1f} s",
        )
    if value.startswith("pointing"):
        return Label(
            value, agreed, f"arm stretched out sideways in {count}/{len(votes)} images"
        )
    seen = sum(1 for g in votes if g == "none")
    return Label(
        "none",
        agreed and seen == count,
        f"arms down in {count}/{len(votes)} images"
        + ("" if seen == count else ", an arm not always seen"),
    )


# --- VLM ---------------------------------------------------------------------------

ATTRIBUTES_PROMPT = """{intro} The background is darkened.
Answer three questions about this person.
1. posture: "standing", "sitting" or "lying".
2. gesture: "waving" (a raised hand moving from side to side{motion}), "raising_left_arm", "raising_right_arm", "raising_both_arms" (hand held up above the shoulder), "pointing_left", "pointing_right" (an arm stretched out sideways), or "none". Left and right are the person's own sides: a person facing the camera holds their left arm on the right side of the photo.
3. upper_garments: each garment visible on the upper body (torso and arms), outermost first, with "type" one of {types} or "other", and "colour" one of {colours}. Do not list trousers, shorts, skirts or shoes. If the upper body is hidden or out of the photo, give an empty list.
Reply with JSON only: {{"posture": "...", "gesture": "...", "upper_garments": [{{"type": "...", "colour": "..."}}]}}"""


def attributes_prompt(count: int, interval: float) -> str:
    if count <= 1:
        intro = "The photo shows one person."
        motion = "; a single photo cannot show this, so do not answer waving"
    else:
        intro = (
            f"Photos 1 to {count} show the same person, oldest first, "
            f"about {interval:.1f} s apart."
        )
        motion = " between the photos"
    return ATTRIBUTES_PROMPT.format(
        intro=intro,
        motion=motion,
        types=", ".join(f'"{t}"' for t in GARMENT_TYPES),
        colours=", ".join(COLOURS),
    )


def _word(value):
    if not isinstance(value, str):
        return None
    return " ".join(re.sub(r"[_\-]", " ", value.lower()).split())


def normalize_garment_type(value):
    """A GARMENT_TYPES entry for spellings such as 't-shirt' or 'tshirt', else None."""
    word = _word(value)
    if word is None:
        return None
    for candidate in (word, word[:-1] if word.endswith("s") else None):
        candidate = GARMENT_SYNONYMS.get(candidate, candidate)
        if candidate in GARMENT_TYPES:
            return candidate
    return None


def normalize_colour(value):
    word = _word(value)
    if word is None:
        return None
    word = {"grey": "gray"}.get(word, word)
    return word if word in COLOURS else None


@dataclass
class VlmAttributes:
    posture: str | None
    gesture: str | None
    garments: list[str]


def parse_attributes(text: str) -> VlmAttributes:
    data = json_object(text)
    posture = _word(data.get("posture"))
    gesture = _word(data.get("gesture"))
    gesture = gesture.replace(" ", "_") if gesture else None
    garments = []
    raw = data.get("upper_garments")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            colour = normalize_colour(item.get("colour", item.get("color")))
            kind = normalize_garment_type(item.get("type")) or OTHER_GARMENT
            if colour is None:
                continue
            phrase = f"{colour} {kind}"
            if phrase not in garments:
                garments.append(phrase)
    return VlmAttributes(
        posture=posture if posture in POSTURES else None,
        gesture=gesture if gesture in GESTURES else None,
        garments=garments,
    )


# --- Decision --------------------------------------------------------------------


def decide(rule: Label | None, vlm_value: str | None, rule_source: str):
    """(value, source): a confident rule, else the VLM, else a weak rule."""
    if rule is not None and rule.confident:
        return rule.value, rule_source
    if vlm_value:
        return vlm_value, "vlm"
    if rule is not None:
        return rule.value, rule_source
    return "", ""


def decide_gesture(rule: Label | None, vlm_value: str | None):
    """(value, source) like decide, but the VLM never changes the side of an arm.

    An unsure raising_* only lacks the motion that tells it from waving, and an
    unsure pointing_* only lacks agreement over the images; the keypoints did
    see which arm it is, while VLMs often confuse the person's left and right.
    So the VLM may only turn a raised arm into waving.
    """
    if rule is not None and not rule.confident:
        if rule.value.startswith("raising"):
            if vlm_value == "waving":
                return "waving", "vlm"
            return rule.value, "keypoints"
        if rule.value.startswith("pointing"):
            return rule.value, "keypoints"
    return decide(rule, vlm_value, "keypoints")
