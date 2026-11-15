"""Simulated person tracks from ground-truth poses (ROS free, numpy).

Used by scripts/sim_person_detector.py, the debug stand-in for
person_perception.py. A person model is a vertical body axis at its model
origin: it is seen when a point of the axis projects into the image and the
depth there is not closer than that point minus the body radius
(``person_visible``). ``TruthTracker`` gives the tracks the same life as
person_perception's: a model gets a new track ID after it was not seen for
``forget_after`` s, a remembered name follows the model to its next track at
once (where person_perception asks the VLM), and the published tracks keep
the ones seen recently but not now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch

import numpy as np

from aha_perception.sim_object_detection import point_visible, project

AXIS_SHARES = (0.15, 0.35, 0.55, 0.75, 0.92)  # of the height, on the body axis
CENTRE_SHARE = 0.55  # body centre, like the median of a standing person's mask
HEAD_SHARE = 0.92  # top of the body, like the median of the mask's upper points
HIDDEN_BELOW = -1.0  # m; the sims hide people under the floor


def is_person(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def axis_points(height: float, shares=AXIS_SHARES) -> np.ndarray:
    """Homogeneous points (4, N) on the body axis, in the model frame."""
    points = np.zeros((4, len(shares)))
    points[2] = [share * height for share in shares]
    points[3] = 1.0
    return points


def person_visible(
    depth: np.ndarray,
    K,
    T_cam_model: np.ndarray,
    height: float,
    radius: float,
    tolerance: float,
    margin_px: float = 0.0,
) -> bool:
    """Whether a point of the body axis is in view and not hidden behind something."""
    points = T_cam_model @ axis_points(height)
    return any(
        point_visible(depth, K, points[:3, i], radius, tolerance, margin_px)
        for i in range(points.shape[1])
    )


def body_bbox(K, T_cam_model: np.ndarray, height: float, radius: float, shape):
    """[x0, y0, x1, y1] pixels around the projected body axis, clipped to the image."""
    points = T_cam_model @ axis_points(height, (0.0, 0.5, 1.0))
    pixels = []
    for i in range(points.shape[1]):
        for side in (-radius, radius):
            uv = project(K, points[:3, i] + np.array([side, 0.0, 0.0]))
            if uv is not None:
                pixels.append(uv)
    if not pixels:
        return [0, 0, 0, 0]
    us, vs = zip(*pixels)
    h, w = shape
    return [
        int(np.clip(min(us), 0, w - 1)),
        int(np.clip(min(vs), 0, h - 1)),
        int(np.clip(max(us), 0, w - 1)),
        int(np.clip(max(vs), 0, h - 1)),
    ]


@dataclass
class Sighting:
    model: str
    centre: np.ndarray  # in the output frame
    head: np.ndarray
    bbox: list


@dataclass
class TruthTrack:
    id: int
    model: str
    centre: np.ndarray
    head: np.ndarray
    bbox: list
    last_seen: float
    name: str = ""


@dataclass
class TruthTracker:
    forget_after: float = 3.0
    tracks: dict = field(default_factory=dict)  # model -> TruthTrack
    names: dict = field(default_factory=dict)  # model -> remembered name
    next_id: int = 1

    def update(self, sightings, stamp: float) -> set:
        """Tracks of one image; returns the IDs seen in it."""
        for model in [
            m for m, t in self.tracks.items() if stamp - t.last_seen > self.forget_after
        ]:
            del self.tracks[model]
        seen = set()
        for s in sightings:
            track = self.tracks.get(s.model)
            if track is None:
                track = TruthTrack(
                    self.next_id, s.model, s.centre, s.head, s.bbox, stamp
                )
                self.next_id += 1
                self.tracks[s.model] = track
            track.centre, track.head, track.bbox = s.centre, s.head, s.bbox
            track.last_seen = stamp
            seen.add(track.id)
        held = {t.name for t in self.tracks.values() if t.name}
        for track in self.tracks.values():
            name = self.names.get(track.model, "")
            if name and not track.name and name not in held:
                track.name = name
                held.add(name)
        return seen

    def by_id(self, track_id: int):
        for track in self.tracks.values():
            if track.id == track_id:
                return track
        return None

    def remember(self, track_id: int, name: str) -> str:
        """Name a track and its model; returns an error message or ''."""
        track = self.by_id(track_id)
        if not name:
            return "name must not be empty"
        if track is None:
            return f"person {track_id} is not tracked"
        for model, other in list(self.names.items()):
            if other == name:
                del self.names[model]
        for other in self.tracks.values():
            if other.name == name:
                other.name = ""
        self.names[track.model] = name
        track.name = name
        return ""
