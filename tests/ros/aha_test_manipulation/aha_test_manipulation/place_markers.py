"""RViz markers of a place goal (/aha/manipulation/place/markers).

``SpotBoard`` keeps the spots of the running goal in the order the place
server tries them: the number is that order (1 first), the color the state
(pending, trying, failed, placed); spots dropped before trying (out of reach,
blocked) are small grey dots without a number. The goal's near point and its
group and avoid points are drawn too. Points are in the planning frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

PENDING = "pending"
TRYING = "trying"
FAILED = "failed"
PLACED = "placed"
DROPPED = "dropped"

COLORS = {
    PENDING: (0.1, 0.5, 1.0, 0.9),
    TRYING: (1.0, 0.85, 0.1, 1.0),
    FAILED: (0.9, 0.15, 0.1, 0.9),
    PLACED: (0.1, 0.9, 0.2, 1.0),
    DROPPED: (0.6, 0.6, 0.6, 0.6),
}
NEAR_COLOR = (1.0, 1.0, 1.0, 0.9)
GROUP_COLOR = (0.1, 0.9, 0.2, 0.5)
AVOID_COLOR = (0.9, 0.15, 0.1, 0.5)
SPOT_SIZE = 0.025  # m, sphere diameter
DROPPED_SIZE = 0.012
RING_SIZE = 0.06  # m, disc diameter at the reference points
RING_HEIGHT = 0.004
TEXT_HEIGHT = 0.035
TEXT_OFFSET = 0.05  # m above the spot


@dataclass
class Spot:
    point: tuple[float, float, float]
    state: str = PENDING
    note: str = ""


@dataclass
class SpotBoard:
    """Spots of one goal. ``spots`` holds the numbered ones (tried and
    queued, in order), ``dropped`` the others."""

    spots: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    near: tuple | None = None
    group: list = field(default_factory=list)
    avoid: list = field(default_factory=list)

    def reset(self, near=None, group=(), avoid=()):
        self.spots, self.dropped = [], []
        self.near = None if near is None else tuple(float(v) for v in near)
        self.group = [tuple(float(v) for v in p) for p in group]
        self.avoid = [tuple(float(v) for v in p) for p in avoid]

    def queue(self, points):
        """The spots still to try (replaces the pending ones; the tried
        ones keep their numbers)."""
        self.spots = [s for s in self.spots if s.state != PENDING]
        self.spots += [Spot(tuple(float(v) for v in p)) for p in points]

    def drop(self, point, note=""):
        self.dropped.append(Spot(tuple(float(v) for v in point), DROPPED, note))

    def start(self) -> int:
        """Mark the first pending spot as tried now; its number (0: none)."""
        for k, spot in enumerate(self.spots):
            if spot.state == PENDING:
                spot.state = TRYING
                return k + 1
        return 0

    def note(self, text):
        """Note on the spot being tried (the latest reason it failed)."""
        spot = self.current()
        if spot is not None:
            spot.note = text

    def finish(self, state, note=None):
        spot = self.current()
        if spot is not None:
            spot.state = state
            if note is not None:
                spot.note = note

    def current(self) -> Spot | None:
        return next((s for s in self.spots if s.state == TRYING), None)

    def markers(self, frame: str) -> MarkerArray:
        header = Header(frame_id=frame)
        out = [Marker(header=header, action=Marker.DELETEALL)]
        for k, spot in enumerate(self.spots):
            out.append(sphere(header, "spot", k, spot.point, SPOT_SIZE, spot.state))
            label = f"{k + 1}" + (f" {spot.note}" if spot.note else "")
            out.append(text(header, "number", k, spot.point, label))
        for k, spot in enumerate(self.dropped):
            out.append(sphere(header, "dropped", k, spot.point, DROPPED_SIZE, DROPPED))
        if self.near is not None:
            out.append(ring(header, "near", 0, self.near, NEAR_COLOR))
        z = self.near[2] if self.near is not None else 0.0
        for k, p in enumerate(self.group):
            out.append(ring(header, "group", k, (p[0], p[1], z), GROUP_COLOR))
        for k, p in enumerate(self.avoid):
            out.append(ring(header, "avoid", k, (p[0], p[1], z), AVOID_COLOR))
        return MarkerArray(markers=out)


def rgba(values) -> ColorRGBA:
    r, g, b, a = values
    return ColorRGBA(r=r, g=g, b=b, a=a)


def sphere(header, ns, k, point, size, state) -> Marker:
    m = Marker(header=header, ns=ns, id=k, type=Marker.SPHERE, action=Marker.ADD)
    m.pose.position = Point(x=point[0], y=point[1], z=point[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = size
    m.color = rgba(COLORS[state])
    return m


def text(header, ns, k, point, label) -> Marker:
    m = Marker(
        header=header, ns=ns, id=k, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD
    )
    m.pose.position = Point(x=point[0], y=point[1], z=point[2] + TEXT_OFFSET)
    m.pose.orientation.w = 1.0
    m.scale.z = TEXT_HEIGHT
    m.color = rgba((1.0, 1.0, 1.0, 1.0))
    m.text = label
    return m


def ring(header, ns, k, point, color) -> Marker:
    m = Marker(header=header, ns=ns, id=k, type=Marker.CYLINDER, action=Marker.ADD)
    m.pose.position = Point(x=point[0], y=point[1], z=point[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = RING_SIZE
    m.scale.z = RING_HEIGHT
    m.color = rgba(color)
    return m
