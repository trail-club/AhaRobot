"""Where the head looks during a task: a point, a person or the path ahead.

The runner owns one Gaze, and every head target of the skills goes through it
to /aha/perception/look_at. head_look_at.py turns the head and keeps a map
target in view while the base moves. A follow mode lasts until the next call,
across subgoals, so a skill that runs meanwhile (a conversation) keeps the gaze:

- person: the face (FACE_DROP below Person.head) of a person from
  /aha/perception/people, by track, else by name, else the nearest visible
  person within REACQUIRE of the last sighting who has no other name
  (person_perception gives a lost person a new track).
- path: a point LOOKAHEAD along the latest Nav2 plan (/plan) at LOOK_HEIGHT.
  Past the end it continues in the travel direction (from the plan's start to
  its end), or along the goal heading once the plan is shorter than ARRIVING
  (Nav2 turns on the spot), so the head ends level and forward instead of
  looking down at the goal under the robot.

The follow modes update in the executor thread on a steady (wall) clock
timer, not the runner's sim time: head_look_at.py and the people's staleness
run on wall time too, and a slow or stopped /clock must not stall the gaze.
Until the first plan of a navigation arrives, the head looks past the goal
along the goal heading.
"""

import math
import threading
import time

LOOK_AT_TOPIC = "/aha/perception/look_at"
PLAN_TOPIC = "/plan"
PERIOD = 0.2  # s (wall clock) between follow updates
LOOK_FRAME = "base_footprint"  # on the floor, so LOOK_HEIGHT is above the floor
LOOK_DISTANCE, LOOK_HEIGHT = 2.0, 1.2  # m; about level from the head camera
LOOKAHEAD = 1.5  # m along the plan
ARRIVING = 0.3  # m of plan left when the head turns to the goal heading
REACQUIRE = 0.5  # m from the last sighting
FACE_DROP = 0.1  # m; Person.head is the top of the mask, the face is lower
MIN_MOVE = 0.01  # m; smaller target changes are not sent again


def path_lookahead(points, distance, goal_yaw, arriving=ARRIVING):
    """(x, y) `distance` along `points` from the first.

    Past the end it continues from the first point towards the last (the
    travel direction; the last steps of a grid plan point anywhere), or along
    goal_yaw if they are closer than `arriving` (turning at the goal).
    """
    if not points:
        return None
    left = distance
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        step = math.hypot(x1 - x0, y1 - y0)
        if step >= left and step > 0.0:
            f = left / step
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
        left -= step
    (x0, y0), (x, y) = points[0], points[-1]
    if math.hypot(x - x0, y - y0) < arriving:
        heading = goal_yaw
    else:
        heading = math.atan2(y - y0, x - x0)
    return (x + left * math.cos(heading), y + left * math.sin(heading))


def follow_choice(people, track=None, name="", last=None, reacquire=REACQUIRE):
    """The person to keep looking at: by track, by name, then near `last` (x, y)."""
    visible = [p for p in people if p.visible]
    for person in visible:
        if track is not None and person.id == track:
            return person
    if name:
        for person in visible:
            if person.name == name:
                return person
    if last is None:
        return None
    near = [
        (math.dist((p.position.x, p.position.y), last), p)
        for p in visible
        if not p.name or p.name == name
    ]
    near = [(d, p) for d, p in near if d <= reacquire]
    return min(near, key=lambda item: item[0])[1] if near else None


class Gaze:
    """Head targets of one runner; methods may be called from any thread."""

    def __init__(self, node):
        from geometry_msgs.msg import Point, PointStamped
        from nav_msgs.msg import Path
        from rclpy.clock import Clock, ClockType

        self.Point, self.PointStamped = Point, PointStamped
        self.publisher = node.create_publisher(PointStamped, LOOK_AT_TOPIC, 10)
        node.create_subscription(Path, PLAN_TOPIC, self._on_plan, 10)
        node.create_timer(
            PERIOD, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )
        self.lock = threading.Lock()
        self.mode = None  # None, "person" or "path"
        self.plan = None  # (received, frame, [(x, y), ...])
        self.sent = None  # (frame, (x, y, z)) of the last target
        self.people = None
        self.track, self.name, self.last, self.seen = None, "", None, None
        self.goal, self.started = None, 0.0

    # -- targets -----------------------------------------------------------

    def look_at(self, frame, point):
        """Look at a point and stop following."""
        with self.lock:
            self.mode = None
            self._send(frame, (point.x, point.y, point.z), force=True)

    def look_towards(self, pan):
        """Look `pan` rad (+ = left) from the base's front, about level."""
        self.look_at(
            LOOK_FRAME,
            self.Point(
                x=LOOK_DISTANCE * math.cos(pan),
                y=LOOK_DISTANCE * math.sin(pan),
                z=LOOK_HEIGHT,
            ),
        )

    def follow_person(self, people, track=None, name=""):
        """Keep looking at a person (aha_tasks.people.People supplies the tracks)."""
        with self.lock:
            self.mode, self.people = "person", people
            self.track, self.name, self.last, self.seen = track, name, None, None
        self._tick()

    def follow_path(self, goal):
        """Look ahead along the plan of the navigation to goal [x, y, yaw] (map).

        Until its first plan, the head looks LOOKAHEAD past the goal along the
        goal heading, as for a plan shorter than ARRIVING: the goal itself at
        LOOK_HEIGHT is next to the head when the robot is already there (a
        restore that Nav2 ends at once), and the head would keep that target.
        """
        ahead = path_lookahead([(goal[0], goal[1])], LOOKAHEAD, goal[2])
        with self.lock:
            self.mode, self.goal, self.started = "path", goal, time.monotonic()
            self._send("map", (*ahead, LOOK_HEIGHT), force=True)

    def hold(self):
        """Stop following; the head keeps the last target."""
        with self.lock:
            self.mode = None

    def task_state(self, state):
        """Stop following unless the task runs (TaskEngine.state).

        Paused, aborted or finished, the head is the operator's: a target sent
        now would turn it back from the panel or teleop. After a resume the head
        keeps the last target until a skill sets a new one.
        """
        if state != "running":
            self.hold()

    # -- updates -----------------------------------------------------------

    def _on_plan(self, message):
        points = [(p.pose.position.x, p.pose.position.y) for p in message.poses]
        with self.lock:
            self.plan = (time.monotonic(), message.header.frame_id, points)

    def _tick(self):
        with self.lock:
            if self.mode == "person":
                self._follow_person()
            elif self.mode == "path":
                self._follow_path()

    def _follow_person(self):
        message = self.people.latest()
        if message is None or message is self.seen:
            return
        self.seen = message
        person = follow_choice(message.people, self.track, self.name, self.last)
        if person is None:
            return  # keep the last target until the person is seen again
        self.track = person.id
        self.last = (person.position.x, person.position.y)
        head = person.head
        self._send(message.header.frame_id, (head.x, head.y, head.z - FACE_DROP))

    def _follow_path(self):
        if self.plan is None or self.plan[0] < self.started:
            return  # the goal target from follow_path stays
        _, frame, points = self.plan
        ahead = path_lookahead(points, LOOKAHEAD, self.goal[2])
        if ahead is not None:
            self._send(frame or "map", (*ahead, LOOK_HEIGHT))

    def _send(self, frame, xyz, force=False):
        if (
            not force
            and self.sent is not None
            and self.sent[0] == frame
            and math.dist(self.sent[1], xyz) < MIN_MOVE
        ):
            return
        target = self.PointStamped()
        target.header.frame_id = frame
        target.point = self.Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        self.publisher.publish(target)
        self.sent = (frame, tuple(xyz))
