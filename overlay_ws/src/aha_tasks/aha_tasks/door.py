"""Wait outside the arena until a referee opens the entrance door (rulebook 3.6.8).

The robot stands outside, facing the closed door. It says that it waits for
the door, looks straight ahead with the head camera and watches the space in
front of it: a corridor `width` m wide, from NEAR in front of the base centre,
between LOW and HIGH above the floor (above the floor, below a door's lintel).
The door counts as open once that corridor is free for `distance` m in FRAMES
consecutive frames over at least HOLD s of one sensor, and in every sensor
that delivers data: the head depth camera (the real robot has no lidar) and,
in simulation, /scan. Missing depth is never free space: a depth image shows
the corridor free only with pixels that look through it and see something
beyond `distance` (the floor or the room behind the open door). A frame that
shows neither that nor an obstacle (a black or glossy door without depth)
counts as a closed door: the robot keeps waiting. A sensor that delivers no
usable frame (no image or TF, the head not looking ahead, a blind camera)
for NO_DATA_TIMEOUT fails the skill.
Before it waits, the robot folds its arms in front of the body to fit through
the door, and fails if they cannot be folded. The sensors are subscribed only
while the skill runs. The skill does not move the base: the next navigate
subgoal drives through the door.

ROS and numpy are imported on use, so task definitions load without them.
"""

import math
import threading
import time

from aha_tasks.core.skill import Skill, SkillResult

DEPTH_TOPIC = "/camera/depth_registered/image_rect"
CAMERA_INFO_TOPIC = "/camera/color/camera_info"
SCAN_TOPIC = "/scan"
BASE_FRAME = "base_footprint"

DISTANCE = 1.5  # m from the base centre that must be free ahead
WIDTH = 0.5  # m; corridor width, below a 0.9 m door (rulebook 3.2.2, appendix)
NEAR = 0.3  # m; the corridor starts in front of the robot's own body
LOW, HIGH = 0.1, 1.8  # m above the floor
DEPTH_MIN_POINTS = 20  # a few stray depth points do not block the corridor
SCAN_MIN_POINTS = 2
DEPTH_STRIDE = 4  # px; every 4th row and column (160 x 120 of 640 x 480)
MIN_VALID = 0.05  # fraction of valid depth pixels; less is a blind camera
HIDDEN = -math.inf  # free distance of a corridor without depth: not open
MAX_PAN = 0.2  # rad of the camera axis from the base's front, left or right
MAX_UP, MAX_DOWN = 0.35, 0.6  # rad of the camera axis above / below level
FRAMES = 5  # consecutive frames with the corridor free
HOLD = 1.0  # s (sensor stamps) from the first to the last of them
STALE_AFTER = 1.0  # s (wall clock) without a message: the sensor is not used
NO_DATA_TIMEOUT = 10.0  # s (wall clock) without any usable frame: fail
CLOCK_STALL_TIMEOUT = 10.0  # s (wall clock) without the wait clock advancing
LOG_PERIOD = 30.0  # s (wall clock) between reports while waiting
POLL = 0.05  # s
TF_WAIT = 0.1  # s for the transform at an image's stamp
WAITING_TEXT = "I am waiting for the door to open."
# Arms folded to pass a door, mirrored, at the lift's height: the upper arms
# swung 0.9 rad back, the forearms forward along the body and the grippers
# turned in, about 0.38 m to each side (meshes; 0.53 m at zero, the
# simulation's start, which does not fit through a 0.9 to 1.0 m door). Within
# the real arms' limits (aha_manipulation/config/arm.yaml), which MoveIt's
# "standby" (joint_r3 1.5) is not: the left elbow folds only to 1.08 rad.
FOLDED = {
    "/right_arm_controller/follow_joint_trajectory": {
        "joint_r2": -0.9,
        "joint_r3": 1.06,
        "joint_r4": -0.5,
        "joint_r5": 0.0,
        "joint_r6": 0.0,
    },
    "/left_arm_controller/follow_joint_trajectory": {
        "joint_l2": 0.9,
        "joint_l3": -1.06,
        "joint_l4": 0.5,
        "joint_l5": 0.0,
        "joint_l6": 0.0,
    },
}
ARM_SPEED = 1.0  # rad/s, half the joint_limits.yaml limit (as offer_seat)
MIN_DURATION = 1.0  # s per motion
FOLDED_WITHIN = 0.02  # rad; closer arms do not move
SERVER_TIMEOUT = 2.0  # s (wall) for an arm controller or the joint states
RESULT_MARGIN = 5.0  # s (wait clock) after the motion's duration
STATUS_SUCCEEDED = 4  # action_msgs/msg/GoalStatus


def depth_rays(depth, fx, fy, cx, cy, stride=DEPTH_STRIDE):
    """(rays, ranges) of every `stride`-th pixel of a depth image in metres.

    rays are (N, 3) in the optical frame (x right, y down, z forward) with
    z = 1, so that ray * range is the point; ranges are the depths, NaN where
    invalid. Nothing at all and too near or too far (0, NaN or inf) are all
    invalid in a depth image.
    """
    import numpy as np

    depth = np.asarray(depth, dtype=np.float64)
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    sample = depth[::stride, ::stride].ravel()
    rays = np.column_stack(
        [
            (cols.ravel() - cx) / fx,
            (rows.ravel() - cy) / fy,
            np.ones(sample.size),
        ]
    )
    ranges = np.where(np.isfinite(sample) & (sample > 0.0), sample, np.nan)
    return rays, ranges


def depth_free(rays, ranges, rotation, translation, distance, width=WIDTH):
    """Free distance ahead (m) seen in depth rays; HIDDEN or None if not seen.

    The camera pose (rotation, translation) is in the base frame. An obstacle
    in the corridor (DEPTH_MIN_POINTS points) gives its distance. Without one,
    the corridor is free only if as many rays pass through its cross-section at
    `distance` and see something beyond it. Otherwise the image shows nothing
    of the corridor (no depth on a black or glossy door, a door too near):
    HIDDEN, a closed door. None (unusable) if fewer than MIN_VALID of all rays
    have depth: the camera is blind.
    """
    import numpy as np

    rotation, translation = np.asarray(rotation), np.asarray(translation)
    directions = np.asarray(rays, dtype=np.float64) @ rotation.T
    ranges = np.asarray(ranges, dtype=np.float64)
    valid = np.isfinite(ranges)
    if valid.size == 0 or valid.mean() < MIN_VALID:
        return None
    points = translation + directions[valid] * ranges[valid, None]
    free = free_distance(points, width=width, min_points=DEPTH_MIN_POINTS)
    if free < distance:
        return free
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = (distance - translation[0]) / directions[:, 0]
    cross = translation + directions * reach[:, None]
    through = (
        (directions[:, 0] > 0.0)
        & (np.abs(cross[:, 1]) <= width / 2.0)
        & (cross[:, 2] >= LOW)
        & (cross[:, 2] <= HIGH)
    )
    beyond = through & valid & (np.nan_to_num(ranges, nan=-1.0) > reach)
    return free if np.count_nonzero(beyond) >= DEPTH_MIN_POINTS else HIDDEN


def scan_points(ranges, angle_min, angle_increment, range_min, range_max):
    """(N, 3) points of the returns of a scan, in the laser frame.

    A beam without a return (inf, beyond range_max) gives no point, so it
    counts as free; NaN and returns below range_min are ignored.
    """
    import numpy as np

    ranges = np.asarray(ranges, dtype=np.float64)
    angles = angle_min + angle_increment * np.arange(ranges.size)
    hit = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    r, a = ranges[hit], angles[hit]
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.zeros(r.size)])


def to_frame(points, rotation, translation):
    """Points (N, 3) moved by a rotation matrix and a translation."""
    import numpy as np

    return np.asarray(points) @ np.asarray(rotation).T + np.asarray(translation)


def rotation_matrix(x, y, z, w):
    """3 x 3 rotation of a unit quaternion."""
    import numpy as np

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def free_distance(points, width=WIDTH, near=NEAR, low=LOW, high=HIGH, min_points=1):
    """Free distance (m, base x) straight ahead in the corridor; inf if nothing.

    points are (N, 3) in the base frame (x forward, z up from the floor). The
    distance is that of the `min_points`-th nearest point in the corridor, so
    that a few stray points (depth noise) do not block it.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = (x >= near) & (np.abs(y) <= width / 2.0) & (z >= low) & (z <= high)
    ahead = x[inside]
    if ahead.size < min_points:
        return math.inf
    return float(np.partition(ahead, min_points - 1)[min_points - 1])


def looks_ahead(rotation, max_pan=MAX_PAN, max_up=MAX_UP, max_down=MAX_DOWN):
    """Whether a camera (optical frame rotation in the base frame) looks ahead."""
    import numpy as np

    axis = np.asarray(rotation)[:, 2]  # the optical z axis
    pan = math.atan2(axis[1], axis[0])
    down = math.atan2(-axis[2], math.hypot(axis[0], axis[1]))
    return abs(pan) <= max_pan and -max_up <= down <= max_down


class DoorWatch:
    """Frames of one sensor; open after FRAMES free frames over HOLD s."""

    def __init__(self, distance=DISTANCE, frames=FRAMES, hold=HOLD):
        self.distance, self.frames, self.hold = distance, frames, hold
        self.count, self.since, self.stamp = 0, None, None
        self.free = None  # free distance of the last frame; None: unusable

    def update(self, free, stamp):
        """Add a frame (free distance, or None if unusable); True once open."""
        if self.since is not None and stamp < self.since:  # the sim restarted
            self.count, self.since = 0, None
        self.free = free
        if free is None or free < self.distance:
            self.count, self.since = 0, None
            return False
        if self.count == 0:
            self.since = stamp
        self.count += 1
        self.stamp = stamp
        return self.open

    @property
    def open(self):
        return self.count >= self.frames and self.stamp - self.since >= self.hold


def door_open(watches):
    """Every sensor whose last frame was usable sees the door open; one at least.

    A sensor that cannot see now (head turned away, blind camera) does not
    count, but its frames start again once it can.
    """
    usable = [watch for watch in watches if watch.free is not None]
    return bool(usable) and all(watch.open for watch in usable)


def describe(free):
    if free is None:
        return "no usable frame"
    if free == HIDDEN:
        return "no depth in the corridor (closed)"
    if math.isinf(free):
        return "nothing ahead"
    return f"{free:.2f} m free ahead"


def fold_duration(current, target, speed=ARM_SPEED, minimum=MIN_DURATION):
    """Seconds to move the joints from `current` to `target` (name -> rad).

    0 if every joint is within FOLDED_WITHIN already; unknown joints count as
    the full way from 0.
    """
    moves = [abs(value - current.get(name, 0.0)) for name, value in target.items()]
    if all(move < FOLDED_WITHIN for move in moves):
        return 0.0
    return max(minimum, max(moves) / speed)


def number(params, key, default, minimum=0.0):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number: {value!r}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be at least {minimum}: {value!r}")
    return float(value)


class Sensors:
    """Latest head depth image and /scan with their transforms, for one runner.

    Subscribed only from start() to stop(): the runner's single executor would
    otherwise take every depth image and TF message for the rest of the task.
    """

    def __init__(self, node):
        self.node = node
        self.lock = threading.Lock()
        self.latest = {}  # source -> (message, wall time received)
        self.info = None
        self.subscriptions = []
        self.tf = self.listener = None

    def start(self):
        """Subscribe to the sensors and TF; nothing if subscribed already."""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, LaserScan
        from tf2_ros import Buffer, TransformListener

        if self.subscriptions:
            return
        for topic, kind, source in (
            (DEPTH_TOPIC, Image, "depth"),
            (SCAN_TOPIC, LaserScan, "scan"),
        ):
            self.subscriptions.append(
                self.node.create_subscription(
                    kind,
                    topic,
                    lambda message, source=source: self._store(source, message),
                    qos_profile_sensor_data,
                )
            )
        self.subscriptions.append(
            self.node.create_subscription(
                CameraInfo, CAMERA_INFO_TOPIC, self._on_info, qos_profile_sensor_data
            )
        )
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self.node)

    def stop(self):
        """Unsubscribe and forget the frames."""
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
        self.subscriptions = []
        if self.listener is not None:
            self.listener.unregister()
        self.tf = self.listener = None
        with self.lock:
            self.latest.clear()
            self.info = None

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "door_sensors", None) is None:
            runtime.door_sensors = cls(runtime.node)
        return runtime.door_sensors

    def _store(self, source, message):
        with self.lock:
            self.latest[source] = (message, time.monotonic())

    def _on_info(self, message):
        with self.lock:
            self.info = message

    def fresh(self):
        """source -> latest message received within STALE_AFTER."""
        now = time.monotonic()
        with self.lock:
            return {
                source: message
                for source, (message, received) in self.latest.items()
                if now - received <= STALE_AFTER
            }

    def transform(self, frame, stamp):
        """(rotation, translation) of frame in BASE_FRAME at stamp, else latest."""
        import numpy as np

        from rclpy.duration import Duration
        from rclpy.time import Time
        from tf2_ros import TransformException

        if self.tf is None:
            return None
        try:
            t = self.tf.lookup_transform(
                BASE_FRAME, frame, Time.from_msg(stamp), Duration(seconds=TF_WAIT)
            ).transform
        except TransformException:
            try:
                t = self.tf.lookup_transform(BASE_FRAME, frame, Time()).transform
            except TransformException:
                return None
        q, p = t.rotation, t.translation
        return rotation_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])

    def depth_free(self, message, width, distance):
        """Free distance ahead in a depth image, or None if it is unusable."""
        with self.lock:
            info = self.info
        if info is None or message.width == 0 or message.height == 0:
            return None
        depth = depth_array(message)
        if depth is None:
            return None
        pose = self.transform(message.header.frame_id, message.header.stamp)
        if pose is None or not looks_ahead(pose[0]):
            return None
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        if info.width and info.width != message.width:  # e.g. 640 info, 320 depth
            scale = message.width / info.width
            fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
        rays, ranges = depth_rays(depth, fx, fy, cx, cy)
        return depth_free(rays, ranges, *pose, distance, width)

    def scan_free(self, message, width, distance):
        """Free distance ahead in a scan; a beam without a return is far."""
        pose = self.transform(message.header.frame_id, message.header.stamp)
        if pose is None:
            return None
        points = scan_points(
            message.ranges,
            message.angle_min,
            message.angle_increment,
            message.range_min,
            message.range_max,
        )
        return free_distance(
            to_frame(points, *pose), width=width, min_points=SCAN_MIN_POINTS
        )


def depth_array(message):
    """Depth in metres of a 32FC1 (m) or 16UC1 (mm) image; None otherwise."""
    import numpy as np

    kinds = {"32FC1": (np.float32, 1.0), "16UC1": (np.uint16, 0.001)}
    if message.encoding not in kinds:
        return None
    kind, scale = kinds[message.encoding]
    dtype = np.dtype(kind).newbyteorder(">" if message.is_bigendian else "<")
    row = message.step // dtype.itemsize
    data = np.frombuffer(bytes(message.data), dtype=dtype)
    if row < message.width or data.size < row * message.height:
        return None
    image = data[: row * message.height].reshape(message.height, row)
    return image[:, : message.width].astype(np.float64) * scale


def missing_joints(current):
    """The FOLDED joints without a position in `current`."""
    return [n for target in FOLDED.values() for n in target if n not in current]


class Arms:
    """Folds both arms (FOLDED) for one runner; no collision checking."""

    def __init__(self, node):
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient

        self.node = node
        self.FollowJointTrajectory = FollowJointTrajectory
        self.clients = {
            action: ActionClient(node, FollowJointTrajectory, action)
            for action in FOLDED
        }
        self.lock = threading.Lock()
        self.joints = {}
        self.subscription = None

    @classmethod
    def of(cls, runtime):
        if getattr(runtime, "door_arms", None) is None:
            runtime.door_arms = cls(runtime.node)
        return runtime.door_arms

    def _subscribe(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        with self.lock:
            self.joints = {}
        self.subscription = self.node.create_subscription(
            JointState, "/joint_states", self._on_joint_states, qos_profile_sensor_data
        )

    def _unsubscribe(self):
        if self.subscription is not None:
            self.node.destroy_subscription(self.subscription)
        self.subscription = None

    def _on_joint_states(self, message):
        with self.lock:
            self.joints.update(zip(message.name, message.position))

    def _goal(self, target, duration):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        point = JointTrajectoryPoint(positions=list(target.values()))
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int(duration % 1.0 * 1e9)
        goal = self.FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=list(target), points=[point])
        return goal

    def current(self):
        """Joint positions from /joint_states, waiting up to SERVER_TIMEOUT for all."""
        names = [name for target in FOLDED.values() for name in target]
        deadline = time.monotonic() + SERVER_TIMEOUT
        while True:
            with self.lock:
                if all(name in self.joints for name in names):
                    return dict(self.joints)
            if time.monotonic() >= deadline:
                with self.lock:
                    return dict(self.joints)
            time.sleep(0.02)

    def folded(self):
        """Whether /joint_states shows both arms at FOLDED."""
        current = self.current()
        if missing_joints(current):
            return False
        return all(fold_duration(current, target) == 0.0 for target in FOLDED.values())

    def fold(self, cancelled):
        """(ok, message): ok once /joint_states shows both arms folded.

        A failure (no controller, a rejected goal, an error, no result in time,
        cancellation) cancels every goal that still runs, and counts only if
        the arms are not folded anyway.
        """
        self._subscribe()
        try:
            ok, message = self._fold(cancelled)
            if not ok and not cancelled.is_set() and self.folded():
                return True, f"Folded, though {message}"
            return ok, message
        finally:
            self._unsubscribe()

    def _fold(self, cancelled):
        from aha_tasks.interaction import cancel_if_accepted

        current = self.current()
        missing = missing_joints(current)
        if missing:
            # The duration comes from where the joints are: without them the
            # arm could move far faster than ARM_SPEED.
            return False, f"no /joint_states of {', '.join(missing)}: arms not moved"
        goals = []  # (action, handle, result future, duration)
        try:
            for action, target in FOLDED.items():
                duration = fold_duration(current, target)
                if duration == 0.0:
                    continue
                client = self.clients[action]
                if not client.wait_for_server(timeout_sec=SERVER_TIMEOUT):
                    return False, f"{action} is not served"
                sent = client.send_goal_async(self._goal(target, duration))
                if not wait_done(sent, SERVER_TIMEOUT, cancelled):
                    sent.add_done_callback(cancel_if_accepted)
                    if cancelled.is_set():
                        return False, "Cancelled"
                    return False, f"{action}: no answer"
                handle = sent.result()
                if not handle.accepted:
                    return False, f"{action} rejected the goal"
                goals.append((action, handle, handle.get_result_async(), duration))
            clock = WaitClock(self.node)
            for action, _, result, duration in goals:
                while not result.done():
                    if cancelled.is_set():
                        return False, "Cancelled"
                    waited, stalled = clock.elapsed()
                    if stalled or waited > duration + RESULT_MARGIN:
                        return False, f"{action} did not finish in time"
                    time.sleep(0.02)
                response = result.result()
                code = response.result.error_code
                # A cancelled or aborted goal may end with error_code 0.
                if response.status != STATUS_SUCCEEDED or code != 0:
                    return False, f"{action} failed (status {response.status}, {code})"
            return True, "Folded" if goals else "Already folded"
        finally:
            for _, handle, result, _ in goals:  # only after a failure
                if not result.done():
                    handle.cancel_goal_async()


def wait_done(future, timeout, cancelled):
    deadline = time.monotonic() + timeout
    while not future.done():
        if cancelled.is_set() or time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


class WaitDoorOpen(Skill):
    """Wait outside the arena for the referee to open the door (start signal).

    params (optional): {"distance": m free ahead (1.5), "width": m (0.5),
    "timeout": s (0: until the task is cancelled), "fold": false to leave the
    arms as they are (no arm controllers)}
    """

    @classmethod
    def validate(cls, params, spec):
        unknown = set(params) - {"distance", "width", "timeout", "fold"}
        if unknown:
            raise ValueError(
                f"wait_door_open may have distance, width, timeout and fold: {params}"
            )
        if number(params, "distance", DISTANCE) <= NEAR:
            raise ValueError(f"distance must be beyond {NEAR} m: {params}")
        if number(params, "width", WIDTH) <= 0.0:
            raise ValueError(f"width must be positive: {params}")
        number(params, "timeout", 0.0)
        if not isinstance(params.get("fold", True), bool):
            raise ValueError(f"fold must be true or false: {params}")

    def ready(self):
        from aha_tasks.interaction import Interaction

        Arms.of(self.runtime)
        Interaction.of(self.runtime).find_tts()

    def execute(self, ctx):
        from aha_tasks.interaction import Interaction

        distance = number(ctx.params, "distance", DISTANCE)
        width = number(ctx.params, "width", WIDTH)
        timeout = number(ctx.params, "timeout", 0.0)
        sensors = Sensors.of(self.runtime)
        # From the start, so that frames are there once the arms are folded.
        sensors.start()
        try:
            gaze = getattr(self.runtime, "gaze", None)
            if gaze is not None:
                gaze.look_towards(0.0)  # level, straight ahead
            if ctx.params.get("fold", True):
                ok, message = Arms.of(self.runtime).fold(ctx.cancelled)
                if ctx.cancelled.is_set():
                    return SkillResult(False, "Cancelled")
                if not ok:
                    return SkillResult(False, f"Could not fold the arms: {message}")
                ctx.log(f"Arms: {message}")
            ok, message = Interaction.of(self.runtime).say(WAITING_TEXT, ctx.cancelled)
            if ctx.cancelled.is_set():
                return SkillResult(False, "Cancelled")
            if not ok:
                ctx.log(f"Could not say that the robot waits for the door: {message}")
            return self.watch(ctx, sensors, distance, width, timeout)
        finally:
            sensors.stop()

    def watch(self, ctx, sensors, distance, width, timeout):
        watches = {}  # source -> DoorWatch
        seen = {}  # source -> last message processed
        logged = set()
        clock = WaitClock(self.runtime.node)
        started = usable = last_log = time.monotonic()

        def report(sources):
            text = ", ".join(f"{s}: {describe(watches[s].free)}" for s in sources)
            return text or "no sensor data"

        while True:
            if ctx.cancelled.is_set():
                return SkillResult(False, "Cancelled")
            now = time.monotonic()
            messages = sensors.fresh()
            for source, message in messages.items():
                if seen.get(source) is message:
                    continue
                seen[source] = message
                if source == "depth":
                    free = sensors.depth_free(message, width, distance)
                else:
                    free = sensors.scan_free(message, width, distance)
                stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
                if free is not None and source not in logged:
                    logged.add(source)  # the first view, e.g. the closed door
                    ctx.log(f"{source}: {describe(free)}")
                watch = watches.setdefault(source, DoorWatch(distance))
                watch.update(free, stamp)
                if free is not None:
                    usable = now
            current = [s for s in messages if s in watches]
            if door_open([watches[s] for s in current]):
                return SkillResult(
                    True, f"Door open after {now - started:.1f} s ({report(current)})"
                )
            if now - usable > NO_DATA_TIMEOUT:
                return SkillResult(
                    False,
                    f"No usable {DEPTH_TOPIC} (head looking ahead) or {SCAN_TOPIC} "
                    f"for {NO_DATA_TIMEOUT:.0f} s",
                )
            waited, stalled = clock.elapsed()
            if stalled:
                return SkillResult(
                    False,
                    f"No /clock progress for {CLOCK_STALL_TIMEOUT:.0f} s; "
                    "is simulation running?",
                )
            if timeout > 0.0 and waited >= timeout:
                return SkillResult(
                    False, f"Door not open within {timeout:.0f} s ({report(watches)})"
                )
            if now - last_log >= LOG_PERIOD:
                last_log = now
                ctx.log(f"Waiting for the door: {report(current)}")
            time.sleep(POLL)


class WaitClock:
    """Seconds of the node clock (sim time if active, else monotonic) since creation."""

    def __init__(self, node):
        self.node = node
        self.start = self.last = self.now()
        self.progress = time.monotonic()

    def now(self):
        clock = self.node.get_clock()
        if not getattr(clock, "ros_time_is_active", False):
            return time.monotonic()
        return clock.now().nanoseconds * 1e-9

    def elapsed(self):
        """(seconds waited, whether the clock stalled for CLOCK_STALL_TIMEOUT)."""
        now, wall = self.now(), time.monotonic()
        if now > self.last:
            self.last, self.progress = now, wall
        return now - self.start, wall - self.progress >= CLOCK_STALL_TIMEOUT
