"""wait_door_open: the free corridor ahead, the decision and the skill (no ROS graph)."""

import math
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks import door  # noqa: E402
from aha_tasks.core import load_spec  # noqa: E402
from aha_tasks.core.skill import SkillContext  # noqa: E402
from aha_tasks.door import (  # noqa: E402
    FOLDED,
    Arms,
    DoorWatch,
    WaitDoorOpen,
    depth_array,
    depth_free,
    depth_rays,
    door_open,
    fold_duration,
    free_distance,
    looks_ahead,
    rotation_matrix,
    scan_points,
    to_frame,
)
from aha_tasks.runner import load_task  # noqa: E402
from aha_tasks.sim_door import EntranceDoor, load_config  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config"
# Optical frame (x right, y down, z forward) of a level camera in the base frame.
LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
CAMERA = np.array([0.1, 0.0, 1.3])  # m above the floor, near the base centre
FX = FY = 554.0  # px; 640 x 480 with a 60 deg field of view
CX, CY = 319.5, 239.5


def pitched(rotation, down):
    """A camera rotation turned `down` rad about the base's y axis."""
    c, s = math.cos(down), math.sin(down)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ rotation


def render(wall_x=None, gap=None, floor=True, far=10.0, black=False):
    """Depth image of a level CAMERA: a wall at base x `wall_x` with an
    opening |y| < gap / 2 (if given), the floor beyond, nothing above.
    A black wall returns no depth."""
    rows, cols = np.mgrid[0:480, 0:640].astype(float)
    # Ray directions in the base frame for each pixel.
    rays = np.stack([(cols - CX) / FX, (rows - CY) / FY, np.ones_like(cols)], -1)
    rays = rays @ LEVEL.T
    depth = np.full(rows.shape, np.inf)  # optical z of the first hit
    if floor:
        t = np.where(
            rays[..., 2] < 0, -CAMERA[2] / np.minimum(rays[..., 2], -1e-9), np.inf
        )
        depth = np.minimum(depth, t)
    if wall_x is not None:
        t = (wall_x - CAMERA[0]) / rays[..., 0]
        y = CAMERA[1] + t * rays[..., 1]
        z = CAMERA[2] + t * rays[..., 2]
        hit = (t > 0) & (z >= 0.0) & (z <= 2.0)
        if gap is not None:
            hit &= np.abs(y) >= gap / 2
        if black:
            depth = np.where(hit, np.nan, depth)
        else:
            depth = np.where(hit, np.minimum(depth, t), depth)
    depth[depth > far] = np.inf  # beyond the camera's range
    return depth


def free_ahead(depth, rotation=LEVEL, camera=CAMERA, distance=door.DISTANCE):
    rays, ranges = depth_rays(depth, FX, FY, CX, CY)
    return depth_free(rays, ranges, rotation, camera, distance)


class CorridorTests(unittest.TestCase):
    def test_closed_door_is_near(self):
        self.assertAlmostEqual(free_ahead(render(wall_x=1.0)), 1.0, places=2)

    def test_open_door_frees_the_corridor(self):
        # Seen through the doorway: the floor beyond `distance`.
        self.assertEqual(free_ahead(render(wall_x=1.0, gap=0.9)), math.inf)

    def test_missing_depth_is_not_free(self):
        # A black or glossy door returns no depth, but the frame and the floor
        # around do: the frame is unusable, not an open door.
        self.assertIsNone(free_ahead(render(wall_x=1.0, black=True)))
        self.assertIsNone(free_ahead(np.full((480, 640), np.nan)))  # blind
        self.assertIsNone(free_ahead(np.zeros((480, 640))))  # all too near
        # Nothing at all within range beyond the doorway (no floor): unusable.
        self.assertIsNone(free_ahead(render(wall_x=1.0, gap=0.9, floor=False)))

    def test_partly_open_door_still_blocks(self):
        self.assertAlmostEqual(free_ahead(render(wall_x=1.0, gap=0.3)), 1.0, places=2)

    def test_robot_off_the_door_centre_sees_the_frame(self):
        depth = render(wall_x=1.0, gap=0.9)  # door 0.3 m left
        free = free_ahead(depth, camera=CAMERA + [0.0, 0.3, 0.0])
        self.assertAlmostEqual(free, 1.0, places=2)

    def test_a_door_beyond_distance_counts_as_open(self):
        # The robot must stand nearer to the door than `distance`.
        self.assertAlmostEqual(free_ahead(render(wall_x=2.0)), 2.0, places=2)

    def test_rays_and_ranges(self):
        depth = np.array([[1.0, 0.0], [np.inf, np.nan]])
        rays, ranges = depth_rays(depth, 1.0, 1.0, 0.0, 0.0, stride=1)
        np.testing.assert_allclose(rays, [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        np.testing.assert_equal(ranges, [1.0, np.nan, np.nan, np.nan])

    def test_a_few_stray_points_do_not_block(self):
        points = np.array([[0.8, 0.0, 1.0]] * 5 + [[3.0, 0.1, 1.0]] * 30)
        self.assertEqual(free_distance(points, min_points=20), 3.0)
        self.assertEqual(free_distance(points, min_points=1), 0.8)

    def test_corridor_limits(self):
        points = np.array(
            [
                [0.2, 0.0, 1.0],  # the robot's own body, before NEAR
                [1.0, 0.3, 1.0],  # beside the corridor
                [1.0, 0.0, 0.05],  # the floor
                [1.0, 0.0, 2.1],  # a lintel above the door
            ]
        )
        self.assertEqual(free_distance(points), math.inf)
        self.assertEqual(free_distance(points, width=0.7), 1.0)

    def test_scan_without_returns_is_free(self):
        ranges = [math.inf, float("nan"), 0.05, 1.0, 20.0]
        points = scan_points(ranges, -0.2, 0.1, 0.12, 12.0)
        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(points[0], [1.0 * math.cos(0.1), math.sin(0.1), 0])
        laser = np.array([0.08, 0.0, 0.35])
        free = free_distance(to_frame(points, np.eye(3), laser))
        self.assertAlmostEqual(free, 0.08 + math.cos(0.1))

    def test_the_head_must_look_ahead(self):
        self.assertTrue(looks_ahead(LEVEL))
        self.assertTrue(looks_ahead(pitched(LEVEL, 0.3)))  # slightly down
        self.assertFalse(looks_ahead(pitched(LEVEL, 0.9)))  # at the floor
        self.assertFalse(looks_ahead(pitched(LEVEL, -0.5)))  # at the ceiling
        yaw = 0.5
        panned = (
            np.array(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            @ LEVEL
        )
        self.assertFalse(looks_ahead(panned))

    def test_quaternion_rotation(self):
        q = (0.0, 0.0, math.sin(0.3), math.cos(0.3))  # 0.6 rad about z
        np.testing.assert_allclose(
            rotation_matrix(*q) @ [1.0, 0.0, 0.0], [math.cos(0.6), math.sin(0.6), 0.0]
        )

    def test_depth_encodings(self):
        def message(encoding, data, width=3, height=2, step=None, big=False):
            itemsize = 4 if encoding == "32FC1" else 2
            return SimpleNamespace(
                encoding=encoding,
                width=width,
                height=height,
                step=step or width * itemsize,
                is_bigendian=big,
                data=data,
            )

        metres = np.arange(6, dtype="<f4").reshape(2, 3)
        np.testing.assert_allclose(
            depth_array(message("32FC1", metres.tobytes())), metres
        )
        mm = np.array([[1000, 0, 2500, 9], [1, 2, 3, 9]], dtype="<u2")  # padded rows
        np.testing.assert_allclose(
            depth_array(message("16UC1", mm.tobytes(), step=8)),
            [[1.0, 0.0, 2.5], [0.001, 0.002, 0.003]],
        )
        self.assertIsNone(depth_array(message("rgb8", b"")))
        self.assertIsNone(depth_array(message("32FC1", b"\0" * 8)))  # too short


class DecisionTests(unittest.TestCase):
    def test_open_after_frames_over_hold(self):
        watch = DoorWatch(distance=1.5, frames=3, hold=1.0)
        self.assertFalse(watch.update(1.0, 0.0))  # closed
        self.assertFalse(watch.update(2.0, 0.1))
        self.assertFalse(watch.update(math.inf, 0.2))
        self.assertFalse(watch.update(math.inf, 0.6))  # 3 frames, 0.5 s
        self.assertTrue(watch.update(math.inf, 1.1))

    def test_closed_or_unusable_frames_start_again(self):
        watch = DoorWatch(distance=1.5, frames=2, hold=0.0)
        watch.update(3.0, 0.0)
        self.assertFalse(watch.update(None, 0.1))  # head moved away
        self.assertFalse(watch.update(3.0, 0.2))
        self.assertTrue(watch.update(3.0, 0.3))
        self.assertFalse(watch.update(1.0, 0.4))  # the door swung back
        self.assertFalse(watch.open)

    def test_a_clock_reset_starts_again(self):
        watch = DoorWatch(distance=1.5, frames=2, hold=0.5)
        watch.update(3.0, 100.0)
        self.assertFalse(watch.update(3.0, 1.0))  # sim restarted
        self.assertFalse(watch.update(3.0, 1.2))
        self.assertTrue(watch.update(3.0, 1.6))

    def test_every_usable_sensor_must_agree(self):
        depth, scan = DoorWatch(frames=1, hold=0.0), DoorWatch(frames=1, hold=0.0)
        self.assertFalse(door_open([]))
        depth.update(3.0, 0.0)
        scan.update(1.0, 0.0)
        self.assertFalse(door_open([depth, scan]))
        scan.update(math.inf, 0.1)
        self.assertTrue(door_open([depth, scan]))
        depth.update(None, 0.2)  # the camera does not see now: the scan decides
        self.assertTrue(door_open([depth, scan]))
        scan.update(None, 0.3)
        self.assertFalse(door_open([depth, scan]))


class FakeSensors:
    """Free distances per source and frame instead of messages."""

    def __init__(self, frames):
        self.frames = frames  # source -> list of free distances (or None)
        self.index = 0
        self.lock = threading.Lock()
        self.subscribed = False

    def start(self):
        self.subscribed = True

    def stop(self):
        self.subscribed = False

    def fresh(self):
        with self.lock:
            self.index += 1
            messages = {}
            for source, values in self.frames.items():
                if values:
                    value = values[min(self.index, len(values)) - 1]
                    stamp = SimpleNamespace(sec=self.index, nanosec=0)
                    messages[source] = SimpleNamespace(
                        value=value, header=SimpleNamespace(stamp=stamp)
                    )
            return messages

    def depth_free(self, message, width, distance):
        return message.value

    scan_free = depth_free


def context(params=None):
    return SkillContext(
        subgoal="start.wait_door",
        params=params or {},
        memory={},
        cancelled=threading.Event(),
        log=lambda text: None,
    )


def skill():
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace())  # wall clock
    return WaitDoorOpen(SimpleNamespace(node=node))


@patch.object(door, "POLL", 0.0)
class SkillTests(unittest.TestCase):
    def run_watch(self, frames, params=None):
        params = params or {}
        return skill().watch(
            context(params),
            FakeSensors(frames),
            params.get("distance", door.DISTANCE),
            door.WIDTH,
            params.get("timeout", 0.0),
        )

    def test_waits_while_closed_then_enters(self):
        # FakeSensors stamps 1 s per poll: FRAMES frames span HOLD.
        result = self.run_watch(
            {"depth": [0.95] * 5 + [4.0] * 10, "scan": [0.9] * 6 + [math.inf] * 10}
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("depth: 4.00 m free ahead", result.message)
        self.assertIn("scan: nothing ahead", result.message)

    def test_closed_door_times_out(self):
        result = self.run_watch({"depth": [0.95]}, {"timeout": 0.2})
        self.assertFalse(result.success)
        self.assertIn("Door not open within", result.message)
        self.assertIn("0.95 m free ahead", result.message)

    @patch.object(door, "NO_DATA_TIMEOUT", 0.2)
    def test_no_usable_frames_fail(self):
        result = self.run_watch({"depth": [None]})
        self.assertFalse(result.success)
        self.assertIn("No usable", result.message)
        result = self.run_watch({})
        self.assertIn("No usable", result.message)

    def test_cancel(self):
        ctx = context()
        ctx.cancelled.set()
        result = skill().watch(
            ctx, FakeSensors({"depth": [0.95]}), door.DISTANCE, door.WIDTH, 0.0
        )
        self.assertEqual((result.success, result.message), (False, "Cancelled"))

    def test_params(self):
        spec = SimpleNamespace(locations={})
        WaitDoorOpen.validate({}, spec)
        WaitDoorOpen.validate({"distance": 2.0, "width": 0.6, "timeout": 30}, spec)
        for params in (
            {"distance": 0.2},
            {"width": 0},
            {"timeout": -1},
            {"timeout": True},
            {"fold": "no"},
            {"door": "entrance"},
        ):
            with self.subTest(params), self.assertRaises(ValueError):
                WaitDoorOpen.validate(params, spec)
        WaitDoorOpen.validate({"fold": False}, spec)
        self.assertEqual(WaitDoorOpen(None).expected({}), {})  # stays outside

    def execute(self, params, arms, frames):
        sensors = FakeSensors(frames)
        runtime = SimpleNamespace(
            node=SimpleNamespace(get_clock=lambda: SimpleNamespace()),
            door_sensors=sensors,
            door_arms=arms,
            interaction=SimpleNamespace(say=lambda text, cancelled: (True, "shown")),
        )
        result = WaitDoorOpen(runtime).execute(context(params))
        self.assertFalse(sensors.subscribed)  # unsubscribed once done
        return result

    @patch.object(door, "SERVER_TIMEOUT", 0.05)
    def test_the_skill_fails_without_folded_arms(self):
        arms = FakeArms(right=FakeClient(served=False))
        result = self.execute({}, arms, {"depth": [4.0]})
        self.assertFalse(result.success)
        self.assertIn("Could not fold the arms", result.message)
        result = self.execute({}, FakeArms(), {"depth": [4.0]})
        self.assertTrue(result.success, result.message)

    def test_fold_false_leaves_the_arms(self):
        arms = SimpleNamespace(fold=None)  # calling it would fail
        result = self.execute({"fold": False}, arms, {"depth": [4.0]})
        self.assertTrue(result.success, result.message)


class FakeFuture:
    def __init__(self, value=None, done=True):
        self.value, self.finished, self.callbacks = value, done, []

    def done(self):
        return self.finished

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)


class FakeHandle:
    def __init__(self, accepted=True, code=0, finish=True):
        self.accepted = accepted
        self.cancelled = False
        result = SimpleNamespace(result=SimpleNamespace(error_code=code))
        self.result = FakeFuture(result, done=finish)

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancelled = True


class FakeClient:
    """A FollowJointTrajectory client; on_send runs when a goal is sent."""

    def __init__(self, served=True, answer=True, on_send=None, **handle):
        self.served, self.answer, self.on_send = served, answer, on_send
        self.handle = FakeHandle(**handle)
        self.goals, self.futures = [], []

    def wait_for_server(self, timeout_sec):
        return self.served

    def send_goal_async(self, goal):
        self.goals.append(goal)
        if self.on_send is not None:
            self.on_send()
        future = FakeFuture(self.handle, done=self.answer)
        self.futures.append(future)
        return future


class FakeArms(Arms):
    """Arms with fake controllers and joint states, at `joints` when subscribed."""

    def __init__(self, joints=None, right=None, left=None):
        self.node = SimpleNamespace(get_clock=lambda: SimpleNamespace())  # wall
        self.lock = threading.Lock()
        self.start_joints = dict(joints or {})
        self.joints = {}
        self.subscription = None
        right_action, left_action = FOLDED
        self.right, self.left = right or FakeClient(), left or FakeClient()
        self.clients = {right_action: self.right, left_action: self.left}

    def _subscribe(self):
        self.joints = dict(self.start_joints)
        self.subscription = "/joint_states"

    def _unsubscribe(self):
        self.subscription = None

    def _goal(self, target, duration):
        return dict(target), duration


ALL_FOLDED = {
    name: value for target in FOLDED.values() for name, value in target.items()
}
NEARLY_FOLDED = {name: value - 0.1 for name, value in ALL_FOLDED.items()}


@patch.object(door, "SERVER_TIMEOUT", 0.05)
class ArmTests(unittest.TestCase):
    def test_folds_both_arms(self):
        arms = FakeArms()
        self.assertEqual(arms.fold(threading.Event()), (True, "Folded"))
        self.assertEqual(len(arms.right.goals), 1)
        self.assertEqual(arms.left.goals[0], (FOLDED[next(reversed(FOLDED))], 1.5))
        self.assertIsNone(arms.subscription)  # /joint_states only while folding

    def test_folded_arms_do_not_move(self):
        arms = FakeArms(ALL_FOLDED, right=FakeClient(served=False))
        self.assertEqual(arms.fold(threading.Event()), (True, "Already folded"))
        self.assertEqual(arms.left.goals, [])

    def test_failures_fail_unless_the_arms_are_folded(self):
        for client, text in (
            (FakeClient(served=False), "is not served"),
            (FakeClient(accepted=False), "rejected the goal"),
            (FakeClient(code=-4), "failed (-4)"),
        ):
            with self.subTest(text):
                ok, message = FakeArms(right=client).fold(threading.Event())
                self.assertFalse(ok)
                self.assertIn(text, message)
        arms = FakeArms(right=FakeClient(code=-4))
        arms.right.on_send = lambda: arms.joints.update(ALL_FOLDED)  # moved anyway
        ok, message = arms.fold(threading.Event())
        self.assertTrue(ok)
        self.assertIn("Folded, though", message)

    @patch.object(door, "RESULT_MARGIN", 0.05)
    def test_a_late_arm_cancels_every_running_goal(self):
        arms = FakeArms(
            NEARLY_FOLDED, right=FakeClient(finish=False), left=FakeClient(finish=False)
        )
        ok, message = arms.fold(threading.Event())  # 1 s motions
        self.assertFalse(ok)
        self.assertIn("did not finish in time", message)
        self.assertTrue(arms.right.handle.cancelled and arms.left.handle.cancelled)

    def test_cancelling_stops_both_arms(self):
        arms = FakeArms(right=FakeClient(finish=False), left=FakeClient(finish=False))
        cancelled = threading.Event()
        threading.Timer(0.1, cancelled.set).start()
        self.assertEqual(arms.fold(cancelled), (False, "Cancelled"))
        self.assertTrue(arms.right.handle.cancelled and arms.left.handle.cancelled)

    def test_an_unanswered_goal_cancels_the_sent_one(self):
        arms = FakeArms(right=FakeClient(finish=False), left=FakeClient(answer=False))
        ok, message = arms.fold(threading.Event())
        self.assertFalse(ok)
        self.assertIn("no answer", message)
        self.assertTrue(arms.right.handle.cancelled)
        # The late answer of the other is cancelled when it comes.
        self.assertEqual(len(arms.left.futures[0].callbacks), 1)
        self.assertFalse(arms.left.handle.cancelled)
        arms.left.futures[0].callbacks[0](arms.left.futures[0])
        self.assertTrue(arms.left.handle.cancelled)


class FakeNode:
    """Records the subscriptions of Sensors."""

    def __init__(self):
        self.topics = []

    def create_subscription(self, kind, topic, callback, qos, **kwargs):
        self.topics.append(topic)
        return topic

    def destroy_subscription(self, subscription):
        # As rclpy: False for one destroyed already (TransformListener.__del__).
        if subscription not in self.topics:
            return False
        self.topics.remove(subscription)
        return True


class SensorSubscriptionTests(unittest.TestCase):
    def test_subscribed_only_while_running(self):
        node = FakeNode()
        sensors = door.Sensors(node)
        self.assertEqual(node.topics, [])
        for _ in range(2):  # a retry subscribes again
            sensors.start()
            sensors.start()
            self.assertIn(door.DEPTH_TOPIC, node.topics)
            self.assertIn(door.SCAN_TOPIC, node.topics)
            self.assertIn("/tf", node.topics)
            self.assertEqual(len(node.topics), len(set(node.topics)))
            sensors.stop()
            self.assertEqual(node.topics, [])
            self.assertIsNone(sensors.transform("camera", None))


class FoldTests(unittest.TestCase):
    def test_duration_from_the_farthest_joint(self):
        right, left = FOLDED.values()
        self.assertEqual(fold_duration({}, right), 1.5)  # from zero: joint_r3 1.5 rad
        self.assertEqual(fold_duration(dict(right), right), 0.0)  # already folded
        near = {name: value + 0.01 for name, value in left.items()}
        self.assertEqual(fold_duration(near, left), 0.0)
        near["joint_l2"] += 0.2
        self.assertEqual(fold_duration(near, left), 1.0)  # at least MIN_DURATION

    def test_the_left_arm_mirrors_the_right(self):
        right, left = FOLDED.values()
        for (r_joint, r_value), (l_joint, l_value) in zip(right.items(), left.items()):
            self.assertEqual((r_joint[-1], r_value), (l_joint[-1], -l_value))


def status(state, phase=None, current=None):
    return {
        "current": current,
        "phase": phase,
        "subgoals": [{"id": "start.wait_door", "status": state}],
    }


class SimDoorTests(unittest.TestCase):
    def test_opens_after_the_delay_of_the_wait(self):
        door_ = EntranceDoor("start.wait_door", delay=5.0)
        self.assertFalse(door_.update(None, 0.0))  # no runner yet
        self.assertFalse(door_.update(status("pending"), 1.0))
        self.assertFalse(door_.update(status("running", None, "start.wait_door"), 2.0))
        running = status("running", "executing", "start.wait_door")
        self.assertFalse(door_.update(running, 3.0))
        self.assertFalse(door_.update(running, 7.9))
        self.assertTrue(door_.update(running, 8.0))
        self.assertTrue(door_.update(status("succeeded"), 9.0))
        self.assertFalse(door_.update(status("pending"), 10.0))  # a new run

    def test_open_when_not_waited_for_or_going_back_out(self):
        door_ = EntranceDoor("start.wait_door", delay=5.0)
        for state in ("bypassed", "unsupported", "skipped", "failed"):
            self.assertTrue(door_.update(status(state), 0.0), state)
        restoring = status("running", "restoring", "start.wait_door")
        self.assertTrue(door_.update(restoring, 1.0))
        running = status("running", "executing", "start.wait_door")
        self.assertFalse(door_.update(running, 2.0))  # closes for the wait

    def test_entrance_scene_matches_the_tasks(self):
        scene = load_config(CONFIG / "sim" / "entrance.json")
        for name in ("gpsr", "pick_and_place", "doing_laundry"):
            with self.subTest(name):
                spec = load_spec(CONFIG / f"{name}.json")
                self.assertIn(scene["location"], spec.locations)
                self.assertIn(scene["trigger"], [s.id for s in spec.subgoals])
                start, *_ = spec.subgoals
                self.assertEqual(start.id, scene["trigger"])
                self.assertEqual(
                    spec.skills[start.skill], "aha_tasks.door:WaitDoorOpen"
                )

    def test_runner_starts_outside_on_request(self):
        spec, _ = load_task(CONFIG / "gpsr.json", "start_outside")
        self.assertEqual(spec.initial_state["robot.location"], "start_outside")
        spec, _ = load_task(CONFIG / "gpsr.json")
        self.assertEqual(spec.initial_state["robot.location"], "spawn")
        with self.assertRaises(ValueError):
            load_task(CONFIG / "gpsr.json", "garden")


if __name__ == "__main__":
    unittest.main()
