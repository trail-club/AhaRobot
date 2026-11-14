"""Simulation only: a NavigateToPose server that teleports the robot in Gazebo.

Stands in for the Nav2 planner, controller and BT navigator in debug
end-to-end runs (task.launch.py e2e:=debug), so that a goal takes seconds
instead of driving at 0.2 m/s. The goal pose in `map` is the Gazebo world
pose, as in task.launch.py, which spawns the robot and starts AMCL at the
same coordinates. With `speed` > 0 the robot first stays where it is for the
straight-line distance / speed, so that people walking at that speed
(aha_tasks sim_guests) are not overtaken.

AMCL publishes map -> odom stamped transform_tolerance (2 s) ahead, so the TF
at an image's time shows a new pose only that long after AMCL took it. The
pose therefore goes to AMCL first (/initialpose; standing still, AMCL does not
update from the scans until the robot moves), and the robot model is moved
with /world/<world>/set_pose as soon as AMCL's map -> base_footprint agrees
with the goal. Images before and after the move then get the right TF, except
for the image or two in between; the goal succeeds `settle` s later, so that
the people topics have replaced what they measured from those. A goal whose
footprint is not free on the map (occupied or unknown cells) is aborted, as
the Nav2 planner would. Nothing in between is checked: the path, doors and
people on the way.

All waits are on the simulation clock and give up only when /clock stops for
CLOCK_STALL_TIMEOUT wall seconds, so a slow simulation only takes longer. The
node also answers /bt_navigator/get_state, which aha_tasks Navigation polls
before the task can start.
"""

import math
import threading
import time

import numpy as np

from aha_tasks.people import CLOCK_STALL_TIMEOUT

STATE_UNCONFIGURED, STATE_ACTIVE = 1, 3  # lifecycle_msgs/msg/State
FREE_BELOW = 50  # occupancy (0-100); unknown is -1
INITIAL_COVARIANCE = 0.05**2  # m^2 and rad^2 on /initialpose
SET_POSE_ATTEMPTS = 3
WORLD_TIMEOUT = 60.0  # s (wall clock) to find the Gazebo world; then exit (respawn)


def yaw_of(q):
    """Yaw (rad) of a quaternion with x, y, z, w attributes."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def pose_error(a, b):
    """(planar distance m, absolute yaw difference rad) between [x, y, yaw] poses."""
    return (
        math.hypot(a[0] - b[0], a[1] - b[1]),
        abs(math.remainder(a[2] - b[2], math.tau)),
    )


def footprint_blocked(grid, x, y, radius):
    """Why a disc on an occupancy grid is not free, or None if it is.

    grid: dict with data (row-major, row 0 at origin y), width, height,
    resolution and origin [x, y] (the map's origin yaw must be 0).
    """
    resolution = grid["resolution"]
    ox, oy = grid["origin"]
    cells = np.asarray(grid["data"], dtype=np.int16).reshape(
        grid["height"], grid["width"]
    )
    reach = int(math.ceil(radius / resolution))
    ci = int(math.floor((x - ox) / resolution))
    cj = int(math.floor((y - oy) / resolution))
    unknown = occupied = 0
    for j in range(cj - reach, cj + reach + 1):
        for i in range(ci - reach, ci + reach + 1):
            cx = ox + (i + 0.5) * resolution
            cy = oy + (j + 0.5) * resolution
            if math.hypot(cx - x, cy - y) > radius:
                continue
            if not (0 <= i < grid["width"] and 0 <= j < grid["height"]):
                unknown += 1
                continue
            value = cells[j, i]
            if value < 0:
                unknown += 1
            elif value >= FREE_BELOW:
                occupied += 1
    if occupied:
        return f"{occupied} occupied cell(s) within {radius:.2f} m"
    if unknown:
        return f"{unknown} unknown cell(s) within {radius:.2f} m"
    return None


def wait_sim(
    sim_now,
    seconds=math.inf,
    done=lambda: False,
    cancelled=lambda: False,
    stall=CLOCK_STALL_TIMEOUT,
    wall_now=time.monotonic,
    sleep=time.sleep,
    poll=0.05,
):
    """Wait on the simulation clock `sim_now()` (s).

    Returns "done" when done() holds, "elapsed" after `seconds` of simulation
    time, "cancelled", or "stalled" when the clock does not advance for
    `stall` wall seconds. A slow clock only makes the wait longer.
    """
    start = last = sim_now()
    progress = wall_now()
    while True:
        if cancelled():
            return "cancelled"
        if done():
            return "done"
        now = sim_now()
        if now - start >= seconds:
            return "elapsed"
        if now > last:
            last, progress = now, wall_now()
        elif wall_now() - progress >= stall:
            return "stalled"
        sleep(poll)


def main(argv=None):
    from aha_perception.gazebo_poses import GazeboPoses
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from lifecycle_msgs.srv import GetState
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import OccupancyGrid
    import rclpy
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener

    from aha_tasks.guest_node import set_pose

    rclpy.init(args=argv)
    node = Node("sim_teleport")
    log = node.get_logger()
    p = node.declare_parameter
    world = p("world", "").value  # empty: the first world on the partition
    robot_model = p("robot_model", "aha_robot").value
    global_frame = p("global_frame", "map").value
    base_frame = p("base_frame", "base_footprint").value
    radius = p("robot_radius", 0.40).value  # nav2.yaml robot_radius
    tolerance_xy = p("tolerance_xy", 0.05).value
    tolerance_yaw = p("tolerance_yaw", 0.05).value
    localize_timeout = p("localize_timeout", 10.0).value  # s, simulation clock
    speed = p("speed", 0.0).value  # m/s on the simulation clock; 0: at once
    settle = p("settle", 1.5).value  # s on the simulation clock after the move

    from gz.transport13 import Node as GzNode

    gazebo = GazeboPoses(world, WORLD_TIMEOUT, 10.0)
    requests = GzNode()  # apart from the pose subscription
    lock = threading.Lock()
    latest = {"map": None}
    group = ReentrantCallbackGroup()

    def on_map(message):
        info = message.info
        grid = {
            "data": message.data,
            "width": info.width,
            "height": info.height,
            "resolution": info.resolution,
            "origin": [info.origin.position.x, info.origin.position.y],
            "rotated": abs(yaw_of(info.origin.orientation)) > 1e-6,
        }
        if grid["rotated"]:
            log.warning("the map origin is rotated; goals are not checked")
        with lock:
            latest["map"] = grid

    node.create_subscription(
        OccupancyGrid,
        "/map",
        on_map,
        QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        ),
        callback_group=group,
    )
    initial_pose = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    def robot_pose():
        """Gazebo's [x, y, z, yaw] of the robot model, or None."""
        poses, _ = gazebo.latest()
        T = poses.get(robot_model)
        if T is None:
            return None
        return [T[0, 3], T[1, 3], T[2, 3], math.atan2(T[1, 0], T[0, 0])]

    def localized():
        """AMCL's [x, y, yaw] of the base in the global frame, or None."""
        try:
            t = tf_buffer.lookup_transform(global_frame, base_frame, Time()).transform
        except TransformException:
            return None
        return [t.translation.x, t.translation.y, yaw_of(t.rotation)]

    def on_get_state(request, response):
        with lock:
            ready = latest["map"] is not None
        active = ready and robot_pose() is not None and localized() is not None
        response.current_state.id = STATE_ACTIVE if active else STATE_UNCONFIGURED
        response.current_state.label = "active" if active else "unconfigured"
        return response

    node.create_service(
        GetState, "/bt_navigator/get_state", on_get_state, callback_group=group
    )

    def move(x, y, z, yaw):
        for attempt in range(1, SET_POSE_ATTEMPTS + 1):
            if set_pose(requests, gazebo.world, robot_model, x, y, z, yaw):
                return True
            log.warning(f"set_pose attempt {attempt} failed")
        return False

    def send_initial_pose(pose):
        message = PoseWithCovarianceStamped()
        message.header.frame_id = global_frame
        message.header.stamp = node.get_clock().now().to_msg()
        message.pose.pose.position.x, message.pose.pose.position.y = pose[:2]
        message.pose.pose.orientation.z = math.sin(pose[2] / 2.0)
        message.pose.pose.orientation.w = math.cos(pose[2] / 2.0)
        for index in (0, 7, 35):  # x, y, yaw
            message.pose.covariance[index] = INITIAL_COVARIANCE
        initial_pose.publish(message)

    def sim_now():
        return node.get_clock().now().nanoseconds * 1e-9

    busy = threading.Lock()

    def execute(goal_handle):
        result = NavigateToPose.Result()
        pose = goal_handle.request.pose
        frame = pose.header.frame_id or global_frame
        goal = [
            pose.pose.position.x,
            pose.pose.position.y,
            yaw_of(pose.pose.orientation),
        ]

        def fail(message, restore=None):
            if restore is not None:
                send_initial_pose(restore)
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                log.error(message)
                goal_handle.abort()
            return result

        with busy:
            if frame != global_frame:
                return fail(f"goal in {frame}; only {global_frame} is supported")
            with lock:
                grid = latest["map"]
            robot = robot_pose()
            if grid is None or robot is None:
                return fail(
                    f"cannot teleport: no {'/map' if grid is None else robot_model}"
                )
            reason = (
                None
                if grid["rotated"]
                else footprint_blocked(grid, goal[0], goal[1], radius)
            )
            if reason:
                return fail(f"cannot teleport to {goal}: {reason}")
            here = [robot[0], robot[1], robot[3]]
            distance, angle = pose_error(here, goal)
            moves = distance > tolerance_xy or angle > tolerance_yaw

            def cancelled():
                return goal_handle.is_cancel_requested

            started = time.monotonic()
            if speed > 0.0:
                outcome = wait_sim(sim_now, distance / speed, cancelled=cancelled)
                if outcome != "elapsed":
                    return fail("the simulation clock stopped")
            sent = [-math.inf]

            def agrees():
                if time.monotonic() - sent[0] >= 1.0:
                    send_initial_pose(goal)  # again if AMCL missed it
                    sent[0] = time.monotonic()
                estimate = localized()
                if estimate is None:
                    return False
                error = pose_error(estimate, goal)
                return error[0] <= tolerance_xy and error[1] <= tolerance_yaw

            outcome = wait_sim(sim_now, localize_timeout, agrees, cancelled)
            if outcome != "done":
                return fail(
                    f"AMCL did not take {goal} within {localize_timeout:.0f} s"
                    if outcome == "elapsed"
                    else "the simulation clock stopped",
                    restore=here,
                )
            if moves:
                if not move(goal[0], goal[1], robot[2], goal[2]):
                    return fail(f"Gazebo did not move {robot_model}", restore=here)
                if wait_sim(sim_now, settle, cancelled=cancelled) != "elapsed":
                    return fail("the simulation clock stopped")
            log.info(
                f"teleported to [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}] "
                f"in {time.monotonic() - started:.1f} s"
            )
            goal_handle.succeed()
            return result

    ActionServer(
        node,
        NavigateToPose,
        "/navigate_to_pose",
        execute_callback=execute,
        goal_callback=lambda request: GoalResponse.ACCEPT,
        cancel_callback=lambda handle: CancelResponse.ACCEPT,
        callback_group=group,
    )
    log.info(
        f"teleporting {robot_model} in Gazebo world {gazebo.world} for NavigateToPose"
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
