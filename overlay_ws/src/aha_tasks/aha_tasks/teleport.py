"""Simulation only: a NavigateToPose server that teleports the robot in Gazebo.

Stands in for the Nav2 planner, controller and BT navigator in debug
end-to-end runs (task.launch.py e2e:=debug), so that a goal takes about a
second instead of driving at 0.2 m/s. The goal pose in `map` is the Gazebo
world pose, as in task.launch.py, which spawns the robot and starts AMCL at
the same coordinates. With `speed` > 0 the robot first stays where it is for
the straight-line distance / speed (simulation time), so that people walking
at that speed (aha_tasks sim_guests) are not overtaken.

AMCL publishes map -> odom stamped transform_tolerance (2 s) ahead, so the TF
at an image's time shows a new pose only that long after AMCL took it. The
pose therefore goes to AMCL first (/initialpose; standing still, AMCL does not
update from the scans until the robot moves), and the robot model is moved
with /world/<world>/set_pose as soon as AMCL's map -> base_footprint agrees
with the goal. Images before and after the move then get the right TF, except
for the image or two in between; the goal succeeds `settle` s (simulation
time) later, so that the people topics have replaced what they measured from
those. A goal whose footprint is not free on the map (occupied or unknown
cells) is aborted, as the Nav2 planner would. Nothing in between is checked:
the path, doors and people on the way.

The node also answers /bt_navigator/get_state, which aha_tasks Navigation
polls before the task can start.
"""

import math
import threading
import time

import numpy as np

STATE_UNCONFIGURED, STATE_ACTIVE = 1, 3  # lifecycle_msgs/msg/State
FREE_BELOW = 50  # occupancy (0-100); unknown is -1
INITIAL_COVARIANCE = 0.05**2  # m^2 and rad^2 on /initialpose
REQUEST_TIMEOUT = 2000  # ms, Gazebo services
REQUEST_ATTEMPTS = 3


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


def main(argv=None):
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GzNode
    from gz.transport13 import SubscribeOptions
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

    rclpy.init(args=argv)
    node = Node("sim_teleport")
    log = node.get_logger()
    p = node.declare_parameter
    world = p("world", "").value
    robot_model = p("robot_model", "aha_robot").value
    global_frame = p("global_frame", "map").value
    base_frame = p("base_frame", "base_footprint").value
    radius = p("robot_radius", 0.40).value  # nav2.yaml robot_radius
    tolerance_xy = p("tolerance_xy", 0.05).value
    tolerance_yaw = p("tolerance_yaw", 0.05).value
    settle_timeout = p("settle_timeout", 10.0).value  # s, wall clock
    speed = p("speed", 0.0).value  # m/s on the simulation clock; 0: at once
    settle = p("settle", 1.5).value  # s on the simulation clock after the move

    gz = GzNode()
    lock = threading.Lock()
    latest = {"robot": None, "map": None}

    if not world:
        deadline = time.monotonic() + 60.0
        while not world and time.monotonic() < deadline:
            for topic in gz.topic_list():
                parts = topic.split("/")
                if len(parts) == 4 and parts[1] == "world" and parts[3] == "clock":
                    world = parts[2]
            time.sleep(0.5)
        if not world:
            raise RuntimeError("no Gazebo world found (check GZ_PARTITION)")

    def on_poses(message):
        for pose in message.pose:
            if pose.name == robot_model:
                with lock:
                    latest["robot"] = pose
                return

    options = SubscribeOptions()
    options.msgs_per_sec = 10
    if not gz.subscribe(Pose_V, f"/world/{world}/pose/info", on_poses, options):
        raise RuntimeError(f"cannot subscribe to /world/{world}/pose/info")

    group = ReentrantCallbackGroup()

    def on_map(message):
        info = message.info
        if abs(yaw_of(info.origin.orientation)) > 1e-6:
            log.error("the map origin is rotated; goals are not checked")
            return
        with lock:
            latest["map"] = {
                "data": message.data,
                "width": info.width,
                "height": info.height,
                "resolution": info.resolution,
                "origin": [info.origin.position.x, info.origin.position.y],
            }

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

    def localized():
        """AMCL's [x, y, yaw] of the base in the global frame, or None."""
        try:
            t = tf_buffer.lookup_transform(global_frame, base_frame, Time()).transform
        except TransformException:
            return None
        return [t.translation.x, t.translation.y, yaw_of(t.rotation)]

    def on_get_state(request, response):
        with lock:
            ready = latest["map"] is not None and latest["robot"] is not None
        active = ready and localized() is not None
        response.current_state.id = STATE_ACTIVE if active else STATE_UNCONFIGURED
        response.current_state.label = "active" if active else "unconfigured"
        return response

    node.create_service(
        GetState, "/bt_navigator/get_state", on_get_state, callback_group=group
    )

    def set_pose(x, y, z, yaw):
        request = Pose(name=robot_model)
        request.position.x, request.position.y, request.position.z = x, y, z
        request.orientation.z = math.sin(yaw / 2.0)
        request.orientation.w = math.cos(yaw / 2.0)
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            ok, reply = gz.request(
                f"/world/{world}/set_pose", request, Pose, Boolean, REQUEST_TIMEOUT
            )
            if ok and reply.data:
                return True
            log.warning(
                f"set_pose attempt {attempt}: "
                + ("refused" if ok else f"no reply in {REQUEST_TIMEOUT} ms")
            )
        return False

    def send_initial_pose(goal):
        message = PoseWithCovarianceStamped()
        message.header.frame_id = global_frame
        message.header.stamp = node.get_clock().now().to_msg()
        message.pose.pose.position.x, message.pose.pose.position.y = goal[:2]
        message.pose.pose.orientation.z = math.sin(goal[2] / 2.0)
        message.pose.pose.orientation.w = math.cos(goal[2] / 2.0)
        for index in (0, 7, 35):  # x, y, yaw
            message.pose.covariance[index] = INITIAL_COVARIANCE
        initial_pose.publish(message)

    def travel(seconds, goal_handle):
        """Wait `seconds` of simulation time; False if cancelled or the clock stalls."""
        clock = node.get_clock()
        end = clock.now().nanoseconds * 1e-9 + seconds
        wall_end = time.monotonic() + seconds + settle_timeout
        while clock.now().nanoseconds * 1e-9 < end:
            if goal_handle.is_cancel_requested or time.monotonic() > wall_end:
                return False
            time.sleep(0.05)
        return True

    def stop(goal_handle, result):
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            log.error("the simulation clock stopped")
            goal_handle.abort()
        return result

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
        with busy:
            if frame != global_frame:
                log.error(f"goal in {frame}; only {global_frame} is supported")
                goal_handle.abort()
                return result
            with lock:
                grid, robot = latest["map"], latest["robot"]
            reason = (
                None
                if grid is None
                else footprint_blocked(grid, goal[0], goal[1], radius)
            )
            if grid is None or robot is None or reason:
                log.error(
                    f"cannot teleport to {goal}: "
                    + (
                        reason
                        or ("no /map" if grid is None else f"no pose of {robot_model}")
                    )
                )
                goal_handle.abort()
                return result
            here = [robot.position.x, robot.position.y, yaw_of(robot.orientation)]
            distance, angle = pose_error(here, goal)
            moves = distance > tolerance_xy or angle > tolerance_yaw
            if speed > 0.0 and not travel(distance / speed, goal_handle):
                return stop(goal_handle, result)
            started = time.monotonic()
            next_send = started
            while True:
                if goal_handle.is_cancel_requested:
                    send_initial_pose(here)
                    goal_handle.canceled()
                    return result
                if time.monotonic() - started > settle_timeout:
                    send_initial_pose(here)
                    log.error(f"AMCL did not take {goal} within {settle_timeout:.0f} s")
                    goal_handle.abort()
                    return result
                if time.monotonic() >= next_send:
                    send_initial_pose(goal)  # again if AMCL missed it
                    next_send = time.monotonic() + 1.0
                estimate = localized()
                if estimate is not None:
                    distance, angle = pose_error(estimate, goal)
                    if distance <= tolerance_xy and angle <= tolerance_yaw:
                        break
                time.sleep(0.05)
            if moves:
                if not set_pose(goal[0], goal[1], robot.position.z, goal[2]):
                    send_initial_pose(here)
                    log.error(f"Gazebo did not move {robot_model} to {goal}")
                    goal_handle.abort()
                    return result
                if not travel(settle, goal_handle):
                    return stop(goal_handle, result)
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
    log.info(f"teleporting {robot_model} in Gazebo world {world} for NavigateToPose")
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
