"""Subgoal task runner: executes a task definition and serves status/commands."""

import argparse
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from aha_tasks.core import POLICIES, TaskEngine, load_spec, resolve, select
from aha_tasks.skills import ARM_OUT, LOCATION, check_locations, go_to

STATUS_TOPIC = "/aha/tasks/status"
COMMAND_SERVICE = "/aha/tasks/command"
ARM_OK = "arm_ok"  # command: the operator moved the arm back home


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def load_task(path):
    """Task definition and its skill classes, fully validated without ROS."""
    spec = load_spec(path)
    classes = {name: resolve(target) for name, target in spec.skills.items()}
    for subgoal in spec.subgoals:
        skill = classes[subgoal.skill]
        if skill is not None:
            try:
                skill.validate(subgoal.params, spec)
            except ValueError as error:
                raise ValueError(f"{subgoal.id}: {error}") from None
    if spec.locations:
        check_locations(spec)
    return spec, classes


def guard_navigation(navigation, runtime):
    """No base motion while the runtime's ARM_OUT is set (see Navigation.blocked)."""
    navigation.blocked = lambda: getattr(runtime, ARM_OUT, None)


def arm_ok(engine, runtime):
    """The arm_ok command: the operator checked that the arm is home and that
    no pick, place or handover goal still runs. Clears the runtime's ARM_OUT
    and what the hands hold back (aha_tasks.handling.Hands), so that the base
    moves and goals are sent again. Refused while a subgoal runs."""
    if engine.state == "running":
        return False, f"{ARM_OK} is not available while a subgoal runs"
    hands = getattr(runtime, "hands", None)
    was = getattr(runtime, ARM_OUT, None) or (hands and hands.blocked())
    if hands is not None:
        hands.arm_ok()
    setattr(runtime, ARM_OUT, None)
    if not was:
        return True, "The arm was not marked out"
    engine.notify(f"Arm back home ({ARM_OK} from the operator); cleared: {was}")
    return True, "Arm marked home: the base may move again"


def describe(spec, selection):
    lines = [
        f"{spec.title} ({spec.task}): "
        f"{len(selection)}/{len(spec.subgoals)} subgoal(s) selected"
    ]
    for subgoal in spec.subgoals:
        mode = "RUN " if subgoal.id in selection else "mock"
        binding = spec.skills[subgoal.skill] or "NOT IMPLEMENTED"
        params = json.dumps(subgoal.params, ensure_ascii=False)
        lines.append(f"[{mode}] {subgoal.id}: {subgoal.skill} {params} -> {binding}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--subgoals",
        default="",
        help="Default selection: comma-separated ids, globs (guest1.*) or "
        "ranges (first..last); empty selects all",
    )
    parser.add_argument(
        "--on-failure", choices=POLICIES, default="pause", help="Default failure policy"
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without ROS or motion"
    )
    parser.add_argument("--startup-timeout", type=positive_seconds, default=300.0)
    parser.add_argument("--goal-timeout", type=positive_seconds, default=600.0)
    args, ros_args = parser.parse_known_args(argv)
    spec, classes = load_task(args.config)
    selection = select(spec, args.subgoals)
    if args.dry_run:
        print("DRY RUN: no navigation goals will be sent")
        print(describe(spec, selection))
        return 0

    from aha_msgs.srv import TaskCommand
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String

    from aha_tasks.gaze import Gaze
    from aha_tasks.navigation import Navigation

    # Keep ROS alive during KeyboardInterrupt so the active goal can be cancelled.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node("task_runner")
    log = node.get_logger()
    status_publisher = node.create_publisher(
        String,
        STATUS_TOPIC,
        QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        ),
    )

    gaze = Gaze(node)

    def publish(status):
        gaze.task_state(status["state"])
        status_publisher.publish(
            String(data=json.dumps(status, ensure_ascii=False, default=str))
        )

    navigation = (
        Navigation(node, spec.locations, args.startup_timeout, args.goal_timeout)
        if spec.locations
        else None
    )
    runtime = SimpleNamespace(node=node, spec=spec, navigation=navigation, gaze=gaze)
    if navigation is not None:
        guard_navigation(navigation, runtime)
    skills = {name: cls and cls(runtime) for name, cls in classes.items()}
    runtime.skills = skills
    restorers = {}
    if navigation is not None:
        restorers[LOCATION] = lambda value, cancelled: go_to(runtime, value, cancelled)

    def logger(level, message):
        # rclpy fixes the severity per call site, so each level needs its own call.
        if level == "error":
            log.error(message)
        elif level == "warn":
            log.warning(message)
        else:
            log.info(message)

    engine = TaskEngine(
        spec,
        skills,
        restorers,
        publish=publish,
        logger=logger,
        default_selection=selection,
        default_policy=args.on_failure,
        max_retries=args.max_retries,
    )

    def on_command(request, response):
        if request.command == "start":
            response.accepted, response.message = engine.start(
                list(request.subgoals), request.on_failure or None
            )
        elif request.command == ARM_OK:
            response.accepted, response.message = arm_ok(engine, runtime)
        else:
            response.accepted, response.message = engine.command(
                request.command, request.target or None
            )
        return response

    node.create_service(TaskCommand, COMMAND_SERVICE, on_command)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    code = 0
    try:
        if navigation is not None:
            engine.notify("Waiting for active Nav2")
            navigation.ready()
        for skill in skills.values():
            if skill is not None:
                skill.ready()
        engine.set_ready("Ready")
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        engine.set_error(f"{type(error).__name__}: {error}")
        code = 1
    finally:
        if not engine.shutdown(timeout=30.0):
            log.error("Task did not stop within 30 s")
            code = 1
        if navigation is not None:
            try:
                navigation.cancel()
            except Exception as error:
                log.error(f"Could not confirm navigation cancellation: {error}")
                code = 1
        executor.shutdown()
        spinner.join(timeout=5.0)  # before tearing down the node
        node.destroy_node()
        rclpy.try_shutdown()
    return code
