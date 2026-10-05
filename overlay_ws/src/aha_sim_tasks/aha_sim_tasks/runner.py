"""Evaluate policies in isolated, freshly started Gazebo episodes."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from ament_index_python.packages import get_package_share_directory

from .evaluation import Evaluator, load_tasks
from .policy_process import PolicyProcess
from .processes import policy_partition, stop_group


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_episode(args, task):
    import rclpy
    from .ros_environment import RosEnvironment

    started = time.monotonic()
    result = {
        "task_id": task.task_id,
        "policy": args.policy,
        "status": "error",
        "metrics": {},
    }
    simulator = None
    policy = None
    node = None
    evaluator = None
    with (args.output.parent / "simulator.log").open("w") as log:
        try:
            policy = PolicyProcess(
                args.policy,
                task.task_id,
                task.instruction,
                args.policy_ros_domain_id,
                args.output.parent,
                os.environ["GZ_PARTITION"],
                startup_timeout=args.policy_startup_timeout,
                action_timeout=args.policy_timeout,
            )
            while not policy.ready:
                policy.poll()
                time.sleep(0.01)
            simulator = subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    "aha_sim_tasks",
                    "environment.launch.py",
                    "headless:=" + ("true" if args.headless else "false"),
                    "cameras:=" + ("true" if args.cameras else "false"),
                    "camera_view:=" + ("true" if args.camera_view else "false"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            rclpy.init()
            node = RosEnvironment(
                task,
                args.head_image_topic,
                cameras=args.cameras,
                camera_timeout=args.camera_timeout,
            )
            deadline = time.monotonic() + 90
            while not node.ready():
                if simulator.poll() is not None:
                    raise RuntimeError(
                        "Simulator exited during startup; see simulator.log"
                    )
                if time.monotonic() >= deadline:
                    missing = node.required - node.received.keys()
                    raise RuntimeError(
                        f"Simulation readiness timed out; missing observations: {sorted(missing)}; see simulator.log"
                    )
                rclpy.spin_once(node, timeout_sec=0.1)
            evaluator = Evaluator(task, node.world.sim_time)
            episode_started = time.monotonic()
            last_camera_bundle = episode_started
            next_action = episode_started
            previous_phase = None
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.02)
                if simulator.poll() is not None:
                    raise RuntimeError("Simulator exited during the episode")
                if not node.fresh():
                    raise RuntimeError(
                        f"Simulation observations stopped arriving: {node.stale_sources()}"
                    )
                observation_ready = node.observation_ready()
                now = time.monotonic()
                if observation_ready:
                    last_camera_bundle = now
                elif now - last_camera_bundle >= args.camera_timeout:
                    raise RuntimeError(
                        f"Camera bundle unavailable for {args.camera_timeout:g} wall seconds; "
                        "synchronized images or their timestamped transforms are missing"
                    )
                action = policy.poll()
                if action is not None:
                    node.apply(action)
                    if policy.phase != previous_phase:
                        node.get_logger().info(f"Policy phase: {policy.phase}")
                        previous_phase = policy.phase
                    next_action = time.monotonic() + 0.1
                if (
                    abs(node.get_clock().now().nanoseconds * 1e-9 - node.world.sim_time)
                    > 0.5
                ):
                    raise RuntimeError(
                        "ROS simulation clock diverged from evaluation state timestamps"
                    )
                status = evaluator.update(node.world)
                if status is not None:
                    result["status"] = status
                    break
                if time.monotonic() - episode_started >= args.wall_timeout:
                    result["status"] = "wall_timeout"
                    break
                if (
                    policy.pending is None
                    and time.monotonic() >= next_action
                    and observation_ready
                ):
                    policy.act(node.observation())
            else:
                raise RuntimeError("ROS context shut down during evaluation")
        except Exception as error:
            result["reason"] = f"{type(error).__name__}: {error}"
        finally:
            if evaluator is not None:
                result["metrics"] = evaluator.metrics
            if node is not None:
                if rclpy.ok():
                    node.stop()
                    rclpy.spin_once(node, timeout_sec=0.1)
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if policy is not None:
                policy.close()
            if simulator is not None and simulator.poll() is None:
                simulator.send_signal(signal.SIGINT)
                try:
                    simulator.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    simulator.terminate()
            result["wall_seconds"] = time.monotonic() - started
            write_json(args.output, result)
    return 0 if result["status"] == "success" else 1


def main():
    tasks = load_tasks(
        Path(get_package_share_directory("aha_sim_tasks")) / "config/tasks.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *tasks], default="all")
    parser.add_argument("--policy", default="scripted")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes per task; each starts a fresh simulator",
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--camera-view",
        action="store_true",
        help="Open head and wrist image panels in the episode's Gazebo partition",
    )
    parser.add_argument(
        "--cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render and observe head RGB-D and both wrist RGB cameras",
    )
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=10,
        help="Wall seconds allowed without camera frames or a complete TF-valid bundle",
    )
    parser.add_argument(
        "--head-image-topic",
        default="",
        help="External legacy head Image topic (requires --no-cameras)",
    )
    parser.add_argument(
        "--wall-timeout",
        type=float,
        default=240,
        help="Wall seconds per episode, excluding startup",
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=87,
        help="Isolated evaluation ROS domain (0..232)",
    )
    parser.add_argument(
        "--policy-ros-domain-id",
        type=int,
        help="Worker ROS domain (0..232); defaults to evaluation domain + 1, wrapping to 0",
    )
    parser.add_argument(
        "--policy-timeout",
        type=float,
        default=30,
        help="Wall seconds allowed for each policy.act call",
    )
    parser.add_argument(
        "--policy-startup-timeout",
        type=float,
        default=120,
        help="Wall seconds allowed for policy import, construction, and reset",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--episode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.camera_view and not args.cameras:
        parser.error("--camera-view requires cameras; remove --no-cameras")
    if args.cameras and args.head_image_topic:
        parser.error(
            "--head-image-topic requires --no-cameras; built-in cameras supply head_image"
        )
    if args.policy_ros_domain_id is None:
        args.policy_ros_domain_id = (args.ros_domain_id + 1) % 233
    if (
        args.episodes < 1
        or not 0 <= args.ros_domain_id <= 232
        or not 0 <= args.policy_ros_domain_id <= 232
        or args.policy_ros_domain_id == args.ros_domain_id
        or any(
            not 0 < timeout < float("inf")
            for timeout in (
                args.wall_timeout,
                args.policy_timeout,
                args.policy_startup_timeout,
                args.camera_timeout,
            )
        )
    ):
        parser.error(
            "episodes and timeouts must be positive; ROS domains must be distinct and in 0..232"
        )
    if args.list_tasks:
        print(
            json.dumps({name: asdict(task) for name, task in tasks.items()}, indent=2)
        )
        return 0
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.episode:
        if args.task == "all":
            parser.error("internal episode requires a single task")
        return run_episode(args, tasks[args.task])
    selected = list(tasks) if args.task == "all" else [args.task]
    report = {
        "policy": args.policy,
        "ros_domain_id": args.ros_domain_id,
        "policy_ros_domain_id": args.policy_ros_domain_id,
        "cameras": args.cameras,
        "camera_view": args.camera_view,
        "camera_timeout": args.camera_timeout,
        "tasks": {name: asdict(tasks[name]) for name in selected},
        "episodes": [],
    }
    run_id = uuid.uuid4().hex
    for name in selected:
        for index in range(1, args.episodes + 1):
            directory = (
                args.output.parent
                / (args.output.stem + "_episodes")
                / run_id
                / f"{name}_{index:03d}"
            )
            directory.mkdir(parents=True)
            episode_result = directory / "result.json"
            environment = dict(os.environ)
            environment.update(
                {
                    "ROS_DOMAIN_ID": str(args.ros_domain_id),
                    "GZ_PARTITION": f"aha-evaluation-{run_id}-{name}-{index}",
                    "ROS_LOG_DIR": str(directory / "ros_logs"),
                    "ROS_HOME": str(directory / "ros_home"),
                    "GZ_SIM_LOG_PATH": str(directory / "gz_logs"),
                }
            )
            command = [
                sys.executable,
                "-m",
                "aha_sim_tasks.runner",
                "--episode",
                "--task",
                name,
                "--policy",
                args.policy,
                "--ros-domain-id",
                str(args.ros_domain_id),
                "--policy-ros-domain-id",
                str(args.policy_ros_domain_id),
                "--policy-timeout",
                str(args.policy_timeout),
                "--policy-startup-timeout",
                str(args.policy_startup_timeout),
                "--output",
                str(episode_result),
                "--wall-timeout",
                str(args.wall_timeout),
                "--head-image-topic",
                args.head_image_topic,
                "--headless" if args.headless else "--no-headless",
                "--cameras" if args.cameras else "--no-cameras",
                "--camera-timeout",
                str(args.camera_timeout),
                *(["--camera-view"] if args.camera_view else []),
            ]
            with (directory / "episode.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process.wait(
                        timeout=args.wall_timeout + args.policy_startup_timeout + 110
                    )
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    stop_group(
                        process,
                        environment["GZ_PARTITION"],
                        other_partitions=(
                            policy_partition(environment["GZ_PARTITION"]),
                        ),
                    )
            if episode_result.is_file():
                result = json.loads(episode_result.read_text())
            else:
                result = {
                    "task_id": name,
                    "policy": args.policy,
                    "status": "error",
                    "reason": "Episode process exited or exceeded its deadline; see episode.log",
                    "metrics": {},
                }
            result.update(episode=index, artifacts=str(directory))
            report["episodes"].append(result)
            successes = sum(
                episode["status"] == "success" for episode in report["episodes"]
            )
            report["summary"] = {
                "successes": successes,
                "episodes": len(report["episodes"]),
                "success_rate": successes / len(report["episodes"]),
            }
            write_json(args.output, report)
            print(f"{name} episode {index}: {result['status']}", flush=True)
    print(f"Results: {args.output}")
    return 0 if report["summary"]["success_rate"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
