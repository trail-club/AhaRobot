"""Evaluate policies in isolated, freshly started Gazebo episodes."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from ament_index_python.packages import get_package_share_directory

from .codex_policy import CLI_TIMEOUT as CODEX_CLI_TIMEOUT
from .evaluation import Evaluator, load_tasks
from .policy_process import PolicyProcess
from .processes import policy_partition, stop_group
from .stepping import GazeboStepper, run_steps


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
    recorder = None
    with (args.output.parent / "simulator.log").open("w") as log:
        try:
            if getattr(args, "record_video", False):
                from .video import VideoRecorder

                recorder = VideoRecorder(args.output.parent, task.task_id, args.policy)
            policy = PolicyProcess(
                args.policy,
                task.task_id,
                task.instruction,
                args.policy_ros_domain_id,
                args.output.parent,
                os.environ["GZ_PARTITION"],
                startup_timeout=args.policy_startup_timeout,
                action_timeout=args.policy_timeout,
                step_duration=getattr(args, "step_duration", 0),
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
                    "ground_truth:="
                    + ("true" if args.policy == "ground_truth_ik" else "false"),
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
                ground_truth=args.policy == "ground_truth_ik",
                video_recorder=recorder,
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
            stepper = None
            if getattr(args, "step_duration", 0):
                stepper = GazeboStepper()
                stepper.pause()
                # Consume messages already in flight when Gazebo paused.
                settle_deadline = time.monotonic() + 0.2
                while time.monotonic() < settle_deadline:
                    rclpy.spin_once(node, timeout_sec=0.01)
            evaluator = Evaluator(task, node.world.sim_time)
            if recorder is not None:
                recorder.start(node.world.sim_time)
            if stepper is not None:
                result["status"] = run_steps(
                    args, node, policy, simulator, evaluator, stepper, recorder=recorder
                )
            else:
                episode_started = time.monotonic()
                last_camera_bundle = episode_started
                next_action = episode_started
                previous_phase = None
                while rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.02)
                    if recorder is not None:
                        recorder.capture(node.world.sim_time)
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
                        if recorder is not None:
                            recorder.action(action, policy.phase)
                        node.apply(action)
                        if policy.phase != previous_phase:
                            node.get_logger().info(f"Policy phase: {policy.phase}")
                            previous_phase = policy.phase
                        next_action = time.monotonic() + 0.1
                    if (
                        abs(
                            node.get_clock().now().nanoseconds * 1e-9
                            - node.world.sim_time
                        )
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
            if recorder is not None:
                try:
                    final_time = (
                        node.world.sim_time
                        if node is not None and node.world is not None
                        else None
                    )
                    result["video"] = recorder.close(final_time, result["status"])
                except Exception as error:
                    result.setdefault(
                        "reason",
                        f"Video finalization error: {type(error).__name__}: {error}",
                    )
                    result["status"] = "error"
            if node is not None:
                if rclpy.ok():
                    try:
                        node.stop()
                        rclpy.spin_once(node, timeout_sec=0.1)
                    except Exception as error:
                        # A malformed state callback may still be pending. Keep
                        # its original failure report and complete all cleanup.
                        result.setdefault(
                            "reason", f"Shutdown error: {type(error).__name__}: {error}"
                        )
                        result["status"] = "error"
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
    parser.add_argument(
        "--policy",
        default="scripted",
        help="scripted, noop, codex, ground_truth_ik, or module:factory",
    )
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
        "--record-video",
        action="store_true",
        help="Save a 10 fps simulation-time H.264 video with camera views and policy notes",
    )
    parser.add_argument(
        "--cameras",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render and observe head RGB-D and both wrist RGB cameras "
        "(default: on, off when --head-image-topic is given)",
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
        help="External legacy head Image topic (disables built-in cameras; "
        "cannot be combined with --cameras)",
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
        help="Worker domain (0..232): ground_truth_ik shares the evaluation domain; other policies default to evaluation + 1, wrapping to 0",
    )
    parser.add_argument(
        "--policy-timeout",
        type=float,
        help="Wall seconds allowed for each policy.act call. "
        "Default: 150 for codex (its CLI call allows 120), 30 for other policies",
    )
    parser.add_argument(
        "--policy-startup-timeout",
        type=float,
        default=120,
        help="Wall seconds allowed for policy import, construction, and reset",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        help="Simulation seconds per action, in multiples of 0.1; inference is paused. "
        "Default: 2 for codex, 0 (continuous) for other policies",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--episode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step_duration is None:
        args.step_duration = 2.0 if args.policy == "codex" else 0.0
    if args.policy_timeout is None:
        args.policy_timeout = 150.0 if args.policy == "codex" else 30.0
    elif args.policy == "codex" and args.policy_timeout <= CODEX_CLI_TIMEOUT:
        parser.error(
            f"codex requires --policy-timeout above {CODEX_CLI_TIMEOUT} seconds, "
            "the Codex CLI call limit"
        )
    if (
        not math.isfinite(args.step_duration)
        or args.step_duration < 0
        or abs(args.step_duration * 10 - round(args.step_duration * 10)) > 1e-6
    ):
        parser.error("--step-duration must be zero or a positive multiple of 0.1")
    if args.policy == "ground_truth_ik" and args.step_duration:
        parser.error("ground_truth_ik requires continuous simulation")
    if args.cameras is None:
        args.cameras = not args.head_image_topic and args.policy != "ground_truth_ik"
    if args.policy == "ground_truth_ik":
        if args.task != "pick_apple":
            parser.error("ground_truth_ik supports --task pick_apple only")
        if args.cameras or args.head_image_topic or args.camera_view:
            parser.error(
                "ground_truth_ik uses Gazebo truth only; camera options are unavailable"
            )
    if args.camera_view and not args.cameras:
        parser.error("--camera-view requires cameras; remove --no-cameras")
    if args.record_video and not args.cameras:
        parser.error("--record-video requires all built-in cameras")
    if args.cameras and args.head_image_topic:
        parser.error(
            "--head-image-topic cannot be combined with --cameras; "
            "built-in cameras supply head_image"
        )
    if args.policy == "codex" and (not args.cameras or not args.step_duration):
        parser.error(
            "codex requires all built-in cameras and a positive --step-duration"
        )
    if args.policy_ros_domain_id is None:
        args.policy_ros_domain_id = (
            args.ros_domain_id
            if args.policy == "ground_truth_ik"
            else (args.ros_domain_id + 1) % 233
        )
    if (
        args.episodes < 1
        or not 0 <= args.ros_domain_id <= 232
        or not 0 <= args.policy_ros_domain_id <= 232
        or (
            (args.policy_ros_domain_id == args.ros_domain_id)
            != (args.policy == "ground_truth_ik")
        )
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
            "episodes and timeouts must be positive; ROS domains must be in 0..232; ground_truth_ik shares the evaluation domain, other policies require a distinct domain"
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
        "observation_source": (
            "gazebo_ground_truth" if args.policy == "ground_truth_ik" else "sensors"
        ),
        "ros_domain_id": args.ros_domain_id,
        "policy_ros_domain_id": args.policy_ros_domain_id,
        "cameras": args.cameras,
        "camera_view": args.camera_view,
        "record_video": args.record_video,
        "camera_timeout": args.camera_timeout,
        "step_duration": args.step_duration,
        "tasks": {name: asdict(tasks[name]) for name in selected},
        "episodes": [],
    }
    if args.policy == "codex":
        from .codex_policy import MODEL, REASONING_EFFORT

        report.update(model=MODEL, reasoning_effort=REASONING_EFFORT)
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
                "--step-duration",
                str(args.step_duration),
                *(["--camera-view"] if args.camera_view else []),
                *(["--record-video"] if args.record_video else []),
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
