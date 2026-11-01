"""Shared parts of the Gazebo trial scripts (tests/manipulation/place_sim,
tests/orchestration/place_llm): start and stop the test bed sim, the CSV of
the trial rows, fold the arms, wait for the trajectory controllers and stop a
place goal that does not finish.

The caller's rclpy node must not be spun by another executor (``fold_arms``
spins it through the SimHolder while waiting).
"""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import time
from pathlib import Path

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState

ARENA_FURNITURE = "config/furniture/rcjo2026_arena.yaml"  # in aha_perception's share
TEST_XACRO = (
    "$(ros2 pkg prefix aha_test_description)"
    "/share/aha_test_description/urdf/aha_robot_test.urdf.xacro"
)
# Folded at the lowest lift the hand hits the base; at this lift (m) the
# hand is about 0.43 m high, below a table in front of the robot.
FOLD_LIFT = 0.3
STOP_GRACE_S = 2.0  # wall s for the children after the shell exits
PLACE_TIMEOUT_S = 900.0  # wall s for a PlaceObject result
CANCEL_TIMEOUT_S = 60.0  # wall s for a canceled goal to stop


class PlaceServerBusy(RuntimeError):
    """The place server still runs a goal and rejects new ones (this sim's
    later trials are not valid)."""


def launch_sim(
    log: Path,
    world: str,
    spawn,
    depth_noise: str = "d435i_filtered",
    furniture: bool = False,
    extra: str = "",
) -> subprocess.Popen:
    """Headless sim with the test bed robot (aha_test_description), plane
    detection, the test bed MoveIt and place server (aha_test_manipulation)
    and head_look_at.py, in a new process group. ``furniture`` adds world_tf
    and the arena furniture config; ``extra`` more sim.launch.py arguments."""
    x, y, yaw = spawn
    arguments = (
        f"use_perception:=true planes:=true depth_noise:={depth_noise} "
        f"world:={world} spawn_x:={x} spawn_y:={y} spawn_yaw:={yaw:.4f} "
    )
    if furniture:
        arguments += (
            "world_tf:=true furniture_config:=$(ros2 pkg prefix aha_perception)"
            f"/share/aha_perception/{ARENA_FURNITURE} "
        )
    if extra:
        arguments += extra + " "
    command = (
        "ros2 launch aha_bringup sim.launch.py headless:=true rviz:=false "
        f"{arguments}robot_xacro:={TEST_XACRO} & "
        "ros2 launch aha_test_manipulation manip.launch.py rviz:=false & "
        "ros2 run aha_perception head_look_at.py --ros-args --params-file "
        "$(ros2 pkg prefix aha_perception)/share/aha_perception/config/head.yaml & wait"
    )
    return subprocess.Popen(
        ["bash", "-c", command],
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_sim(process: subprocess.Popen):
    """SIGINT to the process group, then SIGKILL to what is left of it (the
    launched processes can outlive the shell)."""
    for sig, wait in ((signal.SIGINT, 30), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(wait)
        except subprocess.TimeoutExpired:
            continue
        time.sleep(STOP_GRACE_S)


def open_csv(path: Path, fields: list[str]):
    """Append stream and writer; a file with other columns is rewritten first.

    Rows as long as the current columns are taken as written with them (rows
    appended after a column change), the others with the file's header.
    Values of columns that are gone are dropped; the file is replaced only
    once the rewrite is complete."""
    if path.exists():
        with path.open(newline="") as stream:
            header, *rows = list(csv.reader(stream)) or [[]]
        if header != fields:
            records = []
            for row in rows:
                names = fields if len(row) == len(fields) else header
                records.append({k: v for k, v in zip(names, row) if k in fields})
            rewritten = path.with_name(path.name + ".tmp")
            with rewritten.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, restval="")
                writer.writeheader()
                writer.writerows(records)
            os.replace(rewritten, path)
    new = not path.exists()
    stream = path.open("a", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, restval="")
    if new:
        writer.writeheader()
    return stream, writer


def cancel_goal(holder, handle, future, timeout: float = CANCEL_TIMEOUT_S):
    """Cancel a place goal that ran too long and wait for its (CANCELED)
    result ``future``; PlaceServerBusy if the goal keeps running."""
    handle.cancel_goal_async()
    if not holder.spin_until(future.done, timeout):
        raise PlaceServerBusy("the place goal timed out and ignored the cancel")


def wait_controllers(node, controllers, timeout: float) -> bool:
    """Whether the FollowJointTrajectory servers of ``controllers`` are up."""
    ok = True
    for controller in controllers:
        client = ActionClient(
            node, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory"
        )
        ok = ok and client.wait_for_server(timeout_sec=timeout)
        client.destroy()
    return ok


def fold_arms(node, holder):
    """Fold both arms at FOLD_LIFT (or higher), then raise them to carry.

    At the spawn pose (home) the hands reach under a table in front of the
    robot; lifting them there pushes the robot. ``holder`` is the
    sim_held_object.SimHolder of ``node``."""
    joints = {}
    sub = node.create_subscription(
        JointState,
        "/joint_states",
        lambda m: joints.update(zip(m.name, m.position)),
        10,
    )
    try:
        holder.spin_until(lambda: "joint_r1" in joints, 30.0)
    finally:
        node.destroy_subscription(sub)
    for arm in ("right", "left"):
        lift = f"joint_{arm[0]}1"
        folded = holder.named_state(arm, "carry")
        folded[lift] = max(joints.get(lift, 0.0), FOLD_LIFT)
        if not (holder.move_joints(arm, folded) and holder.move_named(arm, "carry")):
            raise RuntimeError(f"could not fold the {arm} arm")
