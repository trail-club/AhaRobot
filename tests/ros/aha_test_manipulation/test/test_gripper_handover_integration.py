"""gripper_handover.py against astra_controller's dry_run_node standing in
for the right arm_node: the dry run node takes the arm and finger commands
on the arm_node's topics (it rejects other joint names) and echoes them as
the joint states. Both run on a domain of their own (never the robot's), on
this host only; skipped without astra_controller (needs the installed
aha_test_manipulation: colcon test)."""

import os
import subprocess
from pathlib import Path

import pytest
from ament_index_python.packages import PackageNotFoundError, get_package_prefix

TEST_DOMAIN = "75"
# dry_run_node's joint_names: [0] the lift, [1:6] the arm (arm/joint_command),
# [6] the commanded finger (arm/gripper_joint_command), [7] its mirror.
JOINTS = ["joint_r1", *(f"joint_r{i}" for i in range(2, 7)), "joint_r7r", "joint_r7l"]


def executable(package, name):
    try:
        prefix = Path(get_package_prefix(package))
    except PackageNotFoundError:
        pytest.skip(f"{package} is not installed")
    return str(prefix / "lib" / package / name)


@pytest.fixture
def env():
    return {
        **os.environ,
        "ROS_DOMAIN_ID": TEST_DOMAIN,
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
    }


@pytest.fixture
def dry_run(env, tmp_path):
    log = tmp_path / "dry_run_node.log"
    with log.open("w") as out:
        node = subprocess.Popen(
            [
                executable("astra_controller", "dry_run_node"),
                "--ros-args",
                "-r",
                "__ns:=/right",
                "-r",
                "joint_states:=/right/arm/joint_states",
                "-p",
                f"joint_names:=[{','.join(JOINTS)}]",
            ],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    # gripper_handover.py waits only 3 s for a state: start it once they flow
    # (with the type given, echo waits for the topic instead of failing).
    try:
        subprocess.run(
            [
                "ros2",
                "topic",
                "echo",
                "--once",
                "--no-daemon",
                "/right/arm/joint_states",
                "sensor_msgs/msg/JointState",
            ],
            env=env,
            capture_output=True,
            timeout=30,
            check=True,
        )
    except Exception:
        node.kill()
        node.wait()
        raise
    yield node, log
    node.terminate()
    try:
        node.wait(timeout=10)
    except subprocess.TimeoutExpired:
        node.kill()
        node.wait()


def handover(env, *args):
    done = subprocess.run(
        [executable("aha_test_manipulation", "gripper_handover.py"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def test_open_then_close_to_a_width(dry_run, env):
    node, log = dry_run
    assert "final opening: 0.0300" in handover(env, "open", "--open", "0.03")
    assert "estimated object width: 0.0200" in handover(env, "close", "--width", "0.02")
    # It took every arm and finger command.
    assert node.poll() is None, log.read_text()
