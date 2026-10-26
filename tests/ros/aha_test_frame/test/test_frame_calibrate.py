"""ROS-free parts of scripts/frame_calibrate.py."""

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "frame_calibrate.py"
spec = importlib.util.spec_from_file_location("frame_calibrate", SCRIPT)
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)

URDF = """
<robot name="r">
  <joint name="joint_l2" type="revolute"><origin xyz="0.068 0 0" rpy="0 0 -0.18555"/></joint>
  <joint name="joint_l3" type="revolute"><origin xyz="0.29812 0 0" rpy="0 0 -1.3852"/></joint>
  <joint name="joint_r4" type="revolute"><origin xyz="0 0 0" rpy="3.1416 0 0"/></joint>
</robot>
"""


def test_mount_yaw_points_the_upper_arm_forward():
    offset = calibrate.origin_yaw(URDF, "joint_l2")
    # With the Astra mount (+90 deg) the shoulder reads 90 deg - offset back.
    shoulder = -math.pi / 2 - offset
    assert calibrate.mount_yaw(offset, shoulder) == pytest.approx(math.pi / 2)
    assert calibrate.mount_yaw(offset, 0.0) == pytest.approx(0.18555)


def test_forearm_in_line_reads_zero_error():
    offset = calibrate.origin_yaw(URDF, "joint_l3")
    assert calibrate.straight_error(offset, 1.3852) == pytest.approx(0.0)
    assert calibrate.straight_error(offset, 0.0) == pytest.approx(-1.3852)


def test_outward_change_has_the_side_sign():
    assert calibrate.outward_error("left", 0.1, 0.1 + math.pi / 2) == pytest.approx(0.0)
    assert calibrate.outward_error("right", 0.1, 0.1 - math.pi / 2) == pytest.approx(
        0.0
    )
    # A mirrored sign reads 180 deg off.
    assert abs(calibrate.outward_error("left", 0.0, -math.pi / 2)) == pytest.approx(
        math.pi
    )


def test_tilted_origin_is_rejected():
    with pytest.raises(ValueError, match="not a pure z rotation"):
        calibrate.origin_yaw(URDF, "joint_r4")


def test_replace_yaw_keeps_the_rest_of_the_line():
    text = (
        "arms:\n"
        "  left: {shoulder_xyz: [0.03, 0.232, 0.244], yaw: 1.5708}\n"
        "  right: {shoulder_xyz: [0.03, -0.232, 0.222], yaw: -1.5708}  # note\n"
    )
    out = calibrate.replace_yaw(text, "right", 0.5)
    assert "right: {shoulder_xyz: [0.03, -0.232, 0.222], yaw: 0.5000}  # note" in out
    assert "left: {shoulder_xyz: [0.03, 0.232, 0.244], yaw: 1.5708}" in out
    with pytest.raises(ValueError):
        calibrate.replace_yaw("arms: {}\n", "left", 0.0)
