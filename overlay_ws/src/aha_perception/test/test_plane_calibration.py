"""plane_calibration recovers a tilted, offset camera (ROS free) and the
helpers of scripts/camera_plane_calibrate.py."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aha_perception.plane_calibration import (
    PlaneFit,
    correction,
    corrected_mount,
    matrix_rpy,
    rotation,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "camera_plane_calibrate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("camera_plane_calibrate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def pose(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def seen_plane(true_camera, model_camera, height):
    """The horizontal plane z = height as the model reports it."""
    n, d = np.array([0.0, 0.0, 1.0]), height
    n_cam = true_camera[:3, :3].T @ n
    d_cam = d - n @ true_camera[:3, 3]
    n_model = model_camera[:3, :3] @ n_cam
    return PlaneFit.of(n_model, d_cam + n_model @ model_camera[:3, 3])


# A camera looking forward and 40 deg down (optical z forward, x right).
LOOK = rpy_matrix(-math.pi / 2 - math.radians(40), 0.0, -math.pi / 2)


@pytest.mark.parametrize("roll, pitch, dz", [(0.0, 0.03, 0.0), (0.02, -0.04, 0.012)])
def test_correction_recovers_the_true_camera(roll, pitch, dz):
    model = pose(LOOK, [0.15, 0.03, 0.80])
    true = pose(rotation(roll, pitch) @ LOOK, [0.15, 0.03, 0.80 + dz])
    plane = seen_plane(true, model, 0.38)
    fix = correction(plane, model[:3, 3], 0.38)
    assert fix.roll == pytest.approx(roll, abs=1e-9)
    assert fix.pitch == pytest.approx(pitch, abs=1e-9)
    assert fix.dz == pytest.approx(dz, abs=1e-9)
    assert fix.distance == pytest.approx(0.80 + dz - 0.38)


def test_a_right_model_needs_no_correction():
    model = pose(LOOK, [0.15, 0.03, 0.80])
    fix = correction(seen_plane(model, model, 0.0), model[:3, 3], 0.0)
    assert (fix.roll, fix.pitch, fix.dz, fix.tilt_deg) == pytest.approx((0, 0, 0, 0), abs=1e-9)


def test_corrected_mount_moves_only_the_camera():
    base_to_parent = pose(rpy_matrix(0.3, 0.5, -0.2), [0.09, 0.0, 0.8])
    parent_to_camera = pose(rpy_matrix(math.pi / 2, 0, 0), [0.0525, 0, 0])
    fix = SimpleNamespace(roll=0.01, pitch=-0.02, dz=0.005)
    mount = corrected_mount(base_to_parent, parent_to_camera, fix)
    before = base_to_parent @ parent_to_camera
    after = base_to_parent @ mount
    assert after[:3, :3] == pytest.approx(rotation(0.01, -0.02) @ before[:3, :3])
    assert after[:3, 3] == pytest.approx(before[:3, 3] + [0, 0, 0.005])


def test_matrix_rpy_round_trip():
    rpy = (0.4, -0.3, 1.2)
    assert matrix_rpy(rpy_matrix(*rpy)) == pytest.approx(rpy)


def test_non_finite_inputs_are_errors():
    with pytest.raises(ValueError):
        PlaneFit.of([0, 0, 1], float("nan"))
    with pytest.raises(ValueError):
        correction(PlaneFit.of([0, 0, 1], 0.0), [0, 0, float("nan")], 0.0)


def test_camera_below_the_plane_is_an_error():
    plane = PlaneFit.of([0, 0, 1], 1.0)
    with pytest.raises(ValueError):
        correction(plane, [0, 0, 0.5], 1.0)


def plane_msg(height, area, normal=(0.0, 0.0, 1.0)):
    n = SimpleNamespace(x=normal[0], y=normal[1], z=normal[2])
    return SimpleNamespace(normal=n, offset=-height, height=height, area=area)


def test_pick_plane_takes_the_largest_level_plane_at_the_reference():
    script = load_script()
    msg = SimpleNamespace(
        planes=[
            plane_msg(0.38, 0.2),
            plane_msg(0.40, 0.5),
            plane_msg(0.75, 2.0),
            plane_msg(0.38, 3.0, normal=(0.0, 1.0, 0.0)),  # a wall
            plane_msg(float("nan"), 9.0),
        ]
    )
    fit = script.pick_plane(msg, 0.38, 0.08)
    assert fit.offset == pytest.approx(0.40)
    assert script.pick_plane(msg, 0.0, 0.08) is None


def test_set_camera_replaces_or_adds_the_entry():
    script = load_script()
    text = "base_height: 0.38\n\n# camera: design values\n\narms:\n  left: {}\n"
    added = script.set_camera(text, [0.05, 0.0, -0.01], [1.57, 0.0, 0.0])
    assert "camera: {xyz: [0.0500, 0.0000, -0.0100], rpy: [1.5700, 0.0000, 0.0000]}\n\narms:" in added
    replaced = script.set_camera(added, [0.06, 0.0, 0.0], [1.5, 0.0, 0.0])
    assert replaced.count("\ncamera:") == 1 and "xyz: [0.0600" in replaced
