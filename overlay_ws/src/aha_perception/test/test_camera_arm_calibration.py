"""camera_arm_calibration recovers the camera's horizontal offset and yaw
from gripper points (ROS free) and the helpers of
scripts/camera_arm_calibrate.py."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from aha_perception.camera_arm_calibration import (
    HorizontalCorrection,
    corrected_mount,
    facing_point,
    horizontal_correction,
    nearest,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "camera_arm_calibrate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("camera_arm_calibrate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # The script's dataclass resolves its string annotations via sys.modules.
    sys.modules[spec.name] = module
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


def yaw_motion(dyaw, pivot, shift):
    """Rotation by dyaw about the vertical through pivot, then shift."""
    R = rpy_matrix(0.0, 0.0, dyaw)
    p = np.array([pivot[0], pivot[1], 0.0])
    return pose(R, p - R @ p + np.array([shift[0], shift[1], 0.0]))


def apply(T, points):
    points = np.asarray(points, float)
    return points @ T[:3, :3].T + T[:3, 3]


# Head chain as on the frame: link_head_tilt pitched down, camera in front.
BASE_TO_PARENT = pose(rpy_matrix(0.0, math.radians(18), 0.0), [0.12, 0.0, 0.78])
PARENT_TO_CAMERA = pose(rpy_matrix(math.pi / 2, 0.0, 0.0), [0.0476, -0.0146, 0.0])
CAMERA = BASE_TO_PARENT @ PARENT_TO_CAMERA
# Gripper tips over the reach of both arms (base_footprint, m).
GRIPPERS = np.array(
    [
        [0.35, 0.20, 0.45],
        [0.45, 0.10, 0.40],
        [0.40, -0.15, 0.42],
        [0.30, -0.25, 0.50],
        [0.50, -0.05, 0.38],
        [0.38, 0.02, 0.55],
    ]
)


def seen_through_model(true_camera, model_camera, points):
    """The true points as the modelled camera reports them."""
    return apply(model_camera @ np.linalg.inv(true_camera), points)


@pytest.mark.parametrize(
    "dyaw, shift", [(0.0, (0.0, 0.0)), (math.radians(2.0), (0.012, -0.008))]
)
def test_correction_recovers_the_true_camera(dyaw, shift):
    pivot = CAMERA[:3, 3]
    true = yaw_motion(dyaw, pivot, shift) @ CAMERA
    seen = seen_through_model(true, CAMERA, GRIPPERS)
    fix = horizontal_correction(seen, GRIPPERS, pivot)
    assert fix.dyaw == pytest.approx(dyaw, abs=1e-12)
    assert (fix.dx, fix.dy) == pytest.approx(shift, abs=1e-12)
    assert fix.rms == pytest.approx(0.0, abs=1e-12)
    assert fix.dz == pytest.approx(np.zeros(len(GRIPPERS)), abs=1e-12)
    mount = corrected_mount(BASE_TO_PARENT, PARENT_TO_CAMERA, fix)
    assert BASE_TO_PARENT @ mount == pytest.approx(true, abs=1e-12)


def test_the_motion_does_not_depend_on_the_pivot():
    true = yaw_motion(0.03, CAMERA[:3, 3], (0.01, 0.02)) @ CAMERA
    seen = seen_through_model(true, CAMERA, GRIPPERS)
    at_camera = horizontal_correction(seen, GRIPPERS, CAMERA[:3, 3])
    at_origin = horizontal_correction(seen, GRIPPERS, (0.0, 0.0))
    assert at_origin.motion() == pytest.approx(at_camera.motion(), abs=1e-12)
    assert at_origin.dyaw == pytest.approx(at_camera.dyaw)
    assert (at_origin.dx, at_origin.dy) != pytest.approx((at_camera.dx, at_camera.dy))


def test_noisy_clicks_give_a_close_correction():
    rng = np.random.default_rng(0)
    pivot = CAMERA[:3, 3]
    dyaw, shift = math.radians(-1.5), (-0.015, 0.006)
    true = yaw_motion(dyaw, pivot, shift) @ CAMERA
    seen = seen_through_model(true, CAMERA, GRIPPERS)
    seen = seen + rng.normal(0.0, 0.003, seen.shape)
    fix = horizontal_correction(seen, GRIPPERS, pivot)
    assert fix.dyaw == pytest.approx(dyaw, abs=math.radians(1.0))
    assert (fix.dx, fix.dy) == pytest.approx(shift, abs=0.004)
    assert 0.0005 < fix.rms < 0.006
    assert fix.residuals.shape == (len(GRIPPERS), 2)


def test_the_correction_keeps_tilt_and_height():
    fix = HorizontalCorrection(
        dx=0.02, dy=-0.01, dyaw=0.05, pivot=CAMERA[:2, 3], residuals=None, dz=None
    )
    mount = corrected_mount(BASE_TO_PARENT, PARENT_TO_CAMERA, fix)
    after = BASE_TO_PARENT @ mount
    assert after[2, 3] == pytest.approx(CAMERA[2, 3])
    assert after[:2, 3] == pytest.approx(CAMERA[:2, 3] + [0.02, -0.01])
    # Each camera axis keeps its angle to the vertical.
    assert after[2, :3] == pytest.approx(CAMERA[2, :3])


@pytest.mark.parametrize(
    "seen, model",
    [
        ([[0.4, 0.0, 0.4]], [[0.4, 0.0, 0.4]]),  # one pair
        ([[0.4, 0.0, 0.4], [0.4, 0.2, 0.4]], [[0.4, 0.0, 0.4]]),  # count mismatch
        ([[0.4, 0.0], [0.4, 0.2]], [[0.4, 0.0], [0.4, 0.2]]),  # not 3D
        ([[0.4, 0.0, 0.4], [0.4, float("nan"), 0.4]], [[0.4, 0, 0.4], [0.4, 0.2, 0.4]]),
        (
            [[0.4, 0.0, 0.4], [0.41, 0.0, 0.6]],
            [[0.4, 0, 0.4], [0.41, 0, 0.6]],
        ),  # spread
    ],
)
def test_degenerate_inputs_are_errors(seen, model):
    with pytest.raises(ValueError):
        horizontal_correction(seen, model, (0.17, 0.0))


def test_invalid_pivot_is_an_error():
    with pytest.raises(ValueError):
        horizontal_correction(GRIPPERS, GRIPPERS, (float("inf"), 0.0))


def test_facing_point_moves_toward_the_camera():
    point = facing_point([0.5, 0.0, 0.4], [0.1, 0.0, 0.8], 0.01)
    assert point == pytest.approx([0.49, 0.0, 0.4])
    assert facing_point([0.5, 0.1, 0.4], [0.1, 0.0], 0.0) == pytest.approx(
        [0.5, 0.1, 0.4]
    )


def test_nearest_picks_the_only_close_link():
    links = {"link_ree": [0.4, -0.2, 0.4], "link_lee": [0.4, 0.2, 0.4]}
    assert nearest([0.42, -0.19, 0.43], links, 0.1) == "link_ree"
    with pytest.raises(ValueError):
        nearest([0.4, 0.0, 0.8], links, 0.1)  # none close
    with pytest.raises(ValueError):
        nearest([0.4, 0.0, 0.4], links, 0.3)  # both close


def test_head_moved():
    script = load_script()
    assert not script.head_moved(CAMERA, CAMERA)
    nudged = CAMERA.copy()
    nudged[0, 3] += 0.001
    assert not script.head_moved(CAMERA, nudged)
    turned = pose(rpy_matrix(0.0, 0.0, math.radians(1.0)), [0, 0, 0]) @ CAMERA
    assert script.head_moved(CAMERA, turned)
