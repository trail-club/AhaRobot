"""arm_cloud_calibration recovers the arm offsets from synthetic point clouds
of the installed frame model (aha_test_frame frame_table_202610.yaml), and
the ROS-free helpers of scripts/arm_cloud_calibrate.py."""

import importlib.util
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from aha_test_frame.frame import frame_robot, load_frame
from aha_test_frame.arm_cloud_calibration import (
    ArmModel,
    Snapshot,
    UrdfTree,
    camera_motion,
    cloud_normals,
    fit,
    free_parameters,
    prepare,
    read_stl,
    sample_surface,
    support_height,
    voxel_filter,
)
from aha_test_frame.camera_arm_calibration import HorizontalCorrection

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "arm_cloud_calibrate.py"
CAMERA_FRAME = "camera_color_optical_frame"
ARM = ("joint_r2", "joint_r3", "joint_r4", "joint_r5", "joint_r6")
# Right arm over the front table, gripper in the camera's view; the upper arm
# is mostly out of view, as on the robot.
POSES = [
    (0.8, -0.3, -0.3, 0.5, 0.2),
    (1.0, 0.0, 0.0, 0.8, 0.0),
    (1.2, -0.9, 0.4, 0.6, -0.2),
    (1.4, -0.6, -0.2, 0.5, 0.0),
    (1.6, -1.5, 0.0, 0.7, 0.3),
]
HEAD_TILT = 0.45  # rad, looking down at the table
TABLE = 0.346  # m, front table top
# D435i color camera (the cloud is in camera_color_optical_frame).
FX, WIDTH, HEIGHT = 615.0, 640, 480
BLOCK = 4  # px per rendered depth sample


def load_script():
    spec = importlib.util.spec_from_file_location("arm_cloud_calibrate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frame_urdf(config: dict) -> str:
    import xacro
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory("aha_description"))
    urdf = xacro.process_file(
        str(share / "urdf" / "aha_robot.urdf.xacro"), mappings={"sim": "false"}
    ).toxml()
    return frame_robot(urdf, config)


@pytest.fixture(scope="module")
def config():
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory("aha_test_frame"))
    config = load_frame(share / "config" / "frame_table_202610.yaml")
    # The poses above are in view with this right arm, whatever the robot's
    # calibration in the config.
    right = {"shoulder_xyz": [0.03, -0.232, 0.222], "yaw": -1.5239}
    return dict(config, arms=dict(config["arms"], right=right))


@pytest.fixture(scope="module")
def urdf(config):
    return frame_urdf(config)


@pytest.fixture(scope="module")
def model(urdf):
    return ArmModel.from_urdf(urdf, "right")


@pytest.fixture(scope="module")
def truth(urdf):
    """Densely sampled model to render the clouds from."""
    return ArmModel.from_urdf(urdf, "right", spacing=0.0015, seed=1)


@pytest.fixture(scope="module")
def camera(model):
    return model.tree.poses([CAMERA_FRAME], {"joint_head_tilt": HEAD_TILT})[
        CAMERA_FRAME
    ]


def render(truth, joints, params, camera, rng, noise=0.0015, held=True):
    """Cloud (target frame) of the arm posed with the true ``params``, the
    table, a held box and outliers, as the camera at ``camera`` sees it:
    nearest surface per pixel block in the image, depth noise along the ray."""
    surface, _, _ = truth.surface(joints, params)
    grid = np.mgrid[0.3:0.9:0.003, -0.4:0.3:0.003].reshape(2, -1).T
    parts = [surface, np.column_stack([grid, np.full(len(grid), TABLE)])]
    if held:
        ee = truth.tree.poses(["link_ree"], joints)["link_ree"]
        box = rng.uniform(-0.02, 0.02, (3000, 3)) + [0.0, 0.0, 0.03]
        parts.append(box @ ee[:3, :3].T + ee[:3, 3])
    points = np.concatenate(parts)
    inverse = np.linalg.inv(camera)
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    local = local[local[:, 2] > 0.1]
    u = FX * local[:, 0] / local[:, 2] + WIDTH / 2
    v = FX * local[:, 1] / local[:, 2] + HEIGHT / 2
    inside = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    local, u, v = local[inside], u[inside], v[inside]
    pixel = (v // BLOCK).astype(int) * (WIDTH // BLOCK) + (u // BLOCK).astype(int)
    order = np.lexsort((local[:, 2], pixel))
    pixel, local = pixel[order], local[order]
    local = local[np.r_[True, pixel[1:] != pixel[:-1]]]
    local *= (1.0 + rng.normal(0.0, noise, len(local)) / local[:, 2])[:, None]
    cloud = local @ camera[:3, :3].T + camera[:3, 3]
    outliers = rng.uniform([0.2, -0.5, 0.3], [0.9, 0.3, 0.8], (300, 3))
    return np.concatenate([cloud, outliers])


def snapshot(truth, q, params, camera, rng, **kwargs) -> Snapshot:
    """A snapshot seen through the modelled ``camera`` while the true camera
    is off by the camera parameters in ``params``."""
    joints = dict(zip(ARM, q))
    true_camera = camera_motion(params, camera[:3, 3]) @ camera
    cloud = render(truth, joints, params, true_camera, rng, **kwargs)
    seen = camera @ np.linalg.inv(true_camera)
    return Snapshot(cloud @ seen[:3, :3].T + seen[:3, 3], joints, camera[:3, 3])


@pytest.fixture(scope="module")
def offset_snapshots(model, truth, camera):
    rng = np.random.default_rng(3)
    true = {"shoulder": 0.07, "elbow": -0.12}
    return [prepare(model, snapshot(truth, q, true, camera, rng)) for q in POSES]


def test_shoulder_and_elbow_are_recovered_from_several_poses(model, offset_snapshots):
    result = fit(model, offset_snapshots)
    assert result.params["shoulder"] == pytest.approx(0.07, abs=0.003)
    assert result.params["elbow"] == pytest.approx(-0.12, abs=0.003)
    assert result.params["wrist"] == 0.0 and result.params["camera_yaw"] == 0.0
    assert result.reliable, result.warnings
    assert result.leave_one_out["elbow"] < 0.01
    for stats in result.snapshots:
        assert stats.support == pytest.approx(TABLE, abs=0.003)
        assert stats.rms_after < 0.6 * stats.rms_before
        assert stats.inliers_after > 100


def test_one_pose_is_flagged(model, offset_snapshots):
    result = fit(model, offset_snapshots[2:3], leave_one_out=False)
    assert not result.reliable
    assert any("one pose" in w for w in result.warnings)
    assert result.leave_one_out is None
    # The fit still moves the model onto the cloud.
    assert result.snapshots[0].rms_after < 0.6 * result.snapshots[0].rms_before


def test_camera_and_shoulder_are_not_separable(model, offset_snapshots):
    result = fit(
        model,
        offset_snapshots[:3],
        ("shoulder", "elbow", "camera"),
        leave_one_out=False,
    )
    assert {"shoulder", "camera_yaw"} <= set(result.unobservable)
    assert "elbow" not in result.unobservable
    assert not result.reliable


def test_camera_correction_with_a_known_shoulder(model, truth, camera):
    rng = np.random.default_rng(4)
    true = {"elbow": -0.05, "camera_x": 0.012, "camera_y": -0.008, "camera_yaw": 0.03}
    snaps = [prepare(model, snapshot(truth, q, true, camera, rng)) for q in POSES[1:4]]
    result = fit(model, snaps, ("elbow", "camera"), leave_one_out=False)
    for name, value in true.items():
        tolerance = (
            0.002 if name.startswith("camera_") and name != "camera_yaw" else 0.004
        )
        assert result.params[name] == pytest.approx(value, abs=tolerance), name


def test_wrist_offset(model, truth, camera):
    rng = np.random.default_rng(5)
    true = {"shoulder": -0.03, "elbow": 0.04, "wrist": 0.08}
    snaps = [prepare(model, snapshot(truth, q, true, camera, rng)) for q in POSES]
    result = fit(model, snaps, ("shoulder", "elbow", "wrist"), leave_one_out=False)
    for name, value in true.items():
        # The wrist is the least observable (only the hand is in view).
        assert result.params[name] == pytest.approx(value, abs=0.008), name


def test_arm_height(model, truth, camera):
    rng = np.random.default_rng(6)
    true = {"elbow": 0.15, "height": -0.012}
    snaps = [prepare(model, snapshot(truth, q, true, camera, rng)) for q in POSES]
    result = fit(model, snaps, ("shoulder", "elbow", "height"), leave_one_out=False)
    assert result.params["height"] == pytest.approx(-0.012, abs=0.002)
    assert result.params["elbow"] == pytest.approx(0.15, abs=0.004)
    assert result.params["shoulder"] == pytest.approx(0.0, abs=0.004)


def test_parameters_match_the_frame_config(config, model):
    """A fitted shoulder delta goes to ``yaw``, the elbow and wrist to
    ``joint_offsets``: the URDF from the updated config poses the arm like
    the model with the parameters."""
    right = dict(config["arms"]["right"])
    right["yaw"] = float(right["yaw"]) + 0.07
    right["joint_offsets"] = {"joint_r3": -0.12, "joint_r4": 0.05}
    right["shoulder_xyz"] = [
        *right["shoulder_xyz"][:2],
        right["shoulder_xyz"][2] - 0.01,
    ]
    updated = UrdfTree(frame_urdf(dict(config, arms=dict(config["arms"], right=right))))
    joints = dict(zip(ARM, POSES[2]))
    expected = updated.poses(model.links, joints)
    poses = model.link_poses(
        joints, {"shoulder": 0.07, "elbow": -0.12, "wrist": 0.05, "height": -0.01}
    )
    for link in model.links:
        assert poses[link] == pytest.approx(expected[link], abs=1e-5), link


def test_camera_motion_is_the_horizontal_correction():
    fix = HorizontalCorrection(
        dx=0.01,
        dy=-0.02,
        dyaw=0.05,
        pivot=np.array([0.17, 0.03]),
        residuals=np.zeros((0, 2)),
        dz=np.zeros(0),
    )
    params = {"camera_x": 0.01, "camera_y": -0.02, "camera_yaw": 0.05}
    assert camera_motion(params, [0.17, 0.03, 0.77]) == pytest.approx(fix.motion())


def test_free_parameters():
    assert free_parameters(["elbow", "shoulder"]) == ("shoulder", "elbow")
    assert free_parameters(["camera"]) == ("camera_x", "camera_y", "camera_yaw")
    with pytest.raises(ValueError):
        free_parameters(["knee"])
    with pytest.raises(ValueError):
        free_parameters([])


def test_read_stl_binary_and_ascii(tmp_path):
    triangle = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
    binary = bytearray(80) + struct.pack("<I", 1) + struct.pack("<3f", 0, 0, 1)
    for vertex in triangle:
        binary += struct.pack("<3f", *vertex)
    binary += b"\0\0"
    (tmp_path / "b.stl").write_bytes(bytes(binary))
    ascii_text = "solid t\nfacet normal 0 0 1\nouter loop\n"
    ascii_text += "".join(f"vertex {x} {y} {z}\n" for x, y, z in triangle)
    ascii_text += "endloop\nendfacet\nendsolid t\n"
    (tmp_path / "a.stl").write_text(ascii_text)
    for name in ("b.stl", "a.stl"):
        assert read_stl(tmp_path / name) == pytest.approx(np.array([triangle]))
    (tmp_path / "bad.stl").write_text("nothing")
    with pytest.raises(ValueError):
        read_stl(tmp_path / "bad.stl")


def test_samples_are_area_weighted_and_deterministic():
    small = [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]
    large = [[0, 0, 1], [0.03, 0, 1], [0, 0.03, 1]]  # 9 times the area
    triangles = np.array([small, large], float)
    points, normals = sample_surface(triangles, 0.0005, np.random.default_rng(0))
    again, _ = sample_surface(triangles, 0.0005, np.random.default_rng(0))
    assert points == pytest.approx(again)
    assert np.mean(points[:, 2] > 0.5) == pytest.approx(0.9, abs=0.03)
    assert np.abs(normals[:, 2]) == pytest.approx(np.ones(len(normals)))


def test_voxel_filter_and_normals():
    rng = np.random.default_rng(0)
    plane = np.column_stack([rng.uniform(0, 0.1, (5000, 2)), np.zeros(5000)])
    plane = np.vstack([plane, [[np.nan, 0.0, 0.0]]])
    points = voxel_filter(plane, 0.005)
    assert 300 < len(points) <= 400 and np.all(np.isfinite(points))
    normals, valid = cloud_normals(points)
    assert valid.mean() > 0.9
    assert np.abs(normals[valid, 2]) == pytest.approx(np.ones(valid.sum()), abs=1e-6)


def test_support_height_is_the_densest_height_under_the_arm():
    rng = np.random.default_rng(1)
    arm = np.column_stack(
        [rng.uniform(0.4, 0.6, 500), np.zeros(500), np.full(500, 0.45)]
    )
    table = np.column_stack(
        [rng.uniform([0.3, -0.2], [0.7, 0.2], (4000, 2)), np.full(4000, 0.346)]
    )
    floor = np.column_stack([rng.uniform(0.3, 0.7, (1000, 2)), np.zeros(1000)])
    assert support_height(np.vstack([table, floor, arm]), arm) == pytest.approx(0.346)
    assert support_height(arm[:10], arm) is None


def test_script_helpers():
    script = load_script()
    text = (
        "arms:\n"
        "  left: {shoulder_xyz: [0.03, 0.232, 0.244], yaw: 1.5708}\n"
        "  right: {shoulder_xyz: [0.03, -0.232, 0.222], yaw: -1.5239, "
        "joint_offsets: {joint_r3: -0.0100}}\n"
    )
    params = {"shoulder": 0.07, "elbow": -0.12, "wrist": 0.0}
    out = script.updated_config(text, "right", ("shoulder", "elbow"), params)
    right = yaml.safe_load(out)["arms"]["right"]
    assert right["yaw"] == pytest.approx(-1.4539)
    assert right["joint_offsets"] == {"joint_r3": pytest.approx(-0.13)}
    assert yaml.safe_load(out)["arms"]["left"]["yaw"] == pytest.approx(1.5708)
    out = script.updated_config(text, "right", ("height",), {"height": -0.012})
    assert yaml.safe_load(out)["arms"]["right"]["shoulder_xyz"] == pytest.approx(
        [0.03, -0.232, 0.210]
    )
    # A snapshot needs still and live joints (readings at 20 Hz).
    times = np.arange(30) * 0.05
    still = np.column_stack([0.5 + 1e-4 * np.sin(times * 7), np.full(30, 0.1)])
    assert script.joint_problem(times, still) is None
    moving = still + np.column_stack([np.linspace(0, 0.02, 30), np.zeros(30)])
    assert "moved" in script.joint_problem(times, moving)
    assert "frozen" in script.joint_problem(times, np.full((30, 2), 0.3))
    assert "frozen" not in (
        script.joint_problem(times[:15], np.full((15, 2), 0.3)) or ""
    )
    assert "no fresh" in script.joint_problem(times[:2], still[:2])
    mean = script.circular_mean([math.pi - 0.01, -math.pi + 0.01])
    assert math.cos(mean) == pytest.approx(-1.0)
