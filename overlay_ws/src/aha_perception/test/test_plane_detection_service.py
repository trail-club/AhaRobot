"""get_place_candidates of plane_detection_node.py on a synthetic detection
(needs rclpy and the ROS messages; no other node)."""

import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("aha_msgs")

from aha_msgs.msg import PlaceObstacle  # noqa: E402
from aha_msgs.srv import GetPlaceCandidates  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402

from aha_perception.furniture import Match  # noqa: E402
from aha_perception.planes import DepthNoiseModel, PlaneLabel  # noqa: E402
from aha_perception.planes.segmentation import make_plane  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plane_detection_node.py"
Response = GetPlaceCandidates.Response
STEP = 0.01
CAMERA = (0.0, 0.0, 1.4)
STAMP = TimeMsg(sec=100, nanosec=0)
# frame base_footprint: the furniture TF is the identity.
FURNITURE = """
frame: base_footprint
furniture:
  - name: table
    pose: {x: 1.4, y: 0.0}
    size: {x: 0.8, y: 0.6}
    levels: [0.75]
  - name: shelf
    pose: {x: 1.2, y: 1.5}
    size: {x: 0.35, y: 0.8}
    levels: [0.40, 0.62]
    board_thickness: 0.02
"""


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("plane_detection_node", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def node(module, tmp_path_factory):
    config = tmp_path_factory.mktemp("furniture") / "furniture.yaml"
    config.write_text(FURNITURE)
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"furniture_config:={config}",
            "-p",
            "tracking_frame:=base_footprint",
        ]
    )
    node = module.PlaneDetection()
    yield node
    node.stop()
    node.destroy_node()
    rclpy.try_shutdown()


@pytest.fixture(autouse=True)
def fresh(node):
    node.tf_buffer.clear()
    with node.latest_lock:
        node.latest = None
    yield


def grid(x0, x1, y0, y1, z):
    x, y = np.meshgrid(np.arange(x0, x1 + 1e-9, STEP), np.arange(y0, y1 + 1e-9, STEP))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def detection(module, surface_ids=(7,), furniture=None, parts=None, age=0.0):
    """Latest detection of a table (1.0-1.8 x -0.3-0.3 m at 0.75 m), the
    cloud given in base_footprint (camera_to_base identity)."""
    parts = parts or [(grid(1.0, 1.8, -0.3, 0.3, 0.75), PlaneLabel.TABLE)]
    points, planes, start = [], [], 0
    for part, label in parts:
        points.append(part)
        if label is not None:
            inliers = np.arange(start, start + len(part))
            plane = make_plane(part, inliers, CAMERA, np.array([0.0, 0.0, 1.0]))
            plane.label = label
            planes.append(plane)
        start += len(part)
    cloud = np.concatenate(points)[None]
    return module.Detection(
        stamp=STAMP,
        planes=planes,
        cloud=cloud,
        camera_to_base=np.eye(4),
        noise=DepthNoiseModel(),
        wall_time=time.monotonic() - age,
        surface_ids=list(surface_ids),
        surfaces=[],
        furniture=furniture,
    )


def set_latest(node, detection):
    with node.latest_lock:
        node.latest = detection


def request(**fields):
    req = GetPlaceCandidates.Request()
    req.object_size.x, req.object_size.y, req.object_size.z = 0.06, 0.06, 0.1
    for name, value in fields.items():
        setattr(req, name, value)
    return req


def call(node, req):
    return node.on_get_place_candidates(req, Response())


def odom_tf(node, x, sec):
    """base_footprint <- odom: odom's origin at (x, 0) of base_footprint."""
    tf = TransformStamped()
    tf.header.frame_id = "base_footprint"
    tf.header.stamp = TimeMsg(sec=sec)
    tf.child_frame_id = "odom"
    tf.transform.translation.x = x
    tf.transform.rotation.w = 1.0
    node.tf_buffer.set_transform(tf, "test")


def test_without_or_with_an_old_detection_is_an_error(node, module):
    response = call(node, request())
    assert not response.success and response.status == Response.STATUS_ERROR
    assert "no plane detection" in response.message
    set_latest(node, detection(module, age=10.0))
    response = call(node, request())
    assert response.status == Response.STATUS_ERROR and "old" in response.message


def test_candidates_on_the_table(node, module):
    set_latest(node, detection(module))
    response = call(node, request(max_candidates=3))
    assert response.success and response.status == Response.STATUS_OK
    assert response.stamp == STAMP
    assert len(response.candidates) == 3
    first = response.candidates[0]
    assert first.surface_id == 7 and first.plane_id == 0 and not first.oriented
    assert first.pose.header.stamp == STAMP
    assert [p.id for p in response.planes.planes] == [0]


def test_argument_errors(node, module):
    set_latest(node, detection(module))
    response = call(node, request(furniture_level=1))
    assert response.status == Response.STATUS_ERROR
    bad = request()
    bad.object_size.z = 0.0
    assert call(node, bad).status == Response.STATUS_ERROR
    assert call(node, request(surface_labels=["sofa"])).status == Response.STATUS_ERROR
    # Non-finite inputs are errors, not silently ignored boxes or yaws.
    nan = float("nan")
    region = PlaceObstacle()
    region.pose.position.x, region.size.x = nan, 0.1
    assert call(node, request(exclude_regions=[region])).status == Response.STATUS_ERROR
    oriented = request(yaw_offsets=[nan])
    oriented.approach_from.header.frame_id = "base_footprint"
    assert call(node, oriented).status == Response.STATUS_ERROR


def test_missing_surface_is_no_surface(node, module):
    set_latest(node, detection(module))
    response = call(node, request(surface_ids=[3]))
    assert response.success and response.status == Response.STATUS_NO_SURFACE
    assert "S3 is not tracked" in response.message
    assert response.stamp == STAMP


def test_furniture_requests(node, module):
    # No association in the detection (no TF to the config frame).
    set_latest(node, detection(module))
    response = call(node, request(furniture="table"))
    assert response.status == Response.STATUS_ERROR
    assert "not associated" in response.message
    # The table's level is associated with the surface.
    set_latest(node, detection(module, furniture={7: Match("table", 1, 0.0, 0.0)}))
    response = call(node, request(furniture="table", furniture_level=1))
    assert response.status == Response.STATUS_OK
    assert {c.surface_id for c in response.candidates} == {7}
    assert response.view_point.header.frame_id == "base_footprint"
    # A configured level not in the detection.
    response = call(node, request(furniture="shelf", furniture_level=1))
    assert response.status == Response.STATUS_NO_SURFACE
    # Unknown furniture, and an object taller than the shelf level's room.
    assert call(node, request(furniture="sofa")).status == Response.STATUS_ERROR
    tall = request(furniture="shelf", furniture_level=1)
    tall.object_size.z = 0.3
    assert call(node, tall).status == Response.STATUS_NO_SPACE
    # A bad size is an error before the furniture's room is checked.
    tall.object_size.x = 0.0
    response = call(node, tall)
    assert (
        response.status == Response.STATUS_ERROR and "object size" in response.message
    )


def test_near_is_transformed_at_the_detection_stamp(node, module):
    set_latest(node, detection(module))
    near = request()
    near.near.header.frame_id = "odom"
    near.near.point.x, near.near.point.y, near.near.point.z = 0.0, 0.2, 0.75
    response = call(node, near)
    assert response.status == Response.STATUS_ERROR and "no TF" in response.message
    # At the detection stamp odom is 1.2 m ahead; later the robot moved.
    odom_tf(node, 1.2, STAMP.sec)
    odom_tf(node, 0.4, STAMP.sec + 2)
    response = call(node, near)
    assert response.status == Response.STATUS_OK
    position = response.candidates[0].pose.pose.position
    assert math.hypot(position.x - 1.2, position.y - 0.2) < 0.08


def test_oriented_candidates_head_the_checked_yaw(node, module):
    set_latest(node, detection(module))
    req = request(max_candidates=2)
    req.approach_from.header.frame_id = "base_footprint"
    response = call(node, req)
    assert response.status == Response.STATUS_OK
    for c in response.candidates:
        assert c.oriented
        q, p = c.pose.pose.orientation, c.pose.pose.position
        heading = math.atan2(2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.y**2 + q.z**2))
        assert heading == pytest.approx(math.atan2(p.y, p.x), abs=0.05)


def test_degenerate_support_is_reported(node, module):
    set_latest(node, detection(module))
    region = PlaceObstacle()
    region.pose.position.x, region.pose.position.z = 1.4, 0.75
    region.size.x = region.size.y = 2.0
    region.size.z = 0.2
    response = call(node, request(exclude_regions=[region]))
    assert response.status == Response.STATUS_NO_SURFACE
    assert "without valid inliers" in response.message


def test_exclude_region_without_orientation_is_axis_aligned(module):
    region = PlaceObstacle()
    region.pose.position.x = 1.0
    region.size.x, region.size.y, region.size.z = 0.2, 0.1, 0.3
    (box,) = module.exclude_boxes([region])
    np.testing.assert_allclose(box.rotation, np.eye(3))
    np.testing.assert_allclose(box.size, [0.2, 0.1, 0.3])


def test_level_ceilings_follow_the_measured_height(node, module):
    shelf = [(grid(1.05, 1.35, 1.1, 1.9, 0.42), PlaneLabel.SHELF)]
    found = detection(
        module,
        surface_ids=(4,),
        parts=shelf,
        furniture={4: Match("shelf", 1, 0.02, 0.0)},
    )
    # Room 0.62 - 0.02 - 0.40 = 0.20 m above the level top, the plane 2 cm above it.
    assert node.level_ceilings(found) == {0: pytest.approx(0.18)}
    # A 0.19 m object fits the configured room but not under the board seen
    # from this plane, also without a furniture request.
    set_latest(node, found)
    tall = request()
    tall.object_size.z = 0.19
    assert call(node, tall).status == Response.STATUS_NO_SPACE
    assert call(node, request()).status == Response.STATUS_OK


def test_plane_off_its_footprint_loses_the_association(node, module):
    found = detection(module, furniture={7: Match("shelf", 1, 0.0, 0.0)})
    planes, furniture = node.clip_to_furniture(
        found.planes, found.surface_ids, found.furniture, STAMP
    )
    assert furniture == {} and planes[0] is found.planes[0]
    # On its footprint the hull is clipped (grown by furniture_clip_margin).
    found = detection(module, furniture={7: Match("table", 1, 0.0, 0.0)})
    wide = detection(
        module, parts=[(grid(0.9, 1.9, -0.4, 0.4, 0.75), PlaneLabel.TABLE)]
    )
    planes, furniture = node.clip_to_furniture(
        wide.planes, wide.surface_ids, found.furniture, STAMP
    )
    assert furniture == found.furniture
    hull = planes[0].hull_points()
    assert hull[:, 0].min() == pytest.approx(1.0 - node.clip_margin, abs=1e-6)
    assert hull[:, 1].max() == pytest.approx(0.3 + node.clip_margin, abs=1e-6)
