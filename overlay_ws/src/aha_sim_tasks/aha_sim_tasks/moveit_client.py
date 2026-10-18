"""ROS 2 MoveIt client for the privileged, simulator ground-truth baseline."""

import math

from aha_gazebo.model import LIMIT_TOLERANCE
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AttachedCollisionObject,
    PlanningSceneComponents,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    RobotState,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetPlanningScene,
    GetPositionFK,
    GetPositionIK,
)
from rclpy.action import ActionClient
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from .api import JOINT_LIMITS, clamp_joint_near_limits

ARM_JOINTS = tuple(f"joint_r{i}" for i in range(1, 7))
FRAME = "base_footprint"
TIP = "link_ree"


def multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate(q, v):
    return multiply(multiply(q, (*v, 0.0)), (-q[0], -q[1], -q[2], q[3]))[:3]


def relative_pose(root, world):
    inverse = (-root[3], -root[4], -root[5], root[6])
    return (
        *rotate(inverse, tuple(world[i] - root[i] for i in range(3))),
        *multiply(inverse, world[3:]),
    )


def world_pose(root, local):
    offset = rotate(root[3:], local[:3])
    return (
        *(root[i] + offset[i] for i in range(3)),
        *multiply(root[3:], local[3:]),
    )


def pose_message(values):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, values[:3])
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = map(float, values[3:])
    return pose


def pose_values(pose):
    return (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )


def robot_state(joints):
    # Physics can report joints up to its limits outside the URDF ones;
    # every service gets a full state.
    values = {
        name: clamp_joint_near_limits(name, value, LIMIT_TOLERANCE)
        for name, value in joints.items()
    }
    return RobotState(
        joint_state=JointState(name=list(values), position=list(values.values())),
        is_diff=False,
    )


def scene_request():
    request = ApplyPlanningScene.Request()
    request.scene.is_diff = True
    request.scene.robot_state.is_diff = True
    return request


def collision_object(name, frame, shape, pose):
    obj = CollisionObject(id=name)
    obj.header.frame_id = frame
    obj.operation = CollisionObject.ADD
    obj.primitives = [shape]
    obj.primitive_poses = [pose_message(pose)]
    return obj


class MoveItClient:
    def __init__(self):
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node(
            "ground_truth_moveit_policy", context=self.context
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.fk = self.node.create_client(GetPositionFK, "/compute_fk")
        self.ik = self.node.create_client(GetPositionIK, "/compute_ik")
        self.scene = self.node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.get_scene = self.node.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        self.move = ActionClient(self.node, MoveGroup, "/move_action")
        self.pending_goal = self.pending_result = self.goal = None
        self.attached = False

    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=10.0):
            raise TimeoutError(f"MoveIt service unavailable: {client.srv_name}")
        future = client.call_async(request)
        self.executor.spin_until_future_complete(future, timeout_sec=10.0)
        if not future.done():
            raise TimeoutError(f"MoveIt service timed out: {client.srv_name}")
        return future.result()

    def forward(self, joints):
        request = GetPositionFK.Request()
        request.header.frame_id = FRAME
        request.fk_link_names = [TIP]
        request.robot_state = robot_state(joints)
        response = self.call(self.fk, request)
        if (
            response.error_code.val != MoveItErrorCodes.SUCCESS
            or not response.pose_stamped
        ):
            raise RuntimeError(f"MoveIt FK failed: {response.error_code.val}")
        return pose_values(response.pose_stamped[0].pose)

    def inverse(self, target, joints):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = "right_arm"
        ik.ik_link_name = TIP
        ik.robot_state = robot_state(joints)
        ik.robot_state.is_diff = True
        ik.pose_stamped.header.frame_id = FRAME
        ik.pose_stamped.pose = pose_message(target)
        ik.avoid_collisions = True
        ik.timeout.sec = 2
        response = self.call(self.ik, request)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt KDL IK failed: {response.error_code.val}")
        solution = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
            )
        )
        for name in ARM_JOINTS:
            value = solution.get(name)
            if (
                value is None
                or not math.isfinite(value)
                or not JOINT_LIMITS[name][0] <= value <= JOINT_LIMITS[name][1]
            ):
                raise RuntimeError(f"Invalid MoveIt IK solution: {name}={value}")
        return {name: solution[name] for name in ARM_JOINTS}

    def start(self, goal_joints, joints):
        if self.pending_goal is not None or self.pending_result is not None:
            raise RuntimeError("MoveIt motion is already active")
        if not self.move.wait_for_server(timeout_sec=10.0):
            raise TimeoutError("MoveIt move_action unavailable")
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = "right_arm"
        request.pipeline_id = "ompl"
        request.planner_id = "RRTConnectkConfigDefault"
        request.start_state = robot_state(joints)
        request.start_state.is_diff = True
        request.num_planning_attempts = 3
        request.allowed_planning_time = 3.0
        request.max_velocity_scaling_factor = 0.4
        request.max_acceleration_scaling_factor = 0.4
        request.goal_constraints = [
            Constraints(
                joint_constraints=[
                    JointConstraint(
                        joint_name=name,
                        position=float(goal_joints[name]),
                        tolerance_above=0.002,
                        tolerance_below=0.002,
                        weight=1.0,
                    )
                    for name in ARM_JOINTS
                ]
            )
        ]
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self.pending_goal = self.move.send_goal_async(goal)

    def completed(self):
        self.executor.spin_once(timeout_sec=0.0)
        if self.pending_goal is not None and self.pending_goal.done():
            self.goal = self.pending_goal.result()
            self.pending_goal = None
            if not self.goal.accepted:
                self.goal = None
                raise RuntimeError("MoveIt rejected the motion goal")
            self.pending_result = self.goal.get_result_async()
        if self.pending_result is not None and self.pending_result.done():
            result = self.pending_result.result()
            self.pending_result = None
            self.goal = None
            if (
                result.status != GoalStatus.STATUS_SUCCEEDED
                or result.result.error_code.val != MoveItErrorCodes.SUCCESS
            ):
                raise RuntimeError(
                    f"MoveIt execution failed: {result.result.error_code.val}"
                )
            return True
        return False

    def update_scene(self, root, scene, apple):
        request = scene_request()
        diff = request.scene
        for name, location in (
            ("source_table", scene["source_position"]),
            ("destination_table", scene["destination_position"]),
            ("apple", apple),
        ):
            if name == "apple" and self.attached:
                continue
            if name == "apple":
                shape = SolidPrimitive(
                    type=SolidPrimitive.SPHERE, dimensions=[scene["apple_radius"]]
                )
                center = tuple(location)
            else:
                shape = SolidPrimitive(
                    type=SolidPrimitive.BOX, dimensions=scene["table_size"]
                )
                center = (*location[:2], scene["table_size"][2] / 2)
            diff.world.collision_objects.append(
                collision_object(
                    name, FRAME, shape, relative_pose(root, (*center, 0, 0, 0, 1))
                )
            )
        diff.allowed_collision_matrix = self._grasp_collision_matrix()
        if not self.call(self.scene, request).success:
            raise RuntimeError("MoveIt rejected the planning scene")

    def _grasp_collision_matrix(self):
        # Only intended apple/finger contact is allowed. All other collisions remain checked.
        query = GetPlanningScene.Request()
        query.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        matrix = self.call(self.get_scene, query).scene.allowed_collision_matrix
        names = ["apple", "link_r7r", "link_r7l"]
        for name in names:
            if name not in matrix.entry_names:
                matrix.entry_names.append(name)
                for row in matrix.entry_values:
                    row.enabled.append(False)
                matrix.entry_values.append(
                    AllowedCollisionEntry(enabled=[False] * len(matrix.entry_names))
                )
        for finger in names[1:]:
            i, j = matrix.entry_names.index("apple"), matrix.entry_names.index(finger)
            matrix.entry_values[i].enabled[j] = True
            matrix.entry_values[j].enabled[i] = True
        return matrix

    def attach_apple(self, root, joints, apple, radius):
        """Attach collision geometry in MoveIt only; Gazebo grasp stays physical."""
        tip = world_pose(root, self.forward(joints))
        obj = collision_object(
            "apple",
            TIP,
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[radius]),
            relative_pose(tip, (*apple, 0, 0, 0, 1)),
        )
        request = scene_request()
        request.scene.robot_state.attached_collision_objects = [
            AttachedCollisionObject(
                link_name=TIP, object=obj, touch_links=["link_r7r", "link_r7l"]
            )
        ]
        if not self.call(self.scene, request).success:
            raise RuntimeError("MoveIt rejected the attached apple collision geometry")
        self.attached = True

    def close(self):
        try:
            self._cancel_motion()
        finally:
            self.move.destroy()
            self.executor.shutdown()
            self.node.destroy_node()
            self.context.shutdown()

    def _cancel_motion(self):
        if self.pending_goal is not None:
            self.executor.spin_until_future_complete(self.pending_goal, timeout_sec=2.0)
            if self.pending_goal.done():
                goal = self.pending_goal.result()
                self.goal = goal if goal.accepted else None
        if self.goal is not None:
            future = self.goal.cancel_goal_async()
            self.executor.spin_until_future_complete(future, timeout_sec=2.0)
