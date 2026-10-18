"""Ground-truth pick through MoveIt KDL, OMPL and ROS trajectory actions."""

import math

from .api import Action
from .moveit_client import ARM_JOINTS, MoveItClient, rotate, relative_pose, world_pose
from .scene import load_scene


def clip(value, limit):
    return max(-limit, min(limit, value))


def yaw(pose):
    _, _, _, x, y, z, w = pose
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class GroundTruthIKPolicy:
    def reset(self, task_id, instruction):
        if task_id != "pick_apple":
            raise ValueError("ground_truth_ik supports pick_apple only")
        self.scene = load_scene()
        self.settings = self.scene["scripted"]
        self.config = self.scene["ground_truth_ik"]
        self.client = MoveItClient()
        self.phase = "prepare"
        self.phase_time = None
        self.target = None
        self.motion_done = False

    def close(self):
        client = getattr(self, "client", None)
        if client is not None:
            client.close()

    def _prepare(self, observation):
        root = observation.ground_truth.robot_world_pose
        joints = observation.joint_positions
        self.home = self.client.forward({**joints, **dict.fromkeys(ARM_JOINTS, 0.0)})
        self.heading = yaw(root)
        self.heading_rotation = (
            0,
            0,
            math.sin(self.heading / 2),
            math.cos(self.heading / 2),
        )
        self.orientation = world_pose((0, 0, 0, *self.heading_rotation), self.home)[3:]
        self.client.update_scene(
            root, self.scene, observation.ground_truth.object_position
        )
        self.client.start(
            {
                name: self.settings["transport_lift"] if name == "joint_r1" else 0.0
                for name in ARM_JOINTS
            },
            joints,
        )
        self.phase_time = observation.sim_time

    def _advance(self, phase, observation, position=None):
        self.phase = phase
        self.phase_time = observation.sim_time
        if position is None:
            return
        root = observation.ground_truth.robot_world_pose
        joints = observation.joint_positions
        self.target = (*position, *self.orientation)
        self.client.update_scene(
            root, self.scene, observation.ground_truth.object_position
        )
        goal = self.client.inverse(relative_pose(root, self.target), joints)
        print(f"MoveIt {phase}: target_world={position}, joints={goal}", flush=True)
        self.client.start(goal, joints)
        self.motion_done = False

    def _target_reached(self, root, joints):
        if self.target is None or not self.motion_done:
            return False
        current = world_pose(root, self.client.forward(joints))
        dot = abs(sum(a * b for a, b in zip(current[3:], self.target[3:])))
        return (
            math.dist(current[:3], self.target[:3]) < self.config["position_tolerance"]
            and 2 * math.acos(min(1.0, dot)) < 0.04
        )

    def _approach_error(self, truth):
        offset = rotate(self.heading_rotation, self.home[:3])
        heading = (math.cos(self.heading), math.sin(self.heading))
        delta = tuple(
            truth.object_position[i]
            - offset[i]
            - self.config["arm_reach"] * heading[i]
            - truth.robot_world_pose[i]
            for i in range(2)
        )
        lateral = -delta[0] * heading[1] + delta[1] * heading[0]
        if abs(lateral) > self.config["max_lateral_error"]:
            raise ValueError(
                "MoveIt baseline scene must align with the initial base heading"
            )
        distance = sum(delta[i] * heading[i] for i in range(2))
        error = self.heading - yaw(truth.robot_world_pose)
        return distance, math.atan2(math.sin(error), math.cos(error))

    def act(self, observation):
        truth = observation.ground_truth
        if truth is None:
            raise ValueError("ground_truth_ik requires Gazebo ground truth")
        time = observation.sim_time
        root = truth.robot_world_pose
        joints = observation.joint_positions
        if self.phase_time is None:
            self._prepare(observation)
        self.motion_done = self.client.completed() or self.motion_done
        age = time - self.phase_time
        if age > self.config["phase_timeout"] and self.phase != "hold":
            raise RuntimeError(f"MoveIt pick phase {self.phase} timed out")
        opening = self.settings["open_position"]
        closed = self.settings["grasp_position"]
        opened = (
            joints["joint_r7r"] >= opening - self.settings["open_tolerance"]
            and joints["joint_r7l"] <= -opening + self.settings["open_tolerance"]
        )
        velocity = angular = 0.0

        reached = self.phase in ("pregrasp", "grasp", "lift") and self._target_reached(
            root, joints
        )
        if self.phase == "prepare":
            if self.motion_done and opened and age > 2:
                self._advance("approach", observation)
        elif self.phase == "approach":
            distance, error = self._approach_error(truth)
            if abs(distance) < 0.006 and age > 1:
                # Return a stop command before blocking on scene/IK services.
                self._advance("settle", observation)
            else:
                velocity, angular = clip(distance, 0.08), clip(1.5 * error, 0.15)
        elif self.phase == "settle":
            if age > 0.5:
                self._advance(
                    "pregrasp",
                    observation,
                    (
                        *truth.object_position[:2],
                        truth.object_position[2] + self.config["pregrasp_height"],
                    ),
                )
        elif self.phase == "pregrasp":
            if reached and opened and age > 2:
                self._advance("grasp", observation, truth.object_position)
        elif self.phase == "grasp":
            if reached and age > 2:
                self._advance("close", observation)
        elif self.phase == "close":
            opening = closed
            if truth.finger_contacts == 2 and age > 1:
                self.client.attach_apple(
                    root, joints, truth.object_position, self.scene["apple_radius"]
                )
                self._advance(
                    "lift",
                    observation,
                    (*self.target[:2], self.target[2] + self.config["lift_height"]),
                )
        elif self.phase == "lift":
            opening = closed
            if reached and age > 2:
                self._advance("hold", observation)
        elif self.phase == "hold":
            opening = closed
        return Action(
            linear_velocity=velocity,
            angular_velocity=angular,
            joint_positions={"joint_r7r": opening, "joint_r7l": -opening},
        )
