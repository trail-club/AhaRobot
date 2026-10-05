"""Serialize only the public policy API across the process boundary."""

import base64
import json
from types import MappingProxyType

from .api import Action, Observation


def encode(message):
    return json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"


def observation_message(observation):
    image = None
    if observation.head_image is not None:
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import Image

        if not isinstance(observation.head_image, Image):
            raise TypeError("head_image must be a sensor_msgs/msg/Image")
        image = base64.b64encode(serialize_message(observation.head_image)).decode()
    return {
        "op": "act",
        "observation": {
            "sim_time": observation.sim_time,
            "task_id": observation.task_id,
            "instruction": observation.instruction,
            "joint_positions": dict(observation.joint_positions),
            "joint_velocities": dict(observation.joint_velocities),
            "odometry": list(observation.odometry),
            "head_image": image,
        },
    }


def decode_observation(message):
    image = message["head_image"]
    if image is not None:
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image

        image = deserialize_message(base64.b64decode(image, validate=True), Image)
    return Observation(
        sim_time=message["sim_time"],
        task_id=message["task_id"],
        instruction=message["instruction"],
        joint_positions=MappingProxyType(message["joint_positions"]),
        joint_velocities=MappingProxyType(message["joint_velocities"]),
        odometry=tuple(message["odometry"]),
        head_image=image,
    )


def action_message(action):
    if not isinstance(action, Action):
        raise TypeError("policy.act must return aha_sim_tasks.api.Action")
    return {
        "linear_velocity": action.linear_velocity,
        "angular_velocity": action.angular_velocity,
        "joint_positions": dict(action.joint_positions),
    }


def decode_action(message):
    action = Action(**message)
    action.validate()
    return action
