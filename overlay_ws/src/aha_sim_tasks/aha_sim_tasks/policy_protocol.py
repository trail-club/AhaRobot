"""Serialize only the public policy API across the process boundary."""

import base64
import json
from types import MappingProxyType

from .api import Action, CameraObservation, Observation


def encode(message):
    return json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"


def _encode_ros(message, message_type):
    from rclpy.serialization import serialize_message

    if not isinstance(message, message_type):
        raise TypeError(f"Expected {message_type.__name__} camera message")
    return base64.b64encode(serialize_message(message)).decode()


def _decode_ros(message, message_type):
    from rclpy.serialization import deserialize_message

    return deserialize_message(base64.b64decode(message, validate=True), message_type)


def observation_message(observation):
    cameras = {}
    if observation.cameras or observation.head_image is not None:
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import CameraInfo, Image

        for name, camera in observation.cameras.items():
            if not isinstance(camera, CameraObservation):
                raise TypeError("cameras must contain CameraObservation values")
            cameras[name] = {
                "image": _encode_ros(camera.image, Image),
                "camera_info": _encode_ros(camera.camera_info, CameraInfo),
                "base_transform": _encode_ros(camera.base_transform, TransformStamped),
                "depth_image": (
                    _encode_ros(camera.depth_image, Image)
                    if camera.depth_image is not None
                    else None
                ),
            }
    # Avoid transferring the head RGB pixels twice in every policy call.
    head = observation.cameras.get("head")
    image = (
        _encode_ros(observation.head_image, Image)
        if observation.head_image is not None
        and (head is None or observation.head_image is not head.image)
        else None
    )
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
            "cameras": cameras,
        },
    }


def decode_observation(message):
    cameras = {}
    image = message["head_image"]
    if message.get("cameras") or image is not None:
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import CameraInfo, Image

        for name, camera in message.get("cameras", {}).items():
            cameras[name] = CameraObservation(
                image=_decode_ros(camera["image"], Image),
                camera_info=_decode_ros(camera["camera_info"], CameraInfo),
                base_transform=_decode_ros(camera["base_transform"], TransformStamped),
                depth_image=(
                    _decode_ros(camera["depth_image"], Image)
                    if camera["depth_image"] is not None
                    else None
                ),
            )
        image = _decode_ros(image, Image) if image is not None else None
    if image is None and "head" in cameras:
        image = cameras["head"].image
    return Observation(
        sim_time=message["sim_time"],
        task_id=message["task_id"],
        instruction=message["instruction"],
        joint_positions=MappingProxyType(message["joint_positions"]),
        joint_velocities=MappingProxyType(message["joint_velocities"]),
        odometry=tuple(message["odometry"]),
        head_image=image,
        cameras=MappingProxyType(cameras),
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
