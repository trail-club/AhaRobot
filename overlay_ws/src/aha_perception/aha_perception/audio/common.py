"""aha_perception の音声スタック (audio) の各ノードで共有する topic 名・QoS・小さなヘルパ."""

from __future__ import annotations

import json
import re
import shlex
import sys

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .model_registry import Registry
from .remote import LOCAL, SSH_PORT, make_worker

TOPIC_TRANSCRIPT = "/aha/perception/audio/transcript"
TOPIC_DEBUG = "/aha/perception/audio/debug"
TOPIC_MODELS = "/aha/perception/audio/models"
TOPIC_TTS_ACTIVE = "/aha/perception/audio/tts_active"
ACTION_LISTEN = "/aha/perception/audio/listen"
ACTION_SPEAK = "/aha/perception/audio/speak"

# ASR が書き出し、デバッグ UI が配信する認識区間の WAV
SEGMENTS_DIR = "/tmp/aha_audio/segments"
SEGMENT_NAME = re.compile(r"^\d{6}\.wav$")


def segment_filename(seg_id: int) -> str:
    return f"{seg_id:06d}.wav"


def latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def json_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def notice_json(level: str, text: str) -> str:
    """デバッグ UI にトーストを出すメッセージ."""
    return json_text({"type": "notice", "level": level, "text": text})


class AudioModelNode(Node):
    """モデルを読み込むノード (asr / tts) の共通部分."""

    def __init__(self, name: str):
        super().__init__(name)
        self.declare_parameter("models_file", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("target", "dgx")  # dgx | trailpc | local
        self.declare_parameter("ssh_host", "")  # 空なら target のアドレス
        self.declare_parameter("ssh_port", SSH_PORT)
        self.declare_parameter("ssh_user", "")
        self.declare_parameter("identity_file", "")
        self.declare_parameter("request_timeout", 30.0)
        self.declare_parameter("startup_timeout", 1800.0)
        self.declare_parameter("worker_command", "")  # テスト用: SSH の代わりのコマンド
        self.registry = Registry.load(self.get_parameter("models_file").value)
        self.device = self.get_parameter("device").value
        self.target = self.get_parameter("target").value
        self.request_timeout = self.get_parameter("request_timeout").value
        self.startup_timeout = self.get_parameter("startup_timeout").value
        self.worker = None
        self.pub_debug = self.create_publisher(String, TOPIC_DEBUG, 50)
        self.pub_models = self.create_publisher(String, TOPIC_MODELS, latched_qos())

    def start(self) -> None:
        """Load the models. Separate from __init__ so that an interrupted load still
        reaches destroy_node, which removes the worker container."""
        raise NotImplementedError

    @property
    def remote(self) -> bool:
        return self.target != LOCAL

    @property
    def error(self) -> str | None:
        """Why this node cannot go on: the connection to the worker was lost."""
        return self.worker.failure if self.worker is not None else None

    def start_worker(self, roles: list[str]) -> None:
        get = lambda name: self.get_parameter(name).value  # noqa: E731
        self.worker = make_worker(
            self.target,
            roles,
            device=self.device,
            host=get("ssh_host"),
            port=get("ssh_port"),
            user=get("ssh_user"),
            identity_file=get("identity_file"),
            log=self.get_logger().info,
            command=shlex.split(get("worker_command")) or None,
        )
        self.registry.packages = self.worker.start(self.startup_timeout)["packages"]

    def destroy_node(self):
        if self.worker is not None and not self.worker.close():
            self.get_logger().error(
                "worker cleanup was incomplete; see the messages above"
            )
        super().destroy_node()

    def debug(self, payload: dict) -> None:
        self.pub_debug.publish(String(data=json_text(payload)))

    def publish_models(self) -> None:
        """読み込んだモデルの記録を /aha/perception/audio/models に出してログにも残す."""
        record = self.registry.runtime_record(self.get_name())
        self.pub_models.publish(String(data=record))
        self.get_logger().info(f"models: {record}")

    def serve_action(
        self, action_type, name: str, execute_callback, goal_callback
    ) -> ActionServer:
        return ActionServer(
            self,
            action_type,
            name,
            execute_callback=execute_callback,
            goal_callback=goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )


def spin_multithreaded(node_cls, args=None) -> None:
    rclpy.init(args=args)
    node = node_cls()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    name = node.get_name()
    error = None
    try:
        node.start()
        while rclpy.ok() and (error := node.error) is None:
            executor.spin_once(timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    if error is not None:
        print(f"{name} stopped: {error}", file=sys.stderr)
        sys.exit(1)
