"""aha_perception の音声スタック (audio) の各ノードで共有する topic 名・QoS・小さなヘルパ."""

from __future__ import annotations

import json
import re

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .model_registry import Registry

TOPIC_TRANSCRIPT = '/aha/perception/audio/transcript'
TOPIC_DEBUG = '/aha/perception/audio/debug'
TOPIC_MODELS = '/aha/perception/audio/models'
TOPIC_TTS_ACTIVE = '/aha/perception/audio/tts_active'
ACTION_LISTEN = '/aha/perception/audio/listen'
ACTION_SPEAK = '/aha/perception/audio/speak'

# ASR が書き出し、デバッグ UI が配信する認識区間の WAV
SEGMENTS_DIR = '/tmp/aha_audio/segments'
SEGMENT_NAME = re.compile(r'^\d{6}\.wav$')


def segment_filename(seg_id: int) -> str:
    return f'{seg_id:06d}.wav'


def latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(depth=depth, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      reliability=ReliabilityPolicy.RELIABLE)


def json_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def notice_json(level: str, text: str) -> str:
    """デバッグ UI にトーストを出すメッセージ."""
    return json_text({'type': 'notice', 'level': level, 'text': text})


class AudioModelNode(Node):
    """モデルを読み込むノード (asr / tts) の共通部分."""

    def __init__(self, name: str):
        super().__init__(name)
        self.declare_parameter('models_file', '')
        self.declare_parameter('device', 'cuda:0')
        self.registry = Registry.load(self.get_parameter('models_file').value)
        self.device = self.get_parameter('device').value
        self.pub_debug = self.create_publisher(String, TOPIC_DEBUG, 50)
        self.pub_models = self.create_publisher(String, TOPIC_MODELS, latched_qos())

    def debug(self, payload: dict) -> None:
        self.pub_debug.publish(String(data=json_text(payload)))

    def publish_models(self) -> None:
        """読み込んだモデルの記録を /aha/perception/audio/models に出してログにも残す."""
        record = self.registry.runtime_record(self.get_name())
        self.pub_models.publish(String(data=record))
        self.get_logger().info(f'models: {record}')

    def serve_action(self, action_type, name: str, execute_callback, goal_callback) -> ActionServer:
        return ActionServer(
            self, action_type, name, execute_callback=execute_callback,
            goal_callback=goal_callback, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())


def spin_multithreaded(node_cls, args=None) -> None:
    rclpy.init(args=args)
    node = node_cls()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
