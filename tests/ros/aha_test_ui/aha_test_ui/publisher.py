"""Publish robot monitor records (aha_test_ui.records) from a node (ROS).

    ui = UiPublisher(node, "place_runner")
    ui.status("deciding", "asking the LLM")
    ui.event("placed next to O1 lemon", "info")
    ui.chat("apple-1", "apple (C)", messages, notes={2: [note("place OK", "ok")]})

Status and chat publishers are transient local (depth 1 and 10), so a
monitor started later gets the latest status and recent chats; events are
volatile.
"""

from __future__ import annotations

import json

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from aha_test_ui import records


def _qos(depth: int, latched: bool) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE
        ),
        reliability=ReliabilityPolicy.RELIABLE,
    )


STATUS_QOS = _qos(1, True)
CHAT_QOS = _qos(10, True)
EVENT_QOS = _qos(50, False)


class UiPublisher:
    def __init__(self, node, source: str):
        self.source = source
        self.status_pub = node.create_publisher(
            String, records.STATUS_TOPIC, STATUS_QOS
        )
        self.chat_pub = node.create_publisher(String, records.CHAT_TOPIC, CHAT_QOS)
        self.event_pub = node.create_publisher(String, records.EVENT_TOPIC, EVENT_QOS)

    @staticmethod
    def _send(pub, record):
        pub.publish(String(data=json.dumps(record, ensure_ascii=False)))

    def status(self, state: str, detail: str = ""):
        self._send(self.status_pub, records.status(self.source, state, detail))

    def event(self, text: str, level: str = "info"):
        self._send(self.event_pub, records.event(self.source, text, level))

    def chat(self, chat_id, title, messages, notes=None, done=False):
        self._send(
            self.chat_pub,
            records.chat(self.source, chat_id, title, messages, notes, done),
        )
