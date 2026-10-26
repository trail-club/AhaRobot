#!/usr/bin/env python3
"""Robot monitor window: module states, LLM chats and events (Qt + rclpy).

Subscribes the aha_test_ui topics (aha_test_ui.records): /aha/ui/status shows one row
per source (state, detail, age), /aha/ui/chat the LLM conversations (the
combo box keeps those seen since start, newest first; a new one is selected
when it starts; "system" also shows system prompts), /aha/ui/events a log
(the last MAX_EVENTS lines). Runs on the wall clock.
"""

import json
import signal
import sys

import rclpy
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String

from aha_test_ui import records

MAX_CHATS = 50
MAX_EVENTS = 500


def subscription_qos(depth, latched):
    return QoSProfile(
        depth=depth,
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE
        ),
        reliability=ReliabilityPolicy.RELIABLE,
    )


class Monitor(QWidget):
    def __init__(self, node):
        super().__init__()
        self.setWindowTitle("AhaRobot monitor")
        self.states = {}  # source -> status record
        self.chats = []  # newest first

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["source", "state", "detail", "age"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self.render_chat)
        self.system = QCheckBox("system")
        self.system.toggled.connect(self.render_chat)
        self.chat = QTextBrowser()
        chat_box = QWidget()
        chat_layout = QVBoxLayout(chat_box)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.combo, 1)
        row.addWidget(self.system)
        chat_layout.addLayout(row)
        chat_layout.addWidget(self.chat, 1)

        self.events = QTextBrowser()
        self.events.document().setMaximumBlockCount(MAX_EVENTS)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(chat_box)
        splitter.addWidget(self.events)
        splitter.setSizes([150, 450, 180])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.resize(640, 860)

        node.create_subscription(
            String, records.STATUS_TOPIC, self.on_status, subscription_qos(20, True)
        )
        node.create_subscription(
            String, records.CHAT_TOPIC, self.on_chat, subscription_qos(20, True)
        )
        node.create_subscription(
            String, records.EVENT_TOPIC, self.on_event, subscription_qos(100, False)
        )
        self.spin_timer = QTimer(self)
        self.spin_timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
        self.spin_timer.start(50)
        self.age_timer = QTimer(self)
        self.age_timer.timeout.connect(self.render_states)
        self.age_timer.start(1000)

    @staticmethod
    def parse(msg):
        try:
            return json.loads(msg.data)
        except ValueError:
            return None

    def on_status(self, msg):
        record = self.parse(msg)
        if record and record.get("source"):
            self.states[record["source"]] = record
            self.render_states()

    def render_states(self):
        rows = sorted(self.states.values(), key=lambda r: r["source"])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            values = (
                r["source"],
                r.get("state", ""),
                r.get("detail", ""),
                records.age_text(r.get("stamp", 0.0)),
            )
            for j, value in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))

    def on_chat(self, msg):
        record = self.parse(msg)
        if not record:
            return
        key = (record.get("source"), record.get("id"))
        index = next(
            (
                i
                for i, c in enumerate(self.chats)
                if (c.get("source"), c.get("id")) == key
            ),
            None,
        )
        if index is not None:
            self.chats[index] = record
            self.combo.setItemText(index, self.title(record))
            if self.combo.currentIndex() == index:
                self.render_chat()
            return
        self.chats.insert(0, record)
        del self.chats[MAX_CHATS:]
        self.combo.blockSignals(True)
        self.combo.insertItem(0, self.title(record))
        while self.combo.count() > len(self.chats):
            self.combo.removeItem(self.combo.count() - 1)
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)
        self.render_chat()

    @staticmethod
    def title(record):
        state = "" if record.get("done") else " …"
        return (
            f"{record.get('title') or record.get('id')} [{record.get('source')}]{state}"
        )

    def render_chat(self):
        index = self.combo.currentIndex()
        if not 0 <= index < len(self.chats):
            return
        bar = self.chat.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 4
        self.chat.setHtml(records.chat_html(self.chats[index], self.system.isChecked()))
        if at_end:
            bar.setValue(bar.maximum())

    def on_event(self, msg):
        record = self.parse(msg)
        if record:
            self.events.append(records.event_html(record))


def main():
    rclpy.init(args=sys.argv)
    app = QApplication(remove_ros_args(sys.argv))
    node = rclpy.create_node("aha_robot_monitor")
    monitor = Monitor(node)
    monitor.show()
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    rc = app.exec_()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
