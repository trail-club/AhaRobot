#!/usr/bin/env python3
"""Simulation control panel: base joystick + head pan/tilt (Qt + rclpy).

Base:
  - Drag the pad: up/down = forward/back, left/right = turn left/right.
    The robot moves while the knob is held; dragging outside the pad keeps
    the knob on the rim. Releasing the mouse button stops it.
  - WASD while the window has focus (hold to move, release to stop).
  - Publishes geometry_msgs/TwistStamped to /diff_drive_controller/cmd_vel at
    10 Hz while active, then one zero command. Losing window focus or the stop
    button also stops. The stamp is zero, so diff_drive_controller stamps the
    command with its own (sim) clock on receipt and its cmd_vel_timeout stops
    the base when the panel stops publishing.
Head:
  - Pan/tilt sliders in degrees and presets, limits from config/head.yaml.
  - Sliders follow /joint_states except while being dragged or while a
    command sent from this panel is still executing, so they do not fight
    with head_look_at.py or teleop_head.py.

Runs on the wall clock (no use_sim_time): base commands are zero-stamped
and head trajectories are unstamped (start now).

Sizes follow the font metrics, so the layout scales with the font DPI
(e.g. Xft.dpi: 192), also when it changes at runtime; the window never
shrinks below its content.
"""

import math
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from python_qt_binding.QtCore import QEvent, QPointF, QSize, Qt, QTimer, Signal
from python_qt_binding.QtGui import QBrush, QColor, QPainter, QPen
from python_qt_binding.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from aha_perception.head_config import (
    TRAJECTORY_TOPIC,
    declare_head_config,
    describe,
    head_position,
)

CMD_VEL_TOPIC = "/diff_drive_controller/cmd_vel"
CMD_PERIOD_MS = 100
KEY_SPEED_RATIO = 0.5
SLIDER_THROTTLE_S = 0.25
JOINT_STATE_TIMEOUT_S = 2.0
# Knob radius relative to the pad's travel radius.
KNOB_RATIO = 0.22
PAD_MARGIN_PX = 4.0
# Pad side in font heights (minimum / preferred), with pixel floors.
PAD_MIN_LINES = 12
PAD_HINT_LINES = 16
PAD_MIN_PX = 200
PAD_HINT_PX = 250

KEY_DIRECTIONS = {
    Qt.Key_W: (1, 0),
    Qt.Key_S: (-1, 0),
    Qt.Key_A: (0, 1),
    Qt.Key_D: (0, -1),
}


def base_command(linear, angular):
    """TwistStamped with a zero stamp (diff_drive_controller fills in its now())."""
    msg = TwistStamped()
    msg.twist.linear.x = float(linear)
    msg.twist.angular.z = float(angular)
    return msg


class JoystickPad(QWidget):
    """Spring-loaded pad. Emits (x, y) in [-1, 1] (right +, up +) while held.

    The mouse grab keeps the drag alive outside the pad (knob clamped to the
    rim); releasing the button returns the knob to the center.
    """

    moved = Signal(float, float)
    released = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.NoFocus)
        self._knob = QPointF(0.0, 0.0)
        self._active = False

    def _side(self, lines, floor_px):
        return max(floor_px, lines * self.fontMetrics().height())

    def minimumSizeHint(self):
        side = self._side(PAD_MIN_LINES, PAD_MIN_PX)
        return QSize(side, side)

    def sizeHint(self):
        side = self._side(PAD_HINT_LINES, PAD_HINT_PX)
        return QSize(side, side)

    @property
    def active(self):
        return self._active

    def _radius(self):
        # Travel radius; the knob centered on the rim must still fit in the widget.
        return (min(self.width(), self.height()) / 2.0 - PAD_MARGIN_PX) / (
            1.0 + KNOB_RATIO
        )

    def _center(self):
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    def _drag_to(self, pos):
        v = QPointF(pos) - self._center()
        r = self._radius()
        length = math.hypot(v.x(), v.y())
        if length > r:
            v = v * (r / length)
        self._knob = v
        self.update()
        self.moved.emit(v.x() / r, -v.y() / r)

    def release(self):
        if not self._active:
            return
        self._active = False
        self._knob = QPointF(0.0, 0.0)
        self.update()
        self.released.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._active = True
            self._drag_to(event.pos())

    def mouseMoveEvent(self, event):
        if not self._active:
            return
        self._drag_to(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.release()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = self._center()
        r = self._radius()
        p.setPen(QPen(QColor(120, 120, 120), 2))
        p.setBrush(QBrush(QColor(235, 235, 235)))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(190, 190, 190), 1))
        p.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
        knob_r = r * KNOB_RATIO
        p.setPen(QPen(QColor(40, 90, 160), 2))
        p.setBrush(
            QBrush(QColor(70, 130, 210) if self._active else QColor(150, 175, 210))
        )
        p.drawEllipse(c + self._knob, knob_r, knob_r)


class ControlPanel(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        # Stay under the URDF wheel joint limit (10 rad/s * 0.042 m = 0.42 m/s;
        # turning in place 2 * 0.42 / 0.40 m = 2.1 rad/s). diff_drive_controller
        # itself has no velocity limit configured.
        self.max_linear = node.declare_parameter("max_linear", 0.4).value
        self.max_angular = node.declare_parameter("max_angular", 1.5).value
        self.head = declare_head_config(node)

        self.cmd_pub = node.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)
        self.head_pub = node.create_publisher(JointTrajectory, TRAJECTORY_TOPIC, 10)
        node.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_states,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self._joy = (0.0, 0.0)
        self._keys = set()
        self._moving = False
        self._cmd = (0.0, 0.0)
        self._head_current = None
        self._head_target = None
        self._last_js_time = None
        self._hold_until = 0.0
        self._last_slider_send = 0.0

        self._build_ui()

        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._spin_ros)
        self._spin_timer.start(20)
        self._cmd_timer = QTimer(self)
        self._cmd_timer.timeout.connect(self._publish_cmd)
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._refresh)
        self._ui_timer.start(150)

    # UI ---------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("AhaRobot sim 操作パネル")
        self.setFocusPolicy(Qt.StrongFocus)

        base_box = QGroupBox("台車")
        base_layout = QVBoxLayout(base_box)
        self.pad = JoystickPad()
        self.pad.moved.connect(self._on_pad_moved)
        self.pad.released.connect(self._update_motion)
        base_layout.addWidget(self.pad, 1)
        self.stop_button = QPushButton("停止")
        self.stop_button.setFocusPolicy(Qt.NoFocus)
        self.stop_button.setStyleSheet(
            "background-color: #d9534f; color: white; font-weight: bold;"
        )
        self.stop_button.clicked.connect(self.stop_base)
        base_layout.addWidget(self.stop_button)
        key_label = QLabel(
            f"キー操作: W A S D\n"
            f"最大 {self.max_linear:.1f} m/s, {self.max_angular:.1f} rad/s"
        )
        key_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        base_layout.addWidget(key_label)

        head_box = QGroupBox("頭の向き")
        head_layout = QGridLayout(head_box)
        cfg = self.head
        self.pan_slider = QSlider(Qt.Horizontal)
        self.pan_slider.setRange(
            round(math.degrees(cfg.pan_min)), round(math.degrees(cfg.pan_max))
        )
        self.tilt_slider = QSlider(Qt.Vertical)
        self.tilt_slider.setRange(
            round(math.degrees(cfg.tilt_min)), round(math.degrees(cfg.tilt_max))
        )
        # Minimum (up) at the top.
        self.tilt_slider.setInvertedAppearance(True)
        for slider in (self.pan_slider, self.tilt_slider):
            slider.setFocusPolicy(Qt.NoFocus)
            slider.setValue(0)
            slider.valueChanged.connect(self._on_slider_changed)
            slider.sliderReleased.connect(self._send_slider_target)

        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("左"))
        pan_row.addWidget(self.pan_slider, 1)
        pan_row.addWidget(QLabel("右"))
        tilt_col = QVBoxLayout()
        tilt_col.addWidget(QLabel("上"), 0, Qt.AlignHCenter)
        tilt_col.addWidget(self.tilt_slider, 1, Qt.AlignHCenter)
        tilt_col.addWidget(QLabel("下"), 0, Qt.AlignHCenter)

        self.head_label = QLabel()
        head_layout.addLayout(tilt_col, 0, 0, 2, 1)
        head_layout.addLayout(pan_row, 0, 1)
        head_layout.addWidget(self.head_label, 1, 1)

        preset_row = QHBoxLayout()
        for preset in cfg.presets:
            button = QPushButton(preset.label)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(
                lambda _checked=False, p=preset: self.send_head(p.pan, p.tilt)
            )
            preset_row.addWidget(button)
        head_layout.addLayout(preset_row, 2, 0, 1, 2)

        self.status = QLabel()
        self.status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        # Keep the window at least as large as the layout's minimum.
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        top = QHBoxLayout()
        top.addWidget(base_box, 1)
        top.addWidget(head_box, 1)
        layout.addLayout(top, 1)
        layout.addWidget(self.status)
        self._refresh()

        app = QApplication.instance()
        for screen in app.screens():
            self._watch_dpi(screen)
        app.screenAdded.connect(self._watch_dpi)

    def _watch_dpi(self, screen):
        screen.logicalDotsPerInchChanged.connect(
            lambda _dpi: QTimer.singleShot(0, self._remeasure)
        )

    def _remeasure(self):
        # Qt 5 does not re-measure widgets when the logical DPI changes at
        # runtime (Xft/DPI via XSETTINGS): text is drawn at the new size but
        # cached size hints stay. A FontChange drops them and re-runs the
        # layout, which also grows the window to the new minimum.
        for widget in [self, *self.findChildren(QWidget)]:
            QApplication.sendEvent(widget, QEvent(QEvent.FontChange))
        self.layout().activate()

    # Base ---------------------------------------------------------------

    def _on_pad_moved(self, x, y):
        self._joy = (x, y)
        self._update_motion()

    def _command(self):
        if self.pad.active:
            x, y = self._joy
            return y * self.max_linear, -x * self.max_angular
        if self._keys:
            lin = sum(KEY_DIRECTIONS[k][0] for k in self._keys)
            ang = sum(KEY_DIRECTIONS[k][1] for k in self._keys)
            return (
                lin * KEY_SPEED_RATIO * self.max_linear,
                ang * KEY_SPEED_RATIO * self.max_angular,
            )
        return None

    def _update_motion(self):
        cmd = self._command()
        if cmd is not None:
            self._cmd = cmd
            if not self._moving:
                self._moving = True
                self._publish_cmd()
                self._cmd_timer.start(CMD_PERIOD_MS)
        elif self._moving:
            self._moving = False
            self._cmd_timer.stop()
            self._cmd = (0.0, 0.0)
            self._publish_cmd()

    def _publish_cmd(self):
        self.cmd_pub.publish(base_command(*self._cmd))

    def stop_base(self):
        self._keys.clear()
        self.pad.release()
        self._update_motion()

    def keyPressEvent(self, event):
        if event.key() in KEY_DIRECTIONS:
            if not event.isAutoRepeat():
                self._keys.add(event.key())
                self._update_motion()
            return
        if event.key() == Qt.Key_Space:
            self.stop_base()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() in KEY_DIRECTIONS:
            if not event.isAutoRepeat():
                self._keys.discard(event.key())
                self._update_motion()
            return
        super().keyReleaseEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.stop_base()
        super().changeEvent(event)

    def focusOutEvent(self, event):
        self.stop_base()
        super().focusOutEvent(event)

    # Head ---------------------------------------------------------------

    def _on_joint_states(self, msg):
        pos = head_position(msg)
        if pos is not None:
            self._head_current = pos
            self._last_js_time = time.monotonic()

    def _on_slider_changed(self, _value):
        slider = self.sender()
        if slider.isSliderDown():
            now = time.monotonic()
            if now - self._last_slider_send < SLIDER_THROTTLE_S:
                return
        self._send_slider_target()

    def _send_slider_target(self):
        self.send_head(
            math.radians(self.pan_slider.value()),
            math.radians(self.tilt_slider.value()),
        )

    def send_head(self, pan, tilt):
        pan, tilt, _ = self.head.clamp(pan, tilt)
        duration = self.head.duration(self._head_current, (pan, tilt))
        self.head_pub.publish(self.head.trajectory(pan, tilt, duration))
        now = time.monotonic()
        self._last_slider_send = now
        self._hold_until = now + duration + 0.5
        self._head_target = (pan, tilt)
        self._set_sliders(pan, tilt)

    def _set_sliders(self, pan, tilt):
        for slider, value in ((self.pan_slider, pan), (self.tilt_slider, tilt)):
            if slider.isSliderDown():
                continue
            slider.blockSignals(True)
            slider.setValue(round(math.degrees(value)))
            slider.blockSignals(False)

    # Periodic -------------------------------------------------------------

    def _spin_ros(self):
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def _refresh(self):
        connected = (
            self._last_js_time is not None
            and time.monotonic() - self._last_js_time < JOINT_STATE_TIMEOUT_S
        )
        if connected and time.monotonic() > self._hold_until:
            self._set_sliders(*self._head_current)

        target = describe(*self._head_target) if self._head_target else "-"
        current = describe(*self._head_current) if connected else "-"
        self.head_label.setText(f"現在: {current}\n目標: {target}")

        head_txt = (
            "head_controller 接続中"
            if connected
            else "head_controller 未接続 (/joint_states なし)"
        )
        if self._moving:
            base_txt = f"台車 {self._cmd[0]:+.2f} m/s, {self._cmd[1]:+.2f} rad/s"
        else:
            base_txt = "台車 停止"
        self.status.setText(f"{head_txt}  |  {base_txt}")

    def shutdown(self):
        self._ui_timer.stop()
        self._spin_timer.stop()
        if self._moving:
            self.stop_base()


def main():
    rclpy.init(args=sys.argv)
    app = QApplication(remove_ros_args(sys.argv))
    node = rclpy.create_node("aha_sim_control_panel")
    panel = ControlPanel(node)
    # Preferred size from the font-based size hints (pad above its minimum).
    panel.adjustSize()
    panel.show()
    # The spin timer returns to Python regularly, so SIGINT is handled promptly.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    rc = app.exec_()
    panel.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
