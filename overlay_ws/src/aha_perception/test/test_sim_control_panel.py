"""sim_control_panel.py: base command and slider hold (no Qt window, no ROS transport)."""

import importlib.util
import math
import os
import sys
import unittest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Test the source tree's aha_perception module (also without a build).
sys.path.insert(0, PKG)

_spec = importlib.util.spec_from_file_location(
    "sim_control_panel", os.path.join(PKG, "scripts", "sim_control_panel.py")
)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)


class BaseCommandTest(unittest.TestCase):
    def test_stamp_is_zero(self):
        # diff_drive_controller replaces a zero stamp with its own clock, so
        # cmd_vel_timeout works on sim time.
        msg = panel.base_command(0.2, -0.5)
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nanosec), (0, 0))
        self.assertEqual(msg.twist.linear.x, 0.2)
        self.assertEqual(msg.twist.angular.z, -0.5)

    def test_stop_command(self):
        msg = panel.base_command(0, 0)
        self.assertEqual((msg.twist.linear.x, msg.twist.angular.z), (0.0, 0.0))
        self.assertEqual((msg.header.stamp.sec, msg.header.stamp.nanosec), (0, 0))


class CommandHoldTest(unittest.TestCase):
    TARGET = (0.5, 0.2)

    def setUp(self):
        self.hold = panel.CommandHold()
        self.hold.start(self.TARGET, duration=1.0, now=100.0)

    def test_idle(self):
        self.assertFalse(panel.CommandHold().active((0.0, 0.0), 0.0))

    def test_holds_past_the_duration_until_reached(self):
        # Sim at RTF 0.5: halfway when the wall-clock duration is over.
        self.assertTrue(self.hold.active((0.25, 0.1), 101.6))
        self.assertTrue(self.hold.active(None, 102.0))
        self.assertFalse(self.hold.active((0.5 - math.radians(1.0), 0.2), 102.5))
        # Once reached, the sliders follow the joint states again.
        self.assertFalse(self.hold.active((0.0, 0.0), 102.6))

    def test_timeout(self):
        timeout = panel.HOLD_TIMEOUT_RATIO * 1.0 + panel.HOLD_TIMEOUT_MARGIN_S
        self.assertTrue(self.hold.active((0.0, 0.0), 100.0 + timeout - 0.01))
        self.assertFalse(self.hold.active((0.0, 0.0), 100.0 + timeout + 0.01))

    def test_new_target_restarts(self):
        self.assertFalse(self.hold.active(self.TARGET, 100.5))
        self.hold.start((-0.5, 0.0), duration=1.0, now=101.0)
        self.assertTrue(self.hold.active(self.TARGET, 101.1))


if __name__ == "__main__":
    unittest.main()
