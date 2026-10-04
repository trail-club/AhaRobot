"""sim_control_panel.py: base command message (no Qt window, no ROS transport)."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
