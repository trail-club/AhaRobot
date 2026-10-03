#!/usr/bin/env python3
"""AstraArmController zero-point initialization (setupTorque(128)) -- single-arm version that does not automatically enable torque.

diff with  examples/02_record_zero.py:
  - only for 1 arm (default: /dev/ttyUSB0)
  - automatically set_torque(1)
  - set_torque(0) when finish

before exec:
  - initial all joint and wrist
  - ready for emergency bottom

utilize:
  uv run init_arm.py [port]
"""

import os
import sys
import time

REPO = os.path.expanduser("~/aharobot/AhaRobot")
sys.path.insert(0, os.path.join(REPO, "upstream", "astra_controller"))

from astra_controller.arm_controller import ArmController  # noqa: E402

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"

print("=" * 60)
print("About to run zero-point initialization (writes to servo EEPROM and ESP32 LittleFS)")
print("  - Is the gripper opened to 60 mm total (30 mm per jaw)?")
print("    (Init will declare the CURRENT physical position as the 60-mm-open midpoint.)")
print("  - Are joint0/1 set to the URDF zero pose (upper arm tilted 10.6°, forearm perpendicular to the mount x-axis), and the base fixed?")
print("  - Is your hand near the power switch?")
if input("Type yes to continue after confirming all of the above: ").strip() != "yes":
    sys.exit("Cancelled")

ctrl = ArmController(port, do_init=True)  # returns immediately after sending set_torque(128)
print(f"\nInit command sent. Waiting 15 seconds. Firmware output below "
      "(cut power immediately if you see 'Gap is too wide' or 'Maybe cause wrong init_pos0'):\n")
time.sleep(15)

print("\nSending set_torque(0); releasing all servos")
ctrl.set_torque(0)
time.sleep(1.0)
print("Done.")
