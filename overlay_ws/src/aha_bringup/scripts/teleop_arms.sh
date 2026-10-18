#!/usr/bin/env bash
# Keyboard teleop for both arms through arm_node.
#
# Prereq: `ros2 launch aha_bringup arms.launch.py` is already publishing
# /joint_states. Run this in an interactive shell. `ros2 launch` does not
# deliver keypresses, so this is `ros2 run` on purpose.
#
#   w/s e/d r/f t/g u/j y/h   left arm
#   W/S E/D R/F T/G U/J Y/H   right arm
#   space hold, 0 torque off, q quit
exec ros2 run aha_arm_teleop keyboard_teleop "$@"
