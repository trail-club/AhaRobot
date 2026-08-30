#!/usr/bin/env bash
# Keyboard teleop for the mobile base.
#
# Prereq: `sim.launch.py` is already running.
# Run this in an interactive shell (docker exec -it aharobot-aha_project-1 bash),
# then use i/j/k/l/, keys per teleop_twist_keyboard's on-screen help.
exec ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args \
    -p stamped:=true \
    -r /cmd_vel:=/diff_drive_controller/cmd_vel
