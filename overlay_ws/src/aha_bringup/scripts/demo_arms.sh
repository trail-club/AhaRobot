#!/usr/bin/env bash
# Send a small demo trajectory to head, left arm, and right arm.
# Prereq: sim.launch.py is running with all controllers active.
set -e

echo "[demo] head pan -> +0.4 rad"
ros2 action send_goal --feedback /head_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [joint_head_pan, joint_head_tilt],
                 points: [{positions: [0.4, 0.0], time_from_start: {sec: 2}}]}}" \
  > /dev/null

echo "[demo] left arm to a small stretch pose"
ros2 action send_goal /left_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [joint_l2, joint_l3, joint_l4, joint_l5, joint_l6],
                 points: [{positions: [0.3, -0.5, 0.0, 0.3, 0.0], time_from_start: {sec: 3}}]}}" \
  > /dev/null

echo "[demo] right arm mirror"
ros2 action send_goal /right_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [joint_r2, joint_r3, joint_r4, joint_r5, joint_r6],
                 points: [{positions: [-0.3, 0.5, 0.0, -0.3, 0.0], time_from_start: {sec: 3}}]}}" \
  > /dev/null

echo "[demo] lift up 0.2 m"
ros2 action send_goal /lift_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [joint_l1, joint_r1],
                 points: [{positions: [0.2, 0.2], time_from_start: {sec: 2}}]}}" \
  > /dev/null

echo "[demo] open right gripper"
ros2 topic pub --once /right_gripper_controller/commands \
  std_msgs/msg/Float64MultiArray '{data: [0.06, -0.06]}' > /dev/null

echo "[demo] done"
