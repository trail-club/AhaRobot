# aha_bringup

## Apple pick demo

Build the workspace and launch the staged task:

```bash
cd /app/overlay_ws
colcon build --symlink-install --packages-up-to aha_bringup
source install/setup.bash
ros2 launch aha_bringup apple_pick.launch.py
```

Use `headless:=true` to run Gazebo without its GUI. The scripted policy opens
the right gripper, raises the lift to clear the table, drives forward about
50 cm using wheel odometry, aligns the gripper with the apple's height, closes
the gripper, then raises the lift by 12 cm.

Success means Gazebo reports the apple's center at least 0.24 m above the
world floor continuously for 1 second after lifting. The policy exits with an
error if it cannot verify that condition within 10 seconds.

Gazebo's detachable-joint system models a grasp by attaching the apple to the
right gripper after the close command. The height check validates that the
simulated apple stays lifted; this demo does not model finger contact or
verify a physical grasp.

The apple starts beyond the right gripper's initial reach. This is a
straight-line odometry approach demo; it does not use navigation, perception,
or inverse kinematics. Gazebo world poses provide the success check; they are
not a perception input to the policy.
