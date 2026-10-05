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
the gripper, then raises the lift by 13 cm.
Gripper stages wait for measured finger positions before proceeding.

Success means Gazebo reports the apple's center at least 0.24 m above the
world floor continuously for 1 second after lifting. The policy exits with an
error if it cannot verify that condition within 10 seconds.

The apple remains a free rigid body: finger collisions, friction, and gravity
determine whether closing the gripper holds it. The height check verifies that
the simulated apple stays lifted. For independent scoring that also checks
two-finger contact and release, use [aha_sim_tasks](../aha_sim_tasks/README.md).

The apple starts beyond the right gripper's initial reach. This is a
straight-line odometry approach demo; it does not use navigation, perception,
or inverse kinematics. Gazebo world poses provide the success check; they are
not a perception input to the policy.
