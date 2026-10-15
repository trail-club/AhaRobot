# aha_bringup

`sim.launch.py` starts Gazebo, spawns AhaRobot, and activates its controllers.
Build and launch instructions are in the [development-container guide](../../../docs/docker.md).
World and spawn settings are documented in the [Japan Open simulation guide](../../../docs/sobits-rcjo2026.md).
`robot_name` overrides the Gazebo model name (default `aha_robot`).
Gazebo receives the model from [`aha_gazebo/model.py`](../aha_gazebo/aha_gazebo/model.py): joint position
limits are 0.05 m / rad wider than the URDF and the lifts have 60 N of Coulomb friction, so joints commanded
to a URDF limit keep following commands. robot_state_publisher, ros2_control and MoveIt use the URDF limits.
The lifts' position commands start at 0 m (`initial_value` in `aha_description/urdf/ros2_control.xacro`), so
`lift_controller` returns them to 0 m after they dip when the robot lands at the spawn.
`cameras:=false` omits the simulated head and wrist camera sensors.
To bridge their images and calibration, use `use_perception:=true`; camera topics
and display options are described in [aha_perception](../aha_perception/README.md).

`use_manip:=true` starts [MoveIt right-arm planning and execution](../aha_manipulation/README.md).

For approach, apple picking, and pick-and-place demonstrations with independent
success checks, use [aha_sim_tasks](../aha_sim_tasks/README.md). That package
owns the task scene and scripted policy.
