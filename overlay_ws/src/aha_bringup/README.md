# aha_bringup

`sim.launch.py` starts Gazebo, spawns AhaRobot, and activates its controllers.
Build and launch instructions are in the [development-container guide](../../../docs/docker.md).
World and spawn settings are documented in the [Japan Open simulation guide](../../../docs/sobits-rcjo2026.md).
`robot_name` overrides the Gazebo model name (default `aha_robot`).
`cameras:=false` omits the simulated head and wrist camera sensors.
To bridge their images and calibration, use `use_perception:=true`; camera topics
and display options are described in [aha_perception](../aha_perception/README.md).

For approach, apple picking, and pick-and-place demonstrations with independent
success checks, use [aha_sim_tasks](../aha_sim_tasks/README.md). That package
owns the task scene and scripted policy.
