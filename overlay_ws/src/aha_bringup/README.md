# aha_bringup

`sim.launch.py` starts Gazebo, spawns AhaRobot, and activates its controllers.
Build and launch instructions are in the [development-container guide](../../../docs/docker.md).
World and spawn settings are documented in the [Japan Open simulation guide](../../../docs/sobits-rcjo2026.md).
`robot_name` overrides the Gazebo model name (default `aha_robot`).
Gazebo receives the model from [`aha_gazebo/model.py`](../aha_gazebo/aha_gazebo/model.py), built from the same
xacro output as `robot_description`: joint position limits are 0.005 m / rad wider than the URDF and the lifts
hold up to 10 kN, so joints commanded to a URDF limit keep following commands.
robot_state_publisher, ros2_control and MoveIt use the URDF limits;
MoveIt's state input moves positions up to the physics limits onto them (`limit_tolerance` of aha_manipulation).
`initial_lift` sets both lift joints' initial position in metres (default `0.0`).
`cameras:=false` omits the simulated head and wrist camera sensors.
To bridge their images and calibration, use `use_perception:=true`; camera topics
and display options are described in [aha_perception](../aha_perception/README.md).

`use_manip:=true` starts [MoveIt right-arm planning and execution](../aha_manipulation/README.md).
Adding `manip_rviz:=true` (only together with `use_manip:=true`) also opens the [RViz manipulation view](../aha_manipulation/README.md#rviz操作画面)
and, unless `use_perception:=true` already does, bridges the camera images.
With `use_perception:=true` the perception RViz still opens, but only the manipulation
view's control panel starts.

For approach, apple picking, and pick-and-place demonstrations with independent
success checks, use [aha_sim_tasks](../aha_sim_tasks/README.md). That package
owns the task scene and scripted policy.

## Real robot on the frame

`real.launch.py` starts the real robot as currently built: arms and head on a frame of 30 mm aluminum profiles
on a table, without the mobile base and the lifts ([measurements](../../../docs/context/frame.md)).

```bash
ros2 launch aha_bringup real.launch.py
ros2 launch aha_bringup real.launch.py arms:=false head:=manual    # no servo boards
```

It runs robot_state_publisher with the frame model, a static `odom` → `base_footprint` (the frame does not move),
`joint_state_publisher` merging `/head_controller/joint_states` and the arms' `/<side>/arm/joint_states`
into `/joint_states`, [`arms.launch.py`](launch/arms.launch.py), the head servos, the D435i, the point cloud,
plane detection and RViz (topics in [ROS interfaces](../../../docs/interfaces.md#実機の頭部)).
`base_footprint` is on the floor and `base_link` on the table top below the camera column.

| Argument | Default | |
| --- | --- | --- |
| `frame_config` | `aha_description/config/frame_table_202610.yaml` | Frame model, see below |
| `arms` / `use_left` / `use_right` | `true` | Arms (`arms.launch.py`) |
| `right_device` / `left_device` / `head_port` | `/dev/ttyUSB2` / `/dev/ttyUSB0` / `/dev/ttyUSB1` | The three boards as first connected ([record](../../../docs/context/motor.md)); the numbers follow the USB enumeration, so check them after replugging (`/dev/serial/by-path/` names the hub port) |
| `trajectory_bridge` / `record` | `true` / `false` | Passed to `arms.launch.py` (whose own `record` default is `true`) |
| `head` | `servo` | `servo`, `manual` (commanded angles as the state), `imu` (tilt from the D435i IMU, no servos) or `none` |
| `camera` / `planes` / `rviz` | `true` | D435i (its IMU only with `head:=imu`) and point cloud / plane detection (`furniture_config`) / RViz |

The arms need the host-side sign patch on `upstream/astra_controller`
([patches](../../../patches/README.md#astra_controller用パッチ)); without it the arms read mirrored angles.
The head board must be in Serial Forwarding mode after each power-up ([aha_perception](../aha_perception/README.md#接続)).

### Frame model

[`aha_description/frame.py`](../aha_description/aha_description/frame.py) rewrites the xacro URDF (`sim:=false`)
with the YAML `frame_config`: the table height, the head pan axis, an optional camera mount, each arm's shoulder axis
and mount yaw (the lifts become fixed joints) and the frame members as slotted profiles (the collision is their outer box).
The head base (Astra's HeadYaw part) is cut out of the upstream body mesh at build time.
The YAML marks which values were measured and which come from photos.

### Calibration

With `real.launch.py` running (arms with torque off: `ros2 topic pub --once /<side>/arm/torque_enable std_msgs/msg/UInt8 "{data: 0}"`):

```bash
# Arm mount yaw: upper arm forward with the forearm in line, then outward, per arm
ros2 run aha_description frame_calibrate.py --config <source>/aha_description/config/frame_table_202610.yaml [--write]
# Camera tilt and height from the table top (0.38 m) or the floor (0), head pointed at a free area of it
ros2 run aha_perception camera_plane_calibrate.py --reference 0.38 --config <frame yaml> [--write]
```

`frame_calibrate.py` also checks the shoulder sign (+90° to the left arm's outside, −90° for the right arm) and the elbow zero.
`camera_plane_calibrate.py` writes `camera` (link_head_tilt → camera_link); the plane does not give the camera's
horizontal position or yaw, and the result holds for the head pose it was measured in.
Rebuild or restart the launch after `--write` to load the new values.
