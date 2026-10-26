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
and mount yaw (the lifts become fixed joints), optional joint reading offsets of the vertical arm joints (`joint_offsets`)
and the frame members as slotted profiles (the collision is their outer box).
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

The camera's horizontal position and yaw come next from clicks on the gripper tip, with the head in its working pose.
The RViz of `real.launch.py` sends Publish Point to the head look-at (a click turns the head), so start RViz on its own:

```bash
ros2 launch aha_bringup real.launch.py rviz:=false
rviz2 -d $(ros2 pkg prefix --share aha_perception)/rviz/sim_view.rviz -f base_footprint
ros2 run aha_perception camera_arm_calibrate.py --config <frame yaml> [--write]
```

Hide the RobotModel display (a click on the mesh returns the model's point, not the camera's).
For each of 6 or more poses spread over both arms' reach, hold the gripper (torque off) still with its tip pointing down,
and click the tip in the point cloud with Publish Point.
The script pairs each click with the nearest of `link_ree` / `link_lee` from TF (`--offset` moves the point in that link,
`--radius` toward the camera, as the click lands on the surface facing it), rejects pairs taken while the gripper or the head moved,
and on `q` fits the camera's horizontal shift and yaw (`u` drops the last pair).
It prints per-pair residuals and the height difference (not applied), and `--write` writes `camera` with the tilt and height kept.
With the tip pointing down, an error in the tip position along the gripper only shows in the height difference.

When the arm model and the arm in the point cloud still disagree (mount yaw or a joint zero off), fit them from the cloud,
after the camera tilt and height (and, with both arms usable, its horizontal pose) are calibrated:

```bash
ros2 run aha_perception arm_cloud_calibrate.py --config <frame yaml> --arm right [--free shoulder,elbow[,wrist]] [--save snaps.npz] [--write]
```

Hold the arm (torque off) still in 3 or more clearly different poses over a table, varying the shoulder and the elbow,
with the forearm and the gripper in the camera's view and nothing in front of them, and press Enter for each
(`u` drops the last snapshot, `q` fits). A snapshot is the cloud in base_footprint and the arm's joints averaged over 0.5 s;
it is rejected when the joints moved, the cloud is stale, the pose repeats an earlier one, or the head moved
(joints reading exactly the same for a second only warn: an arm at rest reads so, as do the frozen readings after an arm board reset).
The fit puts the cloud points (above the table under the arm, near the model; `--exclude-fingers` leaves the fingers out) on the model surface
and prints per-snapshot RMS distances and inlier counts, the deltas (`shoulder` is added to `yaw`, `elbow` / `wrist` to
`joint_offsets` of joint_<l|r>3 / 4), their correlation and how much they change with a snapshot left out.
`--write` is refused while a check warns (one pose rarely separates the shoulder from the elbow; `--force` writes anyway)
or when `/robot_description` differs from the config. `--load snaps.npz` fits saved snapshots again.
With one arm, a camera yaw error moves the arm like a shoulder offset: `--free camera` (horizontal shift and yaw,
written to `camera`) only works with the shoulder fixed, e.g. `--free elbow,camera`.

Rebuild or restart the launch after `--write` to load the new values.
