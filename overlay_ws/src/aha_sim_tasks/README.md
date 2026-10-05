# aha_sim_tasks

Simple Gazebo Harmonic tasks with independent success checks and a Python policy
interface. Run inside the [development container](../../../docs/docker.md).

## Evaluate the scripted policy

```bash
cd /app/overlay_ws
colcon build --symlink-install --packages-up-to aha_sim_tasks
source install/setup.bash
ros2 run aha_sim_tasks evaluate --task all --policy scripted --episodes 2 \
  --output /tmp/aha-evaluation/results.json
```

The runner starts a fresh simulator for **each episode**, including the robot,
controllers, object, and contact state.
The evaluation launch activates the command controllers as one group and waits
for their active states before calling the policy.
Simulation and scoring use ROS domain 87 by default (`--ros-domain-id` changes it).
Each policy runs in a separate Python subprocess, using ROS domain 88 by default
and a different Gazebo transport partition. `--policy-ros-domain-id` changes the
worker domain; otherwise it is the evaluation domain plus one, wrapping 232 to 0.
The two domains must differ and be unused by other ROS applications.
Both Gazebo partitions are unique per episode.
The scene publishes `/clock` at 20 Hz together with the scoring state so that
clock traffic remains bounded on CPU-only hosts.

All tasks use the same two-table scene: a brown source table, green destination
table, and apple. Entity names, object/table positions, robot spawn, and scripted
gripper settings are defined in [config/scene.json](config/scene.json).
The launch expands [worlds/task_tables.sdf.xacro](worlds/task_tables.sdf.xacro)
from that configuration and passes the configured robot spawn explicitly.
The default run renders the head RGB-D and both wrist RGB cameras with Ogre2.
Headless runs use EGL and need a working rendering backend; GPU acceleration is
recommended. Software rendering is supported but can substantially slow episodes.
`--no-cameras` disables camera sensors, the rendering system, and camera
observations for a CPU-only physics evaluation.
Use `--no-headless` to display Gazebo.

To watch pick-and-place, run:

```bash
ros2 run aha_sim_tasks evaluate --task place_apple --policy scripted \
  --episodes 1 --no-headless --camera-view --output /tmp/aha-evaluation/watch.json
```

`--camera-view` opens a separate Gazebo window with head and both wrist RGB
feeds selected automatically. The viewer inherits the episode's Gazebo partition;
an independently started `gz gui -s ImageDisplay` cannot discover these isolated
topics. The image viewer also works with the default headless server when the
3D window is not needed. It requires cameras and a working GUI display.

The server and GUI windows stop at the end of an episode, before the next task starts.
Cleanup also stops Gazebo processes that leave the original process group,
identified by the episode's unique transport partition. Configure the
container's display using the development-container GUI instructions linked above.

`--task pick_apple` selects one task. `--list-tasks` prints the installed definitions.
The JSON report records each task's criteria, episode status, final measurements,
errors, artifact directory, and overall success rate. Logs and individual results
are stored beside the report under `<report-stem>_episodes/`. Exit code is zero
only when every requested episode succeeds; failure returns 1. A `noop` policy
is available to exercise the timeout path.

## Tasks and success criteria

Definitions are in [config/tasks.json](config/tasks.json). All distances are metres,
angles radians, and task deadlines/hold durations use **simulation time**.
Scoring uses Gazebo world poses and measured finger contacts, independently of the
policy. Poses, contact count, and simulation timestamp arrive in one atomic
`/evaluation/state` sample. A policy's completion claim cannot produce success. Duplicate or
out-of-order samples cannot advance a hold; a gap over 0.5 s in state samples
restarts it.

| Task | Every scoring sample satisfies this for the hold duration | Hold | Deadline |
| --- | --- | --- | --- |
| `approach_apple` | Robot world x/y within 0.04 m of the source approach target derived from the scene and gripper offset, yaw within 0.08 rad of the spawn heading, base speed at most 0.02 m/s | 1 s | 45 s |
| `pick_apple` | The contact window includes both right-gripper fingers and the apple's center is at least 0.24 m above the floor | 1 s | 60 s |
| `place_apple` | Previously lifted to 0.24 m with both fingers in a contact window; now contact windows are empty, center within 0.08 m in x and y and 0.02 m in z of the configured destination position, speed at most 0.03 m/s | 2 s | 90 s |

Startup has a 90 s wall deadline. Missing/stale observations cause an error;
`--wall-timeout` (default 240 s after startup) also bounds slow or paused
simulations. Policy import, construction, and `reset` have a separate 120 s wall
deadline (`--policy-startup-timeout`); each `act` call has a 30 s wall deadline
(`--policy-timeout`). Increase these for models with longer loading or inference.
The evaluator continues receiving ROS updates and scoring while inference runs;
at most one policy call is outstanding. Worker errors, crashes, invalid commands,
and policy timeouts fail the episode with status `error`.
The parent process also bounds the entire episode. Simulator and policy processes
are stopped on completion or error, including descendants in either partition.

The scripted policy opens the right gripper, raises the lift, approaches using
wheel odometry with heading correction, lowers and closes the gripper, then
lifts. It aims the head down and right toward the tables and right gripper;
`scripted.head_pan` and `scripted.head_tilt` in `config/scene.json` set the angles
in radians. A level head sees only the gray background above the low tables.
For placement it drives to the destination, lowers, opens, and withdraws
the gripper by raising the lift. It reads proprioception and odometry and derives
waypoints from static scene geometry and the configured gripper offset; it never
reads live world state. This baseline drives straight along the spawn heading.
Scene targets requiring lateral motion or backward driving fail explicitly;
they need a policy supporting those motions. `--list-tasks` displays resolved
numeric scoring targets.

## Grasp model

The apple is a free rigid body. Finger collisions, friction, controller motion,
and gravity determine whether it is lifted, slips, or falls when the fingers open.
The world loads `aha_task_evaluation`, a passive Gazebo system that requests the
apple collision's contact data from the physics engine. At each physics step it
collects the distinct right-gripper fingers touching the apple. Every 50 ms it
publishes their union (0, 1, or 2) together with end-of-window poses and a timestamp,
then clears the contact window. Thus a single solver step without contact does
not erase evidence from the same window; no evidence carries into the next one.
Two fingers means each touched at least once during that window, rather than
necessarily touching on the same physics step. Release requires an entire window
with no finger contact, followed by the task's release hold duration. Table contact and
multiple contact points on a single finger do not count as a two-finger grasp.
The system never attaches the object or changes its motion.
Unresolved model/link/collision paths are logged on first failure, whenever the
missing set changes, and every 5 wall seconds while resolution keeps failing.

The world uses primitive table/apple geometry and the upstream robot collision
meshes. Apple friction is set to 1.0; geometry, friction, and actuator response
have not been calibrated against real hardware. Contact-based scoring verifies
simulated holding and release, rather than real robot grasp quality.

## Connect another policy

`--policy my_package.my_policy:create_policy` imports and calls a factory only in
the policy worker. Built-in policies use the same worker boundary. The returned
object implements this interface from [api.py](aha_sim_tasks/api.py):

```python
from aha_sim_tasks.api import Action, Observation


class MyPolicy:
    def reset(self, task_id: str, instruction: str) -> None:
        self.instruction = instruction

    def act(self, observation: Observation) -> Action:
        return Action(linear_velocity=0.0, joint_positions={"joint_r1": 0.18})


def create_policy():
    return MyPolicy()
```

Place the module on the runner's Python path, which the worker inherits. Each
episode creates and resets a new policy. `act` runs at up to 10 Hz in wall time.
Observation fields are simulation time, task ID, language instruction, joint positions/velocities,
wheel odometry `(x, y, yaw)` in `odom`, `cameras`, and optional `head_image`. Joint mappings
are read-only snapshots. Object/world poses, finger contacts, and score are absent
from policy observations.

Only these observation fields cross a private IPC socket; the worker returns an
`Action`, validated by the evaluator before publishing controller commands.
Joint mappings remain read-only snapshots, and optional images retain their
`sensor_msgs/msg/Image` type after serialization. Policy stdout/stderr and
tracebacks are written to `policy.log` in the episode's artifact directory.
The worker has no evaluator Python frames or evaluator-owned `RosEnvironment`
object, and its default ROS/Gazebo connections cannot discover evaluation topics.
No ROS topics are bridged into its domain: ROS-based policy components must receive their input
from `Observation` and return commands through `Action`.

This separation prevents accidental use of privileged state by team-owned
policies. It is not a security sandbox: the processes share the host and
filesystem, and policy code could deliberately join the evaluation transports.

By default, `observation.cameras` is a read-only mapping with keys `head`,
`left_wrist`, and `right_wrist`. Each `CameraObservation` contains:

- `image`: RGB `sensor_msgs/msg/Image` with its simulation timestamp and optical frame.
- `camera_info`: matching `sensor_msgs/msg/CameraInfo` with intrinsics.
- `base_transform`: `geometry_msgs/msg/TransformStamped` from `base_link` to that
  optical frame, looked up at the image timestamp.
- `depth_image`: aligned `32FC1` depth in metres for the head; `None` for the wrists.

RGB, calibration, and head depth are paired by exact timestamp and optical frame.
Policy calls wait until all views have transforms, camera timestamps differ by
at most 0.1 s, and images are no more than 0.5 s behind the simulation clock
(up to 0.1 s ahead is allowed for clock transport delay).
Joint states and odometry remain the latest snapshots, rather than samples
synchronized to image exposure. Camera liveness has a 10 s wall deadline
(`--camera-timeout` changes it), allowing dropped frames on slow renderers while
the simulation-time age/skew limits still prevent stale images reaching a policy.
Joint, odometry, and scoring-state liveness retain their 2 s wall deadline.
A stopped required camera stream fails the episode.
`head_image` aliases `cameras["head"].image` for existing policies. All camera
messages cross the same private IPC boundary; evaluation topics remain excluded.
Camera streams and simulated mounts are described in
[aha_perception](../aha_perception/README.md#シミュレーション).

With `--no-cameras`, `cameras` is empty and `head_image` is `None`.
For a legacy external image publisher in the evaluation ROS domain, use
`--no-cameras --head-image-topic /your/image/topic`; the latest Image becomes
`head_image`, without calibration or transform pairing. The runner waits for
that stream and treats a stopped stream as an error.

Actions command base linear/angular velocity and named position targets for the
arms, lifts, head, and grippers. Positions use the existing controller joint
names. Omitted joint targets hold their previous command (initially the measured
position); omitted base velocities are zero.

If a joint target remains unreached, the adapter reissues its trajectory after
3 s of simulation time. Each position trajectory has a 2 s duration.
Commands outside joint limits or base limits (0.3 m/s, 0.5 rad/s), unknown joints, and non-finite values fail the
episode. A result has status `success`, `timeout`, `wall_timeout`, or `error`.

Add a JSON task entry to reuse an existing criterion (`approach`, `pick`, `place`).
New criteria are implemented in [evaluation.py](aha_sim_tasks/evaluation.py);
scene geometry and baseline settings belong in `config/scene.json`, with world
structure in `worlds/`. The bundled scripted policy
supports the three listed tasks; another task needs a corresponding policy.

## Checks

The package's unit tests run through the repository's `make test` / colcon checks.
The evaluation command above runs the robot, controllers, physical contacts,
and success checks in Gazebo. A failure baseline can be run with:

```bash
ros2 run aha_sim_tasks evaluate --task pick_apple --policy noop \
  --output /tmp/aha-evaluation/noop.json
```

This is expected to return 1 with a simulation-time timeout and zero successes.
