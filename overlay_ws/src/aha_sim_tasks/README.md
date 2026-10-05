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
controllers, object, and grasp state.
The evaluation launch activates the command controllers as one group and waits
for their active states before calling the policy.
Evaluation uses ROS domain 87 by default
(`--ros-domain-id` changes it) and a separate Gazebo transport partition per
episode. The domain must be unused by other ROS applications.
The scene publishes `/clock` at 20 Hz together with the scoring poses so that
clock traffic remains bounded on CPU-only hosts.

All tasks use the same two-table scene; the robot starts at `(0, 0, 0.05)` m with
yaw zero. The apple
starts at `(0.962, -0.426, 0.16)` m on the brown source table. The green destination
table is 0.4 m farther along x. No GPU is required for the default server-only run.
Use `--no-headless` to display Gazebo.

To watch pick-and-place, run:

```bash
ros2 run aha_sim_tasks evaluate --task place_apple --policy scripted \
  --episodes 1 --no-headless --output /tmp/aha-evaluation/watch.json
```

The server and GUI stop at the end of an episode, before the next task starts.
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
Scoring uses Gazebo world poses and the grasp system state, independently of the
policy. A policy's completion claim cannot produce success. Duplicate or
out-of-order samples cannot advance a hold; a gap over 0.5 s in pose samples
restarts it.

| Task | Success, continuously for the hold duration | Hold | Deadline |
| --- | --- | --- | --- |
| `approach_apple` | Robot world x/y within 0.04 m of `(0.49, 0)`, yaw within 0.08 rad of zero, base speed at most 0.02 m/s | 1 s | 45 s |
| `pick_apple` | Apple grasped and its center at least 0.24 m above the floor | 1 s | 60 s |
| `place_apple` | Previously grasped and lifted to 0.24 m; now released, center within 0.08 m in x and y and 0.02 m in z of `(1.362, -0.426, 0.16)`, speed at most 0.03 m/s | 2 s | 90 s |

Startup has a 90 s wall deadline. Missing/stale observations cause an error;
`--wall-timeout` (default 240 s after startup) also bounds slow or paused
simulations. The parent process imposes an additional deadline that covers a
policy adapter hanging. Simulator processes are stopped on completion or error.

The scripted policy opens the right gripper, raises the lift, approaches using
wheel odometry with heading correction, lowers and closes the gripper, then
lifts. For placement it drives another 0.4 m, lowers, opens, and withdraws the
gripper by raising the lift. It reads proprioception and odometry and uses fixed
scene waypoints.

## Grasp model

The world loads `aha_task_grasp`, an AhaRobot-specific Gazebo system. The apple
starts released. After observing open fingers, the system creates a fixed grasp
joint only when both fingers close to at most 0.047 m displacement and the apple
center is within 0.055 m of the right gripper center. Opening both fingers to at
least 0.055 m releases the joint. The system reads measured joint positions;
there is no policy-facing attach command and no object teleportation.

This proximity latch is an abstraction for testing task execution and scoring.
It does not validate finger contact, grasp force, or real robot grasp quality.
The world uses primitive table/apple geometry and the current robot model.

## Connect another policy

`--policy my_package.my_policy:create_policy` imports and calls a factory. The
returned object implements this interface from [api.py](aha_sim_tasks/api.py):

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

Place the module on the evaluation process's Python path. Each episode creates
and resets a new policy. `act` runs at up to 10 Hz in wall time. Observation fields
are simulation time, task ID, language instruction, joint positions/velocities,
wheel odometry `(x, y, yaw)` in `odom`, and optional `head_image`. Joint mappings
are read-only snapshots. Object/world poses, grasp state, and score are absent
from policy observations.

The built-in scene does not render a head camera. If an image publisher is
available in the evaluation ROS domain, `--head-image-topic /your/image/topic`
subscribes with sensor-data QoS and passes the latest `sensor_msgs/msg/Image` as
`head_image`; otherwise it is `None`. With a topic specified, evaluation waits
for images and treats a stopped image stream as an error. This API does not
require a particular model or image representation.

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
scene changes belong in `worlds/` and the launch. The bundled scripted policy
supports the three listed tasks; another task needs a corresponding policy.

## Checks

The package's unit tests run through the repository's `make test` / colcon checks.
The evaluation command above runs the actual robot, controllers, grasp plugin,
and success checks in Gazebo. A failure baseline can be run with:

```bash
ros2 run aha_sim_tasks evaluate --task pick_apple --policy noop \
  --output /tmp/aha-evaluation/noop.json
```

This is expected to return 1 with a simulation-time timeout and zero successes.
