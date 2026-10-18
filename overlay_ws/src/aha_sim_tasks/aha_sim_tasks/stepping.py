"""Pause inference and advance the task world in 100 ms physics batches."""

import time


class GazeboStepper:
    quantum = 0.1

    def __init__(self):
        from gz.transport13 import Node
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.world_control_pb2 import WorldControl

        self.node = Node()
        self.control_type = WorldControl
        self.response_type = Boolean

    def control(self, steps=0):
        request = self.control_type(pause=True, multi_step=steps)
        ok, response = self.node.request(
            "/world/task_tables/control",
            request,
            self.control_type,
            self.response_type,
            5000,
        )
        if not ok or not response.data:
            raise RuntimeError("Gazebo world control request failed")

    def pause(self):
        self.control()

    def advance(self):
        # task_tables.sdf.xacro uses a 1 ms physics timestep.
        self.control(100)


def run_steps(args, node, policy, simulator, evaluator, stepper, recorder=None):
    import rclpy

    deadline = time.monotonic() + args.wall_timeout

    def check():
        if not rclpy.ok() or simulator.poll() is not None:
            raise RuntimeError("Simulator exited during stepped evaluation")
        return time.monotonic() >= deadline

    while True:
        # Streams intentionally stop during inference. Freshness is required
        # after physics advances, rather than while the world is paused.
        camera_deadline = time.monotonic() + args.camera_timeout
        while not node.observation_ready():
            if check():
                return "wall_timeout"
            if time.monotonic() >= camera_deadline:
                raise RuntimeError(
                    f"Camera bundle unavailable after simulation step: {node.camera_diagnostics()}"
                )
            rclpy.spin_once(node, timeout_sec=0.02)
        policy.act(node.observation())
        while True:
            if check():
                return "wall_timeout"
            rclpy.spin_once(node, timeout_sec=0.02)
            action = policy.poll()
            if action is not None:
                if recorder is not None:
                    recorder.action(action, getattr(policy, "phase", None))
                break
        for _ in range(round(args.step_duration / stepper.quantum)):
            node.apply(action)
            # Give ROS bridges an opportunity to deliver the command before physics.
            rclpy.spin_once(node, timeout_sec=0.01)
            target = node.world.sim_time + stepper.quantum
            stepper.advance()
            progress_deadline = time.monotonic() + args.camera_timeout
            while node.world.sim_time < target - 1e-6:
                if check():
                    return "wall_timeout"
                if time.monotonic() >= progress_deadline:
                    raise RuntimeError("Simulation state stopped during physics step")
                rclpy.spin_once(node, timeout_sec=0.02)
                status = evaluator.update(node.world)
                if status is not None:
                    return status
            # Rendering and the image bridge finish asynchronously after physics.
            # Drain their callbacks before queuing another batch; otherwise the
            # RGB/depth/info synchronizers can lose matching frames at a pause.
            drain_deadline = time.monotonic() + 0.05
            while time.monotonic() < drain_deadline:
                if check():
                    return "wall_timeout"
                rclpy.spin_once(node, timeout_sec=0.005)
                status = evaluator.update(node.world)
                if status is not None:
                    return status
            # Do not let a missing sensor be hidden by inference pauses.
            while not node.fresh():
                if check():
                    return "wall_timeout"
                if time.monotonic() >= progress_deadline:
                    raise RuntimeError(
                        f"Simulation observations stopped: {node.stale_sources()}"
                    )
                rclpy.spin_once(node, timeout_sec=0.02)
            if recorder is not None:
                recorder.capture(node.world.sim_time)
