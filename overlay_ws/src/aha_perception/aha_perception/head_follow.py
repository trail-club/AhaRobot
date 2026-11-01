"""Head position that follows commanded trajectories without servos.

Shared by manual_head.py and imu_head.py: moves to the last point of each
head trajectory at the configured max_speed, clamped to the head.yaml limits.
"""

from aha_perception.head_config import PAN_JOINT, TILT_JOINT


class CommandedHead:
    def __init__(self, head, pan=0.0, tilt=0.0):
        self.head = head
        pan, tilt, _ = head.clamp(pan, tilt)
        self.position = [pan, tilt]
        self.target = list(self.position)

    def command(self, trajectory):
        """Take a trajectory_msgs/JointTrajectory; return True if clamped."""
        if not trajectory.points:
            self.target = list(self.position)
            return False
        point = trajectory.points[-1]
        target = list(self.target)
        for name, value in zip(trajectory.joint_names, point.positions):
            if name == PAN_JOINT:
                target[0] = value
            elif name == TILT_JOINT:
                target[1] = value
        pan, tilt, clamped = self.head.clamp(*target)
        self.target = [pan, tilt]
        return clamped

    def step(self, dt):
        """Advance by dt seconds; return [pan, tilt]."""
        step = self.head.max_speed * dt
        for i in range(2):
            delta = self.target[i] - self.position[i]
            self.position[i] += max(-step, min(step, delta))
        return list(self.position)
