"""AccelStepper 1.64 (moveTo / run) for the lift fake, in wall-clock time.

A port of AccelStepper::computeNewSpeed and runSpeed as the
AstraLiftController firmware uses them: moveTo with a new target recomputes
the speed, every step recomputes it again. The motor only speeds up while it
can still stop at the target with `acceleration`, so it lags a moving target.
Units are steps, steps/s and steps/s^2.
"""

import math


class AccelStepper:
    def __init__(self, max_speed, acceleration, position=0):
        self.position = position
        self.target = position
        self.speed = 0.0
        self.acceleration = acceleration
        self.c0 = 0.676 * math.sqrt(2.0 / acceleration)  # s
        self.cmin = 1.0 / max_speed
        self.cn = 0.0
        self.n = 0
        self.direction = 1
        self.interval = 0.0  # s, 0 when stopped
        self.last_step = None

    def move_to(self, target, now):
        self.run_until(now)
        if target != self.target:
            self.target = target
            self._compute()

    def run_until(self, now):
        if self.last_step is None or not self.interval:
            self.last_step = now
            return
        while self.interval and self.last_step + self.interval <= now:
            self.last_step += self.interval
            self.position += self.direction
            self._compute()
        if not self.interval:
            self.last_step = now

    def _compute(self):
        distance = self.target - self.position
        to_stop = int(self.speed * self.speed / (2.0 * self.acceleration))
        if distance == 0 and to_stop <= 1:
            self.interval = 0.0
            self.speed = 0.0
            self.n = 0
            return
        if distance > 0:
            if self.n > 0:
                if to_stop >= distance or self.direction < 0:
                    self.n = -to_stop
            elif self.n < 0 and to_stop < distance and self.direction > 0:
                self.n = -self.n
        elif distance < 0:
            if self.n > 0:
                if to_stop >= -distance or self.direction > 0:
                    self.n = -to_stop
            elif self.n < 0 and to_stop < -distance and self.direction < 0:
                self.n = -self.n
        if self.n == 0:
            self.cn = self.c0
            self.direction = 1 if distance > 0 else -1
        else:
            self.cn = max(self.cn - 2.0 * self.cn / (4.0 * self.n + 1), self.cmin)
        self.n += 1
        self.interval = self.cn
        self.speed = self.direction / self.cn
