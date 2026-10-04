"""Fake STS servo bus on a pty for tests without hardware.

FakeStsBus({12: 2100, 13: 2000}).port is a tty path that answers ping, read,
write and sync write like STS3215 servos at those start positions. With torque
on, each servo moves toward its goal at the goal speed (steps/s, 0 = max).
Failure hooks: set_max_speed() slows a servo down, set_silent() drops every
packet (no reply, no write) like a disconnected bus.
"""

import os
import select
import threading
import time
import tty

from aha_servo import sts

MAX_SPEED = 3000
STEP_S = 0.005


class FakeServo:
    def __init__(self, servo_id, position):
        self.id = servo_id
        self.mem = bytearray(256)
        self.pos = float(position)
        self.mem[sts.ADDR_GOAL_POS : sts.ADDR_GOAL_POS + 2] = sts.le16(position)
        self.mem[sts.ADDR_TORQUE_LIMIT : sts.ADDR_TORQUE_LIMIT + 2] = sts.le16(1000)
        self.mem[sts.ADDR_VOLT] = 120
        self.mem[sts.ADDR_TEMP] = 30
        self.goals = []  # goal positions in write order
        self.max_speed = MAX_SPEED  # steps/s, caps the goal speed
        self._store_state(0.0)

    def _store_state(self, speed):
        pos = round(self.pos)
        spd = min(abs(round(speed)), 0x7FFF) | (0x8000 if speed < 0 else 0)
        self.mem[sts.ADDR_STATE : sts.ADDR_STATE + 4] = sts.le16(pos) + sts.le16(spd)

    @property
    def torque(self):
        return self.mem[sts.ADDR_TORQUE_ENABLE]

    @property
    def goal(self):
        return sts.u16(self.mem, sts.ADDR_GOAL_POS)

    def write(self, addr, data):
        self.mem[addr : addr + len(data)] = data
        if addr <= sts.ADDR_GOAL_POS < addr + len(data):
            self.goals.append(self.goal)

    def step(self, dt):
        if not self.torque:
            self._store_state(0.0)
            return
        speed = min(sts.u16(self.mem, sts.ADDR_GOAL_SPEED) or MAX_SPEED, self.max_speed)
        delta = self.goal - self.pos
        move = max(-speed * dt, min(speed * dt, delta))
        self.pos += move
        self._store_state(move / dt if dt > 0 else 0.0)


class FakeStsBus:
    def __init__(self, servos):
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        self.port = os.ttyname(self.slave)
        self.servos = {sid: FakeServo(sid, pos) for sid, pos in servos.items()}
        self.lock = threading.Lock()
        self.bad_packets = 0
        self.silent = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        os.close(self.master)
        os.close(self.slave)

    def servo(self, servo_id):
        return self.servos[servo_id]

    def move_by_hand(self, servo_id, position):
        """Set the position as if moved by hand (visible to the next read)."""
        with self.lock:
            servo = self.servos[servo_id]
            servo.pos = float(position)
            servo._store_state(0.0)

    def set_max_speed(self, servo_id, steps_per_s=MAX_SPEED):
        with self.lock:
            self.servos[servo_id].max_speed = steps_per_s

    def set_silent(self, silent):
        """Drop every incoming packet while True (servos keep moving)."""
        with self.lock:
            self.silent = silent

    def _reply(self, servo_id, params=b""):
        os.write(self.master, sts.build_packet(servo_id, 0, params))

    def _handle(self, servo_id, instruction, params):
        servo = self.servos.get(servo_id)
        if instruction == sts.INST_PING and servo:
            self._reply(servo_id)
        elif instruction == sts.INST_READ and servo:
            addr, length = params[0], params[1]
            self._reply(servo_id, bytes(servo.mem[addr : addr + length]))
        elif instruction == sts.INST_WRITE and servo:
            servo.write(params[0], params[1:])
            self._reply(servo_id)
        elif instruction == sts.INST_SYNC_WRITE and servo_id == sts.BROADCAST_ID:
            addr, length = params[0], params[1]
            body = params[2:]
            for i in range(0, len(body) - length, length + 1):
                target = self.servos.get(body[i])
                if target:
                    target.write(addr, body[i + 1 : i + 1 + length])

    def _run(self):
        buf = bytearray()
        last = time.monotonic()
        while not self._stop.is_set():
            if select.select([self.master], [], [], STEP_S)[0]:
                try:
                    buf += os.read(self.master, 1024)
                except OSError:
                    pass
            with self.lock:
                if self.silent:
                    buf.clear()
                while True:
                    try:
                        packet, used = sts.parse_status(buf)
                    except sts.StsError:
                        self.bad_packets += 1
                        del buf[:2]
                        continue
                    del buf[:used]
                    if packet is None:
                        break
                    self._handle(*packet)
                now = time.monotonic()
                for servo in self.servos.values():
                    servo.step(now - last)
                last = now
