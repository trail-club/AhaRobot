"""Minimal Feetech STS (STS3215) protocol over a stdlib serial port.

No pyserial / scservo_sdk (the dev container has neither): the port is a raw
termios fd. Register addresses and the access pattern follow
tests/perception/test_head_servo.py, which passes on the real head.

Instruction packet: FF FF id len instr params... checksum
Status packet:      FF FF id len error params... checksum
  len = len(params) + 2, checksum = ~(id + len + instr/error + params) & 0xFF.
Multi-byte values are little endian (SCS protocol end 0).

Not thread safe: callers serialize access to one StsBus.
"""

import os
import select
import termios
import time

BROADCAST_ID = 0xFE

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_WRITE = 0x83

ADDR_MODE = 33
ADDR_TORQUE_ENABLE = 40
ADDR_ACC = 41
ADDR_GOAL_POS = 42
ADDR_GOAL_TIME = 44
ADDR_GOAL_SPEED = 46
ADDR_TORQUE_LIMIT = 48
# pos(2) speed(2) load(2), always read as one block: separate 2-byte reads can
# pick up a late reply from the previous read with a valid checksum
# (tools/servo/README.md).
ADDR_STATE = 56
STATE_LEN = 6
ADDR_VOLT = 62
ADDR_TEMP = 63

POS_MIN, POS_MAX = 0, 4095
# Goal speed 0 means "no limit" on STS servos, so commanded speeds start at 1.
SPEED_MIN, SPEED_MAX = 1, 32767

BAUDS = (
    9600,
    19200,
    38400,
    57600,
    115200,
    230400,
    460800,
    500000,
    576000,
    921600,
    1000000,
)
# The subset this platform's termios can set.
BAUD_RATES = {
    baud: getattr(termios, f"B{baud}") for baud in BAUDS if hasattr(termios, f"B{baud}")
}


class StsError(Exception):
    """Communication failure: timeout, bad checksum or malformed reply."""


def checksum(body):
    return ~sum(body) & 0xFF


def build_packet(servo_id, instruction, params=b""):
    body = bytes([servo_id, len(params) + 2, instruction]) + bytes(params)
    return b"\xff\xff" + body + bytes([checksum(body)])


def ping_packet(servo_id):
    return build_packet(servo_id, INST_PING)


def read_packet(servo_id, addr, length):
    return build_packet(servo_id, INST_READ, bytes([addr, length]))


def write_packet(servo_id, addr, data):
    return build_packet(servo_id, INST_WRITE, bytes([addr]) + bytes(data))


def sync_write_packet(addr, data_by_id):
    """One broadcast write of equal-length data to several servos (no reply)."""
    lengths = {len(d) for d in data_by_id.values()}
    if len(lengths) != 1:
        raise ValueError("sync write needs data of one common, non-zero length")
    params = bytes([addr, lengths.pop()])
    for servo_id, data in data_by_id.items():
        params += bytes([servo_id]) + bytes(data)
    return build_packet(BROADCAST_ID, INST_SYNC_WRITE, params)


def parse_status(buf):
    """Find the first status packet in buf.

    Returns ((id, error, params), consumed) when a whole packet is present,
    otherwise (None, discard) where discard leading bytes can be dropped.
    Raises StsError on a bad length or checksum.
    """
    start = buf.find(b"\xff\xff")
    if start < 0:
        # Keep a trailing 0xFF: it may be the first header byte.
        return None, len(buf) - 1 if buf.endswith(b"\xff") else len(buf)
    # FF FF FF id ...: the id is never 0xFF, so the header starts later.
    while start + 2 < len(buf) and buf[start + 2] == 0xFF:
        start += 1
    if len(buf) < start + 4:
        return None, start
    length = buf[start + 3]
    if length < 2:
        raise StsError(f"bad status length {length}")
    end = start + 4 + length
    if len(buf) < end:
        return None, start
    body = bytes(buf[start + 2 : end - 1])
    if checksum(body) != buf[end - 1]:
        raise StsError(f"checksum mismatch in reply {bytes(buf[start:end]).hex(' ')}")
    return (body[0], body[2], body[3:]), end


def le16(value):
    """Two bytes, little endian; raises ValueError outside 0..65535."""
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{value} does not fit in 16 bits")
    return bytes([value & 0xFF, value >> 8])


def u16(data, offset=0):
    return data[offset] | (data[offset + 1] << 8)


def sign_magnitude(value, sign_bit):
    """Feetech encodes negatives as magnitude plus a sign bit."""
    return -(value & ((1 << sign_bit) - 1)) if value & (1 << sign_bit) else value


def decode_state(data):
    """(position, speed, load) from the 6-byte block at ADDR_STATE.

    Position in steps, speed in steps/s, load in 0.1 % of max torque.
    """
    return (
        sign_magnitude(u16(data, 0), 15),
        sign_magnitude(u16(data, 2), 15),
        sign_magnitude(u16(data, 4), 10),
    )


def goal_block(position, speed, acc):
    """7 bytes from ADDR_ACC: acc, goal position, goal time (0 = use speed), goal speed."""
    position = min(max(int(position), POS_MIN), POS_MAX)
    speed = min(max(int(speed), SPEED_MIN), SPEED_MAX)
    return bytes([min(max(int(acc), 0), 254)]) + le16(position) + le16(0) + le16(speed)


class SerialPort:
    """Raw 8N1 serial port on a termios fd (also works on a pty)."""

    def __init__(self, path, baud):
        if baud not in BAUD_RATES:
            raise ValueError(
                f"unsupported baud rate {baud} (supported: {sorted(BAUD_RATES)})"
            )
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(self.fd)
            attrs[0] = 0  # iflag: no break/parity/CR handling, no XON/XOFF
            attrs[1] = 0  # oflag: no output processing
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[3] = 0  # lflag: no echo, no canonical mode, no signals
            attrs[4] = attrs[5] = BAUD_RATES[baud]
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except Exception:
            os.close(self.fd)
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def flush_input(self):
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except termios.error as e:
            # e.g. the USB adapter was unplugged
            raise OSError(*e.args) from e

    def write(self, data, timeout=0.5):
        view = memoryview(data)
        deadline = time.monotonic() + timeout
        while view:
            try:
                n = os.write(self.fd, view)
                view = view[n:]
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StsError(f"{self.path}: write timed out")
                select.select([], [self.fd], [], remaining)

    def read(self, timeout):
        """Bytes available within timeout (possibly empty)."""
        if not select.select([self.fd], [], [], max(timeout, 0.0))[0]:
            return b""
        try:
            return os.read(self.fd, 256)
        except BlockingIOError:
            return b""


class StsBus:
    """STS servos on one serial bus.

    read/write raise StsError after `retries` failed attempts; ping returns
    False instead. `status_error` holds the error byte of the last reply
    (servo alarms such as overload or overheat; 0 when fine).
    """

    def __init__(self, port, baud, timeout=0.05, retries=3):
        self.serial = SerialPort(port, baud)
        self.timeout = timeout
        self.retries = max(1, retries)
        self.status_error = 0

    def close(self):
        self.serial.close()

    def _transact(self, packet, servo_id, n_params):
        self.serial.flush_input()
        self.serial.write(packet)
        buf = bytearray()
        deadline = time.monotonic() + self.timeout
        while True:
            status, used = parse_status(buf)
            del buf[:used]
            if status is not None:
                sid, error, params = status
                # A TX echo (half-duplex adapter) parses as a status with the
                # instruction as error byte; a ping's echo would pass as the
                # reply. A ping reply with error 0x01 has the same bytes and
                # is dropped too.
                if build_packet(sid, error, params) == packet:
                    continue
                # Ignore stray replies (other id or length) and keep reading.
                if sid == servo_id and len(params) == n_params:
                    self.status_error = error
                    return params
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StsError(
                    f"ID{servo_id}: no reply within {self.timeout * 1000:.0f} ms"
                )
            buf += self.serial.read(remaining)

    def _retry(self, packet, servo_id, n_params):
        error = None
        for _ in range(self.retries):
            try:
                return self._transact(packet, servo_id, n_params)
            except StsError as e:
                error = e
        raise error

    def ping(self, servo_id):
        try:
            self._retry(ping_packet(servo_id), servo_id, 0)
        except StsError:
            return False
        return True

    def read(self, servo_id, addr, length):
        return self._retry(read_packet(servo_id, addr, length), servo_id, length)

    def write(self, servo_id, addr, data):
        self._retry(write_packet(servo_id, addr, data), servo_id, 0)

    def sync_write(self, addr, data_by_id):
        self.serial.write(sync_write_packet(addr, data_by_id))

    def read_state(self, servo_id):
        """(position, speed, load), see decode_state."""
        return decode_state(self.read(servo_id, ADDR_STATE, STATE_LEN))

    def set_torque(self, servo_id, enable):
        self.write(servo_id, ADDR_TORQUE_ENABLE, bytes([1 if enable else 0]))

    def write_goal_position(self, servo_id, position):
        self.write(
            servo_id, ADDR_GOAL_POS, le16(min(max(int(position), POS_MIN), POS_MAX))
        )

    def sync_write_goals(self, goals, acc):
        """goals: {id: (position, speed)}; position in steps, speed in steps/s."""
        if goals:
            self.sync_write(
                ADDR_ACC,
                {
                    sid: goal_block(pos, speed, acc)
                    for sid, (pos, speed) in goals.items()
                },
            )
