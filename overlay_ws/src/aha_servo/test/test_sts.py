"""STS protocol: packet bytes, reply parsing and a round trip over the fake bus."""

import os
import unittest

from aha_servo import sts
from fake_bus import FakeStsBus


def hexbytes(text):
    return bytes.fromhex(text)


class PacketTests(unittest.TestCase):
    # Examples from the Feetech SCS/STS protocol manual.
    def test_ping(self):
        self.assertEqual(sts.ping_packet(1), hexbytes("ff ff 01 02 01 fb"))

    def test_read_present_position(self):
        self.assertEqual(
            sts.read_packet(1, 0x38, 2), hexbytes("ff ff 01 04 02 38 02 be")
        )

    def test_write_goal_position_time_speed(self):
        # goal 2048, time 0, speed 1000 at address 42
        data = sts.le16(2048) + sts.le16(0) + sts.le16(1000)
        self.assertEqual(
            sts.write_packet(1, sts.ADDR_GOAL_POS, data),
            hexbytes("ff ff 01 09 03 2a 00 08 00 00 e8 03 d5"),
        )

    def test_sync_write(self):
        data = sts.le16(2048) + sts.le16(1000)
        packet = sts.sync_write_packet(0x2A, {1: data, 2: data})
        # len = (4 + 1) * 2 + 4; checksum = ~(0xfe + 0x0e + 0x83 + ... ) & 0xff
        self.assertEqual(
            packet, hexbytes("ff ff fe 0e 83 2a 04 01 00 08 e8 03 02 00 08 e8 03 59")
        )

    def test_sync_write_needs_equal_lengths(self):
        with self.assertRaises(ValueError):
            sts.sync_write_packet(0x2A, {1: b"\x00", 2: b"\x00\x00"})

    def test_goal_block(self):
        self.assertEqual(
            sts.goal_block(2048, 400, 30),
            bytes([30]) + sts.le16(2048) + sts.le16(0) + sts.le16(400),
        )
        # Position cut to 0..4095, speed at least 1 (0 would mean unlimited).
        self.assertEqual(
            sts.goal_block(5000, 0, 300),
            bytes([254]) + sts.le16(4095) + sts.le16(0) + sts.le16(1),
        )
        self.assertEqual(sts.goal_block(-20, 10, 0)[1:3], sts.le16(0))


class ParseTests(unittest.TestCase):
    REPLY = hexbytes("ff ff 01 04 00 00 08 f2")  # ID1, no error, params 00 08

    def test_reply(self):
        status, used = sts.parse_status(bytearray(self.REPLY))
        self.assertEqual(status, (1, 0, hexbytes("00 08")))
        self.assertEqual(used, len(self.REPLY))

    def test_garbage_and_split(self):
        buf = bytearray(hexbytes("12 34 ff") + self.REPLY[:5])
        status, discard = sts.parse_status(buf)
        self.assertIsNone(status)
        del buf[:discard]
        buf += self.REPLY[5:]
        status, _ = sts.parse_status(buf)
        self.assertEqual(status, (1, 0, hexbytes("00 08")))

    def test_no_header_keeps_trailing_ff(self):
        self.assertEqual(sts.parse_status(bytearray(hexbytes("01 02 ff"))), (None, 2))
        self.assertEqual(sts.parse_status(bytearray(hexbytes("01 02"))), (None, 2))

    def test_extra_ff_before_id(self):
        status, _ = sts.parse_status(bytearray(b"\xff" + self.REPLY))
        self.assertEqual(status[0], 1)

    def test_bad_checksum(self):
        with self.assertRaises(sts.StsError):
            sts.parse_status(bytearray(self.REPLY[:-1] + b"\x00"))

    def test_decode_state(self):
        data = sts.le16(2048) + sts.le16(0x8000 | 100) + sts.le16(0x400 | 50)
        self.assertEqual(sts.decode_state(data), (2048, -100, -50))
        self.assertEqual(
            sts.decode_state(sts.le16(10) + sts.le16(5) + sts.le16(7)), (10, 5, 7)
        )


class FakeBusRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeStsBus({12: 2100, 13: 2000})
        self.bus = sts.StsBus(self.fake.port, 921600, timeout=0.05, retries=2)

    def tearDown(self):
        self.bus.close()
        self.fake.close()

    def test_ping(self):
        self.assertTrue(self.bus.ping(12))
        self.assertTrue(self.bus.ping(13))
        self.assertFalse(self.bus.ping(14))

    def test_read_write(self):
        self.assertEqual(self.bus.read_state(12)[0], 2100)
        self.assertEqual(self.bus.read(13, sts.ADDR_VOLT, 1), bytes([120]))
        self.bus.set_torque(12, True)
        self.assertEqual(self.fake.servo(12).torque, 1)
        self.bus.write_goal_position(12, 2100)
        self.assertEqual(self.fake.servo(12).goal, 2100)

    def test_sync_write_goals(self):
        self.bus.set_torque(12, True)
        self.bus.set_torque(13, True)
        self.bus.sync_write_goals({12: (2200, 3000), 13: (1900, 3000)}, acc=0)
        # Sync write has no reply; the next read goes through the same queue.
        self.bus.read_state(12)
        self.assertEqual(self.fake.servo(12).goal, 2200)
        self.assertEqual(self.fake.servo(13).goal, 1900)

    def test_ignores_stray_reply(self):
        # A late reply of another servo in the input must not be taken as ours.
        self.bus.serial.flush_input = lambda: None
        os.write(self.fake.master, sts.build_packet(13, 0, sts.le16(1234) + bytes(4)))
        self.assertEqual(self.bus.read_state(12)[0], 2100)

    def test_unsupported_baud(self):
        with self.assertRaises(ValueError):
            sts.StsBus(self.fake.port, 12345)


if __name__ == "__main__":
    unittest.main()
