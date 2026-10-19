"""Offline regression tests; no serial port or hardware is accessed."""

import json
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from servo_bus import ServoBus, load_config, make_packet, request


ROOT = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def test_register_reads_and_writes_share_one_serial_transaction_lock(self):
        bus = object.__new__(ServoBus)
        bus.lock = threading.RLock()
        bus.ser = MagicMock()
        first = threading.Event()
        release = threading.Event()
        second_attempt = threading.Event()
        second_request = threading.Event()
        def respond(*args):
            if not first.is_set():
                first.set()
                release.wait(1)
            else:
                second_request.set()
            return bytes(8)
        def write():
            second_attempt.set()
            bus._write(1, 40, [0])
        with patch("servo_bus.request", side_effect=respond):
            reader = threading.Thread(target=bus._read_u16, args=(1, 56))
            writer = threading.Thread(target=write)
            try:
                reader.start()
                self.assertTrue(first.wait(1))
                writer.start()
                self.assertTrue(second_attempt.wait(1))
                self.assertFalse(second_request.wait(.05))
            finally:
                release.set()
                reader.join(1)
                if writer.ident is not None:
                    writer.join(1)
            self.assertTrue(second_request.is_set())

    def test_sync_write_sends_all_targets_in_one_big_endian_packet_without_reply(self):
        bus = object.__new__(ServoBus)
        bus.lock = threading.RLock()
        bus.ser = MagicMock()
        targets = {1: [3, 254, 0, 0, 0, 100], 2: [0, 120, 0, 0, 0, 100]}
        bus._sync_write(42, targets)
        packet = bus.ser.write.call_args.args[0]
        self.assertEqual(packet[2:7], bytes([254, 18, 0x83, 42, 6]))
        self.assertEqual(packet[7:-1], bytes([1, *targets[1], 2, *targets[2]]))
        self.assertEqual(sum(packet[2:]) & 255, 255)
        bus.ser.write.assert_called_once()
        bus.ser.flush.assert_called_once()
        bus.ser.read.assert_not_called()

    def test_servo_alarm_stops_without_repeating_the_command(self):
        for code in (1, 4, 33):
            with self.subTest(code=code):
                ser = MagicMock()
                ser.read.return_value = make_packet(2, code)
                with patch("servo_bus.time.sleep") as sleep:
                    with self.assertRaisesRegex(RuntimeError, ("电压异常" if code & 1 else "0x04") + ".*写入寄存器 0x28"):
                        request(ser, make_packet(2, 3, [40, 1]), 6)
                self.assertEqual(ser.write.call_count, 1)
                sleep.assert_not_called()

    def test_other_servo_alarm_is_rejected_before_status_is_interpreted(self):
        ser = MagicMock()
        ser.read.side_effect = [make_packet(3, 1), make_packet(2, 0)]
        with patch("servo_bus.time.sleep"):
            reply = request(ser, make_packet(2, 3, [40, 0]), 6)
        self.assertEqual(reply[2:5], bytes([2, 2, 0]))
        self.assertEqual(ser.write.call_count, 2)

    def test_other_servo_reply_cannot_confirm_torque_disabled(self):
        ser = MagicMock()
        ser.read.side_effect = [make_packet(2, 0, [0]), make_packet(1, 0, [1])]
        with patch("servo_bus.time.sleep"):
            reply = request(ser, make_packet(1, 2, [40, 1]), 7)
        self.assertEqual(reply[5], 1)
        self.assertEqual(ser.write.call_count, 2)

    def test_inconsistent_length_field_is_rejected_even_with_valid_checksum(self):
        ser = MagicMock()
        reply = bytearray(make_packet(1, 0, [0]))
        reply[3] += 1
        reply[-1] = (~sum(reply[2:-1])) & 255
        ser.read.return_value = bytes(reply)
        with self.assertRaisesRegex(RuntimeError, "长度字段"):
            request(ser, make_packet(1, 2, [40, 1]), 7, attempts=1)

    def test_known_torque_enable_packet(self):
        packet = make_packet(1, 0x03, [0x28, 0x01])
        self.assertEqual(packet, bytes.fromhex("FF FF 01 04 03 28 01 CE"))

    def test_checksum(self):
        packet = make_packet(6, 0x02, [0x38, 0x02])
        self.assertEqual((sum(packet[2:-1]) + packet[-1]) & 0xFF, 0xFF)

    def test_shipping_config(self):
        config = load_config(ROOT / "config.json")
        self.assertEqual(config["baud_rate"], 1_000_000)
        self.assertEqual([item["id"] for item in config["servos"]], [1, 2, 3, 4, 5, 6])
        json.dumps(config, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
