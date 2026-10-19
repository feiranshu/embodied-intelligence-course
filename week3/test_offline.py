"""Offline regression tests; no serial port or hardware is accessed."""

import json
import threading
import unittest
from pathlib import Path

from homework_control import interpolate_positions, move_one_servo, normalized_targets
from servo_bus import ServoBus, load_config, make_packet


ROOT = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
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

    def test_sync_position_packet_is_big_endian(self):
        class FakeSerial:
            def __init__(self):
                self.packet = b""

            def reset_input_buffer(self):
                pass

            def write(self, packet):
                self.packet = packet
                return len(packet)

            def flush(self):
                pass

        bus = ServoBus.__new__(ServoBus)
        bus.lock = threading.RLock()
        bus.ser = FakeSerial()
        bus.sync_write_positions({1: 400, 2: 800})
        self.assertEqual(
            bus.ser.packet,
            bytes.fromhex("FF FF FE 0A 83 2A 02 01 01 90 02 03 20 91"),
        )


class HomeworkControlTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")

    def test_normalized_targets_include_reversed_fixture(self):
        self.config["servos"][0]["position_calibration"] = {
            "end_a": 307,
            "end_b": 695,
            "home": 501,
        }
        self.config["servos"][2]["position_calibration"] = {
            "end_a": 884,
            "end_b": 248,
            "home": 566,
        }
        targets = normalized_targets(self.config, [0, 0, 0, 0, 0, 0])
        self.assertEqual(targets[1], 307)
        self.assertEqual(targets[3], 884)
        targets = normalized_targets(self.config, [1, 1, 1, 1, 1, 1])
        self.assertEqual(targets[1], 695)
        self.assertEqual(targets[3], 248)
        targets = normalized_targets(self.config, [0.5] * 6)
        self.assertEqual(targets[3], 566)

    def test_normalized_value_must_be_in_range(self):
        with self.assertRaises(ValueError):
            normalized_targets(self.config, [0, 0, 0, 0, 0, 1.01])

    def test_interpolation_reaches_exact_target(self):
        start = {1: 100, 2: 200}
        target = {1: 300, 2: 100}
        self.assertEqual(interpolate_positions(start, target, 10, 10), target)

    def test_move_one_only_addresses_selected_servo_and_releases_torque(self):
        class FakeBus:
            def __init__(self):
                self.position_packets = []
                self.torque_calls = []
                self.readings = iter([800, 800, 800, 800, 800, 850])

            def mode(self, servo_id):
                self.mode_id = servo_id
                return "position", 0, 1023

            def read_position(self, servo_id):
                self.read_ids = getattr(self, "read_ids", []) + [servo_id]
                return next(self.readings)

            def sync_write_positions(self, positions):
                self.position_packets.append(dict(positions))

            def set_torque(self, servo_id, enabled):
                self.torque_calls.append((servo_id, enabled))

        item = self.config["servos"][0]
        item["position_calibration"] = {"end_a": 700, "end_b": 1000, "home": 850}
        bus = FakeBus()
        actual = move_one_servo(bus, item, 0.5, duration=0, rate_hz=25)
        self.assertEqual(actual, 850)
        self.assertTrue(all(set(packet) == {1} for packet in bus.position_packets))
        self.assertEqual(bus.position_packets[-1], {1: 850})
        self.assertEqual(bus.torque_calls, [(1, True), (1, False)])


if __name__ == "__main__":
    unittest.main()
