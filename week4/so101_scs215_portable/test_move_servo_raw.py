"""Offline tests for move_servo_raw.py; no hardware is accessed."""

import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

import move_servo_raw


class FakeBus:
    opened = []

    def __init__(self, config, port, *, stop_on_close):
        self.config = config
        self.port = port
        self.stop_on_close = stop_on_close
        self.positions = iter([400, 480, 499, 500, 500])
        self.calls = []
        self.closed = False
        self.__class__.opened.append(self)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.closed = True

    def read_position(self, servo_id):
        self.calls.append(("read_position", servo_id))
        return next(self.positions)

    def mode(self, servo_id):
        self.calls.append(("mode", servo_id))
        return "position", 0, 1023

    def move_position(self, servo_id, position, duration_ms, speed):
        self.calls.append(("move_position", servo_id, position, duration_ms, speed))

    def heartbeat(self, servo_id):
        self.calls.append(("heartbeat", servo_id))

    def stop(self, selector):
        self.calls.append(("stop", selector))
        return []


class RawMoveTests(unittest.TestCase):
    def setUp(self):
        FakeBus.opened.clear()
        self.config = {
            "port": "auto",
            "baud_rate": 1_000_000,
            "servos": [{"id": item, "name": f"joint_{item}"} for item in range(1, 7)],
        }

    def arguments(self, **overrides):
        values = {
            "id": 4,
            "position": 500,
            "duration_ms": 1000,
            "speed": 100,
            "tolerance": 5,
            "config": "unused.json",
            "port": None,
            "execute": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_preview_does_not_open_serial_bus(self):
        with patch("move_servo_raw.load_config", return_value=self.config), \
             patch("move_servo_raw.get_servo_bus_class") as get_bus_class, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            result = move_servo_raw.run(self.arguments())
        self.assertEqual(result, 0)
        get_bus_class.assert_not_called()
        self.assertIn("不会连接串口", output.getvalue())

    def test_execute_uses_raw_target_and_waits_for_stable_feedback(self):
        with patch("move_servo_raw.load_config", return_value=self.config), \
             patch("move_servo_raw.get_servo_bus_class", return_value=FakeBus), \
             patch("move_servo_raw.time.sleep"), \
             patch("move_servo_raw.time.monotonic", side_effect=[0.0, 0.1, 0.2, 0.3, 0.4]), \
             contextlib.redirect_stdout(io.StringIO()):
            result = move_servo_raw.run(self.arguments(execute=True))
        self.assertEqual(result, 0)
        bus = FakeBus.opened[0]
        self.assertIn(("move_position", 4, 500, 1000, 100), bus.calls)
        self.assertEqual(bus.calls.count(("heartbeat", 4)), 4)
        self.assertIn(("stop", 4), bus.calls)
        self.assertFalse(bus.stop_on_close)
        self.assertTrue(bus.closed)

    def test_unconfigured_id_is_rejected_before_serial_is_opened(self):
        with patch("move_servo_raw.load_config", return_value=self.config), \
             patch("move_servo_raw.get_servo_bus_class") as get_bus_class:
            with self.assertRaisesRegex(ValueError, "不在配置中"):
                move_servo_raw.run(self.arguments(id=7, execute=True))
        get_bus_class.assert_not_called()

    def test_named_arguments_and_dry_run_aliases(self):
        args = move_servo_raw.parse_arguments([
            "--id", "4", "--position", "500", "--duration", "3000", "--dry-run"
        ])
        self.assertEqual((args.id, args.position, args.duration_ms), (4, 500, 3000))
        self.assertTrue(args.dry_run)
        self.assertFalse(args.execute)

    def test_parser_rejects_position_1024(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                move_servo_raw.parse_arguments(["4", "1024"])


if __name__ == "__main__":
    unittest.main()
