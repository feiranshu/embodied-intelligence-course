import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from jarvis.hardware import Camera, HardwareMotion


def hardware_config():
    return {"enabled": True, "max_step_mm": 3, "max_travel_mm": 6,
            "command_timeout_s": 0.1,
            "calibration": {"verified": True,
                            "image_to_base_mm": [[10, 0], [0, -10], [0, 0]],
                            "workspace_mm": [[-10, 10], [-10, 10], [-10, 10]]}}


class FakeArm:
    def __init__(self, config, bus):
        self.xyz = np.zeros(3)
        self.cancel = threading.Event()
        self.last_move = None
        self.stops = 0
        self.model = SimpleNamespace(forward=lambda positions: positions)

    def positions(self):
        return self.xyz.copy()

    def stop(self):
        self.stops += 1
        self.cancel.set()


class FakeRunner:
    def __init__(self, arm):
        self.arm = arm
        self.thread = None
        self.error = None
        self.calls = []

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def submit(self, command):
        self.arm.cancel.clear()
        self.calls.append(command)
        def run():
            self.arm.xyz += command[1]
            self.arm.last_move = {"projected": False}
        self.thread = threading.Thread(target=run)
        self.thread.start()
        return True

    def wait(self):
        self.thread.join()
        if self.error:
            raise self.error

    def stop(self):
        self.arm.stop()


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch("vendor.scs215.advanced.Arm", FakeArm),
                        patch("vendor.scs215.advanced.CommandRunner", FakeRunner),
                        patch("vendor.scs215.servo_bus.ServoBus")]
        self.mocks = [item.start() for item in self.patches]
        for item in self.patches:
            self.addCleanup(item.stop)

    def test_unverified_calibration_never_opens_port(self):
        for change in ({"enabled": False}, {"calibration": {"verified": False}}):
            with self.assertRaises(ValueError):
                HardwareMotion({**hardware_config(), **change})
        self.mocks[-1].assert_not_called()

    def test_invalid_matrix_never_opens_port(self):
        config = hardware_config()
        config["calibration"]["image_to_base_mm"] = [[1, 0], [2, 0], [3, 0]]
        with self.assertRaises(ValueError):
            HardwareMotion(config)
        self.mocks[-1].assert_not_called()

    def test_calibrated_direction_and_step_bound(self):
        motion = HardwareMotion(hardware_config())
        self.addCleanup(motion.close)
        motion.step(0.8, 0.6)
        delta = motion.runner.calls[0][1]
        np.testing.assert_allclose(delta, [2.4, -1.8, 0])
        self.assertAlmostEqual(np.linalg.norm(delta), 3)
        self.assertEqual(motion.last_feedback, {"projected": False})

    def test_stop_remains_latched_despite_runner_reset(self):
        motion = HardwareMotion(hardware_config())
        self.addCleanup(motion.close)
        motion.stop()
        with self.assertRaises(InterruptedError):
            motion.step(0.1, 0.1)
        self.assertEqual(motion.runner.calls, [])
        self.assertTrue(motion.arm.cancel.is_set())

    def test_workspace_rejected_before_motion(self):
        config = hardware_config()
        config["calibration"]["workspace_mm"] = [[-1, 1], [-1, 1], [-1, 1]]
        motion = HardwareMotion(config)
        self.addCleanup(motion.close)
        with self.assertRaises(ValueError):
            motion.step(0.5, 0)
        self.assertEqual(motion.runner.calls, [])
        self.assertTrue(motion._stopped.is_set())

    def test_total_travel_rejected_before_extra_motion(self):
        motion = HardwareMotion(hardware_config())
        self.addCleanup(motion.close)
        motion.step(0.5, 0)
        motion.step(-0.5, 0)
        with self.assertRaises(ValueError):
            motion.step(0.1, 0)
        self.assertEqual(len(motion.runner.calls), 2)

    def test_runner_failure_is_not_reported_as_success(self):
        motion = HardwareMotion(hardware_config())
        self.addCleanup(motion.close)
        motion.runner.error = RuntimeError("feedback failure")
        with self.assertRaisesRegex(RuntimeError, "feedback failure"):
            motion.step(0.1, 0)
        self.assertTrue(motion._stopped.is_set())

    def test_timeout_stops_motion_and_rejects_later_commands(self):
        motion = HardwareMotion(hardware_config())
        self.addCleanup(motion.close)
        def submit(command):
            motion.runner.thread = threading.Thread(target=motion.arm.cancel.wait)
            motion.runner.thread.start()
            return True
        motion.runner.submit = submit
        with self.assertRaises(TimeoutError):
            motion.step(0.1, 0)
        motion.runner.thread.join(timeout=1)
        self.assertFalse(motion.runner.busy)
        with self.assertRaises(InterruptedError):
            motion.step(0.1, 0)

    def test_failed_stop_feedback_is_propagated(self):
        motion = HardwareMotion(hardware_config())
        self.addCleanup(motion.close)
        motion.runner.error = RuntimeError("move failure")
        with patch.object(motion.runner, "stop", side_effect=RuntimeError("torque feedback failure")):
            with self.assertRaisesRegex(RuntimeError, "move failure.*torque feedback failure"):
                motion.step(0.1, 0)
        self.assertTrue(motion._stopped.is_set())


class CameraTests(unittest.TestCase):
    def test_disconnect_does_not_refresh_timestamp_or_sequence(self):
        capture = SimpleNamespace(isOpened=lambda: True, set=lambda *args: None,
                                  release=lambda: None)
        first = np.zeros((12, 16, 3), dtype=np.uint8)
        supplied = threading.Event()
        def read():
            if not supplied.is_set():
                supplied.set()
                return True, first
            return False, None
        capture.read = read
        with patch("cv2.VideoCapture", return_value=capture):
            camera = Camera(timeout_s=0.05, max_age_s=1)
        self.addCleanup(camera.close)
        frame = camera.read()
        with self.assertRaises(TimeoutError):
            camera.read()
        self.assertEqual(camera._frame.sequence, frame.sequence)
        self.assertEqual(camera._frame.captured_at, frame.captured_at)


if __name__ == "__main__":
    unittest.main()
