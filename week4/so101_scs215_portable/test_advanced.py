"""Offline acceptance checks for speech dispatch, motion and visual feedback."""

import contextlib
import copy
import io
import json
import threading
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from advanced import Arm, CommandRunner, Kinematics, PositionHoldTimeout, load_advanced, parse_command
from servo_bus import GOAL_POSITION, RUNNING_TIME, ServoBus, load_config
from vision_control import (VisionHTTPServer, correction, detect_ball,
                            pointing_target, synthetic_frame, track, track_joints,
                            validate_jacobian)
from voice_control import dispatch_result, listen, result_text


class FeedbackBus(ServoBus):
    def __init__(self, config, raw):
        self.config = config
        self.items = {int(item["id"]): item for item in config["servos"]}
        self.lock = threading.RLock()
        self.running_id = None
        self.last_heartbeat = 0
        self.raw = list(raw)
        self.writes = []
        self.batches = []
        self.wheel = False
        self.mode_by_id = {}
        self.wheel_positions = {}
        self.stall = False
        self.fail = False
        self.registers = {}

    def _write(self, servo_id, address, values):
        self.writes.append((servo_id, address, list(values)))
        for offset, value in enumerate(values):
            self.registers[servo_id, address + offset] = value
        if self.fail and address == GOAL_POSITION:
            raise RuntimeError("injected communication fault")
        if address == GOAL_POSITION and not self.stall and self.mode(servo_id)[0] == "position":
            self.raw[servo_id - 1] = (values[0] << 8) | values[1]
        if address == 11:
            self.mode_by_id[servo_id] = values == [0, 0]

    def _sync_write(self, address, values):
        self.batches.append((address, copy.deepcopy(values)))
        for joint, data in values.items():
            self._write(joint, address, data)

    def _read_u8(self, servo_id, address):
        return self.registers.get((servo_id, address), 1 if address == 48 else 0)

    def _read_u16(self, servo_id, address):
        return self._read_u8(servo_id, address) * 256 + self._read_u8(servo_id, address + 1)

    def mode(self, servo_id):
        return ("wheel", 0, 0) if self.mode_by_id.get(servo_id, self.wheel) else ("position", 0, 1023)

    def read_position(self, servo_id):
        if (self.running_id in (servo_id, "all") and self.mode(servo_id)[0] == "wheel"
                and self._read_u8(servo_id, 40) != 0
                and self._read_u16(servo_id, RUNNING_TIME) != 0
                and self.wheel_positions.get(servo_id)):
            self.raw[servo_id - 1] = self.wheel_positions[servo_id].pop(0)
        return self.raw[servo_id - 1]


def sampled_endpoint(bus, joint, trace, prompt):
    input(prompt)
    values = [bus.read_position(joint) for _ in range(5)]
    target = values[2]
    delta = (target - trace.raw + 512) % 1024 - 512
    while abs(delta) > 100:
        step = 100 if delta > 0 else -100
        trace.update((trace.raw + step) % 1024)
        delta -= step
    trace.update(target)
    return trace.raw, trace.position


class AdvancedTests(unittest.TestCase):
    def test_cartesian_six_direction_paths_with_base_clearance(self):
        config = load_advanced()
        model = Kinematics(config)
        start = np.array(config["poses"]["初始状态"])
        origin = model.forward(start)
        for axis in range(3):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    current = start.copy()
                    for distance in range(5, 51, 5):
                        delta = np.eye(3)[axis] * sign * distance
                        target = model.inverse(origin + delta, current, allow_crossing=False)
                        np.testing.assert_array_equal(target[4:], start[4:])
                        np.testing.assert_allclose(target - current, model.unwrap(target) - model.unwrap(current))
                        np.testing.assert_allclose(model.forward(target) - origin, delta, atol=2)
                        current = target

    def test_cartesian_home_returns_to_saved_base_direction(self):
        for base in (1022, 962, 950):
            with self.subTest(base=base):
                config = load_advanced()
                raw = [base, 899, 320, 660, 77, 272]
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                with patch.object(arm.cancel, "wait", return_value=False):
                    arm.execute(("home", "初始状态"))
                self.assertEqual(bus.raw, config["poses"]["初始状态"])
                self.assertEqual(any(j == 1 and (a == GOAL_POSITION or (a == 40 and v == [1]))
                                     for j, a, v in bus.writes), base != config["poses"]["初始状态"][0])

    def test_cartesian_left_from_forward_pose_moves_base_without_folding_wrist(self):
        config = load_advanced()
        config["poses"]["初始状态"][0] = 1022
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        before = bus.raw.copy()
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.execute(("move", [0, 10, 0]))
        self.assertLess(bus.raw[0], before[0] - 4)
        self.assertLessEqual(abs(bus.raw[3] - before[3]), 1)
        self.assertTrue(arm.last_joint_move["targets_verified"])
        self.assertEqual(arm.last_joint_move["phase"], "tracking")

    def test_cartesian_wrist_departure_stops_before_waiting_for_arrival(self):
        config = load_advanced()
        raw = config["poses"]["初始状态"]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        target = arm.model.inverse(arm.model.forward(raw) + [0, 5, 0], raw, allow_crossing=False)
        sync = bus._sync_write
        clock = [0.0]
        def departed(address, values):
            sync(address, values)
            if address == GOAL_POSITION:
                bus.raw[0] = raw[0]
                bus.raw[3] = int(target[3]) + 20
        def wait(seconds):
            clock[0] += seconds
            return False
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.prepare(cartesian=True)
        with patch.object(bus, "_sync_write", side_effect=departed), \
             patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(PositionHoldTimeout, "ID 4 反馈偏离本次运动范围"):
                arm.hold_pose(target, cartesian=True)
        self.assertLess(clock[0], 0.2)
        self.assertEqual(arm.last_joint_move["feedback"][3], int(target[3]) + 20)
        self.assertTrue(all(bus._read_u8(j, 40) == 0 for j in range(1, 7)))

    def test_cartesian_partial_feedback_cannot_report_full_distance(self):
        arm = Arm(load_advanced())
        origin = arm.model.forward(arm.positions())
        with patch.object(arm, "move_cartesian", return_value=False), \
             patch.object(arm.model, "forward", side_effect=[origin, origin, origin + [1, 0, 0], origin + [1, 0, 0]]):
            with self.assertRaisesRegex(PositionHoldTimeout, "末端未达到请求位移"):
                arm.move_linear([5, 0, 0])

    def test_cartesian_preparation_drift_does_not_start_planned_motion(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        target = arm.model.inverse(arm.model.forward(bus.raw) + [0, 5, 0], bus.raw, allow_crossing=False)
        with patch.object(arm, "prepare", side_effect=lambda **kwargs: bus.raw.__setitem__(3, 937)):
            with self.assertRaisesRegex(PositionHoldTimeout, "ID 4 准备阶段偏离起点"):
                arm.hold_pose(target, cartesian=True)
        self.assertEqual(bus.batches, [])
        self.assertFalse(arm.last_joint_move["targets_verified"])
        self.assertTrue(all(bus._read_u8(j, 40) == 0 for j in range(1, 7)))

    def test_invalid_cartesian_start_reports_each_joint_without_motion(self):
        bus = FeedbackBus(load_config("config.json"), [1022, 901, 319, 658, 77, 269])
        arm = Arm(load_advanced(), bus)
        with self.assertRaises(ValueError) as error:
            arm.cartesian_start_pose()
        for detail in ("ID 4", "658", "680"):
            self.assertIn(detail, str(error.exception))
        self.assertEqual(bus.writes, [])

    def test_cartesian_fixed_gripper_feedback_does_not_change_hand_targets(self):
        config = load_advanced()
        model = Kinematics(config)
        raw = np.array(config["poses"]["初始状态"])
        raw[5] = 269
        target = model.inverse(model.forward(raw) + [5, 0, 0], raw, allow_crossing=False)
        np.testing.assert_array_equal(target[4:], raw[4:])
        target = model.project_step(np.array([5, 0, 0]), raw, allow_crossing=False)
        np.testing.assert_array_equal(target[4:], raw[4:])
        with self.assertRaises(ValueError):
            model.validate(raw)
        raw[5] = 260
        with self.assertRaises(ValueError):
            model.inverse(model.forward(raw) + [5, 0, 0], raw, allow_crossing=False)

    def test_integral_residual_correction_waits_for_zero_output(self):
        for output in (0, 1):
            with self.subTest(output=output):
                config = load_advanced()
                target = list(config["poses"]["初始状态"])
                bus = FeedbackBus(load_config("config.json"), target)
                bus.registers[2, 61] = output
                arm = Arm(config, bus)
                write = bus._write
                clock = [0.0]
                def residual(joint, address, values):
                    write(joint, address, values)
                    if joint == 2 and address == GOAL_POSITION:
                        bus.raw[1] = ((values[0] << 8) | values[1]) + 3
                def wait(seconds):
                    clock[0] += seconds
                    return False
                with patch.object(bus, "_write", side_effect=residual), patch.object(arm.cancel, "wait", side_effect=wait), patch("time.monotonic", side_effect=lambda: clock[0]):
                    if output:
                        with self.assertRaises(PositionHoldTimeout):
                            arm.hold_pose(target, cartesian=True)
                    else:
                        arm.hold_pose(target, cartesian=True)
                corrections = [v for j, a, v in bus.writes if j == 2 and a == GOAL_POSITION and len(v) == 2]
                self.assertEqual(bool(corrections), output == 0)

    def test_batch_rejects_missing_base_target_or_enable(self):
        for fault in ("goal", "torque"):
            with self.subTest(fault=fault):
                bus = FeedbackBus(load_config("config.json"), [990, 920, 300, 740, 80, 276])
                sync = bus._sync_write
                def corrupt(address, values):
                    sync(address, values)
                    if fault == "goal":
                        bus.registers[1, GOAL_POSITION + 1] ^= 1
                    else:
                        bus.registers[1, 40] = 0
                with patch.object(bus, "_sync_write", side_effect=corrupt), self.assertRaises(RuntimeError):
                    bus.move_positions({1:980, 3:302}, 250, 250)
                self.assertTrue(all(bus._read_u8(i, 40) == 0 for i in range(1, 7)))

    def test_cartesian_accepts_base_deadband_with_small_endpoint_error(self):
        config = load_advanced()
        raw = [990, 920, 300, 740, 80, 276]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        target = raw.copy()
        target[0] = 980
        sync = bus._sync_write
        def residual(address, values):
            sync(address, values)
            if address == GOAL_POSITION:
                bus.raw[0] = target[0] + 2
        with patch.object(bus, "_sync_write", side_effect=residual), patch.object(arm.cancel, "wait", return_value=False):
            arm.hold_pose(target, cartesian=True, duration_s=.25, speed=250)
        self.assertLessEqual(np.linalg.norm(arm.model.forward(bus.raw) - arm.model.forward(target)), 3)

    def test_cartesian_entry_preserves_current_pose_without_preparation(self):
        raw = [1022, 920, 300, 740, 80, 276]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(load_advanced(), bus)
        np.testing.assert_array_equal(arm.cartesian_start_pose(), raw)
        with patch.object(arm, "hold_pose") as prepare, patch.object(arm, "move_linear") as move:
            arm.execute(("move", [5, 0, 0]))
        prepare.assert_not_called()
        move.assert_called_once_with([5, 0, 0])
        self.assertEqual(bus.writes, [])

    def test_cartesian_invalid_start_never_moves_to_another_pose(self):
        for raw in ([1022, 896, 314, 659, 83, 275], [1022, 900, 320, 657, 81, 269]):
            bus = FeedbackBus(load_config("config.json"), raw)
            arm = Arm(load_advanced(), bus)
            with patch.object(arm, "hold_pose") as prepare, patch.object(arm, "move_linear") as move:
                with self.assertRaises(ValueError):
                    arm.execute(("move", [5, 0, 0]))
            prepare.assert_not_called()
            move.assert_not_called()
            self.assertEqual(bus.writes, [])

    def test_forward_at_base_endpoint_has_no_lateral_preparation(self):
        raw = [1022, 920, 300, 740, 80, 276]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(load_advanced(), bus)
        origin = arm.model.forward(raw)
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.execute(("move", [50, 0, 0]))
        actual = arm.model.forward(bus.raw) - origin
        np.testing.assert_allclose(actual, [50, 0, 0], atol=2)
        self.assertFalse(any(j == 1 and (a == GOAL_POSITION or (a == 40 and v == [1]))
                             for j, a, v in bus.writes))

    def test_idle_and_arrived_base_do_not_keep_position_hold_energized(self):
        config = load_advanced()
        raw = [980, 899, 330, 680, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        bus.registers[1, 40] = 1
        arm = Arm(config, bus)
        arm.prepare()
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertTrue(all(bus._read_u8(j, 40) == 1 for j in range(2, 7)))
        target = list(raw)
        target[0] = 990
        arm.move_pose(target, quiet=True)
        self.assertEqual(bus.raw, target)
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertTrue(all(bus._read_u8(j, 40) == 1 for j in range(2, 7)))

    def test_position_stop_only_disables_torque_without_goal_or_time_writes(self):
        bus = FeedbackBus(load_config("config.json"), [512] * 6)
        bus.registers[1, 40] = 1
        bus.registers[1, GOAL_POSITION] = 3
        bus.registers[1, GOAL_POSITION + 1] = 232
        write = bus._write

        def write_checked(joint, address, values):
            if address in (RUNNING_TIME, GOAL_POSITION):
                self.assertEqual(bus._read_u8(joint, 40), 0)
            write(joint, address, values)

        with patch.object(bus, "_write", side_effect=write_checked):
            self.assertEqual(bus.stop(1), [])
        self.assertEqual(bus._read_u16(1, GOAL_POSITION), 1000)
        self.assertEqual(bus.raw[0], 512)
        self.assertEqual(bus.writes, [(1, 40, [0]), (1, 40, [0])])

    def test_wheel_stop_disables_again_and_verifies_after_last_drive_write(self):
        for final_disable_fails in (False, True):
            with self.subTest(final_disable_fails=final_disable_fails):
                bus = FeedbackBus(load_config("config.json"), [512] * 6)
                bus.wheel = True
                bus.running_id = 1
                write = bus._write
                disables = 0

                def firmware(joint, address, values):
                    nonlocal disables
                    if address == 40:
                        disables += 1
                    write(joint, address, values)
                    if address == RUNNING_TIME or (address == 40 and disables == 2 and final_disable_fails):
                        bus.registers[1, 40] = 1

                with patch.object(bus, "_write", side_effect=firmware):
                    failures = bus.stop(1)
                self.assertEqual(bool(failures), final_disable_fails)
                self.assertEqual(bus.writes[-1], (1, 40, [0]))
                self.assertEqual(bus.running_id, 1 if final_disable_fails else None)

    def test_failed_torque_disable_does_not_retime_position_or_clear_watchdog(self):
        bus = FeedbackBus(load_config("config.json"), [512] * 6)
        bus.running_id = 1
        with patch.object(bus, "_read_u8", return_value=1):
            self.assertTrue(bus.stop())
        self.assertEqual(bus.running_id, 1)
        self.assertFalse(any(address == RUNNING_TIME for _, address, _ in bus.writes))

    def test_wheel_stop_still_zeros_drive_when_torque_disable_fails(self):
        bus = FeedbackBus(load_config("config.json"), [512] * 6)
        bus.wheel = True
        bus.running_id = 1
        with patch.object(bus, "set_torque", side_effect=RuntimeError("no reply")):
            self.assertTrue(bus.stop(1))
        self.assertIn((1, RUNNING_TIME, [0, 0]), bus.writes)
        self.assertEqual(bus.running_id, 1)

    def test_run_and_mode_change_abort_if_stopping_another_joint_fails(self):
        for operation in (lambda bus: bus.run(1, "left", 100),
                          lambda bus: bus.set_mode(1, "position")):
            bus = FeedbackBus(load_config("config.json"), [512] * 6)
            bus.wheel = True
            read = bus._read_u8
            with patch.object(bus, "_read_u8", side_effect=lambda j, a: 1 if j == 2 and a == 40 else read(j, a)):
                with self.assertRaisesRegex(RuntimeError, "ID 2"):
                    operation(bus)
            self.assertFalse(any(address in (9, 11, 48) or (address == 40 and values == [1])
                                 for _, address, values in bus.writes))

    def test_wheel_enable_uses_new_drive_and_lost_ack_triggers_stop(self):
        for lose_ack in (False, True):
            bus = FeedbackBus(load_config("config.json"), [512] * 6)
            bus.wheel = True
            bus.registers[1, RUNNING_TIME] = 7
            bus.registers[1, RUNNING_TIME + 1] = 232
            torque = bus.set_torque

            def enable(joint, enabled):
                if enabled:
                    self.assertEqual(bus._read_u16(joint, RUNNING_TIME), 100)
                    self.assertEqual(bus.running_id, joint)
                torque(joint, enabled)
                if enabled and lose_ack:
                    raise RuntimeError("lost torque acknowledgement")

            with patch.object(bus, "set_torque", side_effect=enable):
                if lose_ack:
                    with self.assertRaisesRegex(RuntimeError, "lost torque"):
                        bus.run(1, "left", 100)
                    self.assertEqual(bus._read_u8(1, 40), 0)
                    self.assertEqual(bus._read_u16(1, RUNNING_TIME), 0)
                    self.assertIsNone(bus.running_id)
                else:
                    bus.run(1, "left", 100)
                    self.assertEqual(bus.running_id, 1)

    def test_wheel_drive_lost_ack_never_enables_torque(self):
        bus = FeedbackBus(load_config("config.json"), [512] * 6)
        bus.wheel = True
        write = bus._write

        def lost_ack(joint, address, values):
            write(joint, address, values)
            if address == RUNNING_TIME and values != [0, 0]:
                raise RuntimeError("lost drive acknowledgement")

        with patch.object(bus, "_write", side_effect=lost_ack):
            with self.assertRaisesRegex(RuntimeError, "lost drive"):
                bus.run(1, "left", 100)
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertEqual(bus._read_u16(1, RUNNING_TIME), 0)
        self.assertIsNone(bus.running_id)
        self.assertNotIn((1, 40, [1]), bus.writes)

    def test_verified_crossing_uses_wheel_feedback_then_resumes_position_control(self):
        from joint_mapping import mapped_position, raw_position
        for joint, initial, goal in [(2, 1019, 135), (3, 136, 997)]:
            with self.subTest(joint=joint):
                config = load_advanced()
                raw = [1022, 899, 315, 661, 86, 272]
                raw[joint - 1] = initial
                target = list(raw)
                target[joint - 1] = goal
                bus = FeedbackBus(load_config("config.json"), raw)
                mapping = config["joint_mapping"][str(joint)]
                start, end = mapped_position(initial, mapping), mapped_position(goal, mapping)
                count = int(np.ceil(abs(end - start) / 8))
                bus.wheel_positions[joint] = [raw_position(round(p), mapping)
                                              for p in np.linspace(start, end, count + 1)[1:]]
                arm = Arm(config, bus)
                with patch.object(arm.cancel, "wait", return_value=False):
                    arm.move_pose(target, quiet=True)
                self.assertEqual(bus.raw, target)
                self.assertEqual(bus.mode(joint)[0], "position")
                self.assertEqual(bus._read_u8(joint, 40), 1)
                self.assertIsNone(bus.running_id)
                self.assertTrue(any(j == joint and address == RUNNING_TIME and values != [0, 0]
                                    for j, address, values in bus.writes))
                expected_direction = mapping["wheel_positive"]
                if end < start:
                    expected_direction = "right" if expected_direction == "left" else "left"
                drive = mapping["wheel_speed"] + (1024 if expected_direction == "right" else 0)
                self.assertIn((joint, RUNNING_TIME, [drive >> 8, drive & 255]), bus.writes)
                self.assertFalse(any(j != joint and address in (9, 11, 48)
                                     for j, address, _ in bus.writes))

    def test_recorded_poses_complete_with_feedback_across_zero(self):
        from joint_mapping import mapped_position, raw_position
        for name in ("姿态2",):
            with self.subTest(name=name):
                config = load_advanced()
                config["joint_mapping"]["1"]["wheel_positive"] = "right"
                bus = FeedbackBus(load_config("config.json"), config["pose_paths"]["立正"][0])
                arm = Arm(config, bus)
                clock = [0.0]
                crossings = []
                cross = arm.move_across_zero

                def crossing(joint, target):
                    mapping = config["joint_mapping"][str(joint)]
                    start = mapped_position(bus.raw[joint - 1], mapping)
                    end = mapped_position(target, mapping)
                    count = max(1, int(np.ceil(abs(end - start) / 6)))
                    bus.wheel_positions[joint] = [raw_position(round(p), mapping)
                                                  for p in np.linspace(start, end, count + 1)[1:]]
                    crossings.append(joint)
                    cross(joint, target)

                read = bus.read_position

                def feedback(joint):
                    if joint == 1 and bus.running_id == 1 and bus.mode(1)[0] == "wheel":
                        drive = bus._read_u16(1, RUNNING_TIME)
                        bus.raw[0] = (bus.raw[0] + (2 if drive == 1124 else -2)) % 1024
                    return read(joint)

                def wait(seconds):
                    clock[0] += seconds
                    return False

                with patch.object(arm, "move_across_zero", side_effect=crossing), \
                     patch.object(bus, "read_position", side_effect=feedback), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    arm.execute(("pose", name))
                self.assertEqual(bus.raw[1:], config["poses"][name][1:])
                self.assertLessEqual(abs(bus.raw[0] - config["poses"][name][0]), 3)
                self.assertIn(2, crossings)
                if name == "立正":
                    self.assertIn(3, crossings)
                    self.assertIn(5, crossings)
                self.assertTrue(all(bus.mode(j)[0] == "position" for j in range(1, 7)))
                self.assertEqual(bus._read_u8(1, 40), 0)
                self.assertIsNone(bus.running_id)

    def test_crossing_faults_never_resume_position_targets(self):
        for fault in ("cancel", "reverse", "jump", "stall", "read", "drift", "restore"):
            with self.subTest(fault=fault):
                bus = FeedbackBus(load_config("config.json"), [1022, 1019, 315, 661, 86, 272])
                arm = Arm(load_advanced(), bus)
                arm.prepare()
                clock = [0.0]
                fault_index = [None]
                read = bus.read_position
                set_mode = bus.set_mode

                def mode(joint, value, **kwargs):
                    result = set_mode(joint, value, **kwargs)
                    if fault == "restore" and value == "position":
                        bus.registers[joint, 40] = 1
                        raise RuntimeError("injected mode restore failure")
                    return result

                def feedback(joint):
                    if joint == 2 and bus.running_id == 2 and bus.mode(2)[0] == "wheel":
                        fault_index[0] = len(bus.writes)
                        if fault == "read":
                            raise RuntimeError("injected read failure")
                        step = {"reverse": -4, "jump": 40, "stall": 0}.get(fault, 6)
                        bus.raw[1] = (bus.raw[1] + step) % 1024
                    elif joint == 2 and fault == "drift" and fault_index[0] is not None:
                        bus.raw[1] = 80
                    return read(joint)

                def wait(seconds):
                    clock[0] += seconds
                    if fault == "cancel":
                        fault_index[0] = len(bus.writes)
                        arm.cancel.set()
                        return True
                    return False

                with patch.object(bus, "read_position", side_effect=feedback), \
                     patch.object(bus, "set_mode", side_effect=mode), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]):
                    with self.assertRaises((InterruptedError, PositionHoldTimeout, RuntimeError)):
                        arm.move_pose([1022, 135, 330, 661, 86, 272], quiet=True)
                self.assertEqual(bus._read_u8(2, 40), 0)
                self.assertEqual(bus._read_u16(2, RUNNING_TIME), 0)
                self.assertIsNone(bus.running_id)
                self.assertEqual(bus.raw[2], 315)
                self.assertIsNotNone(fault_index[0])
                self.assertFalse(any(address == GOAL_POSITION or (address == 40 and values == [1])
                                     for _, address, values in bus.writes[fault_index[0]:]))

    def test_recorded_standing_requires_measured_direction_before_earlier_steps(self):
        config = load_advanced()
        config["joint_mapping"]["2"]["wheel_positive"] = None
        raw = config["pose_paths"]["立正"][0]
        bus = FeedbackBus(load_config("config.json"), raw)
        with self.assertRaisesRegex(PositionHoldTimeout, "实测连续模式方向.*动作未启动"):
            Arm(config, bus).execute(("pose", "立正"))
        self.assertEqual(bus.writes, [])
        self.assertEqual(bus.raw, raw)

    def test_boundary_fault_releases_base_without_reenabling_or_releasing_load_joints(self):
        config = load_advanced()
        config["joint_mapping"]["2"]["wheel_positive"] = None
        bus = FeedbackBus(load_config("config.json"), config["pose_paths"]["立正"][0])
        for joint in range(1, 7):
            bus.registers[joint, 40] = 1
        runner = CommandRunner(Arm(config, bus))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.submit(("pose", "立正"))
            with self.assertRaisesRegex(PositionHoldTimeout, "实测连续模式方向"):
                runner.wait()
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertTrue(all(bus._read_u8(joint, 40) == 1 for joint in range(2, 7)))
        self.assertTrue(all(joint == 1 and address in (RUNNING_TIME, GOAL_POSITION, 40)
                            for joint, address, values in bus.writes))

    def test_cartesian_feedback_jump_stops_before_chasing_target(self):
        config = load_advanced()
        raw = list(config["poses"]["初始状态"])
        raw[0] = 723
        target = raw.copy()
        target[0] = 960
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        move = bus.move_positions
        def jump(*args, **kwargs):
            result = move(*args, **kwargs)
            bus.raw[0] = 0
            return result
        with patch.object(bus, "move_positions", side_effect=jump), \
             patch.object(arm.cancel, "wait", return_value=False):
            with self.assertRaisesRegex(PositionHoldTimeout, "反馈偏离本次运动范围"):
                arm.hold_pose(target, cartesian=True, home=True, duration_s=0.5, speed=250)
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
        self.assertEqual(len(bus.batches), 1)

    def test_cartesian_integral_is_verified_and_restored_for_presets(self):
        config = load_advanced()
        config["kinematics"]["position_control"] = {
            joint: {**parameters, "i": 1} for joint, parameters in config["position_control"].items()}
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.prepare(cartesian=True)
            self.assertEqual([bus._read_u8(joint, 23) for joint in (2, 3)], [1, 1])
            write = bus._write
            def residual(joint, address, values):
                write(joint, address, values)
                if joint == 2 and address == GOAL_POSITION:
                    bus.raw[1] = config["poses"]["初始状态"][1] + 2
            with patch.object(bus, "_write", side_effect=residual):
                arm.hold_pose(config["poses"]["初始状态"], cartesian=True)
            self.assertFalse(any(address == GOAL_POSITION and len(values) == 2 for _, address, values in bus.writes))
            arm.prepare()
            self.assertEqual([bus._read_u8(joint, 23) for joint in (2, 3)],
                             [config["position_control"][str(joint)].get("i", 0) for joint in (2, 3)])
        self.assertFalse(any(address == 48 for _, address, _ in bus.writes))

    def test_cartesian_extra_joint_parameters_restore_for_presets(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        for joint in (1, 4):
            for address, values in ((21, [15]), (23, [0]), (24, [0, 30]), (26, [1, 2])):
                bus._write(joint, address, values)
        arm = Arm(config, bus)
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.prepare(cartesian=True)
            self.assertEqual(bus._read_u16(4, 24), 180)
            self.assertEqual(bus._read_u8(4, 23), 4)
            arm.prepare()
            for joint in (1, 4):
                self.assertEqual(bus._read_u16(joint, 24), 30)
                self.assertEqual(bus._read_u8(joint, 23), 0)
                self.assertEqual(bus._read_u16(joint, 26), 258)
        self.assertFalse(any(address == 48 for _, address, _ in bus.writes))

    def test_cartesian_timeout_releases_all_drives_and_invalidates_initial_pose(self):
        for command in (("home", "初始状态"), ("pose", "初始状态"), ("move", [5, 0, 0])):
            for failed_stop in (False, True):
                with self.subTest(command=command, failed_stop=failed_stop):
                    config = load_advanced()
                    bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
                    for joint in range(1, 7):
                        bus.registers[joint, 40] = 1
                    arm = Arm(config, bus)
                    arm.cartesian_initialized = True
                    runner = CommandRunner(arm)
                    read = bus._read_u8
                    def readback(joint, address):
                        return 1 if failed_stop and joint == 2 and address == 40 else read(joint, address)
                    with patch.object(arm, "execute", side_effect=PositionHoldTimeout("stalled")), \
                         patch.object(bus, "_read_u8", side_effect=readback), contextlib.redirect_stdout(io.StringIO()):
                        runner.submit(command)
                        with self.assertRaisesRegex(RuntimeError if failed_stop else PositionHoldTimeout,
                                                   "全部驱动关闭失败" if failed_stop else "全部驱动关闭回读已确认"):
                            runner.wait()
                    self.assertFalse(arm.cartesian_initialized)
                    self.assertTrue(all(bus.registers[joint, 40] == 0 for joint in range(1, 7)))
                    self.assertFalse(any(address == 40 and values == [1] for _, address, values in bus.writes))

    def test_fault_does_not_claim_base_disabled_when_readback_stays_enabled(self):
        bus = FeedbackBus(load_config("config.json"), load_advanced()["poses"]["姿态2"])
        arm = Arm(load_advanced(), bus)
        runner = CommandRunner(arm)
        with patch.object(arm, "execute", side_effect=PositionHoldTimeout("motion failed")), \
             patch.object(bus, "_read_u8", return_value=1), contextlib.redirect_stdout(io.StringIO()):
            runner.submit(("pose", "姿态2"))
            with self.assertRaisesRegex(RuntimeError, "底座驱动关闭失败"):
                runner.wait()
        self.assertNotIn("底座驱动已关闭", str(runner.error))
        self.assertFalse(any(j != 1 and address == 40 and values == [0] for j, address, values in bus.writes))

    def test_pose_two_finishes_without_base_sweep_or_direction_calibration(self):
        config = load_advanced()
        raw = list(config["poses"]["姿态2"])
        raw[1] += 10
        bus = FeedbackBus(load_config("config.json"), raw)
        runner = CommandRunner(Arm(config, bus))
        with patch.object(bus, "run") as run, patch.object(bus, "set_mode") as mode, \
             contextlib.redirect_stdout(io.StringIO()):
            dispatch_result('{"text":"进行动作2"}', runner)
            runner.wait()
        run.assert_not_called()
        mode.assert_not_called()
        self.assertEqual(bus.raw, config["poses"]["姿态2"])
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertTrue(all(bus._read_u8(j, 40) == 1 for j in range(2, 7)))

    def test_pose_three_closes_gripper_before_returning_wrist(self):
        config = load_advanced()
        saved = copy.deepcopy(config["poses"])
        raw = list(config["poses"]["姿态3"])
        raw[4:6] = [78, config["poses"]["姿态2"][5]]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        with patch.object(arm.cancel, "wait", return_value=False), \
             patch.object(bus, "move_position", wraps=bus.move_position) as move, \
             contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("pose", "姿态3"))
        self.assertEqual([(call.args[0], call.args[1]) for call in move.call_args_list],
                         [(5, 345), (6, 380), (6, config["poses"]["姿态2"][5]), (5, 78)])
        self.assertEqual(bus.raw, raw)
        self.assertEqual(config["poses"], saved)

    def test_pose_two_corrects_loaded_shoulder_within_mapped_limits(self):
        from joint_mapping import check_position
        for bias in (5, 8, -5, "stall"):
            with self.subTest(bias=bias):
                config = load_advanced()
                target = config["poses"]["姿态2"]
                raw = list(target)
                raw[1] -= 20
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                write = bus._write
                clock, corrections = [0.0], []
                def firmware(joint, address, values):
                    write(joint, address, values)
                    if joint == 2 and address == GOAL_POSITION and bus.running_id == 2:
                        goal = (values[0] << 8) | values[1]
                        if len(values) == 2:
                            check_position(goal, config["joint_mapping"]["2"])
                            corrections.append(goal)
                        bus.raw[1] = target[1] + 5 if bias == "stall" else goal + bias
                def wait(seconds):
                    clock[0] += seconds
                    self.assertLess(clock[0], 5)
                    return False
                with patch.object(bus, "_write", side_effect=firmware), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if bias in (-5, "stall"):
                        with self.assertRaises(PositionHoldTimeout):
                            arm.execute(("pose", "姿态2"))
                    else:
                        arm.execute(("pose", "姿态2"))
                        self.assertLessEqual(abs(bus.raw[1] - target[1]), 3)
                        self.assertTrue(corrections)
                self.assertTrue(all(abs(goal - target[1]) <= 8 for goal in corrections))

    def test_pose_three_corrects_shoulder_residual_without_accepting_stall(self):
        for fault in (7, -7, 8, -8, "deadband", "jitter", "stall", 9, "unsafe", "cancel", "alarm"):
            with self.subTest(fault=fault):
                config = load_advanced()
                saved = copy.deepcopy(config["poses"])
                target = config["poses"]["姿态3"]
                raw = list(target)
                raw[1] -= 7 if fault == "deadband" else 20
                raw[4:] = [78, config["poses"]["姿态2"][5]]
                if fault == "unsafe":
                    config["kinematics"]["cartesian_raw_ranges"]["2"][1] = 940
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                runner = CommandRunner(arm)
                write, read = bus._write, bus.read_position
                clock, reads, corrections, stopped = [0.0], [0], [], []
                def firmware(joint, address, values):
                    before = bus.raw[joint - 1]
                    write(joint, address, values)
                    if joint == 2 and address == GOAL_POSITION and bus.running_id == 2:
                        goal = (values[0] << 8) | values[1]
                        if len(values) == 2:
                            corrections.append(goal)
                        if fault == "deadband":
                            bus.raw[1] = before if abs(goal - before) <= 8 else goal
                        elif type(fault) is int or fault == "jitter":
                            bus.raw[1] = goal - (7 if fault == "jitter" else fault)
                        else:
                            bus.raw[1] = target[1] - 7
                    if joint == 6 and address == GOAL_POSITION and values[:2] == [1, 124]:
                        self.assertLessEqual(abs(bus.raw[1] - target[1]), 3)
                def feedback(joint):
                    position = read(joint)
                    if joint == 2 and bus.running_id == 2:
                        reads[0] += 1
                        if fault == "jitter":
                            position += 1 if reads[0] % 2 else -1
                        if reads[0] == 4 and fault in ("cancel", "alarm"):
                            stopped.append(len(bus.writes))
                            if fault == "alarm":
                                raise RuntimeError("ID 2 返回错误码 0x01")
                            arm.cancel.set()
                    return position
                def wait(seconds):
                    clock[0] += seconds
                    self.assertLess(clock[0], 8.0)
                    return arm.cancel.is_set()
                with patch.object(bus, "_write", side_effect=firmware), \
                     patch.object(bus, "read_position", side_effect=feedback), \
                     patch.object(bus, "move_position", wraps=bus.move_position) as move, \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.submit(("pose", "姿态3"))
                    if fault in ("stall", 9, "unsafe", "cancel", "alarm"):
                        expected = {"cancel": InterruptedError, "alarm": RuntimeError}.get(fault, PositionHoldTimeout)
                        with self.assertRaises(expected):
                            runner.wait()
                    else:
                        runner.wait()
                self.assertTrue(all(abs(goal - target[1]) <= 8 for goal in corrections))
                self.assertEqual(config["poses"], saved)
                self.assertIsNone(bus.running_id)
                if fault in ("stall", 9, "unsafe", "cancel", "alarm"):
                    self.assertTrue(all(call.args[0] == 2 for call in move.call_args_list))
                    if fault in (9, "unsafe", "cancel", "alarm"):
                        self.assertEqual(corrections, [])
                    if fault == "stall":
                        self.assertEqual(corrections, [942, 945, 947])
                    self.assertLess(clock[0], 4.0)
                else:
                    self.assertTrue(corrections)
                    self.assertLessEqual(abs(bus.raw[1] - target[1]), 3)
                    self.assertEqual(bus.raw[4:], raw[4:])
                    self.assertEqual([(call.args[0], call.args[1]) for call in move.call_args_list if call.args[0] in (5, 6)],
                                     [(5, 345), (6, 380), (6, config["poses"]["姿态2"][5]), (5, 78)])
                if stopped:
                    self.assertFalse(any(address == GOAL_POSITION for _, address, _ in bus.writes[stopped[0]:]))
                    self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))

    def test_pose_three_waits_for_stable_feedback_without_accepting_drift(self):
        for fault in ("settling", "slow_reads", "oscillation", "drift", "cancel"):
            with self.subTest(fault=fault):
                config = load_advanced()
                target = config["poses"]["姿态3"]
                raw = list(target)
                raw[4:] = [78, config["poses"]["姿态2"][5]]
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                runner = CommandRunner(arm)
                move, positions = arm.move_pose, arm.positions
                clock, stage, began, samples, started_writes = [0.0], [0], [], [], []
                def pose(*args, **kwargs):
                    if stage[0] == 1:
                        self.assertEqual(samples[-5:], [True] * 5)
                        stage[0] = 2
                    move(*args, **kwargs)
                    if stage[0] == 0:
                        stage[0] = 1
                        began.append(clock[0])
                        started_writes.append(len(bus.writes))
                def feedback():
                    actual = positions()
                    if stage[0] == 1:
                        if fault == "slow_reads":
                            clock[0] += 0.22
                        if (fault == "drift" or fault == "settling" and len(samples) < 8
                                or fault == "oscillation" and len(samples) % 2 == 0):
                            actual[1] = target[1] - 4
                        samples.append(abs(actual[1] - target[1]) <= 3)
                        if fault == "cancel" and len(samples) == 3:
                            arm.cancel.set()
                    return actual
                def wait(seconds):
                    clock[0] += seconds
                    self.assertLess(clock[0], 6.0)
                    return arm.cancel.is_set()
                with patch.object(arm, "move_pose", side_effect=pose), \
                     patch.object(arm, "positions", side_effect=feedback), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.submit(("pose", "姿态3"))
                    if fault == "cancel":
                        with self.assertRaises(InterruptedError):
                            runner.wait()
                    elif fault in ("oscillation", "drift"):
                        with self.assertRaisesRegex(TimeoutError, "整臂未能稳定保持目标姿态"):
                            runner.wait()
                    else:
                        runner.wait()
                if fault in ("settling", "slow_reads"):
                    self.assertEqual(stage[0], 2)
                    self.assertLessEqual(abs(bus.raw[1] - raw[1]), 3)
                    self.assertEqual(bus.raw[:1] + bus.raw[2:], raw[:1] + raw[2:])
                else:
                    self.assertEqual(stage[0], 1)
                    self.assertLess(clock[0] - began[0], 3.11)
                    self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
                    self.assertFalse(any(joint == 6 and address == GOAL_POSITION and values[:2] == [config["poses"]["姿态2"][5] >> 8, config["poses"]["姿态2"][5] & 255]
                                         for joint, address, values in bus.writes[started_writes[0]:]))

    def test_pose_three_cancel_during_closing_does_not_return_wrist(self):
        config = load_advanced()
        raw = list(config["poses"]["姿态3"])
        raw[4:6] = [78, config["poses"]["姿态2"][5]]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        move = bus.move_position

        def closing(joint, target, duration, speed):
            result = move(joint, target, duration, speed)
            if joint == 6 and target == config["poses"]["姿态2"][5]:
                arm.stop()
            return result

        with patch.object(arm.cancel, "wait", side_effect=lambda _: arm.cancel.is_set()), \
             patch.object(bus, "move_position", side_effect=closing) as calls:
            with self.assertRaises(InterruptedError):
                arm.execute(("pose", "姿态3"))
        self.assertEqual([(call.args[0], call.args[1]) for call in calls.call_args_list],
                         [(5, 345), (6, 380), (6, config["poses"]["姿态2"][5])])
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))

    def test_pose_two_shoulder_target_is_inside_measured_limits(self):
        from joint_mapping import check_position
        config = load_advanced()
        check_position(config["poses"]["姿态2"][1], config["joint_mapping"]["2"])

    def test_pose_settling_corrects_load_change_after_elbow_motion(self):
        for name, bias, loaded_joint in (("姿态2", 5, 3), ("姿态3", -5, 3), ("姿态3", -5, 6)):
            for stalled in (False, True):
                with self.subTest(name=name, stalled=stalled, loaded_joint=loaded_joint):
                    config = load_advanced()
                    target = config["poses"][name]
                    raw = list(target)
                    raw[2] += 10
                    bus = FeedbackBus(load_config("config.json"), raw)
                    arm = Arm(config, bus)
                    write = bus._write
                    clock, loaded, corrections = [0.0], [False], []
                    def firmware(joint, address, values):
                        write(joint, address, values)
                        if joint == loaded_joint and address == GOAL_POSITION and bus.running_id == loaded_joint:
                            loaded[0] = True
                            bus.raw[1] = target[1] + bias
                        if joint == 2 and address == GOAL_POSITION and loaded[0]:
                            goal = (values[0] << 8) | values[1]
                            bus.raw[1] = (target[1] if stalled else goal) + bias
                            if bus.running_id is None:
                                corrections.append(goal)
                    def wait(seconds):
                        clock[0] += seconds
                        self.assertLess(clock[0], 6)
                        return False
                    with patch.object(bus, "_write", side_effect=firmware), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()) as output:
                        if stalled:
                            with self.assertRaisesRegex(TimeoutError, "整臂未能稳定保持目标姿态"):
                                arm.execute(("pose", name))
                            self.assertNotIn("到达姿态", output.getvalue())
                        else:
                            arm.execute(("pose", name))
                            self.assertLessEqual(abs(bus.raw[1] - target[1]), 3)
                    self.assertTrue(corrections)
                    self.assertTrue(all(abs(goal - target[1]) <= 8 for goal in corrections))

    def test_preset_gripper_targets_allow_cartesian_entry_with_closed_feedback(self):
        from joint_mapping import check_position
        config = load_advanced()
        for name in ("立正", "姿态2", "初始状态"):
            check_position(config["poses"][name][5], config["joint_mapping"]["6"])
        raw = list(config["poses"]["初始状态"])
        raw[5] = config["poses"]["姿态2"][5] - 5
        arm = Arm(config, FeedbackBus(load_config("config.json"), raw))
        arm.cartesian_start_pose()
        self.assertEqual(arm.bus.writes, [])

    def test_base_sweep_returns_through_mapping_with_only_base_pulses(self):
        from joint_mapping import mapped_position
        config = load_advanced()
        config["joint_mapping"]["1"]["wheel_positive"] = "right"
        origin = list(config["poses"]["姿态2"])
        bus = FeedbackBus(load_config("config.json"), origin)
        arm = Arm(config, bus)
        clock = [0.0]
        stages = []

        def feedback(joint):
            if joint == 1 and bus.running_id == 1 and bus._read_u8(1, 40):
                drive = bus._read_u16(1, RUNNING_TIME)
                bus.raw[0] = (bus.raw[0] + (2 if drive == 1124 else -2)) % 1024
            return bus.raw[joint - 1]

        def wait(seconds):
            clock[0] += seconds
            if seconds == 0.4:
                stages.append(mapped_position(bus.raw[0], config["joint_mapping"]["1"]))
            return False

        with patch.object(bus, "read_position", side_effect=feedback), \
             patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]):
            arm.sweep_base()
        center = mapped_position(origin[0], config["joint_mapping"]["1"])
        for actual, expected in zip(stages, [center + 25, center - 25, center]):
            self.assertLessEqual(abs(actual - expected), 3)
        self.assertEqual(len(stages), 3)
        self.assertEqual(bus.raw[1:], origin[1:])
        self.assertEqual(bus.mode(1)[0], "position")
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertFalse(any(j != 1 and address == 40 and values == [0]
                             for j, address, values in bus.writes))
        drives = [(values[0] << 8) | values[1] for _, address, values in bus.writes
                  if address == RUNNING_TIME and values != [0, 0]]
        self.assertEqual(set(drives), {100, 1124})
        self.assertGreater(clock[0], 5)

    def test_base_sweep_faults_stop_without_reverse_or_return(self):
        for fault in ("cancel", "reverse", "jump", "stall", "load_drift"):
            with self.subTest(fault=fault):
                config = load_advanced()
                config["joint_mapping"]["1"]["wheel_positive"] = "right"
                bus = FeedbackBus(load_config("config.json"), config["poses"]["姿态2"])
                arm = Arm(config, bus)
                clock = [0.0]

                def feedback(joint):
                    if joint == 1 and bus.running_id == 1 and bus._read_u8(1, 40):
                        step = {"reverse": -2, "jump": 20, "stall": 0}.get(fault, 2)
                        bus.raw[0] = (bus.raw[0] + step) % 1024
                    return bus.raw[joint - 1]

                def wait(seconds):
                    clock[0] += seconds
                    if fault == "cancel" and seconds == 0.02:
                        arm.cancel.set()
                        return True
                    if fault == "load_drift" and seconds == 0.2:
                        bus.raw[1] += 10
                    return False

                with patch.object(bus, "read_position", side_effect=feedback), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]):
                    with self.assertRaises((InterruptedError, PositionHoldTimeout)):
                        arm.sweep_base()
                self.assertEqual(bus._read_u8(1, 40), 0)
                self.assertIsNone(bus.running_id)
                self.assertEqual(bus._read_u16(1, RUNNING_TIME), 0)
                self.assertNotIn((1, RUNNING_TIME, [0, 100]), bus.writes)
                self.assertFalse(any(j != 1 and address == 40 and values == [0]
                                     for j, address, values in bus.writes))



    def test_pose_two_failure_does_not_start_base_sweep(self):
        arm = Arm(load_advanced())
        with patch.object(arm, "move_pose", side_effect=PositionHoldTimeout("not arrived")), \
             patch.object(arm, "sweep_base") as sweep:
            with self.assertRaises(PositionHoldTimeout):
                arm.execute(("pose", "姿态2"))
        sweep.assert_not_called()

    def test_elbow_long_move_waits_for_progress_but_bounds_stall_and_slow_motion(self):
        for goal in (19, 43):
            for step in (20, 0, 2):
                with self.subTest(goal=goal, step=step):
                    raw = [980, 899, 315, 680, 100, 280]
                    bus = FeedbackBus(load_config("config.json"), raw)
                    bus.stall = True
                    arm = Arm(load_advanced(), bus)
                    arm.prepare()
                    target = list(raw)
                    target[2] = goal
                    clock = [0.0]
                    read = bus.read_position

                    def feedback(joint):
                        if joint == 3 and bus.running_id == 3:
                            bus.raw[2] = max(goal, bus.raw[2] - step)
                        return read(joint)

                    def wait(seconds):
                        clock[0] += 0.4
                        return False

                    with patch.object(bus, "read_position", side_effect=feedback), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]):
                        if step == 20:
                            arm.move_pose(target, duration_s=0.2, quiet=True)
                            self.assertEqual(bus.raw[2], goal)
                            self.assertGreater(clock[0], 2.0)
                        else:
                            with self.assertRaises(PositionHoldTimeout):
                                arm.move_pose(target, duration_s=0.2, quiet=True)
                            self.assertGreater(bus.raw[2], goal + 2)
                            self.assertLessEqual(clock[0], 10.5 if step else 2.5)
                    self.assertIsNone(bus.running_id)

    def test_position_tuning_only_changes_configured_joint_without_unlocking(self):
        bus = FeedbackBus(load_config("config.json"), [980, 899, 330, 680, 100, 280])
        Arm(load_advanced(), bus).prepare()
        for joint, p, force, deadband in ((2, 15, 150, 1), (3, 30, 180, 2)):
            self.assertEqual(bus._read_u8(joint, 21), p)
            self.assertEqual(bus._read_u16(joint, 24), force)
            self.assertEqual(bus._read_u16(joint, 26), deadband * 257)
        self.assertTrue(all(joint in (2, 3) for joint, address, _ in bus.writes if address in (21, 24, 26)))
        self.assertFalse(any(address in (9, 11, 48) for _, address, _ in bus.writes))

    def test_prepare_staggers_torque_and_holds_each_joints_latest_position(self):
        bus = FeedbackBus(load_config("config.json"), [1022, 896, 315, 695, 83, 275])
        arm = Arm(load_advanced(), bus)
        write = bus._write
        starting = [False]
        def startup(joint, address, values):
            if address == 40 and values == [1]:
                if starting[0]:
                    raise RuntimeError("overlapping servo startup")
                starting[0] = True
                if joint == 2:
                    bus.raw[3] = 659
            write(joint, address, values)
        def wait(seconds):
            self.assertGreaterEqual(seconds, 0.08)
            starting[0] = False
            return False
        with patch.object(bus, "_write", side_effect=startup), \
             patch.object(arm.cancel, "wait", side_effect=wait):
            arm.prepare()
        self.assertEqual(bus._read_u16(4, GOAL_POSITION), 659)
        self.assertEqual(bus.raw[3], 659)
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertTrue(all(bus._read_u8(joint, 40) == 1 for joint in range(2, 7)))

    def test_position_tuning_refuses_unlocked_eeprom_before_writes(self):
        bus = FeedbackBus(load_config("config.json"), [980, 899, 330, 680, 100, 280])
        bus.registers[2, 48] = 0
        with self.assertRaisesRegex(ValueError, "EEPROM"):
            Arm(load_advanced(), bus).prepare()
        self.assertEqual(bus.writes, [])

    def test_home_profile_preserves_existing_hold_targets(self):
        bus = FeedbackBus(load_config("config.json"), [1022, 120, 874, 986, 82, 276])
        arm = Arm(load_advanced(), bus)
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.prepare()
            bus._write(2, GOAL_POSITION, [0, 117])
            bus.raw[1] = 120
            bus.set_torque(4, False)
            bus.raw[3] = 983
            bus.writes.clear()
            arm.prepare(cartesian=True)
        self.assertEqual(bus._read_u16(2, GOAL_POSITION), 117)
        self.assertEqual(bus._read_u16(4, GOAL_POSITION), 983)
        self.assertFalse(any(j == 2 and (a == GOAL_POSITION or a == 40) for j, a, _ in bus.writes))
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertEqual(bus._read_u8(4, 40), 1)

    def test_position_tuning_readback_failure_prevents_torque_enable(self):
        bus = FeedbackBus(load_config("config.json"), [980, 899, 330, 680, 100, 280])
        read = bus._read_u16
        with patch.object(bus, "_read_u16", side_effect=lambda joint, address: 181 if address == 24 else read(joint, address)):
            with self.assertRaisesRegex(RuntimeError, "回读不一致"):
                Arm(load_advanced(), bus).prepare()
        self.assertFalse(any(address == 40 for _, address, _ in bus.writes))

    def test_wrist_long_arc_does_not_fold_at_half_turn(self):
        from joint_mapping import PositionTrace, mapped_position, check_position
        trace = PositionTrace(89)
        for raw in [189, 289, 389, 489, 589, 607]:
            trace.update(raw)
        self.assertEqual(trace.position, 518)
        for raw in [507, 407, 307, 207, 107, 7, 931, 831, 731, 699]:
            trace.update(raw)
        self.assertEqual(trace.position, -414)
        mapping = {"zero_raw": 89, "sign": -1, "limits": [-515, 411], "unwrapped": True}
        self.assertEqual(mapped_position(607, mapping), -518)
        self.assertEqual(mapped_position(699, mapping), 414)
        self.assertEqual(check_position(89, mapping), 0)
        self.assertEqual(check_position(81, mapping), 8)
        with self.assertRaises(ValueError):
            check_position(650, mapping)

    def test_trace_rejects_unobservable_jump_and_full_cycle(self):
        from joint_mapping import PositionTrace
        with self.assertRaises(ValueError):
            PositionTrace(89).update(607)
        trace = PositionTrace(89)
        with self.assertRaises(ValueError):
            for offset in range(100, 1100, 100):
                trace.update((89 + offset) % 1024)

    def test_endpoint_tolerance_does_not_expand_motion_limits(self):
        from joint_mapping import check_position
        shoulder = {"zero_raw": 902, "sign": -1, "limits": [3, 468]}
        elbow = {"zero_raw": 315, "sign": -1, "limits": [-504, -3]}
        for raw, mapping in [(903, shoulder), (901, shoulder), (315, elbow)]:
            check_position(raw, mapping, tolerance=6)
            with self.assertRaises(ValueError):
                check_position(raw, mapping)
        with self.assertRaises(ValueError):
            check_position(910, shoulder, tolerance=6)

    def test_mapped_joint_at_endpoint_only_allows_inward_motion(self):
        for command_sign, should_run in [(1, False), (-1, True)]:
            with self.subTest(command_sign=command_sign):
                config = copy.deepcopy(self.config)
                config["motion_calibrated"] = True
                config["joint_mapping"] = {"1": {"zero_raw": 300, "sign": 1,
                    "limits": [3, 30], "wheel_positive": "right"}}
                bus = FeedbackBus(load_config("config.json"), self.arm.positions())
                bus.raw[0] = 300
                bus.wheel = True
                arm = Arm(config, bus)
                with patch.object(bus, "run", wraps=bus.run) as run, \
                     patch.object(arm.cancel, "wait", return_value=True):
                    arm.execute(("jog_joint", (1, command_sign)))
                self.assertEqual(run.call_count, int(should_run))
                self.assertIsNone(bus.running_id)

    @patch("calibration.trace_endpoint", new=sampled_endpoint)
    def test_calibration_reference_can_be_an_endpoint(self):
        from calibration import calibrate_ranges
        readings = [900] * 5 + [300] * 25
        readings += [900] * 10 + [432] * 5 + [666] * 5
        readings += ([300] * 5 + [200] * 5 + [400] * 5 + [300] * 5) * 5
        readings += [666] * 5 + [300] * 25
        with patch("calibration.ServoBus") as bus, \
             patch("calibration.request", return_value=bytes(7)), \
             patch("calibration.time.sleep"), patch("builtins.input", side_effect=[""] * 25 + ["SAVE"]), \
             patch("calibration.save_advanced") as save, contextlib.redirect_stdout(io.StringIO()):
            bus.return_value.__enter__.return_value.read_position.side_effect = readings
            calibrate_ranges(load_advanced(), load_config("config.json"), "unused.json")
            mapping = save.call_args.args[1]["joint_mapping"]["1"]
            self.assertEqual(mapping["zero_raw"], 900)
            self.assertEqual(mapping["limits"], [3, 465])
            self.assertFalse(mapping["crosses_zero"])
            self.assertEqual(save.call_count, 7)

    @patch("calibration.trace_endpoint", new=sampled_endpoint)
    def test_calibration_resumes_completed_base(self):
        import tempfile
        from pathlib import Path
        from calibration import calibrate_ranges
        config = load_advanced()
        mapping = {"zero_raw": 1017, "sign": 1, "limits": [-319, 301],
                   "crosses_zero": True, "wheel_positive": None}
        progress = {"joint_mapping": {"1": mapping}, "joint_directions": {"1": -1},
                    "limits_raw": config["limits_raw"]}
        readings = [1017] * 5 + [300] * 25
        readings += ([300] * 5 + [200] * 5 + [400] * 5 + [300] * 5) * 5
        readings += [1017] * 5 + [300] * 25
        with tempfile.TemporaryDirectory(dir=".") as folder:
            path = Path(folder) / "advanced.json"
            path.with_suffix(".calibration.json").write_text(json.dumps(progress), encoding="utf-8")
            with patch("calibration.ServoBus") as bus, \
                 patch("calibration.request", return_value=bytes(7)), patch("calibration.time.sleep"), \
                 patch("builtins.input", side_effect=[""] * 21 + ["SAVE"]) as prompt, \
                 patch("calibration.save_advanced") as save, contextlib.redirect_stdout(io.StringIO()):
                bus.return_value.__enter__.return_value.read_position.side_effect = readings
                calibrate_ranges(config, load_config("config.json"), path)
                self.assertEqual(save.call_args.args[1]["joint_mapping"]["1"], mapping)
                self.assertFalse(any("ID 1 底座" in str(c) for c in prompt.call_args_list))

    def test_signed_mapping_crosses_zero_and_rejects_wrong_arc(self):
        from joint_mapping import mapped_position, raw_position, check_position
        mapping = {"zero_raw": 1022, "sign": 1, "limits": [-323, 294]}
        for raw, signed in [(696, -326), (1022, 0), (0, 2), (295, 297)]:
            self.assertEqual(mapped_position(raw, mapping), signed)
            self.assertEqual(raw_position(signed, mapping), raw)
        self.assertEqual(check_position(0, mapping), 2)
        with self.assertRaises(ValueError):
            check_position(500, mapping)

    def test_mapped_wheel_crossing_never_writes_position_target(self):
        config = copy.deepcopy(self.config)
        config["motion_calibrated"] = True
        config["joint_mapping"] = {"1": {"zero_raw": 1022, "sign": 1,
            "limits": [-20, 7], "crosses_zero": True, "wheel_positive": "right"}}
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        bus.wheel = True
        arm = Arm(config, bus)
        samples = []
        for raw in [1021, 1022, 0, 2]:
            pose = self.arm.positions().copy()
            pose[0] = raw
            samples.append(pose)
        with patch.object(arm, "positions", side_effect=samples), \
             patch.object(arm.cancel, "wait", return_value=False), \
             patch.object(bus, "run", wraps=bus.run) as run:
            arm.execute(("jog_joint", (1, -1)))
        run.assert_called_once_with(1, "right", 100)
        self.assertIsNone(bus.running_id)
        self.assertFalse(any(address == GOAL_POSITION for _, address, _ in bus.writes))

    def test_mapped_joint_requires_verified_wheel_direction(self):
        config = copy.deepcopy(self.config)
        config["motion_calibrated"] = True
        config["joint_mapping"] = {"1": {"zero_raw": 1022, "limits": [-323, 294], "wheel_positive": None}}
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        bus.wheel = True
        arm = Arm(config, bus)
        with self.assertRaisesRegex(ValueError, "正方向"):
            arm.execute(("jog_joint", (1, 1)))
        self.assertEqual(bus.writes, [])

    @patch("calibration.trace_endpoint", new=sampled_endpoint)
    def test_calibration_retries_unstable_and_out_of_range_sample(self):
        from calibration import calibrate_ranges
        readings = [300] * 30
        readings += [300] * 5 + [200] * 5 + [400] * 5
        readings += [300, 310, 300, 310, 300] + [450] * 5 + [300] * 5
        readings += ([300] * 5 + [200] * 5 + [400] * 5 + [300] * 5) * 5
        readings += [300] * 30
        replies = [""] * 27 + ["SAVE"]
        with patch("calibration.ServoBus") as bus, \
             patch("calibration.request", return_value=bytes(7)), \
             patch("calibration.time.sleep"), patch("builtins.input", side_effect=replies) as prompt, \
             patch("calibration.save_advanced") as save, contextlib.redirect_stdout(io.StringIO()):
            bus.return_value.__enter__.return_value.read_position.side_effect = readings
            calibrate_ranges(load_advanced(), load_config("config.json"), "unused.json")
            self.assertEqual(prompt.call_count, 28)
            saved = save.call_args.args[1]
            self.assertEqual(saved["limits_raw"], [[203, 397]] * 6)
            self.assertTrue(saved["motion_calibrated"])

    def setUp(self):
        self.arm = Arm(load_advanced())
        self.config = self.arm.config
        self.config["llm"] = {}

    def test_manual_jog_step_works_outside_mapping_limits(self):
        config = load_advanced()
        config["motion_calibrated"] = True
        raw = list(config["poses"]["立正"])
        raw[1] = 200  # outside the saved shoulder mapping range, inside the servo range
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)

        with self.assertRaisesRegex(ValueError, "超出安全范围"):
            arm.validate_pose(raw, tolerance=0)
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.manual_jog_step(2, 1, 8)

        self.assertEqual(bus.raw[1], 192)
        with self.assertRaisesRegex(ValueError, "超出安全范围"):
            arm.validate_pose(bus.raw, tolerance=0)

    def test_manual_jog_step_crosses_encoder_seam(self):
        config = load_advanced()
        config["motion_calibrated"] = True
        raw = list(config["poses"]["立正"])
        raw[4] = 5
        bus = FeedbackBus(load_config("config.json"), raw)
        bus.wheel_positions[5] = [4, 3, 2, 1, 0, 1023, 1022, 1021]
        arm = Arm(config, bus)

        with patch.object(arm.cancel, "wait", return_value=False):
            arm.manual_jog_step(5, 1, 8)

        self.assertEqual(bus.raw[4], 1021)
        self.assertEqual(bus.mode(5)[0], "position")

    def test_named_pose_action_and_stop(self):
        for text, expected in [("立正", ("pose", "立正")), ("请招手", ("action", "挥手")),
                               ("机械臂，请停止！", ("stop", None))]:
            with self.subTest(text=text):
                self.assertEqual(parse_command(text, self.config), expected)

    def test_all_cartesian_directions(self):
        for text, axis, sign in [("向前移动", 0, 1), ("向后移动", 0, -1),
                                  ("向左移动", 1, 1), ("向右移动", 1, -1),
                                  ("向上移动", 2, 1), ("向下移动", 2, -1)]:
            with self.subTest(text=text):
                arm = Arm(load_advanced())
                before = arm.model.forward(arm.positions())
                with contextlib.redirect_stdout(io.StringIO()):
                    arm.execute(parse_command(text + "五毫米", arm.config))
                delta = arm.model.forward(arm.positions()) - before
                self.assertGreater(delta[axis] * sign, 3)
                self.assertLess(np.linalg.norm(np.delete(delta, axis)), 2)

    def test_continuous_direction_and_explicit_distance(self):
        self.assertEqual(parse_command("向前移动", self.config), ("jog", [5, 0, 0]))
        self.assertEqual(parse_command("向前移动五毫米", self.config), ("move", [5, 0, 0]))
        self.assertEqual(parse_command("向左转", self.config), ("jog_joint", (1, 1)))
        self.assertEqual(parse_command("别动了", self.config), ("stop", None))

    def test_continuous_joint_stops_at_boundary(self):
        start = int(self.arm.positions()[0])
        self.config["limits_raw"][0] = [start - 8, start + 8]
        self.config["joint_directions"] = {"1": -1}
        with contextlib.redirect_stdout(io.StringIO()):
            self.arm.execute(("jog_joint", (1, 1)))
        self.assertEqual(self.arm.positions()[0], start - 8)
        self.assertTrue(self.arm.cancel.is_set())

    def test_continuous_motion_stops_without_another_step(self):
        runner = CommandRunner(self.arm)
        moved = threading.Event()
        with patch.object(self.arm, "move_cartesian", side_effect=lambda _: moved.set()) as move:
            runner.submit(("jog", [5, 0, 0]))
            self.assertTrue(moved.wait(1))
            runner.submit(("stop", None))
            runner.wait()
            self.assertEqual(move.call_count, 1)
        self.assertFalse(runner.busy)

    def test_continuous_real_joint_requires_calibration(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        self.config["joint_directions"] = {"1": 1}
        arm = Arm(self.config, bus)
        with self.assertRaisesRegex(ValueError, "motion_calibrated"):
            arm.execute(("jog_joint", (1, 1)))
        self.assertFalse(any(address == 40 and values == [1] for _, address, values in bus.writes))
        np.testing.assert_equal(bus.raw, self.arm.positions())

    def test_numbers_units_and_asr_spaces(self):
        for text, expected in [("向 前 移动 五 毫米", 5), ("向前移动两厘米", 20),
                               ("向前移动1.5厘米。", 15), ("向前移动十五毫米", 15)]:
            with self.subTest(text=text):
                self.assertEqual(parse_command(text, self.config), ("move", [expected, 0, 0]))

    def test_non_commands_negations_and_multiple_commands_rejected(self):
        for text in ["不要向前移动", "不要停止", "向前移动然后向后移动", "向前移动50厘米",
                     "向前移动零毫米", "向前移动负五毫米", "随便聊一下", "立正挥手"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_command(text, self.config)

    def test_inverse_roundtrip_and_nearest_branch(self):
        raw = self.arm.positions()
        for offset in [(0, 0, 0), (15, -10, 8), (-15, 10, -8)]:
            xyz = self.arm.model.forward(raw) + offset
            solution = self.arm.model.inverse(xyz, raw)
            self.assertLess(np.linalg.norm(self.arm.model.forward(solution) - xyz), 2)
            self.assertLess(np.linalg.norm(solution[:4] - raw[:4]), 70)

    def test_inverse_uses_calibrated_directions(self):
        config = copy.deepcopy(self.config)
        config["kinematics"]["directions"] = [-1, -1, 1, -1, 1]
        model = Kinematics(config)
        raw = self.arm.positions()
        raw[:5] = np.rint(model.zero + (raw[:5] - model.zero) / model.signs).astype(int)
        xyz = model.forward(raw)
        result = model.inverse(xyz + [0, 5, 0], raw)
        self.assertGreater(result[0], raw[0])
        self.assertLess(np.linalg.norm(model.forward(result) - xyz - [0, 5, 0]), 2)

    def test_unreachable_nonfinite_and_limits(self):
        for xyz in [[2000, 0, 0], [float("nan"), 0, 0]]:
            with self.assertRaises(ValueError):
                self.arm.model.inverse(xyz, self.arm.positions())
        with self.assertRaises(ValueError):
            self.arm.move_pose([0] * 6)

    def test_cartesian_step_limit(self):
        original = self.arm.positions()
        with self.assertRaises(ValueError):
            self.arm.move_cartesian([21, 0, 0])
        np.testing.assert_equal(self.arm.positions(), original)

    def test_urdf_zero_pose_and_tool_offset(self):
        raw = np.r_[self.arm.model.zero, 463]
        np.testing.assert_allclose(self.arm.model.forward(raw),
                                   [391.361470, -0.009212, 226.469710], atol=0.0001)
        rotated = raw.copy()
        rotated[4] += 90 / self.arm.config["kinematics"]["degrees_per_raw"]
        self.assertGreater(np.linalg.norm(self.arm.model.forward(rotated) - self.arm.model.forward(raw)), 1)

    def test_fifty_mm_lines_include_individual_joint_paths(self):
        for axis in range(3):
            for sign in [-1, 1]:
                with self.subTest(axis=axis, sign=sign):
                    arm = Arm(load_advanced())
                    origin = arm.model.forward(arm.positions())
                    samples = []
                    original_move = arm.move_pose

                    def trace(target, *args, **kwargs):
                        raw = arm.positions()
                        self.assertLessEqual(np.max(np.abs(target - raw)), 20)
                        for fraction in np.linspace(0, 1, 21):
                            samples.append(arm.model.forward(raw + (target - raw) * fraction))
                        original_move(target, *args, **kwargs)

                    delta = np.zeros(3)
                    delta[axis] = sign * 50
                    with patch.object(arm, "move_pose", side_effect=trace), contextlib.redirect_stdout(io.StringIO()):
                        arm.execute(("move", delta))
                    self.assertLess(np.linalg.norm(arm.model.forward(arm.positions()) - origin - delta), 2)
                    lateral = np.delete(np.array(samples) - origin, axis, axis=1)
                    self.assertLess(np.max(np.linalg.norm(lateral, axis=1)), 3.5)

    def test_unreachable_diagonal_continues_in_feasible_direction(self):
        raw = self.arm.positions()
        self.config["limits_raw"][0] = [int(raw[0]), 700]
        self.arm.model = Kinematics(self.config)
        origin = self.arm.model.forward(raw)
        with self.assertRaisesRegex(ValueError, "IK 无解"):
            self.arm.model.inverse(origin + [3, 3, 0], raw)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(self.arm.move_cartesian([3, 3, 0]))
        actual = self.arm.model.forward(self.arm.positions()) - origin
        self.assertGreater(actual[0], 1)
        self.assertLess(abs(actual[1]), 0.01)
        self.assertEqual(self.arm.positions()[0], raw[0])
        self.assertIn("IK 无解", output.getvalue())

    def test_mapped_fifty_mm_lines_stay_in_one_branch_and_hold_wrist_and_gripper(self):
        shifts = np.array([380, 100, 100, 100, 300, 0])
        raw = (self.arm.positions() + shifts) % 1024
        config = copy.deepcopy(self.config)
        config["kinematics"]["zero_raw"] = ((np.array(config["kinematics"]["zero_raw"]) + shifts[:5]) % 1024).tolist()
        config["joint_mapping"] = {str(i + 1): {"zero_raw": int((512 + shifts[i]) % 1024),
                                               "sign": -1 if i % 2 else 1, "limits": [-411, 411],
                                               "unwrapped": True} for i in range(6)}
        config["poses"]["立正"] = raw.tolist()
        for axis in range(3):
            for sign in [-1, 1]:
                with self.subTest(axis=axis, sign=sign):
                    arm = Arm(config, FeedbackBus(load_config("config.json"), raw))
                    arm.bus = None
                    origin = arm.model.forward(raw)
                    np.testing.assert_allclose(origin, self.arm.model.forward(self.arm.positions()), atol=1e-9)
                    samples, crossings = [], []
                    original_move = arm.move_pose

                    def trace(target, *args, **kwargs):
                        previous = arm.positions()
                        self.assertLessEqual(np.max(np.abs(arm.model.unwrap(target) - arm.model.unwrap(previous))), 20)
                        arm.model.validate(target)
                        np.testing.assert_equal(target[4:], raw[4:])
                        crossings.append(np.max(np.abs(target - previous)) > 1000)
                        change = arm.model.unwrap(target) - arm.model.unwrap(previous)
                        for fraction in np.linspace(0, 1, 21):
                            samples.append(arm.model.forward((previous + change * fraction) % 1024))
                        original_move(target, *args, **kwargs)

                    delta = np.zeros(3)
                    delta[axis] = sign * 50
                    with patch.object(arm, "move_pose", side_effect=trace), contextlib.redirect_stdout(io.StringIO()):
                        arm.move_linear(delta)
                    self.assertLess(np.linalg.norm(arm.model.forward(arm.positions()) - origin - delta), 2)
                    self.assertLess(np.max(np.linalg.norm(np.delete(np.array(samples) - origin, axis, axis=1), axis=1)), 3.5)
                    self.assertFalse(arm.last_move["projected"])
                    self.assertFalse(any(crossings))

    def test_mapped_ik_projection_respects_measured_boundary_and_original_direction(self):
        config = copy.deepcopy(self.config)
        config["joint_mapping"] = {str(i): {"zero_raw": 512, "sign": 1,
                                           "limits": [0, 411] if i == 1 else [-411, 411]} for i in range(1, 7)}
        model = Kinematics(config)
        raw = self.arm.positions()
        origin = model.forward(raw)
        with self.assertRaisesRegex(ValueError, "IK 无解"):
            model.inverse(origin + [3, 3, 0], raw)
        target = model.project_step(np.array([3, 3, 0]), raw)
        model.validate(target)
        actual = model.forward(target) - origin
        self.assertGreater(actual[0], 0)
        self.assertLess(abs(actual[1]), 0.01)
        self.assertEqual(target[0], raw[0])
        with self.assertRaisesRegex(ValueError, "没有保留原方向"):
            model.project_step(np.array([0, 3, 0]), raw)

    def test_no_feasible_direction_stops_without_motion(self):
        raw = self.arm.positions()
        self.config["limits_raw"][0] = [int(raw[0]), 700]
        self.arm.model = Kinematics(self.config)
        runner = CommandRunner(self.arm)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.submit(("move", [0, 3, 0]))
            with self.assertRaisesRegex(ValueError, "没有保留原方向"):
                runner.wait()
        self.assertTrue(self.arm.cancel.is_set())
        np.testing.assert_array_equal(self.arm.positions(), raw)

    def test_stop_interrupts_long_distance_waypoints(self):
        with patch.object(self.arm, "move_pose", side_effect=lambda *a, **k: self.arm.stop()) as move:
            with self.assertRaises(InterruptedError), contextlib.redirect_stdout(io.StringIO()):
                self.arm.execute(("move", [0, 0, 50]))
            self.assertEqual(move.call_count, 1)

    def test_distance_limit_and_negation_never_reach_language_model(self):
        config = load_advanced()
        with patch("urllib.request.urlopen") as request:
            self.assertEqual(parse_command("向上移动五厘米", config), ("move", [0, 0, 50]))
            for text in ["向前移动五十厘米", "向前移动负五毫米", "不要向前移动"]:
                with self.assertRaises(ValueError):
                    parse_command(text, config)
            request.assert_not_called()

    def test_keyboard_release_focus_loss_and_repeat(self):
        from types import SimpleNamespace
        from advanced import keyboard_control

        tk = MagicMock()
        window, entry = tk.Tk.return_value, tk.Entry.return_value
        runner = MagicMock(arm=self.arm, busy=False, error=None)
        pending = []
        window.after.side_effect = lambda ms, fn: pending.append((ms, fn)) or str(len(pending))
        window.after_idle.side_effect = lambda fn: fn()

        def exercise():
            callbacks = dict(call.args for call in window.bind.call_args_list)
            key = lambda name, widget=window: SimpleNamespace(keysym=name, widget=widget)
            callbacks["<KeyPress>"](key("w"))
            callbacks["<KeyPress>"](key("w"))
            runner.submit.assert_called_once_with(("jog", [5, 0, 0]))
            callbacks["<KeyRelease>"](key("w"))
            self.assertEqual(pending[-1][0], 25)
            pending[-1][1]()
            runner.submit.assert_called_with(("stop", None))
            callbacks["<KeyPress>"](key("d"))
            window.focus_displayof.return_value = None
            callbacks["<FocusOut>"](key(""))
            runner.submit.assert_called_with(("stop", None))
            runner.submit.reset_mock()
            callbacks["<KeyPress>"](key("1"))
            callbacks["<KeyPress>"](key("1"))
            runner.submit.assert_called_once_with(("pose", "立正"))
            callbacks["<KeyPress>"](key("w", entry))
            self.assertEqual(runner.submit.call_count, 1)
            callbacks["<KeyPress>"](key("space", entry))
            runner.submit.assert_called_with(("stop", None))

        window.mainloop.side_effect = exercise
        with patch.dict("sys.modules", {"tkinter": tk}):
            keyboard_control(runner)
        runner.stop.assert_called_once()

    def test_legacy_planar_config_still_loads(self):
        config = load_advanced()
        config["kinematics"].pop("urdf")
        config["kinematics"]["zero_raw"] = config["kinematics"]["zero_raw"][:4]
        config["kinematics"]["directions"] = config["kinematics"]["directions"][:4]
        arm = Arm(config)
        target = arm.model.forward(arm.positions()) + [0, 5, 0]
        raw = arm.model.inverse(target, arm.positions())
        self.assertLess(np.linalg.norm(arm.model.forward(raw) - target), 2)

    def test_cross_zero_pose_remains_blocked_before_writes(self):
        self.config["motion_calibrated"] = True
        self.config["joint_mapping"] = {"1": {"crosses_zero": True}}
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        arm = Arm(self.config, bus)
        with self.assertRaisesRegex(ValueError, "跨零"):
            arm.move_pose(self.config["poses"]["向左转"])
        self.assertEqual(bus.writes, [])

    def test_mapped_joint_jog_supports_position_mode_without_wheel_direction(self):
        config = load_advanced()
        raw = [980, 880, 330, 680, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        original_move = arm.move_pose

        def move_once(*args, **kwargs):
            original_move(*args, **kwargs)
            arm.cancel.set()

        with patch.object(arm, "move_pose", side_effect=move_once), contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("jog_joint", (1, 1)))
        self.assertEqual(bus.raw[0], 968)
        self.assertIsNone(bus.running_id)
        self.assertTrue(arm.cancel.is_set())

    def test_mapped_pose_uses_current_limits_without_crossing_feedback_zero(self):
        config = load_advanced()
        config["joint_mapping"]["1"]["wheel_positive"] = None
        config["kinematics"]["calibrated"] = False
        raw = [980, 880, 330, 680, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        target = list(raw)
        target[0] = 990
        with contextlib.redirect_stdout(io.StringIO()):
            arm.move_pose(target, duration_s=0.2)
        self.assertEqual(bus.raw, target)
        self.assertIsNone(bus.running_id)
        bus.writes.clear()
        target[0] = 10
        with self.assertRaisesRegex(PositionHoldTimeout, "实测连续模式方向"):
            arm.move_pose(target)
        self.assertEqual(bus.writes, [])
        target[0] = 500
        with self.assertRaisesRegex(ValueError, "映射位置"):
            arm.move_pose(target)
        self.assertEqual(bus.writes, [])
        with self.assertRaisesRegex(ValueError, "末端控制前"):
            arm.move_cartesian([0, 0, 1])

    def test_standing_keeps_resending_support_targets_after_arrival(self):
        config = load_advanced()
        raw = [1022, 880, 330, 680, 100, 280]
        target = [1022, 960, 250, 716, 957, 380]
        expected = target[:4] + raw[4:]
        config["poses"]["立正"] = target
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        clock, held = [0.0], [0]
        def wait(seconds):
            clock[0] += seconds
            if arm.holding:
                self.assertEqual(bus._read_u8(1, 40), 0)
                self.assertTrue(all(bus._read_u8(joint, 40) == 1 for joint in range(2, 7)))
                held[0] += 1
                if held[0] == 1:
                    bus.raw[1] += 8
                if held[0] >= 15:
                    arm.cancel.set()
                    return True
            return False
        with patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("pose", "立正"))
        self.assertGreaterEqual(len(bus.batches), 3)
        for address, values in bus.batches:
            self.assertEqual(address, GOAL_POSITION)
            self.assertEqual(set(values), set(range(2, 7)))
            self.assertEqual([(values[joint][0] << 8) | values[joint][1] for joint in range(2, 7)], expected[1:])
        self.assertEqual(bus.raw, expected)
        self.assertEqual(config["poses"]["立正"], target)
        self.assertFalse(arm.holding)
        self.assertIsNone(bus.running_id)
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))

    def test_pose_two_crossing_hands_over_before_far_target(self):
        config = load_advanced()
        raw = [1019, 1019, 19, 729, 78, 270]
        bus = FeedbackBus(load_config("config.json"), raw)
        bus.wheel_positions[2] = [1023, 5, 13, 21, 49, 77, 105, 100]
        arm = Arm(config, bus)
        clock = [0.0]
        transition = []
        mode = bus.set_mode
        def restore(joint, value, **kwargs):
            if value == "position":
                transition.append(bus.raw[joint - 1])
            return mode(joint, value, **kwargs)
        def wait(seconds):
            clock[0] += seconds
            return False
        with patch.object(bus, "set_mode", side_effect=restore), \
             patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("pose", "姿态2"))
        self.assertEqual(transition, [21])
        self.assertEqual(bus.wheel_positions[2], [49, 77, 105, 100])
        self.assertEqual(bus.raw, config["poses"]["姿态2"])

    def test_crossing_hands_over_after_small_overshoot_on_either_side(self):
        for start, target, feedback in [(1019, 16, [1023, 7, 23]), (4, 1007, [0, 1018, 999])]:
            with self.subTest(start=start, target=target):
                config = load_advanced()
                raw = [1022, start, 315, 696, 80, 276]
                bus = FeedbackBus(load_config("config.json"), raw)
                bus.wheel_positions[2] = list(feedback)
                arm = Arm(config, bus)
                goal = list(raw)
                goal[1] = target
                with patch.object(arm.cancel, "wait", return_value=False):
                    arm.move_pose(goal, quiet=True)
                self.assertEqual(bus.raw, goal)
                self.assertEqual(bus.mode(2)[0], "position")
                self.assertIsNone(bus.running_id)

    def test_standing_crossing_uses_recorded_shoulder_waypoint_and_restores_modes(self):
        from joint_mapping import mapped_position, raw_position
        config = load_advanced()
        raw = config["pose_paths"]["立正"][0]
        target = config["poses"]["立正"]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        for joint in (2, 3):
            mapping = config["joint_mapping"][str(joint)]
            start = mapped_position(1007 if joint == 2 else 16, mapping)
            end = mapped_position(target[joint - 1], mapping)
            count = max(1, int(np.ceil(abs(end - start) / 6)))
            bus.wheel_positions[joint] = [raw_position(round(value), mapping)
                                          for value in np.linspace(start, end, count + 1)[1:]]
        write = bus._write
        def firmware(joint, address, values):
            if address == GOAL_POSITION:
                self.assertEqual(bus.mode(joint)[0], "position")
            if joint == 3 and address == 11 and values == [0, 0]:
                self.assertEqual(bus.mode(2)[0], "position")
                self.assertEqual(bus.raw[1], 55)
                self.assertEqual(bus._read_u16(2, GOAL_POSITION), 55)
                self.assertEqual(bus._read_u8(2, 40), 1)
            write(joint, address, values)
            if joint == 2 and address == 40 and values == [1] and not bus.batches:
                bus.raw[1] -= 20
        clock = [0.0]
        def wait(seconds):
            clock[0] += seconds
            if arm.holding and len(bus.batches) >= 3:
                return True
            self.assertLess(clock[0], 16.0)
            return False
        with patch.object(bus, "_write", side_effect=firmware), \
             patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("pose", "立正"))
        self.assertEqual(bus.batches[0][0], GOAL_POSITION)
        self.assertEqual(set(bus.batches[0][1]), set(range(2, 7)))
        self.assertEqual([(bus.batches[0][1][joint][0] << 8) | bus.batches[0][1][joint][1] for joint in (2, 3)], [1007, 16])
        self.assertIn((RUNNING_TIME, {2: [4, 250]}), bus.batches)
        self.assertIn((RUNNING_TIME, {3: [1, 244]}), bus.batches)
        for address, values in bus.batches:
            if address == RUNNING_TIME:
                self.assertTrue(set(values) <= {2, 3})
                self.assertTrue(all(len(data) == 2 for data in values.values()))
        self.assertFalse(any(joint in (5, 6) and address in (9, 11, 48) for joint, address, _ in bus.writes))
        self.assertEqual(bus.raw, raw[:1] + target[1:4] + raw[4:])
        self.assertTrue(all(bus.mode(joint)[0] == "position" for joint in range(1, 7)))
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
        self.assertIsNone(bus.running_id)

    def test_standing_repositions_shoulder_before_elbow_crossing_from_pose_two(self):
        config = load_advanced()
        raw = [1022, 135, 19, 729, 78, 276]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        bus.wheel_positions[3] = [10, 2, 1022, 1017, 1011, 1005, 999]
        write = bus._write
        def command(joint, address, values):
            if joint == 3 and address == 11 and values == [0, 0]:
                self.assertEqual(bus.raw[1], 55)
                self.assertEqual(bus._read_u16(2, GOAL_POSITION), 55)
                self.assertEqual(bus._read_u8(2, 40), 1)
            write(joint, address, values)
        clock = [0.0]
        def wait(seconds):
            clock[0] += seconds
            self.assertLess(clock[0], 4)
            return arm.holding
        with patch.object(bus, "_write", side_effect=command), \
             patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("pose", "立正"))
        self.assertEqual(bus.raw, raw[:1] + config["poses"]["立正"][1:4] + raw[4:])
        self.assertTrue(all(bus._read_u8(j, 40) == 0 for j in range(1, 7)))

    def test_standing_approach_tolerance_keeps_final_pose_tolerance(self):
        from joint_mapping import mapped_position, raw_position
        for joint in (2, 3):
            for offset, final_offset in ((5, 0), (6, 0), (7, 0), (5, 5)):
                with self.subTest(joint=joint, offset=offset, final_offset=final_offset):
                    config = load_advanced()
                    raw = [1022, 900, 330, 700, 100, 280]
                    target = list(raw)
                    target[joint - 1] = 120 if joint == 2 else 874
                    config["poses"]["立正"] = target
                    edge, direction = (1007, -1) if joint == 2 else (16, 1)
                    feedback = edge + direction * offset
                    goal = target[joint - 1]
                    bus = FeedbackBus(load_config("config.json"), raw)
                    arm = Arm(config, bus)
                    mapping = config["joint_mapping"][str(joint)]
                    start, end = mapped_position(feedback, mapping), mapped_position(goal, mapping)
                    count = max(1, int(np.ceil(abs(end - start) / 6)))
                    bus.wheel_positions[joint] = [raw_position(round(value), mapping)
                                                  for value in np.linspace(start, end, count + 1)[1:]]
                    write = bus._write
                    def firmware(servo, address, values):
                        write(servo, address, values)
                        if servo == joint and address == GOAL_POSITION:
                            value = (values[0] << 8) | values[1]
                            if value == edge:
                                bus.raw[joint - 1] = feedback
                            elif value == goal:
                                bus.raw[joint - 1] = goal + direction * final_offset
                    clock, held = [0.0], [False]
                    def wait(seconds):
                        clock[0] += seconds
                        held[0] = arm.holding
                        self.assertLess(clock[0], 6.0)
                        return held[0]
                    with patch.object(bus, "_write", side_effect=firmware), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if offset == 7 or final_offset:
                            expected_goal = edge if offset == 7 else goal
                            expected_feedback = feedback if offset == 7 else goal + direction * final_offset
                            with self.assertRaisesRegex(TimeoutError, f"ID {joint} 未在期限内到位：目标 {expected_goal}，反馈 {expected_feedback}"):
                                arm.execute(("pose", "立正"))
                        else:
                            arm.execute(("pose", "立正"))
                    self.assertEqual(held[0], offset <= 6 and final_offset == 0)
                    if offset == 7:
                        self.assertFalse(any(address in (9, 11, 48) for _, address, _ in bus.writes))
                    else:
                        self.assertTrue(any(servo == joint and address == 11 and values == [0, 0]
                                            for servo, address, values in bus.writes))
                    self.assertTrue(all(bus.mode(servo)[0] == "position" for servo in range(1, 7)))
                    self.assertTrue(all(bus._read_u8(servo, 40) == 0 for servo in range(1, 7)))
                    self.assertIsNone(bus.running_id)

    def test_standing_crossing_stall_reverse_and_jump_stop_all_joints(self):
        for fault, feedback in (("stall", []), ("reverse", [1006]), ("jump", [135]), ("rebound", [1018, 909])):
            with self.subTest(fault=fault):
                config = load_advanced()
                raw = [1022, 900 if fault == "rebound" else 1010, 330, 700, 100, 280]
                config["poses"]["立正"] = [1018, 120, 330, 716, 957, 380]
                bus = FeedbackBus(load_config("config.json"), raw)
                bus.wheel_positions[2] = feedback
                arm = Arm(config, bus)
                clock = [0.0]
                def wait(seconds):
                    clock[0] += seconds
                    self.assertLess(clock[0], 3.0)
                    return False
                with patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]):
                    message = ("上次 1018，反馈 909，映射步进 109" if fault == "rebound" else
                               "突跳或方向反转.*上次 1010.*异常反馈 135.*复核 135" if fault == "jump" else
                               "未取得有效进展.*起点 1010.*反馈 1010" if fault == "stall" else "突跳或方向反转.*起点 1010")
                    with self.assertRaisesRegex((RuntimeError, TimeoutError), message):
                        arm.execute(("pose", "立正"))
                self.assertFalse(arm.holding)
                self.assertIsNone(bus.running_id)
                self.assertTrue(all(bus.mode(joint)[0] == "position" for joint in range(1, 7)))
                self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
                self.assertEqual(bus.raw[4:], raw[4:])

    def test_standing_crossing_checks_sample_time_and_hands_over_near_target(self):
        cases = (
            ("seam", [1013, 1019, 50, 58, 66, 74, 82, 90], {50: 0.03}, None),
            ("exit", [1013, 1019, 3, 12, 20, 72, 80, 88, 96, 104], {72: 0.03}, None),
            ("fast_exit", [1013, 1019, 3, 12, 20, 50, 80, 109, 138, 170], {}, None),
            ("near_target", [1013, 1019, 3, 12, 32], {}, None),
            ("overshoot", [1013, 1019, 3, 12, 32], {}, "偏离目标"),
            ("fast_jump", [1013, 1019, 3, 12, 20, 72], {}, "突跳或方向反转"),
            ("later_read_delay", [1013, 1019, 3, 12, 20, 72], {}, "突跳或方向反转"),
            ("large_jump", [1013, 1019, 130], {130: 0.2}, "突跳或方向反转"),
            ("reverse", [1013, 1006], {1006: 0.1}, "突跳或方向反转"),
            ("stale", [1013], {1013: 0.81}, "反馈间隔过长"),
        )
        for joint in (2, 3):
            for name, trace, delays, message in cases:
                with self.subTest(joint=joint, case=name):
                    config = load_advanced()
                    raw = [1022, 900, 330, 700, 100, 280]
                    target = list(raw)
                    target[joint - 1] = 120 if joint == 2 else 874
                    config["poses"]["立正"] = target
                    trace, delays = list(trace), dict(delays)
                    if name == "near_target":
                        goal = 120 if joint == 2 else 149
                        trace += [goal + 4, goal + 50]
                        delays[goal + 4] = 0.06
                    elif name == "overshoot":
                        goal = 120 if joint == 2 else 149
                        trace.append(goal + 10)
                        delays[goal + 10] = 0.1
                    bus = FeedbackBus(load_config("config.json"), raw)
                    bus.wheel_positions[joint] = trace if joint == 2 else [1023 - value for value in trace]
                    arm = Arm(config, bus)
                    read, clock, held = bus.read_position, [0.0], [False]
                    def feedback(servo):
                        moving = bool(servo == joint and bus.mode(servo)[0] == "wheel"
                                      and bus._read_u8(servo, 40) and bus.wheel_positions[servo])
                        value = read(servo)
                        if moving:
                            clock[0] += delays.get(value if joint == 2 else 1023 - value, 0.0)
                        if name == "later_read_delay" and servo == 6 and bus.raw[joint - 1] == (72 if joint == 2 else 951):
                            clock[0] += 0.1
                        return value
                    def wait(seconds):
                        clock[0] += seconds
                        held[0] = arm.holding
                        self.assertLess(clock[0], 3.0)
                        return held[0]
                    with patch.object(bus, "read_position", side_effect=feedback), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if message:
                            with self.assertRaisesRegex((RuntimeError, TimeoutError), message):
                                arm.execute(("pose", "立正"))
                        else:
                            arm.execute(("pose", "立正"))
                    self.assertEqual(held[0], message is None)
                    if message is None:
                        self.assertEqual(bus.raw, target)
                    if name == "near_target":
                        self.assertEqual(bus.wheel_positions[joint], [170 if joint == 2 else 824])
                    if name == "fast_exit":
                        remaining = [80, 109, 138, 170]
                        self.assertEqual(bus.wheel_positions[joint], remaining if joint == 2 else [1023 - value for value in remaining])
                    self.assertTrue(all(bus.mode(servo)[0] == "position" for servo in range(1, 7)))
                    self.assertTrue(all(bus._read_u8(servo, 40) == 0 for servo in range(1, 7)))
                    self.assertIsNone(bus.running_id)

    def test_standing_large_wrap_requires_consistent_readback(self):
        for joint in (2, 3):
            for fault in (None, "spike", "reverse", "overshoot", "cancel", "missing", "slow"):
                with self.subTest(joint=joint, fault=fault):
                    config = load_advanced()
                    raw = [1022, 900, 330, 700, 100, 280]
                    target = list(raw)
                    target[joint - 1] = 120 if joint == 2 else 874
                    config["poses"]["立正"] = target
                    trace = [1013, 1018, 59, 61, 65, 69, 73, 77, 81, 85]
                    if fault == "spike":
                        trace[3] = 1018
                    elif fault == "reverse":
                        trace[3] = 50
                    elif fault == "overshoot":
                        trace[3] = 130 if joint == 2 else 159
                    bus = FeedbackBus(load_config("config.json"), raw)
                    bus.wheel_positions[joint] = trace if joint == 2 else [1023 - value for value in trace]
                    arm = Arm(config, bus)
                    read, clock, held, rechecked = bus.read_position, [0.0], [False], []
                    def feedback(servo):
                        moving = bool(servo == joint and bus.mode(servo)[0] == "wheel"
                                      and bus._read_u8(servo, 40) and bus.wheel_positions[servo])
                        if moving and bus.raw[servo - 1] == (59 if joint == 2 else 964):
                            rechecked.append(len(bus.writes))
                            if fault == "missing":
                                raise RuntimeError("wrap readback missing")
                            if fault == "cancel":
                                arm.cancel.set()
                            if fault == "slow":
                                clock[0] += 0.9
                        value = read(servo)
                        if moving:
                            clock[0] += 0.012
                        return value
                    def wait(seconds):
                        clock[0] += seconds
                        held[0] = arm.holding
                        self.assertLess(clock[0], 3.0)
                        return held[0]
                    with patch.object(bus, "read_position", side_effect=feedback), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if fault:
                            message = {"cancel": "动作已停止", "missing": "wrap readback missing", "slow": "反馈间隔过长"}.get(fault, "边界回读不一致")
                            with self.assertRaisesRegex((RuntimeError, TimeoutError, InterruptedError), message):
                                arm.execute(("pose", "立正"))
                        else:
                            arm.execute(("pose", "立正"))
                    self.assertTrue(rechecked)
                    self.assertEqual(held[0], fault is None)
                    if fault is None:
                        self.assertEqual(bus.raw, target)
                    else:
                        self.assertFalse(any(address == GOAL_POSITION or (address == 40 and values == [1])
                                             for _, address, values in bus.writes[rechecked[0]:]))
                    self.assertTrue(all(bus.mode(servo)[0] == "position" for servo in range(1, 7)))
                    self.assertTrue(all(bus._read_u8(servo, 40) == 0 for servo in range(1, 7)))
                    self.assertIsNone(bus.running_id)

    def test_sequential_crossing_seam_feedback_keeps_direction_and_stall_guards(self):
        cases = [
            (1007, [1012, 1019, 4, 0, 5, 11, 18], None),
            (1007, [1003], "方向反转"),
            (980, [990, 986], "偏离目标"),
            (1007, [1012, 1019, 4, 0], "未取得有效进展"),
            (1007, [1012, 40], "突跳"),
        ]
        for joint in (2, 3):
            for initial, samples, error in cases:
                with self.subTest(joint=joint, samples=samples):
                    raw = [1022, 900, 315, 740, 82, 276]
                    raw[joint - 1] = initial if joint == 2 else 1023 - initial
                    bus = FeedbackBus(load_config("config.json"), raw)
                    bus.wheel_positions[joint] = list(samples) if joint == 2 else [1023 - value for value in samples]
                    arm = Arm(load_advanced(), bus)
                    clock = [0.0]
                    def wait(seconds):
                        clock[0] += seconds
                        self.assertLess(clock[0], 2.0)
                        return False
                    with patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]):
                        if error:
                            with self.assertRaisesRegex(PositionHoldTimeout, error):
                                arm.move_across_zero(joint, 135 if joint == 2 else 874)
                        else:
                            arm.move_across_zero(joint, 135 if joint == 2 else 874)
                            self.assertEqual(bus.raw[joint - 1], 18 if joint == 2 else 1005)
                    self.assertEqual(bus.mode(joint)[0], "position")
                    self.assertEqual(bus._read_u8(joint, 40), 0)
                    self.assertIsNone(bus.running_id)

    def test_crossing_transient_requires_three_torque_off_samples_before_resuming(self):
        for standing in (False, True):
            for joint in (2, 3):
                with self.subTest(standing=standing, joint=joint):
                    config = load_advanced()
                    raw = [1022, 900, 330, 700, 100, 280]
                    raw[joint - 1] = 1007 if joint == 2 else 16
                    target = list(raw)
                    target[joint - 1] = 120 if joint == 2 else 874
                    config["poses"]["立正"] = target
                    bus = FeedbackBus(load_config("config.json"), raw)
                    arm = Arm(config, bus)
                    trace = ([1018, 339, 0, 0, 0, 5, 11, 17, 21] if joint == 2 else
                             [5, 291, 1023, 1023, 1023, 1018, 1012, 1006, 1002])
                    sampled, clock, active = [], [0.0], [False]
                    read, write = bus.read_position, bus._write
                    def feedback(servo):
                        if servo == joint and active[0] and bus.mode(joint)[0] == "wheel" and trace:
                            value = trace.pop(0)
                            sampled.append((value, bus._read_u8(joint, 40)))
                            bus.raw[joint - 1] = value
                            return value
                        return read(servo)
                    def command(servo, address, values):
                        write(servo, address, values)
                        if servo == joint and address == RUNNING_TIME and values != [0, 0]:
                            active[0] = True
                    def wait(seconds):
                        clock[0] += seconds
                        self.assertLess(clock[0], 4)
                        return arm.holding
                    with patch.object(bus, "read_position", side_effect=feedback), \
                         patch.object(bus, "_write", side_effect=command), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if standing:
                            arm.execute(("pose", "立正"))
                        else:
                            arm.move_pose(target, quiet=True)
                    self.assertEqual([torque for _, torque in sampled[2:5]], [0, 0, 0])
                    self.assertEqual(sampled[5][1], 1)
                    self.assertEqual(bus.raw, target)
                    self.assertEqual(bus.mode(joint)[0], "position")
                    self.assertIsNone(bus.running_id)

    def test_crossing_transient_recheck_rejects_unconfirmed_feedback_without_reenable(self):
        from joint_mapping import mapped_position
        for fault in ("wrong_edge", "jitter", "cancel", "deadline", "stop_failure"):
            with self.subTest(fault=fault):
                config = load_advanced()
                bus = FeedbackBus(load_config("config.json"), [1022, 1018, 315, 740, 82, 276])
                arm = Arm(config, bus)
                bus.set_mode(2, "wheel", hold_others=True)
                bus.run(2, "right", 250, hold_others=True)
                bus.writes.clear()
                mapping = config["joint_mapping"]["2"]
                trace = [1019] if fault == "wrong_edge" else [0, 5, 0] if fault == "jitter" else [0, 0, 0]
                clock = [0.0]
                def wait(seconds):
                    clock[0] += 0.13 if fault == "deadline" else seconds
                    if fault == "cancel":
                        arm.cancel.set()
                        return True
                    return False
                read = bus._read_u8
                def register(servo, address):
                    if fault == "stop_failure" and servo == 2 and address == 40:
                        return 1
                    return read(servo, address)
                with patch.object(bus, "read_position", side_effect=lambda _: trace.pop(0)), \
                     patch.object(bus, "_read_u8", side_effect=register), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]):
                    with self.assertRaises((RuntimeError, InterruptedError)):
                        arm.confirm_crossing_sample(2, 339, mapped_position(1018, mapping), mapped_position(120, mapping), -1)
                self.assertFalse(any(a == 40 and v == [1] or a == RUNNING_TIME and v != [0, 0]
                                     for _, a, v in bus.writes))

    def test_standing_seam_bounce_requires_progress_and_rejects_exit_faults(self):
        for joint in (2, 3):
            for fault, tail, message in (("bounce", [3, 7, 11, 17], None),
                                         ("false_exit", [25, 25, 1, 1, 1, 4, 7, 11, 15, 19, 23, 26, 29, 32, 35], None),
                                         ("stuck", [8, 0] * 50, "未取得有效进展"),
                                         ("reverse", [1010, 1005], "突跳或方向反转"),
                                         ("jump", [50], "突跳或方向反转")):
                with self.subTest(joint=joint, fault=fault):
                    config = load_advanced()
                    raw = [1022, 900, 330, 700, 100, 280]
                    target = list(raw)
                    target[joint - 1] = 120 if joint == 2 else 874
                    config["poses"]["立正"] = target
                    bus = FeedbackBus(load_config("config.json"), raw)
                    trace = [1012, 1017, 1022, 1017, 1, 8, 0] + tail
                    bus.wheel_positions[joint] = trace if joint == 2 else [1023 - value for value in trace]
                    arm = Arm(config, bus)
                    clock, held = [0.0], [False]
                    def wait(seconds):
                        clock[0] += seconds
                        held[0] = arm.holding
                        self.assertLess(clock[0], 2.0)
                        return held[0]
                    with patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if message:
                            with self.assertRaisesRegex((RuntimeError, TimeoutError), message):
                                arm.execute(("pose", "立正"))
                        else:
                            arm.execute(("pose", "立正"))
                    self.assertEqual(held[0], message is None)
                    if message is None:
                        self.assertEqual(bus.raw, target)
                        remaining = [26, 29, 32, 35] if fault == "false_exit" else []
                        self.assertEqual(bus.wheel_positions[joint], remaining if joint == 2 else [1023 - value for value in remaining])
                    elif fault == "stuck":
                        self.assertTrue(bus.wheel_positions[joint])
                    else:
                        self.assertEqual(bus.raw[joint - 1], tail[-1] if joint == 2 else 1023 - tail[-1])
                    self.assertTrue(all(bus.mode(servo)[0] == "position" for servo in range(1, 7)))
                    self.assertTrue(all(bus._read_u8(servo, 40) == 0 for servo in range(1, 7)))
                    self.assertIsNone(bus.running_id)

    def test_standing_crossing_preserves_loaded_drive_in_both_directions(self):
        for invert, load, expected in ((False, 330, 500), (True, 1354, 500),
                                       (False, 990, 1000), (False, 0, 500), (False, 1354, 500)):
            with self.subTest(invert=invert, load=load):
                config = load_advanced()
                raw = [1022, 900, 330, 700, 100, 280]
                target = list(raw)
                target[2] = 874
                config["poses"]["立正"] = target
                bus = FeedbackBus(load_config("config.json"), raw)
                bus.items[3]["invert_direction"] = invert
                bus.wheel_positions[3] = [16, 8, 0, 1022, 1016, 1007, 1001, 995, 989, 983, 977]
                arm = Arm(config, bus)
                read, position, write = bus._read_u16, bus.read_position, bus._write
                sampled = []
                def read_load(servo, address):
                    if servo == 3 and address == 60:
                        sampled.append((bus.mode(3)[0], bus._read_u8(3, 40)))
                        return load
                    return read(servo, address)
                def firmware(servo, address, values):
                    write(servo, address, values)
                    if servo == 3 and address == GOAL_POSITION and values[:2] == [0, 16]:
                        bus.raw[2] = 21
                def feedback(servo):
                    if (servo == 3 and bus.mode(3)[0] == "wheel" and bus._read_u8(3, 40)
                            and bool(load & 1024) == invert
                            and bus._read_u16(3, RUNNING_TIME) & 1023 <= load & 1023):
                        bus.raw[2] = 24
                        return 24
                    return position(servo)
                clock, held = [0.0], [False]
                def wait(seconds):
                    clock[0] += seconds
                    held[0] = arm.holding
                    self.assertLess(clock[0], 3.0)
                    return held[0]
                with patch.object(bus, "_read_u16", side_effect=read_load), \
                     patch.object(bus, "_write", side_effect=firmware), \
                     patch.object(bus, "read_position", side_effect=feedback), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    arm.execute(("pose", "立正"))
                self.assertEqual(sampled, [("position", 1)])
                self.assertTrue(held[0])
                self.assertEqual(bus.raw, target)
                drives = [(values[0] << 8) | values[1] for servo, address, values in bus.writes
                          if servo == 3 and address == RUNNING_TIME and values != [0, 0]]
                self.assertGreaterEqual(len(drives), 2)
                self.assertEqual(set(drives), {expected + (1024 if invert else 0)})
                self.assertTrue(all(bus._read_u8(servo, 40) == 0 for servo in range(1, 7)))

    def test_standing_elbow_assists_loaded_seam_without_extending_stall_deadline(self):
        for invert, initial, required, resumed, stalled in (
                (False, 285, 500, {500}, {500}),
                (True, 285, 500, {500}, {500}),
                (False, 485, 500, {500}, {500}),
                (False, 630, 630, {630}, {630})):
            faults = (None, "stall", "voltage", "cancel", "missing", "slow_read") if initial == 285 else (None, "stall")
            for fault in faults:
                with self.subTest(invert=invert, initial=initial, fault=fault):
                    config = load_advanced()
                    raw = [1022, 120, 330, 986, 100, 280]
                    target = raw[:2] + [874] + raw[3:]
                    config["poses"]["立正"] = target
                    bus = FeedbackBus(load_config("config.json"), raw)
                    bus.items[3]["invert_direction"] = invert
                    bus.wheel_positions[3] = [12, 4, 0, 1022, 1017]
                    arm = Arm(config, bus)
                    read, position, write = bus._read_u16, bus.read_position, bus._write
                    clock, held, stopped_at, stalled_at = [0.0], [False], [], []
                    def load(servo, address):
                        if servo != 3 or address != 60:
                            return read(servo, address)
                        return initial - 30 + (1024 if invert else 0)
                    def firmware(servo, address, values):
                        write(servo, address, values)
                        if servo == 3 and address == GOAL_POSITION and values[:2] == [0, 16]:
                            bus.raw[2] = 20
                    def feedback(servo):
                        if (servo == 3 and bus.mode(3)[0] == "wheel" and bus._read_u8(3, 40)
                                and read(3, RUNNING_TIME) and not bus.wheel_positions[3]):
                            if fault in ("voltage", "cancel", "missing", "slow_read"):
                                stopped_at.append(len(bus.writes))
                            if fault == "voltage":
                                raise RuntimeError("ID 3 返回错误码 0x01")
                            if fault == "cancel":
                                arm.cancel.set()
                            if fault == "missing":
                                raise RuntimeError("elbow feedback missing")
                            if fault == "slow_read":
                                clock[0] += 0.9
                            if fault is None and read(3, RUNNING_TIME) & 1023 >= required:
                                bus.raw[2] -= 8
                            return bus.raw[2]
                        value = position(servo)
                        if servo == 3 and value == 1017 and not stalled_at:
                            stalled_at.append(clock[0])
                        return value
                    def wait(seconds):
                        clock[0] += seconds
                        held[0] = arm.holding
                        self.assertLess(clock[0], 3.0)
                        return held[0]
                    with patch.object(bus, "_read_u16", side_effect=load), \
                         patch.object(bus, "_write", side_effect=firmware), \
                         patch.object(bus, "read_position", side_effect=feedback), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if fault:
                            message = {"voltage": "0x01", "cancel": "动作已停止", "missing": "elbow feedback missing", "slow_read": "反馈间隔过长"}.get(fault, "未取得有效进展")
                            with self.assertRaisesRegex((RuntimeError, TimeoutError, InterruptedError), message):
                                arm.execute(("pose", "立正"))
                        else:
                            arm.execute(("pose", "立正"))
                    drives = [(values[0] << 8) | values[1] for servo, address, values in bus.writes
                              if servo == 3 and address == RUNNING_TIME and values != [0, 0]]
                    self.assertTrue(all(bool(drive & 1024) == invert for drive in drives))
                    magnitudes = {drive & 1023 for drive in drives}
                    if fault is None:
                        self.assertEqual(bus.raw, target)
                        self.assertEqual(magnitudes, resumed)
                    elif fault == "stall":
                        self.assertEqual(magnitudes, stalled)
                    else:
                        self.assertEqual(magnitudes, {max(500, initial)})
                    if fault == "stall":
                        self.assertLessEqual(clock[0] - stalled_at[0], 0.83)
                    if stopped_at:
                        self.assertFalse(any(address == RUNNING_TIME and values != [0, 0]
                                             for _, address, values in bus.writes[stopped_at[0]:]))
                    self.assertEqual(held[0], fault is None)
                    self.assertTrue(all(bus.mode(servo)[0] == "position" for servo in range(1, 7)))
                    self.assertTrue(all(bus._read_u8(servo, 40) == 0 for servo in range(1, 7)))
                    self.assertIsNone(bus.running_id)

    def test_standing_load_feedback_fault_or_cancel_prevents_mode_switch(self):
        for fault in (1001, 2048, "missing", "cancel"):
            with self.subTest(fault=fault):
                config = load_advanced()
                raw = [1022, 900, 330, 700, 100, 280]
                target = list(raw)
                target[2] = 874
                config["poses"]["立正"] = target
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                read = bus._read_u16
                stopped = []
                def feedback(servo, address):
                    if servo == 3 and address == 60:
                        stopped.append(len(bus.writes))
                        if fault == "missing":
                            raise RuntimeError("load feedback missing")
                        if fault == "cancel":
                            arm.cancel.set()
                            return 330
                        return fault
                    return read(servo, address)
                with patch.object(bus, "_read_u16", side_effect=feedback), \
                     patch.object(arm.cancel, "wait", return_value=False):
                    with self.assertRaises((RuntimeError, InterruptedError)):
                        arm.execute(("pose", "立正"))
                self.assertEqual(len(stopped), 1)
                self.assertTrue(all(address == 40 and values == [0]
                                    for _, address, values in bus.writes[stopped[0]:]))
                self.assertFalse(any(address in (9, 11, 48) for _, address, _ in bus.writes))
                self.assertTrue(all(bus._read_u8(servo, 40) == 0 for servo in range(1, 7)))
                self.assertIsNone(bus.running_id)

    def test_standing_stalled_position_approach_never_enters_wheel_mode(self):
        config = load_advanced()
        raw = [1022, 900, 330, 700, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        bus.stall = True
        arm = Arm(config, bus)
        clock = [0.0]
        def wait(seconds):
            clock[0] += seconds
            self.assertLess(clock[0], 5.0)
            return False
        with patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, "ID 2 未在期限内到位：目标 1007，反馈 900"):
                arm.execute(("pose", "立正"))
        self.assertFalse(any(address in (9, 11, 48) for _, address, _ in bus.writes))
        self.assertTrue(all(address == GOAL_POSITION and set(values) == set(range(2, 7))
                            for address, values in bus.batches))
        self.assertIsNone(bus.running_id)
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))

    def test_non_crossing_pose_does_not_release_or_switch_verified_wheel_joint(self):
        config = load_advanced()
        raw = [980, 899, 330, 680, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        target = list(raw)
        target[1] = 919
        arm.move_pose(target, duration_s=0.2, quiet=True)
        self.assertEqual(bus.raw, target)
        self.assertFalse(any(address in (9, 11, 48) or (joint != 1 and address == 40 and values == [0])
                             for joint, address, values in bus.writes))

    def test_named_pose_rejects_drift_of_previously_arrived_joint(self):
        config = load_advanced()
        config.pop("pose_paths", None)
        raw = [980, 899, 330, 680, 100, 280]
        target = list(raw)
        target[5] = 290
        config["poses"]["姿态2"] = target
        bus = FeedbackBus(load_config("config.json"), raw)
        original_write = bus._write

        def write(joint, address, values):
            original_write(joint, address, values)
            if joint == 6 and address == GOAL_POSITION and bus.raw[5] == 290:
                bus.raw[1] = 890

        bus._write = write
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(TimeoutError, "ID 2 目标 899，反馈 890"):
            Arm(config, bus).execute(("pose", "姿态2"))
        self.assertNotIn("到达姿态", output.getvalue())

    def test_pose_two_accepts_three_count_error_but_rejects_eight(self):
        for error in (3, 8):
            with self.subTest(error=error):
                config = load_advanced()
                target = config["poses"]["姿态2"]
                raw = list(target)
                raw[1] += 12
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                write = bus._write
                clock = [0.0]
                def offset(joint, address, values):
                    write(joint, address, values)
                    if joint == 2 and address == GOAL_POSITION:
                        bus.raw[1] = target[1] + error
                def wait(seconds):
                    clock[0] += seconds
                    return False
                with patch.object(bus, "_write", side_effect=offset), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if error == 3:
                        arm.execute(("pose", "姿态2"))
                        self.assertEqual(bus.raw[1], target[1] + 3)
                    else:
                        with self.assertRaises(PositionHoldTimeout):
                            arm.execute(("pose", "姿态2"))

    def test_standing_base_is_not_driven_when_arrived(self):
        for start in (1022, 950):
            with self.subTest(start=start):
                config = load_advanced()
                raw = [start, 899, 310, 698, 100, 280]
                config["poses"]["立正"] = [1018, *raw[1:]]
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                clock, held = [0.0], []
                def wait(seconds):
                    clock[0] += seconds
                    if arm.holding:
                        held.append(bus._read_u8(1, 40))
                        if len(held) >= 8:
                            arm.cancel.set()
                            return True
                    self.assertLess(clock[0], 3)
                    return False
                with patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    arm.execute(("pose", "立正"))
                self.assertEqual(held, [0] * 8)
                if start == 1022:
                    self.assertFalse(any(j == 1 and a == 40 and v == [1] for j, a, v in bus.writes))
                self.assertFalse(any(1 in values for _, values in bus.batches[1:]))

    def test_standing_base_path_escape_stops_before_resending(self):
        config = load_advanced()
        raw = [950, 899, 310, 698, 100, 280]
        config["poses"]["立正"] = [1018, 120, 874, 986, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        read = bus.read_position
        def feedback(joint):
            if bus.batches and joint == 1:
                return 940
            return read(joint)
        with patch.object(bus, "read_position", side_effect=feedback), \
             patch.object(arm.cancel, "wait", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "ID 1 底座反馈偏离本次运动范围"):
                arm.execute(("pose", "立正"))
        self.assertEqual(len(bus.batches), 1)
        self.assertTrue(all(bus._read_u8(j, 40) == 0 for j in range(1, 7)))

    def test_fixed_pose_base_is_not_driven_when_arrived(self):
        for name in ("姿态2", "姿态3"):
            with self.subTest(name=name):
                config = load_advanced()
                target = config["poses"][name]
                raw = [1022, *target[1:]]
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                with patch.object(arm.cancel, "wait", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    arm.execute(("pose", name))
                self.assertEqual(bus.raw[0], raw[0])
                self.assertFalse(any(j == 1 and (a == GOAL_POSITION or a == 40 and v == [1])
                                     for j, a, v in bus.writes))
                self.assertEqual(bus._read_u8(1, 40), 0)

    def test_fixed_pose_base_path_escape_stops_before_next_joint(self):
        for displaced in (940, 0):
            with self.subTest(displaced=displaced):
                config = load_advanced()
                bus = FeedbackBus(load_config("config.json"), [950, 920, 300, 740, 80, 276])
                arm = Arm(config, bus)
                read = bus.read_position
                def feedback(joint):
                    if joint == 1 and bus.running_id == 1:
                        return displaced
                    return read(joint)
                with patch.object(bus, "read_position", side_effect=feedback), \
                     patch.object(bus, "move_position", wraps=bus.move_position) as move, \
                     patch.object(arm.cancel, "wait", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "ID 1 底座反馈偏离本次运动范围"):
                        arm.move_pose(config["poses"]["姿态2"], quiet=True)
                self.assertEqual([call.args[0] for call in move.call_args_list], [1])
                self.assertTrue(all(bus._read_u8(j, 40) == 0 for j in range(1, 7)))

    def test_fixed_pose_base_settles_with_torque_off_after_first_arrival(self):
        config = load_advanced()
        raw = [950, 920, 300, 740, 80, 276]
        target = [1019, *raw[1:]]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        read = bus.read_position
        samples = []
        def feedback(joint):
            if joint == 1 and bus.running_id == 1:
                samples.append(bus._read_u8(1, 40))
                bus.raw[0] = 1015
            return read(joint)
        with patch.object(bus, "read_position", side_effect=feedback), \
             patch.object(arm.cancel, "wait", return_value=False):
            arm.move_pose(target, quiet=True, skip_arrived_base=True)
        self.assertEqual(samples, [1, 0, 0])
        self.assertEqual(bus._read_u8(1, 40), 0)

    def test_standing_stops_the_group_on_voltage_alarm_without_resending(self):
        config = load_advanced()
        raw = [980, 880, 330, 680, 100, 280]
        config["poses"]["立正"] = [998, 899, 310, 698, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        read = bus.read_position
        failed_at = [None]
        def feedback(joint):
            if bus.running_id == "all" and bus.batches:
                failed_at[0] = len(bus.writes)
                raise RuntimeError("ID 2 返回错误码 0x01：电压异常")
            return read(joint)
        with patch.object(bus, "read_position", side_effect=feedback), \
             patch.object(arm.cancel, "wait", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "电压异常"):
                arm.execute(("pose", "立正"))
        self.assertEqual(len(bus.batches), 1)
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
        self.assertFalse(any(address == GOAL_POSITION or (address == 40 and values == [1])
                             for _, address, values in bus.writes[failed_at[0]:]))
        self.assertFalse(arm.holding)
        self.assertIsNone(bus.running_id)

    def test_standing_cancel_during_crossing_restores_modes_without_new_goals(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["pose_paths"]["立正"][0])
        arm = Arm(config, bus)
        stopped_at = [None]
        def wait(seconds):
            if bus.running_id == "all" and any(bus.mode(joint)[0] == "wheel" for joint in (2, 3)):
                stopped_at[0] = len(bus.writes)
                arm.cancel.set()
                return True
            return False
        with patch.object(arm.cancel, "wait", side_effect=wait):
            arm.execute(("pose", "立正"))
        self.assertEqual(len(bus.batches), 1)
        self.assertTrue(all(bus.mode(joint)[0] == "position" for joint in range(1, 7)))
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
        self.assertIsNone(bus.running_id)
        self.assertFalse(any(address == GOAL_POSITION or (address == 40 and values == [1])
                             for _, address, values in bus.writes[stopped_at[0]:]))

    def test_standing_cancel_during_start_feedback_never_sends_the_group(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["pose_paths"]["立正"][0])
        arm = Arm(config, bus)
        read = bus.read_position
        samples = [0]
        def feedback(joint):
            if joint == 6 and bus.running_id == "all":
                samples[0] += 1
                if samples[0] == 2:
                    arm.cancel.set()
            return read(joint)
        with patch.object(bus, "read_position", side_effect=feedback), \
             patch.object(arm.cancel, "wait", return_value=False):
            with self.assertRaises(InterruptedError):
                arm.execute(("pose", "立正"))
        self.assertEqual(bus.batches, [])
        self.assertIsNone(bus.running_id)
        self.assertTrue(all(bus.mode(joint)[0] == "position" for joint in range(1, 7)))
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))

    def test_group_watchdog_releases_every_joint_on_heartbeat_loss(self):
        bus = FeedbackBus(load_config("config.json"), [512] * 6)
        bus.running_id = "all"
        for joint in range(1, 7):
            bus.registers[joint, 40] = 1
        bus._watchdog_exit = MagicMock()
        bus._watchdog_exit.wait.side_effect = [False, True]
        with patch("time.monotonic", return_value=1.0), contextlib.redirect_stdout(io.StringIO()):
            bus._watchdog_loop()
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
        self.assertIsNone(bus.running_id)

    def test_next_voice_command_replaces_standing_hold(self):
        config = load_advanced()
        raw = [980, 880, 330, 680, 100, 280]
        config["poses"]["立正"] = [998, 880, 330, 698, 100, 280]
        config["poses"]["姿态2"] = raw
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        runner = CommandRunner(arm)
        self.addCleanup(runner.close)
        held = threading.Event()
        wait = arm.cancel.wait
        def capture(seconds):
            if arm.holding:
                held.set()
            return wait(seconds)
        with patch.object(arm.cancel, "wait", side_effect=capture), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(runner.submit(("pose", "立正")))
            self.assertTrue(held.wait(2))
            transition = len(bus.writes)
            self.assertTrue(runner.submit(("pose", "姿态2")))
            runner.wait()
        self.assertFalse(any(joint in range(2, 7) and address == 40 and values == [0]
                             for joint, address, values in bus.writes[transition:]))
        self.assertEqual(bus.raw, raw)
        self.assertFalse(arm.holding)
        self.assertFalse(runner.busy)

    def test_stop_during_standing_handoff_cancels_next_pose(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        runner = CommandRunner(arm)
        arm.holding = True
        bus.running_id = "all"
        for joint in range(1, 7):
            bus.set_torque(joint, True)
        runner.thread = MagicMock()
        runner.thread.is_alive.return_value = True
        def stop_on_join(**kwargs):
            runner.stop()
            runner.thread.is_alive.return_value = False
        runner.thread.join.side_effect = stop_on_join
        with patch.object(arm, "execute") as execute:
            self.assertFalse(runner.submit(("pose", "初始状态")))
        execute.assert_not_called()
        self.assertTrue(arm.cancel.is_set())
        self.assertFalse(arm.handoff)
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))

    def test_mapped_pose_commands_small_target_offsets(self):
        config = load_advanced()
        raw = [980, 880, 330, 680, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        target = list(raw)
        target[0] += 8
        target[1] -= 2
        with patch.object(bus, "move_position", wraps=bus.move_position) as move:
            arm.move_pose(target, duration_s=0.2, quiet=True)
        self.assertEqual([call.args[0] for call in move.call_args_list], [1, 2])
        self.assertEqual(bus.raw, target)

    def test_mapped_pose_leaves_rest_drift_without_relaxing_target_limits(self):
        config = load_advanced()
        raw = [1022, 899, 315, 657, 85, 267]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        with self.assertRaises(ValueError):
            arm.move_pose(raw, quiet=True)
        self.assertEqual(bus.writes, [])
        target = list(raw)
        target[3] = 677
        arm.move_pose(target, duration_s=0.2, quiet=True)
        self.assertEqual(bus.raw, target)

    def test_holding_mode_changes_refuse_another_active_joint(self):
        bus = FeedbackBus(load_config("config.json"), [512] * 6)
        bus.set_mode(2, "wheel", hold_others=True)
        bus.run(2, "left", 180, hold_others=True)
        bus.writes.clear()
        with self.assertRaisesRegex(ValueError, "上一关节"):
            bus.set_mode(3, "wheel", hold_others=True)
        with self.assertRaisesRegex(ValueError, "一个舵机"):
            bus.set_mode("all", "wheel", hold_others=True)
        self.assertEqual(bus.writes, [])

    def test_mapped_pose_accepts_measured_six_count_servo_deadband(self):
        config = load_advanced()
        raw = [980, 880, 330, 680, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        original_write = bus._write

        def write(joint, address, values):
            before = bus.raw[joint - 1]
            original_write(joint, address, values)
            if address == GOAL_POSITION:
                target = bus.raw[joint - 1]
                bus.raw[joint - 1] = before if abs(target - before) <= 6 else target + (6 if before > target else -6)

        bus._write = write
        arm = Arm(config, bus)
        target = list(raw)
        target[0] -= 8
        arm.move_pose(target, duration_s=0.2, quiet=True)
        self.assertEqual(bus.raw[0], target[0] + 6)
        self.assertIsNone(bus.running_id)

    def test_fixed_joint_follow_moves_holds_and_returns_with_real_mapping(self):
        config = load_advanced()
        raw = [980, 880, 330, 680, 100, 280]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        frames = 0
        samples = {}

        def read():
            nonlocal frames
            frames += 1
            if frames in (70, 100, 180):
                samples[frames] = list(bus.raw)
            if frames > 180:
                return False, None
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            center = (370, 240) if 10 <= frames < 100 else (320, 240)
            cv2.circle(frame, center, 35, (0, 0, 0), -1)
            return True, frame

        camera = MagicMock()
        camera.read.side_effect = read
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track_joints(camera, arm, config["vision"], [1], headless=True, counts_per_pixel=0.12)
        self.assertEqual(samples[70][0], 986)
        self.assertEqual(samples[100], samples[70])
        self.assertEqual(samples[180], raw)
        self.assertTrue(arm.cancel.is_set())
        self.assertIsNone(bus.running_id)

    def test_action_preflight_prevents_partial_missing_pose(self):
        del self.config["poses"]["向右转"]
        with patch.object(self.arm, "move_pose") as move:
            with self.assertRaisesRegex(ValueError, "向右转"):
                self.arm.execute(("action", "挥手"))
            move.assert_not_called()

    def test_action_forwards_timing_speed_and_order(self):
        self.config["actions"]["挥手"] = [
            {"pose": "抬起", "duration_s": 0.7, "speed": 80},
            {"pose": "立正", "duration_s": 1.5, "speed": 100}]
        with patch.object(self.arm, "move_pose") as move:
            self.arm.execute(("action", "挥手"))
        self.assertEqual(move.call_args_list[0].args, (self.config["poses"]["抬起"], 0.7, 80))
        self.assertEqual(move.call_args_list[1].args, (self.config["poses"]["立正"], 1.5, 100))

    def test_stop_interrupts_action_and_no_queue(self):
        self.config["actions"]["挥手"][0]["pause_s"] = 10
        moved = threading.Event()
        runner = CommandRunner(self.arm)
        with patch.object(self.arm, "move_pose", side_effect=lambda *a: moved.set()) as move:
            runner.submit(("action", "挥手"))
            self.assertTrue(moved.wait(1))
            self.assertFalse(runner.submit(("pose", "立正")))
            runner.submit(("stop", None))
            with self.assertRaises(InterruptedError):
                runner.wait()
            self.assertEqual(move.call_count, 1)
        runner.close()

    def test_real_motion_requires_calibration(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        arm = Arm(self.config, bus)
        with self.assertRaisesRegex(ValueError, "motion_calibrated"):
            arm.move_pose(self.config["poses"]["向左转"])
        self.assertEqual(bus.writes, [])

    def test_position_mode_no_automatic_eeprom_changes(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        bus.wheel = True
        self.config["motion_calibrated"] = True
        arm = Arm(self.config, bus)
        with self.assertRaises(ValueError):
            arm.move_pose(self.config["poses"]["向左转"])
        self.assertEqual(bus.writes, [])

    def test_goal_register_endian_time_speed_and_single_motion(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        bus.move_position(1, 600, 1000, 100)
        self.assertEqual(bus.writes[0], (1, GOAL_POSITION, [2, 88, 3, 232, 0, 100]))
        with self.assertRaises(ValueError):
            bus.move_position(2, 700, 1000, 100)
        bus.finish_position(1)
        bus.move_position(2, 700, 1000, 100)
        self.assertEqual(bus.running_id, 2)
        self.assertTrue(all(address not in (9, 11, 48) for _, address, _ in bus.writes))

    def test_write_fault_stops_joint(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        bus.fail = True
        with self.assertRaises(RuntimeError):
            bus.move_position(1, 600, 1000, 100)
        self.assertIsNone(bus.running_id)
        self.assertNotIn((1, RUNNING_TIME, [0, 0]), bus.writes)
        self.assertIn((1, 40, [0]), bus.writes)

    def test_position_feedback_confirms_completion(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        self.config["motion_calibrated"] = True
        arm = Arm(self.config, bus)
        with contextlib.redirect_stdout(io.StringIO()):
            arm.move_pose(self.config["poses"]["向左转"], duration_s=0.1)
        self.assertIsNone(bus.running_id)
        np.testing.assert_equal(arm.positions(), self.config["poses"]["向左转"])

    def test_stalled_servo_aborts_later_joints(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        bus.stall = True
        self.config["motion_calibrated"] = True
        arm = Arm(self.config, bus)
        runner = CommandRunner(arm)
        target = self.arm.positions().tolist()
        target[0] += 2
        target[1] += 10
        self.config["poses"]["向左转"] = target
        runner.submit(("pose", "向左转"))
        with self.assertRaises(PositionHoldTimeout):
            runner.wait()
        self.assertTrue(arm.cancel.is_set())
        self.assertIsNone(bus.running_id)
        shoulder_goals = [values for i, address, values in bus.writes if i == 2 and address == GOAL_POSITION]
        self.assertEqual(len(shoulder_goals), 1)
        self.assertEqual((shoulder_goals[0][0] << 8) | shoulder_goals[0][1], bus.raw[1])
        joint_goals = [values for i, address, values in bus.writes if i == 1 and address == GOAL_POSITION]
        self.assertEqual(len(joint_goals), 1)
        self.assertEqual((joint_goals[0][0] << 8) | joint_goals[0][1], target[0])
        self.assertFalse(any(j != 1 and address == 40 and values == [0] for j, address, values in bus.writes))
        self.assertEqual(bus._read_u8(1, 40), 0)
        runner.close()
        self.assertTrue(all((i, 40, [0]) in bus.writes for i in range(1, 7)))

    def test_timeout_hold_write_failure_still_stops_all_joints(self):
        bus = FeedbackBus(load_config("config.json"), self.arm.positions())
        bus.stall = True
        self.config["motion_calibrated"] = True
        target = self.arm.positions().tolist()
        target[1] += 10
        self.config["poses"]["向左转"] = target
        original_write = bus._write
        goal_count = 0

        def write(joint, address, values):
            nonlocal goal_count
            if joint == 2 and address == GOAL_POSITION:
                goal_count += 1
                if goal_count == 3:
                    raise RuntimeError("hold write failed")
            return original_write(joint, address, values)

        bus._write = write
        runner = CommandRunner(Arm(self.config, bus))
        runner.submit(("pose", "向左转"))
        with self.assertRaisesRegex(RuntimeError, "hold write failed"):
            runner.wait()
        self.assertIsNone(bus.running_id)
        self.assertTrue(all((i, 40, [0]) in bus.writes for i in range(1, 7)))
        runner.close()


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.arm = Arm(load_advanced())
        self.config = copy.deepcopy(self.arm.config["vision"])
        self.config["image_jacobian"] = [[-3, 0.4], [0.2, -3]]

    def test_web_start_needs_no_manual_alignment_and_stop_cancels_queued_motion(self):
        web = VisionHTTPServer(self.arm, "unused.json")
        web.runner = MagicMock(busy=False)
        try:
            web.request("start")
            self.assertTrue(web.state["busy"])
            web.request("stop")
            self.assertFalse(web.submit(("home", "初始状态")))
            web.runner.submit.assert_not_called()
            self.assertTrue(self.arm.cancel.is_set())
            self.assertFalse(web.driving)
        finally:
            web.server_close()

    def test_web_heartbeat_loss_stops_even_after_motion_finished(self):
        web = VisionHTTPServer(self.arm, "unused.json")
        try:
            web.driving = True
            web.heartbeat = time.monotonic() - 3
            web.state["busy"] = False
            with patch.object(self.arm, "stop", wraps=self.arm.stop) as stop:
                web.service_actions()
                stop.assert_called_once()
            self.assertEqual(web.pending, ord("p"))
            self.assertFalse(web.driving)
        finally:
            web.server_close()

    def test_web_buttons_http_authentication_and_stop(self):
        web = VisionHTTPServer(self.arm, "unused.json")
        thread = threading.Thread(target=web.serve_forever, daemon=True)
        thread.start()
        try:
            base, query = web.url.split("?", 1)
            with self.assertRaises(urllib.error.HTTPError) as denied:
                urllib.request.urlopen(base + "api/command", data=b'{"action":"home"}', timeout=2)
            self.assertEqual(denied.exception.code, 403)
            self.assertIsNone(web.pending)
            for action in ["start", "stop"]:
                response = urllib.request.urlopen(base + "api/command?" + query,
                                                  data=json.dumps({"action": action}).encode(), timeout=2)
                self.assertTrue(json.load(response)["ok"])
            self.assertTrue(self.arm.cancel.is_set())
            page = urllib.request.urlopen(web.url, timeout=2).read().decode("utf-8")
            self.assertIn('id="start"', page)
            self.assertNotIn('data-action="record"', page)
            self.assertIn('id="preview"', page)
        finally:
            web.shutdown()
            web.server_close()
            thread.join(timeout=2)

    def test_web_automatically_prepares_and_follows_without_manual_records(self):
        for base, dx in ((910, 80), (1022, -80)):
            config = load_advanced()
            config["vision"]["max_steps"] = 1
            arm = Arm(config)
            arm.config, arm.model = config, Kinematics(config)
            arm.sim_raw = [base, 902, 315, 700, 80, 276]
            hold = arm.hold_pose
            def hold_with_residual(target, **kwargs):
                before = arm.positions()[0]
                hold(target, **kwargs)
                if before > 1007 and target[0] <= 1007:
                    arm.sim_raw[0] += 2
            web = VisionHTTPServer(arm, "unused.json")
            count, reference = 0, None
            def read():
                nonlocal count, reference
                count += 1
                if count == 1:
                    web.request("start")
                if count == 20:
                    self.assertEqual(arm.positions()[3], 700)
                if count == 75:
                    reference = arm.positions().copy()
                    self.assertTrue(web.state["active"])
                    self.assertEqual(reference[3], 900)
                frame = np.full((480, 640, 3), 255, dtype=np.uint8)
                if count > 20 and not 130 <= count < 150:
                    fraction = np.clip((count - 80) / 20, 0, 1)
                    center = (round(320 + dx * fraction), round(240 - 80 * fraction))
                    cv2.circle(frame, center, 35, (0, 0, 0), -1)
                if count == 220:
                    web.closed.set()
                return True, frame
            camera = MagicMock()
            camera.read.side_effect = read
            try:
                with self.subTest(base=base), patch.object(arm, "hold_pose", side_effect=hold_with_residual), \
                     patch("vision_control.save_advanced") as save, \
                     contextlib.redirect_stdout(io.StringIO()):
                    track(camera, arm, config["vision"], fixed_camera=True, point_at_ball=True,
                          wait_start=True, headless=True, web=web)
                    save.assert_not_called()
                    expected, _ = pointing_target(arm.model, reference, [dx, -80], .25)
                    np.testing.assert_allclose(arm.positions(), expected, atol=4)
                    self.assertIsNotNone(web.frame)
            finally:
                web.server_close()

    def test_pointing_changes_gripper_axis_and_has_visible_travel(self):
        model = Kinematics(load_advanced())
        reference = np.array([910, 920, 300, 910, 80, 276])
        def direction(raw):
            transform = np.eye(4)
            for i, origin in enumerate(model.chain):
                transform = transform @ origin
                if i < 5:
                    c, s = np.cos(model.angles(raw)[i]), np.sin(model.angles(raw)[i])
                    transform[:3, :3] = transform[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            axis = transform[:3, 2]
            return np.degrees([np.arctan2(axis[1], axis[0]), np.arcsin(axis[2])])
        origin = direction(reference)
        for offset, expected in [([80, 0], [-20, 0]), ([-80, 0], [20, 0]),
                                 ([0, -80], [0, 20]), ([0, 80], [0, -20])]:
            with self.subTest(offset=offset):
                target, limited = pointing_target(model, reference, offset, 0.25)
                self.assertFalse(limited)
                np.testing.assert_allclose(direction(target) - origin, expected, atol=0.3)
                self.assertGreater(np.linalg.norm(model.forward(target) - model.forward(reference)), 50)
                np.testing.assert_array_equal(target[[1, 2, 4, 5]], reference[[1, 2, 4, 5]])
        np.testing.assert_array_equal(pointing_target(model, reference, [0, 0], 0.25)[0], reference)

    def test_pointing_clamps_before_feedback_seam_and_keeps_measured_limits(self):
        model = Kinematics(load_advanced())
        reference = np.array([950, 920, 300, 930, 80, 276])
        for offset in ([1000, 1000], [-1000, -1000]):
            target, limited = pointing_target(model, reference, offset, 0.25)
            self.assertTrue(limited)
            self.assertTrue(np.all(target[:4] >= 16) and np.all(target[:4] <= 1007))
            model.validate(target)
            np.testing.assert_array_equal(model.unwrap(target) - model.unwrap(reference), target - reference)

    def test_pointing_rejects_seam_reference_instead_of_clipping_to_opposite_direction(self):
        model = Kinematics(load_advanced())
        for base in (0, 1022):
            for offset in ([0, 5], [5, 0], [-5, 0]):
                with self.subTest(base=base, offset=offset), self.assertRaisesRegex(ValueError, "反馈端点"):
                    pointing_target(model, [base, 920, 300, 900, 80, 276], offset, 1)

    def test_pointing_tracks_image_offset_then_returns_to_aligned_pose(self):
        config = load_advanced()
        arm = Arm(config)
        arm.config, arm.model = config, Kinematics(config)
        arm.sim_raw = [910, 920, 300, 910, 80, 276]
        original = arm.positions()
        samples, count = {}, 0
        def read():
            nonlocal count
            count += 1
            if count in (85, 180):
                samples[count] = arm.positions()
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.circle(frame, (400, 160) if 10 <= count < 90 else (320, 240), 35, (0, 0, 0), -1)
            return (True, frame) if count <= 180 else (False, None)
        camera = MagicMock()
        camera.read.side_effect = read
        with patch.object(arm, "hold_pose", wraps=arm.hold_pose) as hold, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track(camera, arm, self.config, fixed_camera=True, point_at_ball=True, headless=True)
        expected, _ = pointing_target(arm.model, original, [80, -80], 0.25)
        np.testing.assert_allclose(samples[85], expected, atol=4)
        np.testing.assert_allclose(samples[180], original, atol=4)
        self.assertGreater(hold.call_count, 4)
        previous = original
        for call in hold.call_args_list:
            target = call.args[0]
            self.assertLessEqual(np.max(np.abs(target - previous)), 20)
            self.assertTrue(call.kwargs["cartesian"])
            previous = target

    def test_pointing_manual_alignment_requires_key_and_start_resets_reference(self):
        config = load_advanced()
        arm = Arm(config)
        arm.config, arm.model = config, Kinematics(config)
        arm.sim_raw = [910, 920, 300, 910, 80, 276]
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (400, 180), 35, (0, 0, 0), -1)
        camera = MagicMock()
        camera.read.side_effect = lambda: (time.sleep(0.025) or True, frame.copy())
        with patch.object(arm, "execute", wraps=arm.execute) as execute, patch("cv2.imshow"), \
             patch("cv2.setWindowTitle"), patch("cv2.getWindowProperty", return_value=1), \
             patch("cv2.waitKeyEx", side_effect=[-1] * 3 + [2621440] + [-1] * 20 + [ord("s")] + [-1] * 10 + [32]):
            track(camera, arm, self.config, fixed_camera=True, point_at_ball=True, wait_start=True)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(execute.call_args.args[0][0], "point")
        self.assertGreater(arm.positions()[3], 920)

    def test_pointing_rejects_large_joint_step_before_driving(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        target = arm.positions()
        target[3] += 21
        with patch.object(arm, "hold_pose") as hold:
            with self.assertRaisesRegex(ValueError, "单步转角"):
                arm.execute(("point", target))
            hold.assert_not_called()

    def test_pointing_home_tilts_forward_in_bounded_steps(self):
        config = load_advanced()
        arm = Arm(config)
        arm.config, arm.model = config, Kinematics(config)
        arm.sim_raw = list(config["poses"]["初始状态"])
        with patch.object(arm, "hold_pose", wraps=arm.hold_pose) as hold, contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("point_home", "初始状态"))
        previous = np.asarray(config["poses"]["初始状态"])
        for call in hold.call_args_list:
            target = call.args[0]
            self.assertLessEqual(np.max(np.abs(target - previous)), 20)
            np.testing.assert_array_equal(target[[0, 1, 2, 4, 5]], previous[[0, 1, 2, 4, 5]])
            previous = target
        self.assertGreater(hold.call_count, 1)
        transform = np.eye(4)
        for i, origin in enumerate(arm.model.chain):
            transform = transform @ origin
            if i < 5:
                c, s = np.cos(arm.model.angles(arm.positions())[i]), np.sin(arm.model.angles(arm.positions())[i])
                transform[:3, :3] = transform[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.assertLess(np.degrees(np.arcsin(transform[2, 2])), 55)
        self.assertGreater(np.linalg.norm(transform[:2, 2]), 0.6)
        self.assertEqual(config["poses"]["初始状态"][3], 740)

    def test_pointing_home_stops_without_sending_remaining_segments(self):
        config = load_advanced()
        arm = Arm(config)
        arm.config, arm.model = config, Kinematics(config)
        arm.sim_raw = list(config["poses"]["初始状态"])
        with patch.object(arm, "hold_pose", side_effect=lambda *a, **kw: arm.cancel.set()) as hold, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(InterruptedError):
                arm.execute(("point_home", "初始状态"))
        self.assertEqual(hold.call_count, 1)

    def test_pointing_home_rejects_current_base_outside_measured_range(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), [641, 902, 315, 660, 83, 272])
        arm = Arm(config, bus)
        with patch.object(arm, "move_pose") as move, patch.object(arm, "hold_pose") as hold:
            with self.assertRaises(ValueError):
                arm.execute(("point_home", "初始状态"))
            move.assert_not_called()
            hold.assert_not_called()

    def test_pointing_home_never_drives_base_back_to_saved_direction(self):
        config = load_advanced()
        for base in (0, 720, 1022):
            bus = FeedbackBus(load_config("config.json"), [base, 902, 315, 660, 83, 272])
            arm = Arm(config, bus)
            with self.subTest(base=base), patch.object(arm.cancel, "wait", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                arm.execute(("point_home", "初始状态"))
            self.assertEqual(arm.positions()[0], base)
            self.assertGreater(arm.positions()[3], 895)
            self.assertTrue(bus.batches)
            self.assertTrue(all(1 not in batch for batch in bus.batches))
            self.assertFalse(any(joint == 1 and address == 40 and values == [1]
                                 for joint, address, values in bus.writes))

    def test_pointing_timeout_releases_all_drives(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        for command in (("point", arm.positions()), ("point_home", "初始状态")):
            with self.subTest(command=command[0]), \
                 patch.object(arm, "execute", side_effect=PositionHoldTimeout("failed")), \
                 patch.object(bus, "stop", wraps=bus.stop) as stop, contextlib.redirect_stdout(io.StringIO()):
                runner = CommandRunner(arm)
                runner.submit(command)
                with self.assertRaises(PositionHoldTimeout):
                    runner.wait()
                stop.assert_called_once_with("all")
                self.assertTrue(all(bus._read_u8(i, 40) == 0 for i in range(1, 7)))

    def test_shaded_circle_with_highlight(self):
        frame = synthetic_frame(self.arm, self.arm.model.forward(self.arm.positions()), self.config)
        ball = detect_ball(frame, self.config)
        self.assertIsNotNone(ball)
        np.testing.assert_allclose(ball[0], [410, 175], atol=2)

    def test_missing_rectangle_and_ambiguous_targets(self):
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
        self.assertIsNone(detect_ball(blank, self.config))
        cv2.rectangle(blank, (30, 30), (250, 55), (0, 0, 0), -1)
        self.assertIsNone(detect_ball(blank, self.config))
        cv2.circle(blank, (200, 240), 35, (0, 0, 0), -1)
        cv2.circle(blank, (400, 240), 35, (0, 0, 0), -1)
        self.assertIsNone(detect_ball(blank, self.config))

    def test_partially_visible_balls_at_all_edges(self):
        for center in [(-20, 240), (660, 240), (320, -20), (320, 500), (30, 30)]:
            with self.subTest(center=center):
                frame = np.full((480, 640, 3), 255, dtype=np.uint8)
                cv2.circle(frame, center, 80, (15, 15, 15), -1)
                ball = detect_ball(frame, self.config)
                self.assertIsNotNone(ball)
                self.assertLess(np.linalg.norm(ball[0] - center), self.config["deadband_px"] / 2)
                self.assertAlmostEqual(ball[1], 80, delta=8)

    def test_clipped_rectangle_and_insufficient_arc_rejected(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.rectangle(frame, (0, 100), (120, 250), (0, 0, 0), -1)
        self.assertIsNone(detect_ball(frame, self.config))
        frame[:] = 255
        cv2.circle(frame, (-75, 240), 80, (0, 0, 0), -1)
        self.assertIsNone(detect_ball(frame, self.config))

    def test_rotated_camera_mapping_reduces_error(self):
        for matrix in [[[-3, 0.4], [0.2, -3]], [[0, 3], [-2, 0]], [[2, 1], [1, -2]]]:
            self.config["image_jacobian"] = matrix
            error = np.array([90, -65])
            delta = correction(error, self.config)
            predicted = error + np.array(matrix) @ delta[self.config["control_axes"]]
            self.assertLess(np.linalg.norm(predicted), np.linalg.norm(error))
            self.assertLessEqual(np.linalg.norm(delta), 3.000001)

    def test_deadband_and_bad_calibration(self):
        np.testing.assert_equal(correction([2, 3], self.config), [0, 0, 0])
        for matrix in [None, [[0, 0], [0, 0]], [[1, 1], [1, 1]], [[float("nan"), 0], [0, 1]]]:
            self.config["image_jacobian"] = matrix
            with self.assertRaises(ValueError):
                validate_jacobian(self.config)

    def test_closed_loop_converges_with_quantized_joints(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = track(None, self.arm, self.config, simulate=True, headless=True)
        self.assertTrue(result["converged"])
        self.assertGreater(result["steps"], 2)
        self.assertLess(result["final_error_px"], self.config["deadband_px"])
        self.assertLess(result["final_error_px"], result["initial_error_px"] / 5)

    def test_camera_loss_stops_active_motion(self):
        class LostCamera:
            def read(self):
                return False, None
        with patch.object(self.arm, "stop") as stop:
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track(LostCamera(), self.arm, self.config, headless=True)
            stop.assert_called_once()

    def test_fixed_camera_stationary_off_center_target_does_not_move(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (400, 180), 35, (0, 0, 0), -1)
        camera = MagicMock()
        camera.read.side_effect = [(True, frame.copy()) for _ in range(15)] + [(False, None)]
        self.config["image_jacobian"] = None
        with patch.object(self.arm, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track(camera, self.arm, self.config, fixed_camera=True, headless=True)
            execute.assert_not_called()
        self.assertTrue(self.arm.cancel.is_set())

    def test_fixed_camera_reference_uses_pose_when_ball_first_becomes_visible(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (400, 180), 35, (0, 0, 0), -1)
        count = 0
        def read():
            nonlocal count
            count += 1
            if count == 5:
                self.arm.sim_raw[1] += 8
            if count > 18:
                return False, None
            return True, np.full_like(frame, 255) if count < 5 else frame.copy()
        camera = MagicMock()
        camera.read.side_effect = read
        with patch.object(self.arm, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track(camera, self.arm, self.config, fixed_camera=True, headless=True)
            execute.assert_not_called()

    def test_fixed_camera_stationary_ball_does_not_trigger_drift_corrections(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (320, 240), 35, (0, 0, 0), -1)
        count = 0
        def read():
            nonlocal count
            count += 1
            if count == 8:
                self.arm.sim_raw[1] += 10
            return (True, frame.copy()) if count < 20 else (False, None)
        camera = MagicMock()
        camera.read.side_effect = read
        with patch.object(self.arm, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track(camera, self.arm, self.config, fixed_camera=True, headless=True)
            execute.assert_not_called()

    def test_fixed_camera_tracks_absolute_offset_then_returns_without_drift(self):
        origin = self.arm.model.forward(self.arm.positions())
        samples = {}
        count = 0

        def read():
            nonlocal count
            count += 1
            if count in (100, 120, 220):
                samples[count] = self.arm.model.forward(self.arm.positions()) - origin
            if count > 220:
                return False, None
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            center = (360, 210) if 10 <= count < 120 else (320, 240)
            cv2.circle(frame, center, 35, (0, 0, 0), -1)
            return True, frame

        camera = MagicMock()
        camera.read.side_effect = read
        self.config["image_jacobian"] = None
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track(camera, self.arm, self.config, fixed_camera=True, headless=True)
        self.assertLess(np.linalg.norm(samples[100] - [0, -8, 6]), 3)
        np.testing.assert_allclose(samples[120], samples[100], atol=0.01)
        self.assertLess(np.linalg.norm(samples[220]), 3)
        self.assertTrue(self.arm.cancel.is_set())

    def test_fixed_camera_execution_still_requires_kinematic_calibration(self):
        from vision_control import main
        config = load_advanced()
        config["kinematics"]["calibrated"] = False
        with patch("sys.argv", ["vision_control.py", "--fixed-camera", "--execute"]), \
             patch("vision_control.load_advanced", return_value=config), \
             patch("vision_control.open_arm") as open_arm, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()
        open_arm.assert_not_called()

    def test_fixed_camera_requires_full_circle(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (320, 15), 60, (0, 0, 0), -1)
        self.assertIsNotNone(detect_ball(frame, self.config))
        self.assertIsNone(detect_ball(frame, self.config, allow_clipped=False))

    def test_fixed_camera_waits_for_start_and_captures_new_origin(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (400, 180), 35, (0, 0, 0), -1)
        camera = MagicMock()
        camera.read.return_value = (True, frame)
        with patch.object(self.arm, "execute") as execute, \
             patch.object(self.arm, "cartesian_start_pose", wraps=self.arm.cartesian_start_pose) as start, \
             patch("cv2.imshow"), patch("cv2.setWindowTitle"), patch("cv2.getWindowProperty", return_value=1), \
             patch("cv2.waitKey", side_effect=[-1] * 4 + [ord("s")] + [-1] * 6 + [32]):
            track(camera, self.arm, self.config, fixed_camera=True, wait_start=True)
        start.assert_called_once()
        execute.assert_not_called()

    def test_fixed_camera_invalid_start_stays_open_without_driving(self):
        for key in (ord("s"), ord("h")):
            with self.subTest(key=key):
                config = load_advanced()
                bus = FeedbackBus(load_config("config.json"), [639, 900, 315, 659, 104, 278])
                arm = Arm(config, bus)
                camera = MagicMock()
                camera.read.return_value = (True, np.full((480, 640, 3), 255, dtype=np.uint8))
                with patch.object(arm, "execute") as execute, patch("cv2.imshow"), \
                     patch("cv2.setWindowTitle"), patch("cv2.getWindowProperty", return_value=1), \
                     patch("cv2.waitKey", side_effect=[key, -1, 32]), contextlib.redirect_stdout(io.StringIO()):
                    result = track(camera, arm, self.config, fixed_camera=True, wait_start=True)
                self.assertEqual(camera.read.call_count, 3)
                self.assertEqual(result["steps"], 0)
                execute.assert_not_called()
                self.assertFalse(any(address == GOAL_POSITION or address == 40 and values == [1]
                                     for _, address, values in bus.writes))

    def test_fixed_camera_vertical_drag_moves_tool_with_measured_mapping(self):
        config = load_advanced()
        arm = Arm(config)
        arm.config = config
        arm.model = Kinematics(config)
        arm.sim_raw = list(config["poses"]["初始状态"])
        original = arm.positions()
        origin = arm.model.forward(original)
        samples = {}
        count = 0

        def read():
            nonlocal count
            count += 1
            if count in (95, 190, 285):
                samples[count] = arm.positions()
            if count > 285:
                return False, None
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            y = 140 if 10 <= count < 100 else 340 if 100 <= count < 195 else 240
            cv2.circle(frame, (320, y), 35, (0, 0, 0), -1)
            return True, frame

        camera = MagicMock()
        camera.read.side_effect = read
        with contextlib.redirect_stdout(io.StringIO()), patch.object(arm, "move_linear", wraps=arm.move_linear) as move:
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                track(camera, arm, self.config, fixed_camera=True, headless=True)
        distances = [np.linalg.norm(call.args[0]) for call in move.call_args_list]
        self.assertGreater(max(distances), 3)
        self.assertLessEqual(max(distances), 10.000001)
        for sample, expected in [(95, 20), (190, -20), (285, 0)]:
            raw = samples[sample]
            np.testing.assert_allclose(arm.model.forward(raw) - origin, [0, 0, expected], atol=5)
            np.testing.assert_array_equal(raw[4:], original[4:])
        self.assertGreater(abs(int(samples[95][1]) - int(original[1])), 6)
        self.assertGreater(abs(int(samples[95][3]) - int(original[3])), 6)

    def test_fixed_camera_home_before_start_does_not_require_visible_ball(self):
        camera = MagicMock()
        camera.read.return_value = (True, np.full((480, 640, 3), 255, dtype=np.uint8))
        for pointing in (False, True):
            runner = MagicMock(error=None, busy=False)
            runner.submit.side_effect = lambda _: setattr(runner, "busy", True)
            with self.subTest(pointing=pointing), patch("vision_control.CommandRunner", return_value=runner), \
                 patch("cv2.imshow"), patch("cv2.getWindowProperty", return_value=1), \
                 patch("cv2.waitKeyEx" if pointing else "cv2.waitKey", side_effect=[ord("h"), -1, 32]):
                track(camera, self.arm, self.config, fixed_camera=True, wait_start=True, point_at_ball=pointing)
            runner.submit.assert_called_once_with(("point_home" if pointing else "home", "初始状态"))
            runner.stop.assert_not_called()
            runner.close.assert_called_once()

    def test_target_loss_does_not_command_a_step(self):
        class MissingTarget:
            frames = 0
            def read(self):
                self.frames += 1
                return (True, np.full((480, 640, 3), 255, np.uint8)) if self.frames <= 4 else (False, None)
        with patch.object(self.arm, "execute") as execute:
            with self.assertRaises(RuntimeError):
                track(MissingTarget(), self.arm, self.config, headless=True)
            execute.assert_not_called()


class SpeechTests(unittest.TestCase):
    def test_distance_candidates_recover_units_without_guessing_conflicts(self):
        cases = [(["向前 武林 啊", "向前 五厘米 啊"], "向前 五厘米 啊"),
                 (["向前 武林 啊", "向前 五厘米", "向前 五 毫米"], ""),
                 (["向前 五厘米", "向后 五厘米"], "向前 五厘米"),
                 (["不要向前五厘米", "向前五厘米"], "不要向前五厘米"),
                 (["向前五十厘米", "向前五厘米"], "向前五十厘米"),
                 (["向前五厘米", "向前五十厘米"], "向前五厘米"),
                 (["向前五十米", "向前五厘米"], "向前五十米"),
                 (["向前我的名字", "向前五厘米"], "向前我的名字"),
                 (["向前五厘米然后向后五厘米", "向前五厘米"], "向前五厘米然后向后五厘米"),
                 (["向前", "向前五厘米"], "向前")]
        for candidates, expected in cases:
            with self.subTest(candidates=candidates):
                result = json.dumps({"alternatives": [{"text": text} for text in candidates]}, ensure_ascii=False)
                self.assertEqual(result_text(result, load_advanced(), buttons=True), expected)

    def test_return_to_initial_requires_a_new_explicit_voice_command(self):
        runner = MagicMock()
        runner.arm.config = load_advanced()
        dispatch_result('{"text":"动作二"}', runner, buttons=True)
        runner.submit.assert_called_once_with(("pose", "姿态2"))
        for phrase in ["恢复", "复位", "初始位置", "回到初始位置", "恢复到初始位置", "恢复到初始状态", "恢复 初始 动作", "回到初始动作"]:
            runner.submit.reset_mock()
            dispatch_result(json.dumps({"text": phrase}, ensure_ascii=False), runner, buttons=True)
            runner.submit.assert_called_once_with(("pose", "初始状态"))
        runner.submit.reset_mock()
        for phrase in ["不要恢复", "恢复了吗", "恢复然后动作二", "不要恢复初始动作", "恢复初始动作然后动作二"]:
            dispatch_result(json.dumps({"text": phrase}, ensure_ascii=False), runner, buttons=True)
        runner.submit.assert_not_called()

    def test_button_speech_rejects_reported_garbled_transcripts_without_llm(self):
        from voice_control import parse_voice_command
        config = load_advanced()
        runner = MagicMock()
        runner.arm.config = config
        with patch("urllib.request.urlopen", side_effect=AssertionError("garbled speech must not reach LLM")):
            for text in ["向前 武林 啊", "向前 我 的 名字", "猝死 妆感", "法", "五厘米 啊",
                         "不要完成动作二", "动作二然后动作三", "动作四"]:
                with self.subTest(text=text):
                    dispatch_result(json.dumps({"text": text}, ensure_ascii=False), runner, buttons=True)
            runner.submit.assert_not_called()
            for text, expected in [("完成 动作 二", ("pose", "姿态2")),
                                   ("请完成第二个动作", ("pose", "姿态2")),
                                   ("向前五厘米啊", ("move", [50, 0, 0])),
                                   ("向前移动一下", ("move", [5, 0, 0])),
                                   ("停止啊", ("stop", None))]:
                with self.subTest(text=text):
                    self.assertEqual(parse_voice_command(text, config, buttons=True), expected)

    def test_voice_buttons_use_same_pose_targets_and_finite_direction_moves(self):
        from voice_control import parse_voice_command
        config = load_advanced()
        with patch("urllib.request.urlopen") as request:
            for text, expected in [("进行动作一", ("pose", "立正")), ("进行动作二", ("pose", "姿态2")),
                                   ("进行动作三", ("pose", "姿态3")), ("进入工作姿态", ("pose", "初始状态")),
                                   ("回到工作姿态", ("pose", "初始状态")), ("向上移动五厘米", ("move", [0, 0, 50])),
                                   ("停止", ("stop", None))]:
                with self.subTest(text=text):
                    self.assertEqual(parse_voice_command(text, config, buttons=True), expected)
            for direction, axis, sign in [("前", 0, 1), ("后", 0, -1), ("左", 1, 1),
                                          ("右", 1, -1), ("上", 2, 1), ("下", 2, -1)]:
                expected = [0, 0, 0]
                expected[axis] = config["step_mm"] * sign
                self.assertEqual(parse_voice_command(f"向{direction}移动", config, buttons=True), ("move", expected))
            for text in ["不要执行动作二", "动作二或动作三", "动作二然后动作三", "动作四", "左转", "挥手"]:
                with self.subTest(text=text), self.assertRaises(ValueError):
                    parse_voice_command(text, config, buttons=True)
            request.assert_not_called()

    def test_voice_button_microphone_dispatches_finite_move_without_model_wait(self):
        runner = MagicMock()
        runner.arm.config = load_advanced()
        with patch("vosk.Model"), patch("voice_control.button_grammar", return_value='["[unk]"]'), \
             patch("vosk.KaldiRecognizer") as recognizer, \
             patch("sounddevice.query_devices", return_value={"default_samplerate": 16000}), \
             patch("sounddevice.RawInputStream") as stream, \
             patch("voice_control.ThreadPoolExecutor") as executor:
            recognizer.return_value.AcceptWaveform.side_effect = [True, True, True, KeyboardInterrupt]
            recognizer.return_value.Result.side_effect = [json.dumps({"text": text}, ensure_ascii=False)
                                                        for text in ["进入工作姿态", "向前移动", "停止"]]
            def capture(**kwargs):
                for _ in range(4):
                    kwargs["callback"](b'\0\0', 1, time.monotonic(), None)
                return stream.return_value
            stream.side_effect = capture
            with self.assertRaises(KeyboardInterrupt):
                listen(runner, "models/vosk-model-small-cn-0.22", buttons=True)
            executor.return_value.submit.assert_not_called()
        self.assertEqual([call.args[0] for call in runner.submit.call_args_list],
                         [("pose", "初始状态"), ("move", [5, 0, 0]), ("stop", None)])
        runner.stop.assert_called()

    def test_speech_candidates_preserve_intent_and_reject_number_ambiguity(self):
        cases = [(["进行重挫二", "进行动作二", "姿态二"], "进行动作二"),
                 (["进行重挫二", "进行动作二", "进行动作三"], ""),
                 (["进行动作2", "进行动作3"], ""),
                 (["进行重挫四", "进行动作四", "进行动作二"], ""),
                 (["进行重挫是", "进行动作是"], ""),
                 (["不要进行动作2", "进行动作2"], "不要进行动作2"),
                 (["动作2然后动作3", "动作2"], "动作2然后动作3"),
                 (["动作2是什么", "动作2"], "动作2是什么"),
                 (["他说进行动作2", "进行动作2"], "他说进行动作2"),
                 (["挥手", "进行动作2"], "挥手"),
                 (["进行挥手", "进行动作2"], "进行挥手"),
                 (["帮我往前挪一点", "进行动作2"], "帮我往前挪一点"),
                 (["进行一个挥手动作"], "进行一个挥手动作"),
                 (["停止", "进行动作2"], "停止")]
        with patch("urllib.request.urlopen") as request:
            for texts, expected in cases:
                with self.subTest(texts=texts):
                    result = json.dumps({"alternatives": [{"text": text} for text in texts]}, ensure_ascii=False)
                    self.assertEqual(result_text(result, load_advanced()), expected)
            request.assert_not_called()

    def test_numbered_commands_dispatch_saved_poses_without_language_model(self):
        config = load_advanced()
        config["poses"] = dict(reversed(list(config["poses"].items())))
        runner = MagicMock()
        runner.arm.config = config
        with patch("urllib.request.urlopen") as request:
            for text, name in [("动作1", "立正"), ("进行动作一", "立正"),
                               ("进行动作2", "姿态2"), ("进行 动作 二。", "姿态2"),
                               ("请执行动作两", "姿态2"), ("执行一下第二个动作", "姿态2"),
                               ("姿态二", "姿态2"), ("机械臂，请进行动作3！", "姿态3")]:
                with self.subTest(text=text):
                    runner.submit.reset_mock()
                    dispatch_result(json.dumps({"text": text}, ensure_ascii=False), runner)
                    runner.submit.assert_called_once_with(("pose", name))
            for text in ["动作0", "动作4", "姿态8", "不要执行动作2", "动作2或动作3",
                         "进行动作2然后动作3", "动作2是什么"]:
                with self.subTest(text=text):
                    runner.submit.reset_mock()
                    dispatch_result(json.dumps({"text": text}, ensure_ascii=False), runner)
                    runner.submit.assert_not_called()
            request.assert_not_called()

    def test_initial_pose_speech_uses_fast_home_without_language_model(self):
        for text in ["初始状态", "初始姿态", "回到初始状态", "回到初始姿态",
                     "回初始状态", "回初始姿态", "机械臂，请执行初始姿态"]:
            with self.subTest(text=text):
                bus = FeedbackBus(load_config("config.json"), [1022, 898, 311, 696, 80, 267])
                runner = CommandRunner(Arm(load_advanced(), bus))
                with patch("urllib.request.urlopen") as request, \
                     patch.object(bus, "move_position", wraps=bus.move_position) as move, \
                     patch.object(runner.arm.cancel, "wait", return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    dispatch_result(json.dumps({"text": text}, ensure_ascii=False), runner)
                    runner.wait()
                request.assert_not_called()
                self.assertTrue(runner.arm.cartesian_initialized)
                self.assertEqual(bus.raw, runner.arm.config["poses"]["初始状态"])
                self.assertTrue(move.called)
                self.assertTrue(all(call.args[3] == 180 for call in move.call_args_list))
        runner = MagicMock()
        runner.arm.config = load_advanced()
        with patch("urllib.request.urlopen") as request, contextlib.redirect_stdout(io.StringIO()):
            for text in ["不要回到初始姿态", "回到初始姿态然后向前移动"]:
                dispatch_result(json.dumps({"text": text}, ensure_ascii=False), runner)
            request.assert_not_called()
            runner.submit.assert_not_called()

    def test_microphone_literal_commands_execute_without_interpreter_wait(self):
        runner = MagicMock()
        runner.arm.config = load_advanced()
        with patch("vosk.Model"), patch("vosk.KaldiRecognizer") as recognizer, \
             patch("sounddevice.query_devices", return_value={"default_samplerate": 16000}), \
             patch("sounddevice.RawInputStream") as stream, \
             patch("voice_control.ThreadPoolExecutor") as executor, \
             contextlib.redirect_stdout(io.StringIO()):
            recognizer.return_value.AcceptWaveform.side_effect = [True, True, True, KeyboardInterrupt]
            recognizer.return_value.Result.side_effect = [json.dumps({"text": text}, ensure_ascii=False)
                                                        for text in ["回到初始姿态", "进行动作2", "停止"]]
            def capture(**kwargs):
                for _ in range(4):
                    kwargs["callback"](b'\0\0', 1, time.monotonic(), None)
                return stream.return_value
            stream.side_effect = capture
            with self.assertRaises(KeyboardInterrupt):
                listen(runner, "models/vosk-model-small-cn-0.22")
            executor.return_value.submit.assert_not_called()
        self.assertEqual([call.args[0] for call in runner.submit.call_args_list],
                         [("pose", "初始状态"), ("pose", "姿态2"), ("stop", None)])
        runner.stop.assert_called()

    def test_stop_discards_pending_language_model_result(self):
        runner = CommandRunner(Arm(load_advanced()))
        with patch("vosk.Model"), patch("vosk.KaldiRecognizer") as recognizer, \
             patch("sounddevice.query_devices", return_value={"default_samplerate": 16000}), \
             patch("sounddevice.RawInputStream") as stream, \
             patch("voice_control.ThreadPoolExecutor") as executor, \
             patch.object(runner, "submit") as submit, patch.object(runner, "stop") as stop:
            recognizer.return_value.AcceptWaveform.side_effect = [True, False, True, KeyboardInterrupt]
            recognizer.return_value.Result.side_effect = ['{"text":"帮我往前挪一点"}', '{"text":""}']
            recognizer.return_value.PartialResult.return_value = '{"partial":"停止"}'
            future = executor.return_value.submit.return_value
            future.done.side_effect = [False, True]
            future.cancel.return_value = False
            future.result.return_value = ("move", [5, 0, 0])

            def capture(**kwargs):
                for _ in range(4):
                    kwargs["callback"](b'\0\0', 1, time.monotonic(), None)
                return stream.return_value

            stream.side_effect = capture
            with self.assertRaises(KeyboardInterrupt):
                listen(runner, "models/vosk-model-small-cn-0.22")
            stop.assert_called()
            future.cancel.assert_called_once()
            submit.assert_not_called()
            executor.return_value.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_language_model_intents_and_validation(self):
        from http.client import RemoteDisconnected

        config = Arm(load_advanced()).config
        cases = [
            ("帮我往前挪一点", {"kind": "move", "direction": "前", "mm": 5}, ("move", [5, 0, 0])),
            ("往左边挪一厘米", {"kind": "move", "direction": "左", "mm": 10}, ("move", [0, 10, 0])),
            ("给大家打个招呼", {"kind": "action", "name": "挥手"}, ("action", "挥手")),
            ("回到站好的那个姿势", {"kind": "pose", "name": "立正"}, ("pose", "立正")),
            ("恢复最开始的姿态", {"kind": "pose", "name": "初始状态"}, ("pose", "初始状态")),
            ("别动了", {"kind": "stop"}, ("stop", None)),
            ("往前挪五米", {"kind": "move", "direction": "前", "mm": 5000}, None),
            ("动一下", {"kind": "move", "direction": "前", "mm": True}, None),
            ("翻个跟头", {"kind": "action", "name": "翻跟头"}, None),
            ("随便动", {"kind": [], "mm": 5}, None),
            ("挪过去", {"kind": "clarify", "message": "请说明方向"}, None),
        ]
        with patch.dict("os.environ", {"ARM_LLM_URL": "http://localhost:1234/v1/chat/completions",
                                       "ARM_LLM_MODEL": "test"}):
            for text, result, expected in cases:
                with self.subTest(text=text), patch("urllib.request.urlopen") as request:
                    request.return_value.__enter__.return_value.read.return_value = json.dumps(
                        {"choices": [{"message": {"content": json.dumps(result)}}]}).encode()
                    if expected is None:
                        with self.assertRaises(ValueError):
                            parse_command(text, config)
                    else:
                        self.assertEqual(parse_command(text, config), expected)
            for error in (TimeoutError, RemoteDisconnected):
                with self.subTest(error=error), patch("urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(ValueError):
                        parse_command("帮我往前挪一点", config)
            with patch("urllib.request.urlopen") as request:
                self.assertEqual(parse_command("停止", config), ("stop", None))
                request.assert_not_called()

    def test_real_model_and_synthesized_chinese_audio(self):
        arm = Arm(load_advanced())
        runner = CommandRunner(arm)
        expected = {"forward": ("jog", [5.0, 0.0, 0.0]), "pose": ("pose", "立正"),
                    "action": ("action", "挥手"), "stop": ("stop", None)}
        for name, command in expected.items():
            with self.subTest(sample=name), patch.object(runner, "submit") as submit:
                listen(runner, "models/vosk-model-small-cn-0.22", wav_path=f"verification/voice_{name}.wav")
                submit.assert_called_once_with(command)
        runner.close()

    def test_unrecognized_speech_never_executes(self):
        runner = CommandRunner(Arm(load_advanced()))
        with patch.object(runner, "submit") as submit:
            dispatch_result('{"text":"不要向前移动"}', runner)
            dispatch_result('{"text":""}', runner)
            submit.assert_not_called()


class WebPoseTests(unittest.TestCase):
    def test_cartesian_readiness_uses_actual_feedback_without_motion(self):
        self.arm.config["kinematics"]["calibrated"] = True
        self.bus.raw = [1022, 899, 320, 660, 77, 269]
        state = self.request("/api/state")["cartesian"]
        self.assertFalse(state["ready"])
        self.assertIn("ID 4", state["blocked_reason"])
        self.bus.raw = list(self.arm.config["poses"]["初始状态"])
        self.bus.raw[5] = 269
        state = self.request("/api/state")["cartesian"]
        self.assertTrue(state["ready"])
        self.assertEqual(state["blocked_reason"], "")
        self.assertEqual(self.bus.writes, [])

    def setUp(self):
        from controller import ControlHTTPServer, RequestHandler
        self.bus = FeedbackBus(load_config("config.json"), [980, 880, 330, 680, 100, 280])
        self.bus.port = "offline"
        self.arm = Arm(load_advanced(), self.bus)
        self.arm.config["kinematics"]["calibrated"] = False
        self.runner = CommandRunner(self.arm)
        self.server = ControlHTTPServer(("127.0.0.1", 0), RequestHandler, bind_and_activate=False)
        self.server.pose_runner = self.runner
        self.addCleanup(self.server.server_close)

    def request(self, path, body=None):
        from controller import RequestHandler
        handler = object.__new__(RequestHandler)
        handler.server, handler.bus, handler.path = self.server, self.bus, path
        handler._json_body = lambda: body
        handler._send_json = MagicMock()
        if body is None:
            handler.do_GET()
        else:
            handler.do_POST()
        return handler._send_json.call_args.args[0]

    def test_three_buttons_dispatch_saved_poses_without_modifying_targets(self):
        expected = copy.deepcopy(self.arm.config["poses"])
        with patch.object(self.runner, "submit", return_value=True) as submit:
            for name in ("立正", "姿态2", "姿态3"):
                self.assertTrue(self.request("/api/pose", {"name": name})["ok"])
                submit.assert_called_with(("pose", name))
                self.assertEqual(self.server.pose_name, name)
        self.assertEqual(self.arm.config["poses"], expected)
        self.assertEqual(self.bus.writes, [])

    def test_next_web_pose_ends_standing_hold_and_dispatches_the_new_target(self):
        self.arm.holding = True
        self.bus.running_id = "all"
        self.runner.thread = MagicMock()
        self.runner.thread.is_alive.return_value = True
        self.runner.thread.join.side_effect = lambda **kwargs: setattr(self.runner.thread.is_alive, "return_value", False)
        self.assertTrue(self.request("/api/state")["pose"]["holding"])
        with patch.object(self.runner, "submit", return_value=True) as submit:
            self.assertTrue(self.request("/api/pose", {"name": "姿态2"})["ok"])
            submit.assert_called_once_with(("pose", "姿态2"))
        self.assertFalse(self.arm.holding)
        self.assertIsNone(self.bus.running_id)
        self.assertTrue(all(self.bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))

    def test_pose_rejects_unknown_name_and_overlapping_controls(self):
        self.assertFalse(self.request("/api/pose", {"name": "不存在"})["ok"])
        self.bus.running_id = 1
        self.assertFalse(self.request("/api/pose", {"name": "立正"})["ok"])
        self.bus.running_id = None
        self.runner.thread = MagicMock()
        self.runner.thread.is_alive.return_value = True
        for path, body in [("/api/pose", {"name": "姿态2"}),
                           ("/api/cartesian", {"direction": "上", "distance_mm": 50}),
                           ("/api/run", {"id": 1, "direction": "left", "speed": 100}),
                           ("/api/mode", {"id": "all", "mode": "wheel"})]:
            self.assertFalse(self.request(path, body)["ok"])
        self.assertEqual(self.bus.writes, [])

    def test_cartesian_routes_six_directions_and_distances_without_hardware_writes(self):
        self.arm.cartesian_initialized = False
        simulated = Arm(load_advanced())
        self.arm.config = simulated.config
        self.arm.config["motion_calibrated"] = True
        self.arm.config["kinematics"]["calibrated"] = True
        self.arm.model = simulated.model
        self.bus.raw = simulated.positions().tolist()
        with patch.object(self.runner, "submit", return_value=True) as submit:
            for direction, axis, sign in [("前", 0, 1), ("后", 0, -1), ("左", 1, 1),
                                          ("右", 1, -1), ("上", 2, 1), ("下", 2, -1)]:
                with self.subTest(direction=direction):
                    self.assertTrue(self.request("/api/cartesian", {"direction": direction, "distance_mm": 50})["ok"])
                    delta = [0.0] * 3
                    delta[axis] = sign * 50
                    submit.assert_called_with(("move", delta))
                    self.assertTrue(self.request("/api/cartesian", {"direction": direction, "continuous": True})["ok"])
                    delta[axis] = sign * self.arm.config["step_mm"]
                    submit.assert_called_with(("jog", delta))
        state = self.request("/api/state")
        self.assertEqual(state["pose"]["kind"], "cartesian")
        self.assertTrue(state["cartesian"]["calibrated"])
        self.assertEqual(self.bus.writes, [])

    def test_working_pose_uses_base_for_horizontal_but_not_vertical_motion(self):
        for axis, sign in [(0, 1), (0, -1), (2, 1), (2, -1)]:
            with self.subTest(axis=axis, sign=sign):
                config = load_advanced()
                config["poses"]["初始状态"][0] = 943
                raw = config["poses"]["初始状态"]
                bus = FeedbackBus(load_config("config.json"), raw)
                arm = Arm(config, bus)
                delta = np.eye(3)[axis] * sign * 5
                origin = arm.model.forward(raw)
                with patch.object(arm.cancel, "wait", return_value=False):
                    arm.move_linear(delta)
                np.testing.assert_allclose(arm.model.forward(bus.raw) - origin, delta, atol=2)
                self.assertEqual(any(j == 1 and (a == GOAL_POSITION or (a == 40 and v == [1]))
                                     for j, a, v in bus.writes), axis == 0)

    def test_compact_home_cartesian_deadband_and_stall(self):
        from joint_mapping import mapped_position, raw_position
        for distance, stalled in [(5, False), (10, False), (50, False), (5, True), (10, True)]:
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
                with self.subTest(distance=distance, stalled=stalled, axis=axis, sign=sign):
                    config = load_advanced()
                    config["poses"]["初始状态"][0] = 943
                    config["kinematics"].pop("position_control", None)
                    config["poses"]["初始状态"][1] = 880
                    raw = config["poses"]["初始状态"]
                    bus = FeedbackBus(load_config("config.json"), raw)
                    bus.stall = stalled
                    arm = Arm(config, bus)
                    origin = arm.model.forward(raw)
                    delta = np.eye(3)[axis] * sign * distance
                    arm.model.inverse(origin + delta, raw)
                    write = bus._write
                    crossing = arm.move_across_zero
                    clock = [0.0]
                    def firmware(joint, address, values):
                        current = bus.raw[joint - 1]
                        write(joint, address, values)
                        if address == GOAL_POSITION and joint in (2, 3) and not stalled:
                            difference = ((values[0] << 8) | values[1]) - current
                            bus.raw[joint - 1] = current + int(np.sign(difference)) * max(0, abs(difference) - 3)
                    def feedback_crossing(joint, goal):
                        self.assertEqual(joint, 1)
                        if not stalled:
                            mapping = config["joint_mapping"][str(joint)]
                            a, b = mapped_position(bus.raw[joint - 1], mapping), mapped_position(goal, mapping)
                            step = 1 if b > a else -1
                            bus.wheel_positions[joint] = [raw_position(p, mapping) for p in range(a, b + step, step)]
                        crossing(joint, goal)
                    def wait(seconds):
                        clock[0] += seconds
                        return False
                    with patch.object(bus, "_write", side_effect=firmware), \
                         patch.object(arm, "move_across_zero", side_effect=feedback_crossing), \
                         patch.object(arm.cancel, "wait", side_effect=wait), \
                         patch("time.monotonic", side_effect=lambda: clock[0]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        if stalled:
                            with self.assertRaises(PositionHoldTimeout):
                                arm.move_linear(delta)
                            self.assertEqual(bus.raw, raw)
                        else:
                            arm.move_linear(delta)
                            actual = arm.model.forward(bus.raw) - origin
                            self.assertGreater(actual[axis] * sign, distance * 0.5)
                            self.assertLess(np.linalg.norm(actual - delta), 3 if distance <= 10 else 5)
                            self.assertEqual(bus.raw[4:], raw[4:])
                            self.assertFalse(arm.last_move["projected"])
                            for joint, address, values in bus.writes:
                                if joint in (2, 3, 4) and address == GOAL_POSITION:
                                    low, high = config["kinematics"]["cartesian_raw_ranges"][str(joint)]
                                    self.assertLessEqual(low, (values[0] << 8) | values[1])
                                    self.assertLessEqual((values[0] << 8) | values[1], high)

    def test_residual_correction_keeps_original_deadline_when_feedback_oscillates(self):
        config = load_advanced()
        target = np.array(config["poses"]["初始状态"])
        raw = target.copy()
        raw[1] += 4
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        write = bus._write
        clock, writes = [0.0], []
        def oscillating(joint, address, values):
            write(joint, address, values)
            if joint == 2 and address == GOAL_POSITION:
                writes.append((values[0] << 8) | values[1])
                bus.raw[1] = int(target[1]) + (4 if len(writes) % 2 else -4)
        def wait(seconds):
            clock[0] += seconds
            if clock[0] > 4:
                raise AssertionError("correction ignored its deadline")
            return False
        with patch.object(bus, "_write", side_effect=oscillating), \
             patch.object(arm.cancel, "wait", side_effect=wait), \
             patch("time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(PositionHoldTimeout):
                arm.move_pose(target, quiet=True, settle_tolerance=3, correct_residual=True, residual_limit=8)
        self.assertLess(clock[0], 4)
        self.assertTrue(all(abs(goal - target[1]) <= 8 for goal in writes))

    def test_cartesian_arrival_does_not_hunt_within_tolerance(self):
        config = load_advanced()
        target = np.array(config["poses"]["初始状态"])
        raw = target.copy()
        raw[1] += 12
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        write = bus._write
        def offset(joint, address, values):
            write(joint, address, values)
            if joint == 2 and address == GOAL_POSITION:
                bus.raw[1] += 1
        with patch.object(bus, "_write", side_effect=offset), \
             patch.object(arm.cancel, "wait", return_value=False):
            arm.hold_pose(target, cartesian=True, duration_s=0.25, speed=250)
        self.assertEqual(bus.raw[1], target[1] + 1)
        self.assertFalse(any(address == GOAL_POSITION and len(values) == 2 for _, address, values in bus.writes))
        for joint, p, force, deadband in ((2, 15, 150, 1), (3, 30, 180, 2)):
            self.assertEqual(bus._read_u8(joint, 21), p)
            self.assertEqual(bus._read_u16(joint, 24), force)
            self.assertEqual(bus._read_u16(joint, 26), deadband * 257)
        arm.prepare()
        self.assertEqual(bus._read_u8(2, 21), 15)
        self.assertEqual(bus._read_u16(2, 24), 150)
        self.assertEqual(bus._read_u16(2, 26), 257)
        self.assertEqual(bus._read_u16(3, 26), 514)

    def test_cartesian_batch_shares_timing_and_preserves_support(self):
        config = load_advanced()
        config["poses"]["初始状态"][0] = 943
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        with patch.object(bus, "move_position", side_effect=AssertionError("individual joint command")), \
             patch.object(arm.cancel, "wait", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.move_cartesian([0, 0, 5])
        self.assertEqual(len(bus.batches), 1)
        address, values = bus.batches[0]
        self.assertEqual(address, GOAL_POSITION)
        self.assertEqual(set(values), {2, 3, 4, 5, 6})
        self.assertTrue(all(data[2:] == [0, 250, 0, 250] for data in values.values()))
        self.assertEqual(bus._read_u8(1, 40), 0)
        self.assertTrue(all(bus._read_u8(joint, 40) == 1 for joint in range(2, 7)))
        self.assertIsNone(bus.running_id)

    def test_cartesian_cancel_stops_batch_without_resending(self):
        config = load_advanced()
        config["poses"]["初始状态"][0] = 943
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        sync_write = bus._sync_write
        def cancel(address, values):
            sync_write(address, values)
            arm.cancel.set()
        with patch.object(bus, "_sync_write", side_effect=cancel), self.assertRaises(InterruptedError):
            arm.move_cartesian([0, 0, 5])
        self.assertEqual(len(bus.batches), 1)
        self.assertTrue(all(bus._read_u8(joint, 40) == 0 for joint in range(1, 7)))
        self.assertIsNone(bus.running_id)

    def test_cartesian_motion_overcomes_three_count_command_deadband(self):
        config = load_advanced()
        config["poses"]["初始状态"][0] = 943
        raw = config["poses"]["初始状态"]
        bus = FeedbackBus(load_config("config.json"), raw)
        arm = Arm(config, bus)
        origin = arm.model.forward(raw)
        write = bus._write
        def deadband(joint, address, values):
            before = bus.raw[joint - 1]
            write(joint, address, values)
            if address == GOAL_POSITION and abs(bus.raw[joint - 1] - before) <= 3:
                bus.raw[joint - 1] = before
        with patch.object(bus, "_write", side_effect=deadband), \
             patch.object(arm.cancel, "wait", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.move_linear([0, 0, 5])
        np.testing.assert_allclose(arm.model.forward(bus.raw) - origin, [0, 0, 5], atol=3)
        self.assertGreater(bus.raw[1] - raw[1], 3)
        self.assertFalse(any(address in (9, 11, 48) for _, address, _ in bus.writes))

    def test_home_from_stopped_folded_pose_needs_no_wheel_mode(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), [1022, 896, 314, 659, 89, 275])
        arm = Arm(config, bus)
        with patch.object(bus, "move_position", wraps=bus.move_position) as move, \
             patch.object(arm.cancel, "wait", return_value=False), \
             patch.object(arm, "move_across_zero", side_effect=AssertionError("unexpected crossing")), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("home", "初始状态"))
        self.assertEqual(bus.raw, config["poses"]["初始状态"])
        self.assertTrue(arm.cartesian_initialized)
        self.assertEqual([call.args[0] for call in move.call_args_list], [4, 3, 2, 5, 6, 1])
        self.assertEqual(len(bus.batches), 0)
        self.assertFalse(any(address in (9, 11) for _, address, _ in bus.writes))

    def test_home_from_reported_position_overcomes_small_command_deadband(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), [1022, 898, 311, 696, 80, 267])
        arm = Arm(config, bus)
        write = bus._write
        def deadband(joint, address, values):
            before = bus.raw[joint - 1]
            write(joint, address, values)
            if address == GOAL_POSITION and joint in (2, 3) and abs(bus.raw[joint - 1] - before) <= 8:
                bus.raw[joint - 1] = before
        with patch.object(bus, "_write", side_effect=deadband), \
             patch.object(bus, "move_position", wraps=bus.move_position) as move, \
             patch.object(bus, "move_positions", wraps=bus.move_positions) as batch, \
             patch.object(arm.cancel, "wait", return_value=False), \
             patch.object(arm, "move_across_zero", side_effect=AssertionError("unexpected crossing")), \
             contextlib.redirect_stdout(io.StringIO()):
            arm.execute(("home", "初始状态"))
        self.assertEqual(bus.raw, config["poses"]["初始状态"])
        self.assertTrue(arm.cartesian_initialized)
        batch.assert_not_called()
        self.assertEqual([call.args[0] for call in move.call_args_list], [4, 3, 2, 6, 1])
        self.assertTrue(all(call.args[1] == config["poses"]["初始状态"][call.args[0] - 1]
                            and call.args[3] == 180 for call in move.call_args_list))
        self.assertFalse(any(address in (9, 11) for _, address, _ in bus.writes))

    def test_home_from_fixed_poses_folds_elbow_before_lowering_shoulder(self):
        for raw, stalled in [(raw, stalled) for raw in ([1022, 138, 22, 727, 83, 276],
                                                       [1022, 121, 877, 985, 81, 269]) for stalled in (False, True)]:
            with self.subTest(raw=raw, stalled=stalled):
                config = load_advanced()
                target = config["poses"]["初始状态"]
                bus = FeedbackBus(load_config("config.json"), raw)
                bus.wheel_positions[2] = [14, 9, 3, 0, 1023, 1018, 1010, 1002]
                bus.wheel_positions[3] = [976, 995, 1008, 1016, 1023, 5, 12, 20, 26]
                arm = Arm(config, bus)
                move = bus.move_position
                clock = [0.0]
                lowered = []
                def move_checked(joint, goal, duration, speed):
                    if joint == 2 and abs(goal - target[1]) <= 40:
                        self.assertEqual(bus.raw[2], target[2])
                        lowered.append(goal)
                    result = move(joint, goal, duration, speed)
                    if stalled and joint == 2 and abs(goal - target[1]) <= 40:
                        bus.raw[1] = goal + 13
                    return result
                def wait(seconds):
                    clock[0] += seconds
                    return False
                with patch.object(bus, "move_position", side_effect=move_checked), \
                     patch.object(arm.cancel, "wait", side_effect=wait), \
                     patch("time.monotonic", side_effect=lambda: clock[0]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if stalled:
                        with self.assertRaises(PositionHoldTimeout):
                            arm.execute(("pose", "初始状态"))
                        self.assertFalse(arm.cartesian_initialized)
                        self.assertNotIn(target[1], lowered)
                    else:
                        arm.execute(("pose", "初始状态"))
                        self.assertEqual(bus.raw, target)
                        self.assertTrue(arm.cartesian_initialized)
                self.assertTrue(lowered)
                self.assertIsNone(bus.running_id)

    def test_cartesian_range_rejects_shoulder_zero_without_changing_measured_arc(self):
        config = load_advanced()
        bus = FeedbackBus(load_config("config.json"), config["poses"]["初始状态"])
        arm = Arm(config, bus)
        for shoulder in (0, 4, 1023):
            current = bus.raw.copy()
            current[1] = shoulder
            arm.validate_pose(current)
            with self.assertRaises(ValueError):
                arm.model.inverse(arm.model.forward(current) + [5, 0, 0], current)
            with self.assertRaises(ValueError):
                arm.model.project_step(np.array([5, 0, 0]), current)
        self.assertEqual(bus.writes, [])

    def test_base_raw_seam_is_inside_calibrated_arc_and_accepts_both_directions(self):
        self.arm.config["kinematics"]["calibrated"] = True
        self.arm.cartesian_initialized = True
        self.bus.raw = list(self.arm.config["poses"]["初始状态"])
        for raw, expected in [(1022, 5), (1023, 6), (0, 7), (1, 8)]:
            self.bus.raw[0] = raw
            state = self.request("/api/state")["servos"][0]
            self.assertEqual(state["mapped_position"], expected)
            self.assertEqual(state["mapped_limits"], [-319, 301])
            with patch.object(self.runner, "submit", return_value=True) as submit:
                for direction in ("左", "右"):
                    self.assertTrue(self.request("/api/cartesian", {"direction": direction, "distance_mm": 5})["ok"])
                self.assertEqual(submit.call_count, 2)
        self.assertEqual(self.bus.writes, [])

    def test_cartesian_base_seam_cannot_enter_other_raw_branch(self):
        for start, finish in [(1022, 16), (16, 1022)]:
            config = load_advanced()
            raw = list(config["poses"]["初始状态"])
            raw[0] = start
            bus = FeedbackBus(load_config("config.json"), raw)
            arm = Arm(config, bus)
            target = raw.copy()
            target[0] = finish
            with self.assertRaises(ValueError):
                arm.model.inverse(arm.model.forward(target), raw, allow_crossing=False)
            self.assertEqual(bus.writes, [])

    def test_jog_projection_stops_without_shifting_line_origin(self):
        arm = Arm(load_advanced())
        with patch.object(arm, "move_cartesian", return_value=True) as move:
            with self.assertRaisesRegex(ValueError, "直线运动停止"):
                arm.execute(("jog", [5, 0, 0]))
        move.assert_called_once()
        self.assertTrue(arm.cancel.is_set())

    def test_linear_projection_stops_without_shifting_line_origin(self):
        arm = Arm(load_advanced())
        start = arm.model.forward(arm.positions())
        displacement = np.array([1.0, 1.0, 0.0])
        with patch.object(arm, "move_cartesian", return_value=True) as move, \
             patch.object(arm.model, "forward", side_effect=[start, start, start + displacement]):
            arm.move_linear([50, 0, 0])
        move.assert_called_once()
        np.testing.assert_array_equal(move.call_args.args[0], [5, 0, 0])
        self.assertTrue(arm.last_move["projected"])
        self.assertEqual(arm.last_move["requested_mm"], [50, 0, 0])
        self.assertEqual(arm.last_move["actual_mm"], displacement.tolist())

    def test_linear_corrects_logged_endpoint_errors_without_accepting_stall(self):
        for requested, feedback, corrected, success in [
                ([-5, 0, 0], [-2.32, -0.07, -1.93], [-4.6, 0, 0], True),
                ([-5, 0, 0], [-2.32, -0.07, -1.93], [-2.32, -0.07, -1.93], False),
                ([50, 0, 0], [53.49, 1.64, -2.9], [50.5, 0.1, 0.1], True),
                ([50, 0, 0], [53.49, 1.64, -2.9], [53.49, 1.64, -2.9], False)]:
            with self.subTest(requested=requested, success=success):
                arm = Arm(load_advanced())
                count = int(np.ceil(np.linalg.norm(requested) / 5))
                position = np.zeros(3)
                calls = []
                def move(delta):
                    nonlocal position
                    calls.append(np.array(delta))
                    position = np.array(feedback if len(calls) <= count else corrected)
                    return False
                with patch.object(arm.model, "forward", side_effect=lambda raw: position.copy()), \
                     patch.object(arm, "move_cartesian", side_effect=move), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    if success:
                        arm.move_linear(requested)
                    else:
                        with self.assertRaisesRegex(PositionHoldTimeout, "未达到请求位移"):
                            arm.move_linear(requested)
                self.assertEqual(len(calls), count + 1)
                np.testing.assert_allclose(calls[-1], np.array(requested) - feedback)
                self.assertEqual("模型反馈位移(mm)" in output.getvalue(), success)

    def test_base_follows_allowed_long_arc_instead_of_raw_shortcut(self):
        from joint_mapping import mapped_position, raw_position
        config = load_advanced()
        raw = list(config["poses"]["初始状态"])
        raw[0] = 700
        target = raw.copy()
        target[0] = 290
        bus = FeedbackBus(load_config("config.json"), raw)
        mapping = config["joint_mapping"]["1"]
        a, b = mapped_position(700, mapping), mapped_position(290, mapping)
        self.assertEqual(b - a, 614)
        bus.wheel_positions[1] = [raw_position(round(p), mapping) for p in np.linspace(a, b, 104)]
        arm = Arm(config, bus)
        with patch.object(arm.cancel, "wait", return_value=False):
            arm.move_pose(target, quiet=True)
        self.assertEqual(bus.raw[0], 290)
        self.assertIn((1, RUNNING_TIME, [4, 100]), bus.writes)
        self.assertNotIn((1, RUNNING_TIME, [0, 100]), bus.writes)

    def test_home_accepts_stable_three_count_error_but_rejects_stall(self):
        for error in (3, 8):
            config = load_advanced()
            target = config["poses"]["初始状态"]
            raw = list(target)
            raw[2] += 8
            bus = FeedbackBus(load_config("config.json"), raw)
            arm = Arm(config, bus)
            write = bus._write
            clock = [0.0]
            def offset(joint, address, values):
                write(joint, address, values)
                if joint == 3 and address == GOAL_POSITION:
                    bus.raw[2] += error
            def wait(seconds):
                clock[0] += seconds
                return False
            with patch.object(bus, "_write", side_effect=offset), \
                 patch.object(arm.cancel, "wait", side_effect=wait), \
                 patch("time.monotonic", side_effect=lambda: clock[0]):
                if error == 3:
                    arm.execute(("home", "初始状态"))
                    self.assertTrue(arm.cartesian_initialized)
                    self.assertEqual(bus.raw[2], target[2] + 3)
                else:
                    with self.assertRaises(PositionHoldTimeout):
                        arm.execute(("home", "初始状态"))
                    self.assertFalse(arm.cartesian_initialized)

    def test_cartesian_home_dispatches_recorded_pose_without_enabling_uncalibrated_motion(self):
        with patch.object(self.runner, "submit", return_value=True) as submit:
            self.assertTrue(self.request("/api/cartesian/home", {})["ok"])
            submit.assert_called_once_with(("home", "初始状态"))
        self.assertFalse(self.arm.cartesian_initialized)
        self.assertFalse(self.arm.config["kinematics"]["calibrated"])
        self.assertEqual(self.bus.writes, [])
        self.assertFalse(self.request("/api/cartesian", {"direction": "前", "distance_mm": 5})["ok"])
        self.arm.config["kinematics"]["calibrated"] = True
        self.assertFalse(self.request("/api/cartesian", {"direction": "前", "distance_mm": 5})["ok"])
        self.bus.raw[3] = 684
        with patch.object(self.runner, "submit", return_value=True) as submit:
            result = self.request("/api/cartesian", {"direction": "前", "distance_mm": 5})
            self.assertTrue(result["ok"])
            submit.assert_called_once_with(("move", [5.0, 0.0, 0.0]))
        self.assertEqual(self.bus.writes, [])

    def test_home_arrival_is_distinct_from_kinematic_validation_and_cancel(self):
        target = self.arm.config["poses"]["初始状态"]
        self.bus.raw = [target[i] + (0 if i in (0, 5) else 8) for i in range(6)]
        with patch.object(self.arm.cancel, "wait", return_value=False):
            self.arm.execute(("home", "初始状态"))
        self.assertEqual(self.bus.raw, target)
        self.assertTrue(self.arm.cartesian_initialized)
        self.assertFalse(self.arm.config["kinematics"]["calibrated"])
        self.assertFalse(any(j == 1 and a == GOAL_POSITION for j, a, _ in self.bus.writes))
        np.testing.assert_allclose(self.arm.model.forward(self.arm.config["kinematics"]["zero_raw"] + [target[5]]), [363.46165, -0.39408, 224.70088], atol=0.01)
        self.arm.cancel.set()
        with self.assertRaises(InterruptedError):
            self.arm.execute(("home", "初始状态"))
        self.assertFalse(self.arm.cartesian_initialized)

    def test_cartesian_test_settings_are_session_only_and_do_not_move(self):
        from pathlib import Path
        original_file = Path("advanced.json").read_bytes()
        settings = self.request("/api/state")["cartesian"]["settings"]
        body = dict(settings, confirmed=True)
        self.assertTrue(self.request("/api/cartesian/settings", body)["ok"])
        state = self.request("/api/state")["cartesian"]
        with patch.object(self.runner, "submit") as submit:
            result = self.request("/api/cartesian", {"direction": "前", "distance_mm": 10})
            self.assertFalse(result["ok"])
            submit.assert_not_called()
        self.assertFalse(state["calibrated"])
        self.assertEqual(state["settings"], settings)
        self.assertEqual(self.bus.writes, [])
        self.assertEqual(Path("advanced.json").read_bytes(), original_file)

    def test_changed_kinematics_do_not_retain_previous_validation(self):
        self.arm.config["kinematics"]["calibrated"] = True
        settings = self.request("/api/state")["cartesian"]["settings"]
        settings["directions"] = [-1] * 5
        self.assertTrue(self.request("/api/cartesian/settings", dict(settings, confirmed=True))["ok"])
        self.assertFalse(self.request("/api/state")["cartesian"]["calibrated"])
        self.assertEqual(self.bus.writes, [])

    def test_cartesian_test_settings_reject_invalid_or_unconfirmed_without_changes(self):
        original = copy.deepcopy(self.arm.config)
        settings = self.request("/api/state")["cartesian"]["settings"]
        for patch_values in ({"confirmed": False}, {"directions": [0]*5},
                             {"zero_raw": [1024]*5}, {"degrees_per_raw": float("nan")}):
            body = dict(settings, confirmed=True)
            body.update(patch_values)
            self.assertFalse(self.request("/api/cartesian/settings", body)["ok"])
            self.assertEqual(self.arm.config, original)
        self.bus.running_id = 1
        self.assertFalse(self.request("/api/cartesian/settings", dict(settings, confirmed=True))["ok"])
        self.assertEqual(self.bus.writes, [])

    def test_cartesian_rejects_uncalibrated_invalid_distance_and_overlapping_joint(self):
        self.arm.cartesian_initialized = True
        with patch.object(self.runner, "submit") as submit:
            result = self.request("/api/cartesian", {"direction": "上", "distance_mm": 50})
            self.assertFalse(result["ok"])
            self.assertIn("运动学标定", result["error"])
            self.arm.config["kinematics"]["calibrated"] = True
            for body in [{"direction": "上", "distance_mm": value}
                         for value in [0, -1, 201, float("nan"), float("inf"), "五厘米"]] + [
                             {"direction": "未知", "distance_mm": 50},
                             {"direction": "上", "continuous": 1}]:
                self.assertFalse(self.request("/api/cartesian", body)["ok"])
            self.bus.running_id = 1
            self.assertFalse(self.request("/api/cartesian", {"direction": "上", "distance_mm": 50})["ok"])
            submit.assert_not_called()
        self.assertEqual(self.bus.writes, [])

    def test_stop_cancels_pose_and_releases_all_joints(self):
        with patch.object(self.arm, "execute", side_effect=lambda command: self.arm.cancel.wait(2)):
            self.runner.submit(("pose", "立正"))
            self.assertTrue(self.request("/api/stop", {"id": 2})["ok"])
            self.runner.wait()
        self.assertTrue(self.arm.cancel.is_set())
        self.assertEqual({joint for joint, address, value in self.bus.writes
                          if address == 40 and value == [0]}, set(range(1, 7)))

    def test_stop_readback_failure_is_reported_without_success_status(self):
        with patch.object(self.bus, "_read_u8", return_value=1):
            result = self.request("/api/stop", {"id": "all"})
        self.assertFalse(result["ok"])
        self.assertIn("停止未确认", self.server.pose_stop_reason)
        self.assertNotIn("已确认", self.server.pose_stop_reason)

    def test_browser_heartbeat_loss_stops_pose(self):
        with patch.object(self.arm, "execute", side_effect=lambda command: self.arm.cancel.wait(2)):
            self.runner.submit(("pose", "立正"))
            self.assertTrue(self.request("/api/heartbeat", {"id": "pose"})["ok"])
            self.server.service_actions()
            self.assertFalse(self.arm.cancel.is_set())
            self.server.pose_heartbeat -= 2
            self.server.service_actions()
            self.runner.wait()
        self.assertTrue(self.arm.cancel.is_set())
        self.assertIn("心跳中断", self.request("/api/state")["pose"]["stop_reason"])

    def test_failed_pose_is_reported_as_failed(self):
        self.server.pose_name = "立正"
        self.runner.error = TimeoutError("ID 2 回位未取得有效进展，停止")
        state = self.request("/api/state")
        self.assertFalse(state["pose"]["busy"])
        self.assertIn("ID 2", state["pose"]["error"])
        self.assertEqual(self.bus.writes, [])


if __name__ == "__main__":
    unittest.main()
