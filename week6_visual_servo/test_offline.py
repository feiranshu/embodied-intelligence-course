"""Hardware-free checks for the Week 6 geometry and tracking decisions."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from common import arm_model, base_gripper_pose, board_points, inverse, read_image, transform, write_image
from part2_hand_eye_calibration.web_app import HandEyeState, capture_blocked_joints
from part3_visual_servo.visual_servo import camera_step
from part4_object_tracking.object_tracking import OpticalFlowTracker


class Week6OfflineTests(unittest.TestCase):
    def test_unicode_image_path(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "棋盘照片.png"
            image = np.full((20, 30, 3), 127, np.uint8)
            write_image(path, image)
            np.testing.assert_array_equal(read_image(path), image)

    def test_chessboard_geometry(self):
        points = board_points(9, 6, 25)
        self.assertEqual(points.shape, (54, 3))
        np.testing.assert_array_equal(points[-1], [200, 125, 0])

    def test_full_pose_matches_existing_forward_position(self):
        model = arm_model()
        raw = model.validate([990, 920, 300, 740, 80, 276])
        np.testing.assert_allclose(base_gripper_pose(raw, model)[:3, 3], model.forward(raw), atol=1e-8)

    def test_hand_eye_coordinate_order(self):
        rng = np.random.default_rng(123)
        T_gc = transform(cv2.Rodrigues(np.array([0.2, -0.3, 0.15]))[0], [35, -25, 50])
        T_bt = transform(cv2.Rodrigues(np.array([0.1, 0.4, -0.2]))[0], [280, 50, 230])
        gripper_base, target_camera = [], []
        for _ in range(25):
            R = cv2.Rodrigues(rng.uniform(-0.8, 0.8, 3))[0]
            T_bg = transform(R, rng.uniform([-80, -80, 30], [80, 80, 180]))
            T_ct = inverse(T_gc) @ inverse(T_bg) @ T_bt
            gripper_base.append(T_bg)
            target_camera.append(T_ct)
        R, t = cv2.calibrateHandEye(
            [T[:3, :3] for T in gripper_base], [T[:3, 3] for T in gripper_base],
            [T[:3, :3] for T in target_camera], [T[:3, 3] for T in target_camera],
            method=cv2.CALIB_HAND_EYE_PARK,
        )
        np.testing.assert_allclose(transform(R, t), T_gc, atol=1e-7)

    def test_servo_rule_and_stop(self):
        size = (1000, 800)
        np.testing.assert_array_equal(camera_step([100, 400], size, 10, 300, 200, forward=True), [-10, 0, 0])
        np.testing.assert_array_equal(camera_step([900, 400], size, 10, 300, 200, forward=True), [10, 0, 0])
        np.testing.assert_array_equal(camera_step([500, 50], size, 10, 300, 200, forward=True), [0, -10, 0])
        np.testing.assert_array_equal(camera_step([500, 750], size, 10, 300, 200, forward=True), [0, 10, 0])
        np.testing.assert_array_equal(camera_step([500, 400], size, 10, 300, 200, forward=True), [0, 0, 10])
        self.assertIsNone(camera_step([500, 400], size, 10, 205, 200, forward=True))
        self.assertIsNone(camera_step([500, 400], size, 5, None, 0, forward=False))

    def test_optical_flow_tracks_textured_target(self):
        rng = np.random.default_rng(3)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for point in rng.integers([70, 70], [160, 160], size=(80, 2)):
            cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1)
        tracker = OpticalFlowTracker(frame, (60, 60, 110, 110))
        tracker.detect(frame.copy(), None, None)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 7], [0, 1, -4]]), (320, 240))
        center, depth = tracker.detect(moved, None, None)
        np.testing.assert_allclose(center, [122, 111], atol=1)
        self.assertIsNone(depth)

    def test_web_capture_accepts_detected_board_outside_mapping_and_without_stability(self):
        frame = np.full((60, 80, 3), 127, dtype=np.uint8)
        corners = np.zeros((48, 1, 2), dtype=np.float32)

        class Camera:
            name = "test camera"
            index = 0
            backend = "test"
            found = True

            def snapshot(self):
                return frame.copy(), corners.copy() if self.found else None, 0, ""

        raw = np.array([631, 907, 315, 783, 31, 0], dtype=int)
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            state = object.__new__(HandEyeState)
            state.args = SimpleNamespace(image_size=(80, 60), intrinsics="camera_intrinsics.json")
            state.cols, state.rows = 8, 6
            state.output = Path(folder)
            state.manifest = Path(folder) / "samples.json"
            state.samples = [{"name": "pose_001", "image": "pose_001.png", "raw": raw.tolist()}]
            state.result = {"old": True}
            state.state_lock = RLock()
            state.motion_busy = True
            state.motion_active = True
            state.motion_status = ""
            state.motion_error = ""
            state.capture_busy = False
            state.solve_busy = False
            state.solve_status = "idle"
            state.solve_error = ""
            state.raw = raw.tolist()
            state.read_error = ""
            state.camera = Camera()
            state.bus = SimpleNamespace(port="COM8")
            state.arm = SimpleNamespace(model=arm_model(), positions=lambda: raw.copy())

            status = state.status()
            self.assertEqual(status["board_stable_frames"], 0)
            self.assertFalse(status["pose_issues"])
            self.assertTrue(status["can_capture"])

            with patch("part2_hand_eye_calibration.web_app.chessboard_corners", return_value=corners):
                result = state.capture_pose()

            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["raw"], raw.tolist())
            self.assertTrue((state.output / "pose_002.png").exists())
            self.assertIsNone(state.result)

            # Cached webpage feedback can be allowed while the next live read is blocked.
            # No new image/record may be written in that case; ID6 remains ignored.
            for joint in range(5):
                for value in [0, 30, 1000, 1022]:
                    with self.subTest(joint=joint+1, value=value):
                        raw[:] = [631, 907, 315, 783, 31, 1022]
                        state.raw = raw.tolist()
                        self.assertTrue(state.status()["can_capture"])
                        raw[joint] = value
                        with self.assertRaisesRegex(RuntimeError, f"ID {joint+1}"):
                            state.capture_pose()
                        status = state.status()
                        self.assertFalse(status["can_capture"])
                        self.assertEqual(status["capture_blocked_joints"][0]["id"], joint+1)
                        self.assertEqual(len(state.samples), 2)
                        self.assertFalse((state.output / "pose_003.png").exists())

            raw[:] = [31, 999, 31, 999, 1023, 1022]
            state.raw = raw.tolist()
            self.assertTrue(state.status()["can_capture"])
            state.camera.found = False
            self.assertFalse(state.status()["can_capture"])
            with self.assertRaisesRegex(RuntimeError, "完整棋盘"):
                state.capture_pose()
            self.assertEqual(len(state.samples), 2)

    def test_capture_feedback_exact_bands_and_all_blocked_ids(self):
        self.assertEqual(capture_blocked_joints([31, 999, 1023, 31, 999, 0]), [])
        self.assertEqual(capture_blocked_joints([31, 999, 1023, 31, 999, 1022]), [])
        blocked = capture_blocked_joints([0, 30, 1000, 1022, 500, 1022])
        self.assertEqual([item["id"] for item in blocked], [1, 2, 3, 4])
        self.assertEqual([item["raw"] for item in blocked], [0, 30, 1000, 1022])

    def test_web_feedback_error_identifies_failed_servo(self):
        state = object.__new__(HandEyeState)
        def read(joint):
            if joint == 3:
                raise RuntimeError("无回复")
            return 500
        state.bus = SimpleNamespace(lock=RLock(), read_position=read)
        with self.assertRaisesRegex(RuntimeError, "ID 3.*无回复"):
            state.read_capture_positions()

    def test_delete_samples_archives_files_and_invalidates_old_result(self):
        from common import read_json, write_json
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            state = object.__new__(HandEyeState)
            state.output, state.manifest = root, root / "samples.json"
            state.args = SimpleNamespace(result=root / "hand_eye.json")
            state.state_lock = RLock()
            state.capture_busy = state.solve_busy = False
            state.samples = [{"name":f"pose_{i:03d}","image":f"pose_{i:03d}.png","raw":[500]*6} for i in (1,2)]
            state.result = {"previous":True}
            write_json(state.manifest,{"board_inner_corners":[8,6],"samples":state.samples})
            write_json(state.args.result,state.result)
            for image in (root/'pose_001.png',root/'pose_002.png',root/'hand_eye.png'):
                image.write_bytes(b'test image')
            result = state.delete_samples("pose_001")
            self.assertEqual(result["sample_count"],1)
            self.assertFalse((root/'pose_001.png').exists())
            self.assertTrue((root/result['archive']/'images/pose_001.png').exists())
            self.assertTrue((root/'pose_002.png').exists())
            self.assertFalse(state.args.result.exists())
            self.assertIsNone(state.result)
            self.assertEqual(read_json(state.manifest)['samples'],state.samples)
            self.assertEqual(read_json(state.manifest)['board_inner_corners'],[8,6])
            with self.assertRaisesRegex(ValueError,"没有找到"):
                state.delete_samples("missing")
            state.solve_busy = True
            with self.assertRaisesRegex(RuntimeError,"正在采集或求解"):
                state.delete_samples(delete_all=True)
            state.solve_busy = False
            result = state.delete_samples(delete_all=True)
            self.assertEqual(result['sample_count'],0)
            self.assertEqual(read_json(state.manifest)['samples'],[])
            self.assertFalse((root/'pose_002.png').exists())

    def test_delete_samples_rolls_back_when_manifest_cannot_be_saved(self):
        from common import write_json
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            state = object.__new__(HandEyeState)
            state.output, state.manifest = root, root/'samples.json'
            state.args = SimpleNamespace(result=root/'hand_eye.json')
            state.state_lock = RLock()
            state.capture_busy = state.solve_busy = False
            state.samples = [{'name':'pose_001','image':'pose_001.png','raw':[500]*6}]
            write_json(state.manifest,{'samples':state.samples})
            (root/'pose_001.png').write_bytes(b'original')
            def fail_manifest(path, value):
                if Path(path)==state.manifest:
                    raise OSError('disk error')
                write_json(path,value)
            with patch('part2_hand_eye_calibration.web_app.write_json',side_effect=fail_manifest):
                with self.assertRaisesRegex(OSError,'disk error'):
                    state.delete_samples('pose_001')
            self.assertEqual((root/'pose_001.png').read_bytes(),b'original')
            self.assertEqual(len(state.samples),1)
            state.samples[0]['image']='../outside.png'
            with self.assertRaisesRegex(ValueError,'照片路径'):
                state.delete_samples('pose_001')


if __name__ == "__main__":
    unittest.main()
