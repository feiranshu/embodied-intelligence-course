"""Part 4: track a user-selected textured object with optical flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_utils import DEFAULT_CAMERA, open_camera  # noqa: E402
from common import ARM_CODE, calibration, load_hand_eye, read_json  # noqa: E402
from part3_visual_servo.visual_servo import run_servo_loop  # noqa: E402


HERE = Path(__file__).resolve().parent


class OpticalFlowTracker:
    """Track feature points inside a selected rectangle; no object model needed."""

    def __init__(self, frame: np.ndarray, roi: tuple[int, int, int, int]):
        x, y, w, h = (int(value) for value in roi)
        height, width = frame.shape[:2]
        if w < 20 or h < 20 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError("请选择至少 20×20 像素且完全位于画面内的目标区域")
        self.box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = self._features(self.gray)
        if len(self.points) < 10:
            raise ValueError("所选目标纹理不足；请选择有明显边缘或纹理的区域")

    def _features(self, gray: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.rint(self.box).astype(np.int32), 255)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.01,
                                         minDistance=6, mask=mask, blockSize=5)
        return points if points is not None else np.empty((0, 1, 2), np.float32)

    def detect(self, frame: np.ndarray, _K: np.ndarray, _dist: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new, good, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None,
                                                 winSize=(21, 21), maxLevel=3)
        if new is None or good is None:
            return None
        previous = self.points.reshape(-1, 2)[good.ravel() == 1]
        current = new.reshape(-1, 2)[good.ravel() == 1]
        if len(current) < 8:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(previous, current,
                                                       method=cv2.RANSAC,
                                                       ransacReprojThreshold=3)
        if matrix is None or inliers is None or int(inliers.sum()) < 8:
            return None
        self.box = cv2.transform(self.box[None], matrix)[0]
        height, width = gray.shape
        if (not np.isfinite(self.box).all() or np.min(self.box[:, 0]) < 0
                or np.min(self.box[:, 1]) < 0 or np.max(self.box[:, 0]) >= width
                or np.max(self.box[:, 1]) >= height):
            return None
        self.points = current[inliers.ravel() == 1].reshape(-1, 1, 2).astype(np.float32)
        if len(self.points) < 30:
            replenished = self._features(gray)
            if len(replenished) > len(self.points):
                self.points = replenished
        self.gray = gray
        cv2.polylines(frame, [np.rint(self.box).astype(np.int32)], True, (0, 255, 0), 2)
        center = self.box.mean(axis=0)
        cv2.circle(frame, tuple(np.rint(center).astype(int)), 5, (0, 0, 255), -1)
        return center, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="实机运动；缺省仅预览跟踪")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="默认按 Windows 设备名选择 GENERAL WEBCAM；也可传数字索引")
    parser.add_argument("--intrinsics", type=Path,
                        default=HERE.parent / "part1_camera_calibration" / "camera_intrinsics.json")
    parser.add_argument("--hand-eye", type=Path,
                        default=HERE.parent / "part2_hand_eye_calibration" / "hand_eye.json")
    parser.add_argument("--advanced-config", type=Path, default=ARM_CODE / "advanced.json")
    parser.add_argument("--hardware-config", type=Path, default=ARM_CODE / "config.json")
    parser.add_argument("--port")
    parser.add_argument("--step-mm", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-travel-mm", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=HERE / "run_log.json")
    args = parser.parse_args()
    if not 0 < args.step_mm <= 5 or not 1 <= args.max_steps <= 10 or not 0 < args.max_travel_mm <= 30:
        parser.error("本示例的步长、步数或总行程超过上限")
    K, dist, size = calibration(args.intrinsics)
    camera, selected_index, backend, device_name = open_camera(args.camera, size)
    print(f"使用摄像头“{device_name}”（索引 {selected_index}，{backend}）")
    arm = bus = None
    try:
        ok, frame = camera.read()
        if not ok or (frame.shape[1], frame.shape[0]) != size:
            raise RuntimeError("相机读取失败或分辨率与标定时不同")
        roi = cv2.selectROI("Part 4 - select target then Enter", frame, showCrosshair=True)
        cv2.destroyWindow("Part 4 - select target then Enter")
        tracker = OpticalFlowTracker(frame, roi)
        if args.execute:
            T_gc = load_hand_eye(args.hand_eye)
            if input("确认末端相机固定、目标区域清空、可立即断电；输入 RUN 启动：").strip() != "RUN":
                print("未启动实机运动")
                return
            from advanced import Arm
            from servo_bus import ServoBus, load_config
            bus = ServoBus(load_config(args.hardware_config), args.port)
            arm = Arm(read_json(args.advanced_config), bus)
        else:
            T_gc = None
            print("预览模式：不打开串口，不驱动机械臂")
        run_servo_loop(camera, tracker, K, dist, size, arm=arm, T_gc=T_gc,
                       step_mm=args.step_mm, forward=False,
                       max_steps=args.max_steps, max_travel_mm=args.max_travel_mm,
                       output=args.output if args.execute else None)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if arm is not None:
            try:
                arm.stop()
            finally:
                bus.close()
        elif bus is not None:
            bus.close()


if __name__ == "__main__":
    main()
