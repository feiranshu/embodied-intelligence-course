"""Part 3: ArUco-based simple visual servo (one centimetre camera steps)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_utils import DEFAULT_CAMERA, open_camera  # noqa: E402
from common import (ARM_CODE, base_gripper_pose, calibration, ensure_image_size,
                    load_hand_eye, read_json, write_json)  # noqa: E402
from part3_visual_servo.motion_model import matching_motion_model  # noqa: E402


HERE = Path(__file__).resolve().parent


class MarkerDetector:
    def __init__(self, marker_id: int, marker_mm: float, dictionary_name: str):
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"未知 ArUco 字典：{dictionary_name}")
        self.marker_id = marker_id
        self.marker_mm = marker_mm
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 53
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.minMarkerPerimeterRate = 0.015
        parameters.minDistanceToBorder = 2
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
        self.last_detection_method = None

    def _find(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or self.marker_id not in ids.ravel():
            return None
        index = int(np.flatnonzero(ids.ravel() == self.marker_id)[0])
        return np.asarray(corners[index], dtype=np.float64).reshape(4, 2)

    def detect(self, frame: np.ndarray, K: np.ndarray, dist: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad = self._find(gray)
        self.last_detection_method = 'gray' if quad is not None else None
        if quad is None:
            enhanced = self.clahe.apply(gray)
            quad = self._find(enhanced)
            if quad is not None:self.last_detection_method = 'contrast'
        if quad is None:
            enlarged = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            quad = self._find(enlarged)
            if quad is not None:
                quad /= 2
                self.last_detection_method = 'contrast_2x'
        if quad is None:
            return None
        center = quad.mean(axis=0)
        depth = None
        if self.marker_mm > 0:
            half = self.marker_mm / 2
            object_points = np.array([[-half, half, 0], [half, half, 0],
                                      [half, -half, 0], [-half, -half, 0]], np.float64)
            try:
                ok, _, tvec = cv2.solvePnP(object_points, quad, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error:
                ok, tvec = False, None
            if ok and np.isfinite(tvec).all() and float(tvec[2, 0]) > 0:
                depth = float(tvec[2, 0])
        cv2.polylines(frame, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(frame, tuple(np.rint(center).astype(int)), 5, (0, 0, 255), -1)
        return center, depth


def camera_step(center: np.ndarray, image_size: tuple[int, int], step_mm: float,
                depth_mm: float | None, standoff_mm: float, *, forward: bool,
                center_region_percent: float = 20.) -> np.ndarray | None:
    """Camera optical coordinates: +x right, +y down, +z forward."""
    width, height = image_size
    x, y = float(center[0]) / width, float(center[1]) / height
    if not np.isfinite([x, y]).all() or not 0 <= x <= 1 or not 0 <= y <= 1:
        return None
    if not np.isfinite(center_region_percent) or not 0 < center_region_percent <= 100:
        raise ValueError('中央区域比例需大于0且不超过100')
    low = .5 - center_region_percent / 200
    high = .5 + center_region_percent / 200
    errors = [(low - x, [-step_mm, 0, 0]), (x - high, [step_mm, 0, 0]),
              (low - y, [0, -step_mm, 0]), (y - high, [0, step_mm, 0])]
    deviation, direction = max(errors, key=lambda pair: pair[0])
    if deviation > 0:
        return np.asarray(direction, dtype=float)
    if forward and depth_mm is not None and depth_mm - standoff_mm >= step_mm:
        return np.asarray([0, 0, step_mm], dtype=float)
    return None


def run_servo_loop(camera, detector, K, dist, image_size, *, arm=None, T_gc=None,
                   step_mm=10.0, standoff_mm=200.0, forward=True, max_steps=30,
                   max_travel_mm=60.0, output: Path | None = None):
    records = []
    traveled = 0.0
    stable = 0
    previous = None
    lost = 0
    status = "error"
    if arm is not None:
        arm.cartesian_start_pose()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("相机断开或读取失败，停止控制")
            ensure_image_size(frame, image_size)
            observation = detector.detect(frame, K, dist)
            center = depth = None
            if observation is None:
                lost += 1
                stable = 0
                previous = None
                cv2.putText(frame, "TARGET LOST", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                lost = 0
                center, depth = observation
                stable = stable + 1 if previous is not None and np.linalg.norm(center - previous) < 20 else 1
                previous = center.copy()
            cv2.rectangle(frame, (int(image_size[0] * .4), int(image_size[1] * .4)),
                          (int(image_size[0] * .6), int(image_size[1] * .6)), (255, 255, 0), 1)
            cv2.imshow("Week 6 visual servo - Q/Space/Esc stop", frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27, 32):
                print("操作者停止")
                status = "operator_stop"
                break
            if arm is None:
                continue
            if lost >= 3:
                print("连续丢失目标，停止")
                status = "target_lost"
                break
            if stable < 3:
                continue
            if forward and depth is None:
                # A decoded ID/centre remains useful, but missing depth cannot
                # establish completion of an approach task.
                continue
            delta_camera = camera_step(center, image_size, step_mm, depth, standoff_mm, forward=forward)
            if delta_camera is None:
                print("目标已进入中心区域，或达到预设停止距离")
                status = "target_centered_or_standoff"
                break
            if len(records) >= max_steps or traveled + np.linalg.norm(delta_camera) > max_travel_mm + 1e-6:
                print("达到步数或总行程上限，停止")
                status = "travel_limit"
                break
            raw = arm.positions()
            T_bg = base_gripper_pose(raw, arm.model)
            R_bc = (T_bg @ T_gc)[:3, :3]
            delta_base = R_bc @ delta_camera
            print(f"相机坐标步进 {delta_camera.tolist()} mm；基座坐标请求 {delta_base.round(2).tolist()} mm")
            arm.move_linear(delta_base)
            actual = arm.last_move or {}
            records.append({"center_px": np.asarray(center).round(2).tolist(), "depth_mm": depth,
                            "requested_camera_mm": delta_camera.tolist(),
                            "requested_base_mm": delta_base.round(3).tolist(),
                            "feedback": actual})
            if output is not None:
                write_json(output, {"steps": records, "completed": False})
            traveled += float(np.linalg.norm(delta_camera))
            stable = 0
            previous = None
            for _ in range(4):
                camera.grab()
            if actual.get("projected"):
                print("仅完成可行部分，停止后续运动")
                status = "projected_move"
                break
    finally:
        cv2.destroyAllWindows()
        if output is not None:
            write_json(output, {"steps": records, "completed": status == "target_centered_or_standoff",
                                "status": status,
                                "requested_travel_mm": traveled})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="实机运动；缺省仅预览识别")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="默认按 Windows 设备名选择 GENERAL WEBCAM；也可传数字索引")
    parser.add_argument("--intrinsics", type=Path, default=HERE.parent / "part1_camera_calibration" / "camera_intrinsics.json")
    parser.add_argument("--hand-eye", type=Path, default=HERE.parent / "part2_hand_eye_calibration" / "最终标定结果" / "hand_eye.json")
    parser.add_argument("--advanced-config", type=Path, default=ARM_CODE / "advanced.json")
    parser.add_argument("--hardware-config", type=Path, default=ARM_CODE / "config.json")
    parser.add_argument("--port")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--marker-mm", type=float, default=0, help="ArUco 标记实际边长；运动时必须提供")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--step-mm", type=float, default=10.0)
    parser.add_argument("--standoff-mm", type=float, default=200.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-travel-mm", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=HERE / "run_log.json")
    args = parser.parse_args()
    if not 0 < args.step_mm <= 10 or args.max_steps < 1 or not 0 < args.max_travel_mm <= 100:
        parser.error("步长、次数或总行程超过本示例上限")
    if args.execute and (args.marker_mm <= 0 or args.standoff_mm <= 0):
        parser.error("实机运动必须填写实际 --marker-mm 和正数 --standoff-mm")
    K, dist, size = calibration(args.intrinsics)
    detector = MarkerDetector(args.marker_id, args.marker_mm, args.dictionary)
    camera, selected_index, backend, device_name = open_camera(args.camera, size)
    print(f"使用摄像头“{device_name}”（索引 {selected_index}，{backend}）")
    arm = bus = None
    try:
        if args.execute:
            T_gc = load_hand_eye(args.hand_eye)
            if input("确认末端相机固定、目标固定、运动路径清空、可立即断电；输入 RUN 启动：").strip() != "RUN":
                print("未启动实机运动")
                return
            from advanced import Arm
            from servo_bus import ServoBus, load_config
            bus = ServoBus(load_config(args.hardware_config), args.port)
            arm = Arm(read_json(args.advanced_config), bus)
            arm.model = matching_motion_model(arm.model, read_json(args.hand_eye))
        else:
            T_gc = None
            print("预览模式：不打开串口，不驱动机械臂")
        run_servo_loop(camera, detector, K, dist, size, arm=arm, T_gc=T_gc,
                       step_mm=args.step_mm, standoff_mm=args.standoff_mm,
                       max_steps=args.max_steps, max_travel_mm=args.max_travel_mm,
                       output=args.output if args.execute else None)
    finally:
        camera.release()
        if arm is not None:
            try:
                arm.stop()
            finally:
                bus.close()
        elif bus is not None:
            bus.close()


if __name__ == "__main__":
    main()
