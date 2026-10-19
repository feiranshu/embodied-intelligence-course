"""Read-only manual pose and chessboard recorder for eye-in-hand calibration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_utils import DEFAULT_CAMERA, open_camera  # noqa: E402
from common import (ARM_CODE, board_spec, calibration, chessboard_corners,
                    ensure_image_size, read_json, write_image, write_json)  # noqa: E402


HERE = Path(__file__).resolve().parent


def pose_issues(raw: np.ndarray, model) -> list[str]:
    """Report mapping and capture-range diagnostics; these do not gate recording."""
    issues = []
    try:
        model.validate(raw)
    except ValueError as error:
        issues.append(str(error))
        return issues

    base = int(raw[0])
    if not (30 <= base <= 280 or 720 <= base <= 1000):
        issues.append("ID 1 底座读数需在 30–280 或 720–1000，避开接缝附近")
    if not 730 <= raw[1] <= 1007:
        issues.append("ID 2 肩部读数需在 730–1007，避开已记录的 FK 接缝区")
    if not 16 <= raw[2] <= 800:
        issues.append("ID 3 肘部读数需在 16–800")
    if not 700 <= raw[3] <= 990:
        issues.append("ID 4 腕部俯仰读数需在 700–990")
    if not 16 <= raw[4] <= 1007:
        issues.append("ID 5 腕部旋转读数需在 16–1007，避开编码器接缝")
    return issues


def record(args) -> None:
    intrinsics_path = Path(args.intrinsics)
    K, dist, image_size = calibration(intrinsics_path)
    board = read_json(intrinsics_path)
    cols, rows, _ = board_spec(board)
    camera, selected_index, backend, device_name = open_camera(args.camera, image_size)
    print(f"摄像头：{device_name}（索引 {selected_index}，{backend}），棋盘内角点 {cols}×{rows}")
    print("只读模式：仅读取 ID 1–6 的位置；不会移动舵机、使能/关闭扭矩或切换模式。")

    from servo_bus import ServoBus, load_config

    bus = None
    try:
        # Prevent ServoBus.close() from disabling torque on a stationary robot.
        bus = ServoBus(load_config(args.hardware_config), args.port, stop_on_close=False)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        manifest = output / "samples.json"
        samples = read_json(manifest).get("samples", []) if manifest.exists() else []
        saved_names = {item.get("name") for item in samples}
        next_index = 1
        while f"pose_{next_index:03d}" in saved_names:
            next_index += 1

        current_raw = None
        next_read = 0.0
        read_error = None
        print("检测到完整棋盘后按空格记录；Q 或 Esc 退出。每张图和位置写入 samples/。")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头读取失败")
            ensure_image_size(frame, image_size)
            corners = chessboard_corners(frame, cols, rows)

            now = time.monotonic()
            if now >= next_read:
                try:
                    current_raw = np.asarray([bus.read_position(i) for i in range(1, 7)], dtype=int)
                    read_error = None
                except Exception as error:  # keep the camera preview visible for diagnosis
                    current_raw = None
                    read_error = str(error)
                next_read = now + 0.35

            preview = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, (cols, rows), corners, True)

            ready = corners is not None and current_raw is not None and read_error is None
            if read_error:
                status = "SERVO READ ERROR"
                color = (0, 0, 255)
            elif corners is None:
                status = "FULL BOARD NOT FOUND"
                color = (0, 0, 255)
            else:
                status = "READY - SPACE TO RECORD"
                color = (0, 150, 0)

            cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)
            cv2.putText(preview, f"samples: {len(samples)} / 12", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
            if current_raw is not None:
                pose_line = "raw ID1-6: " + "  ".join(str(int(value)) for value in current_raw)
                cv2.putText(preview, pose_line, (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (20, 20, 20), 1, cv2.LINE_AA)
            if read_error:
                cv2.putText(preview, read_error[:105], (12, 112), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 0, 220), 1, cv2.LINE_AA)
            cv2.imshow("Part 2 - read-only pose recorder", preview)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != 32:
                continue
            if not ready:
                print("未保存：需要检测到完整棋盘并能读取六轴位置。")
                continue

            raw = np.asarray([bus.read_position(i) for i in range(1, 7)], dtype=int)
            ok, capture_frame = camera.read()
            if not ok or capture_frame is None:
                print("未保存：采集图像读取失败。")
                continue
            ensure_image_size(capture_frame, image_size)
            capture_corners = chessboard_corners(capture_frame, cols, rows)
            if capture_corners is None:
                print("未保存：最新画面中棋盘角点不完整。")
                continue

            name = f"pose_{next_index:03d}"
            image_name = f"{name}.png"
            write_image(output / image_name, capture_frame)
            samples.append({"name": name, "image": image_name, "raw": raw.tolist()})
            write_json(manifest, {
                "camera_calibration_file": intrinsics_path.name,
                "capture_mode": "manual_read_only_board_detected",
                "board_inner_corners": [cols, rows],
                "samples": samples,
            })
            saved_names.add(name)
            next_index += 1
            print(f"已保存 {name}：raw={raw.tolist()}；累计 {len(samples)} 张")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        if bus is not None:
            bus.close()

    print(f"采集结束：{len(samples)} 张。手眼求解至少需要 12 张有效且姿态有变化的样本。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path,
                        default=HERE.parent / "part1_camera_calibration" / "camera_intrinsics.json")
    parser.add_argument("--advanced-config", type=Path, default=ARM_CODE / "advanced.json")
    parser.add_argument("--hardware-config", type=Path, default=ARM_CODE / "config.json")
    parser.add_argument("--output", type=Path, default=HERE / "samples")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="默认按设备名选择 GENERAL WEBCAM；也可传数字索引")
    parser.add_argument("--port", help="串口号；缺省使用 config.json 的自动识别")
    args = parser.parse_args()
    record(args)


if __name__ == "__main__":
    main()
