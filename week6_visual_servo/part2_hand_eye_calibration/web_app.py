"""Local web interface for SO-101 pose nudging and hand-eye calibration."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FINAL_RESULTS = HERE / "最终标定结果"
WEEK6 = HERE.parent
ARM_CODE = WEEK6.parent / "week4" / "so101_scs215_portable"
if not ARM_CODE.is_dir():
    ARM_CODE = WEEK6.parent / "so101_scs215_portable"
import sys

for entry in (WEEK6, ARM_CODE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from advanced import Arm, load_advanced  # noqa: E402
from camera_utils import DEFAULT_CAMERA, open_camera  # noqa: E402
from board_orientation import orient_board_corners  # noqa: E402
from common import (board_spec, calibration, chessboard_corners, ensure_image_size,
                    read_json, write_image, write_json)  # noqa: E402
from servo_bus import ServoBus, load_config  # noqa: E402


HTML_PATH = HERE / "web_ui.html"
JOINT_NAMES = ["底座 / Base", "肩部 / Shoulder", "肘部 / Elbow",
               "腕部俯仰 / Wrist pitch", "腕部旋转 / Wrist roll", "夹爪 / Gripper"]


def capture_blocked_joints(raw):
    """Check the inclusive capture bands for all six feedback values."""
    if raw is None:
        return []
    blocked = []
    for index, value in enumerate(raw[:6]):
        value = int(value)
        interval = "0～30" if 0 <= value <= 30 else "1000～1022" if 1000 <= value <= 1022 else None
        if interval is not None:
            blocked.append(dict(id=index+1, name=JOINT_NAMES[index], raw=value, interval=interval))
    return blocked


def capture_block_reason(blocked):
    return "；".join(f"ID {item['id']} {item['name']}：读数 {item['raw']}，位于 {item['interval']}"
                    for item in blocked)


class CameraFeed:
    def __init__(self, source: str, image_size: tuple[int, int], cols: int, rows: int):
        self.source = self.name = source
        self.capture = self.index = None
        self.backend = "等待连接"
        self.image_size = image_size
        self.cols, self.rows = cols, rows
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.frame: np.ndarray | None = None
        self.corners: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.stable_frames = 0
        self.error = ""
        self._previous_corners: np.ndarray | None = None
        self._previous_oriented_corners = None
        self._board_anchored = False
        self.orientation_status = {"source": "unconfirmed", "marker_found": False,
                                   "message": "等待完整棋盘和蓝圈"}

    def start(self):
        self.thread = threading.Thread(target=self._run, name="hand-eye-camera", daemon=True)
        self.thread.start()

    def _run(self):
        if sys.platform != "win32":
            return self._capture_frames()
        try:
            import comtypes
        except ImportError:
            return self._capture_frames()
        comtypes.CoInitialize()
        try:
            self._capture_frames()
        finally:
            comtypes.CoUninitialize()

    def _capture_frames(self):
        while not self.stop_event.is_set():
            if self.capture is None:
                try:
                    self.capture, self.index, self.backend, self.name = open_camera(self.source, self.image_size)
                except Exception as error:
                    with self.lock:
                        self.error = str(error)
                    self.stop_event.wait(2)
                    continue
            ok, frame = self.capture.read()
            if not ok or frame is None:
                with self.lock:
                    self.error = "摄像头读取失败"
                    self.corners = self.frame = self.jpeg = None
                    self.stable_frames = 0
                    self._previous_corners = self._previous_oriented_corners = None
                    self._board_anchored = False
                self.capture.release()
                self.capture = None
                self.stop_event.wait(1)
                continue
            try:
                ensure_image_size(frame, self.image_size)
                corners = chessboard_corners(frame, self.cols, self.rows)
                orientation = None
                if corners is not None:
                    corners, orientation = orient_board_corners(
                        frame, corners, self.cols, self.rows, self._previous_oriented_corners,
                        anchored=self._board_anchored)
                    self._previous_oriented_corners = corners.copy()
                    self._board_anchored |= orientation["marker_found"]
            except Exception as error:
                with self.lock:
                    self.error = str(error)
                time.sleep(0.05)
                continue

            with self.lock:
                if corners is None:
                    self.stable_frames = 0
                    self._previous_corners = None
                elif self._previous_corners is not None:
                    movement = float(np.max(np.linalg.norm(
                        corners.reshape(-1, 2) - self._previous_corners.reshape(-1, 2), axis=1
                    )))
                    self.stable_frames = self.stable_frames + 1 if movement <= 1.5 else 0
                    self._previous_corners = corners.copy()
                else:
                    self.stable_frames = 0
                    self._previous_corners = corners.copy()
                self.frame = frame.copy()
                self.corners = None if corners is None else corners.copy()
                if orientation is not None:
                    self.orientation_status = orientation
                self.error = ""

                preview = frame.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(preview, (self.cols, self.rows), corners, True)
                    origin = tuple(np.rint(corners.reshape(-1, 2)[0]).astype(int))
                    cv2.circle(preview, origin, 8, (255, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(preview, "0", (origin[0] + 10, origin[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2, cv2.LINE_AA)
                    marker = self.orientation_status.get("marker_center")
                    if marker is not None:
                        center = tuple(np.rint(marker).astype(int))
                        cv2.circle(preview, center, 12, (0,255,0), 2, cv2.LINE_AA)
                        cv2.line(preview,origin,center,(0,255,0),1,cv2.LINE_AA)
                    label = {'blue_circle':'ORIGIN: BLUE CIRCLE','tracked':'ORIGIN: TRACKED - BLUE NOT FOUND',
                             'unconfirmed':'ORIGIN: UNCONFIRMED - BLUE NOT FOUND'}[self.orientation_status['source']]
                    cv2.putText(preview,label,(12,52),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,180,0) if marker else (0,180,255),1,cv2.LINE_AA)
                status = (f"FULL BOARD - STABLE {self.stable_frames}" if corners is not None
                          else "FULL BOARD NOT FOUND")
                cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0, 150, 0) if corners is not None else (0, 0, 230),
                            2, cv2.LINE_AA)
                encoded, data = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
                self.jpeg = data.tobytes() if encoded else None
            self.stop_event.wait(0.01)

    def snapshot(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy(),
                    None if self.corners is None else self.corners.copy(),
                    self.stable_frames, self.error)

    def jpeg_frame(self):
        with self.lock:
            return self.jpeg

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.capture is not None:
            self.capture.release()


class HandEyeState:
    def __init__(self, args, config, bus: ServoBus, arm: Arm, camera: CameraFeed,
                 cols: int, rows: int):
        self.args = args
        self.config = config
        self.bus = bus
        self.arm = arm
        self.camera = camera
        self.cols, self.rows = cols, rows
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = self.output / "samples.json"
        self.samples = read_json(self.manifest).get("samples", []) if self.manifest.exists() else []
        result_path = Path(args.result)
        self.result: dict | None = read_json(result_path) if result_path.exists() else None
        self.state_lock = threading.RLock()
        self.motion_lock = threading.Lock()
        self.motion_busy = False
        self.motion_active = False
        self.jog_stop_event: threading.Event | None = None
        self.motion_status = ""
        self.motion_error = ""
        self.capture_busy = False
        self.solve_busy = False
        self.solve_status = "idle"
        self.solve_error = ""
        self.state_stop = threading.Event()
        self.raw: list[int] | None = None
        self.read_error = ""
        self.poll_thread: threading.Thread | None = None
        self.motion_thread: threading.Thread | None = None
        self.jog_hold_id = None
        self.jog_heartbeat_at = 0.0
        self.wheel_pause = None

    def start(self):
        self.camera.start()
        self.poll_thread = threading.Thread(target=self._poll_servo_state,
                                            name="hand-eye-servo-state", daemon=True)
        self.poll_thread.start()

    def read_capture_positions(self):
        """Read the same six registers, preserving the ID of a failed transaction."""
        if self.bus is None:
            raise RuntimeError(self.read_error or "等待舵机串口连接")
        reader = getattr(self.bus, "read_position", None)
        if reader is None:
            return self.arm.positions()
        values = []
        with self.bus.lock:
            for joint_id in range(1, 7):
                try:
                    values.append(reader(joint_id))
                except Exception as error:
                    raise RuntimeError(f"ID {joint_id} {JOINT_NAMES[joint_id-1]} 位置反馈读取失败：{error}") from error
        return np.asarray(values, dtype=int)

    def _poll_servo_state(self):
        while not self.state_stop.is_set():
            with self.state_lock:
                jogging = self.motion_active
            if jogging:
                self.state_stop.wait(0.05)
                continue
            try:
                if self.bus is None:
                    bus = ServoBus(load_config(self.args.hardware_config), self.args.servo_port, stop_on_close=False)
                    try:
                        arm = Arm(self.config, bus)
                    except Exception:
                        bus.close()
                        raise
                    with self.state_lock:
                        self.bus, self.arm = bus, arm
                positions = self.read_capture_positions().astype(int).tolist()
                with self.state_lock:
                    self.raw, self.read_error = positions, ""
            except Exception as error:
                with self.state_lock:
                    self.raw, self.read_error = None, str(error)
                    if self.bus is not None and not self.motion_busy and not self.capture_busy:
                        self.bus.close()
                        self.bus = self.arm = None
                self.state_stop.wait(1.5)
            self.state_stop.wait(0.4)

    def status(self):
        _, corners, stable, camera_error = self.camera.snapshot()
        with self.state_lock:
            raw = None if self.raw is None else list(self.raw)
            read_error = self.read_error
            motion_busy, motion_active = self.motion_busy, self.motion_active
            motion_status, motion_error = self.motion_status, self.motion_error
            wheel_pause = getattr(self, "wheel_pause", None)
            capture_busy = self.capture_busy
            solve_busy, solve_status, solve_error = self.solve_busy, self.solve_status, self.solve_error
            result = self.result
            samples = [dict(item) for item in self.samples]
            sample_count = len(samples)
        audit_mode = getattr(self.args, "record_only", False)
        fit_count = sum(item.get("split") == "fit" for item in samples) if audit_mode else sample_count
        blocked_joints = capture_blocked_joints(raw)
        pose_problems = ([capture_block_reason(blocked_joints)] if blocked_joints else []) if raw is not None else [read_error or "等待位置反馈"]
        return {
            "ok": True,
            "camera_name": self.camera.name,
            "camera_index": self.camera.index,
            "camera_backend": self.camera.backend,
            "camera_error": camera_error,
            "board_found": corners is not None,
            "board_stable_frames": stable,
            "board_ready": corners is not None,
            "board_orientation": getattr(self.camera, "orientation_status", {}),
            "raw": raw,
            "read_error": read_error,
            "pose_issues": pose_problems,
            "capture_blocked_joints": blocked_joints,
            "capture_feedback_ready": raw is not None and not blocked_joints,
            "sample_count": sample_count,
            "audit_mode": audit_mode,
            "fit_count": fit_count,
            "session_name": self.output.name,
            "samples": samples,
            "can_capture": (corners is not None and raw is not None and not blocked_joints
                            and not camera_error and not capture_busy and not solve_busy and not motion_busy),
            "can_solve": fit_count >= 12 and not solve_busy and not motion_busy and not capture_busy,
            "motion_busy": motion_busy,
            "motion_active": motion_active,
            "motion_status": motion_status,
            "motion_error": motion_error,
            "wheel_pause": wheel_pause,
            "capture_busy": capture_busy,
            "solve_busy": solve_busy,
            "solve_status": solve_status,
            "solve_error": solve_error,
            "result": result,
            "servo_port": self.bus.port if self.bus is not None else None,
        }

    def capture_pose(self):
        with self.state_lock:
            if self.solve_busy or self.capture_busy or self.motion_busy:
                raise RuntimeError("正在采集或求解，请稍后再试")
            self.capture_busy = True
        try:
            return self._capture_pose_locked()
        finally:
            with self.state_lock:
                self.capture_busy = False

    def delete_samples(self, name=None, *, delete_all=False):
        """Remove selected records and archive their files for recovery."""
        with self.state_lock:
            if self.capture_busy or self.solve_busy:
                raise RuntimeError("正在采集或求解，请完成后再删除记录")
            selected = list(self.samples) if delete_all else [item for item in self.samples if item.get("name") == name]
            if not selected:
                raise ValueError("没有找到要删除的记录")
            remaining = [] if delete_all else [item for item in self.samples if item.get("name") != name]
            root = self.output.resolve()
            images = set()
            referenced = {item.get("image") for item in remaining}
            for item in selected:
                if item.get("image") and item["image"] not in referenced:
                    image = (root / item["image"]).resolve()
                    if image.parent != root or image.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        raise ValueError("记录中的照片路径不在采样目录内，未执行删除")
                    if image.exists():
                        images.add(image)
            archive = root / "deleted_records" / str(time.time_ns())
            archive.mkdir(parents=True)
            write_json(archive / "records.json", {"samples": selected})
            moved = []
            try:
                for image in sorted(images):
                    target = archive / "images" / image.name
                    target.parent.mkdir(exist_ok=True)
                    image.replace(target)
                    moved.append((image, target))
                result_path = Path(self.args.result)
                for artifact in (result_path, result_path.with_suffix(".png")):
                    if artifact.is_file():
                        target = archive / "previous_result" / artifact.name
                        target.parent.mkdir(exist_ok=True)
                        artifact.replace(target)
                        moved.append((artifact, target))
                metadata = read_json(self.manifest) if self.manifest.exists() else {}
                metadata["samples"] = remaining
                write_json(self.manifest, metadata)
            except Exception:
                for original, target in reversed(moved):
                    target.replace(original)
                raise
            self.samples = remaining
            self.result = None
            self.solve_status, self.solve_error = "idle", ""
            return {"deleted_count": len(selected), "sample_count": len(remaining),
                    "archive": str(archive.relative_to(root))}

    def _capture_pose_locked(self):
        frame_before, corners_before, stable_before, camera_error = self.camera.snapshot()
        if camera_error:
            raise RuntimeError(camera_error)
        if frame_before is None or corners_before is None:
            raise RuntimeError("当前画面未检测到完整棋盘")
        record_only = getattr(self.args, "record_only", False)
        if record_only and stable_before < 2:
            raise RuntimeError("请等机械臂和棋盘画面稳定后再采集")

        raw = self.read_capture_positions().astype(int)
        if raw.shape != (6,) or np.any(raw < 0) or np.any(raw > 1023):
            raise RuntimeError("无法读取六个舵机的有效位置反馈")
        with self.state_lock:
            self.raw, self.read_error = raw.tolist(), ""
        blocked_joints = capture_blocked_joints(raw)
        if blocked_joints:
            raise RuntimeError("本次未采集：" + capture_block_reason(blocked_joints))
        frame_after, corners_after, stable_after, camera_error = self.camera.snapshot()
        if camera_error:
            raise RuntimeError(camera_error)
        if frame_after is None or corners_after is None:
            raise RuntimeError("保存时画面中未检测到完整棋盘，请重试")
        raw_after = self.read_capture_positions()
        if capture_blocked_joints(raw_after) or np.any(np.abs(raw_after - raw) > 1):
            raise RuntimeError("采集时关节反馈发生变化或进入禁采区间，请停稳后重试")
        if record_only:
            displacement = np.linalg.norm(corners_after.reshape(-1, 2) - corners_before.reshape(-1, 2), axis=1)
            if stable_after < 2 or np.max(displacement) > 1.5:
                raise RuntimeError("采集时棋盘画面仍在变化，请停稳后重试")

        cols, rows = self.cols, self.rows
        ensure_image_size(frame_after, self.args.image_size)
        if chessboard_corners(frame_after, cols, rows) is None:
            raise RuntimeError("最新画面中没有完整棋盘角点")
        index = 1
        saved = {item.get("name") for item in self.samples}
        while f"pose_{index:03d}" in saved:
            index += 1
        name = f"pose_{index:03d}"
        image_name = f"{name}.png"
        write_image(self.output / image_name, frame_after)
        record = {"name": name, "image": image_name, "raw": raw.tolist(),
                  "ordered_corners": np.asarray(corners_after).reshape(-1,2).tolist(),
                  "corner_order_reference": dict(getattr(self.camera, "orientation_status", {}))}
        if record_only:
            fit_count = sum(item.get("split") == "fit" for item in self.samples)
            record["split"] = "fit" if fit_count < 12 else "holdout"
        with self.state_lock:
            samples = [*self.samples, record]
            write_json(self.manifest, {
                "camera_calibration_file": Path(self.args.intrinsics).name,
                "capture_mode": "hand_eye_accuracy_audit" if record_only else "web_board_detected_read_pose",
                "board_inner_corners": [self.cols, self.rows],
                "samples": samples,
            })
            self.samples = samples
            self.result = None
            self.solve_status = "idle"
            self.solve_error = ""
            count = len(samples)
        return {"name": name, "raw": raw.tolist(), "sample_count": count,
                "image": image_name}

    def jog(self, joint_id: int, direction: int, step_ticks: int, hold_id: str):
        if joint_id not in range(1, 7) or direction not in (-1, 1) or step_ticks not in (4, 8, 16):
            raise ValueError("舵机 ID、方向或步长无效")
        if not isinstance(hold_id, str) or not 1 <= len(hold_id) <= 64:
            raise ValueError("按键请求缺少标识")
        with self.state_lock:
            if self.raw is None or self.bus is None:
                raise RuntimeError(self.read_error or "舵机尚未连接")
            if (self.solve_busy or self.capture_busy or self.motion_busy
                    or not self.motion_lock.acquire(blocking=False)):
                raise RuntimeError("正在移动、采集或求解")
            bus = self.bus
            self.motion_busy = self.motion_active = True
            self.motion_error = ""
            self.motion_status = f"ID {joint_id} 持续调整中；松开按键停止"
            self.jog_stop_event = stop_event = threading.Event()
            self.jog_hold_id = hold_id
            self.jog_heartbeat_at = time.monotonic()

        def held():
            return not stop_event.is_set() and time.monotonic() - self.jog_heartbeat_at < 0.8

        def work():
            moved = False
            error = ""
            message = "已停止"
            restore_force = None
            try:
                mode, lower, upper = bus.mode(joint_id)
                if mode == "wheel":
                    mapping = self.config.get("joint_mapping", {}).get(str(joint_id), {})
                    positive = mapping.get("raw_positive_wheel_direction")
                    if positive is None:
                        positive = mapping.get("wheel_positive")
                        if positive in {"left", "right"} and mapping.get("sign", 1) < 0:
                            positive = "right" if positive == "left" else "left"
                    if positive not in {"left", "right"}:
                        raise RuntimeError(f"ID {joint_id} 的连续模式方向尚无实测记录")
                    drive = positive if direction > 0 else ("right" if positive == "left" else "left")
                    base_speed = int(mapping.get("wheel_speed", 100))
                    selected_speed = min(1000, base_speed + {4: 0, 8: 20, 16: 40}[step_ticks])
                    speed = min(selected_speed, base_speed + (20 if step_ticks == 16 else 0))
                    reference = bus.read_position(joint_id)
                    with self.state_lock:
                        self.wheel_pause = None
                    if held():
                        moved = True
                        bus.run(joint_id, drive, speed, hold_others=True)
                    window_started = last_progress = time.monotonic()
                    progressed = False
                    while held():
                        bus.heartbeat(joint_id)
                        actual = bus.read_position(joint_id)
                        now = time.monotonic()
                        change = (actual - reference + 512) % 1024 - 512
                        if change * direction >= 2:
                            reference, last_progress, progressed = actual, now, True
                        if now - window_started >= 0.16:
                            if progressed and speed < selected_speed and held():
                                speed = min(speed + 20, selected_speed)
                                bus.update_wheel_speed(joint_id, drive, speed)
                            window_started, progressed = now, False
                        with self.state_lock:
                            if self.raw is not None:
                                self.raw[joint_id - 1] = actual
                            if now - last_progress >= 0.25:
                                self.wheel_pause = {"id": joint_id, "raw": actual, "direction": direction,
                                                    "drive": speed, "milliseconds": int((now - last_progress) * 1000)}
                            self.motion_status = f"ID {joint_id} 连续驱动 {speed}；松键停止"
                        stop_event.wait(0.035)
                    return
                if mode != "position":
                    raise RuntimeError(f"ID {joint_id} 返回未知运动模式：{mode}")
                if not 0 <= lower < upper <= 1023:
                    raise RuntimeError(f"ID {joint_id} 的硬件位置限位异常：{lower}～{upper}")
                endpoint = upper if direction > 0 else lower
                current = bus.read_position(joint_id)
                target = max(lower, min(upper, current + direction * step_ticks))
                if not held():
                    return
                force = self.config.get("position_control", {}).get(str(joint_id), {}).get("start_force")
                if force is not None:
                    initial_force = bus.read_start_force(joint_id)
                    if force != initial_force:
                        restore_force = initial_force
                        bus.set_start_force(joint_id, force)
                if not held():
                    return
                moved = True
                bus.move_position(joint_id, target, 0, int(self.config.get("speed", 100)))
                while held():
                    bus.heartbeat(joint_id)
                    if stop_event.wait(0.04):
                        break
                    actual = bus.read_position(joint_id)
                    with self.state_lock:
                        if self.raw is not None:
                            self.raw[joint_id - 1] = actual
                    # Let small targets advance through deadband while keeping one nearby goal.
                    lead = actual + direction * step_ticks * 3
                    next_target = target + direction * step_ticks
                    candidate = min(next_target, lead) if direction > 0 else max(next_target, lead)
                    candidate = max(lower, min(upper, candidate))
                    if not held():
                        break
                    if (candidate - target) * direction > 0:
                        bus.update_position_goal(joint_id, candidate)
                        target = candidate
                    if target == endpoint and abs(actual - endpoint) <= 2:
                        message = f"ID {joint_id} 已到硬件位置端点 {endpoint}"
                        with self.state_lock:
                            self.motion_status = message + "，松键释放"
            except Exception as exc:
                error = str(exc)
            finally:
                stop_error = ""
                try:
                    failures = bus.stop(joint_id) if moved else []
                    stop_error = "；".join(failures)
                except Exception as exc:
                    stop_error = str(exc)
                restore_error = ""
                if restore_force is not None:
                    try:
                        bus.set_start_force(joint_id, restore_force)
                    except Exception as exc:
                        restore_error = f"恢复启动驱动力失败：{exc}"
                with self.state_lock:
                    self.motion_error = "；".join(filter(None, (error, stop_error, restore_error)))
                    self.motion_status = "停止反馈异常" if stop_error else "已停止" if error else message
                    self.motion_busy = self.motion_active = False
                    self.jog_stop_event = self.jog_hold_id = None
                    self.motion_lock.release()

        self.motion_thread = threading.Thread(target=work, name=f"servo-jog-{joint_id}", daemon=True)
        try:
            self.motion_thread.start()
        except Exception:
            with self.state_lock:
                self.motion_busy = self.motion_active = False
                self.jog_stop_event = self.jog_hold_id = None
                self.motion_lock.release()
            raise
        return {"accepted": True, "joint_id": joint_id, "step_ticks": step_ticks}

    def jog_heartbeat(self, hold_id: str):
        with self.state_lock:
            active = self.jog_hold_id == hold_id and self.jog_stop_event is not None
            if active and not self.jog_stop_event.is_set():
                self.jog_heartbeat_at = time.monotonic()
                return {"active": True}
        return {"active": False}

    def stop_motion(self, hold_id=None):
        with self.state_lock:
            if self.jog_stop_event is None or (hold_id is not None and hold_id != self.jog_hold_id):
                return {"accepted": False, "message": "当前没有运行中的动作"}
            self.jog_stop_event.set()
            worker = self.motion_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=0.2)
        with self.state_lock:
            return {"accepted": True, "active": self.motion_active,
                    "message": self.motion_error or self.motion_status}

    def solve(self, measured_origin: list[float] | None):
        with self.state_lock:
            if (self.motion_busy or self.capture_busy or self.solve_busy
                    or not self.motion_lock.acquire(blocking=False)):
                raise RuntimeError("机械臂正在移动、采集或已有求解任务")
            audit_mode = getattr(self.args, "record_only", False)
            fit_count = sum(item.get("split") == "fit" for item in self.samples) if audit_mode else len(self.samples)
            if fit_count < 12:
                self.motion_lock.release()
                raise RuntimeError(f"当前只有 {fit_count} 张拟合样本；至少需要 12 张")
            self.solve_busy = True
            self.solve_status = "solving"
            self.solve_error = ""

        def work():
            try:
                from part2_hand_eye_calibration.hand_eye import solve as solve_hand_eye
                from argparse import Namespace

                solve_hand_eye(Namespace(
                    intrinsics=Path(self.args.intrinsics),
                    samples=self.manifest,
                    advanced_config=Path(self.args.advanced_config),
                    board_origin_mm=measured_origin,
                    output=Path(self.args.result),
                    fit_only=audit_mode,
                ))
                result = read_json(self.args.result)
                with self.state_lock:
                    self.result = result
                    self.solve_status = "complete"
            except Exception as error:
                with self.state_lock:
                    self.solve_error = str(error)
                    self.solve_status = "failed"
            finally:
                with self.state_lock:
                    self.solve_busy = False
                self.motion_lock.release()

        worker = threading.Thread(target=work, name="hand-eye-solve", daemon=True)
        try:
            worker.start()
        except Exception:
            with self.state_lock:
                self.solve_busy = False
            self.motion_lock.release()
            raise
        return {"accepted": True, "status": "solving"}

    def close(self):
        self.state_stop.set()
        if self.poll_thread is not None:
            self.poll_thread.join(timeout=1)
        self.camera.stop()
        if self.motion_active:
            try:
                self.stop_motion()
            except Exception as error:
                print(f"动作停止失败：{error}")
        if self.motion_thread is not None:
            self.motion_thread.join(timeout=5)
        if self.bus is not None:
            self.bus.close()


class BoardPreview:
    def __init__(self, camera: CameraFeed):
        self.camera = camera

    def status(self):
        camera = self.camera
        with camera.lock:
            found = camera.corners is not None and not camera.error
            return dict(camera=camera.name, image_size=camera.image_size,
                        pattern=[camera.cols, camera.rows], ready=camera.jpeg is not None,
                        corners=len(camera.corners) if found else 0, error=camera.error)


class HandEyeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, app: HandEyeState | BoardPreview):
        self.app = app
        super().__init__(address, handler)


class RequestHandler(BaseHTTPRequestHandler):
    server: HandEyeHTTPServer
    html_path = HTML_PATH

    def log_message(self, fmt, *args):
        print(f"Web {self.address_string()}: {fmt % args}")

    def _send_json(self, payload: dict, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 4096:
            raise ValueError("请求内容无效")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            data = self.html_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/state":
            try:
                self._send_json(self.server.app.status())
            except Exception as error:
                self._send_json({"ok": False, "error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/video.mjpg":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = self.server.app.camera.jpeg_frame()
                    if frame is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.06)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
        else:
            self._send_json({"ok": False, "error": "页面不存在"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path in ("/api/jog", "/api/jog-heartbeat", "/api/stop") and self.headers.get_content_type() != "application/json":
                raise ValueError("运动控制只接受 JSON 请求")
            data = self._read_json()
            if path == "/api/jog":
                result = self.server.app.jog(int(data["id"]), int(data["direction"]), int(data["step_ticks"]), data["hold_id"])
            elif path == "/api/jog-heartbeat":
                result = self.server.app.jog_heartbeat(data["hold_id"])
            elif path == "/api/stop":
                result = self.server.app.stop_motion(data.get("hold_id"))
            elif path == "/api/capture":
                result = self.server.app.capture_pose()
            elif path == "/api/delete-samples":
                result = self.server.app.delete_samples(data.get("name"), delete_all=data.get("all") is True)
            elif path == "/api/solve":
                measured = data.get("board_origin_mm")
                if measured is None:
                    measured = self.server.app.args.board_origin_mm
                if measured is not None:
                    measured = [float(value) for value in measured]
                    if len(measured) != 3 or not np.isfinite(measured).all():
                        raise ValueError("棋盘原点坐标需要三个有限数值，单位 mm")
                result = self.server.app.solve(measured)
            else:
                self._send_json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json({"ok": True, **result}, HTTPStatus.ACCEPTED)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"ok": False, "error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


class BoardPreviewHandler(RequestHandler):
    html_path = HERE / "board_preview.html"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        self._send_json({"ok": False, "error": "棋盘预览仅提供实时画面"}, HTTPStatus.METHOD_NOT_ALLOWED)


class BoardRecording(HandEyeState):
    def status(self):
        state = super().status()
        return dict(camera=state["camera_name"], image_size=self.args.image_size,
                    pattern=[self.cols, self.rows], ready=self.camera.jpeg_frame() is not None,
                    corners=self.cols * self.rows if state["board_found"] else 0,
                    error=state["camera_error"], recording=True,
                    stable_frames=state["board_stable_frames"], raw=state["raw"],
                    pose_issues=state["pose_issues"], sample_count=state["sample_count"],
                    can_capture=state["can_capture"] and state["board_stable_frames"] >= 2)

class BoardRecordingHandler(BoardPreviewHandler):
    def do_POST(self):
        if urlparse(self.path).path == "/api/capture":
            try:
                self._send_json({"ok": True, **self.server.app.capture_pose()}, HTTPStatus.CREATED)
            except (RuntimeError, ValueError, OSError) as error:
                self._send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        return super().do_POST()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, default=FINAL_RESULTS / "camera_intrinsics.json")
    parser.add_argument("--advanced-config", type=Path, default=FINAL_RESULTS / "advanced.json")
    parser.add_argument("--hardware-config", type=Path, default=ARM_CODE / "config.json")
    parser.add_argument("--servo-port", help="舵机串口；缺省自动识别")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="摄像头设备名或 OpenCV 数字索引")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="只预览棋盘角点，不连接舵机")
    mode.add_argument("--record-only", action="store_true", help="只读关节并采集核验样本，不发送运动指令")
    mode.add_argument("--record-with-keyboard", action="store_true", help="核验采样与按住方向键连续控制")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=FINAL_RESULTS)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--board-origin-mm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="棋盘原点相对基座的实测坐标；提供后报告各轴真实误差")
    args = parser.parse_args()
    args.keyboard_jog = not (args.preview or args.record_only)
    args.record_only = not args.preview
    args.result = args.result or args.output / "hand_eye.json"
    return args


def main():
    args = parse_args()
    _, _, args.image_size = calibration(args.intrinsics)
    intrinsics = read_json(args.intrinsics)
    cols, rows, _ = board_spec(intrinsics)
    camera = CameraFeed(args.camera, args.image_size, cols, rows)
    if args.preview:
        try:
            with HandEyeHTTPServer(("127.0.0.1", args.http_port), BoardPreviewHandler, BoardPreview(camera)) as server:
                camera.start()
                print(f"棋盘预览：http://127.0.0.1:{args.http_port}/（{camera.name}）", flush=True)
                server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()
        return
    bus = None
    app = None
    try:
        config = load_advanced(args.advanced_config)
        read_only = args.record_only and not args.keyboard_jog
        app_class, handler = (BoardRecording, BoardRecordingHandler) if read_only else (HandEyeState, RequestHandler)
        app = app_class(args, config, None, None, camera, cols, rows)
        reviewed = next((item for item in reversed(app.samples)
                         if item.get("corner_order_reference", {}).get("source") == "manual_review"), None)
        if reviewed is not None:
            camera._previous_oriented_corners = np.asarray(reviewed["ordered_corners"], dtype=np.float32).reshape(-1, 1, 2)
            camera._board_anchored = True
        server = HandEyeHTTPServer(("127.0.0.1", args.http_port), handler, app)
        app.start()
    except Exception:
        if app is not None:
            app.close()
        else:
            camera.stop()
            if bus is not None:
                bus.close()
        raise

    url = f"http://127.0.0.1:{args.http_port}/"
    print(f"相机：{camera.source}；舵机串口：等待连接")
    print(f"手眼标定页面：{url}")
    if args.record_only:
        print(f"核验样本目录：{args.output}。键盘连续控制：{'开启' if args.keyboard_jog else '关闭'}。Ctrl+C 退出。")
    else:
        print("网页只在本机开放。按住关节控制键时舵机会连续运动，松开即停止。Ctrl+C 退出。")
    try:
        if not args.record_only:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
