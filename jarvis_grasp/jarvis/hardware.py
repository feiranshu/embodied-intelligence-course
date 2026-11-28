from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from .grasp import GRIPPER_MAX_RAW, GRIPPER_MIN_RAW
from .models import Frame


def resolve_camera_source(source) -> tuple[int, int | None, str]:
    """Resolve an integer index or Windows DirectShow camera name (e.g. 'GENERAL WEBCAM')."""
    import os
    import cv2

    if isinstance(source, int) and not isinstance(source, bool):
        backend = cv2.CAP_DSHOW if os.name == "nt" else None
        return int(source), backend, f"Camera {source}"
    text = str(source).strip()
    if text.isdigit():
        idx = int(text)
        backend = cv2.CAP_DSHOW if os.name == "nt" else None
        return idx, backend, f"Camera {idx}"
    if os.name == "nt":
        names: list[str] = []
        try:
            from pygrabber.dshow_graph import FilterGraph

            names = list(FilterGraph().get_input_devices())
        except Exception:
            import subprocess

            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_PnPEntity -Filter \"PNPClass='Camera' OR PNPClass='Image'\" | Where-Object { $_.Status -eq 'OK' } | Select-Object -ExpandProperty Name",
                ],
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            names = [line.strip() for line in out.splitlines() if line.strip()]
        query = text.casefold()
        matches = [(i, n) for i, n in enumerate(names) if query == n.casefold()] or [
            (i, n) for i, n in enumerate(names) if query in n.casefold()
        ]
        if len(matches) == 1:
            return matches[0][0], cv2.CAP_DSHOW, matches[0][1]
        raise RuntimeError(f"无法唯一匹配摄像头“{source}”，当前设备：{names}")
    raise ValueError(f"无效的摄像头标识：{source}")


class Camera:
    def __init__(self, source=0, width=640, height=480, timeout_s=2.0, max_age_s=0.5):
        import cv2

        if not all(np.isfinite(v) and v > 0 for v in (timeout_s, max_age_s, width, height)):
            raise ValueError("相机尺寸和超时必须是正数")
        self.timeout_s = float(timeout_s)
        self.max_age_s = float(max_age_s)
        self._condition = threading.Condition()
        self._closed = threading.Event()
        self._frame = None
        self._sequence = 0
        self._delivered = 0
        self._error = None
        index, backend, device_name = resolve_camera_source(source)
        self.device_name = device_name
        self.capture = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not self.capture.isOpened() and backend is not None:
            self.capture.release()
            self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开指定相机：{source}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self._thread = threading.Thread(target=self._read_loop, name="jarvis-camera", daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._closed.is_set():
            try:
                ok, image = self.capture.read()
                if ok and image is not None:
                    with self._condition:
                        self._sequence += 1
                        self._frame = Frame(image.copy(), self._sequence, time.monotonic())
                        self._error = None
                        self._condition.notify_all()
                    continue
                self._error = "相机没有返回新画面"
            except Exception as error:
                self._error = str(error)
            self._closed.wait(0.02)

    def read(self):
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            while not self._closed.is_set():
                frame = self._frame
                if (
                    frame is not None
                    and frame.sequence > self._delivered
                    and time.monotonic() - frame.captured_at <= self.max_age_s
                ):
                    self._delivered = frame.sequence
                    return Frame(frame.image.copy(), frame.sequence, frame.captured_at)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(self._error or "等待新相机画面超时")
                self._condition.wait(remaining)
        raise RuntimeError("相机已经关闭")

    def close(self):
        self._closed.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=self.timeout_s)
        self.capture.release()


def _base_gripper_pose(raw, model) -> np.ndarray:
    """Compute 4x4 T_base_gripper from URDF joint chain (millimetres)."""
    angles = model.angles(raw)
    T = np.eye(4, dtype=np.float64)
    for index, origin in enumerate(model.chain):
        T = T @ origin
        if index < 5:
            c, s = math.cos(float(angles[index])), math.sin(float(angles[index]))
            rot = np.eye(4, dtype=np.float64)
            rot[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
            T = T @ rot
    return T


class HardwareMotion:
    simulated = False

    def __init__(self, config):
        calibration = config.get("calibration", {})
        if config.get("enabled") is not True or calibration.get("verified") is not True:
            raise ValueError("实机未启用或当前装配的标定尚未验证")
        self.matrix = np.asarray(calibration.get("image_to_base_mm"), dtype=float)
        self.workspace = np.asarray(calibration.get("workspace_mm"), dtype=float)
        if (
            self.matrix.shape != (3, 2)
            or not np.all(np.isfinite(self.matrix))
            or np.linalg.matrix_rank(self.matrix) != 2
        ):
            raise ValueError("需要经实测验证的 3×2 image_to_base_mm 标定矩阵")
        if (
            self.workspace.shape != (3, 2)
            or not np.all(np.isfinite(self.workspace))
            or np.any(self.workspace[:, 0] >= self.workspace[:, 1])
        ):
            raise ValueError("workspace_mm 必须包含三个轴各自经验证的上下界")
        self.max_step = float(config.get("max_step_mm", 3.0))
        self.max_travel = float(config.get("max_travel_mm", 60.0))
        self.timeout_s = float(config.get("command_timeout_s", 15.0))
        if not all(np.isfinite(v) and v > 0 for v in (self.max_step, self.max_travel, self.timeout_s)):
            raise ValueError("运动步长、总行程和超时必须是有限正数")
        if self.max_step > self.max_travel:
            raise ValueError("单步行程不能大于总行程")

        # Optional grasp configuration (separately gated by hardware.grasp.verified)
        grasp_cfg = config.get("grasp") or {}
        self.grasp_verified = grasp_cfg.get("verified") is True
        self.table_z_mm: float | None = None
        self.max_grasp_step = self.max_step
        self.max_grasp_travel = self.max_travel
        self.gripper_open_raw = 430
        self.gripper_close_raw = 278
        self.gripper_held_range_raw = (298, 415)
        if self.grasp_verified:
            self._validate_grasp_hardware(grasp_cfg)

        self._stopped = threading.Event()
        self._submit_lock = threading.Lock()
        self._step_lock = threading.Lock()
        self._closed = False
        self.travel_mm = 0.0
        self.grasp_travel_mm = 0.0
        self.last_feedback = None
        self._gripper_raw = self.gripper_close_raw
        self._gripper_load = 0
        self._holding = False

        # Optional camera intrinsics + hand-eye calibration for 3D localization
        self._camera_matrix = None
        self._dist_coeffs = None
        self._T_gripper_camera = None
        self._hand_eye_data = None
        self._load_optional_hand_eye(calibration)

        from vendor.scs215.advanced import Arm, CommandRunner, load_advanced
        from vendor.scs215.servo_bus import ServoBus, load_config

        root = Path(__file__).resolve().parents[1] / "vendor" / "scs215"
        advanced = load_advanced(root / "advanced.json")
        if advanced.get("motion_calibrated") is not True or advanced["kinematics"].get("calibrated") is not True:
            raise ValueError("SCS215 关节和运动学标定未就绪")
        max_hw_step = min(advanced["max_step_mm"], advanced.get("max_distance_mm", advanced["max_step_mm"]))
        if self.max_step > max_hw_step or self.max_grasp_step > max_hw_step:
            raise ValueError("单步行程超过底层机械臂配置限制")
        hardware = load_config(root / "config.json")
        if [int(item["id"]) for item in hardware["servos"]] != list(range(1, 7)):
            raise ValueError("SCS215 配置必须按顺序包含 ID 1～6")
        self.bus = ServoBus(hardware, config.get("port"), stop_on_close=False)
        try:
            self.arm = Arm(advanced, self.bus)
            mapping = (self._hand_eye_data or {}).get("calibrated_joint_mapping")
            if mapping is not None and hasattr(self.arm.model, "with_joint_calibration"):
                self.arm.model = self.arm.model.with_joint_calibration(mapping)
            self.runner = CommandRunner(self.arm)
            self._check_workspace(self._position())
        except Exception:
            self.bus.close()
            raise

    def _validate_grasp_hardware(self, grasp_cfg: dict):
        table_z = grasp_cfg.get("table_z_mm")
        if not isinstance(table_z, (int, float)) or isinstance(table_z, bool) or not math.isfinite(table_z):
            raise ValueError("实机抓取需要有限数值 table_z_mm（桌面防碰撞平面）")
        self.table_z_mm = float(table_z)
        if self.table_z_mm < self.workspace[2, 0] - 1e-6 or self.table_z_mm >= self.workspace[2, 1]:
            raise ValueError("table_z_mm 必须位于 workspace_mm 的 Z 轴已验证范围内")
        self.max_grasp_step = float(grasp_cfg.get("max_grasp_step_mm", self.max_step))
        self.max_grasp_travel = float(grasp_cfg.get("max_grasp_travel_mm", 120.0))
        if not all(np.isfinite(v) and v > 0 for v in (self.max_grasp_step, self.max_grasp_travel)):
            raise ValueError("实机抓取步长和总行程必须是有限正数")
        if self.max_grasp_step > self.max_grasp_travel:
            raise ValueError("实机抓取单步不能大于总行程")
        open_raw = int(grasp_cfg.get("gripper_open_raw", 430))
        close_raw = int(grasp_cfg.get("gripper_close_raw", 278))
        held = grasp_cfg.get("gripper_held_range_raw", [298, 415])
        if not GRIPPER_MIN_RAW <= close_raw < open_raw <= GRIPPER_MAX_RAW:
            raise ValueError("实机夹爪开合原始读数超出 SCS215 安全限位")
        if not (isinstance(held, (list, tuple)) and len(held) == 2 and close_raw < held[0] < held[1] < open_raw):
            raise ValueError("实机夹持判定区间 gripper_held_range_raw 无效")
        self.gripper_open_raw = open_raw
        self.gripper_close_raw = close_raw
        self.gripper_held_range_raw = (int(held[0]), int(held[1]))

    def _load_optional_hand_eye(self, calibration: dict):
        project_root = Path(__file__).resolve().parents[1]
        intrinsics_path = calibration.get("intrinsics_path")
        hand_eye_path = calibration.get("hand_eye_path")
        if intrinsics_path:
            path = Path(intrinsics_path)
            if not path.is_absolute():
                path = project_root / path
            data = json.loads(path.read_text(encoding="utf-8"))
            K = np.asarray(data["camera_matrix"], dtype=np.float64)
            dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
            if K.shape != (3, 3) or not np.isfinite(K).all():
                raise ValueError("相机内参文件格式无效")
            self._camera_matrix = K
            self._dist_coeffs = dist
        if hand_eye_path:
            path = Path(hand_eye_path)
            if not path.is_absolute():
                path = project_root / path
            data = json.loads(path.read_text(encoding="utf-8"))
            T = np.asarray(data["T_gripper_camera"], dtype=np.float64)
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError("手眼矩阵 T_gripper_camera 格式无效")
            self._T_gripper_camera = T
            self._hand_eye_data = data

    def camera_geometry(self):
        """Return (K, dist, T_base_camera) when intrinsics & hand-eye are loaded, else (None, None, None)."""
        if self._T_gripper_camera is None or not getattr(self.arm.model, "chain", None):
            return self._camera_matrix, self._dist_coeffs, None
        T_bg = _base_gripper_pose(self.arm.positions(), self.arm.model)
        return self._camera_matrix, self._dist_coeffs, T_bg @ self._T_gripper_camera

    def _position(self):
        return np.asarray(self.arm.model.forward(self.arm.positions()), dtype=float)

    def position_mm(self) -> np.ndarray:
        return self._position()

    def _check_workspace(self, point, *, check_table: bool = False):
        if (
            point.shape != (3,)
            or not np.all(np.isfinite(point))
            or np.any(point < self.workspace[:, 0])
            or np.any(point > self.workspace[:, 1])
        ):
            raise ValueError("末端模型坐标超出已验证工作区")
        if check_table and self.table_z_mm is not None and float(point[2]) < self.table_z_mm - 1e-6:
            raise ValueError(f"末端高度 {point[2]:.1f} mm 低于桌面安全平面 {self.table_z_mm:.1f} mm")

    def _opening_from_raw(self, raw: int) -> float:
        span = max(1, self.gripper_open_raw - self.gripper_close_raw)
        return round(float(np.clip((raw - self.gripper_close_raw) / span, 0.0, 1.0)), 3)

    def gripper_state(self) -> dict:
        raw = self._gripper_raw
        load = self._gripper_load
        holding = self._holding
        try:
            positions = self.arm.positions()
            if len(positions) >= 6:
                raw = int(positions[5])
                self._gripper_raw = raw
                if holding:
                    lo, hi = self.gripper_held_range_raw
                    holding = lo <= raw <= hi
                    self._holding = holding
        except Exception:
            pass
        return {
            "raw": int(raw),
            "opening": self._opening_from_raw(raw),
            "load": int(load),
            "holding": bool(holding),
            "held_object": None,
        }

    def step(self, dx, dy):
        if not self._step_lock.acquire(blocking=False):
            raise RuntimeError("已有运动命令正在执行")
        try:
            if self._stopped.is_set() or self._closed:
                raise InterruptedError("运动已锁存停止，重新启动前不能恢复")
            error = np.asarray([dx, dy], dtype=float)
            if not np.all(np.isfinite(error)) or np.any(np.abs(error) > 1):
                raise ValueError("图像误差必须是有限归一化坐标")
            delta = self.matrix @ error
            distance = float(np.linalg.norm(delta))
            if distance == 0:
                return
            delta *= min(1.0, self.max_step / distance)
            distance = float(np.linalg.norm(delta))
            if self.travel_mm + distance > self.max_travel:
                raise ValueError("累计运动已达到本次启动的总行程限制")
            before = self._position()
            self._check_workspace(before)
            self._check_workspace(before + delta)
            with self._submit_lock:
                if self._stopped.is_set():
                    raise InterruptedError("运动已停止")
                if not self.runner.submit(("move", delta)):
                    raise RuntimeError("底层控制器拒绝运动命令")
            self.runner.thread.join(timeout=self.timeout_s)
            if self.runner.busy:
                raise TimeoutError("底层运动命令超时")
            self.runner.wait()
            if self._stopped.is_set():
                raise InterruptedError("运动已停止")
            after = self._position()
            self.travel_mm += max(distance, float(np.linalg.norm(after - before)))
            self._check_workspace(after)
            self.last_feedback = self.arm.last_move
            if self.last_feedback and self.last_feedback.get("projected"):
                raise RuntimeError("只完成了部分可行位移，本次居中任务终止")
        except Exception as error:
            try:
                self.stop()
            except Exception as stop_error:
                raise RuntimeError(f"{error}；停止回读失败：{stop_error}") from error
            raise
        finally:
            self._step_lock.release()

    def move_delta(self, delta_mm):
        """Execute a workspace- and table-checked 3D Cartesian displacement in bounded sub-steps."""
        if not self.grasp_verified:
            raise ValueError("实机抓取未解锁：hardware.grasp.verified 必须为 true")
        if not self._step_lock.acquire(blocking=False):
            raise RuntimeError("已有运动命令正在执行")
        try:
            if self._stopped.is_set() or self._closed:
                raise InterruptedError("运动已锁存停止，重新启动前不能恢复")
            delta = np.asarray(delta_mm, dtype=float)
            if delta.shape != (3,) or not np.all(np.isfinite(delta)):
                raise ValueError("末端位移必须是 3 维有限毫米向量")
            total_distance = float(np.linalg.norm(delta))
            if total_distance == 0:
                return
            if self.grasp_travel_mm + total_distance > self.max_grasp_travel:
                raise ValueError("抓取累计行程超出 max_grasp_travel_mm 限制")
            sub_count = max(1, math.ceil(total_distance / self.max_grasp_step))
            sub_delta = delta / sub_count
            for _ in range(sub_count):
                if self._stopped.is_set() or self._closed:
                    raise InterruptedError("运动已停止")
                before = self._position()
                self._check_workspace(before, check_table=True)
                self._check_workspace(before + sub_delta, check_table=True)
                with self._submit_lock:
                    if self._stopped.is_set():
                        raise InterruptedError("运动已停止")
                    if not self.runner.submit(("move", sub_delta)):
                        raise RuntimeError("底层控制器拒绝末端移动命令")
                self.runner.thread.join(timeout=self.timeout_s)
                if self.runner.busy:
                    raise TimeoutError("底层末端移动命令超时")
                self.runner.wait()
                if self._stopped.is_set():
                    raise InterruptedError("运动已停止")
                after = self._position()
                step_dist = float(np.linalg.norm(sub_delta))
                self.grasp_travel_mm += max(step_dist, float(np.linalg.norm(after - before)))
                self._check_workspace(after, check_table=True)
                self.last_feedback = self.arm.last_move
                if self.last_feedback and self.last_feedback.get("projected"):
                    raise RuntimeError("抓取移动仅完成部分可行位移，任务终止")
        except Exception as error:
            try:
                self.stop()
            except Exception as stop_error:
                raise RuntimeError(f"{error}；停止回读失败：{stop_error}") from error
            raise
        finally:
            self._step_lock.release()

    def set_gripper(
        self,
        target_raw: int,
        *,
        expect_object: bool = False,
        held_range: tuple[int, int] | None = None,
        max_load: int = 350,
        hold_bias: int = 4,
        speed: int = 180,
        duration_s: float = 0.8,
    ) -> dict:
        """Open or close SCS215 ID 6 gripper with contact-load & width-based object detection."""
        if not self.grasp_verified:
            raise ValueError("实机抓取未解锁：hardware.grasp.verified 必须为 true")
        if not self._step_lock.acquire(blocking=False):
            raise RuntimeError("已有运动命令正在执行")
        try:
            if self._stopped.is_set() or self._closed:
                raise InterruptedError("运动已锁存停止，重新启动前不能恢复")
            target_raw = int(target_raw)
            if not GRIPPER_MIN_RAW <= target_raw <= GRIPPER_MAX_RAW:
                raise ValueError(f"夹爪目标值 {target_raw} 超出安全限位 [{GRIPPER_MIN_RAW}, {GRIPPER_MAX_RAW}]")
            lo, hi = held_range if held_range is not None else self.gripper_held_range_raw

            if expect_object and hasattr(self.bus, "move_position") and hasattr(self.bus, "read_position"):
                actual_raw, load_mag = self._close_gripper_with_contact(
                    target_raw, max_load=max_load, hold_bias=hold_bias, speed=speed, duration_s=duration_s
                )
            else:
                current = np.asarray(self.arm.positions(), dtype=int)
                if current.shape != (6,):
                    raise ValueError("需要 6 关节位置反馈以控制夹爪")
                target_pose = current.copy()
                target_pose[5] = target_raw
                with self._submit_lock:
                    if self._stopped.is_set():
                        raise InterruptedError("运动已停止")
                    if not self.runner.submit(("pose_raw", target_pose)):
                        raise RuntimeError("底层控制器拒绝夹爪命令")
                self.runner.thread.join(timeout=self.timeout_s)
                if self.runner.busy:
                    raise TimeoutError("夹爪运动超时")
                self.runner.wait()
                if self._stopped.is_set():
                    raise InterruptedError("运动已停止")
                actual_raw = int(self.arm.positions()[5])
                load_mag = 0

            gripped = bool(expect_object and lo <= actual_raw <= hi)
            self._gripper_raw = actual_raw
            self._gripper_load = load_mag
            self._holding = gripped
            return {
                "raw": actual_raw,
                "load": load_mag,
                "gripped": gripped,
                "opening": self._opening_from_raw(actual_raw),
                "held_object": None,
            }
        except Exception as error:
            try:
                self.stop()
            except Exception as stop_error:
                raise RuntimeError(f"{error}；停止回读失败：{stop_error}") from error
            raise
        finally:
            self._step_lock.release()

    def _close_gripper_with_contact(
        self,
        target_raw: int,
        *,
        max_load: int,
        hold_bias: int,
        speed: int,
        duration_s: float,
    ) -> tuple[int, int]:
        """Close gripper ID 6 while monitoring load & stall so jaws grip gently without overheating."""
        from vendor.scs215.servo_bus import GOAL_POSITION

        duration_ms = max(150, round(float(duration_s) * 1000))
        with self._submit_lock:
            if self._stopped.is_set():
                raise InterruptedError("运动已停止")
            self.bus.move_position(6, int(target_raw), duration_ms, int(speed))
        deadline = time.monotonic() + max(2.0, float(duration_s) + 1.5)
        stalled = 0
        previous = None
        actual = int(target_raw)
        load_mag = 0
        while True:
            if self._stopped.is_set():
                raise InterruptedError("运动已停止")
            time.sleep(0.03)
            with self.bus.lock:
                self.bus.heartbeat(6)
                actual = int(self.bus.read_position(6))
                load_raw = int(self.bus._read_u16(6, 60)) if hasattr(self.bus, "_read_u16") else 0
                load_mag = load_raw & 1023
                if previous is not None and abs(actual - previous) <= 1:
                    stalled += 1
                else:
                    stalled = 0
                previous = actual
                arrived = abs(actual - target_raw) <= 4
                contact = load_mag >= max_load or stalled >= 6
                if arrived or contact or time.monotonic() >= deadline:
                    hold_raw = max(GRIPPER_MIN_RAW, min(GRIPPER_MAX_RAW, max(int(target_raw), actual - int(hold_bias))))
                    if hasattr(self.bus, "_write"):
                        self.bus._write(6, GOAL_POSITION, [hold_raw >> 8, hold_raw & 255])
                    self.bus.finish_position(6)
                    return actual, load_mag

    def stop(self):
        self._stopped.set()
        with self._submit_lock:
            self.runner.stop()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
            thread = self.runner.thread
            if thread is not None:
                thread.join(timeout=self.timeout_s)
                if thread.is_alive():
                    raise RuntimeError("停止后控制线程未退出，不能确认实机已经停止")
        finally:
            self.bus.close()
