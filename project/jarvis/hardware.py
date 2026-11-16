from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .models import Frame


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
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("无法打开指定相机")
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
                if (frame is not None and frame.sequence > self._delivered
                        and time.monotonic() - frame.captured_at <= self.max_age_s):
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


class HardwareMotion:
    simulated = False

    def __init__(self, config):
        calibration = config.get("calibration", {})
        if config.get("enabled") is not True or calibration.get("verified") is not True:
            raise ValueError("实机未启用或当前装配的标定尚未验证")
        self.matrix = np.asarray(calibration.get("image_to_base_mm"), dtype=float)
        self.workspace = np.asarray(calibration.get("workspace_mm"), dtype=float)
        if (self.matrix.shape != (3, 2) or not np.all(np.isfinite(self.matrix))
                or np.linalg.matrix_rank(self.matrix) != 2):
            raise ValueError("需要经实测验证的 3×2 image_to_base_mm 标定矩阵")
        if (self.workspace.shape != (3, 2) or not np.all(np.isfinite(self.workspace))
                or np.any(self.workspace[:, 0] >= self.workspace[:, 1])):
            raise ValueError("workspace_mm 必须包含三个轴各自经验证的上下界")
        self.max_step = float(config.get("max_step_mm", 3.0))
        self.max_travel = float(config.get("max_travel_mm", 60.0))
        self.timeout_s = float(config.get("command_timeout_s", 15.0))
        if not all(np.isfinite(v) and v > 0 for v in (self.max_step, self.max_travel, self.timeout_s)):
            raise ValueError("运动步长、总行程和超时必须是有限正数")
        if self.max_step > self.max_travel:
            raise ValueError("单步行程不能大于总行程")
        self._stopped = threading.Event()
        self._submit_lock = threading.Lock()
        self._step_lock = threading.Lock()
        self._closed = False
        self.travel_mm = 0.0
        self.last_feedback = None

        from vendor.scs215.advanced import Arm, CommandRunner, load_advanced
        from vendor.scs215.servo_bus import ServoBus, load_config

        root = Path(__file__).resolve().parents[1] / "vendor" / "scs215"
        advanced = load_advanced(root / "advanced.json")
        if advanced.get("motion_calibrated") is not True or advanced["kinematics"].get("calibrated") is not True:
            raise ValueError("SCS215 关节和运动学标定未就绪")
        if self.max_step > min(advanced["max_step_mm"], advanced.get("max_distance_mm", advanced["max_step_mm"])):
            raise ValueError("单步行程超过底层机械臂配置限制")
        hardware = load_config(root / "config.json")
        if [int(item["id"]) for item in hardware["servos"]] != list(range(1, 7)):
            raise ValueError("SCS215 配置必须按顺序包含 ID 1～6")
        self.bus = ServoBus(hardware, config.get("port"), stop_on_close=False)
        try:
            self.arm = Arm(advanced, self.bus)
            self.runner = CommandRunner(self.arm)
            self._check_workspace(self._position())
        except Exception:
            self.bus.close()
            raise

    def _position(self):
        return np.asarray(self.arm.model.forward(self.arm.positions()), dtype=float)

    def _check_workspace(self, point):
        if (point.shape != (3,) or not np.all(np.isfinite(point))
                or np.any(point < self.workspace[:, 0]) or np.any(point > self.workspace[:, 1])):
            raise ValueError("末端模型坐标超出已验证工作区")

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
