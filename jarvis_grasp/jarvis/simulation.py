import math
import threading
import time

import cv2
import numpy as np

from .grasp import GRIPPER_MAX_RAW, GRIPPER_MIN_RAW
from .models import Frame

_BGR = {
    "red": (45, 45, 215),
    "blue": (210, 85, 35),
    "green": (65, 165, 45),
    "yellow": (35, 205, 225),
    "orange": (25, 130, 235),
    "purple": (185, 50, 145),
}


class Simulation:
    """Synthetic wrist-camera + 2.5D arm & gripper feedback (not a physics dynamics simulator)."""

    mode = "simulation"
    simulated = True
    names = ("default", "missing", "lost", "stale", "stalled", "grasp_miss", "grasp_slip")

    def __init__(self, scenario="default", clock=time.monotonic, seed=None):
        self.clock = clock
        self.lock = threading.RLock()
        self.reset(scenario, seed)

    def reset(self, scenario="default", seed=None):
        if scenario not in self.names:
            raise ValueError("未知模拟场景")
        with self.lock:
            self.scenario = scenario
            self.offset = np.zeros(2, dtype=float)
            self.height_mm = 160.0
            self.table_z_mm = 10.0
            self.mm_per_norm = 250.0
            self.gripper_open_raw = 430
            self.gripper_close_raw = 278
            self.gripper_raw = self.gripper_close_raw
            self.gripper_load = 20
            self.held_object = None
            self.grasp_active = False
            self.stopped = False
            self.steps = 0
            self.grasp_moves = 0
            self.sequence = 0
            self.frozen = None
            self.objects = [
                ["blue", "square", 0.76, 0.28],
                ["red", "circle", 0.22, 0.36],
                ["red", "square", 0.66, 0.74],
                ["green", "circle", 0.30, 0.75],
            ]
            if seed is not None:
                rng = np.random.default_rng(seed)
                for obj in self.objects:
                    obj[2] += float(rng.uniform(-0.07, 0.07))
                    obj[3] += float(rng.uniform(-0.06, 0.06))
            if scenario == "missing":
                self.objects = [obj for obj in self.objects if obj[0] != "blue"]

    def _opening(self) -> float:
        span = max(1, self.gripper_open_raw - self.gripper_close_raw)
        return round(float(np.clip((self.gripper_raw - self.gripper_close_raw) / span, 0.0, 1.0)), 3)

    def position_mm(self) -> np.ndarray:
        with self.lock:
            return np.array(
                [
                    180.0 - self.offset[1] * self.mm_per_norm,
                    -self.offset[0] * self.mm_per_norm,
                    self.height_mm,
                ],
                dtype=float,
            )

    def gripper_state(self) -> dict:
        with self.lock:
            held = None
            if self.held_object is not None:
                held = {"color": self.held_object[0], "shape": self.held_object[1]}
            return {
                "raw": int(self.gripper_raw),
                "opening": self._opening(),
                "load": int(self.gripper_load),
                "holding": self.held_object is not None,
                "held_object": held,
                "height_mm": round(float(self.height_mm), 1),
            }

    @staticmethod
    def _draw_shape(image, color: str, shape: str, center: tuple[int, int], radius: int):
        bgr = _BGR.get(color, (120, 120, 120))
        cx, cy = center
        if shape == "circle":
            cv2.circle(image, (cx, cy), radius, bgr, -1)
        elif shape == "triangle":
            pts = np.array(
                [[cx, cy - radius], [cx - radius, cy + radius], [cx + radius, cy + radius]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(image, pts, bgr)
        else:
            cv2.rectangle(image, (cx - radius, cy - radius), (cx + radius, cy + radius), bgr, -1)

    def _draw_gripper(self, image):
        opening = self._opening()
        half_gap = int(round(20 + opening * 44))
        jaw_w = 24
        jaw_h = 92
        cy = 240
        left_x2 = 320 - half_gap
        right_x1 = 320 + half_gap
        # Neutral low-saturation metallic tones so HSV color detection ignores the jaws.
        body = (78, 82, 86)
        edge = (52, 56, 60)
        pad = (96, 98, 100)
        cv2.rectangle(image, (max(0, left_x2 - jaw_w), cy - jaw_h // 2), (left_x2, cy + jaw_h // 2), body, -1)
        cv2.rectangle(image, (max(0, left_x2 - jaw_w), cy - jaw_h // 2), (left_x2, cy + jaw_h // 2), edge, 2)
        cv2.rectangle(image, (max(0, left_x2 - 6), cy - jaw_h // 3), (left_x2, cy + jaw_h // 3), pad, -1)

        cv2.rectangle(image, (right_x1, cy - jaw_h // 2), (min(639, right_x1 + jaw_w), cy + jaw_h // 2), body, -1)
        cv2.rectangle(image, (right_x1, cy - jaw_h // 2), (min(639, right_x1 + jaw_w), cy + jaw_h // 2), edge, 2)
        cv2.rectangle(image, (right_x1, cy - jaw_h // 3), (min(639, right_x1 + 6), cy + jaw_h // 3), pad, -1)

    def read(self):
        with self.lock:
            if self.scenario == "stale" and self.sequence >= 3:
                return self.frozen
            image = np.full((480, 640, 3), 235, np.uint8)
            scale = 160.0 / max(60.0, self.height_mm)
            if self.scenario != "lost" or self.steps < 2:
                for color, shape, x, y in self.objects:
                    rel_x = (x - self.offset[0] - 0.5) * scale + 0.5
                    rel_y = (y - self.offset[1] - 0.5) * scale + 0.5
                    center = (round(rel_x * 640), round(rel_y * 480))
                    radius = max(8, round(27 * scale))
                    self._draw_shape(image, color, shape, center, radius)
            if self.held_object is not None:
                self._draw_shape(image, self.held_object[0], self.held_object[1], (320, 240), 34)
            if self.grasp_active or self.held_object is not None or self.height_mm < 155.0:
                self._draw_gripper(image)
            self.sequence += 1
            self.frozen = Frame(image, self.sequence, self.clock())
            return self.frozen

    def step(self, dx, dy):
        with self.lock:
            if self.stopped:
                raise InterruptedError("模拟运动已停止")
            delta = np.clip(np.array([dx, dy], dtype=float) * 0.40, -0.045, 0.045)
            if not np.isfinite(delta).all() or np.any(np.abs(self.offset + delta) > 0.8):
                raise ValueError("超出模拟工作区")
            self.steps += 1
            if self.scenario != "stalled":
                self.offset += delta

    def move_delta(self, delta_mm):
        with self.lock:
            if self.stopped:
                raise InterruptedError("模拟运动已停止")
            delta = np.asarray(delta_mm, dtype=float)
            if delta.shape != (3,) or not np.all(np.isfinite(delta)):
                raise ValueError("位移向量必须是 3 维有限毫米坐标")
            new_offset = self.offset + np.array([-delta[1] / self.mm_per_norm, -delta[0] / self.mm_per_norm])
            new_z = self.height_mm + float(delta[2])
            if np.any(np.abs(new_offset) > 0.9) or not self.table_z_mm <= new_z <= 240.0:
                raise ValueError("目标超出模拟工作区或低于桌面安全高度")
            self.grasp_active = True
            self.offset = new_offset
            self.height_mm = new_z
            self.grasp_moves += 1
            if (
                self.scenario == "grasp_slip"
                and delta[2] > 0
                and self.held_object is not None
                and self.height_mm >= 75.0
            ):
                dropped = list(self.held_object)
                dropped[2] = float(np.clip(0.5 + self.offset[0] + 0.03, 0.1, 0.9))
                dropped[3] = float(np.clip(0.5 + self.offset[1] + 0.02, 0.1, 0.9))
                self.objects.append(dropped)
                self.held_object = None
                self.gripper_raw = self.gripper_close_raw
                self.gripper_load = 30

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
        with self.lock:
            if self.stopped:
                raise InterruptedError("模拟运动已停止")
            target_raw = int(target_raw)
            if not GRIPPER_MIN_RAW <= target_raw <= GRIPPER_MAX_RAW:
                raise ValueError(f"夹爪目标值 {target_raw} 超出允许范围 [{GRIPPER_MIN_RAW}, {GRIPPER_MAX_RAW}]")
            self.grasp_active = True
            if not expect_object:
                if self.held_object is not None and target_raw >= 360:
                    placed = list(self.held_object)
                    placed[2] = float(np.clip(0.5 + self.offset[0], 0.1, 0.9))
                    placed[3] = float(np.clip(0.5 + self.offset[1], 0.1, 0.9))
                    self.objects.append(placed)
                    self.held_object = None
                self.gripper_raw = target_raw
                self.gripper_load = 25
                return {
                    "raw": int(self.gripper_raw),
                    "load": int(self.gripper_load),
                    "gripped": False,
                    "opening": self._opening(),
                    "held_object": None,
                }

            lo, hi = held_range if held_range is not None else (298, 415)
            caught_index = None
            if self.scenario != "grasp_miss" and self.height_mm <= 45.0:
                best_dist = 0.11
                for idx, obj in enumerate(self.objects):
                    dist = math.hypot(obj[2] - self.offset[0] - 0.5, obj[3] - self.offset[1] - 0.5)
                    if dist <= best_dist:
                        best_dist = dist
                        caught_index = idx

            if caught_index is not None:
                obj = self.objects.pop(caught_index)
                self.held_object = obj
                # Square blocks hold slightly wider than circles; stays comfortably inside held_range.
                settled = int(round((lo + hi) / 2)) + (6 if obj[1] == "square" else -4)
                self.gripper_raw = max(lo, min(hi, settled))
                self.gripper_load = min(max_load, 210)
                return {
                    "raw": int(self.gripper_raw),
                    "load": int(self.gripper_load),
                    "gripped": True,
                    "opening": self._opening(),
                    "held_object": {"color": obj[0], "shape": obj[1]},
                }

            self.gripper_raw = target_raw
            self.gripper_load = 35
            return {
                "raw": int(self.gripper_raw),
                "load": int(self.gripper_load),
                "gripped": False,
                "opening": self._opening(),
                "held_object": None,
            }

    def stop(self):
        with self.lock:
            self.stopped = True

    def resume(self):
        with self.lock:
            self.stopped = False

    def close(self):
        self.stop()
