import threading
import time

import cv2
import numpy as np

from .models import Frame


class Simulation:
    """Synthetic image feedback, not a dynamics or robot-accuracy simulation."""

    mode = "simulation"
    names = ("default", "missing", "lost", "stale", "stalled")

    def __init__(self, scenario="default", clock=time.monotonic, seed=None):
        self.clock = clock
        self.lock = threading.RLock()
        self.reset(scenario, seed)

    def reset(self, scenario="default", seed=None):
        if scenario not in self.names:
            raise ValueError("未知模拟场景")
        with self.lock:
            self.scenario = scenario
            self.offset = np.zeros(2)
            self.stopped = False
            self.steps = 0
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
                    obj[2] += rng.uniform(-0.07, 0.07)
                    obj[3] += rng.uniform(-0.06, 0.06)
            if scenario == "missing":
                self.objects = [obj for obj in self.objects if obj[0] != "blue"]

    def read(self):
        with self.lock:
            if self.scenario == "stale" and self.sequence >= 3:
                return self.frozen
            image = np.full((480, 640, 3), 235, np.uint8)
            if self.scenario != "lost" or self.steps < 2:
                for color, shape, x, y in self.objects:
                    x, y = np.array([x, y]) - self.offset
                    center = (round(x * 640), round(y * 480))
                    bgr = {"red": (45, 45, 215), "blue": (210, 85, 35), "green": (65, 165, 45)}[color]
                    if shape == "circle":
                        cv2.circle(image, center, 27, bgr, -1)
                    else:
                        cv2.rectangle(image, (center[0] - 27, center[1] - 27),
                                      (center[0] + 27, center[1] + 27), bgr, -1)
            self.sequence += 1
            self.frozen = Frame(image, self.sequence, self.clock())
            return self.frozen

    def step(self, dx, dy):
        with self.lock:
            if self.stopped:
                raise InterruptedError("模拟运动已停止")
            delta = np.clip(np.array([dx, dy]) * 0.40, -0.045, 0.045)
            if not np.isfinite(delta).all() or np.any(np.abs(self.offset + delta) > 0.8):
                raise ValueError("超出模拟工作区")
            self.steps += 1
            if self.scenario != "stalled":
                self.offset += delta

    def stop(self):
        with self.lock:
            self.stopped = True

    def resume(self):
        with self.lock:
            self.stopped = False

    def close(self):
        self.stop()
