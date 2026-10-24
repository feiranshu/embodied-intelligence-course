"""Camera access shared by the agent tools and the tracking thread."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import agent_common  # noqa: F401  (puts week6 on sys.path)
from camera_utils import open_camera
from common import read_image, write_image


class CameraHub:
    """Read frames continuously in one thread; tools ask for the latest copy."""

    def __init__(self, source="GENERAL WEBCAM", size=(640, 480)):
        self.capture, index, backend, self.name = open_camera(source, size)
        print(f"摄像头“{self.name}”（索引 {index}，{backend}）已打开", flush=True)
        self._frame = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._loop, name="camera-hub", daemon=True)
        self._thread.start()

    def _loop(self):
        failures = 0
        while not self._done.is_set():
            ok, frame = self.capture.read()
            if ok and frame is not None:
                failures = 0
                with self._lock:
                    self._frame, self._stamp, self._error = frame, time.monotonic(), None
            else:
                failures += 1
                if failures >= 30:
                    self._error = "摄像头连续 30 次没有返回画面"
                time.sleep(0.03)

    def latest(self, max_age_s: float = 1.0, timeout_s: float = 3.0):
        """Return (frame, timestamp) newer than max_age_s, waiting up to timeout_s."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and time.monotonic() - self._stamp <= max_age_s:
                    return self._frame.copy(), self._stamp
            time.sleep(0.02)
        raise RuntimeError(self._error or "没有拿到新的摄像头画面")

    def snapshot(self, path: str | Path):
        frame, _ = self.latest()
        write_image(path, frame)
        return frame

    def close(self):
        self._done.set()
        self._thread.join(timeout=2)
        self.capture.release()


class ImageFileCamera:
    """Use a saved photo instead of a camera (offline tests, re-running a demo)."""

    name = "image-file"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.frame = read_image(self.path)

    def latest(self, max_age_s: float = 1.0, timeout_s: float = 3.0):
        return self.frame.copy(), time.monotonic()

    def snapshot(self, path: str | Path):
        write_image(path, self.frame)
        return self.frame.copy()

    def close(self):
        pass


def open_camera_source(source, size=(640, 480)):
    """A path to an existing image file gives ImageFileCamera, otherwise a live camera."""
    if source is not None and Path(str(source)).is_file():
        return ImageFileCamera(source)
    return CameraHub(source if source is not None else "GENERAL WEBCAM", size)
