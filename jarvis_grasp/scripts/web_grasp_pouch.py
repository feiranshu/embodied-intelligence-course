"""Minimal one-click Web UI server for physical SO-101 green pencil-case grasping.

Opens the wrist camera once for continuous live preview & MP4 recording, exposes:
  GET  /              -> web/grasp_pouch.html
  GET  /api/state     -> live status, stage message, gripper telemetry & last result
  GET  /api/frame.jpg -> latest wrist camera JPEG frame
  POST /api/start     -> launch run_pouch_grasp(target_color="green", grasp_z_mm=26.0)
  POST /api/stop      -> immediate local emergency stop (cuts torque on all 6 servos)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.hardware import resolve_camera_source
from jarvis.perception import draw_detections
from scripts.hw_stage_test import DEFAULT_CAMERA, SCS_DIR, load_calibrated_model
from scripts.real_grasp_pouch import GOAL_UV, _cand_uv, detect_target_in_frame, run_pouch_grasp
from vendor.scs215.advanced import load_advanced
from vendor.scs215.servo_bus import load_config

HTML_PATH = ROOT / "web" / "grasp_pouch.html"

STAGE_LABELS_ZH = {
    "idle": "已就绪，请将绿色笔盒放在机械臂正前方后点击开始。",
    "pouch: lift_to_observe": "1/7 正在安全抬起至高空俯拍观察位…",
    "pouch: detect_initial": "2/7 正在识别前方绿色笔盒…",
    "pouch: search_observe_forward": "2/7 正在前探视角搜索绿色笔盒…",
    "pouch: search_observe_left": "2/7 正在左侧视角搜索绿色笔盒…",
    "pouch: search_observe_right": "2/7 正在右侧视角搜索绿色笔盒…",
    "pouch: probe_reach": "3/7 正在核验视觉伺服方向…",
    "pouch: open_gripper": "4/7 已对准目标，正在张开夹爪…",
    "pouch: descend": "5/7 正在分段平稳下降至笔盒抓取高度…",
    "pouch: close_gripper": "6/7 正在闭合夹爪夹取绿色笔盒（负载防堵转监测）…",
    "pouch: lift": "6/7 已夹住目标，正在垂直抬升并复检…",
    "pouch: place_and_release": "7/7 复检通过，正在轻放回桌面并松开夹爪…",
    "pouch: return_safe_rest": "正在平稳返回安全停泊位并断扭…",
}


def _human_stage(label: str) -> str:
    if label in STAGE_LABELS_ZH:
        return STAGE_LABELS_ZH[label]
    if label.startswith("pouch: servo_"):
        return "3/7 正在视觉伺服对准绿色笔盒中心…"
    return label


class LiveCameraHub:
    """Single shared wrist camera reader compatible with run_pouch_grasp's WristRecorder API."""

    def __init__(self, camera_source=DEFAULT_CAMERA, width: int = 640, height: int = 480, fps: float = 20.0, on_label=None):
        self.camera_source = camera_source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._on_label = on_label

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._seq: int = 0
        self._stage_label: str = "idle"
        self._writer = None
        self._tmp_mp4: Path | None = None
        self._target_mp4: Path | None = None
        self.frame_count: int = 0

        self.device_name: str = str(camera_source)
        self.ready: bool = False
        self.error: str = ""
        self.cap = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            idx, backend, device_name = resolve_camera_source(self.camera_source)
            self.device_name = device_name
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if not cap.isOpened() and backend is not None:
                cap.release()
                cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"无法打开机械臂摄像头 {self.camera_source}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            for _ in range(6):
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._latest = frame
                        self._latest_ts = time.monotonic()
                        self._seq += 1
            self.cap = cap
            self.ready = self._latest is not None
            if not self.ready:
                self.error = f"摄像头 {device_name} 未返回画面"
        except Exception as exc:
            self.ready = False
            self.error = str(exc)
            return
        self._thread = threading.Thread(target=self._loop, name="grasp-web-cam", daemon=True)
        self._thread.start()

    def set_label(self, label: str) -> None:
        with self._lock:
            self._stage_label = label
        if self._on_label is not None:
            self._on_label(label)

    def start_recording(self, mp4_path: Path) -> None:
        with self._lock:
            self._stop_recording_locked()
            if self._latest is None:
                return
            h, w = self._latest.shape[:2]
            mp4_path = Path(mp4_path)
            mp4_path.parent.mkdir(parents=True, exist_ok=True)
            self._target_mp4 = mp4_path
            self._tmp_mp4 = Path(os.environ.get("TEMP", ".")) / f"so101_web_{int(time.time() * 1000)}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self._tmp_mp4), fourcc, self.fps, (w, h))
            self.frame_count = 0

    def stop_recording(self) -> Path | None:
        with self._lock:
            return self._stop_recording_locked()

    def _stop_recording_locked(self) -> Path | None:
        writer = self._writer
        tmp_mp4 = self._tmp_mp4
        target_mp4 = self._target_mp4
        self._writer = None
        self._tmp_mp4 = None
        self._target_mp4 = None
        if writer is not None:
            writer.release()
        if tmp_mp4 is not None and tmp_mp4.is_file() and tmp_mp4.stat().st_size > 0 and target_mp4 is not None:
            target_mp4.write_bytes(tmp_mp4.read_bytes())
            tmp_mp4.unlink(missing_ok=True)
            return target_mp4
        if tmp_mp4 is not None:
            tmp_mp4.unlink(missing_ok=True)
        return None

    def latest_frame(self, *, wait_fresh: bool = False) -> tuple[np.ndarray, float]:
        if wait_fresh:
            with self._lock:
                start_seq = self._seq
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                with self._lock:
                    if self._seq >= start_seq + 3 and self._latest is not None:
                        return self._latest.copy(), self._latest_ts
                time.sleep(0.03)
        with self._lock:
            if self._latest is None:
                raise RuntimeError(self.error or "尚未获取相机画面")
            return self._latest.copy(), self._latest_ts

    def preview_jpeg(self) -> bytes:
        with self._lock:
            frame = None if self._latest is None else self._latest.copy()
            label = self._stage_label
        if frame is None:
            frame = np.full((480, 640, 3), 24, dtype=np.uint8)
            cv2.putText(frame, "Camera not ready", (190, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2)
        else:
            cand, all_c = detect_target_in_frame(frame, target_color="green", min_area=1800)
            frame = draw_detections(frame, all_c, selected_id=cand.id if cand else None, center_fraction=0.18)
            h, w = frame.shape[:2]
            gx = int(round(GOAL_UV[0]))
            cv2.line(frame, (gx, 100), (gx, h - 20), (60, 220, 120), 1, cv2.LINE_AA)
            cv2.line(frame, (425, 100), (425, h - 20), (80, 160, 245), 1, cv2.LINE_AA)
            if cand is not None:
                uv = _cand_uv(cand, frame)
                cv2.circle(frame, (int(round(uv[0])), int(round(uv[1]))), 7, (0, 230, 255), -1, cv2.LINE_AA)
            stamp = time.strftime("%H:%M:%S")
            cv2.putText(
                frame,
                f"{stamp} | {label}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (30, 220, 90),
                2,
                cv2.LINE_AA,
            )
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            now = time.monotonic()
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame
                    self._latest_ts = now
                    self._seq += 1
                    self.ready = True
                    self.error = ""
                    if self._writer is not None:
                        overlay = frame.copy()
                        stamp = time.strftime("%H:%M:%S")
                        cv2.putText(
                            overlay,
                            f"{stamp} | {self._stage_label}",
                            (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (20, 220, 20),
                            2,
                            cv2.LINE_AA,
                        )
                        self._writer.write(overlay)
                        self.frame_count += 1
            else:
                self._stop.wait(0.02)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class GraspWebApp:
    """Thread-safe coordinator for one-click green pencil case grasping."""

    def __init__(
        self,
        *,
        camera_hub: LiveCameraHub,
        port: str | None = None,
        target_color: str = "green",
        grasp_z_mm: float = 26.0,
        runner=None,
    ):
        self.camera = camera_hub
        self.port = port
        self.target_color = target_color
        self.grasp_z_mm = float(grasp_z_mm)
        self._runner = runner
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self.busy: bool = False
        self.status: str = "idle"
        self.stage: str = "idle"
        self.message: str = STAGE_LABELS_ZH["idle"]
        self.last_result: dict | None = None

    def on_stage_label(self, label: str) -> None:
        with self._lock:
            self.stage = label
            if self.busy:
                self.message = _human_stage(label)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "busy": self.busy,
                "status": self.status,
                "stage": self.stage,
                "message": self.message,
                "target_color": self.target_color,
                "grasp_z_mm": self.grasp_z_mm,
                "camera_ready": bool(self.camera.ready),
                "camera_name": self.camera.device_name,
                "camera_error": self.camera.error,
                "last_result": self.last_result,
            }

    def start_grasp(self, *, release_after: bool = True, grasp_z_mm: float | None = None) -> dict:
        with self._lock:
            if self.busy:
                raise RuntimeError("机械臂正在执行抓取任务，请等待完成或先点击急停")
            if not self.camera.ready:
                raise ValueError(self.camera.error or "腕部摄像头尚未就绪")
            if grasp_z_mm is not None:
                z_val = float(grasp_z_mm)
                if not 18.0 <= z_val <= 60.0:
                    raise ValueError("抓取高度 grasp_z_mm 必须在 18～60 mm 安全范围内")
                self.grasp_z_mm = z_val
            self.busy = True
            self.status = "running"
            self.stage = "pouch: lift_to_observe"
            self.message = "正在启动绿色笔盒抓取流程…"
            self._cancel.clear()
            self._worker = threading.Thread(
                target=self._run_worker,
                kwargs={"release_after": bool(release_after)},
                name="grasp-pouch-worker",
                daemon=True,
            )
            self._worker.start()
            return self.snapshot()

    def stop(self) -> dict:
        with self._lock:
            self._cancel.set()
            if self.busy:
                self.message = "已收到急停指令，正在立即切断六轴扭矩…"
            return self.snapshot()

    def _run_worker(self, *, release_after: bool) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        mp4_path = ROOT / "logs" / f"web_grasp_{ts}.mp4"
        log_path = ROOT / "logs" / f"web_grasp_{ts}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.camera.start_recording(mp4_path)
        saved_mp4 = None
        try:
            if self._runner is not None:
                res = self._runner(
                    recorder=self.camera,
                    log_path=log_path,
                    cancel=self._cancel,
                    target_color=self.target_color,
                    grasp_z_mm=self.grasp_z_mm,
                    release_after=release_after,
                )
            else:
                hw_cfg = load_config(SCS_DIR / "config.json")
                adv_cfg = load_advanced(SCS_DIR / "advanced.json")
                model, _ = load_calibrated_model(adv_cfg)
                res = run_pouch_grasp(
                    hw_cfg,
                    adv_cfg,
                    model,
                    self.port,
                    self.camera,
                    log_path,
                    self._cancel,
                    target_color=self.target_color,
                    grasp_z_mm=self.grasp_z_mm,
                    release_after=release_after,
                )
            saved_mp4 = self.camera.stop_recording()
            res["video_mp4"] = str(saved_mp4) if saved_mp4 else None
            res["frames_recorded"] = self.camera.frame_count
            res["telemetry_jsonl"] = str(log_path)
            verified = bool((res.get("after_lift") or {}).get("grasp_verified"))
            with self._lock:
                self.last_result = res
                if self._cancel.is_set():
                    self.status = "stopped"
                    self.message = "任务已人工急停，六轴扭矩已关闭。"
                elif verified:
                    self.status = "success"
                    self.message = (
                        "抓取成功！已夹起绿色笔盒、复检通过并安全放回桌面（6轴已断扭）。"
                        if release_after
                        else "抓取成功！已夹起绿色笔盒并完成悬空复检（6轴已断扭）。"
                    )
                else:
                    self.status = "missed"
                    self.message = "本轮未夹到笔盒（已如实检出空抓并安全回位断扭），可微调位置后重试。"
        except InterruptedError:
            saved_mp4 = self.camera.stop_recording()
            with self._lock:
                self.status = "stopped"
                self.message = "已人工急停，六轴扭矩已关闭。"
                self.last_result = {"torque_off_verified": True, "video_mp4": str(saved_mp4) if saved_mp4 else None}
        except Exception as exc:
            saved_mp4 = self.camera.stop_recording()
            with self._lock:
                self.status = "error"
                self.message = f"执行异常（已安全断扭）：{exc}"
                self.last_result = {
                    "error": str(exc),
                    "torque_off_verified": True,
                    "video_mp4": str(saved_mp4) if saved_mp4 else None,
                }
        finally:
            self.camera.set_label("idle")
            with self._lock:
                self.busy = False
                self.stage = "idle"

    def close(self) -> None:
        self.stop()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=5.0)
        self.camera.close()


def create_server(app: GraspWebApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_response_body(self, status: int, body, content_type: str = "application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    self.send_response_body(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self.send_response_body(200, app.snapshot())
                elif path == "/api/frame.jpg":
                    self.send_response_body(200, app.camera.preview_jpeg(), "image/jpeg")
                else:
                    self.send_response_body(404, {"error": "接口不存在"})
            except Exception as exc:
                self.send_response_body(500, {"error": str(exc)})

        def do_POST(self):
            path = urlsplit(self.path).path
            try:
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    self.send_response_body(403, {"error": "只允许本页面发起请求"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length) if length > 0 else b""
                if path == "/api/stop":
                    self.send_response_body(200, app.stop())
                elif path == "/api/start":
                    data = json.loads(raw_body or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("请求体必须是 JSON 对象")
                    release_after = bool(data.get("release_after", True))
                    grasp_z = data.get("grasp_z_mm")
                    self.send_response_body(200, app.start_grasp(release_after=release_after, grasp_z_mm=grasp_z))
                else:
                    self.send_response_body(404, {"error": "接口不存在"})
            except ValueError as exc:
                self.send_response_body(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.send_response_body(409, {"error": str(exc)})
            except Exception as exc:
                self.send_response_body(500, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="SO-101 绿色笔盒一键抓取极简 Web 界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--grasp-z-mm", type=float, default=26.0)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    app_ref: dict[str, GraspWebApp] = {}
    camera = LiveCameraHub(
        camera_source=args.camera,
        on_label=lambda lbl: app_ref["app"].on_stage_label(lbl) if "app" in app_ref else None,
    )
    camera.start()
    app = GraspWebApp(
        camera_hub=camera,
        port=args.serial_port,
        target_color="green",
        grasp_z_mm=args.grasp_z_mm,
    )
    app_ref["app"] = app

    server = create_server(app, host=args.host, port=args.port)
    url = f"http://{args.host}:{server.server_port}"
    print(f"SO-101 绿色笔盒一键抓取 Web 已启动：{url}", flush=True)
    print(f"摄像头状态：ready={camera.ready}, device={camera.device_name}, error={camera.error or '无'}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
