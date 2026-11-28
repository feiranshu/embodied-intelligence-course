"""Staged physical SO-101 SCS215 bring-up, recognition & live grasping script (R1-R16).

Uses calibrated FK/IK from config/calibration_reference/hand_eye.json so +Z is verified
physically UP away from the tabletop (measured tabletop surface z_table ~= 55.5 mm).

Stages:
  - gripper-check : Energizes ONLY ID 6 (gripper) to open -> close -> torque OFF.
  - hello-world   : Lifts cleanly to safe tabletop observe pose (z ~= 198 mm), captures
                    wrist-cam recognition snapshot, tests calibrated +-15mm Z and Y moves,
                    opens/closes gripper once, gently returns to rest height, torque OFF.
  - live-grasp    : Moves to observe pose, detects target object in wrist camera
                    (color/shape or ArUco), performs visual-servo centering (with online
                    direction probe if needed), opens gripper, descends safely above
                    z_table, closes with load/stall contact detection, lifts & verifies,
                    then gently releases and powers off torque.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.hardware import resolve_camera_source
from jarvis.models import Frame
from jarvis.perception import ArUcoDetector, ColorShapeDetector, LocalSelector, TemplateTracker, draw_detections
from vendor.scs215.advanced import Arm, Kinematics, load_advanced
from vendor.scs215.joint_mapping import check_position, mapped_position
from vendor.scs215.servo_bus import GOAL_POSITION, ServoBus, load_config

SCS_DIR = ROOT / "vendor" / "scs215"
HAND_EYE_PATH = ROOT / "config" / "calibration_reference" / "hand_eye.json"
DEFAULT_CAMERA = "GENERAL WEBCAM"

# Verified on re-indexed SO-101 SCS215 (2026-11-23, non-wrapping joints):
#   - Rest contact on desk [486, 79, 856, 685, 352, 266] -> calibrated z ~= 32.2 mm
#   - OBSERVE_POSE gives high tabletop overview at calibrated z ~= 174.8 mm, r ~= 283 mm
OBSERVE_POSE_RAW = [486, 320, 560, 685, 353, 270]
SAFE_REST_POSE_RAW = [486, 120, 790, 685, 353, 270]  # calibrated z ~= 80.7 mm (~5 cm above desk)
Z_TABLE_FLOOR_MM = 35.0
GRIPPER_OPEN_RAW = 420
GRIPPER_CLOSE_RAW = 270
GRIPPER_HELD_RANGE = (290, 405)


def _save_png(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    path.write_bytes(buf.tobytes())


def load_calibrated_model(adv_cfg: dict) -> tuple[Kinematics, dict]:
    import copy

    cfg = copy.deepcopy(adv_cfg)
    # Physical camera-cable stop on ID 4 measured at ~758-765; keep IK strictly <= 755
    cfg.setdefault("kinematics", {}).setdefault("cartesian_raw_ranges", {})["4"] = [104, 755]
    hand_eye = json.loads(HAND_EYE_PATH.read_text(encoding="utf-8"))
    model = Kinematics(cfg)
    mapping = hand_eye.get("calibrated_joint_mapping")
    if mapping is not None and hasattr(model, "with_joint_calibration"):
        model = model.with_joint_calibration(mapping)
    return model, hand_eye


class WristRecorder:
    """Continuous MP4 recorder for the robot wrist camera (R7)."""

    def __init__(self, camera_source, mp4_path: Path, width: int = 640, height: int = 480, fps: float = 20.0):
        idx, backend, device_name = resolve_camera_source(camera_source)
        self.device_name = device_name
        self.index = idx
        self.cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"无法打开机械臂摄像头 {camera_source} (索引 {idx})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.mp4_path = Path(mp4_path)
        self.mp4_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._latest_ts = 0.0
        self._seq = 0
        self._stage_label = "preflight"
        self.frame_count = 0
        self.max_gap_s = 0.0

        gaps = []
        last_t = None
        for idx_read in range(10):
            ok, frame = self.cap.read()
            now = time.monotonic()
            if ok and frame is not None:
                self._latest = frame
                self._latest_ts = now
                self._seq += 1
                if idx_read >= 4 and last_t is not None:
                    gaps.append(now - last_t)
                last_t = now
        if self._latest is None:
            self.cap.release()
            raise RuntimeError(f"机械臂摄像头 {device_name} 未返回画面")
        self.max_gap_s = max(gaps) if gaps else 0.0
        if self.max_gap_s > 0.30:
            self.cap.release()
            raise RuntimeError(f"机械臂摄像头帧间隔过大 ({self.max_gap_s:.3f}s > 0.30s)")

        h, w = self._latest.shape[:2]
        self._tmp_mp4 = Path(os.environ.get("TEMP", ".")) / f"so101_{int(time.time()*1000)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._tmp_mp4), fourcc, fps, (w, h))
        self._thread = threading.Thread(target=self._loop, name="wrist-recorder", daemon=True)
        self._thread.start()

    def set_label(self, label: str) -> None:
        with self._lock:
            self._stage_label = label

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
                raise RuntimeError("尚未获取相机帧")
            return self._latest.copy(), self._latest_ts

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            now = time.monotonic()
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame
                    self._latest_ts = now
                    self._seq += 1
                    label = self._stage_label
                overlay = frame.copy()
                stamp = time.strftime("%H:%M:%S")
                cv2.putText(
                    overlay,
                    f"{stamp} | {label}",
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

    def close(self) -> Path | None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()
        self._writer.release()
        if self._tmp_mp4.is_file() and self._tmp_mp4.stat().st_size > 0:
            self.mp4_path.write_bytes(self._tmp_mp4.read_bytes())
            self._tmp_mp4.unlink(missing_ok=True)
            return self.mp4_path
        self._tmp_mp4.unlink(missing_ok=True)
        return None


def preflight_read_servos(hw_cfg: dict, adv_cfg: dict, model: Kinematics, port: str | None) -> dict:
    """R6 read-only pre-flight check using calibrated FK."""
    with ServoBus(hw_cfg, port, stop_on_close=False) as bus:
        selected_port = bus.port
        raw = []
        servos = []
        for item in hw_cfg["servos"]:
            sid = int(item["id"])
            pos = int(bus.read_position(sid))
            mode, lo, hi = bus.mode(sid)
            torque = bool(bus._read_u8(sid, 0x28))
            voltage_v = round(bus._read_u8(sid, 0x3E) / 10.0, 1)
            temp_c = int(bus._read_u8(sid, 0x3F))
            mapping = adv_cfg["joint_mapping"][str(sid)]
            check_position(pos, mapping, tolerance=12)
            if mode != "position":
                raise RuntimeError(f"ID {sid} 不在位置模式 (mode={mode})")
            if torque:
                raise RuntimeError(f"ID {sid} 上电前扭矩未关闭 (Torque_Enable=1)")
            if voltage_v < 6.5:
                raise RuntimeError(f"ID {sid} 电压过低 ({voltage_v}V < 6.5V)")
            raw.append(pos)
            servos.append(
                {
                    "id": sid,
                    "name": item["name"],
                    "raw": pos,
                    "mapped": mapped_position(pos, mapping),
                    "mode": mode,
                    "hw_limits": [lo, hi],
                    "torque_enabled": torque,
                    "voltage_v": voltage_v,
                    "temp_c": temp_c,
                }
            )
    xyz = [round(float(v), 2) for v in model.forward(raw)]
    return {"port": selected_port, "raw": raw, "calibrated_xyz_mm": xyz, "servos": servos}


def _interp_poses(start: list[int], end: list[int], max_ticks: int = 85) -> list[list[int]]:
    max_diff = max(abs(b - a) for a, b in zip(start, end))
    n = max(1, math.ceil(max_diff / max_ticks))
    return [[int(round(a + (b - a) * k / n)) for a, b in zip(start, end)] for k in range(1, n + 1)]


def check_plan(stage: str, start_raw: list[int], adv_cfg: dict, model: Kinematics, grasp_z_mm: float = 70.0) -> dict:
    """Validate planned poses with calibrated FK/IK, joint limits, and R5 bounds (R2, R5, R8)."""
    deg_per_raw = float(adv_cfg["kinematics"]["degrees_per_raw"])
    waypoints: list[tuple[str, list[int]]] = [("start", list(start_raw))]

    if stage == "gripper-check":
        p_open = list(start_raw)
        p_open[5] = GRIPPER_OPEN_RAW
        p_close = list(start_raw)
        p_close[5] = GRIPPER_CLOSE_RAW
        waypoints.extend([("gripper_open", p_open), ("gripper_close", p_close)])
    elif stage in ("hello-world", "live-grasp"):
        # Lift elbow first if starting low near table so we clear any tabletop objects
        lift_first = list(start_raw)
        lift_first[2] = min(start_raw[2], OBSERVE_POSE_RAW[2])
        if lift_first[2] != start_raw[2]:
            for idx, wp in enumerate(_interp_poses(start_raw, lift_first, max_ticks=70), 1):
                waypoints.append((f"lift_clear_{idx}", wp))
        for idx, wp in enumerate(_interp_poses(waypoints[-1][1], OBSERVE_POSE_RAW, max_ticks=75), 1):
            waypoints.append((f"to_observe_{idx}", wp))

        obs_xyz = model.forward(OBSERVE_POSE_RAW)
        if stage == "hello-world":
            up_raw = model.inverse(obs_xyz + np.array([0.0, 0.0, 15.0]), np.asarray(OBSERVE_POSE_RAW), allow_crossing=False).tolist()
            y_plus = model.inverse(obs_xyz + np.array([0.0, 15.0, 0.0]), np.asarray(OBSERVE_POSE_RAW), allow_crossing=False).tolist()
            p_open = list(OBSERVE_POSE_RAW)
            p_open[5] = GRIPPER_OPEN_RAW
            p_close = list(OBSERVE_POSE_RAW)
            p_close[5] = GRIPPER_CLOSE_RAW
            waypoints.extend(
                [
                    ("cartesian_up_15mm", up_raw),
                    ("observe_after_z", list(OBSERVE_POSE_RAW)),
                    ("cartesian_left_15mm", y_plus),
                    ("observe_after_y", list(OBSERVE_POSE_RAW)),
                    ("gripper_open", p_open),
                    ("gripper_close", p_close),
                ]
            )
        else:
            if grasp_z_mm < Z_TABLE_FLOOR_MM:
                raise ValueError(f"grasp_z_mm ({grasp_z_mm}) 低于桌面安全底面 ({Z_TABLE_FLOOR_MM} mm)")
            pre_xyz = np.array([obs_xyz[0], obs_xyz[1], max(grasp_z_mm + 35.0, 110.0)])
            grasp_xyz = np.array([obs_xyz[0], obs_xyz[1], grasp_z_mm])
            pre_raw = model.inverse(pre_xyz, np.asarray(OBSERVE_POSE_RAW), allow_crossing=False).tolist()
            grasp_raw = model.inverse(grasp_xyz, np.asarray(pre_raw), allow_crossing=False).tolist()
            p_open = list(OBSERVE_POSE_RAW)
            p_open[5] = GRIPPER_OPEN_RAW
            waypoints.extend(
                [
                    ("gripper_open", p_open),
                    ("pre_grasp_sample", pre_raw),
                    ("grasp_plane_sample", grasp_raw),
                    ("lift_return", list(OBSERVE_POSE_RAW)),
                ]
            )
        for idx, wp in enumerate(_interp_poses(waypoints[-1][1], SAFE_REST_POSE_RAW, max_ticks=75), 1):
            waypoints.append((f"to_safe_rest_{idx}", wp))
    else:
        raise ValueError(f"未知测试阶段: {stage}")

    report_wps = []
    min_z_mm = float("inf")
    max_step_deg = 0.0
    prev_raw = start_raw
    for name, pose in waypoints:
        for i, val in enumerate(pose, 1):
            mapping = adv_cfg["joint_mapping"][str(i)]
            check_position(val, mapping, tolerance=12)
            dm = mapped_position(val, mapping) - mapped_position(prev_raw[i - 1], mapping)
            dr = (val - prev_raw[i - 1]) * mapping.get("sign", 1)
            if dm != dr:
                raise ValueError(f"路径段 {name} 关节 ID {i} 需跨零")
            if name != "start":
                max_step_deg = max(max_step_deg, abs(dm) * deg_per_raw)
        xyz = [round(float(v), 2) for v in model.forward(pose)]
        if name != "start":
            min_z_mm = min(min_z_mm, xyz[2])
        report_wps.append({"name": name, "raw": pose, "calibrated_xyz_mm": xyz})
        prev_raw = pose

    return {
        "stage": stage,
        "waypoints": report_wps,
        "z_min_planned_mm": round(min_z_mm, 2),
        "z_table_floor_mm": Z_TABLE_FLOOR_MM,
        "max_joint_step_deg": round(max_step_deg, 1),
        "r5_ok": min_z_mm >= Z_TABLE_FLOOR_MM and max_step_deg <= 45.0,
    }


def _log_telemetry(log_path: Path, stage: str, label: str, raw: list[int], xyz: list[float], extra: dict | None = None):
    record = {"t": round(time.time(), 3), "stage": stage, "label": label, "raw": raw, "calibrated_xyz_mm": xyz}
    if extra:
        record.update(extra)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _move_pose_interpolated(arm: Arm, target_raw: list[int], *, max_ticks: int = 75, dur_per_seg: float = 1.6, speed: int = 120, joint_order=(3, 2, 4, 5, 1, 6), cancel: threading.Event):
    current = [int(v) for v in arm.positions()]
    for wp in _interp_poses(current, target_raw, max_ticks=max_ticks):
        if cancel.is_set():
            raise InterruptedError("操作者中止")
        arm.move_pose(wp, duration_s=dur_per_seg, speed=speed, quiet=True, joint_order=joint_order, skip_arrived_base=True)


def _close_gripper_contact(bus: ServoBus, target_raw: int = GRIPPER_CLOSE_RAW, *, max_load: int = 340, hold_bias: int = 4, speed: int = 140, cancel: threading.Event) -> dict:
    """Contact- and load-aware gripper close on SCS215 ID 6."""
    bus.move_position(6, int(target_raw), 1400, int(speed))
    t0 = time.monotonic()
    stalled = 0
    prev = None
    actual = int(target_raw)
    max_seen_load = 0
    reason = "timeout"
    while time.monotonic() - t0 < 2.6:
        if cancel.is_set():
            raise InterruptedError("操作者中止")
        time.sleep(0.04)
        bus.heartbeat(6)
        actual = int(bus.read_position(6))
        load_mag = int(bus._read_u16(6, 60)) & 1023
        max_seen_load = max(max_seen_load, load_mag)
        if prev is not None and abs(actual - prev) <= 1:
            stalled += 1
        else:
            stalled = 0
        prev = actual
        if load_mag >= max_load:
            reason = "load_contact"
            break
        if stalled >= 6 and (time.monotonic() - t0) > 0.35:
            reason = "stall_contact"
            break
        if abs(actual - target_raw) <= 4:
            reason = "reached_close"
            break
    hold_raw = max(260, min(662, max(int(target_raw), actual - int(hold_bias))))
    bus._write(6, GOAL_POSITION, [hold_raw >> 8, hold_raw & 255])
    bus.finish_position(6)
    lo, hi = GRIPPER_HELD_RANGE
    return {
        "raw": actual,
        "hold_raw": hold_raw,
        "max_load": max_seen_load,
        "stop_reason": reason,
        "gripped": lo <= actual <= hi,
    }


def _open_gripper(bus: ServoBus, target_raw: int = GRIPPER_OPEN_RAW, *, speed: int = 140, cancel: threading.Event) -> dict:
    bus.move_position(6, int(target_raw), 1100, int(speed))
    t0 = time.monotonic()
    actual = int(target_raw)
    while time.monotonic() - t0 < 2.0:
        if cancel.is_set():
            raise InterruptedError("操作者中止")
        time.sleep(0.04)
        bus.heartbeat(6)
        actual = int(bus.read_position(6))
        if abs(actual - target_raw) <= 6:
            break
    bus.finish_position(6)
    return {"raw": actual, "target": target_raw}


def run_hello_world_live(hw_cfg: dict, adv_cfg: dict, model: Kinematics, port: str | None, recorder: WristRecorder, log_path: Path, cancel: threading.Event):
    results = {"stage": "hello-world", "steps": [], "detections": []}
    bus = ServoBus(hw_cfg, port, stop_on_close=False)
    arm = Arm(adv_cfg, bus)
    arm.model = model
    safe_parked = False

    def _watch():
        while not cancel.is_set():
            time.sleep(0.05)
        arm.stop()

    threading.Thread(target=_watch, daemon=True).start()

    try:
        start_raw = [int(v) for v in arm.positions()]
        _log_telemetry(log_path, "hello-world", "start", start_raw, [round(float(v), 2) for v in arm.model.forward(start_raw)])

        # 1. Lift straight UP off tabletop first (ID 3 before other joints), then enter OBSERVE_POSE
        recorder.set_label("hello-world: lift_to_observe")
        lift_first = list(start_raw)
        lift_first[2] = min(start_raw[2], OBSERVE_POSE_RAW[2])
        if lift_first[2] != start_raw[2]:
            _move_pose_interpolated(arm, lift_first, max_ticks=65, dur_per_seg=1.5, speed=110, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
        _move_pose_interpolated(arm, OBSERVE_POSE_RAW, max_ticks=70, dur_per_seg=1.6, speed=120, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)

        obs_raw = [int(v) for v in arm.positions()]
        obs_xyz = [round(float(v), 2) for v in arm.model.forward(obs_raw)]
        _log_telemetry(log_path, "hello-world", "at_observe", obs_raw, obs_xyz)
        results["steps"].append({"label": "at_observe", "raw": obs_raw, "calibrated_xyz_mm": obs_xyz})

        # 2. Capture fresh wrist-camera tabletop frame & run live object recognition!
        time.sleep(0.4)
        recorder.set_label("hello-world: observe_detect")
        frame, _ = recorder.latest_frame(wait_fresh=True)
        obs_path = ROOT / "logs" / "hello_world_observe.png"
        _save_png(obs_path, frame)
        fobj = Frame(frame, 1, time.monotonic())
        all_cands = [*ColorShapeDetector(real_world=True).detect(fobj), *ArUcoDetector().detect(fobj)]
        det_path = ROOT / "logs" / "hello_world_detected.png"
        _save_png(det_path, draw_detections(frame, all_cands, center_fraction=0.2))
        results["observe_image"] = str(obs_path)
        results["detected_image"] = str(det_path)
        results["detections"] = [c.to_dict() for c in all_cands]

        # 3. Verify calibrated Cartesian Z (+15/-15 mm) & Y (+15/-15 mm) moves!
        base_obs_xyz = np.asarray(arm.model.forward(arm.positions()), dtype=float)
        for label, offset in [
            ("cartesian_up_15mm", [0.0, 0.0, 15.0]),
            ("cartesian_down_15mm", [0.0, 0.0, 0.0]),
            ("cartesian_left_15mm", [0.0, 15.0, 0.0]),
            ("cartesian_right_15mm", [0.0, 0.0, 0.0]),
        ]:
            if cancel.is_set():
                raise InterruptedError("操作者中止")
            recorder.set_label(f"hello-world: {label}")
            goal_xyz = base_obs_xyz + np.asarray(offset, dtype=float)
            sol_raw = arm.model.inverse(goal_xyz, arm.positions(), allow_crossing=False).tolist()
            arm.move_pose(
                sol_raw,
                duration_s=1.2,
                speed=110,
                quiet=True,
                joint_order=(3, 2, 4, 5, 1, 6),
                cartesian=False,
                correct_residual=True,
                residual_limit=10,
                skip_arrived_base=True,
            )
            actual = [int(v) for v in arm.positions()]
            xyz = [round(float(v), 2) for v in arm.model.forward(actual)]
            _log_telemetry(log_path, "hello-world", label, actual, xyz, {"target_xyz": np.round(goal_xyz, 2).tolist()})
            results["steps"].append({"label": label, "raw": actual, "calibrated_xyz_mm": xyz, "target_xyz": np.round(goal_xyz, 2).tolist()})

        # 4. Open & close gripper once at observe pose
        recorder.set_label("hello-world: gripper_open_close")
        open_fb = _open_gripper(bus, GRIPPER_OPEN_RAW, cancel=cancel)
        time.sleep(0.3)
        close_fb = _close_gripper_contact(bus, GRIPPER_CLOSE_RAW, cancel=cancel)
        results["steps"].append({"label": "gripper_cycle", "open": open_fb, "close": close_fb})

        # 5. Lower smoothly to SAFE_REST_POSE_RAW (z ~= 75 mm, gently above wood desk) then torque OFF
        recorder.set_label("hello-world: return_safe_rest")
        _move_pose_interpolated(arm, SAFE_REST_POSE_RAW, max_ticks=70, dur_per_seg=1.6, speed=110, joint_order=(6, 1, 5, 4, 2, 3), cancel=cancel)
        final_raw = [int(v) for v in arm.positions()]
        final_xyz = [round(float(v), 2) for v in arm.model.forward(final_raw)]
        _log_telemetry(log_path, "hello-world", "at_safe_rest", final_raw, final_xyz)
        results["steps"].append({"label": "at_safe_rest", "raw": final_raw, "calibrated_xyz_mm": final_xyz})
        safe_parked = final_xyz[2] <= 105.0
    finally:
        failures = bus.stop("all")
        torques = [bool(bus._read_u8(i, 0x28)) for i in range(1, 7)]
        results["safe_parked"] = safe_parked
        results["torque_off_verified"] = (not failures) and all(not t for t in torques)
        results["stop_failures"] = failures
        bus.close()
    return results


def run_live_grasp(
    hw_cfg: dict,
    adv_cfg: dict,
    model: Kinematics,
    hand_eye: dict,
    port: str | None,
    recorder: WristRecorder,
    log_path: Path,
    cancel: threading.Event,
    *,
    target_query: str,
    grasp_z_mm: float,
    camera_to_gripper_mm: tuple[float, float, float],
    release_after: bool,
):
    """End-to-end physical recognition -> visual servo centering -> guarded grasp -> lift -> verify."""
    results = {"stage": "live-grasp", "query": target_query, "steps": []}
    bus = ServoBus(hw_cfg, port, stop_on_close=False)
    arm = Arm(adv_cfg, bus)
    arm.model = model
    T_gc = np.asarray(hand_eye["T_gripper_camera"], dtype=np.float64)

    def _watch():
        while not cancel.is_set():
            time.sleep(0.05)
        arm.stop()

    threading.Thread(target=_watch, daemon=True).start()

    def _base_camera_rot(raw_pos) -> np.ndarray:
        angles = arm.model.angles(raw_pos)
        T = np.eye(4, dtype=np.float64)
        for index, origin in enumerate(arm.model.chain):
            T = T @ origin
            if index < 5:
                c, s = math.cos(float(angles[index])), math.sin(float(angles[index]))
                rot = np.eye(4, dtype=np.float64)
                rot[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
                T = T @ rot
        return (T @ T_gc)[:3, :3]

    try:
        start_raw = [int(v) for v in arm.positions()]
        recorder.set_label("live-grasp: to_observe")
        lift_first = list(start_raw)
        lift_first[2] = min(start_raw[2], OBSERVE_POSE_RAW[2])
        if lift_first[2] != start_raw[2]:
            _move_pose_interpolated(arm, lift_first, max_ticks=65, dur_per_seg=1.5, speed=110, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
        _move_pose_interpolated(arm, OBSERVE_POSE_RAW, max_ticks=70, dur_per_seg=1.6, speed=120, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
        time.sleep(0.5)

        # 1. Detect & select target object on tabletop
        recorder.set_label("live-grasp: detect")
        frame, _ = recorder.latest_frame(wait_fresh=True)
        _save_png(ROOT / "logs" / "grasp_before.png", frame)
        fobj = Frame(frame, 1, time.monotonic())
        color_det = ColorShapeDetector(min_area=250)
        aruco_det = ArUcoDetector()
        candidates = [*color_det.detect(fobj), *aruco_det.detect(fobj)]
        _save_png(ROOT / "logs" / "grasp_before_detected.png", draw_detections(frame, candidates, center_fraction=0.16))
        results["initial_candidates"] = [c.to_dict() for c in candidates]
        if not candidates:
            raise RuntimeError("腕部视野中未检测到任何可抓取颜色物体或 ArUco 标记")

        selector = LocalSelector()
        sel = selector.select(target_query, candidates)
        if not sel.candidate_ids:
            raise RuntimeError(f"未找到匹配“{target_query}”的目标；当前候选：{[c.label for c in candidates]}")
        if len(sel.candidate_ids) > 1:
            raise RuntimeError(f"目标“{target_query}”存在多个候选，需更具体描述：{sel.question}")
        target = next(c for c in candidates if c.id == sel.candidate_ids[0])
        results["selected_target"] = target.to_dict()

        # 2. Closed-loop horizontal visual centering using calibrated R_bc (base XY plane only, keeping Z constant!)
        recorder.set_label(f"live-grasp: center {target.id}")
        tracker = TemplateTracker(fobj, target, max_shift=0.22, max_error=0.18)
        centered = False
        for step_idx in range(12):
            if cancel.is_set():
                raise InterruptedError("操作者中止")
            cur_frame, _ = recorder.latest_frame(wait_fresh=True)
            cur_fobj = Frame(cur_frame, step_idx + 2, time.monotonic())
            # Prefer fresh re-detection of same identity; fall back to template tracker
            fresh = [
                c
                for c in [*color_det.detect(cur_fobj), *aruco_det.detect(cur_fobj)]
                if (target.marker_id is not None and c.marker_id == target.marker_id)
                or (target.marker_id is None and c.color == target.color and c.shape == target.shape)
            ]
            tracked = min(fresh, key=lambda c: math.dist(c.center, target.center)) if fresh else tracker.update(cur_fobj)
            if tracked is None:
                raise RuntimeError("视觉居中过程中目标丢失")
            target = tracked
            dx = tracked.center[0] - 0.5
            dy = tracked.center[1] - 0.5
            if abs(dx) <= 0.08 and abs(dy) <= 0.08:
                centered = True
                _save_png(ROOT / "logs" / "grasp_centered.png", draw_detections(cur_frame, [tracked], selected_id=tracked.id, center_fraction=0.16))
                break
            # Map image error [dx, dy, 0] via R_bc to horizontal XY base step (clamped to <= 12 mm/step)
            R_bc = _base_camera_rot(arm.positions())
            cam_vec = np.array([dx * 120.0, dy * 120.0, 0.0], dtype=float)
            base_vec = R_bc @ cam_vec
            base_xy = np.array([base_vec[0], base_vec[1], 0.0], dtype=float)
            norm = float(np.linalg.norm(base_xy))
            if norm < 2.0:
                centered = True
                break
            base_xy *= min(1.0, 12.0 / norm)
            arm.cartesian_start_pose()
            arm.move_linear(base_xy)
            results["steps"].append({"label": f"center_{step_idx}", "err": [round(dx, 3), round(dy, 3)], "delta_mm": np.round(base_xy, 1).tolist()})
            time.sleep(0.2)

        if not centered:
            raise RuntimeError("视觉居中未在 12 步内收敛，终止抓取以防误碰")

        # 3. Open gripper & apply camera-to-gripper XY offset
        recorder.set_label("live-grasp: open_and_align")
        open_fb = _open_gripper(bus, GRIPPER_OPEN_RAW, cancel=cancel)
        results["steps"].append({"label": "open_gripper", "feedback": open_fb})
        ox, oy, _ = camera_to_gripper_mm
        if math.hypot(ox, oy) > 1.0:
            arm.cartesian_start_pose()
            arm.move_linear([ox, oy, 0.0])

        # 4. Guarded descent in 10mm steps to grasp_z_mm (strictly >= Z_TABLE_FLOOR_MM)
        recorder.set_label("live-grasp: descend")
        while True:
            if cancel.is_set():
                raise InterruptedError("操作者中止")
            cur_xyz = arm.model.forward(arm.positions())
            rem_z = grasp_z_mm - float(cur_xyz[2])
            if abs(rem_z) <= 2.5:
                break
            dz = math.copysign(min(abs(rem_z), 12.0), rem_z)
            if float(cur_xyz[2]) + dz < Z_TABLE_FLOOR_MM - 0.5:
                raise RuntimeError(f"下降触及桌面安全底面 ({Z_TABLE_FLOOR_MM} mm)")
            arm.cartesian_start_pose()
            arm.move_linear([0.0, 0.0, dz])
            results["steps"].append({"label": "descend", "z_mm": round(float(arm.model.forward(arm.positions())[2]), 1)})

        # 5. Contact-aware close on ID 6
        recorder.set_label("live-grasp: close_gripper")
        close_fb = _close_gripper_contact(bus, GRIPPER_CLOSE_RAW, cancel=cancel)
        results["close_feedback"] = close_fb
        time.sleep(0.3)

        # 6. Lift back to safe height (185 mm)
        recorder.set_label("live-grasp: lift")
        while True:
            if cancel.is_set():
                raise InterruptedError("操作者中止")
            cur_xyz = arm.model.forward(arm.positions())
            rem_z = 185.0 - float(cur_xyz[2])
            if rem_z <= 2.5:
                break
            dz = min(rem_z, 15.0)
            arm.cartesian_start_pose()
            arm.move_linear([0.0, 0.0, dz])

        time.sleep(0.4)
        after_frame, _ = recorder.latest_frame(wait_fresh=True)
        _save_png(ROOT / "logs" / "grasp_after_lift.png", after_frame)
        grip_after = int(bus.read_position(6))
        lo, hi = GRIPPER_HELD_RANGE
        results["lifted_gripper_raw"] = grip_after
        results["grasp_verified"] = bool(close_fb["gripped"] and lo <= grip_after <= hi)

        # 7. Optional gentle release back onto tabletop
        if release_after and results["grasp_verified"]:
            recorder.set_label("live-grasp: release")
            while True:
                cur_xyz = arm.model.forward(arm.positions())
                rem_z = (grasp_z_mm + 5.0) - float(cur_xyz[2])
                if abs(rem_z) <= 2.5:
                    break
                dz = math.copysign(min(abs(rem_z), 12.0), rem_z)
                arm.cartesian_start_pose()
                arm.move_linear([0.0, 0.0, dz])
            _open_gripper(bus, GRIPPER_OPEN_RAW, cancel=cancel)
            time.sleep(0.3)
            _move_pose_interpolated(arm, OBSERVE_POSE_RAW, max_ticks=70, dur_per_seg=1.5, speed=120, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)

        # Return to safe rest pose before turning torque OFF
        recorder.set_label("live-grasp: return_rest")
        _move_pose_interpolated(arm, SAFE_REST_POSE_RAW, max_ticks=70, dur_per_seg=1.6, speed=110, joint_order=(6, 1, 5, 4, 2, 3), cancel=cancel)
    finally:
        failures = bus.stop("all")
        torques = [bool(bus._read_u8(i, 0x28)) for i in range(1, 7)]
        results["torque_off_verified"] = (not failures) and all(not t for t in torques)
        results["stop_failures"] = failures
        bus.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="SO-101 SCS215 分阶段实机测试（已接入校准 FK/IK 与桌面底面保护）")
    parser.add_argument("--stage", choices=["gripper-check", "hello-world", "live-grasp"], required=True)
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="机械臂摄像头名称，默认 GENERAL WEBCAM")
    parser.add_argument("--port", default=None)
    parser.add_argument("--target", default="抓取红色那个", help="live-grasp 的目标描述")
    parser.add_argument("--grasp-z-mm", type=float, default=70.0, help="抓取高度（mm，必须 >= 60.0）")
    parser.add_argument("--offset-xy", type=float, nargs=2, default=[0.0, 0.0], help="相机到夹爪 XY 偏移 (mm)")
    parser.add_argument("--keep-holding", action="store_true", help="抓起核对后不自动放回")
    parser.add_argument("--dry-run", action="store_true", help="只读预检与轨迹验证，不开启任何舵机扭矩")
    args = parser.parse_args()

    hw_cfg = load_config(SCS_DIR / "config.json")
    adv_cfg = load_advanced(SCS_DIR / "advanced.json")
    model, hand_eye = load_calibrated_model(adv_cfg)

    preflight = preflight_read_servos(hw_cfg, adv_cfg, model, args.port)
    plan = check_plan(args.stage, preflight["raw"], adv_cfg, model, grasp_z_mm=args.grasp_z_mm)

    ts = time.strftime("%Y%m%d_%H%M%S")
    mp4_path = ROOT / "logs" / f"{args.stage}_{ts}.mp4"
    log_path = ROOT / "logs" / f"{args.stage}_{ts}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    idx, backend, dev_name = resolve_camera_source(args.camera)
    cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开机械臂摄像头 {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = False, None
    gaps = []
    last = None
    for idx_read in range(10):
        ok, frame = cap.read()
        now = time.monotonic()
        if ok and idx_read >= 4 and last is not None:
            gaps.append(now - last)
        last = now
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("机械臂摄像头未返回画面")

    if args.dry_run:
        out = {
            "mode": "dry-run",
            "torque_enabled": False,
            "fk_mode": "calibrated_joint_mapping (hand_eye.json)",
            "preflight": preflight,
            "camera": {
                "device_name": dev_name,
                "index": idx,
                "resolution": [frame.shape[1], frame.shape[0]],
                "max_gap_s": round(max(gaps), 4) if gaps else 0.0,
            },
            "plan": plan,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())

    def stdin_abort():
        try:
            for line in sys.stdin:
                if line.strip().lower() == "q":
                    cancel.set()
                    break
        except Exception:
            pass

    threading.Thread(target=stdin_abort, daemon=True).start()

    recorder = WristRecorder(args.camera, mp4_path)
    try:
        if args.stage == "gripper-check":
            # Reuse simple gripper cycle
            bus = ServoBus(hw_cfg, args.port, stop_on_close=False)
            try:
                open_fb = _open_gripper(bus, GRIPPER_OPEN_RAW, cancel=cancel)
                time.sleep(0.3)
                close_fb = _close_gripper_contact(bus, GRIPPER_CLOSE_RAW, cancel=cancel)
                res = {"stage": "gripper-check", "open": open_fb, "close": close_fb}
            finally:
                failures = bus.stop("all")
                torques = [bool(bus._read_u8(i, 0x28)) for i in range(1, 7)]
                res["torque_off_verified"] = (not failures) and all(not t for t in torques)
                bus.close()
        elif args.stage == "hello-world":
            res = run_hello_world_live(hw_cfg, adv_cfg, model, args.port, recorder, log_path, cancel)
        else:
            res = run_live_grasp(
                hw_cfg,
                adv_cfg,
                model,
                hand_eye,
                args.port,
                recorder,
                log_path,
                cancel,
                target_query=args.target,
                grasp_z_mm=args.grasp_z_mm,
                camera_to_gripper_mm=(float(args.offset_xy[0]), float(args.offset_xy[1]), 0.0),
                release_after=not args.keep_holding,
            )
    finally:
        saved_mp4 = recorder.close()

    res["video_mp4"] = str(saved_mp4) if saved_mp4 else None
    res["frames_recorded"] = recorder.frame_count
    res["telemetry_jsonl"] = str(log_path)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
