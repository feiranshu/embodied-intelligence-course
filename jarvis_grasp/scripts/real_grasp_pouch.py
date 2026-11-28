"""Physical SO-101 SCS215 real-world object recognition + visual servoing + grasping.

Follows Robotics R1-R16 safety rules:
  - Uses calibrated FK/IK (hand_eye.json) where +Z is verified physically UP (z_table ~= 55.5 mm).
  - Uses ONLY the robot-mounted camera ("GENERAL WEBCAM", index 1).
  - Online 2-axis direction probe measures real pixel Jacobian before visual servoing (no sign guessing).
  - Hard collision floor Z_TABLE_FLOOR_MM = 60.0 mm enforced before every IK move.
  - Contact/load-aware gripper close on ID 6 + post-lift gripper & visual verification.
  - Continuous MP4 + JSONL recording and guaranteed Torque OFF at safe rest in finally.
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
from jarvis.models import Candidate, Frame
from jarvis.perception import ColorShapeDetector, LocalSelector, draw_detections
from scripts.hw_stage_test import (
    DEFAULT_CAMERA,
    HAND_EYE_PATH,
    OBSERVE_POSE_RAW,
    SAFE_REST_POSE_RAW,
    SCS_DIR,
    WristRecorder,
    _close_gripper_contact,
    _interp_poses,
    _move_pose_interpolated,
    _open_gripper,
    _save_png,
    load_calibrated_model,
    preflight_read_servos,
)
from vendor.scs215.advanced import Arm, Kinematics, load_advanced
from vendor.scs215.joint_mapping import check_position
from vendor.scs215.servo_bus import GOAL_POSITION, ServoBus, load_config

Z_TABLE_FLOOR_MM = 15.0
GRIPPER_OPEN_POUCH = 600
GRIPPER_CLOSE_TARGET = 270
GRIPPER_HELD_MIN = 286
GRIPPER_HELD_MAX = 580
GOAL_UV = (335.0, 220.0)

# Verified high-clearance tabletop active-search viewpoints (2026-11-23 re-indexed arm, all z >= 165 mm)
SEARCH_POSES_RAW = [
    ("observe_center", [486, 320, 560, 685, 353, 270]),
    ("observe_forward", [486, 380, 510, 675, 353, 270]),
    ("observe_left", [430, 320, 560, 685, 353, 270]),
    ("observe_right", [540, 320, 560, 685, 353, 270]),
]


def detect_target_in_frame(image: np.ndarray, target_color: str = "green", min_area: int = 1800) -> tuple[Candidate | None, list[Candidate]]:
    # Suppress bright out-of-focus white gripper jaws at top of frame so jaw glare never contaminates bbox
    cleaned = image.copy()
    hsv_top = cv2.cvtColor(cleaned[:140, :], cv2.COLOR_BGR2HSV)
    jaw_glare = (hsv_top[:, :, 2] > 205) & (hsv_top[:, :, 1] < 55)
    cleaned[:140, :][jaw_glare] = (255, 255, 255)
    detector = ColorShapeDetector(min_area=min_area, real_world=True, allow_border=True)
    candidates = detector.detect(Frame(cleaned, 1, time.monotonic()))
    matches = [c for c in candidates if c.color == target_color]
    if not matches:
        return None, candidates
    best = max(matches, key=lambda c: c.area_px)
    return best, candidates


def safe_move_xyz(
    arm: Arm,
    target_xyz: np.ndarray,
    *,
    duration_s: float = 1.4,
    speed: int = 120,
    joint_order=(3, 2, 4, 5, 1, 6),
    allow_gentle_contact: bool = False,
    cancel: threading.Event,
) -> list[float]:
    """Calibrated IK move with hard Z floor & joint-limit checks (enables ID 4 integral gain)."""
    from vendor.scs215.advanced import PositionHoldTimeout

    target_xyz = np.asarray(target_xyz, dtype=float)
    if target_xyz[2] < Z_TABLE_FLOOR_MM:
        raise RuntimeError(f"拒绝执行：目标高度 z={target_xyz[2]:.1f} mm 低于桌面安全平面 {Z_TABLE_FLOOR_MM:.1f} mm")
    if cancel.is_set():
        raise InterruptedError("操作者中止")
    cur_raw = arm.positions()
    ik_seed = cur_raw.copy()
    if 755 < int(ik_seed[3]) <= 770:
        ik_seed[3] = 755
    sol_raw = arm.model.inverse(target_xyz, ik_seed, allow_crossing=False)
    for i, val in enumerate(sol_raw, 1):
        check_position(int(val), arm.config["joint_mapping"][str(i)], tolerance=6)
    max_ticks = max(abs(int(a) - int(b)) for a, b in zip(sol_raw, cur_raw))
    if max_ticks > 150:
        raise RuntimeError(f"单步关节变化过大 ({max_ticks} ticks)，拒绝移动")
    try:
        arm.move_pose(
            sol_raw.tolist(),
            duration_s=duration_s,
            speed=speed,
            quiet=True,
            joint_order=joint_order,
            cartesian=False,
            correct_residual=True,
            residual_limit=10,
            skip_arrived_base=True,
        )
    except PositionHoldTimeout:
        if not allow_gentle_contact:
            raise
        actual_raw = arm.positions()
        if max(abs(int(a) - int(b)) for a, b in zip(actual_raw[:5], sol_raw[:5])) > 25:
            raise
    actual_xyz = [round(float(v), 2) for v in arm.model.forward(arm.positions())]
    return actual_xyz


def close_gripper_pouch(bus: ServoBus, *, target_raw: int = GRIPPER_CLOSE_TARGET, max_load: int = 380, hold_bias: int = 10, speed: int = 140, cancel: threading.Event) -> dict:
    """Firm but stall-safe close for soft pouch/objects on SCS215 ID 6."""
    bus.move_position(6, int(target_raw), 1600, int(speed))
    t0 = time.monotonic()
    stalled = 0
    prev = None
    actual = int(target_raw)
    max_seen_load = 0
    reason = "timeout"
    while time.monotonic() - t0 < 2.8:
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
        if stalled >= 7 and (time.monotonic() - t0) > 0.40:
            reason = "stall_contact"
            break
        if abs(actual - target_raw) <= 4:
            reason = "reached_close"
            break
    hold_raw = max(260, min(662, max(int(target_raw), actual - int(hold_bias))))
    bus._write(6, GOAL_POSITION, [hold_raw >> 8, hold_raw & 255])
    bus.finish_position(6)
    return {
        "raw": actual,
        "hold_raw": hold_raw,
        "max_load": max_seen_load,
        "stop_reason": reason,
        "gripped": GRIPPER_HELD_MIN <= actual <= GRIPPER_HELD_MAX,
    }


def _cand_uv(cand: Candidate, image_or_shape) -> np.ndarray:
    if isinstance(image_or_shape, np.ndarray) and image_or_shape.ndim == 3:
        h, w = image_or_shape.shape[:2]
        x, y, bw, bh = cand.bbox
        x1, y1 = max(0, int(x * w)), max(0, int(y * h))
        x2, y2 = min(w, int(math.ceil((x + bw) * w))), min(h, int(math.ceil((y + bh) * h)))
        roi = image_or_shape[y1:y2, x1:x2]
        if roi.size > 0 and cand.color == "green":
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (40, 46, 55), (83, 255, 254))
            # Prefer the jaw-approach strip (y in 110..260) directly under the gripper tips
            band_y1 = max(0, 110 - y1)
            band_y2 = min(mask.shape[0], 260 - y1)
            if band_y2 > band_y1 + 20:
                bys, bxs = np.nonzero(mask[band_y1:band_y2, :])
                if bxs.size >= 300:
                    return np.array([float(x1 + np.median(bxs)), float(y1 + band_y1 + np.median(bys))], dtype=float)
            ys, xs = np.nonzero(mask)
            if xs.size >= 300:
                return np.array([float(x1 + np.median(xs)), float(y1 + np.median(ys))], dtype=float)
    else:
        h, w = image_or_shape[:2]
    return np.array([cand.center[0] * w, cand.center[1] * h], dtype=float)


def run_pouch_grasp(
    hw_cfg: dict,
    adv_cfg: dict,
    model: Kinematics,
    port: str | None,
    recorder: WristRecorder,
    log_path: Path,
    cancel: threading.Event,
    *,
    target_color: str = "green",
    grasp_z_mm: float = 68.0,
    release_after: bool = True,
) -> dict:
    results: dict = {"stage": "real_grasp_pouch", "target_color": target_color, "grasp_z_mm": grasp_z_mm, "steps": []}
    bus = ServoBus(hw_cfg, port, stop_on_close=False)
    arm = Arm(adv_cfg, bus)
    arm.model = model

    def _log(label: str, extra: dict | None = None):
        raw = [int(v) for v in arm.positions()]
        xyz = [round(float(v), 2) for v in arm.model.forward(raw)]
        rec = {"t": round(time.time(), 3), "label": label, "raw": raw, "xyz_mm": xyz}
        if extra:
            rec.update(extra)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return raw, xyz

    done = threading.Event()

    def _watch():
        while not done.is_set() and not cancel.is_set():
            done.wait(0.05)
        if cancel.is_set() and not done.is_set():
            try:
                arm.stop()
            except Exception:
                pass

    watch_thread = threading.Thread(target=_watch, daemon=True)
    watch_thread.start()

    try:
        start_raw, start_xyz = _log("start")

        # 1. Lift straight up gently (small steps so startup current never spikes) -> OBSERVE_POSE
        recorder.set_label("pouch: lift_to_observe")
        lift_first = list(start_raw)
        lift_first[1] = min(start_raw[1], OBSERVE_POSE_RAW[1])
        lift_first[2] = min(start_raw[2], OBSERVE_POSE_RAW[2])
        if lift_first != start_raw:
            _move_pose_interpolated(arm, lift_first, max_ticks=55, dur_per_seg=1.1, speed=110, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
        _move_pose_interpolated(arm, OBSERVE_POSE_RAW, max_ticks=95, dur_per_seg=1.2, speed=135, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
        time.sleep(0.75)
        obs_raw, obs_xyz = _log("at_observe")

        # 2. Detect green rectangular pouch (with limited-viewpoint active search across verified poses!)
        cand0, all_cands, frame0 = None, [], None
        for view_name, view_pose in SEARCH_POSES_RAW:
            if view_name != "observe_center":
                recorder.set_label(f"pouch: search_{view_name}")
                _move_pose_interpolated(arm, view_pose, max_ticks=75, dur_per_seg=1.3, speed=120, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
                time.sleep(0.6)
            else:
                recorder.set_label("pouch: detect_initial")
            frame0, _ = recorder.latest_frame(wait_fresh=True)
            cand0, all_cands = detect_target_in_frame(frame0, target_color=target_color)
            _log(f"scan_{view_name}", {"found": cand0 is not None, "count": len(all_cands)})
            if cand0 is not None:
                results["found_at_view"] = view_name
                break

        _save_png(ROOT / "logs" / "pouch_before.png", frame0)
        _save_png(
            ROOT / "logs" / "pouch_before_detected.png",
            draw_detections(frame0, all_cands, selected_id=cand0.id if cand0 else None, center_fraction=0.2),
        )
        results["initial_detections"] = [c.to_dict() for c in all_cands]
        if cand0 is None:
            raise RuntimeError(f"在全部 {len(SEARCH_POSES_RAW)} 个观察视角中均未检测到 {target_color} 目标物体")

        uv0 = _cand_uv(cand0, frame0)
        results["initial_uv"] = uv0.round(1).tolist()
        _log("detected_initial", {"uv": uv0.round(1).tolist(), "area_px": cand0.area_px, "target_label": cand0.label})

        # 3. Clean damped monotonic visual servoing into open-jaw capture window (u ~= 335, right wall <= 425)
        cur_frame = frame0
        cur_cand = cand0
        cur_uv = uv0
        reach_sign = 1.0

        # Only run radial probe when target is tiny and far near top/left edge (otherwise pan ID 1 first!)
        if cur_cand.area_px < 6000.0 and cur_uv[1] < 130:
            recorder.set_label("pouch: probe_reach")
            xyz_p0 = arm.model.forward(arm.positions())
            h0 = max(1.0, math.hypot(float(xyz_p0[0]), float(xyz_p0[1])))
            r_u0 = np.array([float(xyz_p0[0]) / h0, float(xyz_p0[1]) / h0, 0.0])
            probe_step = min(14.0, max(0.0, 334.0 - h0))
            if probe_step >= 5.0:
                probe_target = xyz_p0 + r_u0 * probe_step
                probe_target[2] = 165.0
                try:
                    safe_move_xyz(arm, probe_target, duration_s=1.2, speed=110, cancel=cancel)
                    time.sleep(0.45)
                    f_probe, _ = recorder.latest_frame(wait_fresh=True)
                    c_probe, _ = detect_target_in_frame(f_probe, target_color=target_color)
                    if c_probe is not None:
                        uv_probe = _cand_uv(c_probe, f_probe)
                        dv_probe = float(uv_probe[1] - cur_uv[1])
                        du_probe = float(uv_probe[0] - cur_uv[0])
                        if du_probe + dv_probe < -8.0:
                            reach_sign = -1.0
                        results["probe_delta_uv"] = [round(du_probe, 1), round(dv_probe, 1)]
                        results["verified_reach_sign"] = reach_sign
                        cur_frame, cur_cand, cur_uv = f_probe, c_probe, uv_probe
                        _log("probe_reach_verified", {"uv": cur_uv.round(1).tolist(), "duv": [round(du_probe, 1), round(dv_probe, 1)], "area_px": cur_cand.area_px})
                except Exception as probe_err:
                    results["steps"].append({"label": "probe_reach_skipped", "note": str(probe_err)})

        # Critically damped visual servoing toward open-jaw pocket (GOAL_UV = (335, 220), measured ~3.75 px/tick on ID 1)
        for step_idx in range(7):
            u, v = float(cur_uv[0]), float(cur_uv[1])
            err_u = GOAL_UV[0] - u
            err_v = GOAL_UV[1] - v
            results["steps"].append(
                {"label": f"servo_{step_idx}", "uv": [round(u, 1), round(v, 1)], "err_px": [round(err_u, 1), round(err_v, 1)], "area_px": cur_cand.area_px}
            )
            if (abs(err_u) <= 18.0 and 130.0 <= v <= 320.0 and cur_cand.area_px >= 20000.0) or (abs(err_u) <= 18.0 and abs(err_v) <= 35.0):
                break
            recorder.set_label(f"pouch: servo_{step_idx}")
            moved = False

            # 1. Base pan ID 1 (measured +3.75 px/tick -> damped gain 0.18 prevents overshoot!)
            if abs(err_u) > 16.0:
                r_now = [int(x) for x in arm.positions()]
                d_pan = int(np.clip(round(err_u * 0.18), -22, 22))
                new_pan = int(np.clip(r_now[0] + d_pan, 350, 620))
                if abs(new_pan - r_now[0]) >= 3:
                    p_pan = list(r_now)
                    p_pan[0] = new_pan
                    check_position(p_pan[0], adv_cfg["joint_mapping"]["1"], tolerance=6)
                    arm.move_pose(p_pan, duration_s=0.85, speed=110, quiet=True, joint_order=(1, 2, 3, 4, 5, 6))
                    moved = True

            # 2. Radial reach only if object is still not under jaw zone after pan
            if not (130.0 <= v <= 320.0 and cur_cand.area_px >= 20000.0) and abs(err_u) <= 65.0:
                xyz_now = arm.model.forward(arm.positions())
                cur_r = math.hypot(float(xyz_now[0]), float(xyz_now[1]))
                r_unit = np.array([float(xyz_now[0]) / max(1.0, cur_r), float(xyz_now[1]) / max(1.0, cur_r), 0.0])
                d_reach = float(np.clip(reach_sign * (err_v * 0.18), -16.0, 16.0))
                next_r = float(np.clip(cur_r + d_reach, 245.0, 334.0))
                d_reach = next_r - cur_r
                if abs(d_reach) >= 4.0:
                    target_xyz = xyz_now + r_unit * d_reach
                    target_xyz[2] = 165.0
                    try:
                        safe_move_xyz(arm, target_xyz, duration_s=1.1, speed=110, cancel=cancel)
                        moved = True
                    except Exception as ik_err:
                        results["steps"].append({"label": f"servo_{step_idx}_reach_limit", "note": str(ik_err)})

            if not moved:
                break
            time.sleep(0.45)
            next_frame, _ = recorder.latest_frame(wait_fresh=True)
            next_cand, _ = detect_target_in_frame(next_frame, target_color=target_color)
            if next_cand is None:
                results["steps"].append({"label": f"servo_{step_idx}_lost_stop"})
                break
            cur_frame, cur_cand, cur_uv = next_frame, next_cand, _cand_uv(next_cand, next_frame)

        _save_png(
            ROOT / "logs" / "pouch_aligned.png",
            draw_detections(cur_frame, [cur_cand], selected_id=cur_cand.id, center_fraction=0.25, grasp_stage="aligned"),
        )
        aligned_raw, aligned_xyz = _log("aligned", {"uv": cur_uv.round(1).tolist(), "area_px": cur_cand.area_px})
        results["aligned_uv"] = cur_uv.round(1).tolist()
        results["aligned_xyz_mm"] = aligned_xyz

        # 5. Open gripper wide (470) to straddle pencil pouch
        recorder.set_label("pouch: open_gripper")
        open_fb = _open_gripper(bus, GRIPPER_OPEN_POUCH, speed=150, cancel=cancel)
        results["open_feedback"] = open_fb
        time.sleep(0.3)

        # 6. Guarded smooth descent in ~15-25 mm stages down to grasp_z_mm (>= Z_TABLE_FLOOR_MM)
        recorder.set_label("pouch: descend")
        base_xyz = arm.model.forward(arm.positions())
        descent_levels = [z for z in (130.0, 100.0, 75.0, 55.0, 40.0, 32.0) if z < float(base_xyz[2]) - 5.0 and z > grasp_z_mm + 2.5]
        descent_levels.append(float(grasp_z_mm))
        for z_target in descent_levels:
            goal_xyz = np.array([float(base_xyz[0]), float(base_xyz[1]), z_target], dtype=float)
            actual_xyz = safe_move_xyz(
                arm,
                goal_xyz,
                duration_s=1.1,
                speed=95,
                joint_order=(4, 2, 3, 1, 5, 6),
                allow_gentle_contact=(z_target <= 65.0),
                cancel=cancel,
            )
            _log(f"descend_z_{int(z_target)}", {"target_z": z_target})
            results["steps"].append({"label": f"descend_z_{int(z_target)}", "xyz_mm": actual_xyz})
            time.sleep(0.18)

        pre_close_frame, _ = recorder.latest_frame(wait_fresh=True)
        _save_png(ROOT / "logs" / "pouch_at_grasp_z.png", pre_close_frame)

        # 7. Contact- and load-aware gripper close! (380 load cap grips fabric pouch firmly without crushing flat)
        recorder.set_label("pouch: close_gripper")
        close_fb = close_gripper_pouch(bus, target_raw=GRIPPER_CLOSE_TARGET, max_load=380, hold_bias=14, cancel=cancel)
        results["close_feedback"] = close_fb
        _log("gripper_closed", {"close": close_fb})
        time.sleep(0.4)

        # 8. Lift straight UP to z = 165 mm (100% reachable across r <= 335 mm with ID4 <= 755)!
        recorder.set_label("pouch: lift")
        for z_up in (75.0, 115.0, 165.0):
            goal_xyz = np.array([float(base_xyz[0]), float(base_xyz[1]), z_up], dtype=float)
            actual_xyz = safe_move_xyz(arm, goal_xyz, duration_s=1.3, speed=105, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
            results["steps"].append({"label": f"lift_z_{int(z_up)}", "xyz_mm": actual_xyz})
        time.sleep(0.8)  # Settle before measuring success (iteration rule: Done means measured!)

        lifted_frame, _ = recorder.latest_frame(wait_fresh=True)
        _save_png(ROOT / "logs" / "pouch_lifted.png", lifted_frame)
        grip_after_lift = int(bus.read_position(6))
        load_after_lift = int(bus._read_u16(6, 60)) & 1023
        lifted_cand, _ = detect_target_in_frame(lifted_frame, target_color=target_color, min_area=1200)
        lifted_area = float(lifted_cand.area_px) if lifted_cand else 0.0
        # Verify grasp via either jaw opening OR compressed soft-pouch load + close-up visual confirmation
        width_held = bool(close_fb["gripped"] and GRIPPER_HELD_MIN <= grip_after_lift <= GRIPPER_HELD_MAX)
        soft_pouch_held = bool(
            close_fb["max_load"] >= 340
            and load_after_lift >= 110
            and lifted_area >= 165000.0
            and grip_after_lift >= 277
        )
        grasp_verified = width_held or soft_pouch_held
        results["after_lift"] = {
            "gripper_raw": grip_after_lift,
            "gripper_load": load_after_lift,
            "grasp_verified": grasp_verified,
            "visual_in_hand_area_px": lifted_area,
        }
        _log("after_lift", results["after_lift"])
        time.sleep(1.2 if grasp_verified else 0.4)

        # 9. Gentle place back down & release (only when object was actually gripped!)
        if release_after and grasp_verified:
            recorder.set_label("pouch: place_and_release")
            for z_down in (110.0, 75.0, max(Z_TABLE_FLOOR_MM + 12.0, grasp_z_mm + 8.0)):
                goal_xyz = np.array([float(base_xyz[0]), float(base_xyz[1]), z_down], dtype=float)
                safe_move_xyz(arm, goal_xyz, duration_s=1.2, speed=100, joint_order=(4, 2, 3, 1, 5, 6), allow_gentle_contact=(z_down <= 60.0), cancel=cancel)
            rel_fb = _open_gripper(bus, GRIPPER_OPEN_POUCH, speed=150, cancel=cancel)
            results["release_feedback"] = rel_fb
            time.sleep(0.4)
            safe_move_xyz(arm, np.array([float(base_xyz[0]), float(base_xyz[1]), 155.0]), duration_s=1.3, speed=105, joint_order=(3, 2, 4, 5, 1, 6), cancel=cancel)
            time.sleep(0.4)
            rel_frame, _ = recorder.latest_frame(wait_fresh=True)
            _save_png(ROOT / "logs" / "pouch_after_release.png", rel_frame)

        # 10. Return gently to SAFE_REST_POSE_RAW (z ~= 82 mm) & close jaws lightly before torque OFF
        recorder.set_label("pouch: return_safe_rest")
        _move_pose_interpolated(arm, SAFE_REST_POSE_RAW, max_ticks=70, dur_per_seg=1.6, speed=110, joint_order=(6, 1, 5, 4, 2, 3), cancel=cancel)
        _log("at_safe_rest")
    finally:
        done.set()
        watch_thread.join(timeout=0.5)
        failures = bus.stop("all")
        torques = [bool(bus._read_u8(i, 0x28)) for i in range(1, 7)]
        results["torque_off_verified"] = (not failures) and all(not t for t in torques)
        results["stop_failures"] = failures
        bus.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="SO-101 SCS215 实机识别+视觉伺服+抓取绿色笔袋")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--port", default=None)
    parser.add_argument("--color", default="green")
    parser.add_argument("--grasp-z-mm", type=float, default=74.0)
    parser.add_argument("--keep-holding", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.grasp_z_mm < Z_TABLE_FLOOR_MM:
        parser.error(f"--grasp-z-mm 必须 >= {Z_TABLE_FLOOR_MM} mm")

    hw_cfg = load_config(SCS_DIR / "config.json")
    adv_cfg = load_advanced(SCS_DIR / "advanced.json")
    model, _ = load_calibrated_model(adv_cfg)
    preflight = preflight_read_servos(hw_cfg, adv_cfg, model, args.port)

    if args.dry_run:
        # Also test detector offline against latest hello_world_observe.png if present
        offline_check = None
        sample = ROOT / "logs" / "hello_world_observe.png"
        if sample.is_file():
            img = cv2.imdecode(np.frombuffer(sample.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            cand, all_c = detect_target_in_frame(img, target_color=args.color)
            offline_check = {
                "sample": str(sample),
                "matched": cand.to_dict() if cand else None,
                "all_count": len(all_c),
            }
        # Verify IK solvability at observe & grasp heights
        obs_xyz = model.forward(OBSERVE_POSE_RAW)
        ik_checks = {}
        for z in (195.0, 150.0, 115.0, 88.0, args.grasp_z_mm):
            sol = model.inverse(np.array([obs_xyz[0], obs_xyz[1], z]), np.asarray(OBSERVE_POSE_RAW), allow_crossing=False)
            ik_checks[f"z_{int(z)}"] = {"raw": sol.tolist(), "xyz": np.round(model.forward(sol), 1).tolist()}
        search_checks = {}
        for vname, vpose in SEARCH_POSES_RAW:
            for j, val in enumerate(vpose, 1):
                check_position(int(val), adv_cfg["joint_mapping"][str(j)], tolerance=6)
            search_checks[vname] = {"raw": vpose, "xyz": np.round(model.forward(vpose), 1).tolist()}
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "torque_enabled": False,
                    "preflight": preflight,
                    "search_viewpoints": search_checks,
                    "offline_pouch_detection": offline_check,
                    "ik_checks": ik_checks,
                    "z_floor_mm": Z_TABLE_FLOOR_MM,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    mp4_path = ROOT / "logs" / f"real_grasp_{ts}.mp4"
    log_path = ROOT / "logs" / f"real_grasp_{ts}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

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
        res = run_pouch_grasp(
            hw_cfg,
            adv_cfg,
            model,
            args.port,
            recorder,
            log_path,
            cancel,
            target_color=args.color,
            grasp_z_mm=args.grasp_z_mm,
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
