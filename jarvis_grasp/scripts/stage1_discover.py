"""Stage 1 read-only hardware discovery (Torque OFF) for SO-101 SCS215 + cameras."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.models import Frame
from jarvis.perception import ArUcoDetector, ColorShapeDetector, draw_detections
from vendor.scs215.advanced import Kinematics, load_advanced
from vendor.scs215.joint_mapping import check_position, mapped_position
from vendor.scs215.servo_bus import ServoBus, available_ports, load_config


def discover_serial_and_servos() -> dict:
    ports = available_ports()
    result = {"ports": ports, "connected": False, "servos": [], "error": None}
    scs_dir = ROOT / "vendor" / "scs215"
    hw_cfg = load_config(scs_dir / "config.json")
    adv_cfg = load_advanced(scs_dir / "advanced.json")
    model = Kinematics(adv_cfg)

    try:
        with ServoBus(hw_cfg, port_override=None, stop_on_close=False) as bus:
            result["connected"] = True
            result["selected_port"] = bus.port
            raw_positions = []
            servos = []
            for item in hw_cfg["servos"]:
                sid = int(item["id"])
                pos = int(bus.read_position(sid))
                raw_positions.append(pos)
                mode, min_lim, max_lim = bus.mode(sid)
                torque = bool(bus._read_u8(sid, 0x28))
                model_num = int(bus._read_u16(sid, 0x03))
                voltage_v = round(bus._read_u8(sid, 0x3E) / 10.0, 1)
                temp_c = int(bus._read_u8(sid, 0x3F))
                load_raw = int(bus._read_u16(sid, 0x3C))
                mapping = adv_cfg.get("joint_mapping", {}).get(str(sid))
                mapped = mapped_position(pos, mapping) if mapping else None
                in_mapping = True
                mapping_note = ""
                if mapping:
                    try:
                        check_position(pos, mapping, tolerance=12)
                    except ValueError as err:
                        in_mapping = False
                        mapping_note = str(err)
                servos.append(
                    {
                        "id": sid,
                        "name": item["name"],
                        "label": item.get("label", item["name"]),
                        "model_number": model_num,
                        "raw": pos,
                        "mapped": mapped,
                        "in_mapping_limits": in_mapping,
                        "mapping_note": mapping_note,
                        "torque_enabled": torque,
                        "mode": mode,
                        "hw_limits": [min_lim, max_lim],
                        "voltage_v": voltage_v,
                        "temp_c": temp_c,
                        "load_mag": load_raw & 1023,
                    }
                )
            result["servos"] = servos
            result["raw_positions"] = raw_positions
            result["all_torque_off"] = all(not s["torque_enabled"] for s in servos)
            result["all_position_mode"] = all(s["mode"] == "position" for s in servos)
            try:
                xyz = [round(float(v), 2) for v in model.forward(raw_positions)]
                result["end_effector_mm"] = xyz
                model.inverse(np.asarray(xyz), np.asarray(raw_positions), allow_crossing=False)
                result["cartesian_ready"] = True
                result["cartesian_note"] = "当前姿态在末端逆解连续工作区内"
            except Exception as err:
                result["end_effector_mm"] = (
                    [round(float(v), 2) for v in model.forward(raw_positions)]
                    if len(raw_positions) == 6
                    else None
                )
                result["cartesian_ready"] = False
                result["cartesian_note"] = str(err)
            # Compare distance to recorded poses in advanced.json
            pose_diffs = {}
            for name, pose in adv_cfg.get("poses", {}).items():
                if len(pose) == 6:
                    diffs = [abs(int(a) - int(b)) for a, b in zip(raw_positions, pose)]
                    pose_diffs[name] = max(diffs)
            result["nearest_pose"] = min(pose_diffs.items(), key=lambda kv: kv[1]) if pose_diffs else None
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _save_png(path: Path, image) -> None:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    path.write_bytes(buf.tobytes())


def _camera_names() -> list[str]:
    if os.name != "nt":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in @('Camera','Image') -and $_.Status -eq 'OK' } | Select-Object -ExpandProperty Name",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def discover_cameras(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _camera_names()
    max_index = max(len(names), 4)
    color_detector = ColorShapeDetector()
    aruco_detector = ArUcoDetector()
    cameras = []

    for index in range(max_index):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Flush buffered frames and measure freshness across 5 reads
        stamps = []
        image = None
        for _ in range(8):
            t0 = time.monotonic()
            ok, frame = cap.read()
            t1 = time.monotonic()
            if ok and frame is not None:
                image = frame
                stamps.append(t1 - t0)
        cap.release()
        if image is None:
            continue

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = round(float(gray.mean()), 1)
        if brightness < 45:
            lighting = "TOO_DARK"
        elif brightness > 225:
            lighting = "TOO_BRIGHT_OR_GLARE"
        else:
            lighting = "OK"

        raw_path = out_dir / f"camera_{index}.png"
        _save_png(raw_path, image)

        frame_obj = Frame(image, 1, time.monotonic())
        color_cands = color_detector.detect(frame_obj)
        try:
            aruco_cands = aruco_detector.detect(frame_obj)
            aruco_error = None
        except Exception as err:
            aruco_cands = []
            aruco_error = str(err)

        annotated = draw_detections(image, [*color_cands, *aruco_cands], center_fraction=0.2)
        det_path = out_dir / f"camera_{index}_detected.png"
        _save_png(det_path, annotated)

        device_name = names[index] if index < len(names) else f"Camera {index}"
        cameras.append(
            {
                "index": index,
                "name": device_name,
                "resolution": [w, h],
                "brightness": brightness,
                "lighting": lighting,
                "max_read_gap_s": round(max(stamps[-4:]), 4) if len(stamps) >= 4 else round(max(stamps), 4),
                "image_path": str(raw_path),
                "detected_path": str(det_path),
                "color_detections": [c.to_dict() for c in color_cands],
                "aruco_detections": [c.to_dict() for c in aruco_cands],
                "aruco_error": aruco_error,
            }
        )
    return cameras


def main():
    out_dir = ROOT / "data" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "serial": discover_serial_and_servos(),
        "cameras": discover_cameras(out_dir),
    }
    out_file = out_dir / "stage1_report.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
