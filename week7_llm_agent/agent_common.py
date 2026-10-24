"""Shared paths and helpers for the Week 7 LLM-agent exercises."""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COURSE = ROOT.parent
WEEK6 = COURSE / "week6_visual_servo"
ARM_CODE = COURSE / "week4" / "so101_scs215_portable"
for path in (ROOT, WEEK6, ARM_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

RUNS = ROOT / "runs"
ADVANCED_CONFIG = ARM_CODE / "advanced.json"
HARDWARE_CONFIG = ARM_CODE / "config.json"
INTRINSICS = WEEK6 / "part1_camera_calibration" / "camera_intrinsics.json"
HAND_EYE = WEEK6 / "part2_hand_eye_calibration" / "最终标定结果" / "hand_eye.json"


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_dir(name: str) -> Path:
    """Create runs/<name>_<timestamp>/ for photos, raw replies and transcripts."""
    path = RUNS / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def jpeg_data_url(frame, max_side: int = 768, quality: int = 85) -> str:
    """Encode a BGR frame as a data URL accepted by OpenAI-compatible VLM APIs."""
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("无法把画面编码为 JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates ```json fences and prose)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型没有返回文本")
    decoder = json.JSONDecoder()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    sources = ([fenced.group(1)] if fenced else []) + [text]
    for source in sources:
        start = source.find("{")
        while start != -1:
            try:
                value, _ = decoder.raw_decode(source[start:])
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                return value
            start = source.find("{", start + 1)
    raise ValueError(f"模型回复中没有可解析的 JSON 对象：{text[:200]}")
