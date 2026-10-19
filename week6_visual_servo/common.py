"""Shared geometry and file helpers for the Week 6 exercises (millimetres)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
COURSE = ROOT.parent
ARM_CODE = COURSE / "week4" / "so101_scs215_portable"
if not ARM_CODE.is_dir():
    ARM_CODE = COURSE / "so101_scs215_portable"
if str(ARM_CODE) not in sys.path:
    sys.path.insert(0, str(ARM_CODE))


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image through Python paths (supports Chinese Windows paths)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"无法解码图像：{path}")
    return frame


def write_image(path: str | Path, frame: np.ndarray) -> None:
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, frame)
    if not ok:
        raise OSError(f"无法编码图像：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = read_json(path)
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    size = tuple(int(x) for x in data["image_size"])
    if K.shape != (3, 3) or len(size) != 2 or min(size) <= 0 or not np.isfinite(K).all():
        raise ValueError("相机内参文件格式错误")
    return K, dist, size


def board_spec(data: dict) -> tuple[int, int, float]:
    """Read board dimensions from either this project or the handoff JSON format."""
    pattern = data.get("inner_corners", data.get("pattern"))
    if not isinstance(pattern, (list, tuple)) or len(pattern) != 2:
        raise ValueError("内参文件缺少 inner_corners 或 pattern 棋盘规格")
    cols, rows = (int(value) for value in pattern)
    square_mm = float(data["square_mm"])
    if cols < 3 or rows < 3 or not math.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("内参文件中的棋盘规格无效")
    return cols, rows, square_mm


def ensure_image_size(frame: np.ndarray, expected: tuple[int, int]) -> None:
    actual = (frame.shape[1], frame.shape[0])
    if actual != expected:
        raise ValueError(f"相机分辨率 {actual} 与标定分辨率 {expected} 不同；请重新标定或设置相机分辨率")


def board_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    if cols < 3 or rows < 3 or not math.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("棋盘格内角点数量或方格边长无效")
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return points


def chessboard_corners(frame: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
    else:
        ok, corners = cv2.findChessboardCorners(gray, pattern)
    if not ok:
        return None
    if not hasattr(cv2, "findChessboardCornersSB"):
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)


def transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inverse(T: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def camera_target_pose(points: np.ndarray, corners: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    ok, rvec, tvec = cv2.solvePnP(points, corners, K, dist)
    if not ok or float(tvec[2, 0]) <= 0:
        raise ValueError("标定板位姿求解失败或位于相机后方")
    return transform(cv2.Rodrigues(rvec)[0], tvec)


def arm_model(advanced_config: str | Path | None = None):
    from advanced import Kinematics
    config = read_json(advanced_config or ARM_CODE / "advanced.json")
    model = Kinematics(config)
    if len(model.chain) != 6:
        raise ValueError("手眼标定需要完整的 SO-101 URDF 关节链")
    return model


def base_gripper_pose(raw: list[int] | np.ndarray, model) -> np.ndarray:
    """Return T_base_gripper (gripper coordinates -> base coordinates)."""
    raw = model.validate(raw)
    angles = model.angles(raw)
    T = np.eye(4)
    for index, origin in enumerate(model.chain):
        T = T @ origin
        if index < 5:
            c, s = math.cos(angles[index]), math.sin(angles[index])
            rotation = np.eye(4)
            rotation[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T = T @ rotation
    return T


def load_hand_eye(path: str | Path) -> np.ndarray:
    T = np.asarray(read_json(path)["T_gripper_camera"], dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("手眼矩阵格式错误")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=0.02):
        raise ValueError("手眼矩阵旋转部分不是正交矩阵")
    return T


def calibrated_gripper_pose(raw, model, hand_eye_data: dict) -> np.ndarray:
    """Offline FK matching an empirical hand-eye result; does not change motor control."""
    mapping = hand_eye_data.get("calibrated_joint_mapping")
    if mapping is None:
        return base_gripper_pose(raw, model)
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (6,) or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1023):
        raise ValueError("标定位姿需要六个0～1023的有限原始读数")
    center = np.asarray(mapping["raw_reference_unwrapped"], dtype=float)
    reference = np.asarray(mapping["angle_reference_deg"], dtype=float)
    rates = np.asarray(mapping["signed_degrees_per_raw"], dtype=float)
    if any(value.shape != (5,) or not np.isfinite(value).all() for value in (center, reference, rates)):
        raise ValueError("手眼结果中的配套关节映射格式错误")
    q = np.radians(reference + (model.unwrap(raw)[:5] - center) * rates)
    pose = np.eye(4)
    for index, origin in enumerate(model.chain):
        pose = pose @ origin
        if index < 5:
            c, s = math.cos(q[index]), math.sin(q[index])
            rotation = np.eye(4)
            rotation[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            pose = pose @ rotation
    return pose
