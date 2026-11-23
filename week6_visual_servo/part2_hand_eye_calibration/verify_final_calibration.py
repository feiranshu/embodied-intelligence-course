"""Reproject the saved final hand-eye result without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import arm_model, board_points, board_spec, calibration, read_image, read_json
from board_orientation import sample_board_corners
from part2_hand_eye_calibration.hand_eye import calibration_angles, score_frozen_pose_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=HERE / "最终标定结果")
    args = parser.parse_args()
    folder = args.folder
    result = read_json(folder / "hand_eye.json")
    manifest = folder / "samples.json"
    dataset_hash = result.get("calibration_dataset", {}).get("manifest_sha256")
    if dataset_hash and hashlib.sha256(manifest.read_bytes()).hexdigest() != dataset_hash:
        raise ValueError("样本清单哈希与最终结果不一致")
    model = arm_model(folder / "advanced.json")
    intrinsics = folder / "camera_intrinsics.json"
    K, dist, _ = calibration(intrinsics)
    cols, rows, square_mm = board_spec(read_json(intrinsics))
    groups = {"fit": [], "holdout": []}
    references = result.get("calibration_unwrap_references", {})
    for sample in read_json(manifest)["samples"]:
        image = read_image(folder / sample["image"])
        corners = sample_board_corners(image, sample, cols, rows)
        if corners is None:
            raise ValueError(f"无法读取 {sample['name']} 的棋盘角点")
        groups[sample["split"]].append({
            "name": sample["name"], "corners": corners,
            "angles": calibration_angles(model, sample["raw"], references),
        })
    if [item["name"] for item in groups["fit"]] != result["samples"]:
        raise ValueError("拟合样本与最终结果不一致")
    solution = {
        "camera": np.asarray(result["T_gripper_camera"]),
        "board": np.asarray(result["T_base_board"]),
        "delta": np.radians(result["joint_offsets_deg"]),
    }
    points = board_points(cols, rows, square_mm)
    checks = {label: score_frozen_pose_model(items, model, points, K, dist, solution)
              for label, items in groups.items()}
    saved = {"fit": result["rms_px_selected"], "holdout": result["holdout_validation"]["rms_px"]}
    for label, rms in saved.items():
        if not np.isclose(rms, checks[label]["rms_px"], atol=0.001):
            raise ValueError(f"{label} 重投影误差与存档不一致")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
