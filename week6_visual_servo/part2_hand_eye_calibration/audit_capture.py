"""Fit from the designated poses and score the frozen model on other poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arm_model, board_points, board_spec, calibration, read_json, write_json
from part2_hand_eye_calibration.hand_eye import (calibration_angles, calibration_unwrap_references,
                                                _solve_with_joint_offsets, score_frozen_pose_model as score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="只读采样目录，须含配置快照和 samples.json")
    parser.add_argument("--output", type=Path, help="可选核验结果 JSON")
    parser.add_argument("--board-origin-mm", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="棋盘 0 号内角点相对基座原点的独立测量值")
    args = parser.parse_args()
    folder = args.folder
    intrinsics = folder / "camera_intrinsics.json"
    model = arm_model(folder / "advanced.json")
    K, dist, _ = calibration(intrinsics)
    cols, rows, square_mm = board_spec(read_json(intrinsics))
    points = board_points(cols, rows, square_mm)
    groups = {"fit": [], "holdout": []}
    names = set()
    for item in read_json(folder / "samples.json")["samples"]:
        name = item["name"]
        if name in names or item["split"] not in groups:
            raise ValueError(f"样本名重复或分组无效：{name}")
        names.add(name)
        raw = np.asarray(item["raw"], dtype=int)
        corners = np.asarray(item["ordered_corners"], dtype=np.float32).reshape(-1, 1, 2)
        if raw.shape != (6,) or np.any((raw < 0) | (raw > 1023)) or len(corners) != len(points) or not np.isfinite(corners).all():
            raise ValueError(f"无效的关节位置或棋盘角点：{name}")
        groups[item["split"]].append({"name": name, "raw": raw, "corners": corners})

    if len(groups["fit"]) < 12 or len(groups["holdout"]) < 6:
        raise ValueError(f"需要至少 12 张拟合图和 6 张留出图，当前为 {len(groups['fit'])} 和 {len(groups['holdout'])}")
    fit_poses = {tuple(item["raw"][:5]) for item in groups["fit"]}
    validation_poses = {tuple(item["raw"][:5]) for item in groups["holdout"]}
    if len(fit_poses) < 12 or len(validation_poses - fit_poses) < 6:
        raise ValueError("拟合和留出需要不同姿态，调整任意一轴的位置后重新采样")

    references = calibration_unwrap_references(model, [item["raw"] for item in groups["fit"]])
    for observations in groups.values():
        for item in observations:
            item["angles"] = calibration_angles(model, item["raw"], references)
    solved = _solve_with_joint_offsets(groups["fit"], model, K, dist, (cols, rows), square_mm)
    solution = solved["selected"]
    result = {"method": solved["selected_method"], "T_gripper_camera": solution["camera"].tolist(),
              "T_base_board": solution["board"].tolist(), "joint_offsets_deg": np.degrees(solution["delta"]).tolist(),
              "calibration_unwrap_references": references}
    for label, observations in groups.items():
        result[label] = score(observations, model, points, K, dist, solution)
        result[label]["joint_angle_spans_deg"] = np.degrees(np.ptp([item["angles"] for item in observations], axis=0)).tolist()
    if args.board_origin_mm is not None:
        difference = solution["board"][:3, 3] - np.asarray(args.board_origin_mm)
        if not np.isfinite(difference).all():
            raise ValueError("棋盘原点需要有限的毫米坐标")
        result["independent_board_origin_error_mm"] = difference.tolist()
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
