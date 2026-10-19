"""Part 2: collect paired robot/camera poses and solve eye-in-hand calibration."""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_utils import DEFAULT_CAMERA, open_camera  # noqa: E402
from board_orientation import sample_board_corners  # noqa: E402
from common import (ARM_CODE, arm_model, base_gripper_pose, board_points, board_spec, calibration,
                    camera_target_pose, chessboard_corners, ensure_image_size,
                    inverse, read_image, read_json, transform, write_image, write_json)  # noqa: E402


HERE = Path(__file__).resolve().parent


def read_plan(path: Path, model) -> list[dict]:
    poses = read_json(path).get("poses", [])
    if len(poses) < 12:
        raise ValueError("姿态计划至少需要 12 个安全姿态")
    names = []
    for index, item in enumerate(poses, 1):
        name = item.get("name", f"pose_{index:02d}")
        if not name.replace("_", "").isalnum() or name in names:
            raise ValueError(f"姿态名称不合法或重复：{name}")
        names.append(name)
        item["name"] = name
        item["raw"] = model.validate(item["raw"]).tolist()
        item["duration_s"] = float(item.get("duration_s", 3.0))
        item["speed"] = int(item.get("speed", 150))
        if not 0.5 <= item["duration_s"] <= 30 or not 1 <= item["speed"] <= 1000:
            raise ValueError(f"姿态 {name} 的时间或速度参数无效")
    return poses


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _fk_pose(model, angles: np.ndarray) -> np.ndarray:
    """Forward kinematics for five revolute joints, with optional angle offsets."""
    pose = np.eye(4, dtype=np.float64)
    for index, origin in enumerate(model.chain):
        pose = pose @ origin
        if index < 5:
            c, s = math.cos(angles[index]), math.sin(angles[index])
            rotation = np.eye(4)
            rotation[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            pose = pose @ rotation
    return pose


def _pose_from_vector(rotation_vector: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return transform(cv2.Rodrigues(np.asarray(rotation_vector, dtype=float).reshape(3, 1))[0],
                     np.asarray(translation, dtype=float).reshape(3))


def _pose_to_vector(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (cv2.Rodrigues(pose[:3, :3])[0].reshape(3), pose[:3, 3].reshape(3))


def _closed_form(observations, model, delta, outline, method):
    grippers = [_fk_pose(model, item["angles"] + delta) for item in observations]
    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in grippers], [pose[:3, 3] for pose in grippers],
        [item["T_cm"][:3, :3] for item in observations],
        [item["T_cm"][:3, 3] for item in observations], method=method)
    camera = transform(rotation, translation)
    boards = [gripper @ camera @ item["T_cm"] for gripper, item in zip(grippers, observations)]
    corners = np.asarray([(board[:3, :3] @ outline.T).T + board[:3, 3] for board in boards])
    spread = float(np.mean(np.linalg.norm(corners - corners.mean(axis=0), axis=2)))
    if not np.isfinite(spread):
        raise ValueError("手眼初值无效")
    medoid_index = int(np.argmin([np.sum(np.linalg.norm(corners - candidate, axis=2))
                                 for candidate in corners]))
    return camera, boards[medoid_index], spread, boards


def _refine_hand_eye(observations, model, matrix, dist, object_points, camera, board,
                     offset_joints, delta0=None):
    """Refine camera, fixed board and selected constant joint-angle offsets from all corners."""
    offset_joints = list(offset_joints)
    delta0 = np.zeros(5) if delta0 is None else np.asarray(delta0, dtype=float)

    def unpack(parameters):
        delta = delta0.copy()
        delta[offset_joints] = parameters[12:]
        camera_pose = _pose_from_vector(parameters[:3], parameters[3:6])
        board_pose = _pose_from_vector(parameters[6:9], parameters[9:12])
        return camera_pose, board_pose, delta

    def residual(parameters):
        camera_pose, board_pose, delta = unpack(parameters)
        errors = []
        for item in observations:
            camera_board = inverse(_fk_pose(model, item["angles"] + delta) @ camera_pose) @ board_pose
            rvec, tvec = _pose_to_vector(camera_board)
            if tvec[2] <= 1:
                errors.append(np.full(item["corners"].size, 1e4))
                continue
            projected = cv2.projectPoints(object_points, rvec, tvec, matrix, dist)[0].reshape(-1, 2)
            errors.append((projected - item["corners"].reshape(-1, 2)).ravel())
        return np.concatenate(errors)

    params = np.concatenate([*_pose_to_vector(camera), *_pose_to_vector(board), delta0[offset_joints]])
    steps = np.asarray([1e-6] * 3 + [1e-4] * 3 + [1e-6] * 3 + [1e-4] * 3
                       + [1e-6] * len(offset_joints))
    damping = 1e-3
    error = residual(params)
    with np.errstate(over="ignore", invalid="ignore"):
        cost = float(error @ error)
    if not np.isfinite(cost):
        raise ValueError("当前手眼初值无法产生有限的重投影误差")

    for _ in range(200):
        jacobian = np.column_stack([
            (residual(params + np.eye(len(params))[index] * steps[index]) - error) / steps[index]
            for index in range(len(params))
        ])
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ error
        improved = False
        previous_cost = cost
        while damping < 1e12:
            diagonal = np.diag(np.diag(normal) + 1e-12)
            try:
                step = -np.linalg.solve(normal + damping * diagonal, gradient)
            except np.linalg.LinAlgError:
                damping *= 4
                continue
            trial_error = residual(params + step)
            with np.errstate(over="ignore", invalid="ignore"):
                trial_cost = float(trial_error @ trial_error)
            if trial_cost < cost:
                params += step
                error, cost = trial_error, trial_cost
                damping = max(damping / 3, 1e-12)
                improved = True
                break
            damping *= 4
        if not improved or previous_cost - cost < 1e-9 * max(previous_cost, 1):
            break

    camera_pose, board_pose, delta = unpack(params)
    # Recompute the final Jacobian at the selected solution for a useful uncertainty estimate.
    jacobian = np.column_stack([
        (residual(params + np.eye(len(params))[index] * steps[index]) - error) / steps[index]
        for index in range(len(params))
    ])
    dof = max(1, error.size - len(params))
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * cost / dof
        standard_deviation = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:
        standard_deviation = np.full(len(params), np.inf)
    per_image_rms = [float(np.sqrt(np.mean(item_error.reshape(-1, 2) ** 2) * 2))
                     for item_error in np.split(error, len(observations))]
    return {
        "camera": camera_pose,
        "board": board_pose,
        "delta": delta,
        "delta_std": dict(zip(offset_joints, standard_deviation[12:])),
        "rms_px": float(math.sqrt(cost / (error.size / 2))),
        "per_image_rms_px": per_image_rms,
    }


def _solve_with_joint_offsets(observations, model, matrix, dist, pattern, square_mm):
    if len(observations) < 12:
        raise ValueError(f"有效视角只有 {len(observations)} 张；至少需要 12 张")
    cols, rows = pattern
    object_points = board_points(cols, rows, square_mm)
    outline = np.asarray([[0, 0, 0], [(cols - 1) * square_mm, 0, 0],
                          [(cols - 1) * square_mm, (rows - 1) * square_mm, 0],
                          [0, (rows - 1) * square_mm, 0]], dtype=float)
    methods = {}
    for item in observations:
        item["T_cm"] = camera_target_pose(object_points, item["corners"], matrix, dist)
    for name, method in HAND_EYE_METHODS.items():
        try:
            camera_pose, board_pose, spread, _ = _closed_form(observations, model, np.zeros(5), outline, method)
            refined = _refine_hand_eye(observations, model, matrix, dist, object_points,
                                       camera_pose, board_pose, [])
            if not np.isfinite(refined["rms_px"]):
                continue
            try:
                offset_refined = _refine_hand_eye(
                    observations, model, matrix, dist, object_points,
                    refined["camera"], refined["board"], [1, 2, 3])
            except ValueError:
                offset_refined = refined
            methods[name] = {"camera": camera_pose, "board_corner_scatter_mm": spread,
                             "nominal": refined, "corrected": offset_refined,
                             "identity_initial_seed": bool(np.array_equal(camera_pose, np.eye(4)))}
        except (cv2.error, ValueError):
            continue
    if not methods:
        raise ValueError("OpenCV 手眼初值均失败；检查姿态旋转多样性和棋盘角点")

    # Compare the actual reprojection fit, rather than selecting an unrefined seed.
    best_method = min(methods, key=lambda name: min(methods[name]["nominal"]["rms_px"],
                                                  methods[name]["corrected"]["rms_px"]))
    nominal = methods[best_method]["nominal"]

    # Grid-search shoulder/elbow bias first: the encoder dead band can produce a large offset.
    grid = np.radians(np.arange(-90, 91, 10))
    start = np.zeros(5)
    start_spread = math.inf
    for shoulder, elbow in itertools.product(grid, repeat=2):
        candidate = np.zeros(5)
        candidate[1], candidate[2] = shoulder, elbow
        try:
            spread = _closed_form(observations, model, candidate, outline,
                                  HAND_EYE_METHODS[best_method])[2]
        except (cv2.error, ValueError):
            continue
        if spread < start_spread:
            start, start_spread = candidate, spread
    camera_pose, board_pose, _, _ = _closed_form(
        observations, model, start, outline, HAND_EYE_METHODS[best_method])
    corrected = methods[best_method]["corrected"]
    try:
        grid_corrected = _refine_hand_eye(observations, model, matrix, dist, object_points,
                                         camera_pose, board_pose, [1, 2, 3], start)
        if grid_corrected["rms_px"] < corrected["rms_px"]:
            corrected = grid_corrected
    except ValueError:
        pass

    unreliable = [joint for joint, std in corrected["delta_std"].items()
                  if not std < math.radians(3)]
    if unreliable:
        # Report uncertainty without silently changing the fitted angle offsets.
        corrected["unobservable"] = unreliable

    selected = corrected if corrected["rms_px"] < nominal["rms_px"] else nominal
    grippers = [_fk_pose(model, item["angles"] + selected["delta"]) for item in observations]
    board_poses = [gripper @ selected["camera"] @ item["T_cm"]
                   for gripper, item in zip(grippers, observations)]
    origin_per_image = np.asarray([pose[:3, 3] for pose in board_poses])
    return {
        "methods": methods,
        "selected_method": best_method,
        "nominal": nominal,
        "corrected": corrected,
        "selected": selected,
        "board_poses": board_poses,
        "board_origin_per_image": origin_per_image,
    }


def _kinematics_patch(config, model, raw_poses, delta):
    kinematics = copy.deepcopy(config["kinematics"])
    raw_poses = np.asarray(raw_poses, dtype=int)
    for index in np.flatnonzero(np.abs(delta) > math.radians(0.5)):
        row = raw_poses[np.argsort(raw_poses[:, index])[len(raw_poses) // 2]]
        kinematics["zero_raw"][index] = int(row[index])
        kinematics["reference_angles_deg"][index] = float(
            math.degrees(model.angles(row)[index] + delta[index]))
    return {key: kinematics[key] for key in ("zero_raw", "reference_angles_deg")}


def collect(args):
    model = arm_model(args.advanced_config)
    poses = read_plan(args.plan, model)
    print(f"已预检 {len(poses)} 个目标；程序不进行碰撞检测。")
    for item in poses:
        print(f"  {item['name']}: {item['raw']}")
    if not args.execute:
        print("仅完成离线预检。现场确认相机固定、棋盘固定、路径清空后加 --execute。")
        return
    K, dist, expected_size = calibration(args.intrinsics)
    board = read_json(args.intrinsics)
    cols, rows, _ = board_spec(board)
    camera, selected_index, backend, device_name = open_camera(args.camera, expected_size)
    print(f"使用摄像头“{device_name}”（索引 {selected_index}，{backend}）")
    ok, probe = camera.read()
    if not ok:
        camera.release()
        raise RuntimeError("相机读取失败")
    try:
        ensure_image_size(probe, expected_size)
    except BaseException:
        camera.release()
        raise
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "samples.json"
    samples = read_json(manifest).get("samples", []) if manifest.exists() else []
    completed = {sample["name"] for sample in samples}
    from advanced import Arm
    from servo_bus import ServoBus, load_config
    bus = None
    arm = None
    try:
        bus = ServoBus(load_config(args.hardware_config), args.port)
        arm = Arm(read_json(args.advanced_config), bus)
        for item in poses:
            if item["name"] in completed:
                continue
            response = input(f"{item['name']}：确认操作者在场、路径清空、可断电后输入 MOVE；Q 退出：").strip()
            if response != "MOVE":
                print("停止采集；已保存的样本可供下次继续。")
                break
            arm.move_pose(item["raw"], duration_s=item["duration_s"], speed=item["speed"])
            time.sleep(0.4)
            frame = None
            for _ in range(5):
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("摄像头读取失败")
            ensure_image_size(frame, expected_size)
            corners = chessboard_corners(frame, cols, rows)
            if corners is None:
                print("棋盘角点不完整，此姿态未保存；请修订计划后重试。")
                continue
            actual = arm.positions().tolist()
            model.validate(actual)
            image_name = f"{item['name']}.png"
            write_image(folder / image_name, frame)
            samples.append({"name": item["name"], "image": image_name, "raw": actual})
            write_json(manifest, {"camera_calibration_file": Path(args.intrinsics).name,
                                  "samples": samples})
            completed.add(item["name"])
            print(f"已保存 {item['name']}：关节反馈 {actual}；累计 {len(samples)} 张")
    finally:
        camera.release()
        if arm is not None:
            try:
                arm.stop()
            finally:
                bus.close()
        elif bus is not None:
            bus.close()
    print(f"最终有效样本：{len(samples)}；求解需要至少 12 张")


def solve(args):
    K, dist, image_size = calibration(args.intrinsics)
    config = read_json(args.intrinsics)
    cols, rows, square_mm = board_spec(config)
    model = arm_model(args.advanced_config)
    manifest = Path(args.samples)
    records = read_json(manifest).get("samples", [])
    observations, raw_poses, names = [], [], []
    for item in records:
        path = manifest.parent / item["image"]
        frame = read_image(path)
        ensure_image_size(frame, image_size)
        corners = sample_board_corners(frame, item, cols, rows)
        if corners is None:
            print(f"跳过无法重新识别的样本：{item['name']}")
            continue
        raw = np.asarray(item["raw"], dtype=int)
        if raw.shape != (6,) or np.any(raw < 0) or np.any(raw > 1023):
            raise ValueError(f"样本 {item['name']} 必须包含六个 0～1023 的原始位置值")
        # Offline pose conversion uses the calibrated angle mapping, not motion limits.
        observations.append({"name": item["name"], "raw": raw,
                             "angles": model.angles(raw), "corners": corners})
        raw_poses.append(raw)
        names.append(item["name"])
    solved = _solve_with_joint_offsets(observations, model, K, dist, (cols, rows), square_mm)
    selected = solved["selected"]
    T_gc = selected["camera"]
    board_pose = selected["board"]
    board_poses = solved["board_poses"]
    locations = solved["board_origin_per_image"]
    origin = board_pose[:3, 3]
    scatter = float(np.sqrt(np.mean(np.sum((locations - locations.mean(axis=0)) ** 2, axis=1))))
    rotation_sum = sum((T[:3, :3] for T in board_poses), np.zeros((3, 3)))
    u, _, vt = np.linalg.svd(rotation_sum)
    R_mean = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    orientation_error = [math.degrees(math.acos(float(np.clip(
        (np.trace(R_mean.T @ T[:3, :3]) - 1) / 2, -1, 1)))) for T in board_poses]
    orientation_scatter = float(np.sqrt(np.mean(np.square(orientation_error))))
    offsets_deg = np.degrees(selected["delta"])
    result = {
        "T_gripper_camera": T_gc.tolist(), "units": "mm",
        "frames": "T_gripper_camera maps OpenCV camera coordinates to gripper_frame_link coordinates; translation in mm",
        "samples": names, "estimated_board_origin_base_mm": origin.tolist(),
        "board_origin_scatter_rms_mm": float(scatter),
        "board_orientation_scatter_rms_deg": orientation_scatter,
        "camera_translation_in_gripper_mm": T_gc[:3, 3].tolist(),
        "camera_origin_distance_to_gripper_mm": float(np.linalg.norm(T_gc[:3, 3])),
        "distance_definition": "Euclidean distance from camera optical origin to URDF gripper_frame_link origin, in mm; not housing-to-finger clearance.",
        "T_base_board": board_pose.tolist(),
        "joint_offsets_deg": offsets_deg.tolist(),
        "rms_px_nominal_fk": float(solved["nominal"]["rms_px"]),
        "rms_px_with_joint_offsets": float(solved["corrected"]["rms_px"]),
        "rms_px_selected": float(selected["rms_px"]),
        "selected_hand_eye_method": solved["selected_method"],
        "solver_candidates": {
            name: {"initial_board_corner_scatter_mm": value["board_corner_scatter_mm"],
                   "refined_nominal_rms_px": value["nominal"]["rms_px"],
                   "refined_with_offsets_rms_px": value["corrected"]["rms_px"],
                   "identity_initial_seed": value["identity_initial_seed"]}
            for name, value in solved["methods"].items()
        },
        "per_image_rms_px": dict(zip(names, selected["per_image_rms_px"])),
        "joint_offset_std_deg": {
            str(joint + 1): float(math.degrees(value))
            for joint, value in solved["corrected"]["delta_std"].items()
        },
        "unobservable_offset_joint_ids": [joint + 1 for joint in solved["corrected"].get("unobservable", [])],
    }
    rotation = T_gc[:3, :3]
    result["camera_rpy_deg_in_gripper"] = np.degrees([
        math.atan2(rotation[2, 1], rotation[2, 2]),
        math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])),
        math.atan2(rotation[1, 0], rotation[0, 0]),
    ]).tolist()
    result["rpy_convention"] = "R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"
    result["kinematics_patch_suggestion"] = _kinematics_patch(
        read_json(args.advanced_config), model, raw_poses, selected["delta"])
    if args.board_origin_mm is not None:
        measured = np.asarray(args.board_origin_mm, dtype=float)
        error = origin - measured
        result["measured_board_origin_base_mm"] = measured.tolist()
        result["board_origin_error_each_axis_mm"] = np.abs(error).tolist()
        result["board_origin_error_norm_mm"] = float(np.linalg.norm(error))
        result["meets_50mm_each_axis"] = bool(np.all(np.abs(error) <= 50))
    write_json(args.output, result)
    report = np.full((620, 1350, 3), 255, np.uint8)
    lines = ["WEEK 6 - EYE IN HAND CALIBRATION", f"Samples: {len(names)}",
             f"Selected method: {solved['selected_method']}; reprojection RMS: {selected['rms_px']:.2f} px",
             f"Nominal FK RMS: {solved['nominal']['rms_px']:.2f} px; offset fit RMS: {solved['corrected']['rms_px']:.2f} px",
             "T_gripper_camera (camera frame -> gripper frame, mm):",
             *["  " + "  ".join(f"{x:9.4f}" for x in row) for row in T_gc],
             f"Camera-to-gripper origin distance (fitted): {np.linalg.norm(T_gc[:3, 3]):.2f} mm",
             f"Joint offsets ID 2-4 [deg]: {np.round(offsets_deg[1:4], 2).tolist()}",
             f"Board origin in base [mm]: {np.round(origin, 2).tolist()}",
             f"Board position scatter RMS: {scatter:.2f} mm",
             f"Board orientation scatter RMS: {orientation_scatter:.2f} deg",
             "Fitted distance is not an independently measured mounting distance."]
    if args.board_origin_mm is not None:
        lines.append(f"Measured board origin [mm]: {np.round(measured, 2).tolist()}")
        lines.append(f"Absolute error by axis [mm]: {np.round(np.abs(error), 2).tolist()}")
    for index, line in enumerate(lines):
        cv2.putText(report, line, (25, 38 + index * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (20, 20, 20), 1, cv2.LINE_AA)
    report_path = Path(args.output).with_suffix(".png")
    write_image(report_path, report)
    print(f"已保存 {args.output} 与 {report_path}；标定板位置散布 RMS {scatter:.2f} mm")
    print(f"名义 FK RMS {solved['nominal']['rms_px']:.2f}px；含关节偏差 RMS "
          f"{solved['corrected']['rms_px']:.2f}px；采用 {solved['selected_method']}，"
          f"关节偏差 ID 2–4={np.round(offsets_deg[1:4], 2).tolist()}°")
    print("运动学修正仅作为 JSON 建议输出；不会自动修改 advanced.json。")
    if args.board_origin_mm is None:
        print("未提供实测标定板原点，无法判定课程要求的各轴 50 mm 外部误差。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--plan", type=Path, default=HERE / "pose_plan.json")
    collect_parser.add_argument("--intrinsics", type=Path, default=HERE.parent / "part1_camera_calibration" / "camera_intrinsics.json")
    collect_parser.add_argument("--advanced-config", type=Path, default=ARM_CODE / "advanced.json")
    collect_parser.add_argument("--hardware-config", type=Path, default=ARM_CODE / "config.json")
    collect_parser.add_argument("--output", type=Path, default=HERE / "samples")
    collect_parser.add_argument("--camera", default=DEFAULT_CAMERA,
                                help="默认按 Windows 设备名选择 GENERAL WEBCAM；也可传数字索引")
    collect_parser.add_argument("--port")
    collect_parser.add_argument("--execute", action="store_true", help="实际驱动机械臂；缺省仅预检计划")
    solve_parser = sub.add_parser("solve")
    solve_parser.add_argument("--samples", type=Path, default=HERE / "samples" / "samples.json")
    solve_parser.add_argument("--intrinsics", type=Path, default=HERE.parent / "part1_camera_calibration" / "camera_intrinsics.json")
    solve_parser.add_argument("--advanced-config", type=Path, default=ARM_CODE / "advanced.json")
    solve_parser.add_argument("--board-origin-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    solve_parser.add_argument("--output", type=Path, default=HERE / "hand_eye.json")
    args = parser.parse_args()
    (collect if args.command == "collect" else solve)(args)


if __name__ == "__main__":
    main()
