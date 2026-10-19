"""Part 1: collect chessboard photos and calibrate camera intrinsics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_utils import DEFAULT_CAMERA, open_camera  # noqa: E402
from common import board_points, chessboard_corners, read_image, read_json, write_image, write_json  # noqa: E402


HERE = Path(__file__).resolve().parent


def capture(args):
    folder = Path(args.images)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "board.json"
    board = {"inner_corners": [args.cols, args.rows], "square_mm": args.square_mm}
    if metadata.exists() and read_json(metadata) != board:
        raise ValueError("图像目录已有不同棋盘规格；请选择新目录")
    camera, selected_index, backend, device_name = open_camera(args.camera)
    print(f"使用摄像头“{device_name}”（索引 {selected_index}，{backend}）")
    existing = len(list(folder.glob("chess_*.png")))
    print("移动棋盘或相机改变视角；角点全部可见时按空格保存，Q/Esc 退出。")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("摄像头读取失败")
            corners = chessboard_corners(frame, args.cols, args.rows)
            preview = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(preview, (args.cols, args.rows), corners, True)
            cv2.putText(preview, f"saved {existing}/{args.count}  corners: {corners is not None}",
                        (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Part 1 - chessboard capture", preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and corners is not None:
                path = folder / f"chess_{existing + 1:03d}.png"
                write_image(path, frame)
                write_json(metadata, board)
                existing += 1
                print(f"保存 {path.name}；请改变距离、位置或倾斜角度")
                if existing >= args.count:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"保存 {existing} 张；至少需要 15 张不同视角")


def calibrate(args):
    folder = Path(args.images)
    metadata = folder / "board.json"
    if metadata.exists() and read_json(metadata) != {
        "inner_corners": [args.cols, args.rows], "square_mm": args.square_mm
    }:
        raise ValueError("命令参数与采集时的棋盘规格不一致")
    points = board_points(args.cols, args.rows, args.square_mm)
    world, image, used = [], [], []
    image_size = None
    for path in sorted(folder.glob("chess_*.png")):
        try:
            frame = read_image(path)
        except (ValueError, FileNotFoundError):
            print(f"跳过无法读取的图像：{path}")
            continue
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            raise ValueError(f"图像尺寸不一致：{path.name} 为 {size}，预期 {image_size}")
        corners = chessboard_corners(frame, args.cols, args.rows)
        if corners is None:
            print(f"跳过未找到完整棋盘的图像：{path.name}")
            continue
        world.append(points)
        image.append(corners)
        used.append(path.name)
    if len(used) < 15:
        raise ValueError(f"仅有 {len(used)} 张有效图像；需要至少 15 张不同视角")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(world, image, image_size, None, None)
    errors = []
    for object_points, corners, rvec, tvec in zip(world, image, rvecs, tvecs):
        predicted, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((predicted - corners) ** 2, axis=2)))))
    result = {
        "image_size": list(image_size), "inner_corners": [args.cols, args.rows],
        "square_mm": args.square_mm, "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(), "rms_reprojection_px": float(rms),
        "views": [{"image": name, "rms_px": error} for name, error in zip(used, errors)],
    }
    write_json(args.output, result)
    print(f"有效图像 {len(used)} 张；整体重投影 RMS = {rms:.3f} px；结果：{args.output}")
    if rms > 1.5:
        print("误差偏高；检查棋盘是否平整、方格边长、角点数、图像清晰度和视角多样性。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "calibrate"):
        item = sub.add_parser(name)
        item.add_argument("--images", type=Path, default=HERE / "images")
        item.add_argument("--cols", type=int, default=8, help="棋盘横向内角点数，不是方格数")
        item.add_argument("--rows", type=int, default=6, help="棋盘纵向内角点数")
        item.add_argument("--square-mm", type=float, default=25.0)
        if name == "capture":
            item.add_argument("--camera", default=DEFAULT_CAMERA,
                              help="默认按 Windows 设备名选择 GENERAL WEBCAM；也可传数字索引")
            item.add_argument("--count", type=int, default=18)
        else:
            item.add_argument("--output", type=Path, default=HERE / "camera_intrinsics.json")
    args = parser.parse_args()
    board_points(args.cols, args.rows, args.square_mm)
    if args.command == "capture":
        if args.count < 15:
            parser.error("--count 不能少于 15")
        capture(args)
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
