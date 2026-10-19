"""Verify the GENERAL WEBCAM DirectShow name and read a frame from it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_utils import DEFAULT_CAMERA, directshow_camera_names, open_camera  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="device name to verify; default: GENERAL WEBCAM")
    args = parser.parse_args()
    names = directshow_camera_names()
    print("Windows DirectShow 摄像头清单：")
    for name in names:
        print(f"  {name}")
    if not names:
        print("当前没有枚举到视频设备；请连接摄像头，并确认 Windows 已识别它。")
        return 1
    capture = None
    try:
        capture, index, backend, name = open_camera(args.camera)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("设备已打开，但未能读取视频帧")
        print(f"已连接目标设备“{name}”：索引 {index}，{frame.shape[1]}×{frame.shape[0]}，{backend}")
        return 0
    except RuntimeError as exc:
        print(f"检查失败：{exc}")
        return 1
    finally:
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
