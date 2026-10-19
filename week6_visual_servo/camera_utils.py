"""Open the named Windows webcam used by the Week 6 exercises."""

from __future__ import annotations

import os

import cv2


DEFAULT_CAMERA = "GENERAL WEBCAM"


def directshow_camera_names() -> list[str]:
    """Enumerate Windows DirectShow capture devices in their index order."""
    if os.name != "nt":
        raise RuntimeError("按设备名称选择摄像头目前仅支持 Windows")
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())
    except Exception as exc:
        raise RuntimeError(
            "无法读取 Windows DirectShow 摄像头清单。确认已在 py312 安装 requirements.txt，"
            f"并从本机 Anaconda Prompt 运行；系统返回：{exc}"
        ) from exc


def _named_index(source: str, names: list[str]) -> tuple[int, str]:
    query = source.casefold().strip()
    exact = [(index, name) for index, name in enumerate(names) if name.casefold().strip() == query]
    matches = exact or [(index, name) for index, name in enumerate(names)
                        if query in name.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"摄像头名称“{source}”匹配到多个设备：{[name for _, name in matches]}")
    raise RuntimeError(f"Windows 摄像头清单中找不到“{source}”；当前设备为：{names or '（无视频设备）'}")


def open_camera(source: str | int = DEFAULT_CAMERA,
                size: tuple[int, int] | None = None):
    """Open by exact camera name (default) or explicit numeric OpenCV index."""
    value = str(source).strip()
    named = not value.isdecimal()
    if named:
        names = directshow_camera_names()
        index, device_name = _named_index(value, names)
        candidates = [(index, cv2.CAP_DSHOW, "DirectShow")]
    else:
        index = int(value)
        device_name = f"OpenCV 索引 {index}"
        if os.name == "nt":
            candidates = [(index, cv2.CAP_DSHOW, "DirectShow"),
                          (index, cv2.CAP_MSMF, "Media Foundation"),
                          (index, cv2.CAP_ANY, "OpenCV auto")]
        else:
            candidates = [(index, cv2.CAP_ANY, "OpenCV auto")]

    failures = []
    for index, backend, backend_name in candidates:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            failures.append(f"{backend_name}: 未打开")
            capture.release()
            continue
        if size is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture, index, backend_name, device_name
        failures.append(f"{backend_name}: 已打开但没有视频帧")
        capture.release()

    details = "；".join(failures)
    raise RuntimeError(
        f"已按名称找到摄像头“{device_name}”，但无法读取画面（{details}）。"
        "确认摄像头已连接、在 Windows‘蓝牙和设备 → 摄像头’中启用，"
        "并允许桌面应用访问摄像头。"
    )
