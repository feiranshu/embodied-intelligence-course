"""Black-circle tracking with an end-effector or fixed desktop camera."""

from __future__ import annotations

import argparse
import copy
import secrets
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from advanced import CommandRunner, ROOT, load_advanced, open_arm, save_advanced
from controller import RequestHandler
from servo_bus import load_config


def detect_ball(frame, config, *, allow_clipped=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.inRange(gray, 0, int(config["threshold"]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        clipped = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
        if clipped and not allow_clipped:
            continue
        if not float(config["min_area"]) <= area <= width * height * (0.65 if clipped else 0.35):
            continue
        if clipped:
            points = contour[:, 0, :]
            points = points[(points[:, 0] > 2) & (points[:, 0] < width - 3)
                            & (points[:, 1] > 2) & (points[:, 1] < height - 3)]
            if len(points) < 12:
                continue
            center, axes, angle = cv2.fitEllipse(points.astype(np.float32))
            a, b = np.asarray(axes) / 2
            if min(a, b) < 5 or max(a, b) > min(width, height) * 0.75 or min(a, b) / max(a, b) < 0.6:
                continue
            offset = points - center
            angle = np.radians(angle)
            u = (np.cos(angle) * offset[:, 0] + np.sin(angle) * offset[:, 1]) / a
            v = (-np.sin(angle) * offset[:, 0] + np.cos(angle) * offset[:, 1]) / b
            residual = np.abs(np.hypot(u, v) - 1)
            angles = np.sort(np.arctan2(v, u))
            coverage = 2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
            if np.median(residual) > 0.08 or np.quantile(residual, 0.9) > 0.2 or coverage < 2 * np.pi / 3:
                continue
            candidates.append((area, np.array(center), float(np.sqrt(a * b))))
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
        if circularity < float(config["min_circularity"]):
            continue
        if not 0.7 <= w / h <= 1.4:
            continue
        moments = cv2.moments(contour)
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        candidates.append((area, center, (area / np.pi) ** 0.5))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > candidates[0][0] * 0.65:
        return None
    return candidates[0][1], candidates[0][2]


def validate_jacobian(config):
    matrix = np.asarray(config["image_jacobian"], dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("需先标定 2×2 图像雅可比矩阵")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] < 0.2 or singular[0] / singular[-1] > 20:
        raise ValueError("相机反馈不足或控制轴退化；检查末端相机安装和 control_axes")
    axes = config["control_axes"]
    if len(axes) != 2 or len(set(axes)) != 2 or any(a not in (0, 1, 2) for a in axes):
        raise ValueError("control_axes 应是两个不同的基座坐标轴编号")
    return matrix


def correction(error, config):
    error = np.asarray(error, dtype=float)
    if error.shape != (2,) or not np.all(np.isfinite(error)):
        raise ValueError("图像误差无效")
    matrix = validate_jacobian(config)
    gain, limit = float(config["gain"]), float(config["step_mm"])
    deadband = float(config["deadband_px"])
    if not 0 < gain <= 1 or not 0 < limit <= 10 or not 1 <= deadband <= 100:
        raise ValueError("视觉增益、步长或死区无效")
    if np.linalg.norm(error) <= deadband:
        return np.zeros(3)
    step = -gain * np.linalg.solve(matrix, error)
    step *= min(1, limit / max(np.linalg.norm(step), 1e-9))
    delta = np.zeros(3)
    delta[config["control_axes"]] = step
    return delta


def synthetic_frame(arm, origin, config):
    frame = np.full((config["height"], config["width"], 3), 255, dtype=np.uint8)
    displacement = (arm.model.forward(arm.positions()) - origin)[config["control_axes"]]
    center = np.array([config["width"] / 2 + 90, config["height"] / 2 - 65])
    center += np.array([[-3.0, 0.4], [0.2, -3.0]]) @ displacement
    cv2.circle(frame, tuple(np.rint(center).astype(int)), 35, (15, 15, 15), -1)
    cv2.circle(frame, tuple(np.rint(center + [-10, -12]).astype(int)), 9, (100, 100, 100), -1)
    return frame


def sample_center(camera, config):
    centers = []
    fresh_after = time.monotonic() + 0.3
    deadline = time.monotonic() + 3
    while len(centers) < 5 and time.monotonic() < deadline:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("无法读取摄像头")
        ball = detect_ball(frame, config)
        if ball is None or time.monotonic() < fresh_after:
            centers.clear()
        else:
            centers.append(ball[0])
        cv2.imshow("SO101 vision", frame)
        if cv2.waitKey(20) & 255 in (27, ord("q")):
            raise InterruptedError("已取消标定")
    if len(centers) < 5 or np.max(np.std(centers, axis=0)) > 3:
        raise RuntimeError("黑球未稳定可见；保持屏幕目标静止")
    return np.mean(centers, axis=0)


def calibrate(camera, arm, config):
    original = arm.positions()
    origin = arm.model.forward(original)
    axes = config["control_axes"]
    if len(axes) != 2 or len(set(axes)) != 2 or any(a not in (0, 1, 2) for a in axes):
        raise ValueError("control_axes 必须包含两个不同的坐标轴")
    targets = []
    for axis in axes:
        delta = np.zeros(3)
        delta[axis] = 5
        targets.append(arm.model.inverse(origin + delta, original))
        if np.max(np.abs(targets[-1][:4] - original[:4])) > 45:
            raise ValueError("标定初始姿态接近奇异位形，请调整姿态")
    pixel_changes, actual_changes = [], []
    for target in targets:
        before = sample_center(camera, config)
        arm.move_pose(target, duration_s=0.8)
        after = sample_center(camera, config)
        pixel_changes.append(after - before)
        actual_changes.append((arm.model.forward(arm.positions()) - origin)[axes])
        arm.move_pose(original, duration_s=0.8)
        returned = sample_center(camera, config)
        if np.linalg.norm(returned - before) > 8:
            raise RuntimeError("回到原姿态后画面未复原，停止标定")
    movements = np.column_stack(actual_changes)
    if np.linalg.cond(movements) > 20:
        raise ValueError("标定运动反馈退化")
    result = copy.deepcopy(config)
    result["image_jacobian"] = (np.column_stack(pixel_changes) @ np.linalg.inv(movements)).tolist()
    validate_jacobian(result)
    result["calibrated"] = True
    return result


class VisionRequestHandler(RequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self._reject_if_unauthorized():
            return
        if urlparse(self.path).path == "/":
            self._send_file(ROOT / "web" / "black_ball.html", "text/html; charset=utf-8")
        elif urlparse(self.path).path == "/api/frame" and self.server.frame is not None:
            frame = self.server.frame
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
        elif urlparse(self.path).path == "/api/state":
            with self.server.motion_lock:
                self._send_json({"ok": True, **self.server.state})
        else:
            self._send_json({"ok": False, "error": "页面不存在"}, 404)

    def do_POST(self):
        if self._reject_if_unauthorized():
            return
        try:
            data = self._json_body()
            if urlparse(self.path).path != "/api/command":
                raise ValueError("未知操作")
            self.server.request(data.get("action"))
            self._send_json({"ok": True})
        except Exception as error:
            self._send_json({"ok": False, "error": str(error)}, 400)


class VisionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, arm, config_path):
        self.arm, self.config_path = arm, config_path
        self.motion_lock = threading.RLock()
        self.pending = None
        self.runner = None
        self.closed = threading.Event()
        self.heartbeat = time.monotonic()
        self.driving = False
        self.frame = None
        self.state = {"active": False, "busy": False, "ball": False, "stable": False,
                      "error": "", "message": "点击启动，自动准备姿态并跟随黑球", "fault": False}
        VisionRequestHandler.access_token = secrets.token_urlsafe(18)
        super().__init__(("127.0.0.1", 0), VisionRequestHandler)
        self.url = f"http://127.0.0.1:{self.server_port}/?token={VisionRequestHandler.access_token}"

    def request(self, action):
        keys = {"start": ord("s")}
        with self.motion_lock:
            self.heartbeat = time.monotonic()
            if action == "heartbeat":
                return
            if action in {"stop", "exit"}:
                self.pending = ord("p")
                self.state["active"] = self.driving = False
                self.arm.stop()
                self.state["message"] = "已停止"
                if action == "exit":
                    self.closed.set()
                return
            if action not in keys:
                raise ValueError("未知操作")
            if self.state["fault"]:
                raise ValueError("控制已停止，请查看错误并重新打开程序")
            if self.pending is not None or self.state["busy"] or self.state["active"]:
                raise ValueError("请先停止跟随或等待当前动作结束")
            self.pending = keys[action]
            self.state["busy"] = True
            self.state["error"] = ""
            self.driving = True

    def submit(self, command):
        with self.motion_lock:
            if not self.driving or self.pending == ord("p") or self.closed.is_set():
                return False
            accepted = self.runner.submit(command)
            self.state["busy"] = self.runner.busy
            return accepted

    def service_actions(self):
        if self.driving and time.monotonic() - self.heartbeat > 2:
            try:
                self.request("stop")
                self.state["message"] = "页面连接中断，已停止"
            except Exception as error:
                self.driving = False
                self.state.update(error=str(error), fault=True)


def pointing_target(model, reference, offset, degrees_per_pixel):
    if not model.chain:
        raise ValueError("夹爪指向需要已标定的 URDF 模型")
    reference = model.validate(reference, fixed_hand=np.asarray(reference)[4:])
    if np.any(reference[:4] < 16) or np.any(reference[:4] > 1007):
        raise ValueError("当前关节位于反馈端点附近，不能开始转向；请先进入连续工作区")
    low, high = model.limits.copy().T
    branch = model.unwrap(reference) - reference
    low = np.maximum(low, branch + [16, 16, 16, 16, 0, 0])
    high = np.minimum(high, branch + [1007, 1007, 1007, 1007, 1023, 1023])
    bounds = model.zero[:, None] + ((model.angle_limits - model.reference[:, None])
                                  / (model.signs[:, None] * model.scale))
    low[:5] = np.maximum(low[:5], np.min(bounds, axis=1))
    high[:5] = np.minimum(high[:5], np.max(bounds, axis=1))
    target = model.unwrap(reference)
    angles = np.asarray(offset, dtype=float) * degrees_per_pixel
    target[[0, 3]] += np.radians(np.clip(angles, -25, 25)) / model.scale
    clipped = np.clip(target, np.ceil(low), np.floor(high))
    limited = np.any(np.abs(angles) > 25) or np.any(np.abs(clipped - target) > 0.5)
    raw = np.rint(clipped - branch).astype(int)
    raw[4:] = reference[4:]
    return model.validate(raw, fixed_hand=reference[4:]), bool(limited)


def track(camera, arm, config, *, simulate=False, preview=False, headless=False, output=None,
          fixed_camera=False, mm_per_pixel=0.2, wait_start=False, point_at_ball=False,
          degrees_per_pixel=0.25, web=None):
    if point_at_ball and (not fixed_camera or not np.isfinite(degrees_per_pixel) or not 0 < degrees_per_pixel <= 1):
        raise ValueError("夹爪指向需要固定相机，灵敏度须为 0～1 度/像素（不含零）")
    if fixed_camera:
        axes = config["control_axes"]
        if len(axes) != 2 or len(set(axes)) != 2 or any(a not in (0, 1, 2) for a in axes):
            raise ValueError("control_axes 应是两个不同的基座坐标轴编号")
        if not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
            raise ValueError("固定相机的毫米/像素比例必须是正数")
        step_mm = float(config.get("fixed_step_mm", 10.0))
        deadband_px = float(config.get("fixed_deadband_px", 25.0))
        if point_at_ball:
            deadband_px = 6.0
        if not (0 < step_mm <= 10 and 1 <= deadband_px <= 100):
            raise ValueError("视觉步长或死区无效")
    elif not preview:
        validate_jacobian(config)
    runner = CommandRunner(arm) if arm is not None else None
    control = runner if web is None else web
    if web is not None:
        web.runner = runner
    origin = arm.model.forward(arm.positions()) if arm is not None else None
    stable = 0
    previous_center = None
    previous_radius = None
    previous_error = None
    growing = 0
    steps = 0
    started = time.monotonic()
    initial_error = None
    final_error = None
    settled = 0
    reference_center = None
    reference_pose = None
    settled_center = None
    following_target = False
    capture_stable = 0
    active = not wait_start
    startup = None
    try:
        while web is not None or preview or not active or time.monotonic() - started < 120:
            if web is not None and web.closed.is_set():
                break
            if web is not None and not web.driving:
                startup = None
            captured_at = time.monotonic()
            if runner is not None and runner.error is not None and not isinstance(runner.error, InterruptedError):
                raise runner.error
            if simulate:
                frame = synthetic_frame(arm, origin, config)
            else:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("摄像头断开，停止视觉控制")
            if not preview and frame.shape[:2] != (config["height"], config["width"]):
                raise ValueError("摄像头分辨率与标定不一致，请重新标定")
            ball = detect_ball(frame, config, allow_clipped=not fixed_camera)
            capture_stable = (capture_stable + 1 if ball is not None and previous_center is not None
                              and np.linalg.norm(ball[0] - previous_center) < 3 else 0)
            center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
            cv2.drawMarker(frame, tuple(center.astype(int)), (0, 160, 0), cv2.MARKER_CROSS, 20, 2)
            status = "PREVIEW" if preview else "SIMULATION" if simulate else "TRACKING"
            if fixed_camera:
                status += " | FIXED CAMERA"
            if point_at_ball:
                status += f" | AIM {degrees_per_pixel:g} deg/px"
            if not active:
                status = "READY | H: initial pose | S: follow | SPACE: exit"
                if point_at_ball:
                    status = "READY | H: prepare | arrows/IJKL: aim | S: follow"
                positions = arm.positions()
                cv2.putText(frame, "JOINTS: " + " / ".join(map(str, positions)), (12, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 80, 190), 1)
                cv2.putText(frame, "HOME: " + " / ".join(map(str, arm.config["poses"]["初始状态"])), (12, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 80, 190), 1)
            if ball is None:
                stable = settled = 0
                status += " | target lost/ambiguous: STOP"
                if runner is not None and ((active and (runner.busy or point_at_ball and previous_center is not None))
                                           or web is not None and startup in {"preparing", "entering"} and runner.busy):
                    runner.stop()
                    if web is not None:
                        active, startup = False, "waiting"
                        web.state["message"] = "目标暂时丢失，已停止；重新识别后自动继续"
                previous_center = previous_radius = None
            else:
                detected, radius = ball
                if not preview and active and previous_center is not None and (
                        np.linalg.norm(detected - previous_center) > 60
                        or not 0.6 <= radius / previous_radius <= 1.7):
                    if not fixed_camera:
                        raise RuntimeError("目标发生突跳，停止并重新确认黑球")
                    runner.stop()
                    stable = 0
                    if web is not None:
                        active, startup = False, "waiting"
                        web.state["message"] = "目标移动过快，已停止；稳定识别后自动继续"
                previous_center, previous_radius = detected, radius
                stable += 1
                if web is not None and startup and not runner.busy and capture_stable >= 5:
                    if startup == "waiting":
                        if control.submit(("point_home", "初始状态")):
                            startup = "preparing"
                            web.state["message"] = "正在自动准备姿态"
                    elif captured_at > runner.finished_at + 0.15:
                        current = arm.cartesian_start_pose()
                        if startup in {"preparing", "entering"} and 1007 < current[0] <= 1023:
                            target = current.copy()
                            target[0] = min(1007, current[0] - 16)
                            if control.submit(("point", target)):
                                startup = "entering"
                                web.state["message"] = "正在小步进入连续工作区"
                        else:
                            pointing_target(arm.model, current, [0, 0], degrees_per_pixel)
                            if reference_pose is None:
                                reference_pose, reference_center = current, detected.copy()
                            settled_center = detected.copy()
                            origin = arm.model.forward(reference_pose)
                            following_target = True
                            active, startup = True, None
                            steps, started = 0, time.monotonic()
                            web.state["message"] = "正在自动跟随；拖动黑球即可"
                if fixed_camera:
                    if web is None and reference_center is None and stable >= 3:
                        reference_center = detected.copy()
                        if arm is not None:
                            reference_pose = arm.positions()
                            origin = arm.model.forward(reference_pose)
                    center = detected if reference_center is None else reference_center
                error = detected - center
                if fixed_camera and not preview and active:
                    if point_at_ball and reference_pose is not None:
                        target, limited = pointing_target(arm.model, reference_pose, error, degrees_per_pixel)
                        current = arm.positions()
                        error = (current - target)[[0, 3]] * np.degrees(arm.model.scale) / degrees_per_pixel
                        if limited:
                            status += " | LIMIT"
                    else:
                        target = origin.copy()
                        target[axes] -= error * mm_per_pixel
                        if np.linalg.norm(target - origin) > float(config["max_travel_mm"]):
                            raise RuntimeError("小球目标超出本次固定相机位移范围")
                        error = (arm.model.forward(arm.positions()) - target) / mm_per_pixel
                norm = float(np.linalg.norm(error))
                final_error = norm
                if initial_error is None:
                    initial_error = norm
                cv2.circle(frame, tuple(np.rint(detected).astype(int)), round(radius), (0, 0, 255), 2)
                cv2.line(frame, tuple(center.astype(int)), tuple(np.rint(detected).astype(int)), (255, 0, 0), 2)
                status += f" | error={norm:.1f}px | steps={steps}"
                if not preview and active and stable >= 3 and not runner.busy and captured_at > runner.finished_at + 0.15:
                    if runner.error is not None and not isinstance(runner.error, InterruptedError):
                        raise runner.error
                    if not fixed_camera and previous_error is not None and norm > previous_error + 3:
                        growing += 1
                    else:
                        growing = 0
                    if growing >= 3:
                        raise RuntimeError("图像误差连续增大；检查相机标定和安装")
                    if fixed_camera:
                        delta = np.zeros(3)
                        if settled_center is None:
                            settled_center = detected.copy()
                        if norm <= deadband_px:
                            following_target = False
                            settled_center = detected.copy()
                        elif following_target or np.linalg.norm(detected - settled_center) > deadband_px:
                            following_target = True
                            if point_at_ball:
                                current = arm.positions()
                                delta = target - current
                                delta[4:] = 0
                                delta = np.rint(delta * min(1, 16 / max(np.max(np.abs(delta)), 1))).astype(int)
                            else:
                                delta = -error * mm_per_pixel
                                delta *= min(1, step_mm / np.linalg.norm(delta))
                    else:
                        delta = correction(error, config)
                    settled = settled + 1 if not np.any(delta) else 0
                    if settled >= 5 and not fixed_camera:
                        print(f"进入目标死区：{norm:.2f} px，停止跟踪", flush=True)
                        if output:
                            cv2.putText(frame, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 80, 190), 1)
                            if not cv2.imwrite(str(output), frame):
                                raise OSError("无法保存视觉验证图")
                        break
                    if np.any(delta):
                        if not point_at_ball:
                            displacement = arm.model.forward(arm.positions()) - origin
                            if np.linalg.norm(displacement + delta) > float(config["max_travel_mm"]):
                                raise RuntimeError("已达到本次视觉控制位移上限")
                        if web is None and steps >= int(config["max_steps"]):
                            raise RuntimeError("已达到视觉迭代次数上限")
                        if point_at_ball:
                            print(f"夹爪指向：误差={error.round(2).tolist()} px，目标关节={(current + delta).tolist()}", flush=True)
                            control.submit(("point", current + delta))
                        else:
                            print(f"视觉反馈：误差={error.round(2).tolist()} px，位移={delta.round(2).tolist()} mm", flush=True)
                            runner.submit(("move", delta))
                        previous_error = norm
                        steps += 1
            cv2.putText(frame, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 80, 190), 1)
            if not headless or web is not None:
                if web is None:
                    cv2.imshow("SO101 vision", frame)
                    key = cv2.waitKeyEx(20) if point_at_ball else cv2.waitKey(20) & 255
                else:
                    with web.motion_lock:
                        web.frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                        web.state.update(active=active, busy=runner.busy or web.pending is not None or startup is not None,
                                         ball=ball is not None, stable=capture_stable >= 5,
                                         limited="LIMIT" in status, motion=arm.last_joint_move)
                        key, web.pending = web.pending, None
                    time.sleep(0.01)
                if key in (27, ord("q"), 32):
                    print("已请求停止视觉控制")
                    break
                if fixed_camera and not preview and not active and not runner.busy:
                    directions = {2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1),
                                  ord("j"): (-1, 0), ord("l"): (1, 0), ord("i"): (0, -1), ord("k"): (0, 1),
                                  ord("J"): (-1, 0), ord("L"): (1, 0), ord("I"): (0, -1), ord("K"): (0, 1)}
                    if point_at_ball and key in directions:
                        try:
                            current = arm.cartesian_start_pose()
                            target, limited = pointing_target(arm.model, current, directions[key],
                                                              1.0 if key in map(ord, "IJKL") else 5.0)
                            if np.any(target != current):
                                control.submit(("point", target))
                            if limited:
                                print("已到该方向关节边界；需要调整整台机械臂与屏幕的相对摆放。", flush=True)
                                if web is not None:
                                    web.state["error"] = "已到该方向关节边界，仍未对准时需调整机械臂与屏幕的摆放"
                        except ValueError as error:
                            print(f"尚不能调整指向：{error}", flush=True)
                            if web is not None:
                                web.state["error"] = str(error)
                    elif key == ord("h"):
                        try:
                            arm.validate_pose(arm.positions(), tolerance=12)
                        except ValueError as error:
                            print(f"不能自动回位：{error}；当前关节超出已测范围，未发送运动命令。", flush=True)
                            if web is None:
                                cv2.setWindowTitle("SO101 vision", "SO101 vision - START OUTSIDE RANGE")
                            else:
                                web.state["error"] = str(error)
                        else:
                            control.submit(("point_home" if point_at_ball else "home", "初始状态"))
                            if web is not None:
                                web.state["message"] = "保留底座方向，上臂回位后腕部向前倾斜；准备完成后需核对屏幕方向"
                    elif key == ord("s") and web is not None:
                        startup = "waiting"
                        reference_center = reference_pose = None
                        web.state["message"] = "等待相机稳定识别黑球，随后自动准备并跟随"
                    elif key == ord("s"):
                        try:
                            origin = arm.model.forward(arm.cartesian_start_pose())
                        except ValueError as error:
                            print(f"尚未开始跟随：{error}；窗口继续显示实时读数。进入已测范围后按 H 回工作姿态，再按 S。", flush=True)
                            if web is None:
                                cv2.setWindowTitle("SO101 vision", "SO101 vision - PRESS H AFTER RANGE CHECK")
                            else:
                                web.state["error"] = str(error)
                        else:
                            reference_center = previous_center = previous_radius = None
                            reference_pose = None
                            stable = 0
                            active = True
                            started = time.monotonic()
                            if web is None:
                                cv2.setWindowTitle("SO101 vision", "SO101 vision - FOLLOWING")
                            print("开始夹爪转向跟随；保持小球静止三帧建立对准基准" if point_at_ball else
                                  "开始末端左右/上下跟随；保持小球静止三帧建立参考位置", flush=True)
                if point_at_ball and key == ord("p"):
                    startup = None
                    runner.stop()
                    active = False
                    reference_center = reference_pose = settled_center = None
                    previous_center = previous_radius = None
                    stable = steps = 0
                    following_target = False
                    if web is None:
                        cv2.setWindowTitle("SO101 vision", "SO101 vision")
                    print("已暂停跟随；方向键调整指向，重新按 S 建立基准。", flush=True)
                if web is None and cv2.getWindowProperty("SO101 vision", cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(0.025)
        else:
            raise TimeoutError("本次视觉控制达到 120 秒上限")
        return {"simulation": simulate, "steps": steps, "initial_error_px": initial_error,
                "final_error_px": final_error, "converged": settled >= 5}
    finally:
        if runner is not None:
            runner.close()


def track_joints(camera, arm, config, joints, *, headless=False, counts_per_pixel=0.5):
    from joint_mapping import mapped_position, raw_position

    if (not 1 <= len(joints) <= 2 or len(set(joints)) != len(joints)
            or any(j not in range(1, 6) for j in joints)
            or not np.isfinite(counts_per_pixel) or not 0 < counts_per_pixel <= 0.5):
        raise ValueError("跟随关节应为一到两个不同的 ID 1～5，灵敏度应在 (0, 0.5] 内")
    origin = arm.validate_pose(arm.positions())
    mappings = [arm.config["joint_mapping"][str(j)] for j in joints]
    runner = CommandRunner(arm)
    reference = previous = None
    stable = steps = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < 120:
            captured_at = time.monotonic()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("摄像头断开，停止关节跟随")
            if frame.shape[:2] != (config["height"], config["width"]):
                raise ValueError("摄像头分辨率不一致")
            if runner.error is not None and not isinstance(runner.error, InterruptedError):
                raise runner.error
            ball = detect_ball(frame, config)
            status = f"FIXED CAMERA | JOINT FOLLOW | steps={steps}"
            if ball is None:
                stable = 0
                previous = None
                if runner.busy:
                    runner.stop()
                status += " | NO TARGET: STOP"
            else:
                center, radius = ball
                if previous is not None and np.linalg.norm(center - previous) > 60:
                    stable = 0
                    if runner.busy:
                        runner.stop()
                previous = center.copy()
                stable += 1
                cv2.circle(frame, tuple(np.rint(center).astype(int)), round(radius), (0, 0, 255), 2)
                if reference is None and stable >= 3:
                    reference = center.copy()
                if reference is not None:
                    cv2.drawMarker(frame, tuple(np.rint(reference).astype(int)), (0, 160, 0))
                if (reference is not None and stable >= 3 and not runner.busy
                        and captured_at > runner.finished_at + 0.15):
                    current = arm.positions()
                    target = current.copy()
                    offset = np.clip((center - reference) * counts_per_pixel * [1, -1], -60, 60)
                    for axis, (joint, mapping) in enumerate(zip(joints, mappings)):
                        desired = np.clip(mapped_position(origin[joint - 1], mapping) + offset[axis],
                                          *mapping["limits"])
                        raw = raw_position(round(desired), mapping)
                        expected = (raw - int(origin[joint - 1])) * mapping.get("sign", 1)
                        if abs(expected - (round(desired) - mapped_position(origin[joint - 1], mapping))) > 1:
                            raw = 1021 if offset[axis] * mapping.get("sign", 1) > 0 else 2
                        raw = int(np.clip(raw, 2, 1021))
                        change = raw - int(current[joint - 1])
                        if abs(change) >= 6:
                            target[joint - 1] += int(np.clip(change, -24, 24))
                    if np.any(target != current):
                        if steps >= int(config["max_steps"]):
                            raise RuntimeError("已达到本次关节跟随步数上限")
                        arm.validate_pose(target)
                        print(f"小球={center.round(1).tolist()}；关节目标={target.tolist()}", flush=True)
                        runner.submit(("pose_raw", target))
                        steps += 1
            cv2.putText(frame, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 80, 190), 1)
            if not headless:
                cv2.imshow("SO101 vision", frame)
                if cv2.waitKey(20) & 255 in (27, ord("q"), 32):
                    break
                if cv2.getWindowProperty("SO101 vision", cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(0.025)
        return {"mode": "fixed_camera_joint_follow", "steps": steps}
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser(description="黑球视觉控制；支持末端相机和桌面固定相机，默认只预览")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--advanced-config", default=str(ROOT / "advanced.json"))
    parser.add_argument("--port")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--fixed-camera", action="store_true", help="桌面固定相机：将小球相对位移映射为末端目标位移")
    parser.add_argument("--wait-start", action="store_true", help="固定相机先等待：H 回初始姿态，S 开始末端跟随")
    parser.add_argument("--mm-per-pixel", type=float, default=0.2, help="固定相机控制灵敏度，默认每像素对应 0.2 mm")
    parser.add_argument("--point-at-ball", action="store_true", help="固定相机控制夹爪转向；开始前用方向键对准黑球")
    parser.add_argument("--web", action="store_true", help="在黑球页面右栏操作回位、对准、启动和停止")
    parser.add_argument("--degrees-per-pixel", type=float, default=0.25, help="夹爪转向灵敏度，默认每像素 0.25 度")
    parser.add_argument("--follow-joints", type=int, nargs="+", help="固定相机的关节跟随：水平、垂直分别对应的关节 ID；可只指定水平关节")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--simulate", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", help="收敛时保存一帧验证图")
    args = parser.parse_args()
    config = load_advanced(args.advanced_config)
    settings = copy.deepcopy(config["vision"])
    if args.fixed_camera and (args.calibrate or args.simulate):
        parser.error("固定相机模式使用真实摄像头，不使用末端相机标定或末端相机模拟")
    if args.follow_joints and (not args.fixed_camera or not args.execute):
        parser.error("--follow-joints 需要 --fixed-camera --execute")
    if args.point_at_ball and (not args.fixed_camera or args.follow_joints):
        parser.error("--point-at-ball 需要 --fixed-camera，且不使用 --follow-joints")
    if args.web and not (args.fixed_camera and args.point_at_ball and args.execute and args.wait_start):
        parser.error("--web 需要 --fixed-camera --point-at-ball --execute --wait-start")
    if args.wait_start and (not args.fixed_camera or not args.execute or args.follow_joints):
        parser.error("--wait-start 需要 --fixed-camera --execute，且不使用 --follow-joints")
    if args.camera is not None:
        if args.camera != settings["camera"]:
            settings["calibrated"] = False
        settings["camera"] = args.camera
    if args.calibrate and not args.execute:
        parser.error("相机标定会实际移动末端，需同时指定 --execute")
    if args.headless and not args.simulate:
        parser.error("--headless 仅用于离线模拟")
    if args.execute and not args.follow_joints and not config["kinematics"]["calibrated"]:
        parser.error("先完成机械臂运动学标定")
    if args.execute and not args.fixed_camera and not args.calibrate and not settings["calibrated"]:
        parser.error("先运行 --execute --calibrate 标定末端相机")
    arm = camera = web = None
    try:
        if args.execute or args.simulate:
            arm = open_arm(config, load_config(args.config), args.simulate, args.port)
        if args.simulate:
            settings["image_jacobian"] = [[-3.0, 0.4], [0.2, -3.0]]
        else:
            camera = cv2.VideoCapture(settings["camera"])
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings["width"])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["height"])
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not camera.isOpened():
                raise RuntimeError("摄像头无法打开；检查索引和系统相机权限")
        if args.follow_joints:
            print(track_joints(camera, arm, settings, args.follow_joints))
        elif args.calibrate:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("无法读取摄像头")
            settings["height"], settings["width"] = frame.shape[:2]
            config["vision"] = calibrate(camera, arm, settings)
            save_advanced(args.advanced_config, config)
            print(f"已保存图像雅可比：{config['vision']['image_jacobian']}")
        else:
            if args.web:
                web = VisionHTTPServer(arm, args.advanced_config)
                threading.Thread(target=web.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
                print(f"黑球控制页面：{web.url}", flush=True)
                webbrowser.open(web.url)
            if args.wait_start and not args.web:
                print("固定摄像头对准屏幕黑球；在相机窗口按 H 回初始姿态，按 S 开始跟随；空格/Q/Esc 退出。", flush=True)
                if args.point_at_ball:
                    print("按 S 前先用方向键（或 I 上/K 下/J 左/L 右）把夹爪对准静止黑球；每次约 5 度。跟随中按 P 暂停并重新对准。", flush=True)
            result = track(camera, arm, settings, simulate=args.simulate,
                           preview=not (args.execute or args.simulate), headless=args.headless, output=args.output,
                           fixed_camera=args.fixed_camera, mm_per_pixel=args.mm_per_pixel, wait_start=args.wait_start,
                           point_at_ball=args.point_at_ball, degrees_per_pixel=args.degrees_per_pixel, web=web)
            print(result)
    except KeyboardInterrupt:
        print("正在停止……")
    except Exception as error:
        if web is None:
            raise
        try:
            arm.stop()
        except Exception as stop_error:
            error = RuntimeError(f"{error}；停止失败：{stop_error}")
        web.state.update(active=False, busy=False, fault=True, error=str(error), message="控制已停止")
        print(f"视觉控制停止：{error}", flush=True)
        web.closed.wait()
    finally:
        try:
            if web is not None:
                web.shutdown()
                web.server_close()
            if arm is not None:
                arm.stop()
        finally:
            if arm is not None and arm.bus is not None:
                arm.bus.close()
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
