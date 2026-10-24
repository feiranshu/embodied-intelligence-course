"""Robot capabilities from weeks 4–6 exposed as LLM function-calling tools.

The LLM never writes motor values directly: it picks a tool, the tool checks
the request against advanced.json limits and runs the existing week4/week6
code.  Every result is a small JSON-serialisable dict so it can be sent back
to the model as the tool message.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np

import agent_common
from agent_common import extract_json, jpeg_data_url, read_json, run_dir, write_json
from arm_backend import DIRECTIONS, JOINT_NAMES, arm_state, available_actions, available_poses, linear_delta

MAX_AGENT_MOVE_MM = 50.0
MAX_TRACK_TRAVEL_MM = 60.0
MAX_JOINT_DELTA_DEG = {
    "shoulder_pan": 25.0,
    "shoulder_lift": 25.0,
    "elbow_flex": 30.0,
    "wrist_flex": 30.0,
    "wrist_roll": 35.0,
    "gripper": 45.0,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_robot_state",
        "description": "读取机械臂当前关节角（度）、末端坐标（mm，基座坐标系：x 前、y 左、z 上）、"
                       "是否可以做末端直线运动，以及视觉追踪状态。做任何动作前后都可以调用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "list_capabilities",
        "description": "列出可用的预设姿态、动作序列、关节角允许范围和单次末端移动上限。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "go_to_pose",
        "description": "让机械臂运动到一个当前关节范围内的已记录预设姿态；名称必须来自 list_capabilities。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "预设姿态名，必须来自 list_capabilities"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "run_action",
        "description": "执行一个已录制的连贯动作序列（由多个预设姿态组成）。",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "动作名，必须来自 list_capabilities"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "move_end_effector",
        "description": "通过逆运动学让末端沿基座坐标系直线移动指定距离。起点必须在末端工作范围内"
                       "（get_robot_state 的 cartesian_ready 为 true），否则先 go_to_pose('初始状态')。",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": list(DIRECTIONS), "description": "前/后/左/右/上/下"},
            "distance_mm": {"type": "number", "description": f"移动距离，毫米，0～{MAX_AGENT_MOVE_MM:g}"}},
            "required": ["direction", "distance_mm"]}}},
    {"type": "function", "function": {
        "name": "set_joint_angles",
        "description": "小幅调整一个或多个关节的目标角度（度），未给出的关节保持不动；超出关节范围会被限制，"
                       "单次变化过大则拒绝。大幅运动应使用已记录的预设姿态。",
        "parameters": {"type": "object", "properties": {
            "joints": {"type": "object", "description": "关节名到角度的映射",
                       "properties": {name: {"type": "number"} for name in JOINT_NAMES}},
            "duration_s": {"type": "number", "description": "运动时间，0.5～5 秒，默认 2"}},
            "required": ["joints"]}}},
    {"type": "function", "function": {
        "name": "look",
        "description": "用腕部相机拍一张照片，让视觉模型回答关于当前画面的问题，例如“桌上有什么”。",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "想了解的问题，默认描述画面"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "detect_object",
        "description": "根据文字描述在腕部相机画面中寻找物体，返回 found、object_id 和像素位置。"
                       "描述中含“ArUco/标记”时用 ArUco 检测，否则用视觉模型定位。",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string", "description": "物体描述，例如“红色的杯子”"}},
            "required": ["description"]}}},
    {"type": "function", "function": {
        "name": "start_tracking",
        "description": "对 detect_object 找到的物体启动第 6 周实现的视觉伺服：每步小幅移动末端，让物体回到画面中央。"
                       "在后台运行，可用 get_robot_state 查看进度、stop_tracking 停止。需要先处于“初始状态”等工作姿态。",
        "parameters": {"type": "object", "properties": {
            "object_id": {"type": "integer"},
            "follow": {"type": "boolean", "description": "true=居中后继续跟随直到停止；false=居中即结束（默认）"},
            "max_steps": {"type": "integer", "description": "最多移动步数，1～40，默认 15"},
            "max_travel_mm": {"type": "number", "description": "累计相机步进上限，5～60 mm，默认 60"}},
            "required": ["object_id"]}}},
    {"type": "function", "function": {
        "name": "stop_tracking",
        "description": "停止当前视觉追踪任务。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "imitate_person",
        "description": "拍一张人的照片，由视觉模型解析手臂动作并让机械臂模仿（任务 4.1 的完整流程）。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "stop_robot",
        "description": "立即停止所有运动和追踪。用户说停、危险或任何异常时必须马上调用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

DETECT_PROMPT = (
    "在图片中寻找：{description}。只输出一个 JSON 对象，不要输出其他文字：\n"
    '{{"found": true或false, "bbox": [x1, y1, x2, y2], "label": "物体名称"}}\n'
    "bbox 是物体外接矩形，坐标归一化到 0～1000（左上角为 0,0，右下角为 1000,1000）。没找到时 found 为 false。"
)


class TrackingJob:
    """Background visual servo: keep the tracked object in the image centre."""

    def __init__(self, toolkit, target: dict, *, follow: bool, max_steps: int, max_travel_mm: float):
        self.toolkit = toolkit
        self.target = target
        self.follow = follow
        self.max_steps = max_steps
        self.max_travel_mm = max_travel_mm
        self.cancel = threading.Event()
        self.status = "starting"
        self.steps = []
        self.error = None
        self.thread = threading.Thread(target=self._run, name="agent-tracking", daemon=True)

    def summary(self) -> dict:
        return {"object_id": self.target["object_id"], "label": self.target["label"], "status": self.status,
                "steps": len(self.steps),
                "travel_mm": round(sum(float(np.linalg.norm(step["camera_mm"])) for step in self.steps), 2),
                "max_travel_mm": self.max_travel_mm,
                "error": self.error, "running": self.thread.is_alive()}

    def _detector(self):
        from part3_visual_servo.visual_servo import MarkerDetector
        from part4_object_tracking.object_tracking import OpticalFlowTracker

        if self.target["kind"] == "aruco":
            return MarkerDetector(self.target["marker_id"], 0, "DICT_4X4_50")
        x1, y1, x2, y2 = self.target["bbox_px"]
        return OpticalFlowTracker(self.target["frame"], (x1, y1, x2 - x1, y2 - y1))

    def _run(self):
        from common import base_gripper_pose, ensure_image_size
        from part3_visual_servo.motion_model import matching_motion_model
        from part3_visual_servo.visual_servo import camera_step

        kit = self.toolkit
        motion = kit.arm.arm
        original_model = motion.model
        try:
            K, dist, size = kit.calibration()
            T_gc = kit.hand_eye_matrix()
            detector = self._detector()
            motion.model = matching_motion_model(original_model, kit.hand_eye_data())
            motion.cancel.clear()
            motion.cartesian_start_pose()
            self.status = "tracking"
            lost, last_stamp, stable, previous = 0, 0.0, 0, None
            previous_move_error, diverging = None, 0
            while not self.cancel.is_set():
                frame, stamp = kit.camera.latest()
                if stamp == last_stamp:
                    time.sleep(0.03)
                    continue
                last_stamp = stamp
                ensure_image_size(frame, size)
                observation = detector.detect(frame, K, dist)
                if observation is None:
                    lost += 1
                    stable, previous = 0, None
                    if lost >= 8:
                        self.status = "target_lost"
                        break
                    continue
                lost = 0
                center = observation[0]
                stable = stable + 1 if previous is not None and np.linalg.norm(center - previous) < 20 else 1
                previous = center.copy()
                if stable < 3:
                    continue
                delta_camera = camera_step(center, size, kit.track_step_mm, None, 0.0, forward=False)
                if delta_camera is None:
                    if not self.follow:
                        self.status = "centered"
                        break
                    self.status = "centered_following"
                    continue
                if len(self.steps) >= self.max_steps:
                    self.status = "step_limit"
                    break
                traveled = sum(float(np.linalg.norm(step["camera_mm"])) for step in self.steps)
                if traveled + float(np.linalg.norm(delta_camera)) > self.max_travel_mm + 1e-6:
                    self.status = "travel_limit"
                    break
                center_error = float(np.linalg.norm(center - np.asarray(size, dtype=float) / 2))
                if previous_move_error is not None and center_error > previous_move_error + 10:
                    diverging += 1
                else:
                    diverging = 0
                if diverging >= 2:
                    self.status = "diverging"
                    break
                previous_move_error = center_error
                R_bc = (base_gripper_pose(motion.positions(), motion.model) @ T_gc)[:3, :3]
                delta_base = R_bc @ delta_camera
                motion.move_linear(delta_base)
                self.steps.append({"center_px": np.round(center, 1).tolist(),
                                   "camera_mm": delta_camera.tolist(),
                                   "base_mm": np.round(delta_base, 2).tolist(),
                                   "feedback": getattr(motion, "last_move", None)})
                self.status = "tracking"
                stable, previous = 0, None
                if (motion.last_move or {}).get("projected"):
                    self.status = "workspace_edge"
                    break
            else:
                self.status = "stopped"
        except Exception as error:
            self.error = str(error)
            self.status = "error"
            try:
                kit.arm.stop()
            except Exception as stop_error:
                self.error += f"；停止失败：{stop_error}，请准备断电"
        finally:
            motion.model = original_model
            kit.motion_lock.release()
            write_json(kit.log_dir / f"tracking_{self.target['object_id']}.json",
                       {**self.summary(), "running": False, "steps_detail": self.steps})


class RobotToolkit:
    def __init__(self, arm, camera=None, vision_llm=None, *, config=None, person_camera=None,
                 intrinsics=agent_common.INTRINSICS, hand_eye=agent_common.HAND_EYE,
                 track_step_mm=5.0, log_dir: Path | None = None, read_only: bool = False):
        self.arm = arm
        self.camera = camera
        self.person_camera = person_camera or camera
        self.vision_llm = vision_llm
        self.config = config or arm.config
        self.intrinsics = Path(intrinsics)
        self.hand_eye = Path(hand_eye)
        self.track_step_mm = float(track_step_mm)
        self.read_only = bool(read_only)
        self.motion_lock = threading.Lock()
        self.objects: dict[int, dict] = {}
        self.tracking: TrackingJob | None = None
        self.log_dir = log_dir or run_dir("agent")
        self.calls = []
        self.handlers = {
            "get_robot_state": self.get_robot_state,
            "list_capabilities": self.list_capabilities,
            "go_to_pose": self.go_to_pose,
            "run_action": self.run_action,
            "move_end_effector": self.move_end_effector,
            "set_joint_angles": self.set_joint_angles,
            "look": self.look,
            "detect_object": self.detect_object,
            "start_tracking": self.start_tracking,
            "stop_tracking": self.stop_tracking,
            "imitate_person": self.imitate_person,
            "stop_robot": self.stop_robot,
        }

    # ---------- dispatch ----------
    def schemas(self) -> list[dict]:
        """Hide tools whose camera, model or current calibration prerequisite is missing."""
        needs_camera = {"look", "detect_object", "start_tracking", "stop_tracking"}
        needs_person_camera = {"imitate_person"}
        needs_vision = {"look", "detect_object", "imitate_person"}
        needs_kinematics = {"move_end_effector", "start_tracking"}
        changes_robot = {"go_to_pose", "run_action", "move_end_effector", "set_joint_angles",
                         "start_tracking", "imitate_person"}
        kinematics_ready = self.config.get("kinematics", {}).get("calibrated") is True
        schemas = [copy.deepcopy(tool) for tool in TOOL_SCHEMAS
                   if not (tool["function"]["name"] in needs_camera and self.camera is None)
                   and not (tool["function"]["name"] in needs_person_camera and self.person_camera is None)
                   and not (tool["function"]["name"] in needs_vision and self.vision_llm is None)
                   and not (tool["function"]["name"] in needs_kinematics and not kinematics_ready)
                   and not (self.read_only and tool["function"]["name"] in changes_robot)]
        for tool in schemas:
            function = tool["function"]
            if function["name"] == "go_to_pose":
                function["parameters"]["properties"]["name"]["enum"] = available_poses(self.config)
            elif function["name"] == "run_action":
                function["parameters"]["properties"]["name"]["enum"] = available_actions(self.config)
        return schemas

    def dispatch(self, name: str, arguments) -> dict:
        started = time.monotonic()
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
            if name not in self.handlers or name not in {t["function"]["name"] for t in self.schemas()}:
                raise ValueError(f"没有名为 {name} 的可用工具")
            result = {"ok": True, **self.handlers[name](**arguments)}
        except TypeError as error:
            result = {"ok": False, "error": f"参数错误：{error}"}
        except Exception as error:
            result = {"ok": False, "error": str(error)}
        self.calls.append({"tool": name, "arguments": arguments, "result": result,
                           "seconds": round(time.monotonic() - started, 2)})
        write_json(self.log_dir / "tool_calls.json", self.calls)
        return result

    def _motion(self, command) -> dict:
        if self.tracking is not None and self.tracking.thread.is_alive():
            raise RuntimeError("视觉追踪正在运行，先调用 stop_tracking")
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError("机械臂正在执行其他动作")
        try:
            self.arm.run(command)
        finally:
            self.motion_lock.release()
        return {"state": arm_state(self.arm)}

    # ---------- state ----------
    def get_robot_state(self) -> dict:
        state = arm_state(self.arm)
        state["tracking"] = self.tracking.summary() if self.tracking else None
        return state

    def list_capabilities(self) -> dict:
        return {"poses": available_poses(self.config), "actions": available_actions(self.config),
                "joint_limits_deg": dict(zip(JOINT_NAMES, self.arm.joints.limits_deg)),
                "max_joint_delta_deg": MAX_JOINT_DELTA_DEG,
                "max_move_mm": self.max_move_mm, "camera": self.camera is not None,
                "kinematics_calibrated": self.config.get("kinematics", {}).get("calibrated") is True,
                "known_objects": {i: o["label"] for i, o in self.objects.items()}}

    @property
    def max_move_mm(self) -> float:
        return min(MAX_AGENT_MOVE_MM, float(self.config.get("max_distance_mm", self.config["max_step_mm"])))

    # ---------- motion ----------
    def go_to_pose(self, name: str) -> dict:
        poses = available_poses(self.config)
        if name not in poses:
            raise ValueError(f"姿态“{name}”不存在或不符合当前关节范围；可用：{poses}")
        return self._motion(("home" if name == "初始状态" else "pose", name))

    def run_action(self, name: str) -> dict:
        if name not in available_actions(self.config):
            raise ValueError(f"没有可执行的动作“{name}”；可用：{available_actions(self.config)}")
        return self._motion(("action", name))

    def move_end_effector(self, direction: str, distance_mm: float) -> dict:
        distance_mm = float(distance_mm)
        if not 0 < distance_mm <= self.max_move_mm:
            raise ValueError(f"distance_mm 必须在 0～{self.max_move_mm:g} 之间")
        result = self._motion(("move", linear_delta(direction, distance_mm)))
        result["last_move"] = getattr(self.arm.arm, "last_move", None)
        return result

    def set_joint_angles(self, joints: dict, duration_s: float = 2.0) -> dict:
        duration_s = float(duration_s)
        if not 0.5 <= duration_s <= 5:
            raise ValueError("duration_s 需在 0.5～5 秒之间")
        if not isinstance(joints, dict) or not joints:
            raise ValueError("joints 至少包含一个关节")
        current_raw = [int(v) for v in self.arm.positions()]
        target, notes = self.arm.joints.to_raw(joints, current_raw)
        current_deg = dict(zip(JOINT_NAMES, self.arm.joints.to_degrees(current_raw)))
        target_deg = dict(zip(JOINT_NAMES, self.arm.joints.to_degrees(target)))
        excessive = {name: round(abs(target_deg[name] - current_deg[name]), 1) for name in joints
                     if abs(target_deg[name] - current_deg[name]) > MAX_JOINT_DELTA_DEG[name]}
        if excessive:
            limits = {name: MAX_JOINT_DELTA_DEG[name] for name in excessive}
            raise ValueError(f"单次关节变化过大：{excessive}°；允许上限：{limits}°。请使用已记录姿态")
        result = self._motion(("joints", (target, duration_s)))
        result["clamped"] = notes
        return result

    def stop_robot(self) -> dict:
        if self.tracking is not None:
            self.tracking.cancel.set()
        self.arm.stop()
        if self.tracking is not None:
            self.tracking.thread.join(timeout=3)
        return {"message": "已请求停止所有运动", "state": arm_state(self.arm)}

    # ---------- vision ----------
    def _frame(self, out_name: str):
        if self.camera is None:
            raise RuntimeError("没有连接摄像头")
        frame, _ = self.camera.latest()
        from common import write_image
        path = self.log_dir / f"{out_name}_{time.strftime('%H%M%S')}.jpg"
        write_image(path, frame)
        return frame, path

    def _ask_vision(self, frame, prompt: str) -> str:
        if self.vision_llm is None:
            raise RuntimeError("没有配置视觉模型")
        completion = self.vision_llm.chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": jpeg_data_url(frame)}},
            {"type": "text", "text": prompt}]}], temperature=0.2)
        return completion.choices[0].message.content or ""

    def look(self, question: str = "请简要描述画面中有什么、它们大致在画面的什么位置。") -> dict:
        frame, path = self._frame("look")
        return {"answer": self._ask_vision(frame, question), "photo": str(path)}

    def detect_object(self, description: str) -> dict:
        frame, path = self._frame("detect")
        height, width = frame.shape[:2]
        object_id = len(self.objects) + 1
        if any(word in description.lower() for word in ("aruco", "标记", "二维码", "marker")):
            import cv2

            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is None:
                return {"found": False, "photo": str(path)}
            quad = np.asarray(corners[0]).reshape(4, 2)
            x1, y1 = quad.min(axis=0)
            x2, y2 = quad.max(axis=0)
            target = {"kind": "aruco", "marker_id": int(ids.ravel()[0]), "label": f"ArUco {int(ids.ravel()[0])}"}
        else:
            reply = self._ask_vision(frame, DETECT_PROMPT.format(description=description))
            data = extract_json(reply)
            if not data.get("found"):
                return {"found": False, "photo": str(path), "model_reply": reply[:200]}
            bbox = data.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
                raise ValueError(f"视觉模型返回的 bbox 无效：{bbox!r}")
            x1, y1, x2, y2 = bbox
            x1, x2 = sorted((x1 / 1000 * width, x2 / 1000 * width))
            y1, y2 = sorted((y1 / 1000 * height, y2 / 1000 * height))
            target = {"kind": "flow", "label": str(data.get("label") or description)}
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width - 1, int(round(x2))), min(height - 1, int(round(y2)))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return {"found": False, "photo": str(path), "message": "目标框太小，无法稳定跟踪"}
        target.update(object_id=object_id, bbox_px=[x1, y1, x2, y2], frame=frame)
        self.objects[object_id] = target
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        return {"found": True, "object_id": object_id, "label": target["label"], "bbox_px": [x1, y1, x2, y2],
                "center_normalized": [round(center[0] / width, 2), round(center[1] / height, 2)],
                "photo": str(path)}

    def calibration(self):
        from common import calibration
        return calibration(self.intrinsics)

    def hand_eye_data(self) -> dict:
        return read_json(self.hand_eye)

    def hand_eye_matrix(self):
        from common import load_hand_eye
        return load_hand_eye(self.hand_eye)

    def start_tracking(self, object_id: int, follow: bool = False, max_steps: int = 15,
                       max_travel_mm: float = MAX_TRACK_TRAVEL_MM) -> dict:
        object_id = int(object_id)
        if object_id not in self.objects:
            raise ValueError(f"未知 object_id {object_id}，先调用 detect_object")
        max_steps = int(max_steps)
        if not 1 <= max_steps <= 40:
            raise ValueError("max_steps 需在 1～40 之间")
        max_travel_mm = float(max_travel_mm)
        if not 5 <= max_travel_mm <= MAX_TRACK_TRAVEL_MM:
            raise ValueError(f"max_travel_mm 需在 5～{MAX_TRACK_TRAVEL_MM:g} 之间")
        if not self.arm.simulated:
            assessment = self.hand_eye_data().get("assessment", {})
            if (assessment.get("independent_accuracy_validated") is not True
                    or assessment.get("weak_joint_directions_not_identified") is True):
                raise RuntimeError("当前手眼结果尚未完成独立精度/弱关节方向验证，实机视觉追踪保持锁定")
        if self.tracking is not None and self.tracking.thread.is_alive():
            raise RuntimeError("已有追踪任务在运行")
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError("机械臂正在执行其他动作")
        try:
            self.arm.arm.cartesian_start_pose()
            self.tracking = TrackingJob(self, self.objects[object_id], follow=bool(follow),
                                        max_steps=max_steps, max_travel_mm=max_travel_mm)
            self.tracking.thread.start()
        except Exception:
            self.motion_lock.release()
            raise
        return {"tracking": self.tracking.summary()}

    def stop_tracking(self) -> dict:
        if self.tracking is None or not self.tracking.thread.is_alive():
            return {"message": "当前没有追踪任务", "tracking": self.tracking.summary() if self.tracking else None}
        self.tracking.cancel.set()
        self.tracking.thread.join(timeout=10)
        if self.tracking.thread.is_alive():
            self.arm.stop()
            self.tracking.thread.join(timeout=3)
        return {"tracking": self.tracking.summary()}

    def imitate_person(self) -> dict:
        from part1_imitation.imitate import imitate_once

        if self.person_camera is None or self.vision_llm is None:
            raise RuntimeError("模仿需要摄像头和视觉模型")
        if self.tracking is not None and self.tracking.thread.is_alive():
            raise RuntimeError("视觉追踪正在运行，先调用 stop_tracking")
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError("机械臂正在执行其他动作")
        try:
            log = imitate_once(self.vision_llm, self.person_camera, self.arm, self.config,
                               out_dir=self.log_dir / f"imitate_{time.strftime('%H%M%S')}")
        finally:
            self.motion_lock.release()
        if not log.get("ok"):
            raise RuntimeError(log.get("error", "模仿失败"))
        return {"description": log["parsed"]["description"], "target_deg": log["target_deg"],
                "executed_pose": log["executed_anchor"]["pose"],
                "safety_notes": log["safety_notes"], "clamped": log["clamp_notes"]}

    def close(self):
        try:
            if self.tracking is not None and self.tracking.thread.is_alive():
                self.stop_tracking()
        finally:
            self.arm.close()
