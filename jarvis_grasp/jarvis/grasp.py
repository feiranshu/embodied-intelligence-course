"""Object grasp planning, 3D ray-plane localization, and closed-loop grasp execution."""

from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np

from .models import Candidate, GraspPlan, GraspResult

# Calibrated physical SCS215 ID 6 (gripper) raw tick limits from vendor/scs215/advanced.json
GRIPPER_MIN_RAW = 260
GRIPPER_MAX_RAW = 662

PLACE_OFFSETS_MM: dict[str, tuple[float, float, float]] = {
    "left": (0.0, 45.0, 0.0),
    "right": (0.0, -45.0, 0.0),
    "front": (35.0, 0.0, 0.0),
    "back": (-35.0, 0.0, 0.0),
    "center": (0.0, 0.0, 0.0),
}


class GraspFailure(RuntimeError):
    def __init__(self, message: str, *, stage: str = "", gripper_raw: int | None = None):
        super().__init__(message)
        self.stage = stage
        self.gripper_raw = gripper_raw


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_grasp_config(config: dict, *, mode: str = "simulation") -> dict[str, Any]:
    """Validate grasp geometry, collision floor, and gripper tick bounds."""
    grasp = dict(config.get("grasp") or {})
    if mode == "hardware":
        hw_grasp = dict((config.get("hardware") or {}).get("grasp") or {})
        if hw_grasp.get("verified") is not True:
            raise ValueError("实机抓取未解锁：hardware.grasp.verified 必须为 true，且需先实测桌面高度与夹爪行程")
        for key in (
            "table_z_mm",
            "grasp_z_mm",
            "pre_grasp_z_mm",
            "lift_z_mm",
            "camera_to_gripper_mm",
            "max_grasp_step_mm",
            "max_grasp_travel_mm",
            "gripper_open_raw",
            "gripper_close_raw",
            "gripper_held_range_raw",
            "gripper_hold_bias_raw",
            "gripper_max_load",
            "gripper_speed",
            "gripper_duration_s",
        ):
            if key in hw_grasp and hw_grasp[key] is not None:
                grasp[key] = hw_grasp[key]
        if "lift_z_mm" in hw_grasp and hw_grasp["lift_z_mm"] is not None:
            grasp["lift_height_mm"] = hw_grasp["lift_z_mm"]
        if "max_grasp_step_mm" in hw_grasp and hw_grasp["max_grasp_step_mm"] is not None:
            grasp["descend_step_mm"] = float(hw_grasp["max_grasp_step_mm"])
            grasp["lift_step_mm"] = float(hw_grasp["max_grasp_step_mm"])
        for required in ("table_z_mm", "grasp_z_mm", "pre_grasp_z_mm", "lift_height_mm", "camera_to_gripper_mm"):
            if grasp.get(required) is None:
                raise ValueError(f"实机抓取缺少已实测参数：{required}")

    defaults = {
        "center_fraction": 0.14,
        "hold_s": 0.6,
        "observe_height_mm": 160.0,
        "pre_grasp_z_mm": 85.0,
        "grasp_z_mm": 25.0,
        "lift_height_mm": 135.0,
        "table_z_mm": 10.0,
        "descend_step_mm": 15.0,
        "lift_step_mm": 15.0,
        "camera_to_gripper_mm": [0.0, 0.0, 0.0],
        "place_offset_mm": [0.0, 0.0, 0.0],
        "gripper_open_raw": 430,
        "gripper_close_raw": 278,
        "gripper_held_range_raw": [298, 415],
        "gripper_hold_bias_raw": 4,
        "gripper_max_load": 350,
        "gripper_speed": 180,
        "gripper_duration_s": 0.8,
        "settle_s": 0.15,
        "verify_after_lift": True,
    }
    merged = {**defaults, **grasp}

    for key in (
        "center_fraction",
        "hold_s",
        "observe_height_mm",
        "pre_grasp_z_mm",
        "grasp_z_mm",
        "lift_height_mm",
        "table_z_mm",
        "descend_step_mm",
        "lift_step_mm",
        "gripper_duration_s",
        "settle_s",
    ):
        if not _is_finite_number(merged[key]):
            raise ValueError(f"grasp.{key} 必须是有限数值")
    if not 0 < merged["center_fraction"] <= 1.0:
        raise ValueError("grasp.center_fraction 必须在 (0, 1] 内")
    if merged["hold_s"] <= 0 or merged["descend_step_mm"] <= 0 or merged["lift_step_mm"] <= 0:
        raise ValueError("grasp.hold_s、descend_step_mm、lift_step_mm 必须为正数")
    if merged["settle_s"] < 0 or not 0.1 <= merged["gripper_duration_s"] <= 10.0:
        raise ValueError("grasp.settle_s 不能为负且 gripper_duration_s 需在 0.1～10 秒内")
    if merged["grasp_z_mm"] < merged["table_z_mm"]:
        raise ValueError("grasp_z_mm 不能低于桌面安全平面 table_z_mm")
    if merged["pre_grasp_z_mm"] < merged["grasp_z_mm"]:
        raise ValueError("pre_grasp_z_mm 必须高于或等于 grasp_z_mm")
    if merged["lift_height_mm"] <= merged["grasp_z_mm"]:
        raise ValueError("lift_height_mm 必须高于 grasp_z_mm")

    offset = np.asarray(merged["camera_to_gripper_mm"], dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("grasp.camera_to_gripper_mm 必须是 3 维有限毫米向量")
    merged["camera_to_gripper_mm"] = tuple(float(v) for v in offset)

    place_offset = np.asarray(merged["place_offset_mm"], dtype=float)
    if place_offset.shape != (3,) or not np.all(np.isfinite(place_offset)):
        raise ValueError("grasp.place_offset_mm 必须是 3 维有限毫米向量")
    merged["place_offset_mm"] = tuple(float(v) for v in place_offset)

    open_raw = merged["gripper_open_raw"]
    close_raw = merged["gripper_close_raw"]
    held_range = merged["gripper_held_range_raw"]
    if (
        isinstance(open_raw, bool)
        or isinstance(close_raw, bool)
        or not isinstance(open_raw, int)
        or not isinstance(close_raw, int)
        or not GRIPPER_MIN_RAW <= close_raw < open_raw <= GRIPPER_MAX_RAW
    ):
        raise ValueError(
            f"夹爪开合原始读数必须满足 {GRIPPER_MIN_RAW} <= close_raw < open_raw <= {GRIPPER_MAX_RAW}"
        )
    if (
        not isinstance(held_range, (list, tuple))
        or len(held_range) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in held_range)
        or not close_raw < held_range[0] < held_range[1] < open_raw
    ):
        raise ValueError("gripper_held_range_raw 必须是位于 (close_raw, open_raw) 内的递增整数区间")
    merged["gripper_held_range_raw"] = (int(held_range[0]), int(held_range[1]))

    max_load = merged["gripper_max_load"]
    speed = merged["gripper_speed"]
    hold_bias = merged["gripper_hold_bias_raw"]
    if isinstance(max_load, bool) or not isinstance(max_load, int) or not 50 <= max_load <= 800:
        raise ValueError("gripper_max_load 必须在 50～800 之间")
    if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= 1000:
        raise ValueError("gripper_speed 必须在 1～1000 之间")
    if isinstance(hold_bias, bool) or not isinstance(hold_bias, int) or not 0 <= hold_bias <= 30:
        raise ValueError("gripper_hold_bias_raw 必须在 0～30 之间")
    return merged


def ray_plane_intersection(
    pixel_uv: tuple[float, float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    T_base_camera: np.ndarray,
    plane_z_mm: float,
) -> tuple[float, float, float]:
    """Back-project an image pixel onto a horizontal tabletop plane z = plane_z_mm in base frame."""
    K = np.asarray(camera_matrix, dtype=np.float64)
    T = np.asarray(T_base_camera, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("相机内参矩阵 K 必须是有限的 3×3 矩阵")
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("T_base_camera 必须是有限的 4×4 齐次变换矩阵")
    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    pts = np.array([[[float(pixel_uv[0]), float(pixel_uv[1])]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, K, dist).reshape(2)
    ray_cam = np.array([undistorted[0], undistorted[1], 1.0], dtype=np.float64)
    origin_base = T[:3, 3]
    dir_base = T[:3, :3] @ ray_cam
    if abs(float(dir_base[2])) < 1e-6:
        raise ValueError("相机视线与桌面平面平行，无法求交")
    t = (float(plane_z_mm) - float(origin_base[2])) / float(dir_base[2])
    if not math.isfinite(t) or t <= 0:
        raise ValueError("视线与桌面交点位于相机后方，请检查标定与桌面高度")
    point = origin_base + t * dir_base
    return tuple(round(float(v), 2) for v in point)


def estimate_grasp_pose(
    candidate: Candidate,
    frame_shape: tuple[int, ...],
    grasp_config: dict[str, Any],
    *,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    T_base_camera: np.ndarray | None = None,
    place_direction: str | None = None,
) -> GraspPlan:
    """Build a validated GraspPlan from a detected candidate and optional 3D camera geometry."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    cx, cy = candidate.center
    estimated_base_xyz = None
    if T_base_camera is not None:
        T = np.asarray(T_base_camera, dtype=np.float64)
        if candidate.position_3d_mm is not None:
            p_cam = np.array([*candidate.position_3d_mm, 1.0], dtype=np.float64)
            p_base = (T @ p_cam)[:3]
            estimated_base_xyz = tuple(round(float(v), 2) for v in p_base)
        elif camera_matrix is not None:
            estimated_base_xyz = ray_plane_intersection(
                (cx * width, cy * height),
                camera_matrix,
                dist_coeffs,
                T,
                float(grasp_config["grasp_z_mm"]),
            )

    place_offset = tuple(float(v) for v in grasp_config["place_offset_mm"])
    if place_direction is not None:
        place_offset = PLACE_OFFSETS_MM.get(place_direction, place_offset)

    return GraspPlan(
        target_id=candidate.id,
        target_label=candidate.label,
        center_norm=(round(float(cx), 4), round(float(cy), 4)),
        angle_deg=round(float(candidate.angle_deg), 1),
        bbox_norm=tuple(round(float(v), 4) for v in candidate.bbox),
        camera_to_gripper_mm=tuple(float(v) for v in grasp_config["camera_to_gripper_mm"]),
        pre_grasp_z_mm=float(grasp_config["pre_grasp_z_mm"]),
        grasp_z_mm=float(grasp_config["grasp_z_mm"]),
        lift_z_mm=float(grasp_config["lift_height_mm"]),
        table_z_mm=float(grasp_config["table_z_mm"]),
        gripper_open_raw=int(grasp_config["gripper_open_raw"]),
        gripper_close_raw=int(grasp_config["gripper_close_raw"]),
        gripper_held_range_raw=tuple(int(v) for v in grasp_config["gripper_held_range_raw"]),
        estimated_base_xyz_mm=estimated_base_xyz,
        place_offset_mm=place_offset,
    )


class GraspExecutor:
    """Staged tabletop pick-and-place state machine with stop checks and grasp verification."""

    def __init__(
        self,
        motion,
        grasp_config: dict[str, Any],
        *,
        check_stop: Callable[[], None],
        sleep: Callable[[float], None],
        on_stage: Callable[[str, str], None] | None = None,
        on_event: Callable[..., None] | None = None,
    ):
        self.motion = motion
        self.cfg = grasp_config
        self.check_stop = check_stop
        self.sleep = sleep
        self.on_stage = on_stage or (lambda stage, message: None)
        self.on_event = on_event or (lambda event, **kwargs: None)

    def _move_z_to(self, target_z_mm: float, step_mm: float, stage: str):
        if target_z_mm < self.cfg["table_z_mm"]:
            raise GraspFailure(
                f"目标高度 {target_z_mm:.1f} mm 低于桌面安全平面 {self.cfg['table_z_mm']:.1f} mm",
                stage=stage,
            )
        while True:
            self.check_stop()
            current_z = float(self.motion.position_mm()[2])
            remaining = target_z_mm - current_z
            if abs(remaining) <= 0.5:
                break
            dz = math.copysign(min(abs(remaining), step_mm), remaining)
            if current_z + dz < self.cfg["table_z_mm"] - 1e-6:
                raise GraspFailure("垂直运动触及桌面安全平面限制", stage=stage)
            self.motion.move_delta([0.0, 0.0, dz])
            self.on_event("grasp_motion", stage=stage, delta_mm=[0.0, 0.0, round(dz, 2)])
            if self.cfg["settle_s"] > 0:
                self.sleep(min(0.05, self.cfg["settle_s"]))

    def _move_xy(self, dx_mm: float, dy_mm: float, step_mm: float, stage: str):
        remaining = np.array([float(dx_mm), float(dy_mm)], dtype=float)
        while float(np.linalg.norm(remaining)) > 0.5:
            self.check_stop()
            dist = float(np.linalg.norm(remaining))
            scale = min(1.0, step_mm / dist)
            step = remaining * scale
            self.motion.move_delta([float(step[0]), float(step[1]), 0.0])
            self.on_event("grasp_motion", stage=stage, delta_mm=[round(float(step[0]), 2), round(float(step[1]), 2), 0.0])
            remaining -= step
            if self.cfg["settle_s"] > 0:
                self.sleep(min(0.05, self.cfg["settle_s"]))

    def _safe_retreat_after_miss(self, plan: GraspPlan):
        try:
            self.check_stop()
            self.motion.set_gripper(
                plan.gripper_open_raw,
                expect_object=False,
                speed=self.cfg["gripper_speed"],
                duration_s=self.cfg["gripper_duration_s"],
            )
            self._move_z_to(plan.lift_z_mm, self.cfg["lift_step_mm"], "retreating")
        except InterruptedError:
            raise
        except Exception:
            pass

    def execute_grasp(
        self,
        plan: GraspPlan,
        *,
        read_frame: Callable[[], Any],
        place_after: bool = False,
    ) -> tuple[GraspResult, Any]:
        """Run open -> align -> descend -> close -> lift -> verify (and optional place)."""
        telemetry: dict[str, Any] = {"plan": plan.to_dict()}
        self.on_event("grasp_plan", plan=plan.to_dict())

        # 1. Open gripper before descending
        self.check_stop()
        self.on_stage("opening", f"正在张开夹爪，准备抓取{plan.target_label}。")
        open_fb = self.motion.set_gripper(
            plan.gripper_open_raw,
            expect_object=False,
            speed=self.cfg["gripper_speed"],
            duration_s=self.cfg["gripper_duration_s"],
        )
        telemetry["open"] = open_fb
        self.on_event("gripper_move", stage="opening", feedback=open_fb)
        if self.cfg["settle_s"] > 0:
            self.sleep(self.cfg["settle_s"])

        # 2. Apply calibrated camera-to-gripper XY offset
        ox, oy, _ = plan.camera_to_gripper_mm
        if math.hypot(ox, oy) > 0.5:
            self.check_stop()
            self.on_stage("aligning", "正在补偿相机到夹爪的相对位置偏移。")
            self._move_xy(ox, oy, self.cfg["descend_step_mm"], "aligning")

        # 3. Descend to pre-grasp clearance then grasp plane
        self.check_stop()
        self.on_stage("descending", f"正在安全下降至{plan.target_label}抓取高度。")
        if float(self.motion.position_mm()[2]) > plan.pre_grasp_z_mm + 0.5:
            self._move_z_to(plan.pre_grasp_z_mm, self.cfg["descend_step_mm"], "descending")
            if self.cfg["settle_s"] > 0:
                self.sleep(self.cfg["settle_s"])
        self._move_z_to(plan.grasp_z_mm, self.cfg["descend_step_mm"], "descending")

        # 4. Close gripper & detect contact / empty grasp
        self.check_stop()
        self.on_stage("closing", f"正在闭合夹爪夹取{plan.target_label}。")
        close_fb = self.motion.set_gripper(
            plan.gripper_close_raw,
            expect_object=True,
            held_range=plan.gripper_held_range_raw,
            max_load=self.cfg["gripper_max_load"],
            hold_bias=self.cfg["gripper_hold_bias_raw"],
            speed=self.cfg["gripper_speed"],
            duration_s=self.cfg["gripper_duration_s"],
        )
        telemetry["close"] = close_fb
        self.on_event("gripper_move", stage="closing", feedback=close_fb)
        if self.cfg["settle_s"] > 0:
            self.sleep(self.cfg["settle_s"])
        if not close_fb.get("gripped"):
            self._safe_retreat_after_miss(plan)
            raise GraspFailure(
                "未夹到目标物体（夹爪空抓），已张开并抬起退回",
                stage="closing",
                gripper_raw=close_fb.get("raw"),
            )

        # 5. Lift object smoothly to safe travel height
        self.check_stop()
        self.on_stage("lifting", f"已夹住{plan.target_label}，正在抬起机械臂。")
        self._move_z_to(plan.lift_z_mm, self.cfg["lift_step_mm"], "lifting")
        if self.cfg["settle_s"] > 0:
            self.sleep(self.cfg["settle_s"])

        # 6. Verify object is still securely held after lift + capture fresh frame
        self.check_stop()
        self.on_stage("verifying", "正在核对抬起后的夹持状态与最新画面。")
        state_after = self.motion.gripper_state()
        telemetry["after_lift"] = state_after
        after_frame = read_frame()
        self.check_stop()
        if self.cfg.get("verify_after_lift", True) and not state_after.get("holding"):
            raise GraspFailure(
                "抬起过程中物体滑落，抓取未通过核对",
                stage="verifying",
                gripper_raw=state_after.get("raw"),
            )

        # 7. Optional immediate placement if user requested a destination
        if place_after:
            after_frame = self.execute_release(
                plan.place_offset_mm,
                read_frame=read_frame,
            )
            result = GraspResult(
                success=True,
                stage="placed",
                target_id=plan.target_id,
                target_label=plan.target_label,
                gripper_raw=self.motion.gripper_state().get("raw"),
                gripper_load=close_fb.get("load"),
                lifted=True,
                verified=True,
                telemetry=telemetry,
            )
            self.on_event("grasp_verified", result=result.to_dict())
            return result, after_frame

        result = GraspResult(
            success=True,
            stage="lifted",
            target_id=plan.target_id,
            target_label=plan.target_label,
            gripper_raw=state_after.get("raw"),
            gripper_load=close_fb.get("load"),
            lifted=True,
            verified=True,
            telemetry=telemetry,
        )
        self.on_event("grasp_verified", result=result.to_dict())
        return result, after_frame

    def execute_release(
        self,
        place_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        read_frame: Callable[[], Any],
    ) -> Any:
        """Move horizontally if requested, descend safely, open gripper, and lift back."""
        self.check_stop()
        px, py, _ = place_offset_mm
        if math.hypot(px, py) > 0.5:
            self.on_stage("placing", "正在移动至放置位置。")
            self._move_xy(px, py, self.cfg["lift_step_mm"], "placing")

        place_z = max(float(self.cfg["table_z_mm"]), float(self.cfg["grasp_z_mm"]) + 2.0)
        self.on_stage("releasing", "正在缓慢下降并松开夹爪。")
        self._move_z_to(place_z, self.cfg["descend_step_mm"], "releasing")
        self.check_stop()
        open_fb = self.motion.set_gripper(
            int(self.cfg["gripper_open_raw"]),
            expect_object=False,
            speed=self.cfg["gripper_speed"],
            duration_s=self.cfg["gripper_duration_s"],
        )
        self.on_event("gripper_move", stage="releasing", feedback=open_fb)
        if self.cfg["settle_s"] > 0:
            self.sleep(self.cfg["settle_s"])

        self.on_stage("retreating", "物体已放下，正在抬起回到观察高度。")
        self._move_z_to(float(self.cfg["observe_height_mm"]), self.cfg["lift_step_mm"], "retreating")
        frame = read_frame()
        self.on_event("release_completed", gripper=self.motion.gripper_state())
        return frame
