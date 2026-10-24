"""One arm interface for the agent: offline simulation or the real SCS215 arm.

Hardware motion always goes through week4's ``Arm`` (limits, cross-zero
handling, feedback checks) and ``CommandRunner`` (stop / torque-off on error).
The simulator validates against the *same* advanced.json joint mapping and IK,
so a command rejected here would also be rejected on the real arm.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import agent_common  # noqa: F401  (puts week4 on sys.path)
from advanced import Kinematics
from joint_mapping import check_position, mapped_position, raw_position

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_LABELS = ["底座旋转", "肩部抬升", "肘部弯曲", "腕部俯仰", "腕部旋转", "夹爪"]
DIRECTIONS = {"前": (0, 1), "后": (0, -1), "左": (1, 1), "右": (1, -1), "上": (2, 1), "下": (2, -1)}


def crossing_joints(config: dict, current, target) -> list[int]:
    """Return 1-based joints whose shortest mapped path crosses raw 0/1023."""
    result = []
    for index, (before, after) in enumerate(zip(current, target), 1):
        mapping = config["joint_mapping"][str(index)]
        mapped_change = mapped_position(after, mapping) - mapped_position(before, mapping)
        direct_change = (int(after) - int(before)) * mapping.get("sign", 1)
        if mapped_change != direct_change:
            result.append(index)
    return result


def within_sync_release_tolerance(actual, target, tolerance: int = 10) -> bool:
    """Return true as soon as every joint is close enough to release torque."""
    if isinstance(tolerance, bool) or not 1 <= int(tolerance) <= 16:
        raise ValueError("同步姿态到位释放容差必须为 1～16 格")
    actual_array = np.asarray(actual, dtype=int)
    target_array = np.asarray(target, dtype=int)
    if actual_array.shape != (6,) or target_array.shape != (6,):
        raise ValueError("同步姿态反馈和目标都必须包含六个关节")
    return bool(np.all(np.abs(actual_array - target_array) <= int(tolerance)))


class JointSpace:
    """Convert raw 0–1023 readings to signed degrees in the calibrated joint mapping."""

    def __init__(self, config: dict):
        mapping = config.get("joint_mapping", {})
        if not all(str(i) in mapping and "limits" in mapping[str(i)] for i in range(1, 7)):
            raise ValueError("advanced.json 缺少六个关节的 joint_mapping，不能做关节角控制")
        self.mappings = [mapping[str(i)] for i in range(1, 7)]
        self.deg_per_raw = float(config["kinematics"]["degrees_per_raw"])

    @property
    def limits_deg(self) -> list[list[float]]:
        return [[round(low * self.deg_per_raw, 1), round(high * self.deg_per_raw, 1)]
                for low, high in (m["limits"] for m in self.mappings)]

    def to_degrees(self, raw) -> list[float]:
        return [round(mapped_position(int(r), m) * self.deg_per_raw, 1) for r, m in zip(raw, self.mappings)]

    def to_raw(self, degrees, current_raw) -> tuple[list[int], list[str]]:
        """Return raw targets plus notes for every value clamped to the joint limits.

        ``degrees`` may be a list of six numbers or a dict keyed by joint name;
        missing joints keep their current position.
        """
        if isinstance(degrees, dict):
            unknown = set(degrees) - set(JOINT_NAMES)
            if unknown:
                raise ValueError(f"未知关节：{sorted(unknown)}；可用 {JOINT_NAMES}")
            values = [degrees.get(name) for name in JOINT_NAMES]
        else:
            values = list(degrees)
            if len(values) != 6:
                raise ValueError("关节角必须是 6 个数")
        target, notes = [], []
        for i, (value, current, mapping) in enumerate(zip(values, current_raw, self.mappings)):
            if value is None:
                target.append(int(current))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{JOINT_NAMES[i]} 的角度不是有效数字：{value!r}")
            position = round(float(value) / self.deg_per_raw)
            low, high = mapping["limits"]
            clipped = int(np.clip(position, low, high))
            # Recorded poses may sit a few ticks outside the mapping (week4 accepts 6); clamp them silently.
            if abs(clipped - position) > 6:
                notes.append(f"{JOINT_NAMES[i]} 请求 {value}° 超出 [{low * self.deg_per_raw:.1f}, "
                             f"{high * self.deg_per_raw:.1f}]°，已限制为 {clipped * self.deg_per_raw:.1f}°")
            target.append(raw_position(clipped, mapping))
        return target, notes


def pose_is_valid(config: dict, name: str, tolerance: int = 6) -> bool:
    """Return whether a recorded pose still fits the current joint mappings."""
    raw = config.get("poses", {}).get(name)
    mappings = config.get("joint_mapping", {})
    if not isinstance(raw, list) or len(raw) != 6 or not all(str(i) in mappings for i in range(1, 7)):
        return False
    try:
        for joint, value in enumerate(raw, 1):
            if isinstance(value, bool) or not 0 <= int(value) <= 1023:
                return False
            check_position(int(value), mappings[str(joint)], tolerance=tolerance)
    except (TypeError, ValueError):
        return False
    return True


def available_poses(config: dict) -> list[str]:
    """Only expose recorded poses that fit the currently confirmed ranges."""
    return [name for name in config.get("poses", {}) if pose_is_valid(config, name)]


def action_step_is_valid(config: dict, step: dict, poses: set[str]) -> bool:
    """Accept a recorded pose step or one absolute raw target for a single joint."""
    if not isinstance(step, dict):
        return False
    if "pose" in step:
        return step.get("pose") in poses
    joint = step.get("joint_id")
    position = step.get("position")
    if isinstance(joint, bool) or isinstance(position, bool):
        return False
    if not isinstance(joint, int) or not isinstance(position, int) or not 1 <= joint <= 6:
        return False
    try:
        check_position(position, config["joint_mapping"][str(joint)], tolerance=6)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def available_actions(config: dict) -> list[str]:
    """Only actions whose every pose or single-joint target is currently valid can run."""
    poses = set(available_poses(config))
    return [name for name, steps in config.get("actions", {}).items()
            if steps and all(action_step_is_valid(config, step, poses) for step in steps)]


def plan_safe_action(config: dict, name: str, current) -> tuple[list[list[int]], list[str]]:
    """Preflight every waypoint before an action starts.

    ID5 is allowed to keep its preceding value when only its recorded wrist-roll
    target would cross raw zero.  Any other crossing rejects the whole action
    before the first waypoint moves.
    """
    if name not in available_actions(config):
        raise ValueError(f"动作“{name}”含未记录或越界的姿态，不能执行")
    previous = [int(value) for value in current]
    targets, notes = [], []
    for number, step in enumerate(config["actions"][name], 1):
        if "pose" in step:
            step_name = str(step["pose"])
            target = [int(value) for value in config["poses"][step_name]]
        else:
            joint = int(step["joint_id"])
            position = int(step["position"])
            step_name = f"ID{joint}={position}"
            target = previous.copy()
            target[joint - 1] = position
        crossings = crossing_joints(config, previous, target)
        if 5 in crossings:
            target[4] = previous[4]
            notes.append(f"动作“{name}”第 {number} 步“{step_name}”保持 ID5={previous[4]}，避免跨零")
            crossings = crossing_joints(config, previous, target)
        if crossings:
            raise ValueError(
                f"动作“{name}”第 {number} 步“{step_name}”需要关节 {crossings} 跨零；整段动作未启动"
            )
        targets.append(target)
        previous = target
    return targets, notes


def linear_delta(direction: str, distance_mm: float) -> list[float]:
    if direction not in DIRECTIONS:
        raise ValueError(f"方向必须是 {'/'.join(DIRECTIONS)} 之一")
    axis, sign = DIRECTIONS[direction]
    delta = [0.0, 0.0, 0.0]
    delta[axis] = sign * float(distance_mm)
    return delta


class SimArm:
    """Offline stand-in with the same validation as the real arm; never opens a port."""

    simulated = True

    def __init__(self, config: dict):
        self.config = config
        self.model = Kinematics(config)
        self.joints = JointSpace(config)
        poses = available_poses(config)
        if not poses:
            raise ValueError("advanced.json 中没有符合当前关节范围的预设姿态")
        initial = "初始状态" if "初始状态" in poses else poses[0]
        self.raw = list(config["poses"][initial])
        self.cancel = threading.Event()
        self.last_move = None
        self.arm = self

    def positions(self):
        return np.array(self.raw, dtype=int)

    def cartesian_start_pose(self):
        if self.config.get("kinematics", {}).get("calibrated") is not True:
            raise ValueError("ID4 物理方向已调整，旧 IK/笛卡尔标定已禁用；重新标定后才能做末端直线运动")
        current = self.positions()
        try:
            self.model.inverse(self.model.forward(current), current, allow_crossing=False)
        except ValueError as error:
            raise ValueError(f"{error}；起点不在末端工作范围，先回到“初始状态”或“居中”") from error
        return current

    def move_linear(self, delta):
        delta = np.asarray(delta, dtype=float)
        limit = float(self.config.get("max_distance_mm", self.config["max_step_mm"]))
        if delta.shape != (3,) or not np.all(np.isfinite(delta)) or not 0 < np.linalg.norm(delta) <= limit:
            raise ValueError("移动距离必须大于零且不超过 max_distance_mm")
        current = self.cartesian_start_pose()
        start = self.model.forward(current)
        target = self.model.inverse(start + delta, current, allow_crossing=False)
        self.raw = [int(v) for v in target]
        actual = self.model.forward(self.positions()) - start
        self.last_move = {"requested_mm": delta.round(2).tolist(), "actual_mm": actual.round(2).tolist(),
                          "projected": False}

    def execute(self, command):
        kind, value = command
        if self.cancel.is_set():
            raise InterruptedError("动作已停止")
        if kind in {"pose", "home"}:
            if value not in available_poses(self.config):
                raise ValueError(f"姿态“{value}”不符合当前关节范围；可用：{available_poses(self.config)}")
            self.raw = list(self.config["poses"][value])
        elif kind == "action":
            targets, _notes = plan_safe_action(self.config, value, self.raw)
            self.raw = list(targets[-1])
        elif kind == "joints":
            raw, _duration = value
            self.joints.to_raw(self.joints.to_degrees(raw), self.raw)  # re-checks limits
            self.raw = [int(v) for v in raw]
        elif kind == "move":
            self.move_linear(value)
        else:
            raise ValueError(f"模拟器不支持命令 {kind}")

    def run(self, command):
        self.cancel.clear()
        self.execute(command)

    def stop(self):
        self.cancel.set()

    def close(self):
        self.stop()


class HardwareArm:
    """Real arm through week4 Arm + CommandRunner; one command at a time, blocking."""

    simulated = False

    def __init__(self, config: dict, port: str | None = None, hardware_config=agent_common.HARDWARE_CONFIG):
        from advanced import Arm, CommandRunner
        from servo_bus import ServoBus, load_config

        class AgentArm(Arm):
            def move_pose_synchronized(self, target, duration_s=None, speed=None):
                """Move a no-cross pose like move_servo_raw, using one sync packet."""
                target = self.validate_pose(target)
                current = self.validate_pose(self.positions(), tolerance=12)
                crossings = crossing_joints(self.config, current, target)
                if crossings:
                    raise ValueError(f"同步姿态含跨零关节 {crossings}，拒绝同步执行")
                duration_s = float(self.config.get("sync_duration_s", self.config["duration_s"])
                                   if duration_s is None else duration_s)
                speed = int(self.config.get("sync_speed", self.config["speed"])
                            if speed is None else speed)
                release_tolerance = int(self.config.get("sync_release_tolerance", 10))
                if not 0.1 <= duration_s <= 10 or not 1 <= speed <= 1000:
                    raise ValueError("同步动作时间或速度无效")
                if not 1 <= release_tolerance <= 16:
                    raise ValueError("同步姿态到位释放容差必须为 1～16 格")
                mappings = self.config["joint_mapping"]
                moving = {i + 1: int(raw) for i, raw in enumerate(target)
                          if abs(int(raw) - int(current[i])) > 3}
                if not moving:
                    self.last_sync_feedback = current.copy()
                    return
                for joint, raw in moving.items():
                    mode, low, high = self.bus.mode(joint)
                    if mode != "position" or not low <= raw <= high:
                        raise ValueError(f"ID {joint} 的目标或模式不符合硬件限位")
                origin_mapped = np.asarray([
                    mapped_position(raw, mappings[str(i + 1)]) for i, raw in enumerate(current)
                ], dtype=int)
                target_mapped = np.asarray([
                    mapped_position(raw, mappings[str(i + 1)]) for i, raw in enumerate(target)
                ], dtype=int)
                target_array = np.asarray(target, dtype=int)
                self.last_sync_trace = []
                try:
                    started_at = time.monotonic()
                    self.bus.move_positions(
                        moving, round(duration_s * 1000), speed, cancel=self.cancel
                    )
                    deadline = started_at + max(2.0, duration_s + 1.5)
                    hard_deadline = started_at + max(10.0, duration_s + 1.5)
                    best = np.abs(target_array - current)
                    while not self.cancel.wait(0.04):
                        with self.lock:
                            self.bus.heartbeat("all")
                            actual = self.validate_pose(self.positions(), tolerance=12)
                        actual_mapped = np.asarray([
                            mapped_position(raw, mappings[str(i + 1)]) for i, raw in enumerate(actual)
                        ], dtype=int)
                        self.last_sync_trace.append({
                            "t": round(time.monotonic() - started_at, 3),
                            "actual": [int(v) for v in actual],
                        })
                        escaped = np.flatnonzero(
                            (actual_mapped < np.minimum(origin_mapped, target_mapped) - 16)
                            | (actual_mapped > np.maximum(origin_mapped, target_mapped) + 16)
                        )
                        if len(escaped):
                            joint = int(escaped[0]) + 1
                            raise RuntimeError(
                                f"ID {joint} 同步反馈偏离本次运动范围："
                                f"起点 {int(current[joint - 1])}，目标 {int(target[joint - 1])}，"
                                f"反馈 {int(actual[joint - 1])}"
                            )
                        errors = np.abs(actual - target_array)
                        # Release on the first acceptable sample.  Waiting for
                        # several consecutive samples keeps torque enabled and
                        # can make gravity-loaded joints hunt around the target.
                        if within_sync_release_tolerance(actual, target_array, release_tolerance):
                            self.last_sync_feedback = actual.copy()
                            return
                        now = time.monotonic()
                        if np.any(errors <= best - 2):
                            best = np.minimum(best, errors)
                            deadline = min(hard_deadline, max(deadline, now + 2.0))
                        if now >= deadline:
                            raise TimeoutError(f"同步姿态未在期限内到位：目标 {target.tolist()}，反馈 {actual.tolist()}")
                finally:
                    self.ready = False
                    failures = self.bus.stop("all")
                    if failures:
                        raise RuntimeError("；".join(failures))

            def execute(self, command):
                if command[0] == "action":
                    self.last_sync_feedback = None
                    name = command[1]
                    steps = self.config.get("actions", {}).get(name, [])
                    targets, notes = plan_safe_action(self.config, name, self.positions())
                    for step in steps:
                        duration = float(step.get("duration_s", self.config["duration_s"]))
                        pause = float(step.get("pause_s", 0))
                        speed = int(step.get("speed", self.config["speed"]))
                        if not 0.1 <= duration <= 60 or not 0 <= pause <= 60 or not 1 <= speed <= 1000:
                            raise ValueError("动作的时间、暂停或速度无效；整段动作未启动")
                    for note in notes:
                        print(f"安全调整：{note}", flush=True)
                    for step, target in zip(steps, targets):
                        actual = self.validate_pose(self.positions(), tolerance=12)
                        crossings = crossing_joints(self.config, actual, target)
                        if crossings:
                            raise ValueError(
                                f"实时位置漂移后，前往“{step['pose']}”需要关节 {crossings} 跨零；停止剩余动作"
                            )
                        self.move_pose_synchronized(
                            target, step.get("duration_s"), step.get("speed")
                        )
                        if self.cancel.wait(float(step.get("pause_s", 0))):
                            raise InterruptedError("动作已停止")
                    return
                if command[0] == "joints":
                    # Never let a later sequential/failing command reuse an earlier
                    # synchronized command's final feedback in the run log.
                    self.last_sync_feedback = None
                    raw, duration = command[1]
                    if crossing_joints(self.config, self.positions(), raw):
                        self.move_pose(raw, duration_s=duration, quiet=True, joint_order=(4, 2, 3, 1, 5, 6),
                                       skip_arrived_base=True)
                    else:
                        self.move_pose_synchronized(raw, duration)
                    return
                super().execute(command)

        hardware = load_config(hardware_config)
        if [int(s["id"]) for s in hardware["servos"]] != list(range(1, 7)):
            raise ValueError("需要 ID 1～6 依次为底座、肩、肘、腕俯仰、腕旋转、夹爪")
        self.bus = ServoBus(hardware, port, stop_on_close=False)
        try:
            self.arm = AgentArm(config, self.bus)
        except Exception:
            self.bus.close()
            raise
        self.config = config
        self.joints = JointSpace(config)
        self.last_feedback = None
        self.runner = CommandRunner(self.arm)

    @property
    def model(self):
        return self.arm.model

    def positions(self):
        return self.arm.positions()

    def run(self, command):
        self.last_feedback = None
        if not self.runner.submit(command):
            raise RuntimeError("机械臂正在执行其他动作，本条指令未执行")
        self.runner.wait()
        feedback = getattr(self.arm, "last_sync_feedback", None)
        self.last_feedback = None if feedback is None else [int(v) for v in feedback]

    def stop(self):
        self.runner.stop()

    def close(self):
        try:
            self.runner.close()
        finally:
            self.bus.close()


def open_arm(config: dict, *, execute: bool, port: str | None = None):
    return HardwareArm(config, port) if execute else SimArm(config)


def arm_state(arm) -> dict:
    feedback = getattr(arm, "last_feedback", None)
    raw = [int(v) for v in (feedback if feedback is not None else arm.positions())]
    if feedback is not None:
        arm.last_feedback = None
    state = {"simulated": arm.simulated, "raw": raw,
             "joints_deg": dict(zip(JOINT_NAMES, arm.joints.to_degrees(raw)))}
    if arm.config.get("kinematics", {}).get("calibrated") is not True:
        state.update(end_effector_mm=None, cartesian_ready=False,
                     cartesian_reason="ID4 调整后尚未重新标定 IK/笛卡尔模型")
        return state
    try:
        state["end_effector_mm"] = [round(float(v), 1) for v in arm.model.forward(raw)]
        arm.model.inverse(arm.model.forward(raw), np.asarray(raw), allow_crossing=False)
        state["cartesian_ready"] = True
    except ValueError:
        state["cartesian_ready"] = False
    return state
