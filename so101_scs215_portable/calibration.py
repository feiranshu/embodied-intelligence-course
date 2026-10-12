"""Interactive read-only sampling of manually demonstrated joint ranges."""

import copy
import json
import time
import threading
from pathlib import Path

from advanced import save_advanced
from joint_mapping import mapped_position, check_position, PositionTrace
from servo_bus import ServoBus, make_packet, request


def trace_endpoint(bus, joint, trace, prompt):
    stop = threading.Event()
    errors = []

    def collect():
        try:
            while not stop.is_set():
                with bus.lock:
                    trace.update(bus.read_position(joint))
                stop.wait(0.04)
        except Exception as error:
            errors.append(error)
            print(f"\n采样中断：{error}。请停止摆动并按 Enter。", flush=True)

    worker = threading.Thread(target=collect, daemon=True)
    worker.start()
    try:
        input(prompt)
    finally:
        stop.set()
        worker.join()
    if errors:
        raise errors[0]
    return trace.raw, trace.position


def calibrate_ranges(config, hardware, path, port=None):
    candidate = copy.deepcopy(config)
    candidate["motion_calibrated"] = False
    candidate["joint_directions"] = {}
    candidate["joint_mapping"] = {}
    progress_path = Path(path).with_suffix(".calibration.json")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for key in ("joint_mapping", "joint_directions", "limits_raw"):
            candidate[key] = progress[key]
        print("恢复已完成关节：", list(candidate["joint_mapping"]), flush=True)
    directions = [("左转", "右转"), ("向后放低", "向前抬起"),
                  ("伸展", "弯曲"), ("向下俯", "向上仰"),
                  ("顺时针", "逆时针"), ("闭合", "张开")]
    print("实机关节范围校准：只读取，不驱动电机、不改写舵机参数。", flush=True)
    print("请托住机械臂，逐关节手动摆动；范围端点应离碰撞、线缆拉紧或硬限位留有余量。", flush=True)
    print("不必寻找机械极限，遇到阻力不要硬掰。Ctrl+C 随时取消，最终确认前不覆盖配置。", flush=True)
    with ServoBus(hardware, port, stop_on_close=False) as bus:
        def sample(joint):
            while True:
                with bus.lock:
                    if request(bus.ser, make_packet(joint, 2, [40, 1]), 7)[5] != 0:
                        raise ValueError(f"ID {joint} 扭矩未释放，请停止其他控制程序后再校准")
                    values = []
                    for _ in range(5):
                        values.append(bus.read_position(joint))
                        time.sleep(0.05)
                offsets = [(v - values[0] + 512) % 1024 - 512 for v in values]
                if max(offsets) - min(offsets) <= 3:
                    return (values[0] + sorted(offsets)[2]) % 1024
                print(f"ID {joint} 读数尚未稳定：{values}；已采端点保留。", flush=True)
                input("请托稳此关节，保持不动，再按 Enter 重试当前采样：")

        input("摆到正常参考姿态即可（不要求几何中点），保持不动，按 Enter 记录未完成关节的零点：")
        origins = [sample(joint) for joint in range(1, 7)]
        for joint, (negative, positive) in enumerate(directions, 1):
            if str(joint) in candidate["joint_mapping"]:
                continue
            label = hardware["servos"][joint - 1]["label"]
            input(f"\nID {joint} {label}：先放到舒适参考姿态，托稳按 Enter，随后开始连续采样：")
            origins[joint - 1] = sample(joint)
            trace = PositionTrace(origins[joint - 1])
            a, first = trace_endpoint(bus, joint, trace,
                f"正在连续采样，请缓慢摆到【{negative}】安全端点，摆稳按 Enter：")
            print(f"已读取第一端点 {a}", flush=True)
            b, second = trace_endpoint(bus, joint, trace,
                f"继续缓慢经过正常活动路径，摆到【{positive}】安全端点，摆稳按 Enter：")
            mapping = {"zero_raw": origins[joint - 1], "sign": 1}
            while abs(first - second) < 12:
                print(f"端点 {a}/{b} 距离不足 12 格；参考零点可位于端点或范围一侧。", flush=True)
                b, second = trace_endpoint(bus, joint, trace,
                    f"请重新缓慢摆到【{positive}】安全端点，摆稳按 Enter：")
            mapping["sign"] = 1 if first < second else -1
            low, high = sorted((first * mapping["sign"], second * mapping["sign"]))
            mapping["limits"] = [low + 3, high - 3]
            mapping["unwrapped"] = True
            mapping["crosses_zero"] = abs(a - b) != high - low
            mapping["wheel_positive"] = None
            candidate["joint_mapping"][str(joint)] = mapping
            if not mapping["crosses_zero"]:
                candidate["limits_raw"][joint - 1] = [min(a, b) + 3, max(a, b) - 3]
            candidate["joint_directions"][str(joint)] = mapping["sign"] * (-1 if joint == 1 else 1)
            print(f"原始端点 {a}/{b}，零点 {mapping['zero_raw']}，映射安全边界 [{low + 3}, {high - 3}]。", flush=True)
            while True:
                input("将此关节放回舒适姿态（可在已测端点），托稳后按 Enter 继续：")
                raw = sample(joint)
                current = mapped_position(raw, mapping)
                if low - 3 <= current <= high + 3:
                    break
                print(f"原始 {raw} / 映射 {current} 超出已测范围及 ±3 格采样容差 [{low - 3}, {high + 3}]；已采端点保留。", flush=True)
            save_advanced(progress_path, {key: candidate[key] for key in
                ("joint_mapping", "joint_directions", "limits_raw")})
            print(f"ID {joint} 进度已保存，中断后可从下一关节继续。", flush=True)
        current = [sample(i) for i in range(1, 7)]
        for i, raw in enumerate(current, 1):
            while True:
                try:
                    check_position(raw, candidate["joint_mapping"][str(i)], tolerance=6)
                    break
                except ValueError as error:
                    print(f"ID {i}：{error}；已完成校准保留。", flush=True)
                    input("请放回该关节的安全范围内部，摆稳按 Enter：")
                    raw = sample(i)
    print("\n待保存映射：", candidate["joint_mapping"], flush=True)
    print("仅完成关节范围/方向采样，末端前后上下移动还需要运动学校准。", flush=True)
    if input("确认这些范围均无碰撞且留有余量，输入 SAVE 保存；其他输入取消：").strip() != "SAVE":
        print("已取消，原配置未修改")
        return
    candidate["motion_calibrated"] = True
    candidate["kinematics"]["calibrated"] = False
    candidate["vision"]["calibrated"] = False
    save_advanced(path, candidate)
    progress_path.unlink(missing_ok=True)
    print("映射范围已保存。实机连续运动还需确认连续模式的电机方向；不会自动改写模式。", flush=True)
