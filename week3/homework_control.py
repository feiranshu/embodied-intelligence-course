"""SO-101 + SCS215 homework workflow: ID, calibration and six-axis motion.

The original controller in this package is intentionally left as a one-motor
continuous-rotation tool.  This file is the assignment-facing position-mode
controller.  It accepts six normalized values in [0, 1] and sends every target
in one SCS sync-write packet at each interpolation step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from servo_bus import POSITION_MODE_MAX, ServoBus, available_ports, load_config


ROOT = Path(__file__).resolve().parent
MIN_CALIBRATION_SPAN = 30
START_POSITION_TOLERANCE = 15


def ordered_servos(config: dict) -> list[dict]:
    servos = sorted(config["servos"], key=lambda item: int(item["id"]))
    ids = [int(item["id"]) for item in servos]
    if ids != [1, 2, 3, 4, 5, 6]:
        raise ValueError(f"作业控制器要求舅机 ID 恰好为 1～6，当前为 {ids}")
    return servos


def calibration_endpoints(item: dict) -> tuple[int, int]:
    calibration = item.get("position_calibration") or {}
    try:
        zero = int(calibration["end_a"])
        one = int(calibration["end_b"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"ID {item.get('id')} 缺少有效的 position_calibration.end_a/end_b"
        ) from error
    if not 0 <= zero <= POSITION_MODE_MAX or not 0 <= one <= POSITION_MODE_MAX:
        raise ValueError(
            f"ID {item['id']} 校准端点必须位于 0～{POSITION_MODE_MAX}"
        )
    if abs(one - zero) < MIN_CALIBRATION_SPAN:
        raise ValueError(
            f"ID {item['id']} 校准跨度过小：{zero} → {one}"
        )
    return zero, one


def normalized_to_raw(item: dict, value: float) -> int:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"ID {item['id']} 的输入 {value!r} 不在 0～1")
    zero, one = calibration_endpoints(item)
    raw = round(zero + value * (one - zero))
    return max(0, min(POSITION_MODE_MAX, raw))


def normalized_targets(config: dict, values: list[float]) -> dict[int, int]:
    servos = ordered_servos(config)
    if len(values) != 6:
        raise ValueError("必须输入六个 0～1 数值")
    return {
        int(item["id"]): normalized_to_raw(item, value)
        for item, value in zip(servos, values, strict=True)
    }


def smoothstep(value: float) -> float:
    """Ease both ends of the host-side trajectory."""
    return value * value * (3.0 - 2.0 * value)


def interpolate_positions(
    start: dict[int, int], target: dict[int, int], step: int, total_steps: int
) -> dict[int, int]:
    if total_steps < 1 or not 1 <= step <= total_steps:
        raise ValueError("插值步数无效")
    ratio = smoothstep(step / total_steps)
    return {
        servo_id: round(start[servo_id] + (target[servo_id] - start[servo_id]) * ratio)
        for servo_id in target
    }


def atomic_write_config(path: Path, config: dict) -> None:
    path = path.resolve()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sample_position(bus: ServoBus, servo_id: int, count: int = 5) -> int:
    readings = []
    for index in range(count):
        readings.append(bus.read_position(servo_id))
        if index + 1 < count:
            time.sleep(0.03)
    return round(statistics.median(readings))


def release_torque(bus: ServoBus, servo_ids: list[int]) -> list[str]:
    failures = []
    for servo_id in servo_ids:
        try:
            bus.set_torque(servo_id, False)
        except Exception as error:  # Best effort safety cleanup.
            failures.append(f"ID {servo_id} 释放扭矩失败：{error}")
    return failures


def print_target_table(config: dict, values: list[float], targets: dict[int, int]) -> None:
    print("ID  关节          归一化目标  原始目标")
    print("--  ------------  ----------  --------")
    for item, value in zip(ordered_servos(config), values, strict=True):
        servo_id = int(item["id"])
        print(
            f"{servo_id:>2}  {item.get('label', item['name']):<12}  "
            f"{value:>10.3f}  {targets[servo_id]:>8}"
        )


def move_arm(
    bus: ServoBus,
    config: dict,
    values: list[float],
    *,
    duration: float,
    rate_hz: float,
) -> dict[int, int]:
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("运动时间必须是非负有限数")
    if not math.isfinite(rate_hz) or not 5 <= rate_hz <= 100:
        raise ValueError("插值频率必须位于 5～100 Hz")

    servos = ordered_servos(config)
    servo_ids = [int(item["id"]) for item in servos]
    targets = normalized_targets(config, values)
    starts = {}

    # Complete all read-only safety checks before enabling any torque.
    for item in servos:
        servo_id = int(item["id"])
        mode, minimum, maximum = bus.mode(servo_id)
        if mode != "position":
            raise RuntimeError(
                f"ID {servo_id} 不是位置模式（min={minimum}, max={maximum}）；"
                "先运行 position-mode"
            )
        current = bus.read_position(servo_id)
        zero, one = calibration_endpoints(item)
        lower, upper = sorted((zero, one))
        if not lower - START_POSITION_TOLERANCE <= current <= upper + START_POSITION_TOLERANCE:
            raise RuntimeError(
                f"ID {servo_id} 当前位置 {current} 超出校准区间 "
                f"[{lower}, {upper}]，拒绝上扭矩"
            )
        starts[servo_id] = current

    cleanup_required = False
    try:
        # Seed each goal with its current position before torque is enabled.  If
        # firmware auto-enables torque on a goal write, this command still asks
        # for zero displacement rather than jumping to a stale target.
        bus.sync_write_positions(starts)
        # Some servo firmware may enable torque as a side effect of a goal
        # write, so cleanup becomes mandatory immediately after this packet.
        cleanup_required = True
        for servo_id in servo_ids:
            bus.set_torque(servo_id, True)

        total_steps = max(1, math.ceil(duration * rate_hz))
        started_at = time.monotonic()
        for step in range(1, total_steps + 1):
            bus.sync_write_positions(
                interpolate_positions(starts, targets, step, total_steps)
            )
            deadline = started_at + duration * step / total_steps
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        # Give the last command a short settling interval before reading back.
        time.sleep(0.2)
        return {servo_id: bus.read_position(servo_id) for servo_id in servo_ids}
    finally:
        if cleanup_required:
            failures = release_torque(bus, servo_ids)
            for failure in failures:
                print(f"警告：{failure}", file=sys.stderr)


def move_one_servo(
    bus: ServoBus,
    item: dict,
    value: float,
    *,
    duration: float,
    rate_hz: float,
) -> int:
    """Move exactly one position-mode servo and always release its torque."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("运动时间必须是非负有限数")
    if not math.isfinite(rate_hz) or not 5 <= rate_hz <= 100:
        raise ValueError("插值频率必须位于 5～100 Hz")

    servo_id = int(item["id"])
    target = normalized_to_raw(item, value)

    # Finish all read-only checks before sending any goal or torque command.
    mode, minimum, maximum = bus.mode(servo_id)
    if mode != "position":
        raise RuntimeError(
            f"ID {servo_id} 不是位置模式（min={minimum}, max={maximum}）"
        )
    readings = [bus.read_position(servo_id) for _ in range(5)]
    if max(readings) - min(readings) > START_POSITION_TOLERANCE:
        raise RuntimeError(
            f"ID {servo_id} 位置反馈不稳定：{readings}；拒绝上扭矩"
        )
    start = round(statistics.median(readings))
    zero, one = calibration_endpoints(item)
    lower, upper = sorted((zero, one))
    if not lower - START_POSITION_TOLERANCE <= start <= upper + START_POSITION_TOLERANCE:
        raise RuntimeError(
            f"ID {servo_id} 当前位置 {start} 超出校准区间 "
            f"[{lower}, {upper}]，拒绝上扭矩"
        )

    cleanup_required = False
    try:
        # A partially transmitted goal packet may still have reached the servo,
        # so cleanup must be armed before the first write attempt.
        cleanup_required = True
        bus.sync_write_positions({servo_id: start})
        bus.set_torque(servo_id, True)

        total_steps = max(1, math.ceil(duration * rate_hz))
        started_at = time.monotonic()
        for step in range(1, total_steps + 1):
            position = interpolate_positions(
                {servo_id: start}, {servo_id: target}, step, total_steps
            )
            bus.sync_write_positions(position)
            deadline = started_at + duration * step / total_steps
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        time.sleep(0.2)
        return bus.read_position(servo_id)
    finally:
        if cleanup_required:
            for failure in release_torque(bus, [servo_id]):
                print(f"警告：{failure}", file=sys.stderr)


def command_ports(_args, _config, _config_path) -> None:
    ports = available_ports()
    if not ports:
        print("未发现串口。")
        return
    for item in ports:
        suffix = ""
        if item["vid"] is not None and item["pid"] is not None:
            suffix = f" VID:PID={item['vid']:04X}:{item['pid']:04X}"
        print(f"{item['device']}: {item['description'] or item['hwid']}{suffix}")


def command_status(args, config, _config_path) -> None:
    with ServoBus(config, args.port, stop_on_close=False) as bus:
        print(f"串口：{bus.port} @ {config['baud_rate']} baud")
        print("ID  关节          位置  模式      校准端点")
        print("--  ------------  ----  --------  -----------")
        for item in ordered_servos(config):
            servo_id = int(item["id"])
            mode, _minimum, _maximum = bus.mode(servo_id)
            position = bus.read_position(servo_id)
            zero, one = calibration_endpoints(item)
            print(
                f"{servo_id:>2}  {item.get('label', item['name']):<12}  "
                f"{position:>4}  {mode:<8}  {zero:>4} → {one:<4}"
            )


def command_set_id(args, config, _config_path) -> None:
    if not args.yes:
        print("设置 ID 时只能连接一只舅机，否则可能造成重复 ID。")
        answer = input(f"请输入 SET {args.old_id}->{args.new_id} 确认：").strip()
        if answer != f"SET {args.old_id}->{args.new_id}":
            raise RuntimeError("已取消 ID 修改")
    with ServoBus(config, args.port, stop_on_close=False) as bus:
        bus.set_servo_id(args.old_id, args.new_id)
        print(f"舅机 ID 已从 {args.old_id} 改为 {args.new_id}，并在新 ID 上通信验证成功。")


def command_position_mode(args, config, _config_path) -> None:
    if not args.yes:
        print("此操作会将 ID 1～6 的 EEPROM 角度限位写为 0/1023。")
        if input("请输入 POSITION 确认：").strip() != "POSITION":
            raise RuntimeError("已取消模式修改")
    with ServoBus(config, args.port, stop_on_close=False) as bus:
        result = bus.set_mode("all", "position")
        print(f"已验证 ID {result['ids']} 均为位置模式。")


def command_calibrate(args, config, config_path: Path) -> None:
    servos = {int(item["id"]): item for item in ordered_servos(config)}
    selected = []
    for servo_id in args.ids:
        if servo_id not in servos:
            raise ValueError(f"无效舅机 ID：{servo_id}")
        if servo_id not in selected:
            selected.append(servo_id)

    print("校准只读取手动摆放的端点，不发送位置运动指令。")
    print("请托住机械臂，确保线缆不受拉扯，不要用力顶到机械死点。")
    with ServoBus(config, args.port, stop_on_close=False) as bus:
        for servo_id in selected:
            item = servos[servo_id]
            bus.set_torque(servo_id, False)
            label = item.get("label", item["name"])
            input(f"\nID {servo_id} {label}：移到希望定义为 0.0 的端点，然后按 Enter……")
            zero = sample_position(bus, servo_id)
            print(f"  读取 0.0 端点：{zero}")
            input(f"ID {servo_id} {label}：移到希望定义为 1.0 的端点，然后按 Enter……")
            one = sample_position(bus, servo_id)
            print(f"  读取 1.0 端点：{one}")
            if abs(one - zero) < MIN_CALIBRATION_SPAN:
                raise RuntimeError(
                    f"ID {servo_id} 两端点只相差 {abs(one - zero)} ticks，"
                    f"小于安全阈值 {MIN_CALIBRATION_SPAN}"
                )
            item["position_calibration"] = {
                "end_a": zero,
                "end_b": one,
                "home": round((zero + one) / 2),
            }

    atomic_write_config(config_path, config)
    print(f"\n校准已原子写入：{config_path}")


def command_move(args, config, _config_path) -> None:
    values = [float(value) for value in args.positions]
    targets = normalized_targets(config, values)
    print_target_table(config, values, targets)
    if args.dry_run:
        print("\n干运行完成：未打开串口，未发送舅机指令。")
        return
    if not args.yes:
        print("\n即将使六个关节同时运动，程序结束时会释放扭矩。")
        if input("确保机械臂周围无人和障碍物，输入 MOVE 继续：").strip() != "MOVE":
            raise RuntimeError("已取消运动")
    with ServoBus(config, args.port, stop_on_close=False) as bus:
        print(f"\n使用串口 {bus.port}，运动时间 {args.duration:.2f} s……")
        actual = move_arm(
            bus,
            config,
            values,
            duration=args.duration,
            rate_hz=args.rate,
        )
    print("运动完成，扭矩已释放。最终位置反馈：")
    print("  " + "  ".join(f"ID {servo_id}={position}" for servo_id, position in actual.items()))


def command_move_one(args, config, _config_path) -> None:
    items = {int(item["id"]): item for item in ordered_servos(config)}
    if args.id not in items:
        raise ValueError(f"无效舵机 ID：{args.id}")
    item = items[args.id]
    target = normalized_to_raw(item, args.position)
    label = item.get("label", item["name"])
    print(
        f"仅测试 ID {args.id} {label}：归一化目标 {args.position:.3f}，"
        f"原始目标 {target}"
    )
    if args.dry_run:
        print("干运行完成：未打开串口，未发送舵机指令。")
        return
    if not args.yes:
        print("只会给这一只舵机发送位置目标；结束后会释放它的扭矩。")
        phrase = f"MOVE {args.id}"
        if input(f"确认周围无障碍且可立即断电，输入 {phrase} 继续：").strip() != phrase:
            raise RuntimeError("已取消运动")
    with ServoBus(config, args.port, stop_on_close=False) as bus:
        print(f"使用串口 {bus.port}，运动时间 {args.duration:.2f} s……")
        actual = move_one_servo(
            bus,
            item,
            args.position,
            duration=args.duration,
            rate_hz=args.rate,
        )
    print(f"运动完成，ID {args.id} 扭矩已释放；最终位置反馈：{actual}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SO-101/SCS215 编号、校准与六关节 0～1 位置控制"
    )
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--port", help="覆盖 config.json 中的串口，例如 COM8")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ports = subparsers.add_parser("ports", help="列出串口，不访问舅机")
    ports.set_defaults(handler=command_ports)

    status = subparsers.add_parser("status", help="只读查看六只舅机的位置和模式")
    status.set_defaults(handler=command_status)

    set_id = subparsers.add_parser("set-id", help="一次只连一只舅机时修改 ID")
    set_id.add_argument("old_id", type=int)
    set_id.add_argument("new_id", type=int)
    set_id.add_argument("--yes", action="store_true", help="跳过交互确认")
    set_id.set_defaults(handler=command_set_id)

    position_mode = subparsers.add_parser(
        "position-mode", help="将 ID 1～6 写入 0～1023 位置模式"
    )
    position_mode.add_argument("--yes", action="store_true", help="跳过交互确认")
    position_mode.set_defaults(handler=command_position_mode)

    calibrate = subparsers.add_parser("calibrate", help="手动记录关节 0.0/1.0 两个端点")
    calibrate.add_argument("ids", nargs="+", type=int, metavar="ID")
    calibrate.set_defaults(handler=command_calibrate)

    move = subparsers.add_parser("move", help="输入六个 0～1 数值并同步运动")
    move.add_argument("positions", nargs=6, type=float, metavar="P")
    move.add_argument("--duration", type=float, default=2.0, help="插值运动时间，默认 2 s")
    move.add_argument("--rate", type=float, default=25.0, help="同步写入频率，默认 25 Hz")
    move.add_argument("--dry-run", action="store_true", help="只计算目标，不打开串口")
    move.add_argument("--yes", action="store_true", help="跳过交互确认")
    move.set_defaults(handler=command_move)

    move_one = subparsers.add_parser("move-one", help="只测试一只舵机的位置运动")
    move_one.add_argument("id", type=int, metavar="ID")
    move_one.add_argument("position", type=float, metavar="P")
    move_one.add_argument("--duration", type=float, default=3.0, help="插值运动时间，默认 3 s")
    move_one.add_argument("--rate", type=float, default=25.0, help="写入频率，默认 25 Hz")
    move_one.add_argument("--dry-run", action="store_true", help="只计算目标，不打开串口")
    move_one.add_argument("--yes", action="store_true", help="跳过交互确认")
    move_one.set_defaults(handler=command_move_one)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        args.handler(args, config, config_path)
        return 0
    except KeyboardInterrupt:
        print("\n用户中断；如果曾开启扭矩，程序已尝试释放。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
