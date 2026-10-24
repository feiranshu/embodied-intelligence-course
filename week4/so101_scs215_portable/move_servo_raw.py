"""Move one configured SCS215 servo to an absolute raw position."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def ranged_int(label: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{label} 必须是整数") from error
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"{label} 必须位于 {minimum}～{maximum}"
            )
        return number

    return parse


def load_config(path: str | Path) -> dict:
    """Load enough hardware configuration for preview without importing pyserial."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = {"port", "baud_rate", "servos"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"配置缺少字段：{', '.join(sorted(missing))}")
    ids = [int(item["id"]) for item in config["servos"]]
    if len(ids) != len(set(ids)):
        raise ValueError("配置中存在重复舵机 ID")
    return config


def get_servo_bus_class():
    """Import hardware dependencies only when --execute is requested."""
    try:
        from servo_bus import ServoBus
    except ModuleNotFoundError as error:
        if error.name == "serial":
            raise RuntimeError(
                "实机执行需要 pyserial；请先运行 python -m pip install -r requirements.txt"
            ) from error
        raise

    return ServoBus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将一只 SCS215 舵机移动到 0～1023 的绝对原始位置"
    )
    parser.add_argument("positional_id", nargs="?", type=ranged_int("ID", 1, 253),
                        metavar="id", help="舵机 ID，也可使用 --id")
    parser.add_argument(
        "positional_position",
        nargs="?",
        type=ranged_int("原始位置", 0, 1023),
        metavar="position",
        help="绝对原始位置（0～1023），也可使用 --position",
    )
    parser.add_argument("--id", dest="option_id", type=ranged_int("ID", 1, 253))
    parser.add_argument("--position", dest="option_position",
                        type=ranged_int("原始位置", 0, 1023))
    parser.add_argument("--duration-ms", "--duration", dest="duration_ms",
                        type=ranged_int("运动时间", 1, 65535), default=1000,
                        help="运动时间，单位毫秒（默认 1000）")
    parser.add_argument("--speed", type=ranged_int("速度", 1, 1000), default=100)
    parser.add_argument("--tolerance", type=ranged_int("到位容差", 0, 100), default=5)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--port", help="覆盖配置中的串口，例如 COM8")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="实际连接并驱动舵机",
    )
    mode.add_argument("--dry-run", action="store_true",
                      help="只显示计划，不导入串口模块或连接硬件（默认行为）")
    return parser


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, label in (("id", "ID"), ("position", "原始位置")):
        positional = getattr(args, f"positional_{name}")
        option = getattr(args, f"option_{name}")
        if positional is None and option is None:
            parser.error(f"必须提供{label}：使用位置参数或 --{name}")
        if positional is not None and option is not None:
            parser.error(f"{label}不能同时使用位置参数和 --{name} 重复提供")
        setattr(args, name, option if option is not None else positional)
    return args


def move_and_wait(bus: Any, servo_id: int, position: int, duration_ms: int,
                  speed: int, tolerance: int) -> tuple[int, int]:
    before = bus.read_position(servo_id)
    mode, minimum, maximum = bus.mode(servo_id)
    print(
        f"ID {servo_id}: 当前={before}，模式={mode}，"
        f"硬件限位={minimum}～{maximum}，目标={position}"
    )
    if mode != "position":
        raise ValueError(f"ID {servo_id} 当前不是位置模式；本工具不会修改 EEPROM")
    if not minimum <= position <= maximum:
        raise ValueError(
            f"目标 {position} 超出 ID {servo_id} 的硬件限位 {minimum}～{maximum}"
        )

    bus.move_position(servo_id, position, duration_ms, speed)
    deadline = time.monotonic() + max(2.0, duration_ms / 1000 + 1.5)
    stable_samples = 0
    actual = before
    while time.monotonic() < deadline:
        time.sleep(0.05)
        bus.heartbeat(servo_id)
        actual = bus.read_position(servo_id)
        if abs(actual - position) <= tolerance:
            stable_samples += 1
            if stable_samples >= 3:
                return before, actual
        else:
            stable_samples = 0
    raise RuntimeError(
        f"ID {servo_id} 未在时限内到位：目标={position}，最后反馈={actual}"
    )


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    configured_ids = {int(item["id"]) for item in config["servos"]}
    if args.id not in configured_ids:
        raise ValueError(
            f"ID {args.id} 不在配置中；可用 ID："
            + ", ".join(str(item) for item in sorted(configured_ids))
        )

    plan = (
        f"计划：ID {args.id} -> 原始位置 {args.position}，"
        f"时间 {args.duration_ms} ms，速度 {args.speed}，到位容差 ±{args.tolerance}"
    )
    if not args.execute:
        print(plan)
        print("当前为预览，不会连接串口或驱动舵机；确认安全后加 --execute。")
        return 0

    print(plan)
    print("即将连接实机；运动期间按 Ctrl+C 可请求停止。")
    servo_bus_class = get_servo_bus_class()
    with servo_bus_class(config, args.port, stop_on_close=False) as bus:
        try:
            before, actual = move_and_wait(
                bus,
                args.id,
                args.position,
                args.duration_ms,
                args.speed,
                args.tolerance,
            )
        finally:
            failures = bus.stop(args.id)
            if failures:
                raise RuntimeError("停止或释放驱动未完全确认：" + "；".join(failures))
    print(f"完成：ID {args.id}，{before} -> {actual}；该舵机驱动已释放。")
    return 0


def main() -> int:
    args = parse_arguments()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n已收到中断；已请求停止并释放驱动。")
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"错误：{error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
