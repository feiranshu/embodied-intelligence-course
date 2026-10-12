"""Read-only hardware and configuration diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

from servo_bus import ServoBus, available_ports, load_config, resolve_port


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="只读检查 SCS215 控制器和六个舵机")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--port", help="覆盖串口，例如 COM8")
    args = parser.parse_args()

    config = load_config(args.config)
    print("检测到的串口：")
    ports = available_ports()
    if not ports:
        print("  （无）")
    for item in ports:
        vid_pid = ""
        if item["vid"] is not None and item["pid"] is not None:
            vid_pid = f" VID:PID={item['vid']:04X}:{item['pid']:04X}"
        print(f"  {item['device']}: {item['description']}{vid_pid}")

    selected, _ = resolve_port(args.port or config["port"])
    print(f"\n选择串口：{selected}")
    print("以下操作只读取寄存器，不会让舵机运动或改写 EEPROM。")
    with ServoBus(config, args.port, stop_on_close=False) as bus:
        states = bus.state()
        print("\nID  关节          位置  模式       最小  最大  电压(V)")
        print("--  ------------  ----  ---------  ----  ----  -------")
        for item in states:
            print(
                f"{item['id']:>2}  {item['label']:<12}  {item['position']:>4}  "
                f"{item['mode']:<9}  {item['min_limit']:>4}  {item['max_limit']:>4}  {bus._read_u8(item['id'], 62) / 10:>5.1f}"
            )
    print("\n检查完成：六只舵机均可通信，串口已释放。")


if __name__ == "__main__":
    main()
