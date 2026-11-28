"""Read-only hardware calibration helper for measuring SCS215 tabletop grasp geometry.

This tool never writes servo goals or enables torque. Move the unpowered arm manually
to each reference pose (observe, pre-grasp, tabletop grasp, open gripper, closed gripper,
holding target object) and record the URDF end-effector XYZ and gripper raw ticks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_arm_snapshot(port: str | None = None) -> dict:
    from vendor.scs215.advanced import Kinematics, load_advanced
    from vendor.scs215.servo_bus import ServoBus, load_config

    scs_dir = ROOT / "vendor" / "scs215"
    advanced = load_advanced(scs_dir / "advanced.json")
    hardware = load_config(scs_dir / "config.json")
    model = Kinematics(advanced)
    with ServoBus(hardware, port, stop_on_close=False) as bus:
        raw = [int(bus.read_position(i)) for i in range(1, 7)]
    xyz = [round(float(v), 2) for v in model.forward(raw)]
    return {"raw": raw, "end_effector_mm": xyz, "gripper_raw": raw[5]}


def main():
    parser = argparse.ArgumentParser(description="SCS215 实机抓取标定只读采样工具（不开启舵机扭矩）")
    parser.add_argument("--port", default=None, help="串口名称，例如 COM3；默认按 config.json 自动查找")
    parser.add_argument("--once", action="store_true", help="仅读取一次当前关节与末端坐标后退出")
    args = parser.parse_args()

    if args.once:
        snap = read_arm_snapshot(args.port)
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return

    print("只读标定助手：不会写入舵机或开启扭矩。请断电或关闭扭矩后手动摆放到各参考位，按回车采样。")
    samples = {}
    steps = [
        ("observe", "1. 摆放到【上方观察姿态】（能看清整个桌面），按回车采样"),
        ("pre_grasp", "2. 摆放到【预抓取高度】（物体正上方约 4～6 cm），按回车采样"),
        ("grasp", "3. 摆放到【抓取接触高度】（夹爪指尖贴近物体中部、高于桌面），按回车采样"),
        ("table", "4. 摆放到【桌面安全下界】（夹爪刚接触桌面但不压桌），按回车采样"),
        ("gripper_open", "5. 手动张开夹爪到【抓取前张开宽度】，按回车采样"),
        ("gripper_held", "6. 手动让夹爪夹住【典型目标物体】，按回车采样"),
        ("gripper_close", "7. 手动完全闭合空夹爪到【空抓闭合位】，按回车采样"),
    ]
    for key, prompt in steps:
        input(prompt + "：")
        samples[key] = read_arm_snapshot(args.port)
        print("   已采样：", samples[key])

    table_z = samples["table"]["end_effector_mm"][2]
    grasp_z = max(table_z + 2.0, samples["grasp"]["end_effector_mm"][2])
    pre_z = max(grasp_z + 10.0, samples["pre_grasp"]["end_effector_mm"][2])
    lift_z = max(pre_z + 10.0, samples["observe"]["end_effector_mm"][2])
    open_raw = samples["gripper_open"]["gripper_raw"]
    held_raw = samples["gripper_held"]["gripper_raw"]
    close_raw = samples["gripper_close"]["gripper_raw"]

    suggestion = {
        "grasp": {
            "verified": False,
            "table_z_mm": round(table_z, 1),
            "grasp_z_mm": round(grasp_z, 1),
            "pre_grasp_z_mm": round(pre_z, 1),
            "lift_z_mm": round(lift_z, 1),
            "camera_to_gripper_mm": [0.0, 0.0, 0.0],
            "gripper_open_raw": open_raw,
            "gripper_close_raw": close_raw,
            "gripper_held_range_raw": [
                max(close_raw + 10, held_raw - 25),
                min(open_raw - 10, held_raw + 25),
            ],
        }
    }
    print("\n建议写入实机配置（核对 camera_to_gripper_mm 与安全范围后再将 verified 设为 true）：")
    print(json.dumps(suggestion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
