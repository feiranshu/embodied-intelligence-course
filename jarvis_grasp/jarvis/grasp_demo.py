"""Standalone CLI demonstration for object recognition, clarification, grasping, and release."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .controller import Assistant
from .perception import ColorShapeDetector, LocalSelector
from .simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]


def run_demo(
    command: str = "抓取蓝色盒子",
    *,
    clarify_answer: str | None = None,
    release_after: bool = True,
    scenario: str = "default",
    output_dir: Path | None = None,
) -> dict:
    config = json.loads((ROOT / "config" / "default.json").read_text(encoding="utf-8-sig"))
    run_dir = output_dir or (ROOT / "runs" / ("grasp_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    sim = Simulation(scenario=scenario)
    detector = ColorShapeDetector()
    selector = LocalSelector()
    app = Assistant(sim, sim, detector, selector, config, run_dir, mode="simulation")
    try:
        print(f"[1/4] 识别桌面物体并执行指令：{command}")
        state = app.run(command)
        print(f"      状态：{state['status']} | 回复：{state['reply']}")
        if state["candidates"]:
            names = ", ".join(f"{c['id']}({c['label']}, {c['angle_deg']}°)" for c in state["candidates"])
            print(f"      候选物体：{names}")
        if state["status"] == "clarifying" and clarify_answer:
            print(f"[2/4] 回答澄清问题：{clarify_answer}")
            state = app.run(clarify_answer)
            print(f"      状态：{state['status']} | 回复：{state['reply']}")
        print(f"[3/4] 夹爪状态：{state['gripper']}")
        if release_after and state.get("held_object") is not None:
            print("[4/4] 执行放下动作：放下")
            state = app.run("放下")
            print(f"      状态：{state['status']} | 回复：{state['reply']}")
            print(f"      最终夹爪状态：{state['gripper']}")
        print(f"运行日志与前后图像已保存至：{run_dir}")
        return state
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description="一键运行桌面物体识别与抓取演示（默认软件模拟）")
    parser.add_argument("--command", default="抓取蓝色盒子", help="识别或抓取指令")
    parser.add_argument("--answer", default=None, help="当目标存在歧义时的澄清回答，例如“圆的”")
    parser.add_argument("--scenario", choices=Simulation.names, default="default")
    parser.add_argument("--keep-holding", action="store_true", help="抓起后保持夹持，不自动执行放下")
    parser.add_argument("--output", type=Path, default=None, help="运行记录与截图保存目录")
    args = parser.parse_args()
    run_demo(
        args.command,
        clarify_answer=args.answer,
        release_after=not args.keep_holding,
        scenario=args.scenario,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
