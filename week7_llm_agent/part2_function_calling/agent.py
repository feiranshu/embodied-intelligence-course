"""Week 7 task 4.2: earlier robot skills as tools, called by an external LLM API.

Typed Chinese instructions go to the LLM together with the tool schemas from
robot_tools.py; every tool call and result is printed so the whole
"LLM → tool_call → robot → tool result → LLM" flow is visible, and saved to
runs/agent_*/ (tool_calls.json, transcript.json).

Without --execute the arm is simulated and no serial port is opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import agent_common  # noqa: E402
from agent_common import read_json, run_dir, write_json  # noqa: E402
from agent_loop import run_turn, text_protocol_instructions  # noqa: E402
from arm_backend import open_arm  # noqa: E402
from llm_client import add_llm_arguments, llm_from_args, make_llm  # noqa: E402
from robot_tools import RobotToolkit  # noqa: E402

SKILL = HERE / "skills" / "robot-arm-tools" / "SKILL.md"


def system_prompt(*, skill: bool, extra: str = "") -> str:
    text = (HERE / "system_prompt.md").read_text(encoding="utf-8")
    if skill:
        text += "\n\n## 已加载的 Skill\n" + SKILL.read_text(encoding="utf-8")
    return text + ("\n\n" + extra if extra else "")


def print_event(kind: str, data: dict):
    if kind == "call":
        print(f"  → 调用工具 {data['name']}({data['arguments']})", flush=True)
    else:
        result = json.dumps(data["result"], ensure_ascii=False)
        print(f"  ← 结果 {result[:400]}{'…' if len(result) > 400 else ''}", flush=True)


def add_robot_arguments(parser):
    group = parser.add_argument_group("机械臂与相机")
    group.add_argument("--execute", action="store_true", help="驱动真实机械臂；缺省为模拟")
    group.add_argument("--port", help="机械臂串口，缺省自动识别")
    group.add_argument("--yes", action="store_true", help="跳过实机运行前的 RUN 确认")
    group.add_argument("--read-only", action="store_true", help="只开放读取/观察工具，代码层隐藏全部运动工具")
    group.add_argument("--camera", help="腕部相机设备名、索引或照片路径；缺省不接相机（视觉类工具隐藏）")
    group.add_argument("--person-camera", help="拍人用于 imitate_person 的相机；缺省与 --camera 相同")
    group.add_argument("--vision-provider", help="look/detect_object 使用的视觉模型平台，缺省与 --provider 相同")
    group.add_argument("--vision-model", help="视觉模型名，缺省用平台预设")
    group.add_argument("--track-step-mm", type=float, default=5.0, help="视觉追踪每步移动距离，1～10 mm")
    return group


def build_toolkit(args, llm, log_dir):
    """Open arm, optional cameras and the vision model; caller closes the toolkit."""
    from camera_hub import open_camera_source
    from common import calibration

    if not 1 <= args.track_step_mm <= 10:
        raise SystemExit("--track-step-mm 需在 1～10 之间")
    config = read_json(agent_common.ADVANCED_CONFIG)
    if args.execute and not args.yes:
        if input("实机模式：Agent 会自主驱动机械臂。确认周围空旷、可随时断电，输入 RUN 继续：").strip() != "RUN":
            raise SystemExit("未启动")
    camera = person_camera = vision = None
    size = calibration(agent_common.INTRINSICS)[2]
    if args.camera is not None:
        camera = open_camera_source(args.camera, size)
    if args.person_camera is not None:
        person_camera = open_camera_source(args.person_camera)
    if camera is not None or person_camera is not None:
        provider = args.vision_provider or args.provider
        same = provider == args.provider and args.vision_model in (None, llm.model)
        vision = llm if same else make_llm(provider, model=args.vision_model, purpose="vision",
                                            thinking=args.thinking)
    arm = open_arm(config, execute=args.execute, port=args.port)
    return RobotToolkit(arm, camera, vision, config=config, person_camera=person_camera,
                        track_step_mm=args.track_step_mm, log_dir=log_dir,
                        read_only=args.read_only), [camera, person_camera]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", help="只执行这一句指令后退出（便于录屏演示）")
    parser.add_argument("--skill", action="store_true", help="把 skills/robot-arm-tools/SKILL.md 加入系统提示词")
    parser.add_argument("--protocol", choices=["native", "text"], default="native",
                        help="native=OpenAI function calling；text=模型不支持工具调用时的文本协议")
    parser.add_argument("--max-rounds", type=int, default=8, help="每句指令最多的工具调用轮数")
    add_llm_arguments(parser, purpose="agent")
    add_robot_arguments(parser)
    args = parser.parse_args()

    llm = llm_from_args(args)
    log_dir = run_dir("agent")
    toolkit, cameras = build_toolkit(args, llm, log_dir)
    extra = text_protocol_instructions(toolkit.schemas()) if args.protocol == "text" else ""
    messages = [{"role": "system", "content": system_prompt(skill=args.skill, extra=extra)}]
    print(f"模型 {llm.provider}/{llm.model}；{'实机' if args.execute else '模拟'}；"
          f"工具 {[t['function']['name'] for t in toolkit.schemas()]}\n日志目录 {log_dir}")
    try:
        while True:
            text = args.once or input("\n你：").strip()
            if text.lower() in {"q", "quit", "exit", "退出"}:
                break
            if not text:
                continue
            messages.append({"role": "user", "content": text})
            try:
                reply = run_turn(llm, messages, toolkit, protocol=args.protocol, max_rounds=args.max_rounds,
                                 on_event=print_event, temperature=0.3)
            except Exception as error:
                message = f"模型调用失败：{error}"
                print(message)
                write_json(log_dir / "error.json", {"ok": False, "provider": llm.provider,
                                                       "model": llm.model, "error": message})
                write_json(log_dir / "transcript.json", messages)
                if args.once:
                    break
                continue
            print(f"Agent：{reply}")
            write_json(log_dir / "transcript.json", messages)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\n操作者中断，停止机械臂")
        toolkit.stop_robot()
    finally:
        toolkit.close()
        for camera in cameras:
            if camera is not None:
                camera.close()


if __name__ == "__main__":
    main()
