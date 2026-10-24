"""Week 7 task 4.1: photograph a person → VLM parses the arm action → arm imitates it.

Sub-commands follow the course task book:
  capture   任务一：打开摄像头拍一张照片并保存
  describe  任务二：调用 VLM API 描述照片中人物的手臂动作
  imitate   任务三/四：拍照 → VLM 给出 6 个关节角 → 选择已记录安全姿态模仿（--prompt detailed 为任务四）

Without --execute the arm is simulated and no serial port is opened.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_common  # noqa: E402
from agent_common import extract_json, jpeg_data_url, read_json, run_dir, write_json  # noqa: E402
from arm_backend import (JOINT_LABELS, JOINT_NAMES, JointSpace, arm_state, available_poses,
                         crossing_joints, open_arm)  # noqa: E402
from camera_hub import ImageFileCamera, open_camera_source  # noqa: E402
from joint_mapping import mapped_position  # noqa: E402
from llm_client import add_llm_arguments, llm_from_args  # noqa: E402
from part1_imitation.prompts import DESCRIBE_PROMPT, EXAMPLES, build_prompt, load_examples  # noqa: E402

DEFAULT_CAMERA = "GENERAL WEBCAM"


def ask_vlm(llm, frame, prompt: str) -> str:
    """Send one image plus a text prompt; return the model's text reply."""
    completion = llm.chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": jpeg_data_url(frame)}},
        {"type": "text", "text": prompt},
    ]}], temperature=0.2)
    return completion.choices[0].message.content or ""


def parse_joint_reply(text: str) -> dict:
    """Validate the VLM JSON: description plus all six joint angles as numbers."""
    data = extract_json(text)
    arm_visible = data.get("arm_visible")
    if not isinstance(arm_visible, bool):
        raise ValueError("模型回复缺少布尔值 arm_visible，安全策略拒绝运动")
    joints = data.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("模型回复缺少 joints 对象")
    missing = [name for name in JOINT_NAMES if name not in joints]
    if missing:
        raise ValueError(f"模型回复缺少关节：{missing}")
    values = {}
    for name in JOINT_NAMES:
        value = joints[name]
        if isinstance(value, str):
            value = float(value.replace("°", "").strip())
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 不是数字：{joints[name]!r}")
        values[name] = float(value)
    return {"arm_visible": arm_visible, "description": str(data.get("description", "")),
            "matched_pose": data.get("matched_pose"),
            "matched_example": data.get("matched_example"), "joints": values}


def select_safe_pose(config: dict, requested: dict, examples_path=EXAMPLES, *,
                     matched_pose=None, matched_example=None) -> dict:
    """Map untrusted model angles to the nearest recorded pose used by the prompt.

    Independent joint limits do not guarantee a collision-free six-joint pose.
    Recorded poses have already gone through the Week 4 motion workflow, so real
    imitation executes one of those poses while retaining the raw VLM angles in
    the log for grading and prompt evaluation.
    """
    space = JointSpace(config)
    ranges = np.asarray([max(1.0, high - low) for low, high in space.limits_deg], dtype=float)
    wanted = np.asarray([requested[name] for name in JOINT_NAMES], dtype=float)
    valid_poses = set(available_poses(config))
    candidates = []
    for item in load_examples(examples_path).get("examples", []):
        pose = item.get("pose")
        if pose not in valid_poses:
            continue
        raw = [int(value) for value in config["poses"][pose]]
        degrees = np.asarray(space.to_degrees(raw), dtype=float)
        score = float(np.sqrt(np.mean(np.square((wanted - degrees) / ranges))))
        candidates.append({"pose": pose, "action": str(item.get("action", pose)),
                           "raw": raw, "degrees": degrees.tolist(), "score": round(score, 4)})
    if not candidates:
        raise ValueError("模仿示例中没有引用任何已记录姿态，安全策略拒绝运动")
    if isinstance(matched_pose, str):
        for candidate in candidates:
            if candidate["pose"] == matched_pose.strip():
                candidate = dict(candidate)
                candidate["selection"] = "model_matched_pose"
                return candidate
    if isinstance(matched_example, str):
        for candidate in candidates:
            if candidate["action"] == matched_example.strip():
                candidate = dict(candidate)
                candidate["selection"] = "model_matched_example"
                return candidate
    candidate = dict(min(candidates, key=lambda item: item["score"]))
    candidate["selection"] = "nearest_joint_angles"
    return candidate


def execution_target(config: dict, anchor: dict, current: list[int]) -> tuple[list[int], list[str]]:
    """Use the recorded pose, but avoid the observed unstable ID5 cross-zero transition."""
    target = list(anchor["raw"])
    notes = []
    mapping = config["joint_mapping"]["5"]
    current_mapped = mapped_position(current[4], mapping)
    target_mapped = mapped_position(target[4], mapping)
    direct_change = (target[4] - current[4]) * mapping.get("sign", 1)
    if target_mapped - current_mapped != direct_change:
        target[4] = int(current[4])
        notes.append("ID5 腕部旋转需要跨零；为避免已观察到的抽动，本次保持当前腕向")
    return target, notes


def plan_execution_pose(config: dict, requested: dict, anchor: dict, current: list[int],
                        examples_path=EXAMPLES) -> tuple[dict, list[int], list[str]]:
    """Require a no-cross target, using only an explicitly configured semantic fallback."""
    target, notes = execution_target(config, anchor, current)
    crossings = crossing_joints(config, current, target)
    if not crossings:
        return anchor, target, notes
    fallback_name = None
    for item in load_examples(examples_path).get("examples", []):
        if item.get("pose") == anchor["pose"]:
            fallback_name = item.get("no_cross_fallback")
            break
    if not isinstance(fallback_name, str) or not fallback_name.strip():
        raise ValueError(f"姿态“{anchor['pose']}”从当前位置需要关节 {crossings} 跨零，且没有已核对的替代姿态；拒绝运动")
    fallback = select_safe_pose(config, requested, examples_path, matched_pose=fallback_name)
    fallback["selection"] = "configured_no_cross_fallback"
    fallback_target, fallback_notes = execution_target(config, fallback, current)
    fallback_crossings = crossing_joints(config, current, fallback_target)
    if fallback_crossings:
        raise ValueError(f"替代姿态“{fallback_name}”仍需关节 {fallback_crossings} 跨零；拒绝运动")
    fallback_notes.insert(
        0, f"原姿态“{anchor['pose']}”需关节 {crossings} 跨零；改用已核对的同侧替代姿态“{fallback_name}”"
    )
    return fallback, fallback_target, fallback_notes


def print_plan(current_deg: dict, requested: dict, target_deg: dict, notes: list[str]):
    print(f"{'关节':<16}{'当前°':>9}{'模型给出°':>11}{'执行目标°':>11}")
    for name, label in zip(JOINT_NAMES, JOINT_LABELS):
        print(f"{name + '(' + label + ')':<16}{current_deg[name]:>9.1f}{requested[name]:>11.1f}{target_deg[name]:>11.1f}")
    for note in notes:
        print("  处理：" + note)


def capture_after_countdown(camera, delay_s: int):
    """Give the operator a visible countdown, then freeze one live frame."""
    if delay_s:
        print("请在倒计时结束前摆好并保持动作：", flush=True)
        for remaining in range(delay_s, 0, -1):
            print(f"  {remaining}...", flush=True)
            time.sleep(1)
    frame, _ = camera.latest()
    print("已拍照，可以放下手臂；正在调用视觉模型。", flush=True)
    return frame


def imitate_once(llm, camera, arm, config: dict, *, mode="detailed", duration_s=3.0,
                 out_dir: Path | None = None, examples_path=EXAMPLES, frame=None) -> dict:
    """Photo → VLM angles → nearest recorded pose → arm motion; return its audit log."""
    out_dir = out_dir or run_dir("imitate")
    if frame is None:
        frame = camera.snapshot(out_dir / "photo.jpg")
    else:
        from common import write_image
        write_image(out_dir / "photo.jpg", frame)
    prompt = build_prompt(config, mode, examples_path)
    started = time.monotonic()
    log = {"mode": mode, "model": llm.model, "prompt": prompt,
           "photo": str(out_dir / "photo.jpg"), "simulated": arm.simulated,
           "requested_duration_s": duration_s, "motion_timing": "recorded_pose_config"}
    try:
        reply = ask_vlm(llm, frame, prompt)
    except Exception as error:
        log.update(ok=False, latency_s=round(time.monotonic() - started, 2),
                   error=f"模型调用失败：{error}")
        print(f"本次模仿未执行：{log['error']}")
        write_json(out_dir / "log.json", log)
        return log
    latency = round(time.monotonic() - started, 2)
    log.update(reply=reply, latency_s=latency)
    try:
        parsed = parse_joint_reply(reply)
        log["parsed"] = parsed
        matched_pose = parsed.get("matched_pose")
        no_match = (isinstance(matched_pose, str)
                    and matched_pose.strip().lower() in {"无", "none", "null", "n/a"})
        if not parsed["arm_visible"] or no_match:
            raise ValueError("画面中没有完整可辨认的手臂，安全策略拒绝运动；请调整取景后重试")
        current = [int(v) for v in arm.positions()]
        requested_raw, clamp_notes = arm.joints.to_raw(parsed["joints"], current)
        requested_deg = dict(zip(JOINT_NAMES, arm.joints.to_degrees(requested_raw)))
        anchor = select_safe_pose(config, parsed["joints"], examples_path,
                                  matched_pose=parsed.get("matched_pose"),
                                  matched_example=parsed.get("matched_example"))
        model_anchor = dict(anchor)
        anchor, target_raw, execution_notes = plan_execution_pose(
            config, parsed["joints"], anchor, current, examples_path
        )
        target_deg = dict(zip(JOINT_NAMES, arm.joints.to_degrees(target_raw)))
        safety_note = (f"模型角度仅用于选择安全示例；实际执行已记录姿态“{anchor['pose']}”"
                       f"（{anchor['action']}，归一化距离 {anchor['score']:.4f}）")
        notes = [*clamp_notes, safety_note, *execution_notes]
        log.update(clamp_notes=clamp_notes, safety_notes=[safety_note],
                   execution_notes=execution_notes,
                   requested_raw=requested_raw, requested_deg=requested_deg,
                   model_selected_anchor=model_anchor, executed_anchor=anchor,
                   target_raw=target_raw, target_deg=target_deg,
                   start_raw=current, execution_policy="nearest_recorded_pose")
        print(f"\nVLM（{latency}s）动作描述：{parsed['description']}")
        if parsed.get("matched_example"):
            print(f"最接近的示例：{parsed['matched_example']}")
        print_plan(dict(zip(JOINT_NAMES, arm.joints.to_degrees(current))), parsed["joints"], target_deg, notes)
        command = ("joints", (target_raw, duration_s))
        arm.run(command)
        log["final"] = arm_state(arm)
        log["ok"] = True
        print(f"{'[模拟] ' if arm.simulated else ''}已到达：{log['final']['joints_deg']}")
    except Exception as error:
        log["ok"] = False
        log["error"] = str(error)
        print(f"本次模仿未执行：{error}")
    write_json(out_dir / "log.json", log)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["capture", "describe", "imitate"])
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="拍人的摄像头：Windows 设备名、数字索引，或已有照片路径（跳过拍照）")
    parser.add_argument("--image", type=Path, help="直接使用已有照片，等价于 --camera 照片路径")
    parser.add_argument("--prompt", choices=["basic", "detailed"], default="detailed",
                        help="basic=任务三基础提示词；detailed=任务四改进提示词（默认）")
    parser.add_argument("--examples", type=Path, default=EXAMPLES, help="任务四的关节说明与示例 JSON")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="兼容参数；安全姿态模式实际使用 week4 已记录姿态的运动时序")
    parser.add_argument("--capture-delay", type=int, default=3,
                        help="实机输入 RUN 后的拍照倒计时秒数，0～10，默认 3")
    parser.add_argument("--repeat", action="store_true", help="按回车重复拍照模仿，输入 q 退出")
    parser.add_argument("--execute", action="store_true", help="驱动真实机械臂；缺省为模拟")
    parser.add_argument("--port", help="机械臂串口，缺省自动识别")
    parser.add_argument("--yes", action="store_true", help="跳过实机运行前的 RUN 确认")
    add_llm_arguments(parser, purpose="vision")
    args = parser.parse_args()
    if not 0.5 <= args.duration <= 10:
        parser.error("--duration 需在 0.5～10 秒之间")
    if not 0 <= args.capture_delay <= 10:
        parser.error("--capture-delay 需在 0～10 秒之间")

    if args.image is not None and not args.image.is_file():
        parser.error(
            f"--image 指定的照片不存在：{args.image}。"
            "请把指南中的 capture_时间 替换为实际目录名，或先执行 capture 命令拍照。"
        )
    camera = open_camera_source(args.image if args.image is not None else args.camera)
    try:
        if args.command == "capture":
            out = run_dir("capture") / "photo.jpg"
            camera.snapshot(out)
            print(f"已保存照片：{out}")
            return
        llm = llm_from_args(args)
        if args.command == "describe":
            out_dir = run_dir("describe")
            frame = camera.snapshot(out_dir / "photo.jpg")
            try:
                reply = ask_vlm(llm, frame, DESCRIBE_PROMPT)
            except Exception as error:
                message = f"模型调用失败：{error}"
                write_json(out_dir / "log.json", {"ok": False, "model": llm.model,
                                                   "prompt": DESCRIBE_PROMPT, "error": message})
                print(f"照片：{out_dir / 'photo.jpg'}\n{message}\n失败日志：{out_dir / 'log.json'}")
                return
            write_json(out_dir / "log.json", {"ok": True, "model": llm.model,
                                               "prompt": DESCRIBE_PROMPT, "reply": reply})
            print(f"照片：{out_dir / 'photo.jpg'}\n模型（{llm.model}）输出：\n{reply}")
            return
        config = read_json(agent_common.ADVANCED_CONFIG)
        if args.execute and not args.yes:
            answer = input("实机模仿：确认机械臂周围空旷、可随时断电；模型给出的角度可能不准。输入 RUN 继续：")
            if answer.strip() != "RUN":
                print("未启动")
                return
        arm = open_arm(config, execute=args.execute, port=args.port)
        try:
            print(f"机械臂当前：{arm_state(arm)['joints_deg']}")
            while True:
                frame = None
                if args.execute and not isinstance(camera, ImageFileCamera):
                    frame = capture_after_countdown(camera, args.capture_delay)
                imitate_once(llm, camera, arm, config, mode=args.prompt, duration_s=args.duration,
                             examples_path=args.examples, frame=frame)
                if not args.repeat or input("\n回车再拍一次，q 退出：").strip().lower() == "q":
                    break
        except KeyboardInterrupt:
            print("\n操作者中断，停止机械臂")
        finally:
            arm.close()
    finally:
        camera.close()


if __name__ == "__main__":
    main()
