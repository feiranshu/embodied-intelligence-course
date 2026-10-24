"""Prompts for Week 7 task 4.1 (course task book tasks 2–4)."""

from __future__ import annotations

import json
from pathlib import Path

import agent_common  # noqa: F401
from arm_backend import JOINT_LABELS, JOINT_NAMES, JointSpace, available_poses

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "imitation_examples.json"

DESCRIBE_PROMPT = (
    "请观察这张照片中的人物，用中文描述他的手臂正在做什么动作。"
    "分别说明左臂和右臂（以人物自身为准）：上臂抬起的高度和方向、肘部弯曲程度、手腕和手掌的状态。"
    "如果画面中看不到人或手臂，请直接说明。"
)

OUTPUT_FORMAT = (
    "只输出一个 JSON 对象，不要输出 Markdown 或其他文字，格式为：\n"
    '{"arm_visible": true或false, "description": "对人物手臂动作的中文描述", '
    '"joints": {' + ", ".join(f'"{name}": 角度数值' for name in JOINT_NAMES) + "}}"
)


def joint_table(config: dict) -> str:
    space = JointSpace(config)
    rows = [f"- {name}（{label}）：允许范围 {low}° ～ {high}°"
            for name, label, (low, high) in zip(JOINT_NAMES, JOINT_LABELS, space.limits_deg)]
    return "\n".join(rows)


def load_examples(path: str | Path = EXAMPLES) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_examples(config: dict, examples: dict) -> list[dict]:
    """Turn example pose names into joint angles from the current advanced.json."""
    space = JointSpace(config)
    valid_poses = set(available_poses(config))
    resolved = []
    for example in examples.get("examples", []):
        if "pose" in example:
            if example["pose"] not in valid_poses:
                continue
            joints = dict(zip(JOINT_NAMES, space.to_degrees(config["poses"][example["pose"]])))
        else:
            joints = {name: float(example["joints"][name]) for name in JOINT_NAMES}
        resolved.append({"action": example["action"], "pose": example.get("pose"), "joints": joints})
    return resolved


def basic_prompt(config: dict) -> str:
    """Task 3: describe the action, then convert it into six joint angles."""
    return (
        "你在控制一台 SO-101 六自由度桌面机械臂模仿照片中人物的手臂动作。\n"
        "第一步：描述照片中人物的手臂在做什么动作。\n"
        "如果画面中没有完整可辨认的手臂，arm_visible 必须为 false；程序会拒绝运动。否则为 true。\n"
        "第二步：把这个动作转换成机械臂 6 个关节的目标角度（单位：度）。关节名称、含义和允许范围：\n"
        f"{joint_table(config)}\n"
        "关节含义：shoulder_pan 让手臂左右转向；shoulder_lift 让大臂抬起或放下；elbow_flex 控制肘部弯曲；"
        "wrist_flex 控制手腕上下；wrist_roll 控制手腕旋转；gripper 控制夹爪开合（数值大为张开）。\n"
        "所有数值必须在允许范围内。\n" + OUTPUT_FORMAT
    )


def detailed_prompt(config: dict, examples: dict | None = None) -> str:
    """Task 4: structure of each joint, human→robot mapping and calibrated examples."""
    examples = examples if examples is not None else load_examples()
    notes = examples.get("joint_notes", {})
    joint_lines = []
    for name, label, (low, high) in zip(JOINT_NAMES, JOINT_LABELS, JointSpace(config).limits_deg):
        joint_lines.append(f"- {name}（{label}，{low}° ～ {high}°）：{notes.get(name, '')}")
    anchors = resolve_examples(config, examples)
    valid_poses = available_poses(config)
    fallback = "初始状态" if "初始状态" in valid_poses else valid_poses[0]
    anchor_lines = [f"- pose={item['pose']}；人的动作“{item['action']}” → " + json.dumps(
        {k: round(v, 1) for k, v in item["joints"].items()}, ensure_ascii=False) for item in anchors]
    rules = "\n".join(f"- {rule}" for rule in examples.get("human_to_robot", []))
    return (
        "你在控制一台 SO-101 六自由度桌面机械臂，让它尽量准确地模仿照片中人物的手臂姿态。\n"
        "## 机械臂结构\n"
        "机械臂从底座到末端依次为：底座旋转 → 肩部（大臂）→ 肘部（小臂）→ 腕部俯仰 → 腕部旋转 → 夹爪。"
        "角度单位为度，以下是每个关节的作用、允许范围和已核对的参考值：\n"
        + "\n".join(joint_lines) +
        "\n## 人与机械臂的对应关系\n" + rules +
        "\n## 已标定的动作示例（照片中的动作与示例相近时，直接使用或在示例基础上微调）\n"
        + "\n".join(anchor_lines) +
        "\n## 推理步骤\n"
        "1. 判断画面中是否有人以及要模仿哪条手臂；\n"
        "2. 依次估计这条手臂的朝向、上臂抬起程度、肘部弯曲、手腕和手掌状态；\n"
        "3. 找到最接近的示例，把它的 pose 名称原样写入 matched_pose；再按差异逐个关节调整；没有相近示例时 matched_pose 写“无”；\n"
        "4. 检查每个数值都在允许范围内。\n"
        f"画面中没有人或看不清完整手臂时，arm_visible 必须为 false、matched_pose 和 matched_example 都写“无”，"
        f"joints 使用“{fallback}”姿态的数值，并在 description 中说明；程序会拒绝运动。能清楚看见要模仿的手臂时 arm_visible 才能为 true。\n"
        "## 输出\n只输出一个 JSON 对象，不要输出 Markdown 或其他文字，格式为：\n"
        '{"arm_visible": true或false, "description": "人物手臂动作的中文描述", "matched_pose": "最接近示例的 pose 名称或“无”", '
        '"matched_example": "最接近的示例动作或“无”", '
        '"joints": {' + ", ".join(f'"{name}": 角度数值' for name in JOINT_NAMES) + "}}"
    )


def build_prompt(config: dict, mode: str, examples_path: str | Path = EXAMPLES) -> str:
    if mode == "basic":
        return basic_prompt(config)
    if mode == "detailed":
        return detailed_prompt(config, load_examples(examples_path))
    raise ValueError("prompt 只能是 basic 或 detailed")
