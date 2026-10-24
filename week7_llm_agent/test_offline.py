"""Offline checks for Week 7: no API key, camera, microphone or serial port needed.

Run from this directory:  python -X utf8 -m unittest test_offline.py -v
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "part2_function_calling"))
import agent_common  # noqa: E402
from agent_common import extract_json, read_json  # noqa: E402
from agent_loop import parse_text_tool_calls, run_turn  # noqa: E402
from arm_backend import (JOINT_NAMES, SimArm, arm_state, available_actions, available_poses,
                         crossing_joints, plan_safe_action, within_sync_release_tolerance)  # noqa: E402
from camera_hub import ImageFileCamera  # noqa: E402
from common import write_image  # noqa: E402
from llm_client import LLM  # noqa: E402
from joint_mapping import mapped_position  # noqa: E402
from part1_imitation.imitate import (capture_after_countdown, execution_target, imitate_once, parse_joint_reply,
                                     plan_execution_pose)  # noqa: E402
from part1_imitation.prompts import build_prompt  # noqa: E402
from robot_tools import RobotToolkit  # noqa: E402

CONFIG = read_json(agent_common.ADVANCED_CONFIG)


def calibrated_test_config():
    """Synthetic IK calibration used only to keep offline Cartesian tests exercised."""
    config = copy.deepcopy(CONFIG)
    config["kinematics"]["calibrated"] = True
    config["kinematics"]["zero_raw"][3] = 758
    config["kinematics"]["reference_angles_deg"][3] = 0.0
    return config


# ---------- fake OpenAI-compatible client ----------
def message(content="", tool_calls=()):
    calls = [SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
             for i, (name, args) in enumerate(tool_calls)]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))])


def chunk(content=None, tool_calls=None, audio=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, audio=audio)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def tool_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        return iter(reply) if kwargs.get("stream") else reply


def fake_llm(replies):
    completions = FakeCompletions(replies)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLM(client, "fake-model", {"thinking": {"type": "disabled"}}, "fake"), completions


def textured_image(path: Path, size=(640, 480)) -> Path:
    rng = np.random.default_rng(7)
    frame = np.full((size[1], size[0], 3), 200, np.uint8)
    patch = (rng.integers(0, 2, (16, 16)) * 255).astype(np.uint8)
    patch = cv2.resize(patch, (120, 120), interpolation=cv2.INTER_NEAREST)
    frame[60:180, 420:540] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    write_image(path, frame)
    return path


class JsonAndPromptTests(unittest.TestCase):
    def test_extract_json_tolerates_fences_and_prose(self):
        self.assertEqual(extract_json('结果如下：```json\n{"a": {"b": 1}}\n```'), {"a": {"b": 1}})
        self.assertEqual(extract_json('先说明 {不是JSON} 然后 {"x": 2} 结束'), {"x": 2})
        with self.assertRaises(ValueError):
            extract_json("没有对象")

    def test_parse_joint_reply_requires_all_six_numbers(self):
        joints = {name: 10 for name in JOINT_NAMES}
        joints["gripper"] = "30°"
        parsed = parse_joint_reply(json.dumps(
            {"arm_visible": True, "description": "举手", "joints": joints}, ensure_ascii=False))
        self.assertEqual(parsed["joints"]["gripper"], 30.0)
        del joints["elbow_flex"]
        with self.assertRaises(ValueError):
            parse_joint_reply(json.dumps({"joints": joints}))
        with self.assertRaisesRegex(ValueError, "arm_visible"):
            parse_joint_reply(json.dumps({"description": "举手", "joints": {name: 10 for name in JOINT_NAMES}}))

    def test_prompts_contain_joints_ranges_and_pose_examples(self):
        basic = build_prompt(CONFIG, "basic")
        detailed = build_prompt(CONFIG, "detailed")
        for name in JOINT_NAMES:
            self.assertIn(name, basic)
            self.assertIn(name, detailed)
        self.assertIn("-93.5", basic)
        self.assertIn("手臂向上高高举起", detailed)
        self.assertIn("matched_example", detailed)
        self.assertIn("matched_pose", detailed)


class ArmBackendTests(unittest.TestCase):
    def test_sync_pose_releases_on_first_sample_within_ten_ticks(self):
        target = [100, 200, 300, 400, 500, 600]
        self.assertTrue(within_sync_release_tolerance([90, 210, 300, 399, 505, 600], target, 10))
        self.assertFalse(within_sync_release_tolerance([89, 210, 300, 399, 505, 600], target, 10))

    def test_recorded_poses_round_trip_through_degrees(self):
        arm = SimArm(CONFIG)
        self.assertNotIn("立正", available_poses(CONFIG))
        self.assertIn("初始状态", available_poses(CONFIG))
        for name in available_poses(CONFIG):
            raw = CONFIG["poses"][name]
            target, notes = arm.joints.to_raw(arm.joints.to_degrees(raw), raw)
            self.assertEqual(notes, [], name)
            np.testing.assert_allclose(arm.joints.to_degrees(target), arm.joints.to_degrees(raw), atol=2.0)

    def test_id4_uses_confirmed_non_crossing_range(self):
        mapping = CONFIG["joint_mapping"]["4"]
        self.assertEqual(CONFIG["limits_raw"][3], [104, 758])
        self.assertEqual(mapping["limits"], [0, 654])
        self.assertFalse(mapping["crosses_zero"])
        arm = SimArm(CONFIG)
        for raw in range(104, 759):
            target, notes = arm.joints.to_raw({"wrist_flex": (raw - 104) * arm.joints.deg_per_raw},
                                              arm.positions())
            self.assertEqual(target[3], raw)
            self.assertEqual(notes, [])

    def test_id2_cross_zero_direction_matches_observed_hardware(self):
        self.assertEqual(CONFIG["joint_mapping"]["2"]["wheel_positive"], "right")

    def test_raise_pose_avoids_id2_and_id3_cross_zero_from_home(self):
        arm = SimArm(CONFIG)
        start = CONFIG["poses"]["初始状态"]
        target = CONFIG["poses"]["举手"]
        for index in (1, 2):
            mapping = CONFIG["joint_mapping"][str(index + 1)]
            mapped_change = mapped_position(target[index], mapping) - mapped_position(start[index], mapping)
            direct_change = (target[index] - start[index]) * mapping.get("sign", 1)
            self.assertEqual(mapped_change, direct_change)
        # The recorded wrist roll can still cross zero from some starting
        # poses; the imitation pipeline freezes ID5 before submitting motion.
        self.assertEqual(crossing_joints(CONFIG, start, target), [5])
        safe_target, notes = execution_target(CONFIG, {"raw": target}, start)
        self.assertEqual(crossing_joints(CONFIG, start, safe_target), [])
        self.assertEqual(safe_target[4], start[4])
        self.assertEqual(len(notes), 1)
        self.assertEqual(crossing_joints(CONFIG, start, CONFIG["poses"]["举手_原始记录"]), [2, 3, 5])

    def test_out_of_range_angles_are_clamped_and_reported(self):
        arm = SimArm(CONFIG)
        target, notes = arm.joints.to_raw({"shoulder_pan": 500, "gripper": -50}, arm.positions())
        degrees = dict(zip(JOINT_NAMES, arm.joints.to_degrees(target)))
        self.assertEqual(len(notes), 2)
        self.assertAlmostEqual(degrees["shoulder_pan"], arm.joints.limits_deg[0][1], delta=0.2)
        self.assertAlmostEqual(degrees["gripper"], arm.joints.limits_deg[5][0], delta=0.2)
        with self.assertRaises(ValueError):
            arm.joints.to_raw({"elbow": 1}, arm.positions())

    def test_only_actions_with_recorded_poses_are_offered(self):
        self.assertIn("左右摆动", available_actions(CONFIG))
        self.assertIn("挥手", available_actions(CONFIG))
        self.assertIn("点头", available_actions(CONFIG))

    def test_task5_gestures_change_only_the_requested_joint(self):
        current = CONFIG["poses"]["任务五基准"]
        wave, wave_notes = plan_safe_action(CONFIG, "挥手", current)
        nod, nod_notes = plan_safe_action(CONFIG, "点头", current)
        self.assertEqual(wave_notes, [])
        self.assertEqual(nod_notes, [])
        self.assertEqual(wave[0], current)
        self.assertEqual(nod[0], current)
        self.assertEqual([target[0] for target in wave], [1008, 929, 1009, 929, 1009])
        self.assertTrue(all(target[1:] == current[1:] for target in wave))
        self.assertEqual([target[3] for target in nod], [472, 522, 422, 522, 472])
        self.assertTrue(all(target[:3] + target[4:] == current[:3] + current[4:] for target in nod))

    def test_action_preflight_freezes_id5_and_removes_all_crossings(self):
        current = [1017, 902, 46, 720, 929, 377]
        config = copy.deepcopy(CONFIG)
        config["actions"]["ID5测试"] = [{"pose": "左侧"}]
        targets, notes = plan_safe_action(config, "ID5测试", current)
        self.assertEqual([target[4] for target in targets], [929])
        self.assertTrue(any("ID5=929" in note for note in notes))
        previous = current
        for target in targets:
            self.assertEqual(crossing_joints(CONFIG, previous, target), [])
            previous = target

    def test_safe_swing_only_moves_the_base_from_recorded_start(self):
        current = [928, 899, 183, 743, 923, 377]
        targets, notes = plan_safe_action(CONFIG, "左右摆动", current)
        self.assertEqual(notes, [])
        self.assertEqual([target[0] for target in targets], [1010, 928, 969])
        self.assertTrue(all(target[1:] == current[1:] for target in targets))


class ImitationTests(unittest.TestCase):
    def test_capture_countdown_waits_then_freezes_one_frame(self):
        camera = SimpleNamespace(latest=lambda: (np.full((2, 3, 3), 17, dtype=np.uint8), 123.0))
        with patch("part1_imitation.imitate.time.sleep") as sleep:
            frame = capture_after_countdown(camera, 3)
        self.assertEqual(sleep.call_count, 3)
        self.assertTrue(np.all(frame == 17))

    def test_full_pipeline_maps_vlm_angles_to_recorded_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "person.jpg"))
            reference_arm = SimArm(CONFIG)
            wanted = dict(zip(JOINT_NAMES, reference_arm.joints.to_degrees(CONFIG["poses"]["姿态3"])))
            llm, completions = fake_llm([message(json.dumps(
                {"arm_visible": True, "description": "手臂向正前方平伸，手掌张开",
                 "matched_example": "手臂向正前方平伸，手掌张开", "joints": wanted},
                ensure_ascii=False))])
            arm = SimArm(CONFIG)
            log = imitate_once(llm, camera, arm, CONFIG, out_dir=folder / "run")
            self.assertTrue(log["ok"], log.get("error"))
            self.assertEqual(arm.positions().tolist(), CONFIG["poses"]["姿态3"])
            self.assertEqual(log["execution_policy"], "nearest_recorded_pose")
            self.assertEqual(log["executed_anchor"]["pose"], "姿态3")
            self.assertIn("模型角度仅用于选择安全示例", log["safety_notes"][0])
            request = completions.requests[0]
            self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertTrue(request["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg"))
            self.assertTrue((folder / "run" / "log.json").is_file())

    def test_model_matched_pose_selects_recorded_raise_and_avoids_id5_crossing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "person.jpg"))
            reference_arm = SimArm(CONFIG)
            misleading = dict(zip(JOINT_NAMES, reference_arm.joints.to_degrees(CONFIG["poses"]["姿态3"])))
            llm, _ = fake_llm([message(json.dumps(
                {"arm_visible": True, "description": "手臂举到头顶上方，手掌张开", "matched_pose": "举手",
                 "matched_example": "手臂向上高高举起，肘部弯曲，手掌张开", "joints": misleading},
                ensure_ascii=False))])
            arm = SimArm(CONFIG)
            wrist_before = int(arm.positions()[4])
            log = imitate_once(llm, camera, arm, CONFIG, out_dir=folder / "run")
            expected = list(CONFIG["poses"]["举手"])
            expected[4] = wrist_before
            self.assertTrue(log["ok"], log.get("error"))
            self.assertEqual(log["executed_anchor"]["pose"], "举手")
            self.assertEqual(log["executed_anchor"]["selection"], "model_matched_pose")
            self.assertEqual(arm.positions().tolist(), expected)
            self.assertTrue(any("ID5" in note for note in log["execution_notes"]))

    def test_rerecorded_pose2_is_directly_reachable_without_fallback(self):
        arm = SimArm(CONFIG)
        current = [978, 899, 165, 725, 926, 276]
        requested = dict(zip(JOINT_NAMES, arm.joints.to_degrees(CONFIG["poses"]["姿态2"])))
        anchor = {
            "pose": "姿态2", "action": "手臂大幅抬起并收拢",
            "raw": list(CONFIG["poses"]["姿态2"]), "degrees": list(requested.values()), "score": 0.0,
        }
        planned, target, notes = plan_execution_pose(CONFIG, requested, anchor, current)
        self.assertEqual(planned["pose"], "姿态2")
        self.assertEqual(crossing_joints(CONFIG, current, target), [])
        self.assertFalse(any("同侧替代姿态" in note for note in notes))
        self.assertEqual(target, CONFIG["poses"]["姿态2"])
        home = list(CONFIG["poses"]["初始状态"])
        planned, target, notes = plan_execution_pose(CONFIG, requested, anchor, home)
        self.assertEqual(planned["pose"], "姿态2")
        self.assertEqual(crossing_joints(CONFIG, home, target), [])
        self.assertEqual(target[4], home[4])
        self.assertTrue(any("ID5" in note for note in notes))

    def test_explicit_no_visible_arm_never_moves(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "face.jpg"))
            joints = {name: 0 for name in JOINT_NAMES}
            llm, _ = fake_llm([message(json.dumps(
                {"arm_visible": False, "description": "只有面部，没有手臂", "matched_pose": "无",
                 "matched_example": "无", "joints": joints}, ensure_ascii=False))])
            arm = SimArm(CONFIG)
            before = arm.positions().tolist()
            log = imitate_once(llm, camera, arm, CONFIG, out_dir=folder / "run")
            self.assertFalse(log["ok"])
            self.assertIn("拒绝运动", log["error"])
            self.assertEqual(arm.positions().tolist(), before)
            self.assertFalse(log["parsed"]["arm_visible"])

    def test_bad_reply_does_not_move(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "person.jpg"))
            llm, _ = fake_llm([message("我看不清楚")])
            arm = SimArm(CONFIG)
            before = arm.positions().tolist()
            log = imitate_once(llm, camera, arm, CONFIG, out_dir=folder / "run")
            self.assertFalse(log["ok"])
            self.assertEqual(arm.positions().tolist(), before)

    def test_api_failure_is_logged_and_does_not_move(self):
        class FailingLLM:
            model = "busy-model"

            @staticmethod
            def chat(_messages, **_kwargs):
                raise RuntimeError("服务暂时繁忙")

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "person.jpg"))
            arm = SimArm(CONFIG)
            before = arm.positions().tolist()
            log = imitate_once(FailingLLM(), camera, arm, CONFIG, out_dir=folder / "run")
            self.assertFalse(log["ok"])
            self.assertIn("模型调用失败", log["error"])
            self.assertEqual(arm.positions().tolist(), before)
            self.assertTrue((folder / "run" / "log.json").is_file())


class ToolkitTests(unittest.TestCase):
    def test_pose_and_action_schemas_list_only_available_names(self):
        schemas = {tool["function"]["name"]: tool for tool in self.kit.schemas()}
        pose_names = schemas["go_to_pose"]["function"]["parameters"]["properties"]["name"]["enum"]
        action_names = schemas["run_action"]["function"]["parameters"]["properties"]["name"]["enum"]
        self.assertEqual(pose_names, self.kit.list_capabilities()["poses"])
        self.assertEqual(action_names, self.kit.list_capabilities()["actions"])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.kit = RobotToolkit(SimArm(CONFIG), log_dir=Path(self.folder.name))

    def tearDown(self):
        self.kit.close()
        self.folder.cleanup()

    def test_vision_tools_hidden_without_camera(self):
        names = {tool["function"]["name"] for tool in self.kit.schemas()}
        self.assertIn("go_to_pose", names)
        self.assertNotIn("look", names)
        self.assertFalse(self.kit.dispatch("look", {})["ok"])

    def test_read_only_mode_hides_and_rejects_motion_tools(self):
        kit = RobotToolkit(SimArm(CONFIG), log_dir=Path(self.folder.name), read_only=True)
        names = {tool["function"]["name"] for tool in kit.schemas()}
        self.assertIn("get_robot_state", names)
        self.assertIn("list_capabilities", names)
        self.assertIn("stop_robot", names)
        self.assertNotIn("go_to_pose", names)
        self.assertNotIn("run_action", names)
        self.assertFalse(kit.dispatch("go_to_pose", {"name": "居中"})["ok"])

    def test_motion_tools_follow_workspace_rules(self):
        names = {tool["function"]["name"] for tool in self.kit.schemas()}
        self.assertNotIn("move_end_effector", names)
        result = self.kit.dispatch("move_end_effector", {"direction": "前", "distance_mm": 20})
        self.assertFalse(result["ok"])
        self.assertTrue(self.kit.dispatch("go_to_pose", '{"name": "初始状态"}')["ok"])
        self.assertFalse(arm_state(self.kit.arm)["cartesian_ready"])
        self.assertFalse(self.kit.dispatch("move_end_effector", {"direction": "上", "distance_mm": 500})["ok"])
        self.assertFalse(self.kit.dispatch("go_to_pose", {"name": "立正"})["ok"])
        self.assertTrue(self.kit.dispatch("run_action", {"name": "挥手"})["ok"])
        self.assertTrue(self.kit.dispatch("run_action", {"name": "点头"})["ok"])
        self.assertTrue(self.kit.dispatch("run_action", {"name": "左右摆动"})["ok"])
        clamped = self.kit.dispatch("set_joint_angles", {"joints": {"gripper": -999}})
        self.assertTrue(clamped["ok"])
        self.assertEqual(len(clamped["clamped"]), 1)
        too_far = self.kit.dispatch("set_joint_angles", {"joints": {"shoulder_pan": 80}})
        self.assertFalse(too_far["ok"])
        self.assertIn("单次关节变化过大", too_far["error"])
        self.assertFalse(self.kit.dispatch("no_such_tool", {})["ok"])
        self.assertTrue(self.kit.dispatch("stop_robot", {})["ok"])
        self.assertTrue((Path(self.folder.name) / "tool_calls.json").is_file())


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.kit = RobotToolkit(SimArm(CONFIG), log_dir=Path(self.folder.name))

    def tearDown(self):
        self.kit.close()
        self.folder.cleanup()

    def test_native_function_calling_round_trip(self):
        llm, completions = fake_llm([
            message("", [("go_to_pose", {"name": "初始状态"})]),
            message("", [("go_to_pose", {"name": "左侧"})]),
            message("已经回到初始状态并转到左侧姿态。"),
        ])
        messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "复位后转到左侧姿态"}]
        reply = run_turn(llm, messages, self.kit)
        self.assertIn("左侧姿态", reply)
        self.assertEqual([m["role"] for m in messages[2:]], ["assistant", "tool", "assistant", "tool", "assistant"])
        self.assertTrue(json.loads(messages[3]["content"])["ok"])
        self.assertEqual(messages[3]["tool_call_id"], messages[2]["tool_calls"][0]["id"])
        self.assertIn("tools", completions.requests[0])

    def test_streamed_tool_call_arguments_are_joined(self):
        llm, _ = fake_llm([
            [chunk("好的，"), chunk(tool_calls=[tool_delta(0, "c1", "go_to_pose", '{"na')]),
             chunk(tool_calls=[tool_delta(0, None, None, 'me": "居中"}')]), SimpleNamespace(choices=[])],
            [chunk(audio={"transcript": "到居中了", "data": "AAAA"})],
        ])
        audio = []
        messages = [{"role": "user", "content": "去居中"}]
        reply = run_turn(llm, messages, self.kit, stream=True, on_audio=audio.append)
        self.assertEqual(reply, "到居中了")
        self.assertEqual(audio, ["AAAA"])
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], '{"name": "居中"}')
        self.assertEqual(self.kit.arm.positions().tolist(), CONFIG["poses"]["居中"])

    def test_text_protocol_for_models_without_tools(self):
        calls = parse_text_tool_calls('好的<tool_call>{"name": "go_to_pose", "arguments": {"name": "居中"}}</tool_call>')
        self.assertEqual(calls[0]["name"], "go_to_pose")
        llm, completions = fake_llm([
            message('<tool_call>{"name": "get_robot_state", "arguments": {}}</tool_call>'),
            message("机械臂当前状态已读取。"),
        ])
        messages = [{"role": "user", "content": "你现在什么姿态"}]
        self.assertEqual(run_turn(llm, messages, self.kit, protocol="text"), "机械臂当前状态已读取。")
        self.assertNotIn("tools", completions.requests[0])
        self.assertTrue(messages[2]["content"].startswith("工具结果"))

    def test_round_limit_stops_loop(self):
        llm, _ = fake_llm([message("", [("get_robot_state", {})])] * 3)
        reply = run_turn(llm, [{"role": "user", "content": "x"}], self.kit, max_rounds=3)
        self.assertIn("上限", reply)

    def test_multiple_motion_calls_in_one_reply_are_all_rejected(self):
        before = self.kit.arm.positions().tolist()
        llm, _ = fake_llm([
            message("", [("go_to_pose", {"name": "居中"}),
                         ("set_joint_angles", {"joints": {"gripper": 10}})]),
            message("本轮没有执行运动。"),
        ])
        messages = [{"role": "user", "content": "同时去居中并调整夹爪"}]
        self.assertIn("本轮已停止", run_turn(llm, messages, self.kit))
        self.assertEqual(self.kit.arm.positions().tolist(), before)
        results = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not result["ok"] for result in results))
        self.assertTrue(all("多个运动工具" in result["error"] for result in results))

    def test_failed_motion_prevents_model_from_trying_recovery_motion(self):
        before = self.kit.arm.positions().tolist()
        llm, completions = fake_llm([
            message("", [("run_action", {"name": "不存在"})]),
            message("", [("go_to_pose", {"name": "居中"})]),
        ])
        messages = [{"role": "user", "content": "执行动作"}]
        reply = run_turn(llm, messages, self.kit)
        self.assertIn("不再自动尝试其他动作", reply)
        self.assertEqual(self.kit.arm.positions().tolist(), before)
        self.assertEqual(len(completions.requests), 1)

    def test_successful_motion_prevents_second_motion_in_same_turn(self):
        llm, completions = fake_llm([
            message('', [("go_to_pose", {"name": "居中"})]),
            message('', [("go_to_pose", {"name": "初始状态"})]),
        ])
        messages = [{"role": "user", "content": "去居中"}]
        reply = run_turn(llm, messages, self.kit, max_motion_calls=1)
        self.assertIn("禁止自动追加第二次运动", reply)
        self.assertEqual(self.kit.arm.positions().tolist(), CONFIG["poses"]["居中"])
        second_request_tools = {tool["function"]["name"] for tool in completions.requests[1]["tools"]}
        self.assertNotIn("go_to_pose", second_request_tools)


class VisionToolTests(unittest.TestCase):
    def test_person_camera_alone_only_enables_imitation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "person.jpg"))
            vision, _ = fake_llm([])
            kit = RobotToolkit(SimArm(CONFIG), camera=None, person_camera=camera,
                               vision_llm=vision, log_dir=folder)
            try:
                names = {tool["function"]["name"] for tool in kit.schemas()}
                self.assertIn("imitate_person", names)
                self.assertNotIn("look", names)
                self.assertNotIn("detect_object", names)
                self.assertNotIn("start_tracking", names)
            finally:
                kit.close()
                camera.close()

    def test_detect_then_track_stops_at_cumulative_travel_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "scene.jpg"))
            # Object occupies x 420–540, y 60–180 of 640×480 → normalised 0–1000 box.
            vision, _ = fake_llm([message('{"found": true, "bbox": [656, 125, 844, 375], "label": "方块"}')])
            config = calibrated_test_config()
            kit = RobotToolkit(SimArm(config), camera, vision, config=config, log_dir=folder, track_step_mm=5)
            try:
                self.assertTrue(kit.dispatch("go_to_pose", {"name": "初始状态"})["ok"])
                found = kit.dispatch("detect_object", {"description": "方块"})
                self.assertTrue(found["found"], found)
                self.assertGreater(found["center_normalized"][0], 0.6)
                started = kit.dispatch("start_tracking", {
                    "object_id": found["object_id"], "max_steps": 40, "max_travel_mm": 10})
                self.assertTrue(started["ok"], started)
                self.assertFalse(kit.dispatch("go_to_pose", {"name": "居中"})["ok"])  # busy while tracking
                kit.tracking.thread.join(timeout=20)
                summary = kit.tracking.summary()
                # The photo never changes, so the simulated servo reaches the 10 mm cumulative limit.
                self.assertEqual(summary["status"], "travel_limit", summary)
                self.assertEqual(summary["steps"], 2)
                self.assertEqual(summary["travel_mm"], 10.0)
                self.assertTrue((folder / f"tracking_{found['object_id']}.json").is_file())
                self.assertTrue(kit.dispatch("go_to_pose", {"name": "居中"})["ok"])
            finally:
                kit.close()

    def test_real_tracking_is_locked_for_unvalidated_hand_eye(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            camera = ImageFileCamera(textured_image(folder / "scene.jpg"))
            vision, _ = fake_llm([message(
                '{"found": true, "bbox": [656, 125, 844, 375], "label": "方块"}')])
            config = calibrated_test_config()
            arm = SimArm(config)
            kit = RobotToolkit(arm, camera, vision, config=config, log_dir=folder)
            try:
                found = kit.dispatch("detect_object", {"description": "方块"})
                self.assertTrue(found["found"], found)
                arm.simulated = False  # Only exercise the safety gate; no serial backend is created.
                started = kit.dispatch("start_tracking", {"object_id": found["object_id"]})
                self.assertFalse(started["ok"])
                self.assertIn("实机视觉追踪保持锁定", started["error"])
                self.assertIsNone(kit.tracking)
            finally:
                kit.close()


class OmniHelpersTests(unittest.TestCase):
    def test_siliconflow_asr(self):
        from part3_omni_agent.asr import SiliconFlowASR

        seen = {}

        class Transcriptions:
            @staticmethod
            def create(**kwargs):
                seen.update(kwargs)
                return SimpleNamespace(text="  向左移动  ")

        client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()))
        asr = SiliconFlowASR(client=client)
        self.assertEqual(asr.transcribe(b"RIFFfake"), "向左移动")
        self.assertEqual(seen["model"], "FunAudioLLM/SenseVoiceSmall")
        self.assertEqual(seen["file"].name, "speech.wav")

    def test_history_compaction_and_audio_encoding(self):
        from part3_omni_agent.audio_io import audio_data_url, wav_bytes
        from part3_omni_agent.omni_agent import compact_history, loggable, user_turn

        wav = wav_bytes(np.zeros(1600, np.int16))
        self.assertTrue(audio_data_url(wav).startswith("data:audio/wav;base64,"))
        frame = np.zeros((48, 64, 3), np.uint8)
        messages = [{"role": "system", "content": "s"}] + [user_turn(wav, frame, None) for _ in range(3)]
        compact_history(messages, keep_audio_turns=2)
        kinds = [[p["type"] for p in m["content"]] for m in messages[1:]]
        self.assertEqual(kinds[0], ["text", "text", "text"])
        self.assertEqual(kinds[1], ["text", "input_audio", "text"])
        self.assertEqual(kinds[2], ["image_url", "input_audio", "text"])
        self.assertNotIn("base64", json.dumps(loggable(messages)))


if __name__ == "__main__":
    unittest.main()
