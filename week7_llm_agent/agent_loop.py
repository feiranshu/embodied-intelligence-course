"""LLM ⇄ tool loop shared by the text agent (4.2) and the omni agent (4.3).

protocol="native": OpenAI function calling (``tools=`` + ``tool_calls``).
protocol="text":   for models without function calling; the model writes
                   <tool_call>{"name": ..., "arguments": {...}}</tool_call>
                   and the results come back as a user message.
"""

from __future__ import annotations

import json
import re
import uuid

from agent_common import extract_json

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
MOTION_TOOLS = {"go_to_pose", "run_action", "move_end_effector", "set_joint_angles",
                "start_tracking", "imitate_person"}


def text_protocol_instructions(schemas: list[dict]) -> str:
    lines = ["你不能直接使用 function calling。需要调用工具时，在回复中输出一行或多行：",
             '<tool_call>{"name": "工具名", "arguments": {参数}}</tool_call>',
             "输出工具调用后立即结束本次回复，等待“工具结果”消息；不需要工具时直接回答用户。可用工具："]
    for tool in schemas:
        function = tool["function"]
        lines.append(f"- {function['name']}：{function['description']} 参数 JSON Schema："
                     + json.dumps(function["parameters"], ensure_ascii=False))
    return "\n".join(lines)


def _audio_part(delta):
    audio = getattr(delta, "audio", None)
    if audio is None and getattr(delta, "model_extra", None):
        audio = delta.model_extra.get("audio")
    return audio if isinstance(audio, dict) else None


def complete(llm, messages, *, tools=None, stream=False, on_text=None, on_audio=None, **kwargs):
    """Return (text, tool_calls) where tool_calls = [{"id", "name", "arguments"}].

    With audio output (omni models), base64 audio chunks go to on_audio and the
    spoken transcript counts as text when the model sends no separate content.
    """
    if tools:
        kwargs["tools"] = tools
    if not stream:
        message = llm.chat(messages, **kwargs).choices[0].message
        calls = [{"id": call.id, "name": call.function.name, "arguments": call.function.arguments or "{}"}
                 for call in (message.tool_calls or [])]
        if on_text and message.content:
            on_text(message.content)
        return message.content or "", calls
    text, transcript, pending = [], [], {}
    for chunk in llm.chat(messages, stream=True, **kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            text.append(delta.content)
            if on_text:
                on_text(delta.content)
        audio = _audio_part(delta)
        if audio:
            if audio.get("data") and on_audio:
                on_audio(audio["data"])
            if audio.get("transcript"):
                transcript.append(audio["transcript"])
                if on_text and not text:
                    on_text(audio["transcript"])
        for call in getattr(delta, "tool_calls", None) or []:
            slot = pending.setdefault(call.index, {"id": None, "name": "", "arguments": ""})
            if call.id:
                slot["id"] = call.id
            if call.function is not None:
                slot["name"] += call.function.name or ""
                slot["arguments"] += call.function.arguments or ""
    calls = []
    for index in sorted(pending):
        slot = pending[index]
        slot["id"] = slot["id"] or f"call_{uuid.uuid4().hex[:12]}"
        slot["arguments"] = slot["arguments"] or "{}"
        calls.append(slot)
    return "".join(text) or "".join(transcript), calls


def parse_text_tool_calls(text: str) -> list[dict]:
    calls = []
    for body in TOOL_CALL_PATTERN.findall(text or ""):
        data = extract_json(body)
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": str(data.get("name", "")),
                      "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)})
    return calls


def run_turn(llm, messages: list, toolkit, *, protocol="native", stream=False, max_rounds=8,
             max_motion_calls=None, on_event=None, on_text=None, **kwargs) -> str:
    """Let the model call tools until it answers in plain text; messages is extended in place."""
    on_event = on_event or (lambda kind, data: None)
    schemas = toolkit.schemas()
    completed_motion = None
    completed_motion_calls = 0
    for _ in range(max_rounds):
        text, calls = complete(llm, messages, tools=schemas if protocol == "native" else None,
                               stream=stream, on_text=on_text, **kwargs)
        if protocol == "text":
            calls = parse_text_tool_calls(text)
        if not calls:
            messages.append({"role": "assistant", "content": text})
            return text
        if protocol == "native":
            messages.append({"role": "assistant", "content": text or "", "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls]})
        else:
            messages.append({"role": "assistant", "content": text})
        results = []
        failed_motion = None
        multiple_motion_calls = sum(call["name"] in MOTION_TOOLS for call in calls) > 1
        for call in calls:
            on_event("call", call)
            motion_limit_reached = (max_motion_calls is not None
                                    and completed_motion_calls >= max_motion_calls)
            if motion_limit_reached and call["name"] in MOTION_TOOLS:
                result = {"ok": False, "error": f"本轮已经成功执行运动工具 {completed_motion}；禁止自动追加第二次运动。"}
            elif multiple_motion_calls and call["name"] in MOTION_TOOLS:
                result = {"ok": False, "error": "同一模型回复包含多个运动工具；为避免连续误动作，本轮未执行。请每次只调用一个运动工具并等待结果。"}
            else:
                result = toolkit.dispatch(call["name"], call["arguments"])
            if call["name"] in MOTION_TOOLS and not result.get("ok", False):
                failed_motion = (call["name"], str(result.get("error", "未知错误")))
            elif call["name"] in MOTION_TOOLS:
                completed_motion = call["name"]
                completed_motion_calls += 1
            on_event("result", {"name": call["name"], "result": result})
            payload_result = result
            if (call["name"] in MOTION_TOOLS and result.get("ok", False)
                    and max_motion_calls is not None and completed_motion_calls >= max_motion_calls):
                payload_result = {**result, "agent_instruction":
                                  "运动已经成功完成。本轮不得再调用任何运动工具；请直接简短确认完成，不要复位或校准。"}
            payload = json.dumps(payload_result, ensure_ascii=False)
            if protocol == "native":
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": payload})
            else:
                results.append(f"{call['name']} → {payload}")
        if protocol == "text":
            messages.append({"role": "user", "content": "工具结果：\n" + "\n".join(results)})
        if max_motion_calls is not None and completed_motion_calls >= max_motion_calls:
            schemas = [tool for tool in schemas if tool["function"]["name"] not in MOTION_TOOLS]
        if failed_motion is not None:
            name, error = failed_motion
            final = f"运动工具 {name} 执行失败，本轮已停止，不再自动尝试其他动作：{error}"
            messages.append({"role": "assistant", "content": final})
            return final
    final = "工具调用轮数达到上限，本轮停止。请换个说法或分步下达指令。"
    messages.append({"role": "assistant", "content": final})
    return final
