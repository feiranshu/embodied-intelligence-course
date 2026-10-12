"""Offline Chinese speech recognition, with live stop-command handling."""

from __future__ import annotations

import json
import os
import queue
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from advanced import parse_command, parse_literal_command


def parse_voice_command(text, config, buttons=False, literal=False):
    if buttons:
        text = re.sub(r"[\s，。！？,!?]", "", text).rstrip(".")
        text = re.sub(r"(?:吧|啊|一下)$", "", text)
        text = re.sub(r"^(?:机械臂)?(?:请)?完成(?=(?:动作|姿态|第))", "", text)
    try:
        command = (parse_literal_command if literal or buttons else parse_command)(text, config)
    except ValueError:
        if not buttons:
            raise
        raise ValueError("未听清一条完整口令；请重说动作1/2/3、进入工作姿态、向前移动五厘米或停止；不会猜测动作和距离") from None
    if buttons:
        kind, value = command
        if kind == "jog":
            return "move", value
        if not (kind in {"stop", "move"}
                or kind == "pose" and value in {"立正", "姿态2", "姿态3", "初始状态"}):
            raise ValueError("这条指令没有对应的按钮；请说动作1/2/3、进入工作姿态、六方向移动或停止")
    return command


def button_grammar(model):
    phrases = ["立正", "恢复", "复位", "初始状态", "初始姿态", "初始位置", "回到初始位置", "恢复到初始位置",
               "回到初始状态", "回到初始姿态", "恢复到初始状态", "恢复初始状态", "恢复初始位置",
               "恢复到初始姿态", "恢复初始动作", "回到初始动作", "回初始位置", "回初始状态", "回初始姿态", "进入工作姿态", "回到工作姿态", "工作姿态",
               "停止", "急停", "停下", "停", "别动", "不要动", "我", "你", "他", "什么"]
    phrases.extend(prefix + noun + number for prefix in ("", "进行", "执行", "完成", "做", "机械臂请进行")
                   for noun in ("动作", "姿态") for number in "零一二两三四五六七八九")
    phrases.extend("请完成第" + number + "个动作" for number in "一二三")
    directions = [prefix + direction + verb for prefix in ("向", "往")
                  for direction in "前后左右上下" for verb in ("", "移动")]
    phrases.extend(directions)
    digits = "零一二三四五六七八九"
    for number in range(1, 100):
        chinese = digits[number] if number < 10 else (
            (digits[number // 10] if number >= 20 else "") + "十" + (digits[number % 10] if number % 10 else ""))
        phrases.extend(direction + chinese + unit for direction in directions for unit in ("厘米", "毫米", "米"))
    sentences = []
    vocabulary = {}
    for phrase in phrases:
        tokens = []
        while phrase:
            for end in range(len(phrase), 0, -1):
                word = phrase[:end]
                if word not in vocabulary:
                    vocabulary[word] = model.vosk_model_find_word(word) >= 0
                if vocabulary[word]:
                    tokens.append(word)
                    phrase = phrase[end:]
                    break
            else:
                tokens = []
                break
        if tokens:
            sentence = " ".join(tokens)
            sentences.extend(prefix + sentence for prefix in ("", "请 ", "不要 ", "不 ", "别 ", "勿 ", "他 说 "))
            sentences.extend(sentence + ending for ending in (" 啊", " 吧", " 吗"))
    return json.dumps(list(dict.fromkeys(sentences)) + ["[unk]"], ensure_ascii=False)


def result_text(result, config, buttons=False):
    result = json.loads(result)
    alternatives = result.get("alternatives", [])
    text = (alternatives[0] if alternatives else result).get("text", "").strip()
    normalized = re.sub(r"\s", "", text)
    if buttons and alternatives and re.match(r"^(?:机械臂)?(?:请)?(?:向|往)?[前后左右上下]", normalized):
        try:
            parse_voice_command(text, config, buttons=True)
        except ValueError:
            pass
        else:
            return text
        if re.search(r"不|别|勿|负|然后|再|或|还是|吗|什么|如果|说|我|你|他|[−-]|\[unk\]", normalized):
            return text
        if re.search(r"[\d零一二两三四五六七八九十]+(?:厘米|毫米|米|cm|mm)", normalized):
            try:
                parse_voice_command(text, config, buttons=True)
            except ValueError:
                return text
        commands = {}
        for alternative in alternatives:
            candidate = alternative.get("text", "").strip()
            try:
                command = parse_voice_command(candidate, config, buttons=True)
            except ValueError:
                if re.fullmatch(r"(?:向|往)?[前后左右上下](?:移动|移)?[\d.零一二两三四五六七八九十]+(?:厘米|毫米|米|cm|mm)(?:啊|吧)?",
                                re.sub(r"\s", "", candidate)):
                    print("不执行：距离候选包含超限或无效数值，请重说", flush=True)
                    return ""
                continue
            if command[0] != "move" or not re.search(r"厘米|毫米|cm|mm", candidate):
                continue
            commands.setdefault(tuple(command[1]), candidate)
        if len(commands) > 1:
            print("不执行：语音候选的方向、数值或单位不一致，请重说", flush=True)
            return ""
        if commands:
            try:
                parse_voice_command(text, config, buttons=True)
            except ValueError:
                candidate = next(iter(commands.values()))
                print(f"语音候选校正：{text} → {candidate}", flush=True)
                return candidate
        return text
    if (not alternatives or re.search(r"不|别|勿|负|然后|再向|或|还是|吗|什么|如果|说", normalized)
            or not re.match(r"^(?:机械臂)?(?:请)?(?:进行|执行|运行|做|动作|姿态|第)", normalized)):
        return text
    numbered = r"(?:动作|姿态)[\d零一二两三四五六七八九十]|第[\d零一二两三四五六七八九十]+个(?:动作|姿态)"
    if not re.search(numbered, normalized):
        try:
            parse_literal_command(text, config)
        except ValueError:
            pass
        else:
            return text
    commands = {}
    numbered_hint = False
    for alternative in alternatives:
        candidate = alternative.get("text", "").strip()
        candidate_normalized = re.sub(r"\s", "", candidate)
        numbered_hint |= bool(re.match(r"(?:机械臂)?(?:请)?(?:(?:进行|执行|运行|做)(?:一下)?)?(?:动作|姿态)", candidate_normalized))
        if not re.fullmatch(r"(?:机械臂)?(?:请)?(?:(?:进行|执行|运行|做)(?:一下)?)?"
                            r"(?:(?:动作|姿态)[\d零一二两三四五六七八九十]+|第[\d零一二两三四五六七八九十]+个(?:动作|姿态))",
                            candidate_normalized):
            continue
        try:
            command = parse_literal_command(candidate, config)
        except ValueError:
            print("不执行：语音中的动作编号未配置，请重说", flush=True)
            return ""
        commands.setdefault(command, candidate)
    if len(commands) > 1:
        print("不执行：语音中的动作编号不明确，请重说", flush=True)
        return ""
    if not commands and numbered_hint:
        print("不执行：未识别到明确的动作编号，请重说", flush=True)
        return ""
    if re.search(numbered, normalized):
        return text
    return next(iter(commands.values()), text)


def dispatch_result(result, runner, buttons=False):
    text = result_text(result, runner.arm.config, buttons)
    if not text:
        return
    print(f"语音转录：{text}", flush=True)
    try:
        command = parse_voice_command(text, runner.arm.config, buttons)
    except ValueError as error:
        print(f"不执行：{error}", flush=True)
        return
    print(f"理解结果：{command}", flush=True)
    runner.submit(command)


def listen(runner, model_path, device=None, wav_path=None, buttons=False):
    from vosk import KaldiRecognizer, Model, SetLogLevel

    model_path = Path(model_path)
    if not (model_path / "am" / "final.mdl").is_file():
        raise FileNotFoundError(f"缺少中文 Vosk 模型：{model_path}；请按 README 安装模型")
    SetLogLevel(-1)
    original_directory = Path.cwd()
    try:
        os.chdir(model_path)
        model = Model(".")
    finally:
        os.chdir(original_directory)
    if buttons:
        print(f"语音按钮控制：动作1＝立正，动作2＝姿态2，动作3＝姿态3；恢复/回到初始位置；六方向移动；停止。"
              f"\n每句话执行一次；未说距离时移动 {runner.arm.config['step_mm']:g} mm，说“五厘米”则移动 50 mm。", flush=True)
    if wav_path:
        with wave.open(str(wav_path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise ValueError("WAV 必须是单声道、16 位无压缩 PCM")
            recognizer = KaldiRecognizer(model, source.getframerate())
            if buttons:
                recognizer.SetGrammar(button_grammar(model))
            recognizer.SetMaxAlternatives(20)
            while chunk := source.readframes(4000):
                if recognizer.AcceptWaveform(chunk):
                    dispatch_result(recognizer.Result(), runner, buttons)
                    runner.wait()
            dispatch_result(recognizer.FinalResult(), runner, buttons)
            runner.wait()
        return

    import sounddevice as sd

    samplerate = int(sd.query_devices(device, "input")["default_samplerate"])
    recognizer = KaldiRecognizer(model, samplerate)
    if buttons:
        recognizer.SetGrammar(button_grammar(model))
    recognizer.SetMaxAlternatives(20)
    audio = queue.Queue(maxsize=8)
    faults = queue.SimpleQueue()

    def callback(data, frames, timestamp, status):
        if status:
            faults.put(str(status))
        try:
            audio.put_nowait((time.monotonic(), bytes(data)))
        except queue.Full:
            faults.put("音频处理落后，停止接收指令")

    pending = None
    interpreter = ThreadPoolExecutor(max_workers=1)
    try:
        with sd.RawInputStream(device=device, samplerate=samplerate, blocksize=round(samplerate / 4),
                               dtype="int16", channels=1, callback=callback):
            print("录音已开始：" + ("请说上方口令，含糊转录不执行；" if buttons else "可用自然语言描述动作；")
                  + "说“停止”或按 Ctrl+C 停止。", flush=True)
            while True:
                if not faults.empty():
                    raise RuntimeError(f"麦克风错误：{faults.get()}")
                try:
                    captured_at, chunk = audio.get(timeout=0.3)
                except queue.Empty:
                    continue
                if time.monotonic() - captured_at > 1.5:
                    raise RuntimeError("音频已过期，请检查麦克风或处理速度")
                if recognizer.AcceptWaveform(chunk):
                    text = result_text(recognizer.Result(), runner.arm.config, buttons)
                    if text:
                        print(f"语音转录：{text}", flush=True)
                        try:
                            immediate = parse_voice_command(text, runner.arm.config, buttons, literal=True)
                        except ValueError as error:
                            if buttons:
                                print(f"不执行：{error}", flush=True)
                                continue
                            immediate = None
                        if immediate == ("stop", None):
                            runner.submit(immediate)
                            if pending is not None:
                                pending.cancel()
                                pending = None
                        elif pending is not None:
                            print("正在理解或执行上一条指令；可说“停止”", flush=True)
                        elif immediate is not None:
                            print(f"理解结果：{immediate}", flush=True)
                            runner.submit(immediate)
                        else:
                            pending = interpreter.submit(parse_voice_command, text, runner.arm.config, buttons)
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial.replace(" ", "") in {"停止", "急停", "停下", "停", "停一下", "别动", "别动了", "不要动"}:
                        runner.stop()
                        if pending is not None:
                            pending.cancel()
                            pending = None
                if pending is not None and pending.done():
                    try:
                        command = pending.result()
                    except ValueError as error:
                        print(f"不执行：{error}", flush=True)
                    else:
                        print(f"理解结果：{command}", flush=True)
                        runner.submit(command)
                    pending = None
    finally:
        try:
            runner.stop()
        finally:
            interpreter.shutdown(wait=False, cancel_futures=True)
