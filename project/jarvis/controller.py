import json
import math
import threading
import time
import uuid
from pathlib import Path

import cv2

from .perception import TemplateTracker


class TaskFailure(RuntimeError):
    pass


class Assistant:
    def __init__(self, camera, motion, detector, selector, config, run_dir,
                 mode="simulation", strategy="clarify", clock=time.monotonic,
                 sleep=time.sleep, tracker_factory=TemplateTracker):
        self.camera, self.motion = camera, motion
        self.detector, self.selector = detector, selector
        self.config = config["control"]
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.mode, self.strategy = mode, strategy
        self.clock, self.sleep = clock, sleep
        self.tracker_factory = tracker_factory
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.worker = None
        self.busy = False
        self.history = []
        self.pending = None
        self.frame = None
        self.events = []
        self.audio_client = None
        self.api_logged = {}
        self.retry_of = None
        self.state = {"status": "idle", "reply": "请描述你想找的物体。", "question": "",
                      "candidates": [], "selected_id": None, "error": None, "metrics": {}}
        for name in ("hold_s", "interval_s", "max_frame_age_s", "timeout_s", "stall_s", "min_progress"):
            value = self.config[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"control.{name} 必须是正数")
        if not 0 < self.config["center_fraction"] <= 1:
            raise ValueError("center_fraction 必须在 (0,1] 内")
        if strategy not in ("clarify", "direct") or (mode != "simulation" and strategy == "direct"):
            raise ValueError("直接选择策略仅用于软件对照实验")

    def snapshot(self):
        with self.lock:
            return {**self.state, "mode": self.mode, "busy": self.busy,
                    "events": list(self.events[-30:])}

    def _state(self, **values):
        with self.lock:
            self.state.update(values)

    def _event(self, event, **values):
        record = {"event": event, "time": self.clock(), **values}
        with self.lock:
            self.events.append(record)
            with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _check_stop(self):
        if self.cancel.is_set():
            raise InterruptedError("用户停止了当前任务")

    def _read(self, previous=None):
        self._check_stop()
        frame = self.camera.read()
        self._check_stop()
        age = self.clock() - frame.captured_at
        if age < 0 or age > self.config["max_frame_age_s"]:
            raise TaskFailure("画面已过期")
        if previous is not None and (frame.sequence <= previous.sequence or frame.captured_at <= previous.captured_at):
            raise TaskFailure("画面停止更新")
        with self.lock:
            self.frame = frame
        return frame

    def preview(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.image.copy()
        frame = self.camera.read()
        with self.lock:
            self.frame = frame
        return frame.image.copy()

    def submit(self, text):
        text = text.strip()
        if not text:
            raise ValueError("请输入目标描述")
        if text in {"停止", "停", "取消", "急停", "stop", "Stop"}:
            self.stop()
            return self.snapshot()
        return self._launch(text)

    def submit_audio(self, wav, model_client):
        return self._launch(None, wav, model_client)

    def _launch(self, text, wav=None, model_client=None):
        with self.lock:
            if self.busy:
                raise RuntimeError("当前任务还在运行，请先停止或等待结束")
            self.busy = True
            self.cancel.clear()
            if self.mode == "simulation":
                self.motion.resume()
            self._state(status="observing", error=None, question="", reply="正在观察最新画面。")
            self.audio_client = model_client
            self.worker = threading.Thread(target=self._work, args=(text, wav, model_client), daemon=True)
            self.worker.start()
        return self.snapshot()

    def run(self, text=None, wav=None, model_client=None):
        if wav is None:
            self.submit(text)
        else:
            self.submit_audio(wav, model_client)
        worker = self.worker
        if worker is not None:
            worker.join()
        return self.snapshot()

    def stop(self):
        started = self.clock()
        with self.lock:
            self.cancel.set()
        stop_error = self._stop_motion()
        self._event("stop_requested", request_duration_s=self.clock() - started, error=stop_error)
        with self.lock:
            if not self.busy and self.pending:
                self._finish("failed" if stop_error else "stopped", stop_error or "用户取消了等待澄清的任务")
            self._state(status="failed" if stop_error else "stopping" if self.busy else "stopped", question="",
                        reply=stop_error or ("已请求本地停止。" if self.busy else "任务已停止。"),
                        error=stop_error)

    def _stop_motion(self):
        try:
            self.motion.stop()
            return None
        except Exception as error:
            return "停止请求发生错误，未确认实机停止：" + str(error)

    def reset(self):
        with self.lock:
            if self.busy:
                raise RuntimeError("任务尚未结束")
            if self.pending:
                self._finish("stopped", "用户重置了对话")
            self.history.clear()
            self.cancel.clear()
            self._state(status="idle", reply="请描述你想找的物体。", question="", error=None,
                        selected_id=None, candidates=[], metrics={})
            self.frame = None

    def _usage(self):
        clients = []
        for obj in (self.detector, self.selector):
            client = getattr(obj, "client", None)
            if client is not None and all(client is not item for item in clients):
                clients.append(client)
        if self.audio_client is not None and all(self.audio_client is not item for item in clients):
            clients.append(self.audio_client)
        totals = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0}
        totals["tokens_fully_reported"] = True
        for client in clients:
            usage = getattr(client, "usage", {})
            for key in ("api_calls", "input_tokens", "output_tokens"):
                totals[key] += usage.get(key, 0)
            totals["tokens_fully_reported"] &= usage.get("tokens_fully_reported", False)
            calls = getattr(client, "calls", [])
            for call in calls[self.api_logged.get(id(client), 0):]:
                with (self.run_dir / "api_calls.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps({"task_id": self.pending["task_id"] if self.pending else None,
                                          **call}, ensure_ascii=False) + "\n")
            self.api_logged[id(client)] = len(calls)
        totals["cost"] = None if clients else 0.0
        return totals

    def _finish(self, status, reason=""):
        if self.pending is None:
            return
        task = self.pending
        usage = self._usage()
        record = {key: value for key, value in task.items() if key not in {"started", "usage_start"}}
        record.update(status=status, reason=reason, duration_s=round(self.clock() - task["started"], 4))
        record.setdefault("first_response_s", record["duration_s"])
        record.update({key: usage[key] - task["usage_start"][key]
                       for key in ("api_calls", "input_tokens", "output_tokens")})
        record["cost"] = usage["cost"]
        record["tokens_fully_reported"] = usage["tokens_fully_reported"]
        record["selection_correct"] = None
        with (self.run_dir / "tasks.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._state(metrics=record)
        self._event("task_finished", **record)
        self.pending = None
        self.history.clear()

    def _work(self, text, wav=None, model_client=None):
        try:
            if self.pending is None:
                self.pending = {"task_id": uuid.uuid4().hex[:12], "started": self.clock(),
                                "mode": self.mode, "strategy": self.strategy,
                                "detector": type(self.detector).__name__, "clarification_count": 0,
                                "selected_id": None, "usage_start": self._usage(), "retry_of": self.retry_of}
                self.retry_of = None
            task_id = self.pending["task_id"]
            if wav is not None:
                self._state(status="transcribing", reply="正在识别录音。")
                (self.run_dir / f"{task_id}_{time.time_ns()}.wav").write_bytes(wav)
                text = model_client.transcribe(wav)
                self._check_stop()
                if not text.strip():
                    raise TaskFailure("录音中没有识别到文字")
            self._event("user_message", task_id=task_id, text=text)
            frame = self._read()
            candidates = self.detector.detect(frame)
            self._check_stop()
            self._state(candidates=[c.to_dict() for c in candidates])
            self._event("observe", task_id=task_id, sequence=frame.sequence,
                        candidates=[c.to_dict() for c in candidates])
            selection = self.selector.select(text, candidates, list(self.history))
            self._check_stop()
            self.pending.setdefault("first_response_s", self.clock() - self.pending["started"])
            self.history.append({"role": "user", "content": text})
            if selection.intent == "stop":
                raise InterruptedError("用户取消了当前任务")
            if selection.intent == "chat":
                self._state(status="idle", reply=selection.reply or "请描述要观察的物体。")
                self._finish("stopped", "没有发起寻物动作")
                return
            by_id = {candidate.id: candidate for candidate in candidates}
            ids = list(dict.fromkeys(selection.candidate_ids))
            if any(candidate_id not in by_id for candidate_id in ids):
                raise TaskFailure("模型返回了画面中不存在的候选编号")
            if not ids and not selection.question:
                reply = "当前画面中未找到符合描述的物体。"
                self._state(status="not_found", reply=reply)
                self._finish("not_found", reply)
                return
            if (len(ids) != 1 or selection.question) and self.strategy == "clarify":
                question = selection.question or "有多个候选，请说明颜色、形状或左右位置。"
                self.pending["clarification_count"] += 1
                self.history.append({"role": "assistant", "content": question})
                self._state(status="clarifying", question=question, reply=question)
                self._event("clarification", task_id=task_id, question=question, candidate_ids=ids)
                return
            if not ids:
                raise TaskFailure("没有可以执行的明确目标")
            target = by_id[ids[0]]
            self.pending["selected_id"] = target.id
            self.pending["selected_label"] = target.label
            self._state(status="centering", selected_id=target.id,
                        reply=f"已选择{target.label}，正在根据新画面居中。")
            self._event("center_target", task_id=task_id, target=target.to_dict())
            self._save_frame(frame, f"{task_id}_before.jpg")
            final = self._center(frame, target)
            self._check_stop()
            self._save_frame(final, f"{task_id}_after.jpg")
            reply = f"{target.label}已进入画面中央，并持续保持 1 秒；新画面核对通过。"
            if self.config["hold_s"] != 1:
                reply = f"{target.label}已居中，保持 {self.config['hold_s']:g} 秒并通过新画面核对。"
            if self.mode == "simulation":
                reply = "软件模拟完成：" + reply
            with self.lock:
                self._check_stop()
                self._state(status="completed", reply=reply)
                self._finish("completed")
        except InterruptedError as error:
            stop_error = self._stop_motion()
            reason = str(error) + ("；" + stop_error if stop_error else "")
            status = "failed" if stop_error else "stopped"
            self._state(status=status, reply=reason, error=stop_error, question="")
            self._finish(status, reason)
        except Exception as error:
            stop_error = self._stop_motion()
            reason = str(error) + ("；" + stop_error if stop_error else "")
            self._state(status="failed", reply="任务未完成：" + reason, error=reason, question="")
            self._finish("failed", reason)
        finally:
            with self.lock:
                self.busy = False

    def _save_frame(self, frame, name):
        ok, data = cv2.imencode(".jpg", frame.image)
        if ok:
            (self.run_dir / name).write_bytes(data.tobytes())

    def _center(self, first, target):
        tracker = self.tracker_factory(first, target)
        started = self.clock()
        progress_at = started
        best_error = float("inf")
        hold_started = None
        previous = first
        half = self.config["center_fraction"] / 2
        while self.clock() - started <= self.config["timeout_s"]:
            self.sleep(self.config["interval_s"])
            frame = self._read(previous)
            previous = frame
            tracked = tracker.update(frame)
            if tracked is None:
                raise TaskFailure("目标丢失或跟踪身份不确定")
            dx, dy = tracked.center[0] - 0.5, tracked.center[1] - 0.5
            error = math.hypot(dx, dy)
            if abs(dx) <= half and abs(dy) <= half:
                if hold_started is None:
                    hold_started = self.clock()
                if self.clock() - hold_started >= self.config["hold_s"]:
                    break
            else:
                hold_started = None
                if error < best_error - self.config["min_progress"]:
                    best_error, progress_at = error, self.clock()
                if self.clock() - progress_at >= self.config["stall_s"]:
                    raise TaskFailure("运动后目标没有持续接近中心")
                self._check_stop()
                self.motion.step(dx, dy)
                self._event("motion_step", task_id=self.pending["task_id"],
                            sequence=frame.sequence, error=[dx, dy],
                            feedback=getattr(self.motion, "last_feedback", None))
        else:
            raise TaskFailure("居中任务超时")
        self._state(status="verifying", reply="正在用新的画面核对目标身份。")
        self.sleep(self.config["interval_s"])
        final = self._read(previous)
        tracked = tracker.update(final)
        if tracked is None:
            raise TaskFailure("结束前目标丢失")
        detected = self.detector.detect(final)
        self._check_stop()
        matches = [candidate for candidate in detected
                   if self._same_identity(target, candidate)
                   and math.dist(candidate.center, tracked.center) < 0.08]
        if len(matches) != 1:
            raise TaskFailure("新画面不能唯一确认原目标身份")
        cx, cy = matches[0].center
        if abs(cx - 0.5) > half or abs(cy - 0.5) > half:
            raise TaskFailure("新画面中目标未保持居中")
        if self.clock() - started > self.config["timeout_s"]:
            raise TaskFailure("最终核对超时")
        self.sleep(self.config["interval_s"])
        latest = self._read(final)
        current = tracker.update(latest)
        if current is None:
            raise TaskFailure("核对返回后目标已丢失")
        if any(abs(value - 0.5) > half for value in current.center):
            raise TaskFailure("核对返回后目标已离开中心")
        self._event("verified", task_id=self.pending["task_id"], sequence=latest.sequence,
                    target=matches[0].to_dict(), hold_s=self.clock() - hold_started)
        return latest

    @staticmethod
    def _same_identity(original, candidate):
        if original.marker_id is not None:
            return original.marker_id == candidate.marker_id
        return (original.color, original.shape, original.label) == (candidate.color, candidate.shape, candidate.label)

    def close(self):
        self.stop()
        if self.worker:
            self.worker.join(timeout=3)
        try:
            self.camera.close()
        finally:
            if self.motion is not self.camera:
                self.motion.close()
