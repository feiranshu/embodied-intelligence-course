import json
from pathlib import Path
import tempfile
import threading
import unittest

from jarvis.controller import Assistant
from jarvis.models import Candidate, Selection
from jarvis.perception import ColorShapeDetector, LocalSelector
from jarvis.simulation import Simulation


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = json.loads((Path(__file__).parents[1] / "config/default.json").read_text())
        self.clock = Clock()

    def make(self, scenario="default", selector=None, detector=None):
        sim = Simulation(scenario, clock=self.clock)
        app = Assistant(sim, sim, detector or ColorShapeDetector(), selector or LocalSelector(),
                        self.config, self.tmp.name, clock=self.clock, sleep=self.clock.sleep)
        return app, sim

    def records(self):
        return [json.loads(line) for line in (Path(self.tmp.name) / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_actual_images_center_and_fresh_verification(self):
        app, sim = self.make()
        state = app.run("帮我看看蓝色盒子")
        self.assertEqual(state["status"], "completed", state)
        self.assertGreater(sim.steps, 0)
        verified = next(event for event in app.events if event["event"] == "verified")
        moves = [event for event in app.events if event["event"] == "motion_step"]
        self.assertGreater(verified["sequence"], moves[-1]["sequence"])
        self.assertGreaterEqual(verified["hold_s"], 1)
        self.assertEqual(self.records()[0]["status"], "completed")

    def test_ambiguity_never_moves_until_answer(self):
        app, sim = self.make()
        self.assertEqual(app.run("红色那个")["status"], "clarifying")
        self.assertEqual(sim.steps, 0)
        self.assertEqual(app.run("圆的")["status"], "completed")
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.records()[0]["clarification_count"], 1)

    def test_absent_target_never_moves(self):
        app, sim = self.make()
        self.assertEqual(app.run("紫色那个")["status"], "not_found")
        self.assertEqual(sim.steps, 0)

    def test_faults_record_failures_without_success(self):
        for scenario, expected in [("lost", "丢失"), ("stale", "停止更新"), ("stalled", "接近中心")]:
            with self.subTest(scenario=scenario):
                app, sim = self.make(scenario)
                state = app.run("蓝色盒子")
                self.assertEqual(state["status"], "failed", state)
                self.assertIn(expected, state["error"])
                self.assertTrue(sim.stopped)
        self.assertEqual(len(self.records()), 3)

    def test_stop_pending_clarification_records_attempt(self):
        app, sim = self.make()
        app.run("红色那个")
        app.stop()
        self.assertEqual(self.records()[0]["status"], "stopped")
        self.assertIsNone(app.pending)
        self.assertEqual(sim.steps, 0)

    def test_stop_during_remote_selection_never_moves(self):
        entered, release = threading.Event(), threading.Event()

        class SlowSelector:
            def select(self, text, candidates, history):
                entered.set()
                release.wait(2)
                return Selection(candidate_ids=[candidates[0].id])

        app, sim = self.make(selector=SlowSelector())
        app.submit("蓝色盒子")
        self.assertTrue(entered.wait(2))
        app.stop()
        release.set()
        app.worker.join(2)
        self.assertEqual(app.snapshot()["status"], "stopped")
        self.assertEqual(sim.steps, 0)

    def test_final_identity_mismatch_must_fail(self):
        class ChangedDetector:
            def __init__(self):
                self.calls = 0

            def detect(self, frame):
                self.calls += 1
                detected = ColorShapeDetector().detect(frame)
                if self.calls > 1:
                    return [Candidate(c.id, "黄色圆形", "yellow", "circle", c.bbox) for c in detected]
                return detected

        app, _ = self.make(detector=ChangedDetector())
        state = app.run("蓝色盒子")
        self.assertEqual(state["status"], "failed")
        self.assertIn("身份", state["error"])

    def test_hallucinated_id_cannot_move(self):
        class WrongSelector:
            def select(self, *args):
                return Selection(candidate_ids=["invented"])

        app, sim = self.make(selector=WrongSelector())
        self.assertEqual(app.run("随便")["status"], "failed")
        self.assertEqual(sim.steps, 0)

    def test_new_task_clears_previous_clarification(self):
        app, sim = self.make()
        app.run("红色那个")
        app.reset()
        self.assertEqual(app.history, [])
        self.assertEqual(self.records()[0]["status"], "stopped")

    def test_target_removed_during_final_model_call(self):
        app, sim = self.make()
        real_detector = app.detector

        class SlowDetector:
            calls = 0

            def detect(inner, frame):
                inner.calls += 1
                result = real_detector.detect(frame)
                if inner.calls == 2:
                    self.clock.sleep(2)
                    sim.objects.clear()
                return result

        app.detector = SlowDetector()
        state = app.run("蓝色盒子")
        self.assertEqual(state["status"], "failed")
        self.assertIn("核对返回后", state["error"])

    def test_stop_during_transcription_cannot_start_old_request(self):
        entered, release = threading.Event(), threading.Event()

        class ASR:
            usage = {}

            def transcribe(self, wav):
                entered.set()
                release.wait(2)
                return "蓝色盒子"

        app, sim = self.make()
        app.submit_audio(b"test", ASR())
        self.assertTrue(entered.wait(2))
        app.stop()
        with self.assertRaises(RuntimeError):
            app.submit("绿色圆形")
        release.set()
        app.worker.join(2)
        self.assertEqual(app.snapshot()["status"], "stopped")
        self.assertEqual(sim.steps, 0)
        self.assertEqual(self.records()[0]["status"], "stopped")

    def test_stop_failure_is_recorded_as_failure(self):
        app, sim = self.make()
        app.run("红色那个")

        def bad_stop():
            raise RuntimeError("反馈异常")

        sim.stop = bad_stop
        app.stop()
        self.assertEqual(app.snapshot()["status"], "failed")
        self.assertIn("未确认实机停止", app.snapshot()["error"])
        self.assertEqual(self.records()[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
