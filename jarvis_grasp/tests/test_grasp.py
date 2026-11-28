import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request

import cv2
import numpy as np

from jarvis.controller import Assistant
from jarvis.grasp import estimate_grasp_pose, ray_plane_intersection, validate_grasp_config
from jarvis.hardware import HardwareMotion
from jarvis.models import Candidate, Frame
from jarvis.perception import ArUcoDetector, ColorShapeDetector, LocalSelector, draw_detections
from jarvis.providers import ModelClient, ModelSelector
from jarvis.server import create_server
from jarvis.simulation import Simulation


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def _config():
    return json.loads((Path(__file__).parents[1] / "config/default.json").read_text(encoding="utf-8"))


class GraspPerceptionTests(unittest.TestCase):
    def test_detects_orientation_and_new_shapes(self):
        image = np.full((480, 640, 3), 240, np.uint8)
        # Elongated blue rectangle (bar/block)
        cv2.rectangle(image, (100, 140), (220, 180), (220, 80, 30), -1)
        # Yellow triangle
        tri = np.array([[420, 120], [380, 200], [460, 200]], dtype=np.int32)
        cv2.fillConvexPoly(image, tri, (30, 210, 230))
        # Red circle
        cv2.circle(image, (300, 340), 28, (40, 40, 220), -1)
        found = ColorShapeDetector().detect(Frame(image, 1, 0.0))
        shapes = {(c.color, c.shape) for c in found}
        self.assertIn(("blue", "rectangle"), shapes)
        self.assertIn(("yellow", "triangle"), shapes)
        self.assertIn(("red", "circle"), shapes)
        self.assertTrue(all(c.area_px > 180 and 0.5 <= c.confidence <= 1.0 for c in found))

    def test_aruco_pnp_3d_position_and_angle(self):
        K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((5, 1), dtype=np.float64)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 3, 120)
        image = np.full((480, 640, 3), 255, np.uint8)
        image[180:300, 260:380] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        found = ArUcoDetector(camera_matrix=K, dist_coeffs=dist, marker_mm=40.0).detect(Frame(image, 1, 0.0))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].marker_id, 3)
        self.assertIsNotNone(found[0].position_3d_mm)
        # Expected depth: f * marker_mm / marker_px = 600 * 40 / 120 = 200 mm
        self.assertAlmostEqual(found[0].position_3d_mm[2], 200.0, delta=10.0)
        self.assertAlmostEqual(found[0].angle_deg, 0.0, delta=2.0)

    def test_aruco_empty_frame_clahe_fallback_returns_empty_list(self):
        blank = np.full((240, 320, 3), 200, np.uint8)
        self.assertEqual(ArUcoDetector().detect(Frame(blank, 1, 0.0)), [])

    def test_selector_grasp_release_and_clarification_preserves_intent(self):
        selector = LocalSelector()
        candidates = [
            Candidate("object_1", "红色圆形", "red", "circle", (0.2, 0.3, 0.08, 0.1)),
            Candidate("object_2", "红色方形", "red", "square", (0.6, 0.3, 0.08, 0.1)),
            Candidate("object_3", "蓝色方形", "blue", "square", (0.7, 0.6, 0.08, 0.1)),
        ]
        # Ambiguous grasp request asks a grasp clarification question
        first = selector.select("抓取红色那个", candidates)
        self.assertEqual(first.intent, "grasp")
        self.assertEqual(len(first.candidate_ids), 2)
        self.assertIn("抓", first.question)
        # Clarification answer inherits "grasp" intent
        second = selector.select(
            "圆的",
            candidates,
            [
                {"role": "user", "content": "抓取红色那个"},
                {"role": "assistant", "content": first.question},
            ],
        )
        self.assertEqual(second.intent, "grasp")
        self.assertEqual(second.candidate_ids, ["object_1"])
        self.assertFalse(second.question)

        # Grasp + place direction does not confuse spatial filtering
        place_sel = selector.select("把蓝色盒子抓起来放到左边", candidates)
        self.assertEqual(place_sel.intent, "grasp")
        self.assertEqual(place_sel.candidate_ids, ["object_3"])
        self.assertEqual(place_sel.place_direction, "left")

        # Release intent
        rel = selector.select("放下", candidates)
        self.assertEqual(rel.intent, "release")

    def test_draw_detections_does_not_mutate_input(self):
        image = np.full((240, 320, 3), 230, np.uint8)
        original = image.copy()
        cand = Candidate("object_1", "蓝色方形", "blue", "square", (0.2, 0.2, 0.15, 0.15), angle_deg=15.0)
        rendered = draw_detections(image, [cand], selected_id="object_1", grasp_stage="closing")
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(rendered, original))

    def test_real_world_detector_ignores_dark_trim_and_keeps_true_blue(self):
        hsv = np.full((240, 320, 3), (32, 25, 195), np.uint8)  # wood desk background
        hsv[140:230, 10:120] = (107, 180, 42)  # dark cool-lit black metal trim
        hsv[20:110, 10:120] = (90, 52, 130)    # cyan-green LED reflection on matte black desk pad
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertEqual(ColorShapeDetector(real_world=True).detect(Frame(bgr, 1, 0.0)), [])
        hsv[60:120, 180:240] = (112, 165, 145)  # real blue cube
        bgr2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        found = ColorShapeDetector(real_world=True).detect(Frame(bgr2, 2, 0.1))
        self.assertEqual([(c.color, c.shape) for c in found], [("blue", "square")])
        hsv[130:190, 180:240] = (66, 54, 232)   # bright auto-exposed green pencil case
        bgr3 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        found3 = ColorShapeDetector(real_world=True).detect(Frame(bgr3, 3, 0.2))
        self.assertIn(("green", "square"), [(c.color, c.shape) for c in found3])


class GraspGeometryTests(unittest.TestCase):
    def test_ray_plane_intersection_exact_recovery(self):
        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        # Camera at base [200, 0, 220] looking straight down (-Z base = +Z cam)
        T_bc = np.array(
            [
                [0.0, -1.0, 0.0, 200.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 220.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Ground-truth point on z=20 plane: [220, -40, 20] -> in cam frame: [40, -20, 200]
        # Pixel: u = 320 + 800*(40/200) = 480, v = 240 + 800*(-20/200) = 160
        pt = ray_plane_intersection((480.0, 160.0), K, None, T_bc, plane_z_mm=20.0)
        self.assertAlmostEqual(pt[0], 220.0, places=2)
        self.assertAlmostEqual(pt[1], -40.0, places=2)
        self.assertAlmostEqual(pt[2], 20.0, places=2)

    def test_estimate_grasp_pose_builds_valid_plan(self):
        cfg = validate_grasp_config(_config(), mode="simulation")
        cand = Candidate("object_1", "蓝色方形", "blue", "square", (0.45, 0.45, 0.1, 0.1), angle_deg=12.5)
        plan = estimate_grasp_pose(cand, (480, 640, 3), cfg, place_direction="left")
        self.assertEqual(plan.target_id, "object_1")
        self.assertEqual(plan.angle_deg, 12.5)
        self.assertEqual(plan.place_offset_mm, (0.0, 45.0, 0.0))

    def test_validate_grasp_rejects_below_table_and_unverified_hardware(self):
        cfg = _config()
        cfg["grasp"]["grasp_z_mm"] = 5.0
        cfg["grasp"]["table_z_mm"] = 10.0
        with self.assertRaisesRegex(ValueError, "桌面安全平面"):
            validate_grasp_config(cfg, mode="simulation")
        with self.assertRaisesRegex(ValueError, "实机抓取未解锁"):
            validate_grasp_config(_config(), mode="hardware")


class GraspControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = _config()
        self.clock = Clock()

    def make(self, scenario="default"):
        sim = Simulation(scenario, clock=self.clock)
        app = Assistant(
            sim,
            sim,
            ColorShapeDetector(),
            LocalSelector(),
            self.config,
            self.tmp.name,
            clock=self.clock,
            sleep=self.clock.sleep,
        )
        return app, sim

    def test_full_recognize_grasp_and_release_cycle(self):
        app, sim = self.make()
        state = app.run("抓取蓝色盒子")
        self.assertEqual(state["status"], "completed", state)
        self.assertEqual(state["action"], "grasp")
        self.assertEqual(state["grasp_stage"], "lifted")
        self.assertIsNotNone(state["held_object"])
        self.assertEqual(state["held_object"]["color"], "blue")
        self.assertTrue(sim.gripper_state()["holding"])
        self.assertEqual(len(sim.objects), 3)
        self.assertTrue(state["metrics"]["grasp_succeeded"])

        # Attempting a second grasp while holding must be rejected cleanly
        second = app.run("抓取绿色圆形")
        self.assertEqual(second["status"], "failed")
        self.assertIn("已夹持", second["error"])

        # Release places the object back on the table
        rel = app.run("放下")
        self.assertEqual(rel["status"], "completed", rel)
        self.assertIsNone(rel["held_object"])
        self.assertFalse(sim.gripper_state()["holding"])
        self.assertEqual(len(sim.objects), 4)

    def test_ambiguous_grasp_clarifies_then_picks_correct_object(self):
        app, sim = self.make()
        first = app.run("抓取红色那个")
        self.assertEqual(first["status"], "clarifying")
        self.assertEqual(sim.steps, 0)
        self.assertIsNone(sim.held_object)
        second = app.run("圆的")
        self.assertEqual(second["status"], "completed", second)
        self.assertEqual(second["held_object"]["color"], "red")
        self.assertEqual(second["held_object"]["shape"], "circle")
        self.assertEqual(second["metrics"]["clarification_count"], 1)

    def test_grasp_and_place_to_left(self):
        app, sim = self.make()
        state = app.run("把绿色球抓起来放到左边")
        self.assertEqual(state["status"], "completed", state)
        self.assertEqual(state["grasp_stage"], "placed")
        self.assertIsNone(state["held_object"])
        self.assertFalse(sim.gripper_state()["holding"])
        self.assertEqual(len(sim.objects), 4)

    def test_grasp_miss_fails_honestly_and_retreats(self):
        app, sim = self.make("grasp_miss")
        state = app.run("抓取蓝色盒子")
        self.assertEqual(state["status"], "failed")
        self.assertIn("空抓", state["error"])
        self.assertIsNone(state["held_object"])
        self.assertGreaterEqual(sim.height_mm, 130.0)
        self.assertFalse(state["metrics"]["grasp_succeeded"])

    def test_grasp_slip_detected_after_lift(self):
        app, sim = self.make("grasp_slip")
        state = app.run("抓取蓝色盒子")
        self.assertEqual(state["status"], "failed")
        self.assertIn("滑落", state["error"])
        self.assertFalse(sim.gripper_state()["holding"])

    def test_release_when_empty_does_not_move(self):
        app, sim = self.make()
        state = app.run("放下")
        self.assertEqual(state["status"], "idle")
        self.assertEqual(sim.grasp_moves, 0)

    def test_stop_mid_grasp_halts_immediately(self):
        app, sim = self.make()
        entered = threading.Event()
        release = threading.Event()
        orig_set_gripper = sim.set_gripper

        def slow_gripper(*args, **kwargs):
            entered.set()
            release.wait(2)
            return orig_set_gripper(*args, **kwargs)

        sim.set_gripper = slow_gripper
        app.submit("抓取蓝色盒子")
        self.assertTrue(entered.wait(2))
        app.stop()
        release.set()
        app.worker.join(2)
        self.assertEqual(app.snapshot()["status"], "stopped")
        self.assertIsNone(sim.held_object)


class ModelGraspIntentTests(unittest.TestCase):
    def test_model_selector_accepts_grasp_and_release_intents(self):
        def make_fake(payload):
            function = SimpleNamespace(name="select_target", arguments=json.dumps(payload))
            resp = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(tool_calls=[SimpleNamespace(function=function)]))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))

        candidates = [Candidate("object_1", "蓝色方形", "blue", "square", (0.2, 0.2, 0.1, 0.1))]
        sel = ModelSelector(
            ModelClient(
                make_fake({"intent": "grasp", "candidate_ids": ["object_1"], "question": "", "reply": "", "place_direction": "left"}),
                model="test-model",
            )
        ).select("抓蓝色盒子放左边", candidates)
        self.assertEqual(sel.intent, "grasp")
        self.assertEqual(sel.place_direction, "left")

        rel = ModelSelector(
            ModelClient(
                make_fake({"intent": "release", "candidate_ids": [], "question": "", "reply": ""}),
                model="test-model",
            )
        ).select("放下", candidates)
        self.assertEqual(rel.intent, "release")


class HardwareGraspSafetyTests(unittest.TestCase):
    def _hw_config(self, *, grasp_verified=True):
        return {
            "enabled": True,
            "max_step_mm": 3.0,
            "max_travel_mm": 60.0,
            "command_timeout_s": 2.0,
            "calibration": {
                "verified": True,
                "image_to_base_mm": [[10.0, 0.0], [0.0, -10.0], [0.0, 0.0]],
                "workspace_mm": [[100.0, 260.0], [-120.0, 120.0], [20.0, 220.0]],
            },
            "grasp": {
                "verified": grasp_verified,
                "table_z_mm": 25.0,
                "grasp_z_mm": 40.0,
                "pre_grasp_z_mm": 80.0,
                "lift_z_mm": 130.0,
                "camera_to_gripper_mm": [0.0, 0.0, 0.0],
                "max_grasp_step_mm": 3.0,
                "max_grasp_travel_mm": 40.0,
                "gripper_open_raw": 430,
                "gripper_close_raw": 278,
                "gripper_held_range_raw": [298, 415],
            },
        }

    def test_hardware_grasp_gated_and_table_floor_enforced(self):
        class FakeArm:
            def __init__(self, config, bus):
                self.xyz = np.array([180.0, 0.0, 30.0])
                self.joints = np.array([500, 600, 400, 500, 500, 430])
                self.cancel = threading.Event()
                self.last_move = None
                self.model = SimpleNamespace(forward=lambda pos: self.xyz.copy())

            def positions(self):
                return self.joints.copy()

            def stop(self):
                self.cancel.set()

        class FakeRunner:
            def __init__(self, arm):
                self.arm = arm
                self.thread = None
                self.error = None
                self.calls = []

            @property
            def busy(self):
                return self.thread is not None and self.thread.is_alive()

            def submit(self, command):
                self.arm.cancel.clear()
                self.calls.append(command)
                def run():
                    if command[0] == "move":
                        self.arm.xyz = self.arm.xyz + command[1]
                        self.arm.last_move = {"projected": False}
                    elif command[0] == "pose_raw":
                        self.arm.joints = np.asarray(command[1], dtype=int)
                self.thread = threading.Thread(target=run)
                self.thread.start()
                return True

            def wait(self):
                self.thread.join()
                if self.error:
                    raise self.error

            def stop(self):
                self.arm.stop()

        class FakeBus:
            def __init__(self, *args, **kwargs):
                self.lock = threading.RLock()
                self.pos = 430
                self.load = 0
                self.finished = False
                self.writes = []

            def move_position(self, sid, target, dur, speed):
                # Simulate jaw closing and hitting an object at raw 348 with load 380
                self.pos = 348
                self.load = 380

            def heartbeat(self, sid):
                pass

            def read_position(self, sid):
                return self.pos

            def _read_u16(self, sid, addr):
                return self.load

            def _write(self, sid, addr, vals):
                self.writes.append((sid, addr, vals))

            def finish_position(self, sid):
                self.finished = True

            def close(self):
                pass

        with patch("vendor.scs215.advanced.Arm", FakeArm), \
             patch("vendor.scs215.advanced.CommandRunner", FakeRunner), \
             patch("vendor.scs215.servo_bus.ServoBus", FakeBus):
            # Unverified grasp refuses move_delta & set_gripper
            m_locked = HardwareMotion(self._hw_config(grasp_verified=False))
            self.addCleanup(m_locked.close)
            with self.assertRaisesRegex(ValueError, "实机抓取未解锁"):
                m_locked.move_delta([0, 0, -2])
            with self.assertRaisesRegex(ValueError, "实机抓取未解锁"):
                m_locked.set_gripper(430)

            # Verified grasp: sub-steps safely, rejects descent below table_z_mm, detects contact grip
            m = HardwareMotion(self._hw_config(grasp_verified=True))
            self.addCleanup(m.close)
            m.move_delta([0, 0, 6.0])  # splits into 2 x 3.0mm steps -> z = 36
            self.assertEqual(len(m.runner.calls), 2)
            self.assertAlmostEqual(m.position_mm()[2], 36.0)

            fb = m.set_gripper(278, expect_object=True, held_range=(298, 415), max_load=350)
            self.assertTrue(fb["gripped"])
            self.assertEqual(fb["raw"], 348)
            self.assertTrue(m.bus.finished)

            # Attempting to go below table_z_mm (25 mm) fails and latches stop
            with self.assertRaisesRegex(ValueError, "桌面安全平面"):
                m.move_delta([0, 0, -15.0])
            self.assertTrue(m._stopped.is_set())


class WebServerTests(unittest.TestCase):
    def test_web_grasp_and_frame_endpoints(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        clock = Clock()
        sim = Simulation("default", clock=clock)
        app = Assistant(sim, sim, ColorShapeDetector(), LocalSelector(), _config(), tmp.name, clock=clock, sleep=clock.sleep)
        server = create_server(app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), app.close()))
        base = f"http://127.0.0.1:{server.server_port}"

        req = urllib.request.Request(
            base + "/api/message",
            data=json.dumps({"text": "抓取蓝色盒子"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
        if app.worker:
            app.worker.join(timeout=5)
        with urllib.request.urlopen(base + "/api/state", timeout=5) as resp:
            state = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["held_object"]["color"], "blue")
        with urllib.request.urlopen(base + "/api/frame.jpg", timeout=5) as resp:
            jpg = resp.read()
        self.assertTrue(jpg.startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
