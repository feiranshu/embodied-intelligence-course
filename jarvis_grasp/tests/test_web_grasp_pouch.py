import json
import threading
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

from scripts.web_grasp_pouch import GraspWebApp, create_server


class FakeCameraHub:
    def __init__(self):
        self.ready = True
        self.device_name = "GENERAL WEBCAM (Mock)"
        self.error = ""
        self.frame_count = 42
        self.labels = []
        self._frame = np.full((240, 320, 3), 120, dtype=np.uint8)

    def set_label(self, label):
        self.labels.append(label)

    def start_recording(self, mp4_path):
        self._mp4 = mp4_path

    def stop_recording(self):
        return None

    def preview_jpeg(self):
        ok, buf = cv2.imencode(".jpg", self._frame)
        assert ok
        return buf.tobytes()

    def close(self):
        pass


class WebGraspPouchTests(unittest.TestCase):
    def test_start_conflict_success_and_stop(self):
        cam = FakeCameraHub()
        gate = threading.Event()
        entered = threading.Event()

        def fake_runner(*, recorder, log_path, cancel, target_color, grasp_z_mm, release_after):
            self.assertEqual(target_color, "green")
            self.assertAlmostEqual(grasp_z_mm, 26.0)
            recorder.set_label("pouch: descend")
            entered.set()
            gate.wait(timeout=3.0)
            if cancel.is_set():
                raise InterruptedError("stopped")
            return {
                "close_feedback": {"raw": 335, "max_load": 480, "gripped": True},
                "after_lift": {"gripper_raw": 331, "gripper_load": 195, "grasp_verified": True},
                "torque_off_verified": True,
            }

        app = GraspWebApp(camera_hub=cam, grasp_z_mm=26.0, runner=fake_runner)
        server = create_server(app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), app.close()))
        base = f"http://127.0.0.1:{server.server_port}"

        # 1. HTML & JPEG & idle state
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            html = r.read().decode("utf-8")
            self.assertIn("开始抓取绿色笔盒", html)
        with urllib.request.urlopen(base + "/api/frame.jpg", timeout=5) as r:
            self.assertTrue(r.read().startswith(b"\xff\xd8"))
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            st = json.loads(r.read().decode("utf-8"))
            self.assertEqual(st["status"], "idle")
            self.assertTrue(st["camera_ready"])

        # 2. Start grasp -> duplicate start returns 409 -> release gate -> completes as success
        req = urllib.request.Request(
            base + "/api/start",
            data=json.dumps({"release_after": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
        self.assertTrue(entered.wait(timeout=3.0))
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 409)
        gate.set()
        app._worker.join(timeout=3.0)

        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            st2 = json.loads(r.read().decode("utf-8"))
            self.assertEqual(st2["status"], "success")
            self.assertFalse(st2["busy"])
            self.assertTrue(st2["last_result"]["after_lift"]["grasp_verified"])

        # 3. Emergency stop during second run
        gate.clear()
        entered.clear()
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
        self.assertTrue(entered.wait(timeout=3.0))
        stop_req = urllib.request.Request(base + "/api/stop", data=b"", method="POST")
        with urllib.request.urlopen(stop_req, timeout=5) as r:
            self.assertEqual(r.status, 200)
        gate.set()
        app._worker.join(timeout=3.0)
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            st3 = json.loads(r.read().decode("utf-8"))
            self.assertEqual(st3["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
