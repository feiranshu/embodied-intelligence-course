import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

from jarvis.models import Candidate, Frame
from jarvis.providers import ModelClient, ModelSelector, VisionDetector


class FakeClient:
    def __init__(self, arguments, tool_name=None, usage=True):
        self.arguments = arguments
        self.tool_name = tool_name
        self.with_usage = usage
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.transcribe))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        name = self.tool_name or kwargs["tool_choice"]["function"]["name"]
        function = SimpleNamespace(name=name, arguments=self.arguments if isinstance(self.arguments, str)
                                   else json.dumps(self.arguments))
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
            tool_calls=[SimpleNamespace(function=function)]))], usage=SimpleNamespace(
                prompt_tokens=12, completion_tokens=8) if self.with_usage else None)

    def transcribe(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text="看看红色的")


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [Candidate("object_1", "红色圆形", "red", "circle", (0.2, 0.2, 0.1, 0.1))]
        self.frame = Frame(np.full((120, 160, 3), 240, np.uint8), 1, 0)

    def test_valid_selection_and_accounting(self):
        fake = FakeClient({"intent": "find", "candidate_ids": ["object_1"], "question": "", "reply": ""})
        client = ModelClient(fake, model="test-model")
        result = ModelSelector(client).select("红色的", self.candidates)
        self.assertEqual(result.candidate_ids, ["object_1"])
        self.assertEqual(client.usage["api_calls"], 1)
        self.assertEqual(client.usage["input_tokens"], 12)
        self.assertIsNone(client.usage["actual_cost"])

    def test_unknown_candidate_rejected(self):
        client = ModelClient(FakeClient({"intent": "find", "candidate_ids": ["invented"],
                                        "question": "", "reply": ""}), model="test-model")
        with self.assertRaises(ValueError):
            ModelSelector(client).select("找它", self.candidates)

    def test_raw_motion_tool_rejected(self):
        client = ModelClient(FakeClient({}, tool_name="set_joint_angles"), model="test-model")
        with self.assertRaises(ValueError):
            ModelSelector(client).select("找它", self.candidates)

    def test_invalid_json_rejected(self):
        client = ModelClient(FakeClient("not-json"), model="test-model")
        with self.assertRaises(ValueError):
            ModelSelector(client).select("找它", self.candidates)

    def test_valid_vision_includes_actual_image(self):
        fake = FakeClient({"objects": [{"label": "蓝色方形", "color": "blue", "shape": "square",
                                        "bbox": [0.2, 0.3, 0.1, 0.2]}]})
        found = VisionDetector(ModelClient(fake, model="test-model")).detect(self.frame)
        self.assertEqual(found[0].color, "blue")
        self.assertTrue(fake.requests[0]["messages"][1]["content"][0]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"))

    def test_bad_vision_boxes_rejected(self):
        for bbox in ([0.9, 0.2, 0.4, 0.1], [True, 0, 0.2, 0.2], [0.1, 0.1, -0.2, 0.2],
                     [0.1, 0.1, float("nan"), 0.2]):
            with self.subTest(bbox=bbox):
                fake = FakeClient({"objects": [{"label": "物体", "color": "blue", "shape": "square", "bbox": bbox}]})
                with self.assertRaises(ValueError):
                    VisionDetector(ModelClient(fake, model="test-model")).detect(self.frame)

    def test_multiple_candidates_require_clarification(self):
        candidates = self.candidates + [Candidate("object_2", "红色方形", "red", "square", (0.5, 0.5, 0.1, 0.1))]
        client = ModelClient(FakeClient({"intent": "find", "candidate_ids": ["object_1", "object_2"],
                                        "question": "", "reply": ""}), model="test-model")
        with self.assertRaises(ValueError):
            ModelSelector(client).select("红色的", candidates)

    def test_usage_missing_is_unknown(self):
        client = ModelClient(FakeClient({"objects": []}, usage=False), model="test-model")
        VisionDetector(client).detect(self.frame)
        self.assertFalse(client.usage["tokens_fully_reported"])
        self.assertIsNone(client.calls[0]["input_tokens"])

    def test_asr_wav_without_device(self):
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 16000)
        client = ModelClient(FakeClient({}), model="test-model")
        with patch.dict("os.environ", {"JARVIS_ASR_MODEL": "test-asr"}):
            self.assertEqual(client.transcribe(stream.getvalue()), "看看红色的")
        self.assertEqual(client.calls[0]["operation"], "transcribe")


if __name__ == "__main__":
    unittest.main()
