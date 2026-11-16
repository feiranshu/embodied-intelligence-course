import tempfile
import unittest
from pathlib import Path

from jarvis.evaluate import read_records, summarize


def record(task, status="completed", **changes):
    return {"task_id": task, "status": status, "mode": "simulation", "detector": "color",
            "strategy": "clarify", "duration_s": 2, "api_calls": 0, "input_tokens": 0,
            "output_tokens": 0, "cost": 0, **changes}


class EvaluationTests(unittest.TestCase):
    def test_failures_remain_in_denominator_and_retry_is_separate(self):
        result = summarize([record("a", "failed", reason="lost"),
                            record("b", retry_of="a", selection_correct=True),
                            record("c", "not_found"), record("d", "stopped", reason="user_stop")])
        group = result["groups"][0]
        self.assertEqual(group["attempts"]["total"], 4)
        self.assertEqual(group["attempts"]["completion_rate"], 0.25)
        self.assertEqual(group["first_attempts"]["completed"], 0)
        self.assertEqual(group["retries"]["completed"], 1)
        self.assertEqual(group["attempts"]["failure_reasons"], {"lost": 1, "user_stop": 1})

    def test_unknown_truth_and_cost_are_not_success_or_free(self):
        stats = summarize([record("a", selection_correct=True, cost=0.1),
                           record("b", selection_correct=False, cost=None),
                           record("c")])["groups"][0]["attempts"]
        self.assertEqual(stats["selection_unknown"], 1)
        self.assertEqual(stats["selection_accuracy_known"], 0.5)
        self.assertEqual(stats["verified_completed"], 1)
        self.assertIsNone(stats["cost"])
        self.assertAlmostEqual(stats["known_cost_subtotal"], 0.1)
        self.assertEqual(stats["cost_unknown"], 1)

    def test_mode_detector_and_strategy_are_separate(self):
        records = [record("a"), record("b", mode="hardware"), record("c", detector="aruco"),
                   record("d", strategy="direct")]
        result = summarize(records)
        self.assertEqual(result["record_count"], 4)
        self.assertEqual(len(result["groups"]), 4)
        self.assertEqual(sum(group["attempts"]["total"] for group in result["groups"]), 4)

    def test_missing_time_not_reported_as_zero(self):
        stats = summarize([record("a", duration_s=None)])["groups"][0]["attempts"]
        self.assertIsNone(stats["mean_duration_s"])
        self.assertEqual(stats["duration_unknown"], 1)

    def test_rejects_incomplete_records(self):
        with self.assertRaises(ValueError):
            summarize([record("a", "tracking")])

    def test_empty_input(self):
        self.assertEqual(summarize([])["groups"], [])

    def test_reader_reports_malformed_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text('{"status": "completed"}\nnot json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "第 2 行"):
                read_records(path)


if __name__ == "__main__":
    unittest.main()
