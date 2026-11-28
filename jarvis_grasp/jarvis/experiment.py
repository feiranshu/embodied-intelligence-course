import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import time

from .controller import Assistant
from .evaluate import summarize
from .perception import ColorShapeDetector, LocalSelector
from .simulation import Simulation


class SimulationClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def run_experiment(output, seed=23):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).resolve().parents[1] / "config/default.json").read_text(encoding="utf-8"))
    cases = []
    for i in range(10):
        color, shape, command = [
            ("blue", "square", "帮我看看蓝色盒子"),
            ("green", "circle", "绿色圆的那个"),
        ][i % 2]
        cases.append((f"find_{i:02d}", i, command, None, color, shape, "clarify", "default"))
    for i in range(10):
        color, shape, command = [
            ("blue", "square", "抓取蓝色盒子"),
            ("green", "circle", "把绿色圆形抓起来放到左边"),
        ][i % 2]
        cases.append((f"grasp_{i:02d}", 50 + i, command, None, color, shape, "clarify", "default"))
    for i in range(10):
        shape = "circle" if i % 2 == 0 else "square"
        for strategy in ("direct", "clarify"):
            cases.append(
                (
                    f"ambiguity_grasp_{i:02d}_{strategy}",
                    100 + i,
                    "抓取红色那个",
                    "圆的" if shape == "circle" else "方的",
                    "red",
                    shape,
                    strategy,
                    "default",
                )
            )
    for i in range(5):
        cases.append((f"missing_{i:02d}", 200 + i, "抓取紫色那个", None, "purple", "circle", "clarify", "default"))
    for scenario in ("lost", "stale", "stalled", "grasp_miss", "grasp_slip"):
        command = "抓取蓝色盒子" if scenario.startswith("grasp_") else "蓝色盒子"
        cases.append((scenario, 300, command, None, "blue", "square", "clarify", scenario))

    random.Random(seed).shuffle(cases)
    records = []
    for case_id, scene_seed, command, answer, color, shape, strategy, scenario in cases:
        clock = SimulationClock()
        sim = Simulation(scenario, clock=clock, seed=scene_seed)
        detector = ColorShapeDetector()
        candidates = detector.detect(sim.read())
        expected = next(
            (candidate for candidate in candidates if (candidate.color, candidate.shape) == (color, shape)),
            None,
        )
        app = Assistant(
            sim,
            sim,
            detector,
            LocalSelector(),
            config,
            output / case_id,
            strategy=strategy,
            clock=clock,
            sleep=clock.sleep,
        )
        wall_start = time.monotonic()
        try:
            state = app.run(command)
            if state["status"] == "clarifying" and answer:
                clock.sleep(2.0)
                state = app.run(answer)
            if app.pending:
                app.stop()
            record = dict(app.snapshot()["metrics"])
            record.update(
                case_id=case_id,
                scene_seed=scene_seed,
                expected_id=expected.id if expected else None,
                selection_correct=(record.get("selected_id") == expected.id) if expected else None,
                clock="virtual",
                simulated_answer_delay_s=2.0 if strategy == "clarify" and answer else 0.0,
                wall_duration_s=round(time.monotonic() - wall_start, 5),
                expected_status="failed" if scenario != "default" else "not_found" if expected is None else "completed",
            )
            record["task_correct"] = (
                record["status"] == record["expected_status"] and record["selection_correct"] is not False
            )
            records.append(record)
        finally:
            app.close()
    (output / "trials.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    report = {
        "note": "合成图像与简化 2.5D 夹爪模拟；虚拟时钟，不是实机抓取成功率或真实交互延迟。",
        "seed": seed,
        "summary": summarize(records),
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="离线识别与抓取评测：10居中、10抓取、10对模糊抓取对照、5不存在、5故障")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / ("experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    run_experiment(args.output, args.seed)
    print(f"识别与抓取合成场景评测完成：{args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
