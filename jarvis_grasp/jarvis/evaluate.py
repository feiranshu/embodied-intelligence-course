"""Summarize recorded attempts without mixing simulation and physical results."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


OUTCOMES = ("completed", "not_found", "failed", "stopped")


def _statistics(records):
    total = len(records)
    outcomes = Counter(record["status"] for record in records)
    correct = sum(record.get("selection_correct") is True for record in records)
    incorrect = sum(record.get("selection_correct") is False for record in records)
    unknown = total - correct - incorrect
    durations = [record["duration_s"] for record in records if record.get("duration_s") is not None]
    costs = [record["cost"] for record in records if record.get("cost") is not None]
    grasp_records = [record for record in records if record.get("action") == "grasp"]
    grasp_succeeded = sum(record.get("grasp_succeeded") is True for record in grasp_records)
    return {
        "total": total,
        **{status: outcomes[status] for status in OUTCOMES},
        "completion_rate": outcomes["completed"] / total if total else None,
        "selection_correct": correct,
        "selection_incorrect": incorrect,
        "selection_unknown": unknown,
        "selection_accuracy_known": correct / (correct + incorrect) if correct + incorrect else None,
        "verified_completed": sum(
            record["status"] == "completed" and record.get("selection_correct") is True for record in records
        ),
        "grasp_attempts": len(grasp_records),
        "grasp_succeeded": grasp_succeeded,
        "grasp_success_rate": grasp_succeeded / len(grasp_records) if grasp_records else None,
        "clarified_attempts": sum(record.get("clarification_count", 0) > 0 for record in records),
        "clarification_count": sum(record.get("clarification_count", 0) for record in records),
        "mean_duration_s": sum(durations) / len(durations) if durations else None,
        "duration_unknown": total - len(durations),
        "api_calls": sum(record.get("api_calls", 0) for record in records),
        "input_tokens": sum(record.get("input_tokens", 0) for record in records),
        "output_tokens": sum(record.get("output_tokens", 0) for record in records),
        "cost": sum(costs) if len(costs) == total and total else None,
        "known_cost_subtotal": sum(costs),
        "cost_unknown": total - len(costs),
        "failure_reasons": dict(
            Counter(
                record.get("reason") or "unspecified"
                for record in records
                if record["status"] in {"failed", "stopped"}
            )
        ),
    }


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        if record.get("status") not in OUTCOMES:
            raise ValueError(f"非终态任务记录：{record.get('task_id', '?')} / {record.get('status')}")
        key = (
            record.get("mode", "unknown"),
            record.get("detector", "unknown"),
            record.get("strategy", "unknown"),
        )
        groups[key].append(record)
    return {
        "record_count": sum(len(group) for group in groups.values()),
        "groups": [
            {
                "mode": key[0],
                "detector": key[1],
                "strategy": key[2],
                "attempts": _statistics(group),
                "first_attempts": _statistics([record for record in group if not record.get("retry_of")]),
                "retries": _statistics([record for record in group if record.get("retry_of")]),
            }
            for key, group in sorted(groups.items())
        ],
        "note": (
            "completion_rate 仅表示软件报告完成；选择正确性以人工或场景真值为准。"
            "未知正确性不计为正确。不同 mode、detector、strategy 分开统计；成本单位沿用记录。"
        ),
    }


def read_records(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是一个任务记录对象")
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description="汇总 JARVIS 识别与抓取任务记录")
    parser.add_argument("path", type=Path, help="任务 JSONL 文件")
    parser.add_argument("--output", type=Path, help="另存 JSON 汇总")
    args = parser.parse_args()
    try:
        result = summarize(read_records(args.path))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
