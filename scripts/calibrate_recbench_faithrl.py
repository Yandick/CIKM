from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate FaithRL calibration constants from dumped RecBench rollout/validation JSONL."
    )
    parser.add_argument("input", type=Path, help="JSONL dumped by verl validation/rollout logging.")
    args = parser.parse_args()

    correctness: list[float] = []
    unfaithfulness: list[float] = []
    parse_success: list[float] = []

    for row in iter_jsonl(args.input):
        correct = as_float(row.get("correctness"))
        if correct is None:
            ndcg1 = as_float(row.get("ndcg@1"))
            if ndcg1 is not None:
                correct = float(ndcg1 > 0.0)
        if correct is None:
            recommendation = as_float(row.get("recommendation"))
            reward_k = as_float(row.get("reward_k"))
            if recommendation is not None and (reward_k is None or reward_k <= 1.0):
                correct = float(recommendation > 0.0)
        if correct is not None:
            correctness.append(correct)

        faithful = as_float(row.get("faithful_binary"))
        if faithful is None:
            faithful_score = as_float(row.get("faithfulness"))
            if faithful_score is not None:
                faithful = float(faithful_score >= 1.0)
        if faithful is None:
            rationale = as_float(row.get("rationale"))
            evidence = as_float(row.get("evidence"))
            fmt = as_float(row.get("format"))
            if rationale is not None and evidence is not None and fmt is not None:
                faithful = float(fmt >= 1.0 and evidence >= 1.0 and rationale >= 1.0)
        if faithful is not None:
            unfaithfulness.append(1.0 - faithful)

        parsed = as_float(row.get("parse_success"))
        if parsed is not None:
            parse_success.append(parsed)

    if not correctness or not unfaithfulness:
        raise SystemExit(
            "No usable correctness/faithfulness fields found. Run validation with recbench_json.py first."
        )

    baseline_correct_rate = sum(correctness) / len(correctness)
    baseline_unfaithful_rate = sum(unfaithfulness) / len(unfaithfulness)
    parse_rate = sum(parse_success) / len(parse_success) if parse_success else None

    print(f"rows_with_correctness={len(correctness)}")
    print(f"rows_with_faithfulness={len(unfaithfulness)}")
    if parse_rate is not None:
        print(f"parse_success_rate={parse_rate:.6f}")
    print(f"BASELINE_CORRECT_RATE={baseline_correct_rate:.6f}")
    print(f"BASELINE_UNFAITHFUL_RATE={baseline_unfaithful_rate:.6f}")


if __name__ == "__main__":
    main()
