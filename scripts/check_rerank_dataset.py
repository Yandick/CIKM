from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check candidate-reranking JSONL quality.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-candidates", type=int, default=20)
    args = parser.parse_args()

    count = 0
    bad_k = 0
    bad_target = 0
    missing_title = 0
    total_items = 0
    history_lens: list[int] = []
    target_positions: Counter[int] = Counter()

    with args.path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            count += 1
            history = record["history"]
            candidates = record["candidates"]
            target_candidate_id = record["label"]["target_candidate_id"]
            candidate_ids = [candidate["candidate_id"] for candidate in candidates]

            history_lens.append(len(history))
            if len(candidates) != args.expected_candidates:
                bad_k += 1
            if target_candidate_id not in candidate_ids:
                bad_target += 1
            else:
                target_positions[candidate_ids.index(target_candidate_id) + 1] += 1

            for item in history + candidates:
                total_items += 1
                metadata = item.get("metadata") or {}
                title = metadata.get("title")
                if not title or title == "Unknown item":
                    missing_title += 1

    avg_history = sum(history_lens) / len(history_lens) if history_lens else 0.0
    missing_title_rate = missing_title / total_items if total_items else 0.0

    print(f"path: {args.path}")
    print(f"count: {count}")
    print(f"bad_candidate_size: {bad_k}")
    print(f"bad_target_membership: {bad_target}")
    print(
        "history_len_min_avg_max: "
        f"{min(history_lens) if history_lens else 0} / {avg_history:.2f} / "
        f"{max(history_lens) if history_lens else 0}"
    )
    print(f"missing_title_rate: {missing_title_rate:.4f} ({missing_title}/{total_items})")
    print("target_position_distribution:")
    for pos in range(1, args.expected_candidates + 1):
        print(f"  {pos:02d}: {target_positions[pos]}")

    if count and target_positions[1] == count:
        raise SystemExit("target is always at position 1; candidate order leakage detected")
    if bad_k or bad_target:
        raise SystemExit("dataset has invalid candidate records")


if __name__ == "__main__":
    main()
