from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def history_items(history: str) -> list[str]:
    if not history:
        return []
    return [item for item in history.split(" ") if item]


def collect_popular_items(train_csv: Path, limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item_id = row.get("parent_asin")
            if item_id:
                counts[item_id] += 1
    return [item for item, _ in counts.most_common(limit)]


def collect_item_universe(train_csv: Path) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item_id = row.get("parent_asin")
            if item_id and item_id not in seen:
                seen.add(item_id)
                items.append(item_id)
    return items


def sample_rows(split_csv: Path, max_examples: int, min_history_len: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if len(history_items(row.get("history", ""))) < min_history_len:
                continue
            rows.append(row)
            if len(rows) >= max_examples:
                break
    return rows


def build_candidates(
    target: str,
    history: list[str],
    item_pool: list[str],
    num_candidates: int,
    rng: random.Random,
    policy: str,
) -> list[str]:
    blocked = set(history)
    blocked.add(target)
    negatives = [item for item in item_pool if item not in blocked]
    if policy == "random_negative":
        negatives = negatives[:]
        rng.shuffle(negatives)
    candidates = [target] + negatives[: num_candidates - 1]
    if len(candidates) != num_candidates:
        raise ValueError(f"Not enough candidates for target {target}")
    rng.shuffle(candidates)
    return candidates


def scan_metadata(metadata_jsonl: Path, needed_items: set[str]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with metadata_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not needed_items:
                break
            record = json.loads(line)
            item_id = record.get("parent_asin")
            if item_id in needed_items:
                metadata[item_id] = {
                    "title": record.get("title") or "Unknown item",
                    "store": record.get("store"),
                    "categories": record.get("categories") or [],
                    "features": (record.get("features") or [])[:5],
                    "description": " ".join(record.get("description") or [])[:280],
                    "average_rating": record.get("average_rating"),
                    "rating_number": record.get("rating_number"),
                    "price": record.get("price"),
                }
                needed_items.remove(item_id)
    return metadata


def letter_id(index: int) -> str:
    return chr(ord("A") + index)


def convert_row(
    row: dict[str, str],
    candidates: list[str],
    metadata: dict[str, dict[str, Any]],
    max_history_len: int,
    instance_id: str,
) -> dict[str, Any]:
    target_item = row["parent_asin"]
    history = history_items(row.get("history", ""))[-max_history_len:]

    candidate_records = []
    target_candidate_id = None
    for idx, item_id in enumerate(candidates):
        candidate_id = letter_id(idx)
        if item_id == target_item:
            target_candidate_id = candidate_id
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "item_id": item_id,
                "metadata": metadata.get(item_id, {"title": "Unknown item"}),
            }
        )

    return {
        "instance_id": instance_id,
        "dataset": "amazon_food_pilot",
        "user_id": row["user_id"],
        "history": [
            {
                "evidence_id": f"H{idx + 1:02d}",
                "item_id": item_id,
                "metadata": metadata.get(item_id, {"title": "Unknown item"}),
            }
            for idx, item_id in enumerate(history)
        ],
        "candidates": candidate_records,
        "label": {
            "target_item_id": target_item,
            "target_candidate_id": target_candidate_id,
        },
    }


def build_split(config: dict[str, Any], split_name: str, split_file_key: str, output_name: str) -> None:
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_csv = raw_dir / config["paths"]["train_file"]
    split_csv = raw_dir / config["paths"][split_file_key]
    metadata_jsonl = raw_dir / config["paths"]["metadata_file"]

    max_examples = int(config["pilot"][f"{split_name}_examples"])
    min_history_len = int(config["sequence"]["min_history_len"])
    max_history_len = int(config["sequence"]["max_history_len"])
    num_candidates = int(config["candidates"]["num_candidates"])
    pool_size = int(config["candidates"]["policies"]["popularity_negative"]["popularity_pool_size"])
    policy = config["candidates"].get("policy", "random_negative")
    seed = int(config["pilot"].get("seed", 42))

    if policy == "random_negative":
        item_pool = collect_item_universe(train_csv)
    elif policy == "popularity_negative":
        item_pool = collect_popular_items(train_csv, pool_size)
    else:
        raise ValueError(f"Unsupported candidate policy for pilot builder: {policy}")

    rows = sample_rows(split_csv, max_examples, min_history_len)

    candidate_sets = []
    needed_items: set[str] = set()
    for idx, row in enumerate(rows):
        history = history_items(row.get("history", ""))[-max_history_len:]
        rng = random.Random(f"{seed}:{split_name}:{idx}:{row['user_id']}:{row['parent_asin']}")
        candidates = build_candidates(
            row["parent_asin"], history, item_pool, num_candidates, rng, policy
        )
        candidate_sets.append(candidates)
        needed_items.update(history)
        needed_items.update(candidates)

    metadata = scan_metadata(metadata_jsonl, needed_items)

    output_path = processed_dir / output_name
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (row, candidates) in enumerate(zip(rows, candidate_sets, strict=True)):
            record = convert_row(
                row,
                candidates,
                metadata,
                max_history_len,
                instance_id=f"amazon_food_{split_name}_{idx:06d}",
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} {split_name} instances to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Amazon Food pilot JSONL files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/amazon_food_pilot.yaml"),
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )
    args = parser.parse_args()

    config = read_yaml(args.config)
    if args.split == "validation":
        build_split(config, "validation", "validation_file", "valid.jsonl")
    else:
        build_split(config, "test", "test_file", "test.jsonl")


if __name__ == "__main__":
    main()
