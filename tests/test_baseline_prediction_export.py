"""Unit tests for deterministic baseline prediction export."""

from __future__ import annotations

import json
from pathlib import Path

from llm4rec.evaluation import (
    BaselinePredictionStrategy,
    build_example_index,
    iter_heuristic_prediction_records,
    load_offline_prediction_records_jsonl,
    score_offline_prediction_records,
    summarize_scored_predictions,
    write_offline_prediction_records_jsonl,
)


def test_oracle_prediction_export_hits_target(tmp_path: Path) -> None:
    config_path = _write_feature_driven_dataset(tmp_path)
    prediction_records = list(
        iter_heuristic_prediction_records(
            config_path,
            split="validation",
            strategy=BaselinePredictionStrategy.ORACLE,
        )
    )
    example_index = build_example_index(config_path, split="validation")
    summary = summarize_scored_predictions(
        score_offline_prediction_records(example_index, prediction_records)
    )

    assert len(prediction_records) == 1
    assert prediction_records[0].selected_item_id == "c7"
    assert summary.hit_rate == 1.0


def test_history_feature_overlap_export_round_trip(tmp_path: Path) -> None:
    config_path = _write_feature_driven_dataset(tmp_path)
    prediction_records = list(
        iter_heuristic_prediction_records(
            config_path,
            split="validation",
            strategy=BaselinePredictionStrategy.HISTORY_FEATURE_OVERLAP,
        )
    )
    output_path = tmp_path / "predictions.jsonl"
    write_offline_prediction_records_jsonl(output_path, prediction_records)
    loaded_records = load_offline_prediction_records_jsonl(output_path)
    example_index = build_example_index(config_path, split="validation")
    summary = summarize_scored_predictions(
        score_offline_prediction_records(example_index, loaded_records)
    )

    assert loaded_records[0].selected_item_id == "c7"
    assert loaded_records[0].metadata["strategy"] == "history_feature_overlap"
    assert loaded_records[0].ranked_item_ids[0] == "c7"
    assert summary.hit_rate == 1.0


def _write_feature_driven_dataset(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "amazon-food"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "train.csv").write_text(
        "\n".join(
            [
                "user_id,parent_asin,rating,timestamp,history",
                "u1,c1,5.0,100,",
                "u1,c2,5.0,200,c1",
                "u1,c3,5.0,300,c1 c2",
                "u1,c4,5.0,400,c1 c2 c3",
                "u1,c5,5.0,500,c1 c2 c3 c4",
                "u1,c6,5.0,600,c1 c2 c3 c4 c5",
                "u2,s1,5.0,110,",
                "u2,s2,5.0,210,s1",
                "u2,s3,5.0,310,s1 s2",
                "u2,s4,5.0,410,s1 s2 s3",
                "u2,s5,5.0,510,s1 s2 s3 s4",
                "u2,s6,5.0,610,s1 s2 s3 s4 s5",
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "valid.csv").write_text(
        "\n".join(
            [
                "user_id,parent_asin,rating,timestamp,history",
                "u1,c7,5.0,700,c1 c2 c3 c4 c5 c6",
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "test.csv").write_text(
        "\n".join(
            [
                "user_id,parent_asin,rating,timestamp,history",
                "u2,s7,5.0,710,s1 s2 s3 s4 s5 s6",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        raw_dir / "meta.jsonl",
        [
            *[
                {
                    "parent_asin": item_id,
                    "title": f"Coffee Item {item_id.upper()}",
                    "main_category": "Grocery",
                    "categories": ["Grocery & Gourmet Food", "Coffee"],
                }
                for item_id in ("c1", "c2", "c3", "c4", "c5", "c6", "c7")
            ],
            *[
                {
                    "parent_asin": item_id,
                    "title": f"Snack Item {item_id.upper()}",
                    "main_category": "Grocery",
                    "categories": ["Grocery & Gourmet Food", "Snack Foods"],
                }
                for item_id in ("s1", "s2", "s3", "s4", "s5", "s6", "s7")
            ],
        ],
    )

    config_path = tmp_path / "amazon_food_test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: amazon_food",
                f"raw_dir: {raw_dir.as_posix()}",
                "interactions_file: train.csv",
                "validation_file: valid.csv",
                "test_file: test.csv",
                "review_file:",
                "metadata_file: meta.jsonl",
                "items_file:",
                'separator: ","',
                "columns:",
                "  user_id: user_id",
                "  item_id: parent_asin",
                "  rating: rating",
                "  timestamp: timestamp",
                "  history: history",
                "preprocessing:",
                "  min_history_len: 2",
                "  max_history_len: 50",
                "  rating_threshold: 4.0",
                "  max_item_snippets: 2",
                "  max_snippet_chars: 80",
                "candidate_constructor:",
                "  strategy: popularity",
                "  num_candidates: 3",
                "  exclude_history: true",
                "  allow_insufficient_candidates: true",
                "  shuffle_candidates: false",
                "  seed: 42",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
