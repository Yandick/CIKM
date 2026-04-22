"""Unit tests for offline verifier scoring and best-by-reward reranking."""

from __future__ import annotations

import json
from pathlib import Path

from llm4rec.evaluation import (
    build_example_index,
    load_offline_prediction_records_jsonl,
    score_offline_prediction_records,
    select_best_predictions_by_reward,
    summarize_scored_predictions,
)


def test_offline_verifier_scores_predictions_and_selects_best(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    example_index = build_example_index(config_path, split="validation")
    example_id = next(iter(example_index))
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "example_id": example_id,
                "response_text": "Answer: i6",
                "group_id": example_id,
                "sample_index": 0,
            },
            {
                "example_id": example_id,
                "response_text": "Answer: i7",
                "group_id": example_id,
                "sample_index": 1,
            },
        ],
    )

    prediction_records = load_offline_prediction_records_jsonl(predictions_path)
    scored_records = score_offline_prediction_records(example_index, prediction_records)
    best_records = select_best_predictions_by_reward(scored_records)
    summary = summarize_scored_predictions(scored_records)

    assert len(scored_records) == 2
    assert best_records[0].verifier.selected_item_id == "i7"
    assert best_records[0].verifier.exact_hit is True
    assert round(summary.hit_rate, 4) == 0.5
    assert round(summary.best_of_n_hit_rate, 4) == 1.0


def _write_toy_amazon_food_dataset(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "amazon-food"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "train.csv").write_text(
        "\n".join(
            [
                "user_id,parent_asin,rating,timestamp,history",
                "u1,i1,5.0,100,",
                "u1,i2,5.0,200,i1",
                "u1,i3,5.0,300,i1 i2",
                "u1,i4,5.0,400,i1 i2 i3",
                "u1,i5,5.0,500,i1 i2 i3 i4",
                "u1,i6,5.0,600,i1 i2 i3 i4 i5",
                "u2,j1,5.0,110,",
                "u2,j2,5.0,210,j1",
                "u2,j3,5.0,310,j1 j2",
                "u2,j4,5.0,410,j1 j2 j3",
                "u2,j5,5.0,510,j1 j2 j3 j4",
                "u2,j6,5.0,610,j1 j2 j3 j4 j5",
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "valid.csv").write_text(
        "\n".join(
            [
                "user_id,parent_asin,rating,timestamp,history",
                "u1,i7,5.0,700,i1 i2 i3 i4 i5 i6",
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "test.csv").write_text(
        "\n".join(
            [
                "user_id,parent_asin,rating,timestamp,history",
                "u2,j7,5.0,710,j1 j2 j3 j4 j5 j6",
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        raw_dir / "meta.jsonl",
        [
            {"parent_asin": "i7", "title": "Morning Coffee Beans", "main_category": "Grocery", "categories": ["Grocery & Gourmet Food", "Coffee"]},
            {"parent_asin": "i6", "title": "Spiced Nut Mix", "main_category": "Grocery", "categories": ["Grocery & Gourmet Food", "Snacks"]},
            {"parent_asin": "j6", "title": "Soup Base Cubes", "main_category": "Grocery", "categories": ["Grocery & Gourmet Food", "Pantry Staples"]},
            {"parent_asin": "j7", "title": "Soft Granola Bites", "main_category": "Grocery", "categories": ["Grocery & Gourmet Food", "Snack Foods"]},
        ],
    )
    _write_jsonl(
        raw_dir / "reviews.jsonl",
        [
            {"parent_asin": "i7", "asin": "i7", "rating": 5.0, "verified_purchase": True, "title": "Great repeat buy", "text": "Would buy again and pair with coffee."},
            {"parent_asin": "i6", "asin": "i6", "rating": 5.0, "verified_purchase": True, "title": "Excellent snack", "text": "Crunchy and easy to recommend."},
            {"parent_asin": "j6", "asin": "j6", "rating": 4.0, "verified_purchase": False, "title": "Solid pantry item", "text": "Useful flavor base."},
            {"parent_asin": "j7", "asin": "j7", "rating": 5.0, "verified_purchase": True, "title": "Good texture", "text": "Balanced sweetness and soft texture."},
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
                "review_file: reviews.jsonl",
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
