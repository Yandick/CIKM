"""Unit tests for Amazon Food preprocessing and processed-data export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4rec.data import (
    build_candidate_constructor_from_train_split,
    build_item_records_from_sources,
    build_item_records_from_reviews,
    build_next_item_splits,
    export_prepared_amazon_food,
    iter_amazon_food_rows,
    load_dataset_config,
    prepare_amazon_food,
)
from llm4rec.reasoning import validate_trace_rec_graph
from llm4rec.evaluation import InterventionType, score_counterfactual_graph_case
from llm4rec.prompts import BaselinePromptStyle, render_baseline_prompt
from llm4rec.training import prompt_record_to_sft_record, score_prediction_text


def test_iter_amazon_food_rows_parses_history(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    config = load_dataset_config(config_path)

    rows = list(iter_amazon_food_rows(config, split="train"))

    assert rows[0].history_item_ids == ()
    assert rows[1].history_item_ids == ("i1",)
    assert rows[2].item_id == "i3"


def test_build_item_records_from_sources_merges_meta_and_reviews(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    config = load_dataset_config(config_path)

    item_records = build_item_records_from_sources(
        config,
        allowed_item_ids={"i6", "j6", "missing"},
    )

    assert item_records["i6"].title == "Spiced Nut Mix"
    assert "category:snacks" in item_records["i6"].feature_refs
    assert "review_count:1-2" in item_records["i6"].feature_refs
    assert "mean_rating:4.5+" in item_records["i6"].feature_refs
    assert item_records["i6"].raw["review_titles"] == ("Excellent snack",)
    assert item_records["i6"].raw["description_snippets"] == ("A savory mixed snack for repeat purchase.",)
    assert item_records["missing"].raw == {}


def test_build_next_item_splits_from_precomputed_csv(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    config = load_dataset_config(config_path)
    item_records = build_item_records_from_reviews(config)
    candidate_constructor = build_candidate_constructor_from_train_split(config)

    train_examples, validation_examples, test_examples = build_next_item_splits(
        config,
        candidate_constructor=candidate_constructor,
        item_records=item_records,
    )

    assert len(train_examples) == 8
    assert len(validation_examples) == 1
    assert len(test_examples) == 1
    assert train_examples[0].seen_item_ids() == ("i1", "i2")
    assert train_examples[0].target_item_id == "i3"
    assert train_examples[0].context["history_timestamp_mode"] == "synthetic_from_order"
    assert train_examples[0].context["available_evidence_refs"] == ("history:0", "history:1")
    assert "i3" in train_examples[0].context["available_feature_refs_by_candidate"]
    assert validation_examples[0].target_item_id == "i7"
    assert test_examples[0].target_item_id == "j7"


def test_prepare_amazon_food_reads_split_files_and_metadata(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)

    prepared = prepare_amazon_food(config_path)

    assert "i1" in prepared.item_records
    assert "j7" in prepared.item_records
    assert prepared.item_records["i7"].title == "Morning Coffee Beans"
    assert len(prepared.train_examples) == 8
    assert len(prepared.validation_examples) == 1
    assert len(prepared.test_examples) == 1
    assert "available_feature_refs_by_candidate" in prepared.train_examples[0].context


def test_amazon_food_example_context_can_drive_graph_validation(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    prepared = prepare_amazon_food(config_path)
    example = prepared.validation_examples[0]
    feature_ref = example.context["available_feature_refs_by_candidate"][example.target_item_id][0]

    payload = {
        "metadata": {
            "task": "next_item_recommendation",
            "user_id": example.user_id,
            "candidate_ids": list(example.candidate_ids()),
            "context": example.context,
        },
        "nodes": [
            {
                "id": "pref-1",
                "type": "preference_state",
                "summary": "recent repeat-buy tendency",
                "polarity": "positive",
                "horizon": "recent",
                "evidence_refs": [example.context["available_evidence_refs"][0]],
                "source": "history",
            },
            {
                "id": "cand-1",
                "type": "candidate_evidence",
                "candidate_id": example.target_item_id,
                "feature_refs": [feature_ref],
                "summary": "candidate matches weak item profile",
            },
            {
                "id": "decision-1",
                "type": "decision",
                "selected_item_id": example.target_item_id,
            },
        ],
        "edges": [
            {"source": "pref-1", "target": "cand-1", "type": "supports"},
            {"source": "cand-1", "target": "decision-1", "type": "selected"},
        ],
    }

    result = validate_trace_rec_graph(payload)

    assert result.valid is True


def test_amazon_food_example_supports_graph_based_counterfactual_scoring(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    prepared = prepare_amazon_food(config_path)
    example = prepared.validation_examples[0]
    candidate_ids = list(example.candidate_ids())
    target_feature_refs = example.context["available_feature_refs_by_candidate"][example.target_item_id]

    original_graph = {
        "metadata": {
            "task": "next_item_recommendation",
            "user_id": example.user_id,
            "candidate_ids": candidate_ids,
            "context": example.context,
        },
        "nodes": [
            {
                "id": "pref-1",
                "type": "preference_state",
                "summary": "recent repeat-buy tendency",
                "polarity": "positive",
                "horizon": "recent",
                "evidence_refs": [example.context["available_evidence_refs"][0]],
                "source": "history",
            },
            {
                "id": "cand-1",
                "type": "candidate_evidence",
                "candidate_id": example.target_item_id,
                "feature_refs": [target_feature_refs[0]],
                "summary": "candidate matches weak item profile",
            },
            {
                "id": "decision-1",
                "type": "decision",
                "selected_item_id": example.target_item_id,
            },
        ],
        "edges": [
            {"source": "pref-1", "target": "cand-1", "type": "supports"},
            {"source": "cand-1", "target": "decision-1", "type": "selected"},
        ],
    }
    updated_graph = {
        "metadata": original_graph["metadata"],
        "nodes": [
            original_graph["nodes"][0],
            {
                "id": "cand-1",
                "type": "candidate_evidence",
                "candidate_id": example.target_item_id,
                "feature_refs": [target_feature_refs[1]],
                "summary": "candidate feature changed",
            },
            {
                "id": "decision-1",
                "type": "decision",
                "selected_item_id": example.target_item_id,
            },
        ],
        "edges": [
            {"source": "pref-1", "target": "cand-1", "type": "supports"},
            {"source": "cand-1", "target": "decision-1", "type": "selected"},
        ],
    }

    assert validate_trace_rec_graph(original_graph).valid is True
    assert validate_trace_rec_graph(updated_graph).valid is True

    score = score_counterfactual_graph_case(
        original_graph,
        updated_graph,
        InterventionType.CANDIDATE_FEATURE_SWAP,
        focus_candidate_id=example.target_item_id,
    )

    assert score.targeted_update_correct is True
    assert score.decision_direction_consistent is True


def test_export_prepared_amazon_food_writes_jsonl(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    output_dir = tmp_path / "processed"

    summary = export_prepared_amazon_food(config_path, output_dir)

    assert summary.item_count >= 14
    assert summary.train_count == 8
    assert summary.validation_count == 1
    assert summary.test_count == 1
    assert (output_dir / "item_records.jsonl").exists()
    assert _count_lines(output_dir / "train.jsonl") == 8
    assert _count_lines(output_dir / "validation.jsonl") == 1
    assert _count_lines(output_dir / "test.jsonl") == 1


def test_render_baseline_prompt_uses_item_metadata(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    prepared = prepare_amazon_food(config_path)
    example = prepared.validation_examples[0]

    prompt_record = render_baseline_prompt(
        example,
        item_records=prepared.item_records,
        style=BaselinePromptStyle.FREE_FORM_COT,
    )

    user_prompt = prompt_record.prompt[1]["content"]
    assert "Morning Coffee Beans" in user_prompt
    assert "Answer: <candidate_id>" in user_prompt
    assert "category:coffee" in user_prompt


def test_prompt_record_to_sft_record_and_prediction_scoring(tmp_path: Path) -> None:
    config_path = _write_toy_amazon_food_dataset(tmp_path)
    prepared = prepare_amazon_food(config_path)
    example = prepared.test_examples[0]

    prompt_record = render_baseline_prompt(
        example,
        item_records=prepared.item_records,
        style=BaselinePromptStyle.FREE_FORM_COT,
    )
    score = score_prediction_text(f"Reasoning: concise.\nAnswer: {example.target_item_id}", prompt_record)

    with pytest.raises(ValueError):
        prompt_record_to_sft_record(prompt_record)
    assert score.parse_success is True
    assert score.hit is True


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
                "u2,j8,2.0,720,j1 j2 j3 j4 j5 j6",
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
            {
                "parent_asin": "i6",
                "title": "Spiced Nut Mix",
                "main_category": "Grocery",
                "store": "Trail House",
                "categories": ["Grocery & Gourmet Food", "Snacks"],
                "features": ["Protein-rich", "Savory coating"],
                "description": ["A savory mixed snack for repeat purchase."],
                "average_rating": 4.8,
                "rating_number": 23,
            },
            {
                "parent_asin": "j6",
                "title": "Soup Base Cubes",
                "main_category": "Grocery",
                "store": "Kitchen Box",
                "categories": ["Grocery & Gourmet Food", "Pantry Staples"],
                "features": ["Fast flavor boost"],
                "description": ["Adds flavor to quick soups."],
                "average_rating": 4.0,
                "rating_number": 8,
            },
            {
                "parent_asin": "i7",
                "title": "Morning Coffee Beans",
                "main_category": "Grocery",
                "store": "Roast Lab",
                "categories": ["Grocery & Gourmet Food", "Coffee"],
                "features": ["Dark roast"],
                "description": ["Bold beans for daily brewing."],
                "average_rating": 4.9,
                "rating_number": 40,
            },
            {
                "parent_asin": "j7",
                "title": "Soft Granola Bites",
                "main_category": "Grocery",
                "store": "Snack Valley",
                "categories": ["Grocery & Gourmet Food", "Snack Foods"],
                "features": ["Soft texture"],
                "description": ["A soft-texture snack with balanced sweetness."],
                "average_rating": 4.6,
                "rating_number": 16,
            },
        ],
    )
    _write_jsonl(
        raw_dir / "reviews.jsonl",
        [
            {
                "parent_asin": "i6",
                "asin": "i6",
                "rating": 5.0,
                "verified_purchase": True,
                "title": "Excellent snack",
                "text": "Crunchy, fresh, and easy to recommend.",
            },
            {
                "parent_asin": "j6",
                "asin": "j6",
                "rating": 4.0,
                "verified_purchase": False,
                "title": "Solid pantry item",
                "text": "Useful flavor base for quick meals.",
            },
            {
                "parent_asin": "i7",
                "asin": "i7",
                "rating": 5.0,
                "verified_purchase": True,
                "title": "Great repeat buy",
                "text": "Would buy again and pair with coffee.",
            },
            {
                "parent_asin": "j7",
                "asin": "j7",
                "rating": 5.0,
                "verified_purchase": True,
                "title": "Good texture",
                "text": "Balanced sweetness and good texture.",
            },
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


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)
