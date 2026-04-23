"""Unit tests for local-model inference contracts and parsing helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.evaluation import build_example_index, load_offline_prediction_records_jsonl, score_offline_prediction_records
from llm4rec.inference import (
    HFLocalModelConfig,
    extract_candidate_id_from_text,
    load_inference_run_config,
    load_local_hf_model_config,
    prediction_record_from_response,
    render_prompt_text,
)
from llm4rec.prompts import BaselinePromptStyle, PromptRecord, load_baseline_prompt_template


def test_load_local_hf_model_and_inference_configs(tmp_path: Path) -> None:
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "name: qwen-local",
                "backend: huggingface_local_causal_lm",
                "model_path: D:/models/qwen-local",
                "local_files_only: true",
                "runtime:",
                "  use_chat_template: true",
                "  max_input_tokens: 1024",
                "generation:",
                "  max_new_tokens: 64",
                "  do_sample: false",
            ]
        ),
        encoding="utf-8",
    )
    inference_config_path = tmp_path / "inference.yaml"
    inference_config_path.write_text(
        "\n".join(
            [
                "name: answer_only_baseline",
                "prompt_style: answer_only",
                "prompt_version: builtin_answer_only_v1",
                "split: validation",
                "max_examples: 12",
            ]
        ),
        encoding="utf-8",
    )

    model_config = load_local_hf_model_config(model_config_path)
    run_config = load_inference_run_config(inference_config_path)

    assert model_config.local_files_only is True
    assert model_config.model_path == "D:/models/qwen-local"
    assert model_config.generation.max_new_tokens == 64
    assert run_config.prompt_style == BaselinePromptStyle.ANSWER_ONLY
    assert run_config.max_examples == 12


def test_load_baseline_prompt_template_from_yaml(tmp_path: Path) -> None:
    prompt_config_path = tmp_path / "prompt.yaml"
    prompt_config_path.write_text(
        "\n".join(
            [
                "name: free_form_cot_v1",
                "style: free_form_cot",
                "system_prompt: You are a helpful recommender.",
                "task_block: |",
                "  Task:",
                "  Think briefly, then answer with Answer: <candidate_id>.",
            ]
        ),
        encoding="utf-8",
    )

    template = load_baseline_prompt_template(prompt_config_path)

    assert template.name == "free_form_cot_v1"
    assert template.style == BaselinePromptStyle.FREE_FORM_COT
    assert "Answer: <candidate_id>" in template.task_block


def test_prediction_record_from_response_parses_candidate_and_metadata() -> None:
    prompt_record = PromptRecord(
        example_id="ex1",
        split="validation",
        style=BaselinePromptStyle.FREE_FORM_COT,
        prompt=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ),
        target_item_id="i3",
        candidate_ids=("i1", "i2", "i3"),
        metadata={"user_id": "u1", "history_length": 6, "target_timestamp": 700},
    )
    model_config = HFLocalModelConfig(
        name="qwen-local",
        backend="huggingface_local_causal_lm",
        model_path="D:/models/qwen-local",
    )

    record = prediction_record_from_response(
        prompt_record,
        "Reasoning.\nAnswer: i2",
        model_config=model_config,
        run_id="run-1",
        prompt_version="builtin_free_form_cot_v1",
        latency_ms=12.5,
        prompt_tokens=100,
        completion_tokens=8,
    )

    assert record.selected_item_id == "i2"
    assert record.prompt_style == "free_form_cot"
    assert record.model_name == "qwen-local"
    assert record.prompt_tokens == 100
    assert record.metadata["user_id"] == "u1"


def test_render_prompt_text_without_chat_template_falls_back_to_role_blocks() -> None:
    prompt_record = PromptRecord(
        example_id="ex1",
        split="validation",
        style=BaselinePromptStyle.ANSWER_ONLY,
        prompt=(
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ),
        target_item_id="i1",
        candidate_ids=("i1", "i2"),
    )

    rendered = render_prompt_text(prompt_record, tokenizer=None, use_chat_template=False)

    assert "SYSTEM:" in rendered
    assert "USER:" in rendered
    assert rendered.strip().endswith("ASSISTANT:")


def test_extract_candidate_id_from_text_supports_answer_line() -> None:
    parsed = extract_candidate_id_from_text(
        "Short reasoning.\nAnswer: i7",
        ("i6", "i7", "i8"),
    )

    assert parsed == "i7"


def test_offline_scoring_rejects_conflicting_selected_item_and_response_text(tmp_path: Path) -> None:
    config_path = _write_toy_dataset(tmp_path)
    example_index = build_example_index(config_path, split="validation")
    example_id = next(iter(example_index))
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        '{"example_id": "%s", "selected_item_id": "i7", "response_text": "Answer: i6"}\n'
        % example_id,
        encoding="utf-8",
    )

    records = load_offline_prediction_records_jsonl(predictions_path)
    with pytest.raises(ValueError):
        score_offline_prediction_records(example_index, records)


def _write_toy_dataset(tmp_path: Path) -> Path:
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
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "valid.csv").write_text(
        "user_id,parent_asin,rating,timestamp,history\nu1,i7,5.0,700,i1 i2 i3 i4 i5 i6\n",
        encoding="utf-8",
    )
    (raw_dir / "test.csv").write_text(
        "user_id,parent_asin,rating,timestamp,history\nu1,i8,5.0,800,i1 i2 i3 i4 i5 i6 i7\n",
        encoding="utf-8",
    )
    (raw_dir / "meta.jsonl").write_text(
        "\n".join(
            [
                '{"parent_asin":"i6","title":"Snack Six","main_category":"Grocery","categories":["Grocery & Gourmet Food","Snacks"]}',
                '{"parent_asin":"i7","title":"Coffee Seven","main_category":"Grocery","categories":["Grocery & Gourmet Food","Coffee"]}',
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "reviews.jsonl").write_text(
        "\n".join(
            [
                '{"parent_asin":"i6","asin":"i6","rating":5.0,"verified_purchase":true,"title":"good","text":"good snack"}',
                '{"parent_asin":"i7","asin":"i7","rating":5.0,"verified_purchase":true,"title":"great","text":"great coffee"}',
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "amazon_food.yaml"
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
