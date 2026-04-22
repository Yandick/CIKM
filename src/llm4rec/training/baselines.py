"""Baseline prompt export, weak SFT packaging, and prediction parsing helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from llm4rec.data import (
    build_candidate_constructor_from_train_split,
    build_item_records_from_sources,
    collect_split_item_ids,
    iter_amazon_food_examples,
    load_dataset_config,
)
from llm4rec.data.schema import DatasetConfig
from llm4rec.prompts import (
    BaselinePromptStyle,
    PromptRecord,
    PromptRenderConfig,
    iter_rendered_prompts,
)


ANSWER_LINE_RE = re.compile(r"answer\s*:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class SFTRecord:
    """Prompt/response pair suitable for a first-stage SFT dataset."""

    example_id: str
    split: str
    style: BaselinePromptStyle
    prompt: tuple[dict[str, str], ...]
    response: str
    target_item_id: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class PredictionScore:
    """Minimal parsing/evaluation result for one generated prediction."""

    parsed_item_id: str | None
    parse_success: bool
    hit: bool


def iter_baseline_prompt_records(
    config: DatasetConfig | str | Path,
    *,
    split: str,
    style: BaselinePromptStyle,
    render_config: PromptRenderConfig | None = None,
    limit: int | None = None,
) -> Iterable[PromptRecord]:
    """Render one Amazon Food split into prompt records without materializing all examples."""

    dataset_config = _ensure_dataset_config(config)
    candidate_constructor = build_candidate_constructor_from_train_split(dataset_config)
    item_ids = collect_split_item_ids(dataset_config)
    item_records = build_item_records_from_sources(
        dataset_config,
        allowed_item_ids=item_ids,
    )
    examples = iter_amazon_food_examples(
        dataset_config,
        split=split,
        candidate_constructor=candidate_constructor,
        item_records=item_records,
    )

    for index, record in enumerate(
        iter_rendered_prompts(
            examples,
            item_records=item_records,
            style=style,
            config=render_config,
        )
    ):
        if limit is not None and index >= limit:
            break
        yield record


def prompt_record_to_sft_record(prompt_record: PromptRecord) -> SFTRecord:
    """Convert one prompt record into a weakly supervised prompt/response pair."""

    if prompt_record.style == BaselinePromptStyle.ANSWER_ONLY:
        response = prompt_record.target_item_id
    elif prompt_record.style == BaselinePromptStyle.FREE_FORM_COT:
        response = f"Answer: {prompt_record.target_item_id}"
    else:
        raise ValueError(f"unsupported style: {prompt_record.style}")

    return SFTRecord(
        example_id=prompt_record.example_id,
        split=prompt_record.split,
        style=prompt_record.style,
        prompt=prompt_record.prompt,
        response=response,
        target_item_id=prompt_record.target_item_id,
        candidate_ids=prompt_record.candidate_ids,
    )


def extract_predicted_candidate_id(text: str, candidate_ids: tuple[str, ...]) -> str | None:
    """Extract one candidate id from model output using answer lines or exact mentions."""

    if not text:
        return None

    match = ANSWER_LINE_RE.search(text)
    if match is not None:
        candidate_id = match.group(1).strip()
        if candidate_id in candidate_ids:
            return candidate_id

    normalized = text.strip()
    if normalized in candidate_ids:
        return normalized

    mentioned = [candidate_id for candidate_id in candidate_ids if candidate_id in text]
    if len(mentioned) == 1:
        return mentioned[0]
    return None


def score_prediction_text(text: str, prompt_record: PromptRecord) -> PredictionScore:
    """Parse a prediction string and score exact next-item hit."""

    parsed_item_id = extract_predicted_candidate_id(text, prompt_record.candidate_ids)
    return PredictionScore(
        parsed_item_id=parsed_item_id,
        parse_success=parsed_item_id is not None,
        hit=parsed_item_id == prompt_record.target_item_id,
    )


def write_prompt_records_jsonl(path: str | Path, records: Iterable[PromptRecord]) -> int:
    """Write prompt records as JSONL."""

    return _write_jsonl(path, (asdict(record) for record in records))


def write_sft_records_jsonl(path: str | Path, records: Iterable[SFTRecord]) -> int:
    """Write weak SFT records as JSONL."""

    return _write_jsonl(path, (asdict(record) for record in records))


def _write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write a stream of dictionaries to JSONL."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _ensure_dataset_config(config: DatasetConfig | str | Path) -> DatasetConfig:
    """Normalize config input into a DatasetConfig object."""

    if isinstance(config, DatasetConfig):
        return config
    return load_dataset_config(config)
