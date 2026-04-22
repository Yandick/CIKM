"""Prompt rendering helpers for baseline recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from llm4rec.data.schema import ItemRecord, NextItemExample


class BaselinePromptStyle(str, Enum):
    """Supported baseline prompt styles for the first experiments."""

    ANSWER_ONLY = "answer_only"
    FREE_FORM_COT = "free_form_cot"


@dataclass(frozen=True)
class PromptRenderConfig:
    """Rendering controls for concise recommendation prompts."""

    max_history_items: int = 10
    max_candidates: int = 20
    max_feature_refs_per_item: int = 3
    max_feature_bullets_per_item: int = 2
    max_description_snippets_per_item: int = 1
    max_review_snippets_per_item: int = 1


@dataclass(frozen=True)
class PromptRecord:
    """One rendered prompt record ready for inference or SFT packaging."""

    example_id: str
    split: str
    style: BaselinePromptStyle
    prompt: tuple[dict[str, str], ...]
    target_item_id: str
    candidate_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def render_baseline_prompt(
    example: NextItemExample,
    *,
    item_records: dict[str, ItemRecord],
    style: BaselinePromptStyle,
    config: PromptRenderConfig | None = None,
) -> PromptRecord:
    """Render one next-item example into a chat-style baseline prompt."""

    render_config = config or PromptRenderConfig()
    history_block = _render_history_block(example, item_records=item_records, config=render_config)
    candidate_block = _render_candidate_block(example, item_records=item_records, config=render_config)
    task_block = _render_task_block(style)

    system_prompt = (
        "You are a recommendation assistant. "
        "Use the user's recent history and the candidate item information to pick the best next item."
    )
    user_prompt = "\n\n".join([history_block, candidate_block, task_block])
    prompt = (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    )

    return PromptRecord(
        example_id=example.example_id,
        split=example.split,
        style=style,
        prompt=prompt,
        target_item_id=example.target_item_id,
        candidate_ids=example.candidate_ids(),
        metadata={
            "user_id": example.user_id,
            "history_length": len(example.history),
            "target_timestamp": example.context.get("target_timestamp"),
        },
    )


def iter_rendered_prompts(
    examples: Iterable[NextItemExample],
    *,
    item_records: dict[str, ItemRecord],
    style: BaselinePromptStyle,
    config: PromptRenderConfig | None = None,
) -> Iterable[PromptRecord]:
    """Render a stream of examples into baseline prompts."""

    for example in examples:
        yield render_baseline_prompt(
            example,
            item_records=item_records,
            style=style,
            config=config,
        )


def _render_history_block(
    example: NextItemExample,
    *,
    item_records: dict[str, ItemRecord],
    config: PromptRenderConfig,
) -> str:
    """Render the chronological user history section."""

    lines = ["User history (oldest to newest):"]
    history_events = example.history[-config.max_history_items :]
    start_index = len(example.history) - len(history_events) + 1
    for offset, event in enumerate(history_events):
        item_record = item_records.get(event.item_id, ItemRecord(item_id=event.item_id))
        title = item_record.title or event.item_id
        rating_text = "" if event.rating is None else f" | rating={event.rating:.1f}"
        lines.append(f"{start_index + offset}. [{event.item_id}] {title}{rating_text}")
    return "\n".join(lines)


def _render_candidate_block(
    example: NextItemExample,
    *,
    item_records: dict[str, ItemRecord],
    config: PromptRenderConfig,
) -> str:
    """Render the candidate set section with weak item profiles."""

    lines = ["Candidates:"]
    for rank, candidate in enumerate(example.candidates[: config.max_candidates], start=1):
        item_record = item_records.get(candidate.item_id, ItemRecord(item_id=candidate.item_id))
        lines.append(f"{rank}. [{candidate.item_id}] {_render_item_profile(item_record, config=config)}")
    return "\n".join(lines)


def _render_item_profile(item_record: ItemRecord, *, config: PromptRenderConfig) -> str:
    """Render one compact item profile from metadata and reviews."""

    title = item_record.title or item_record.item_id
    parts = [title]

    feature_refs = _prioritize_feature_refs(item_record.feature_refs)[
        : config.max_feature_refs_per_item
    ]
    if feature_refs:
        parts.append(f"signals={', '.join(feature_refs)}")

    raw = item_record.raw
    feature_bullets = list(raw.get("feature_bullets", ()))[: config.max_feature_bullets_per_item]
    if feature_bullets:
        parts.append(f"features={'; '.join(feature_bullets)}")

    descriptions = list(raw.get("description_snippets", ()))[: config.max_description_snippets_per_item]
    if descriptions:
        parts.append(f"description={'; '.join(descriptions)}")

    review_snippets = list(raw.get("review_snippets", ()))[: config.max_review_snippets_per_item]
    if review_snippets:
        parts.append(f"reviews={'; '.join(review_snippets)}")

    return " | ".join(parts)


def _prioritize_feature_refs(feature_refs: tuple[str, ...]) -> list[str]:
    """Promote informative category-like features ahead of generic source tags."""

    def feature_key(feature_ref: str) -> tuple[int, str]:
        if feature_ref == "category:grocery_and_gourmet_food":
            return (3, feature_ref)
        if feature_ref.startswith("category:"):
            return (0, feature_ref)
        if feature_ref.startswith("mean_rating:") or feature_ref.startswith("review_count:"):
            return (1, feature_ref)
        return (2, feature_ref)

    return sorted(feature_refs, key=feature_key)


def _render_task_block(style: BaselinePromptStyle) -> str:
    """Render the task instruction block for one baseline style."""

    if style == BaselinePromptStyle.ANSWER_ONLY:
        return (
            "Task:\n"
            "Choose the single best candidate as the next item for the user.\n"
            "Return only the candidate id."
        )

    if style == BaselinePromptStyle.FREE_FORM_COT:
        return (
            "Task:\n"
            "First give a short analysis of the user's likely recent and persistent preferences, "
            "and compare the candidates against them.\n"
            "Then end with a final line in the exact format: Answer: <candidate_id>."
        )

    raise ValueError(f"unsupported style: {style}")
