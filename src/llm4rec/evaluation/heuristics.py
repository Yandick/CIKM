"""Deterministic baseline predictors that emit offline prediction records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from llm4rec.data import (
    build_candidate_constructor_from_train_split,
    build_item_records_from_sources,
    collect_split_item_ids,
    iter_amazon_food_examples,
    load_dataset_config,
)
from llm4rec.data.schema import DatasetConfig, ItemRecord, NextItemExample
from llm4rec.evaluation.offline import OfflinePredictionRecord


class BaselinePredictionStrategy(str, Enum):
    """Supported deterministic strategies for prediction JSONL export."""

    ORACLE = "oracle"
    POPULARITY = "popularity"
    HISTORY_FEATURE_OVERLAP = "history_feature_overlap"


@dataclass(frozen=True)
class CandidateScore:
    """Compact candidate-level score breakdown for one heuristic prediction."""

    item_id: str
    total_score: float
    overlap_score: float
    popularity_score: float
    rating_score: float


def iter_heuristic_prediction_records(
    config: DatasetConfig | str | Path,
    *,
    split: str,
    strategy: BaselinePredictionStrategy,
    limit: int | None = None,
) -> Iterable[OfflinePredictionRecord]:
    """Build deterministic prediction rows for one dataset split."""

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

    for index, example in enumerate(examples):
        if limit is not None and index >= limit:
            break
        yield build_heuristic_prediction_record(
            example,
            item_records=item_records,
            strategy=strategy,
        )


def build_heuristic_prediction_record(
    example: NextItemExample,
    *,
    item_records: dict[str, ItemRecord],
    strategy: BaselinePredictionStrategy,
) -> OfflinePredictionRecord:
    """Construct one offline prediction record from a deterministic baseline."""

    if strategy == BaselinePredictionStrategy.ORACLE:
        ranked_item_ids = tuple(
            [example.target_item_id]
            + [candidate.item_id for candidate in example.candidates if candidate.item_id != example.target_item_id]
        )
        run_id = f"heuristic_{strategy.value}"
        return OfflinePredictionRecord(
            example_id=example.example_id,
            selected_item_id=example.target_item_id,
            ranked_item_ids=ranked_item_ids,
            run_id=run_id,
            model_name=strategy.value,
            prompt_style="answer_only",
            prompt_version="heuristic_v1",
            group_id=run_id,
            sample_index=0,
            metadata={
                "strategy": strategy.value,
                "target_item_id": example.target_item_id,
            },
        )

    scored_candidates = score_candidates(
        example,
        item_records=item_records,
        strategy=strategy,
    )
    ranked_item_ids = tuple(score.item_id for score in scored_candidates)
    selected_item_id = ranked_item_ids[0]
    run_id = f"heuristic_{strategy.value}"
    return OfflinePredictionRecord(
        example_id=example.example_id,
        selected_item_id=selected_item_id,
        ranked_item_ids=ranked_item_ids,
        run_id=run_id,
        model_name=strategy.value,
        prompt_style="answer_only",
        prompt_version="heuristic_v1",
        group_id=run_id,
        sample_index=0,
        metadata={
            "strategy": strategy.value,
            "target_item_id": example.target_item_id,
            "candidate_scores": [
                {
                    "item_id": score.item_id,
                    "total_score": round(score.total_score, 6),
                    "overlap_score": round(score.overlap_score, 6),
                    "popularity_score": round(score.popularity_score, 6),
                    "rating_score": round(score.rating_score, 6),
                }
                for score in scored_candidates[:5]
            ],
        },
    )


def score_candidates(
    example: NextItemExample,
    *,
    item_records: dict[str, ItemRecord],
    strategy: BaselinePredictionStrategy,
) -> list[CandidateScore]:
    """Score all candidates for one example and return them sorted descending."""

    max_rank_prior = max((candidate.rank_prior or 0.0 for candidate in example.candidates), default=0.0)
    history_feature_weights = _history_feature_weights(example, item_records=item_records)

    scored_candidates: list[CandidateScore] = []
    for order, candidate in enumerate(example.candidates):
        candidate_record = item_records.get(candidate.item_id, ItemRecord(item_id=candidate.item_id))
        popularity_score = (
            0.0
            if max_rank_prior <= 0.0
            else float(candidate.rank_prior or 0.0) / max_rank_prior
        )
        rating_score = _mean_rating_score(candidate_record)
        overlap_score = 0.0
        if strategy == BaselinePredictionStrategy.HISTORY_FEATURE_OVERLAP:
            overlap_score = sum(
                history_feature_weights.get(feature_ref, 0.0)
                for feature_ref in _informative_feature_refs(candidate_record.feature_refs)
            )

        total_score = _total_candidate_score(
            strategy=strategy,
            overlap_score=overlap_score,
            popularity_score=popularity_score,
            rating_score=rating_score,
        )
        scored_candidates.append(
            CandidateScore(
                item_id=candidate.item_id,
                total_score=total_score - (order * 1e-9),
                overlap_score=overlap_score,
                popularity_score=popularity_score,
                rating_score=rating_score,
            )
        )

    return sorted(
        scored_candidates,
        key=lambda score: (-score.total_score, score.item_id),
    )


def _history_feature_weights(
    example: NextItemExample,
    *,
    item_records: dict[str, ItemRecord],
) -> dict[str, float]:
    """Build a recency-weighted history feature bag."""

    history = example.history[-10:]
    if not history:
        return {}

    weights: dict[str, float] = {}
    total = float(len(history))
    for offset, event in enumerate(history, start=1):
        item_record = item_records.get(event.item_id, ItemRecord(item_id=event.item_id))
        recency_weight = offset / total
        for feature_ref in _informative_feature_refs(item_record.feature_refs):
            weights[feature_ref] = weights.get(feature_ref, 0.0) + recency_weight
    return weights


def _informative_feature_refs(feature_refs: tuple[str, ...]) -> tuple[str, ...]:
    """Keep feature refs that can plausibly discriminate candidates."""

    informative = tuple(
        feature_ref
        for feature_ref in feature_refs
        if feature_ref
        and feature_ref != "category:grocery_and_gourmet_food"
        and not feature_ref.startswith("review_count:")
        and not feature_ref.startswith("verified_ratio:")
    )
    return informative


def _mean_rating_score(item_record: ItemRecord) -> float:
    """Map mean rating into a small bounded tie-breaker score."""

    value = item_record.raw.get("mean_rating")
    if value is None:
        return 0.0
    rating = float(value)
    return max(0.0, min(1.0, (rating - 3.0) / 2.0))


def _total_candidate_score(
    *,
    strategy: BaselinePredictionStrategy,
    overlap_score: float,
    popularity_score: float,
    rating_score: float,
) -> float:
    """Combine score components for one baseline strategy."""

    if strategy == BaselinePredictionStrategy.POPULARITY:
        return popularity_score + (0.05 * rating_score)
    if strategy == BaselinePredictionStrategy.HISTORY_FEATURE_OVERLAP:
        return overlap_score + (0.1 * popularity_score) + (0.05 * rating_score)
    raise ValueError(f"unsupported strategy for candidate scoring: {strategy}")


def _ensure_dataset_config(config: DatasetConfig | str | Path) -> DatasetConfig:
    """Normalize config input into a DatasetConfig object."""

    if isinstance(config, DatasetConfig):
        return config
    return load_dataset_config(config)
