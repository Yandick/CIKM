"""Offline verifier scoring and reward-guided reranking utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm4rec.data import (
    build_candidate_constructor_from_train_split,
    build_item_records_from_sources,
    collect_split_item_ids,
    iter_amazon_food_examples,
    load_dataset_config,
)
from llm4rec.data.schema import DatasetConfig, NextItemExample
from llm4rec.evaluation.verifier import (
    AnswerOnlyPrediction,
    AnswerOnlyVerifierResult,
    CounterfactualAuditCase,
    VerifierRewardWeights,
    verify_answer_only_prediction,
)
from llm4rec.evaluation.counterfactual import InterventionType
from llm4rec.evaluation.counterfactual import parse_eval_view
from llm4rec.inference.parsing import (
    extract_candidate_id_from_text,
    validate_prediction_consistency,
)



@dataclass(frozen=True)
class OfflinePredictionRecord:
    """One offline prediction input keyed by example id."""

    example_id: str
    selected_item_id: str | None = None
    response_text: str | None = None
    ranked_item_ids: tuple[str, ...] = ()
    audit_graph: dict[str, Any] | None = None
    counterfactual_audits: tuple[CounterfactualAuditCase, ...] = ()
    group_id: str | None = None
    sample_index: int | None = None
    run_id: str | None = None
    model_name: str | None = None
    prompt_style: str | None = None
    prompt_version: str | None = None
    generation_config: dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScoredPredictionRecord:
    """One prediction record after verifier scoring."""

    example_id: str
    group_id: str
    sample_index: int
    prediction: OfflinePredictionRecord
    verifier: AnswerOnlyVerifierResult


@dataclass(frozen=True)
class OfflineVerifierSummary:
    """Aggregate summary for an offline verifier scoring run."""

    total_predictions: int
    total_examples: int
    mean_reward: float
    schema_pass_rate: float
    graph_valid_rate: float
    hit_rate: float
    mean_reciprocal_rank: float
    best_of_n_hit_rate: float
    best_of_n_mean_reciprocal_rank: float


def build_example_index(
    config: DatasetConfig | str | Path,
    *,
    split: str,
    limit: int | None = None,
) -> dict[str, NextItemExample]:
    """Build an example lookup table for one split."""

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

    example_index: dict[str, NextItemExample] = {}
    for index, example in enumerate(examples):
        if limit is not None and index >= limit:
            break
        example_index[example.example_id] = example
    return example_index


def load_offline_prediction_records_jsonl(
    path: str | Path,
) -> list[OfflinePredictionRecord]:
    """Load offline prediction records from JSONL."""

    records: list[OfflinePredictionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            record = OfflinePredictionRecord(
                example_id=str(payload["example_id"]),
                selected_item_id=_optional_string(payload.get("selected_item_id")),
                response_text=_optional_string(payload.get("response_text")),
                ranked_item_ids=tuple(str(item) for item in payload.get("ranked_item_ids", []) or ()),
                audit_graph=_as_mapping(payload.get("audit_graph")),
                counterfactual_audits=_coerce_counterfactual_audits(
                    payload.get("counterfactual_audits", [])
                ),
                group_id=_optional_string(payload.get("group_id")),
                sample_index=_optional_int(payload.get("sample_index")),
                run_id=_optional_string(payload.get("run_id")),
                model_name=_optional_string(payload.get("model_name")),
                prompt_style=_optional_string(payload.get("prompt_style")),
                prompt_version=_optional_string(payload.get("prompt_version")),
                generation_config=dict(payload.get("generation_config", {}))
                if isinstance(payload.get("generation_config"), dict)
                else None,
                finish_reason=_optional_string(payload.get("finish_reason")),
                latency_ms=_optional_float(payload.get("latency_ms")),
                prompt_tokens=_optional_int(payload.get("prompt_tokens")),
                completion_tokens=_optional_int(payload.get("completion_tokens")),
                error=_optional_string(payload.get("error")),
                metadata=dict(payload.get("metadata", {}))
                if isinstance(payload.get("metadata"), dict)
                else None,
            )
            records.append(record)
    return records


def write_offline_prediction_records_jsonl(
    path: str | Path,
    records: Iterable[OfflinePredictionRecord],
) -> int:
    """Write raw offline prediction records as JSONL."""

    def rows() -> Iterable[dict[str, Any]]:
        for record in records:
            yield _json_ready(asdict(record))

    return _write_jsonl(path, rows())


def score_offline_prediction_records(
    example_index: dict[str, NextItemExample],
    prediction_records: Sequence[OfflinePredictionRecord],
    *,
    weights: VerifierRewardWeights | None = None,
) -> list[ScoredPredictionRecord]:
    """Score offline prediction records with the answer-only verifier."""

    scored_records: list[ScoredPredictionRecord] = []
    sample_counters: dict[str, int] = {}
    for record in prediction_records:
        example = example_index.get(record.example_id)
        if example is None:
            raise KeyError(f"prediction example_id not found in split index: {record.example_id}")

        validate_prediction_consistency(
            selected_item_id=record.selected_item_id,
            response_text=record.response_text,
            candidate_ids=example.candidate_ids(),
        )
        selected_item_id = record.selected_item_id or _extract_predicted_candidate_id(
            record.response_text or "",
            example.candidate_ids(),
        )
        if selected_item_id is None:
            selected_item_id = ""

        prediction = AnswerOnlyPrediction(
            selected_item_id=selected_item_id,
            ranked_item_ids=record.ranked_item_ids,
            response_text=record.response_text,
            audit_graph=record.audit_graph,
            metadata={} if record.metadata is None else record.metadata,
        )
        verifier_result = verify_answer_only_prediction(
            example,
            prediction,
            counterfactual_cases=record.counterfactual_audits,
            weights=weights,
        )

        sample_index = record.sample_index
        if sample_index is None:
            sample_index = sample_counters.get(record.example_id, 0)
            sample_counters[record.example_id] = sample_index + 1

        scored_records.append(
            ScoredPredictionRecord(
                example_id=record.example_id,
                group_id=record.group_id or record.run_id or record.example_id,
                sample_index=sample_index,
                prediction=record,
                verifier=verifier_result,
            )
        )
    return scored_records


def select_best_predictions_by_reward(
    scored_records: Sequence[ScoredPredictionRecord],
) -> list[ScoredPredictionRecord]:
    """Select the highest-reward prediction for each example/group pair."""

    best_by_example: dict[tuple[str, str], ScoredPredictionRecord] = {}
    for record in scored_records:
        key = (record.example_id, record.group_id)
        current = best_by_example.get(key)
        if current is None or _sort_key(record) > _sort_key(current):
            best_by_example[key] = record
    return [best_by_example[key] for key in sorted(best_by_example)]


def summarize_scored_predictions(
    scored_records: Sequence[ScoredPredictionRecord],
) -> OfflineVerifierSummary:
    """Aggregate offline verifier scoring outputs into summary metrics."""

    if not scored_records:
        raise ValueError("scored_records must be non-empty")

    total_predictions = len(scored_records)
    total_examples = len({record.example_id for record in scored_records})
    mean_reward = sum(record.verifier.reward for record in scored_records) / float(total_predictions)
    schema_pass_rate = (
        sum(1.0 for record in scored_records if record.verifier.components.schema_passed)
        / float(total_predictions)
    )
    graph_valid_rate = (
        sum(
            1.0
            for record in scored_records
            if (not record.verifier.audit_graph_present) or record.verifier.audit_graph_valid
        )
        / float(total_predictions)
    )
    hit_rate = (
        sum(1.0 for record in scored_records if record.verifier.exact_hit)
        / float(total_predictions)
    )
    mean_reciprocal_rank = (
        sum(record.verifier.reciprocal_rank for record in scored_records)
        / float(total_predictions)
    )

    best_records = select_best_predictions_by_reward(scored_records)
    best_total = float(len(best_records))
    best_of_n_hit_rate = sum(1.0 for record in best_records if record.verifier.exact_hit) / best_total
    best_of_n_mean_reciprocal_rank = (
        sum(record.verifier.reciprocal_rank for record in best_records) / best_total
    )
    return OfflineVerifierSummary(
        total_predictions=total_predictions,
        total_examples=total_examples,
        mean_reward=mean_reward,
        schema_pass_rate=schema_pass_rate,
        graph_valid_rate=graph_valid_rate,
        hit_rate=hit_rate,
        mean_reciprocal_rank=mean_reciprocal_rank,
        best_of_n_hit_rate=best_of_n_hit_rate,
        best_of_n_mean_reciprocal_rank=best_of_n_mean_reciprocal_rank,
    )


def write_scored_prediction_records_jsonl(
    path: str | Path,
    records: Iterable[ScoredPredictionRecord],
) -> int:
    """Write scored prediction records as JSONL."""

    def rows() -> Iterable[dict[str, Any]]:
        for record in records:
            row = _json_ready(asdict(record))
            yield row

    return _write_jsonl(path, rows())


def write_offline_summary_json(
    path: str | Path,
    summary: OfflineVerifierSummary,
) -> None:
    """Write an offline verifier summary as one JSON file."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _coerce_counterfactual_audits(payloads: Any) -> tuple[CounterfactualAuditCase, ...]:
    """Parse a JSON-like payload into counterfactual audit cases."""

    if not isinstance(payloads, list):
        return ()

    cases: list[CounterfactualAuditCase] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        intervention_raw = payload.get("intervention")
        try:
            intervention = InterventionType(str(intervention_raw))
        except Exception:
            continue

        updated_graph = _as_mapping(payload.get("updated_graph"))
        updated_eval_view = None
        if isinstance(payload.get("updated_eval_view"), dict):
            updated_eval_view = parse_eval_view(payload["updated_eval_view"], strict=False)

        cases.append(
            CounterfactualAuditCase(
                intervention=intervention,
                updated_graph=updated_graph,
                updated_eval_view=updated_eval_view,
                focus_candidate_id=_optional_string(payload.get("focus_candidate_id")),
            )
        )
    return tuple(cases)


def _extract_predicted_candidate_id(
    text: str,
    candidate_ids: Sequence[str],
) -> str | None:
    """Extract one candidate id from a text response."""

    return extract_candidate_id_from_text(text, candidate_ids)


def _sort_key(record: ScoredPredictionRecord) -> tuple[float, float, int]:
    """Stable ordering for best-of-n selection."""

    return (
        record.verifier.reward,
        record.verifier.reciprocal_rank,
        -record.sample_index,
    )


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write JSON-serializable rows to JSONL."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _json_ready(value: Any) -> Any:
    """Convert nested dataclasses/enums/tuples into JSON-ready structures."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            return value
    return value


def _ensure_dataset_config(config: DatasetConfig | str | Path) -> DatasetConfig:
    """Normalize config input into a DatasetConfig object."""

    if isinstance(config, DatasetConfig):
        return config
    return load_dataset_config(config)


def _optional_string(value: Any) -> str | None:
    """Convert a scalar to string when present."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    """Convert a scalar to int when present."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    """Convert a scalar to float when present."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_mapping(value: Any) -> dict[str, Any] | None:
    """Normalize arbitrary JSON-like values into dicts when possible."""

    if isinstance(value, dict):
        return value
    return None
