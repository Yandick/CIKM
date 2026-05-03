from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RerankOutput:
    ranking: list[str]
    selected_candidate_id: str
    evidence_refs: list[str]
    raw_text: str


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found")


def parse_final_answer(text: str) -> RerankOutput:
    answer_start = text.lower().rfind("final answer")
    search_text = text[answer_start:] if answer_start >= 0 else text
    obj = _first_json_object(search_text)

    ranking = obj.get("ranking")
    selected = obj.get("selected_candidate_id")
    evidence_refs = obj.get("evidence_refs", [])
    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
        raise ValueError("Final answer field 'ranking' must be a list of strings")
    if not isinstance(selected, str):
        raise ValueError("Final answer field 'selected_candidate_id' must be a string")
    if not isinstance(evidence_refs, list) or not all(isinstance(x, str) for x in evidence_refs):
        raise ValueError("Final answer field 'evidence_refs' must be a list of strings")
    return RerankOutput(
        ranking=ranking, selected_candidate_id=selected, evidence_refs=evidence_refs, raw_text=text
    )


def _dcg(hits: list[int]) -> float:
    return sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))


def ranking_metrics(
    ranking: list[str], positives: set[str], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    first_rank = next((idx for idx, item in enumerate(ranking, start=1) if item in positives), None)
    metrics["mrr"] = 0.0 if first_rank is None else 1.0 / first_rank
    for k in ks:
        top_k = ranking[:k]
        hits = [1 if item in positives else 0 for item in top_k]
        ideal = _dcg([1] * min(len(positives), k))
        metrics[f"hit@{k}"] = float(any(hits))
        metrics[f"recall@{k}"] = 0.0 if not positives else sum(hits) / len(positives)
        metrics[f"ndcg@{k}"] = 0.0 if ideal == 0 else _dcg(hits) / ideal
    return metrics


def format_errors(
    output: RerankOutput, candidate_ids: list[str], evidence_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    candidate_set = set(candidate_ids)
    if output.selected_candidate_id not in candidate_set:
        errors.append("selected_candidate_not_in_pool")
    if not output.ranking:
        errors.append("empty_ranking")
    elif output.ranking[0] != output.selected_candidate_id:
        errors.append("selected_not_first")
    if any(candidate_id not in candidate_set for candidate_id in output.ranking):
        errors.append("ranking_contains_unknown_candidate")
    if len(output.ranking) != len(candidate_ids):
        errors.append("ranking_length_mismatch")
    if len(set(output.ranking)) != len(output.ranking):
        errors.append("ranking_contains_duplicates")
    if set(output.ranking) != candidate_set:
        errors.append("ranking_candidate_set_mismatch")
    if any(ref not in evidence_ids for ref in output.evidence_refs):
        errors.append("invented_evidence_ref")
    if not output.evidence_refs:
        errors.append("empty_evidence_refs")
    return sorted(set(errors))


def evidence_validity(evidence_refs: list[str], evidence_ids: set[str]) -> float:
    if not evidence_refs:
        return 0.0
    return sum(1 for ref in evidence_refs if ref in evidence_ids) / len(evidence_refs)


def compute_score(
    solution_str: str,
    *,
    positive_candidate_ids: list[str],
    candidate_ids: list[str],
    evidence_ids: list[str],
    k: int = 1,
) -> dict[str, Any]:
    try:
        output = parse_final_answer(solution_str)
    except ValueError as exc:
        return {
            "score": -0.5,
            "parse_success": 0.0,
            "parse_error": str(exc),
            f"ndcg@{k}": 0.0,
            "format": 0.0,
            "evidence": 0.0,
        }

    evidence_set = set(evidence_ids) | set(candidate_ids)
    errors = format_errors(output, candidate_ids, evidence_set)
    metrics = ranking_metrics(output.ranking, set(positive_candidate_ids), ks=(1, 3, 5, 10, k))
    fmt = 1.0 if not errors else max(0.0, 1.0 - 0.2 * len(errors))
    evidence = evidence_validity(output.evidence_refs, evidence_set)
    recommendation = metrics.get(f"ndcg@{k}", metrics["ndcg@1"])
    score = recommendation + 0.2 * fmt + 0.2 * evidence
    return {
        "score": score,
        "parse_success": 1.0,
        "recommendation": recommendation,
        "format": fmt,
        "evidence": evidence,
        "selected_candidate_id": output.selected_candidate_id,
        "validation_errors": errors,
        **metrics,
    }


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    k: int = 1,
) -> dict[str, Any]:
    del data_source, ground_truth
    extra_info = extra_info or {}
    return compute_score(
        solution_str,
        positive_candidate_ids=list(extra_info.get("positive_candidate_ids") or []),
        candidate_ids=list(extra_info.get("candidate_ids") or []),
        evidence_ids=list(extra_info.get("evidence_ids") or []),
        k=k,
    )
