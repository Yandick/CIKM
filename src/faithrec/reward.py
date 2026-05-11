from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

ALLOWED_RATIONALE_CLAIMS = {
    "matches_query",
    "partially_matches_query",
    "contrasts_with_query",
    "supports_ranking",
}
DEFAULT_BASELINE_CORRECT_RATE = 0.5
DEFAULT_BASELINE_UNFAITHFUL_RATE = 0.5
DEFAULT_HYBRID_WEIGHT = 0.1
RANKING_FORMAT_ERRORS = {
    "selected_candidate_not_in_pool",
    "empty_ranking",
    "selected_not_first",
    "ranking_contains_unknown_candidate",
    "ranking_length_mismatch",
    "ranking_contains_duplicates",
    "ranking_candidate_set_mismatch",
}


@dataclass
class RerankOutput:
    ranking: list[str]
    selected_candidate_id: str
    evidence_refs: list[str]
    rationale: list[dict[str, Any]]
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
    rationale = obj.get("rationale", [])
    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
        raise ValueError("Final answer field 'ranking' must be a list of strings")
    if not isinstance(selected, str):
        raise ValueError("Final answer field 'selected_candidate_id' must be a string")
    if not isinstance(evidence_refs, list) or not all(isinstance(x, str) for x in evidence_refs):
        raise ValueError("Final answer field 'evidence_refs' must be a list of strings")
    if not isinstance(rationale, list) or not all(isinstance(x, dict) for x in rationale):
        raise ValueError("Final answer field 'rationale' must be a list of objects")
    return RerankOutput(
        ranking=ranking,
        selected_candidate_id=selected,
        evidence_refs=evidence_refs,
        rationale=rationale,
        raw_text=text,
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
        seen_positives: set[str] = set()
        hits = []
        for item in top_k:
            if item in positives and item not in seen_positives:
                hits.append(1)
                seen_positives.add(item)
            else:
                hits.append(0)
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


def rationale_errors(
    output: RerankOutput,
    candidate_ids: list[str],
    evidence_ids: set[str],
    source_evidence_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    candidate_set = set(candidate_ids)
    evidence_ref_set = set(output.evidence_refs)
    source_evidence_set = source_evidence_ids or set()
    selected_has_rationale = False
    selected_has_source_support = False

    if not output.rationale:
        errors.append("empty_rationale")
    if len(output.rationale) > 5:
        errors.append("rationale_too_long")

    for step in output.rationale:
        candidate_id = step.get("candidate_id")
        claim = step.get("claim")
        support = step.get("support")

        if not isinstance(candidate_id, str) or candidate_id not in candidate_set:
            errors.append("rationale_unknown_candidate")
        elif candidate_id == output.selected_candidate_id:
            selected_has_rationale = True

        if claim not in ALLOWED_RATIONALE_CLAIMS:
            errors.append("rationale_unknown_claim")

        if not isinstance(support, list) or not all(isinstance(ref, str) for ref in support):
            errors.append("rationale_support_not_string_list")
            continue
        if not support:
            errors.append("empty_rationale_support")
        if len(support) > 5:
            errors.append("rationale_support_too_long")
        if any(ref not in evidence_ids for ref in support):
            errors.append("rationale_invented_support")
        if any(ref not in evidence_ref_set for ref in support):
            errors.append("rationale_support_not_in_evidence_refs")
        if isinstance(candidate_id, str) and candidate_id in candidate_set and candidate_id not in support:
            errors.append("rationale_missing_candidate_support")
        if candidate_id == output.selected_candidate_id and source_evidence_set.intersection(support):
            selected_has_source_support = True

    if output.rationale and not selected_has_rationale:
        errors.append("selected_missing_rationale")
    if output.rationale and source_evidence_set and not selected_has_source_support:
        errors.append("selected_rationale_missing_source_support")
    return sorted(set(errors))


def rationale_validity(errors: list[str]) -> float:
    return max(0.0, 1.0 - 0.2 * len(set(errors)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _faithrl_geometric_score(
    *,
    top1_score: float,
    faithful_binary_score: float,
    baseline_correct_rate: float,
    baseline_unfaithful_rate: float,
) -> tuple[float, str, float, float]:
    correct = float(top1_score > 0.0)
    faithful = float(faithful_binary_score >= 1.0)
    if correct and faithful:
        return baseline_unfaithful_rate, "correct_faithful", correct, faithful
    if faithful:
        return 0.0, "wrong_faithful", correct, faithful
    return -baseline_correct_rate, "unfaithful", correct, faithful


def _select_score(
    *,
    reward_mode: str,
    weighted_score: float,
    geometric_score: float,
    hybrid_weight: float,
) -> float:
    if reward_mode == "weighted":
        return weighted_score
    if reward_mode == "faithrl":
        return geometric_score
    if reward_mode == "faithrl_hybrid":
        return geometric_score + hybrid_weight * weighted_score
    raise ValueError(f"Unknown reward_mode: {reward_mode}")


def compute_score(
    solution_str: str,
    *,
    positive_candidate_ids: list[str],
    candidate_ids: list[str],
    evidence_ids: list[str],
    source_evidence_ids: list[str] | None = None,
    k: int = 1,
    reward_mode: str = "weighted",
    baseline_correct_rate: float = DEFAULT_BASELINE_CORRECT_RATE,
    baseline_unfaithful_rate: float = DEFAULT_BASELINE_UNFAITHFUL_RATE,
    hybrid_weight: float = DEFAULT_HYBRID_WEIGHT,
) -> dict[str, Any]:
    k = max(1, int(k))
    baseline_correct_rate = _clamp01(baseline_correct_rate)
    baseline_unfaithful_rate = _clamp01(baseline_unfaithful_rate)
    hybrid_weight = max(0.0, float(hybrid_weight))
    try:
        output = parse_final_answer(solution_str)
    except ValueError as exc:
        weighted_score = -0.5
        geometric_score = -baseline_correct_rate
        metric_ks = sorted({1, 3, 5, 10, k})
        metrics = {}
        for metric_k in metric_ks:
            metrics[f"hit@{metric_k}"] = 0.0
            metrics[f"recall@{metric_k}"] = 0.0
            metrics[f"mrr@{metric_k}"] = 0.0
            metrics[f"ndcg@{metric_k}"] = 0.0
        return {
            "score": _select_score(
                reward_mode=reward_mode,
                weighted_score=weighted_score,
                geometric_score=geometric_score,
                hybrid_weight=hybrid_weight,
            ),
            "weighted_score": weighted_score,
            "geometric_score": geometric_score,
            "outcome": "parse_error",
            "correctness": 0.0,
            "faithfulness": 0.0,
            "faithful_binary": 0.0,
            "score_mode": reward_mode,
            "parse_success": 0.0,
            "parse_error": str(exc),
            "recommendation": 0.0,
            "reward_k": k,
            "format": 0.0,
            "evidence": 0.0,
            "rationale": 0.0,
            "selected_candidate_id": "",
            "validation_errors": ["parse_error"],
            **metrics,
        }

    evidence_set = set(evidence_ids) | set(candidate_ids)
    errors = format_errors(output, candidate_ids, evidence_set)
    source_evidence_set = set(source_evidence_ids or []) or (set(evidence_ids) - set(candidate_ids))
    rationale_errs = rationale_errors(
        output, candidate_ids, evidence_set, source_evidence_ids=source_evidence_set
    )
    metrics = ranking_metrics(output.ranking, set(positive_candidate_ids), ks=(1, 3, 5, 10, k))
    ranking_is_valid = not (set(errors) & RANKING_FORMAT_ERRORS)
    if not ranking_is_valid:
        for metric_k in {1, 3, 5, 10, k}:
            metrics[f"hit@{metric_k}"] = 0.0
            metrics[f"recall@{metric_k}"] = 0.0
            metrics[f"ndcg@{metric_k}"] = 0.0
        metrics["mrr"] = 0.0
    fmt = 1.0 if not errors else max(0.0, 1.0 - 0.2 * len(errors))
    evidence = evidence_validity(output.evidence_refs, evidence_set)
    rationale = 0.0 if not output.rationale else rationale_validity(rationale_errs)
    recommendation = metrics.get(f"ndcg@{k}", metrics["ndcg@1"])
    top1_score = metrics["ndcg@1"]
    faithfulness = (fmt + evidence + rationale) / 3.0
    faithful_binary = float(fmt >= 1.0 and evidence >= 1.0 and rationale >= 1.0)
    weighted_score = recommendation + 0.2 * fmt + 0.2 * evidence + 0.2 * rationale
    geometric_score, outcome, correctness, faithful_exact = _faithrl_geometric_score(
        top1_score=top1_score,
        faithful_binary_score=faithful_binary,
        baseline_correct_rate=baseline_correct_rate,
        baseline_unfaithful_rate=baseline_unfaithful_rate,
    )
    score = _select_score(
        reward_mode=reward_mode,
        weighted_score=weighted_score,
        geometric_score=geometric_score,
        hybrid_weight=hybrid_weight,
    )
    return {
        "score": score,
        "weighted_score": weighted_score,
        "geometric_score": geometric_score,
        "outcome": outcome,
        "correctness": correctness,
        "faithfulness": faithfulness,
        "faithful_binary": faithful_exact,
        "score_mode": reward_mode,
        "parse_success": 1.0,
        "parse_error": "",
        "recommendation": recommendation,
        "reward_k": k,
        "format": fmt,
        "evidence": evidence,
        "rationale": rationale,
        "selected_candidate_id": output.selected_candidate_id,
        "validation_errors": sorted(set(errors + rationale_errs)),
        **metrics,
    }


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    k: int = 1,
    reward_mode: str = "weighted",
    baseline_correct_rate: float = DEFAULT_BASELINE_CORRECT_RATE,
    baseline_unfaithful_rate: float = DEFAULT_BASELINE_UNFAITHFUL_RATE,
    hybrid_weight: float = DEFAULT_HYBRID_WEIGHT,
) -> dict[str, Any]:
    del data_source, ground_truth
    extra_info = extra_info or {}
    return compute_score(
        solution_str,
        positive_candidate_ids=list(extra_info.get("positive_candidate_ids") or []),
        candidate_ids=list(extra_info.get("candidate_ids") or []),
        evidence_ids=list(extra_info.get("evidence_ids") or []),
        source_evidence_ids=list(extra_info.get("source_evidence_ids") or []),
        k=k,
        reward_mode=reward_mode,
        baseline_correct_rate=baseline_correct_rate,
        baseline_unfaithful_rate=baseline_unfaithful_rate,
        hybrid_weight=hybrid_weight,
    )
