from __future__ import annotations

import json
import math
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


def _parse(text: str) -> tuple[list[str], str, list[str], list[dict[str, Any]]]:
    answer_start = text.lower().rfind("final answer")
    obj = _first_json_object(text[answer_start:] if answer_start >= 0 else text)
    ranking = obj.get("ranking")
    selected = obj.get("selected_candidate_id")
    evidence_refs = obj.get("evidence_refs", [])
    rationale = obj.get("rationale", [])
    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
        raise ValueError("ranking must be a list of strings")
    if not isinstance(selected, str):
        raise ValueError("selected_candidate_id must be a string")
    if not isinstance(evidence_refs, list) or not all(isinstance(x, str) for x in evidence_refs):
        raise ValueError("evidence_refs must be a list of strings")
    if not isinstance(rationale, list) or not all(isinstance(x, dict) for x in rationale):
        raise ValueError("rationale must be a list of objects")
    return ranking, selected, evidence_refs, rationale


def _dcg(hits: list[int]) -> float:
    return sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))


def _ndcg(ranking: list[str], positives: set[str], k: int) -> float:
    hits = [1 if item in positives else 0 for item in ranking[:k]]
    ideal = _dcg([1] * min(len(positives), k))
    return 0.0 if ideal == 0 else _dcg(hits) / ideal


def _rationale_errors(
    *,
    selected: str,
    evidence_refs: list[str],
    rationale: list[dict[str, Any]],
    candidate_set: set[str],
    evidence_ids: set[str],
    source_evidence_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    evidence_ref_set = set(evidence_refs)
    source_evidence_set = source_evidence_ids or set()
    selected_has_rationale = False
    selected_has_source_support = False

    if not rationale:
        errors.append("empty_rationale")
    if len(rationale) > 5:
        errors.append("rationale_too_long")

    for step in rationale:
        candidate_id = step.get("candidate_id")
        claim = step.get("claim")
        support = step.get("support")

        if not isinstance(candidate_id, str) or candidate_id not in candidate_set:
            errors.append("rationale_unknown_candidate")
        elif candidate_id == selected:
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
        if candidate_id == selected and source_evidence_set.intersection(support):
            selected_has_source_support = True

    if rationale and not selected_has_rationale:
        errors.append("selected_missing_rationale")
    if rationale and source_evidence_set and not selected_has_source_support:
        errors.append("selected_rationale_missing_source_support")
    return sorted(set(errors))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _faithrl_geometric_score(
    *,
    recommendation: float,
    format_score: float,
    evidence_score: float,
    rationale_score: float,
    baseline_correct_rate: float,
    baseline_unfaithful_rate: float,
) -> tuple[float, str, float, float]:
    correct = float(recommendation > 0.0)
    faithful = float(format_score >= 1.0 and evidence_score >= 1.0 and rationale_score >= 1.0)
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
    extra_info: dict[str, Any],
    k: int = 1,
    reward_mode: str = "weighted",
    baseline_correct_rate: float = DEFAULT_BASELINE_CORRECT_RATE,
    baseline_unfaithful_rate: float = DEFAULT_BASELINE_UNFAITHFUL_RATE,
    hybrid_weight: float = DEFAULT_HYBRID_WEIGHT,
) -> dict[str, float | list[str] | str]:
    baseline_correct_rate = _clamp01(baseline_correct_rate)
    baseline_unfaithful_rate = _clamp01(baseline_unfaithful_rate)
    hybrid_weight = max(0.0, float(hybrid_weight))
    candidate_ids = list(extra_info.get("candidate_ids") or [])
    positive_ids = set(extra_info.get("positive_candidate_ids") or [])
    evidence_ids = set(extra_info.get("evidence_ids") or []) | set(candidate_ids)
    source_evidence_ids = set(extra_info.get("source_evidence_ids") or []) or (
        evidence_ids - set(candidate_ids)
    )
    candidate_set = set(candidate_ids)

    try:
        ranking, selected, evidence_refs, rationale = _parse(solution_str)
    except ValueError as exc:
        weighted_score = -0.5
        geometric_score = -baseline_correct_rate
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
            "score_mode": reward_mode,
            "parse_success": 0.0,
            "parse_error": str(exc),
            "ndcg@k": 0.0,
            "rationale": 0.0,
        }

    errors: list[str] = []
    if selected not in candidate_set:
        errors.append("selected_candidate_not_in_pool")
    if not ranking or ranking[0] != selected:
        errors.append("selected_not_first")
    if set(ranking) != candidate_set:
        errors.append("ranking_candidate_set_mismatch")
    if len(ranking) != len(candidate_ids):
        errors.append("ranking_length_mismatch")
    if len(set(ranking)) != len(ranking):
        errors.append("ranking_contains_duplicates")
    if any(ref not in evidence_ids for ref in evidence_refs):
        errors.append("invented_evidence_ref")
    if not evidence_refs:
        errors.append("empty_evidence_refs")
    rationale_errs = _rationale_errors(
        selected=selected,
        evidence_refs=evidence_refs,
        rationale=rationale,
        candidate_set=candidate_set,
        evidence_ids=evidence_ids,
        source_evidence_ids=source_evidence_ids,
    )

    ndcg = _ndcg(ranking, positive_ids, k)
    fmt = 1.0 if not errors else max(0.0, 1.0 - 0.2 * len(set(errors)))
    evidence = (
        0.0
        if not evidence_refs
        else sum(1 for ref in evidence_refs if ref in evidence_ids) / len(evidence_refs)
    )
    rationale_score = 0.0 if not rationale else max(0.0, 1.0 - 0.2 * len(set(rationale_errs)))
    weighted_score = ndcg + 0.2 * fmt + 0.2 * evidence + 0.2 * rationale_score
    geometric_score, outcome, correctness, faithfulness = _faithrl_geometric_score(
        recommendation=ndcg,
        format_score=fmt,
        evidence_score=evidence,
        rationale_score=rationale_score,
        baseline_correct_rate=baseline_correct_rate,
        baseline_unfaithful_rate=baseline_unfaithful_rate,
    )
    return {
        "score": _select_score(
            reward_mode=reward_mode,
            weighted_score=weighted_score,
            geometric_score=geometric_score,
            hybrid_weight=hybrid_weight,
        ),
        "weighted_score": weighted_score,
        "geometric_score": geometric_score,
        "outcome": outcome,
        "correctness": correctness,
        "faithfulness": faithfulness,
        "score_mode": reward_mode,
        "parse_success": 1.0,
        "recommendation": ndcg,
        "format": fmt,
        "evidence": evidence,
        "rationale": rationale_score,
        "ndcg@k": ndcg,
        "validation_errors": sorted(set(errors + rationale_errs)),
    }


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    k=1,
    reward_mode="weighted",
    baseline_correct_rate=DEFAULT_BASELINE_CORRECT_RATE,
    baseline_unfaithful_rate=DEFAULT_BASELINE_UNFAITHFUL_RATE,
    hybrid_weight=DEFAULT_HYBRID_WEIGHT,
):
    del data_source, ground_truth
    return compute_score(
        solution_str,
        extra_info or {},
        k=k,
        reward_mode=reward_mode,
        baseline_correct_rate=baseline_correct_rate,
        baseline_unfaithful_rate=baseline_unfaithful_rate,
        hybrid_weight=hybrid_weight,
    )
