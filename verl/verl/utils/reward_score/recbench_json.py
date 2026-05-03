from __future__ import annotations

import json
import math
from typing import Any


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


def _parse(text: str) -> tuple[list[str], str, list[str]]:
    answer_start = text.lower().rfind("final answer")
    obj = _first_json_object(text[answer_start:] if answer_start >= 0 else text)
    ranking = obj.get("ranking")
    selected = obj.get("selected_candidate_id")
    evidence_refs = obj.get("evidence_refs", [])
    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
        raise ValueError("ranking must be a list of strings")
    if not isinstance(selected, str):
        raise ValueError("selected_candidate_id must be a string")
    if not isinstance(evidence_refs, list) or not all(isinstance(x, str) for x in evidence_refs):
        raise ValueError("evidence_refs must be a list of strings")
    return ranking, selected, evidence_refs


def _dcg(hits: list[int]) -> float:
    return sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))


def _ndcg(ranking: list[str], positives: set[str], k: int) -> float:
    hits = [1 if item in positives else 0 for item in ranking[:k]]
    ideal = _dcg([1] * min(len(positives), k))
    return 0.0 if ideal == 0 else _dcg(hits) / ideal


def compute_score(
    solution_str: str, extra_info: dict[str, Any], k: int = 1
) -> dict[str, float | list[str] | str]:
    candidate_ids = list(extra_info.get("candidate_ids") or [])
    positive_ids = set(extra_info.get("positive_candidate_ids") or [])
    evidence_ids = set(extra_info.get("evidence_ids") or []) | set(candidate_ids)
    candidate_set = set(candidate_ids)

    try:
        ranking, selected, evidence_refs = _parse(solution_str)
    except ValueError as exc:
        return {
            "score": -0.5,
            "parse_success": 0.0,
            "parse_error": str(exc),
            "ndcg@k": 0.0,
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

    ndcg = _ndcg(ranking, positive_ids, k)
    fmt = 1.0 if not errors else max(0.0, 1.0 - 0.2 * len(set(errors)))
    evidence = (
        0.0
        if not evidence_refs
        else sum(1 for ref in evidence_refs if ref in evidence_ids) / len(evidence_refs)
    )
    return {
        "score": ndcg + 0.2 * fmt + 0.2 * evidence,
        "parse_success": 1.0,
        "recommendation": ndcg,
        "format": fmt,
        "evidence": evidence,
        "ndcg@k": ndcg,
        "validation_errors": sorted(set(errors)),
    }


def reward_func(data_source, solution_str, ground_truth, extra_info=None, k=1):
    del data_source, ground_truth
    return compute_score(solution_str, extra_info or {}, k=k)
