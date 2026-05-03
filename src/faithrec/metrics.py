from __future__ import annotations

import math


def rank_of(ranking: list[str], target: str) -> int | None:
    """Return 1-based rank, or None when target is absent."""

    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def hit_at_k(ranking: list[str], target: str, k: int) -> float:
    rank = rank_of(ranking, target)
    return float(rank is not None and rank <= k)


def ndcg_at_k(ranking: list[str], target: str, k: int) -> float:
    rank = rank_of(ranking, target)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr(ranking: list[str], target: str) -> float:
    rank = rank_of(ranking, target)
    if rank is None:
        return 0.0
    return 1.0 / rank


def ranking_metrics(ranking: list[str], target: str, ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    metrics: dict[str, float] = {"mrr": mrr(ranking, target)}
    for k in ks:
        metrics[f"hr@{k}"] = hit_at_k(ranking, target, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranking, target, k)
    return metrics


def targeted_evidence_sensitivity(
    original_ranking: list[str], audited_ranking: list[str], selected_candidate_id: str
) -> float:
    """Return 1 when targeted evidence removal worsens the selected candidate rank."""

    before = rank_of(original_ranking, selected_candidate_id)
    after = rank_of(audited_ranking, selected_candidate_id)
    if before is None or after is None:
        return 0.0
    return float(after > before)


def irrelevant_evidence_stability(
    original_selected: str, audited_ranking: list[str]
) -> float:
    return float(bool(audited_ranking) and audited_ranking[0] == original_selected)
