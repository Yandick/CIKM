from __future__ import annotations

from dataclasses import dataclass

from faithrec.metrics import ndcg_at_k
from faithrec.schema import RerankInstance, RerankOutput, validate_output


@dataclass(frozen=True)
class RewardWeights:
    rank: float = 1.0
    grounding: float = 0.25
    counterfactual: float = 0.25
    format: float = 0.20
    cost: float = 0.05


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    rank: float
    grounding: float
    counterfactual: float
    format: float
    cost: float
    errors: list[str]


def format_reward(output: RerankOutput, instance: RerankInstance) -> tuple[float, list[str]]:
    errors = validate_output(output, instance)
    if not errors:
        return 1.0, []

    penalty = 0.20 * len(set(errors))
    return max(0.0, 1.0 - penalty), errors


def grounding_reward(output: RerankOutput, instance: RerankInstance) -> float:
    """Initial grounding reward based on valid refs and concise evidence use.

    This is deliberately simple. Semantic grounding should be added later through
    category overlap, embedding similarity, or a learned verifier.
    """

    if not output.evidence_refs:
        return 0.0

    valid = [ref for ref in output.evidence_refs if ref in instance.evidence_ids]
    validity = len(valid) / len(output.evidence_refs)
    max_refs = 5
    concision = min(1.0, max_refs / max(len(output.evidence_refs), 1))
    return 0.7 * validity + 0.3 * concision


def rank_reward(output: RerankOutput, instance: RerankInstance, k: int = 5) -> float:
    if instance.target_candidate_id is None:
        return 0.0
    if output.selected_candidate_id == instance.target_candidate_id:
        return 1.0
    return ndcg_at_k(output.ranking, instance.target_candidate_id, k)


def compute_reward(
    output: RerankOutput,
    instance: RerankInstance,
    *,
    weights: RewardWeights = RewardWeights(),
    counterfactual_score: float = 0.0,
    completion_tokens: int = 0,
    completion_token_budget: int = 256,
) -> RewardBreakdown:
    fmt, errors = format_reward(output, instance)
    rank = rank_reward(output, instance)
    ground = grounding_reward(output, instance)
    cost = min(1.0, completion_tokens / max(completion_token_budget, 1))

    total = (
        weights.rank * rank
        + weights.grounding * ground
        + weights.counterfactual * counterfactual_score
        + weights.format * fmt
        - weights.cost * cost
    )
    return RewardBreakdown(
        total=total,
        rank=rank,
        grounding=ground,
        counterfactual=counterfactual_score,
        format=fmt,
        cost=cost,
        errors=errors,
    )
