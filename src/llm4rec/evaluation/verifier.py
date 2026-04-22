"""Programmatic verifier and reward decomposition for answer-only recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from llm4rec.data.schema import NextItemExample
from llm4rec.evaluation.counterfactual import (
    CounterfactualCaseScore,
    CounterfactualEvalView,
    InterventionType,
    score_counterfactual_case,
    score_counterfactual_graph_case,
)
from llm4rec.reasoning.trace_rec_v1 import (
    CandidateEvidenceNode,
    GraphEdgeType,
    PreferenceStateNode,
    TraceRecGraph,
    ValidationResult,
    parse_trace_rec_graph,
    validate_trace_rec_graph,
)


@dataclass(frozen=True)
class AnswerOnlyPrediction:
    """One answer-only policy output, with optional audit artifacts."""

    selected_item_id: str
    ranked_item_ids: tuple[str, ...] = ()
    response_text: str | None = None
    audit_graph: dict[str, Any] | TraceRecGraph | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_ranking(self) -> tuple[str, ...]:
        """Return a de-duplicated ranking surface with the selected item first."""

        ranking: list[str] = []
        if self.selected_item_id:
            ranking.append(self.selected_item_id)
        for item_id in self.ranked_item_ids:
            if item_id and item_id not in ranking:
                ranking.append(item_id)
        return tuple(ranking)


@dataclass(frozen=True)
class CounterfactualAuditCase:
    """One optional counterfactual audit attached to a prediction."""

    intervention: InterventionType
    updated_graph: dict[str, Any] | TraceRecGraph | None = None
    updated_eval_view: CounterfactualEvalView | None = None
    focus_candidate_id: str | None = None


@dataclass(frozen=True)
class GraphEvidenceProfile:
    """Compact graph-side evidence summary for the selected candidate."""

    selected_candidate_present: bool
    selected_support_count: int
    selected_conflict_count: int
    contrasted_conflict_count: int


@dataclass(frozen=True)
class CounterfactualVerifierSummary:
    """Aggregate of counterfactual verifier scores over multiple audits."""

    case_count: int
    targeted_update_accuracy: float
    decision_direction_consistency: float
    non_target_stability: float
    final_choice_change_rate: float


@dataclass(frozen=True)
class VerifierRewardWeights:
    """Default reward weights for offline scoring and later RL."""

    utility_weight: float = 1.0
    counterfactual_weight: float = 1.0
    grounding_weight: float = 0.5
    consistency_weight: float = 0.5
    support_weight: float = 0.5
    conflict_weight: float = 0.25
    locality_weight: float = 0.25
    cost_penalty_weight: float = 0.1


@dataclass(frozen=True)
class VerifierComponentScores:
    """Decomposed reward components before weight aggregation."""

    schema_passed: bool
    utility: float
    grounding: float | None
    consistency: float | None
    support: float | None
    conflict: float | None
    counterfactual: float | None
    locality: float | None
    cost_penalty: float


@dataclass(frozen=True)
class AnswerOnlyVerifierResult:
    """Structured verifier result for one answer-only prediction."""

    selected_item_id: str
    answer_in_candidate_set: bool
    exact_hit: bool
    reciprocal_rank: float
    audit_graph_present: bool
    audit_graph_valid: bool
    audit_graph_selected_matches_prediction: bool | None
    audit_graph_grounded: bool | None
    graph_evidence_profile: GraphEvidenceProfile | None
    graph_validation: ValidationResult | None
    counterfactual_summary: CounterfactualVerifierSummary | None
    counterfactual_case_scores: tuple[CounterfactualCaseScore, ...]
    components: VerifierComponentScores
    reward: float


def verify_answer_only_prediction(
    example: NextItemExample,
    prediction: AnswerOnlyPrediction,
    *,
    counterfactual_cases: Sequence[CounterfactualAuditCase] = (),
    weights: VerifierRewardWeights | None = None,
) -> AnswerOnlyVerifierResult:
    """Verify and reward one answer-only prediction with optional audit artifacts."""

    reward_weights = weights or VerifierRewardWeights()
    candidate_ids = set(example.candidate_ids())
    answer_in_candidate_set = prediction.selected_item_id in candidate_ids
    exact_hit = prediction.selected_item_id == example.target_item_id
    reciprocal_rank = _reciprocal_rank(example.target_item_id, prediction.normalized_ranking())

    parsed_graph: TraceRecGraph | None = None
    graph_validation: ValidationResult | None = None
    graph_valid = False
    graph_selected_matches_prediction: bool | None = None
    graph_grounded: bool | None = None
    graph_profile: GraphEvidenceProfile | None = None

    if prediction.audit_graph is not None:
        graph_validation = validate_trace_rec_graph(prediction.audit_graph)
        graph_valid = graph_validation.valid
        if graph_valid:
            parsed_graph = (
                prediction.audit_graph
                if isinstance(prediction.audit_graph, TraceRecGraph)
                else parse_trace_rec_graph(prediction.audit_graph, strict=True)
            )
            graph_selected_item_id = _selected_item_id(parsed_graph)
            graph_selected_matches_prediction = (
                graph_selected_item_id == prediction.selected_item_id
            )
            graph_grounded = _graph_is_grounded(graph_validation)
            graph_profile = _graph_evidence_profile(
                parsed_graph,
                selected_item_id=prediction.selected_item_id,
            )
        else:
            graph_selected_matches_prediction = False
            graph_grounded = False

    counterfactual_case_scores = _score_counterfactual_cases(
        prediction=prediction,
        counterfactual_cases=counterfactual_cases,
    )
    counterfactual_summary = _summarize_counterfactual_scores(counterfactual_case_scores)

    schema_passed = answer_in_candidate_set and (
        prediction.audit_graph is None
        or (
            graph_valid
            and bool(graph_selected_matches_prediction)
        )
    )
    components = VerifierComponentScores(
        schema_passed=schema_passed,
        utility=reciprocal_rank,
        grounding=(
            None
            if prediction.audit_graph is None
            else 1.0 if graph_grounded else 0.0
        ),
        consistency=(
            None
            if prediction.audit_graph is None
            else 1.0 if graph_selected_matches_prediction else 0.0
        ),
        support=(
            None
            if graph_profile is None
            else 1.0 if graph_profile.selected_support_count > 0 else 0.0
        ),
        conflict=(
            None
            if graph_profile is None
            else min(1.0, float(graph_profile.contrasted_conflict_count))
        ),
        counterfactual=(
            None if counterfactual_summary is None else counterfactual_summary.targeted_update_accuracy
        ),
        locality=(
            None if counterfactual_summary is None else counterfactual_summary.non_target_stability
        ),
        cost_penalty=_cost_penalty(
            prediction=prediction,
            graph=parsed_graph,
        ),
    )
    reward = aggregate_verifier_reward(components, weights=reward_weights)

    return AnswerOnlyVerifierResult(
        selected_item_id=prediction.selected_item_id,
        answer_in_candidate_set=answer_in_candidate_set,
        exact_hit=exact_hit,
        reciprocal_rank=reciprocal_rank,
        audit_graph_present=prediction.audit_graph is not None,
        audit_graph_valid=graph_valid,
        audit_graph_selected_matches_prediction=graph_selected_matches_prediction,
        audit_graph_grounded=graph_grounded,
        graph_evidence_profile=graph_profile,
        graph_validation=graph_validation,
        counterfactual_summary=counterfactual_summary,
        counterfactual_case_scores=counterfactual_case_scores,
        components=components,
        reward=reward,
    )


def aggregate_verifier_reward(
    components: VerifierComponentScores,
    *,
    weights: VerifierRewardWeights | None = None,
) -> float:
    """Aggregate reward components into one scalar with a hard schema gate."""

    if not components.schema_passed:
        return 0.0

    reward_weights = weights or VerifierRewardWeights()
    positive_terms = [
        (components.utility, reward_weights.utility_weight),
        (components.counterfactual, reward_weights.counterfactual_weight),
        (components.grounding, reward_weights.grounding_weight),
        (components.consistency, reward_weights.consistency_weight),
        (components.support, reward_weights.support_weight),
        (components.conflict, reward_weights.conflict_weight),
        (components.locality, reward_weights.locality_weight),
    ]
    positive_total = 0.0
    positive_weight = 0.0
    for value, weight in positive_terms:
        if value is None:
            continue
        positive_total += weight * value
        positive_weight += weight

    if positive_weight == 0.0:
        base_reward = 0.0
    else:
        base_reward = positive_total / positive_weight

    reward = base_reward - (reward_weights.cost_penalty_weight * components.cost_penalty)
    return max(0.0, min(1.0, reward))


def _graph_evidence_profile(
    graph: TraceRecGraph,
    *,
    selected_item_id: str,
) -> GraphEvidenceProfile:
    """Summarize support/conflict evidence for the selected candidate."""

    node_lookup = graph.node_by_id()
    selected_node_ids = {
        node.id
        for node in graph.nodes
        if isinstance(node, CandidateEvidenceNode) and node.candidate_id == selected_item_id
    }
    non_selected_node_ids = {
        node.id
        for node in graph.nodes
        if isinstance(node, CandidateEvidenceNode) and node.candidate_id != selected_item_id
    }

    selected_support_count = 0
    selected_conflict_count = 0
    contrasted_conflict_count = 0
    for edge in graph.edges:
        if edge.type not in {GraphEdgeType.SUPPORTS, GraphEdgeType.CONFLICTS}:
            continue
        source = node_lookup.get(edge.source)
        if not isinstance(source, PreferenceStateNode):
            continue

        if edge.target in selected_node_ids:
            if edge.type == GraphEdgeType.SUPPORTS:
                selected_support_count += 1
            elif edge.type == GraphEdgeType.CONFLICTS:
                selected_conflict_count += 1
        elif edge.target in non_selected_node_ids and edge.type == GraphEdgeType.CONFLICTS:
            contrasted_conflict_count += 1

    return GraphEvidenceProfile(
        selected_candidate_present=bool(selected_node_ids),
        selected_support_count=selected_support_count,
        selected_conflict_count=selected_conflict_count,
        contrasted_conflict_count=contrasted_conflict_count,
    )


def _score_counterfactual_cases(
    *,
    prediction: AnswerOnlyPrediction,
    counterfactual_cases: Sequence[CounterfactualAuditCase],
) -> tuple[CounterfactualCaseScore, ...]:
    """Score all attached counterfactual audits for one prediction."""

    case_scores: list[CounterfactualCaseScore] = []
    for case in counterfactual_cases:
        if case.updated_eval_view is not None:
            case_scores.append(
                score_counterfactual_case(
                    case.updated_eval_view,
                    case.intervention,
                    original_choice=prediction.selected_item_id,
                    focus_candidate_id=case.focus_candidate_id,
                )
            )
            continue

        if case.updated_graph is not None and prediction.audit_graph is not None:
            case_scores.append(
                score_counterfactual_graph_case(
                    prediction.audit_graph,
                    case.updated_graph,
                    case.intervention,
                    focus_candidate_id=case.focus_candidate_id or prediction.selected_item_id,
                )
            )
    return tuple(case_scores)


def _summarize_counterfactual_scores(
    scores: Sequence[CounterfactualCaseScore],
) -> CounterfactualVerifierSummary | None:
    """Aggregate case-level counterfactual scores into split-friendly means."""

    if not scores:
        return None

    total = float(len(scores))
    final_choice_changed = [
        score.final_choice_changed
        for score in scores
        if score.final_choice_changed is not None
    ]
    return CounterfactualVerifierSummary(
        case_count=len(scores),
        targeted_update_accuracy=(
            sum(1.0 for score in scores if score.targeted_update_correct) / total
        ),
        decision_direction_consistency=(
            sum(1.0 for score in scores if score.decision_direction_consistent) / total
        ),
        non_target_stability=(
            sum(1.0 for score in scores if score.non_target_stable) / total
        ),
        final_choice_change_rate=(
            0.0
            if not final_choice_changed
            else sum(1.0 for changed in final_choice_changed if changed)
            / float(len(final_choice_changed))
        ),
    )


def _graph_is_grounded(validation: ValidationResult) -> bool:
    """Whether a validation result is free of unknown-pointer issues."""

    return not any(
        issue.code in {"unknown_evidence_ref", "unknown_feature_ref"}
        for issue in validation.issues
    )


def _cost_penalty(
    *,
    prediction: AnswerOnlyPrediction,
    graph: TraceRecGraph | None,
) -> float:
    """Small compactness penalty used only as a weak regularizer."""

    answer_tokens = 0
    if prediction.response_text:
        answer_tokens = len(prediction.response_text.split())
    elif prediction.selected_item_id:
        answer_tokens = 1

    answer_penalty = max(answer_tokens - 8, 0) / 8.0
    graph_penalty = 0.0
    if graph is not None:
        graph_size = len(graph.nodes) + len(graph.edges)
        graph_penalty = max(graph_size - 8, 0) / 8.0
    return min(1.0, answer_penalty + graph_penalty)


def _reciprocal_rank(target_item_id: str, ranked_item_ids: Sequence[str]) -> float:
    """Return reciprocal rank for a target item within a candidate ranking."""

    for index, item_id in enumerate(ranked_item_ids, start=1):
        if item_id == target_item_id:
            return 1.0 / float(index)
    return 0.0


def _selected_item_id(graph: TraceRecGraph) -> str:
    """Return the selected item id from the decision node."""

    for node in graph.nodes:
        if hasattr(node, "selected_item_id"):
            return str(node.selected_item_id)
    raise ValueError("graph must contain a decision node")
