"""Counterfactual eval_view validation and graph-derived scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from llm4rec.reasoning.trace_rec_v1 import (
    CandidateEvidenceNode,
    GraphEdgeType,
    PreferenceHorizon,
    PreferencePolarity,
    PreferenceStateNode,
    TraceRecGraph,
    ValidationResult,
    parse_trace_rec_graph,
)


class EvalDirection(str, Enum):
    UP = "up"
    SAME = "same"
    DOWN = "down"


class InterventionType(str, Enum):
    RECENT_SUPPORT_REMOVAL = "recent_support_removal"
    LONG_SUPPORT_REMOVAL = "long_support_removal"
    CANDIDATE_FEATURE_SWAP = "candidate_feature_swap"
    HARD_AVERSION_INJECTION = "hard_aversion_injection"


@dataclass(frozen=True)
class CounterfactualEvalView:
    """Compact surface used to score TRACE-Rec and free-form CoT consistently."""

    recent_support: EvalDirection
    long_support: EvalDirection
    aversion: EvalDirection
    candidate_match: EvalDirection
    final_choice: str


@dataclass(frozen=True)
class GraphFactorSnapshot:
    """Graph-derived factor counts for one focus candidate."""

    recent_support: int
    long_support: int
    aversion: int
    candidate_feature_refs: tuple[str, ...]
    final_choice: str


class EvalViewValidationError(ValueError):
    """Raised when an eval_view payload fails strict parsing."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        message = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        super().__init__(message)


def validate_eval_view(payload: dict[str, Any] | CounterfactualEvalView) -> ValidationResult:
    """Validate a raw or parsed eval_view payload."""

    if isinstance(payload, CounterfactualEvalView):
        payload = {
            "recent_support": payload.recent_support.value,
            "long_support": payload.long_support.value,
            "aversion": payload.aversion.value,
            "candidate_match": payload.candidate_match.value,
            "final_choice": payload.final_choice,
        }

    result = ValidationResult()
    if not isinstance(payload, dict):
        result.add("invalid_eval_view", "eval_view must be a mapping", "eval_view")
        return result

    expected_keys = {
        "recent_support",
        "long_support",
        "aversion",
        "candidate_match",
        "final_choice",
    }
    missing = expected_keys.difference(payload)
    extra = set(payload).difference(expected_keys)

    for key in sorted(missing):
        result.add("missing_field", f"{key} is required", f"eval_view.{key}")
    for key in sorted(extra):
        result.add("unexpected_field", f"{key} is not part of eval_view", f"eval_view.{key}")

    for key in ["recent_support", "long_support", "aversion", "candidate_match"]:
        value = payload.get(key)
        try:
            EvalDirection(value)
        except Exception:
            result.add(
                "invalid_direction",
                "direction must be one of up|same|down",
                f"eval_view.{key}",
            )

    final_choice = payload.get("final_choice")
    if not isinstance(final_choice, str) or not final_choice.strip():
        result.add(
            "invalid_final_choice",
            "final_choice must be a non-empty string",
            "eval_view.final_choice",
        )

    return result


def parse_eval_view(
    payload: dict[str, Any], *, strict: bool = True
) -> CounterfactualEvalView:
    """Parse a raw eval_view payload into a typed dataclass."""

    result = validate_eval_view(payload)
    if strict and not result.valid:
        raise EvalViewValidationError(result)

    return CounterfactualEvalView(
        recent_support=EvalDirection(payload["recent_support"]),
        long_support=EvalDirection(payload["long_support"]),
        aversion=EvalDirection(payload["aversion"]),
        candidate_match=EvalDirection(payload["candidate_match"]),
        final_choice=payload["final_choice"],
    )


def graph_factor_snapshot(
    graph: TraceRecGraph | dict[str, Any],
    *,
    focus_candidate_id: str | None = None,
) -> GraphFactorSnapshot:
    """Project a TRACE-Rec graph into compact counts for one focus candidate."""

    parsed_graph = (
        parse_trace_rec_graph(graph, strict=True) if isinstance(graph, dict) else graph
    )
    node_lookup = parsed_graph.node_by_id()
    candidate_id = focus_candidate_id or _selected_item_id(parsed_graph)
    candidate_nodes = [
        node
        for node in parsed_graph.nodes
        if isinstance(node, CandidateEvidenceNode) and node.candidate_id == candidate_id
    ]
    candidate_node_ids = {node.id for node in candidate_nodes}

    recent_support = 0
    long_support = 0
    aversion = 0

    for edge in parsed_graph.edges:
        if edge.type not in {GraphEdgeType.SUPPORTS, GraphEdgeType.CONFLICTS}:
            continue
        if edge.target not in candidate_node_ids:
            continue

        source = node_lookup.get(edge.source)
        if not isinstance(source, PreferenceStateNode):
            continue

        if (
            source.polarity == PreferencePolarity.POSITIVE
            and edge.type == GraphEdgeType.SUPPORTS
            and source.horizon == PreferenceHorizon.RECENT
        ):
            recent_support += 1
        if (
            source.polarity == PreferencePolarity.POSITIVE
            and edge.type == GraphEdgeType.SUPPORTS
            and source.horizon == PreferenceHorizon.PERSISTENT
        ):
            long_support += 1
        if source.polarity == PreferencePolarity.NEGATIVE and edge.type == GraphEdgeType.CONFLICTS:
            aversion += 1

    candidate_feature_refs = tuple(
        sorted({ref for node in candidate_nodes for ref in node.feature_refs})
    )
    return GraphFactorSnapshot(
        recent_support=recent_support,
        long_support=long_support,
        aversion=aversion,
        candidate_feature_refs=candidate_feature_refs,
        final_choice=_selected_item_id(parsed_graph),
    )


def derive_eval_view_from_graphs(
    original_graph: TraceRecGraph | dict[str, Any],
    updated_graph: TraceRecGraph | dict[str, Any],
    *,
    focus_candidate_id: str | None = None,
) -> CounterfactualEvalView:
    """Derive an eval_view directly from an original/updated graph pair."""

    parsed_original = (
        parse_trace_rec_graph(original_graph, strict=True)
        if isinstance(original_graph, dict)
        else original_graph
    )
    target_candidate_id = focus_candidate_id or _selected_item_id(parsed_original)
    original_snapshot = graph_factor_snapshot(
        parsed_original,
        focus_candidate_id=target_candidate_id,
    )
    updated_snapshot = graph_factor_snapshot(
        updated_graph,
        focus_candidate_id=target_candidate_id,
    )
    return CounterfactualEvalView(
        recent_support=_compare_factor(
            original_snapshot.recent_support,
            updated_snapshot.recent_support,
        ),
        long_support=_compare_factor(
            original_snapshot.long_support,
            updated_snapshot.long_support,
        ),
        aversion=_compare_factor(original_snapshot.aversion, updated_snapshot.aversion),
        candidate_match=_compare_feature_refs(
            original_snapshot.candidate_feature_refs,
            updated_snapshot.candidate_feature_refs,
        ),
        final_choice=updated_snapshot.final_choice,
    )


def targeted_factor(intervention: InterventionType) -> str:
    """Return the eval_view factor targeted by an intervention."""

    mapping = {
        InterventionType.RECENT_SUPPORT_REMOVAL: "recent_support",
        InterventionType.LONG_SUPPORT_REMOVAL: "long_support",
        InterventionType.CANDIDATE_FEATURE_SWAP: "candidate_match",
        InterventionType.HARD_AVERSION_INJECTION: "aversion",
    }
    return mapping[intervention]


def expected_direction(intervention: InterventionType) -> EvalDirection:
    """Return the expected direction for the targeted factor."""

    mapping = {
        InterventionType.RECENT_SUPPORT_REMOVAL: EvalDirection.DOWN,
        InterventionType.LONG_SUPPORT_REMOVAL: EvalDirection.DOWN,
        InterventionType.CANDIDATE_FEATURE_SWAP: EvalDirection.DOWN,
        InterventionType.HARD_AVERSION_INJECTION: EvalDirection.UP,
    }
    return mapping[intervention]


def targeted_update_accuracy(
    eval_view: CounterfactualEvalView, intervention: InterventionType
) -> bool:
    """Whether the targeted factor moves in the expected direction."""

    factor = targeted_factor(intervention)
    return getattr(eval_view, factor) == expected_direction(intervention)


def non_target_stability(
    eval_view: CounterfactualEvalView, intervention: InterventionType
) -> bool:
    """Whether all untargeted factors remain unchanged."""

    factor = targeted_factor(intervention)
    for field_name in ["recent_support", "long_support", "aversion", "candidate_match"]:
        if field_name == factor:
            continue
        if getattr(eval_view, field_name) != EvalDirection.SAME:
            return False
    return True


def decision_changed(eval_view: CounterfactualEvalView, original_choice: str) -> bool:
    """Whether the final choice changed relative to the original recommendation."""

    return eval_view.final_choice != original_choice


def decision_direction_consistency(
    eval_view: CounterfactualEvalView,
    intervention: InterventionType,
    *,
    original_choice: str,
    focus_candidate_id: str | None = None,
) -> bool:
    """Whether the final decision moves in the expected direction for the affected candidate."""

    affected_candidate = focus_candidate_id or original_choice
    if eval_view.final_choice != affected_candidate:
        return True
    return getattr(eval_view, targeted_factor(intervention)) == expected_direction(intervention)


@dataclass(frozen=True)
class CounterfactualCaseScore:
    """Minimal per-example score tuple for early TRACE-Rec pilots."""

    targeted_update_correct: bool
    decision_direction_consistent: bool
    non_target_stable: bool
    final_choice_changed: bool | None


def score_counterfactual_case(
    eval_view: CounterfactualEvalView,
    intervention: InterventionType,
    *,
    original_choice: str | None = None,
    focus_candidate_id: str | None = None,
) -> CounterfactualCaseScore:
    """Score one counterfactual case using the compact eval view."""

    final_choice_changed = None
    if original_choice is not None:
        final_choice_changed = decision_changed(eval_view, original_choice)

    return CounterfactualCaseScore(
        targeted_update_correct=targeted_update_accuracy(eval_view, intervention),
        decision_direction_consistent=(
            False
            if original_choice is None
            else decision_direction_consistency(
                eval_view,
                intervention,
                original_choice=original_choice,
                focus_candidate_id=focus_candidate_id,
            )
        ),
        non_target_stable=non_target_stability(eval_view, intervention),
        final_choice_changed=final_choice_changed,
    )


def score_counterfactual_graph_case(
    original_graph: TraceRecGraph | dict[str, Any],
    updated_graph: TraceRecGraph | dict[str, Any],
    intervention: InterventionType,
    *,
    focus_candidate_id: str | None = None,
) -> CounterfactualCaseScore:
    """Score one counterfactual case from graph pairs instead of manual eval labels."""

    parsed_original = (
        parse_trace_rec_graph(original_graph, strict=True)
        if isinstance(original_graph, dict)
        else original_graph
    )
    eval_view = derive_eval_view_from_graphs(
        parsed_original,
        updated_graph,
        focus_candidate_id=focus_candidate_id or _selected_item_id(parsed_original),
    )
    return score_counterfactual_case(
        eval_view,
        intervention,
        original_choice=_selected_item_id(parsed_original),
        focus_candidate_id=focus_candidate_id or _selected_item_id(parsed_original),
    )


def _compare_factor(original_value: int, updated_value: int) -> EvalDirection:
    """Map a numeric factor delta to the eval direction surface."""

    if updated_value > original_value:
        return EvalDirection.UP
    if updated_value < original_value:
        return EvalDirection.DOWN
    return EvalDirection.SAME


def _compare_feature_refs(
    original_refs: tuple[str, ...],
    updated_refs: tuple[str, ...],
) -> EvalDirection:
    """Map candidate feature-ref changes to the eval direction surface."""

    original_set = set(original_refs)
    updated_set = set(updated_refs)
    if updated_set == original_set:
        return EvalDirection.SAME
    if updated_set.issuperset(original_set):
        return EvalDirection.UP
    return EvalDirection.DOWN


def _selected_item_id(graph: TraceRecGraph) -> str:
    """Return the selected item id from the graph decision node."""

    for node in graph.nodes:
        if hasattr(node, "selected_item_id"):
            return str(node.selected_item_id)
    raise ValueError("graph must contain a decision node")
