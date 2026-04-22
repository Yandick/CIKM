"""Unit tests for TRACE-Rec counterfactual eval helpers."""

from __future__ import annotations

import pytest

from llm4rec.evaluation import (
    EvalDirection,
    EvalViewValidationError,
    derive_eval_view_from_graphs,
    InterventionType,
    non_target_stability,
    parse_eval_view,
    score_counterfactual_case,
    score_counterfactual_graph_case,
    targeted_update_accuracy,
    validate_eval_view,
)


def _valid_eval_view() -> dict:
    return {
        "recent_support": "down",
        "long_support": "same",
        "aversion": "same",
        "candidate_match": "same",
        "final_choice": "item-b",
    }


def test_parse_valid_eval_view() -> None:
    eval_view = parse_eval_view(_valid_eval_view())
    assert eval_view.recent_support == EvalDirection.DOWN
    assert eval_view.final_choice == "item-b"


def test_invalid_eval_view_rejects_bad_direction() -> None:
    payload = _valid_eval_view()
    payload["recent_support"] = "lower"

    result = validate_eval_view(payload)

    assert not result.valid
    assert any(issue.code == "invalid_direction" for issue in result.issues)


def test_recent_support_removal_scores_as_targeted_update() -> None:
    eval_view = parse_eval_view(_valid_eval_view())

    assert targeted_update_accuracy(
        eval_view, InterventionType.RECENT_SUPPORT_REMOVAL
    ) is True
    assert non_target_stability(eval_view, InterventionType.RECENT_SUPPORT_REMOVAL) is True


def test_candidate_feature_swap_requires_candidate_match_to_drop() -> None:
    eval_view = parse_eval_view(
        {
            "recent_support": "same",
            "long_support": "same",
            "aversion": "same",
            "candidate_match": "down",
            "final_choice": "item-a",
        }
    )

    score = score_counterfactual_case(
        eval_view,
        InterventionType.CANDIDATE_FEATURE_SWAP,
        original_choice="item-b",
    )

    assert score.targeted_update_correct is True
    assert score.decision_direction_consistent is True
    assert score.non_target_stable is True
    assert score.final_choice_changed is True


def test_parse_eval_view_raises_on_missing_fields_in_strict_mode() -> None:
    payload = _valid_eval_view()
    payload.pop("final_choice")

    with pytest.raises(EvalViewValidationError):
        parse_eval_view(payload, strict=True)


def test_derive_eval_view_from_graphs_detects_recent_support_drop() -> None:
    original_graph = _graph_payload(
        include_recent_support=True,
        include_long_support=True,
        include_aversion=False,
        candidate_feature_refs=["genre:sci-fi"],
        selected_item_id="item-a",
    )
    updated_graph = _graph_payload(
        include_recent_support=False,
        include_long_support=True,
        include_aversion=False,
        candidate_feature_refs=["genre:sci-fi"],
        selected_item_id="item-b",
    )

    eval_view = derive_eval_view_from_graphs(original_graph, updated_graph)

    assert eval_view.recent_support == EvalDirection.DOWN
    assert eval_view.long_support == EvalDirection.SAME
    assert eval_view.final_choice == "item-b"


def test_score_counterfactual_graph_case_uses_graph_deltas() -> None:
    original_graph = _graph_payload(
        include_recent_support=True,
        include_long_support=True,
        include_aversion=False,
        candidate_feature_refs=["genre:sci-fi"],
        selected_item_id="item-a",
    )
    updated_graph = _graph_payload(
        include_recent_support=True,
        include_long_support=True,
        include_aversion=False,
        candidate_feature_refs=["taste:savory"],
        selected_item_id="item-b",
    )

    score = score_counterfactual_graph_case(
        original_graph,
        updated_graph,
        InterventionType.CANDIDATE_FEATURE_SWAP,
    )

    assert score.targeted_update_correct is True
    assert score.decision_direction_consistent is True
    assert score.non_target_stable is True
    assert score.final_choice_changed is True


def _graph_payload(
    *,
    include_recent_support: bool,
    include_long_support: bool,
    include_aversion: bool,
    candidate_feature_refs: list[str],
    selected_item_id: str,
) -> dict:
    nodes = [
        {
            "id": "cand-a",
            "type": "candidate_evidence",
            "candidate_id": "item-a",
            "feature_refs": candidate_feature_refs,
            "summary": "candidate a evidence",
        },
        {
            "id": "cand-b",
            "type": "candidate_evidence",
            "candidate_id": "item-b",
            "feature_refs": ["genre:romance"],
            "summary": "candidate b evidence",
        },
        {
            "id": "decision",
            "type": "decision",
            "selected_item_id": selected_item_id,
        },
    ]
    edges = [
        {
            "source": "cand-a" if selected_item_id == "item-a" else "cand-b",
            "target": "decision",
            "type": "selected",
        }
    ]
    context = {
        "available_evidence_refs": ["history:1", "history:2", "constraint:1"],
        "available_feature_refs_by_candidate": {
            "item-a": ["genre:sci-fi", "taste:savory"],
            "item-b": ["genre:romance"],
        },
    }

    if include_recent_support:
        nodes.append(
            {
                "id": "pref-recent",
                "type": "preference_state",
                "summary": "recent sci-fi preference",
                "polarity": "positive",
                "horizon": "recent",
                "evidence_refs": ["history:1"],
                "source": "history",
            }
        )
        edges.append({"source": "pref-recent", "target": "cand-a", "type": "supports"})

    if include_long_support:
        nodes.append(
            {
                "id": "pref-long",
                "type": "preference_state",
                "summary": "persistent genre preference",
                "polarity": "positive",
                "horizon": "persistent",
                "evidence_refs": ["history:2"],
                "source": "history",
            }
        )
        edges.append({"source": "pref-long", "target": "cand-a", "type": "supports"})

    if include_aversion:
        nodes.append(
            {
                "id": "pref-aversion",
                "type": "preference_state",
                "summary": "avoid sugary products",
                "polarity": "negative",
                "horizon": "persistent",
                "evidence_refs": ["constraint:1"],
                "source": "constraint",
            }
        )
        edges.append({"source": "pref-aversion", "target": "cand-a", "type": "conflicts"})

    return {
        "metadata": {
            "task": "next_item_recommendation",
            "user_id": "u-1",
            "candidate_ids": ["item-a", "item-b"],
            "context": context,
        },
        "nodes": nodes,
        "edges": edges,
    }
