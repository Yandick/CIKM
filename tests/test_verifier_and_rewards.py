"""Unit tests for answer-only verifier logic and reward/advantage utilities."""

from __future__ import annotations

from llm4rec.data.schema import CandidateItem, InteractionEvent, NextItemExample
from llm4rec.evaluation import (
    AnswerOnlyPrediction,
    CounterfactualAuditCase,
    InterventionType,
    VerifierRewardWeights,
    verify_answer_only_prediction,
)
from llm4rec.training import AdvantageConfig, group_relative_advantages, mean_reward


def test_verify_answer_only_prediction_without_audit_graph() -> None:
    example = _toy_example()
    prediction = AnswerOnlyPrediction(selected_item_id="i3", response_text="Answer: i3")

    result = verify_answer_only_prediction(example, prediction)

    assert result.answer_in_candidate_set is True
    assert result.exact_hit is True
    assert result.components.schema_passed is True
    assert result.components.utility == 1.0
    assert result.reward == 1.0


def test_verify_answer_only_prediction_rejects_invalid_candidate_id() -> None:
    example = _toy_example()
    prediction = AnswerOnlyPrediction(selected_item_id="unknown")

    result = verify_answer_only_prediction(example, prediction)

    assert result.answer_in_candidate_set is False
    assert result.components.schema_passed is False
    assert result.reward == 0.0


def test_verify_answer_only_prediction_with_audit_graph_and_counterfactual_case() -> None:
    example = _toy_example()
    original_graph = _valid_graph_payload(selected_item_id="i3", support_feature_ref="category:coffee")
    updated_graph = _updated_graph_payload(selected_item_id="i2", support_feature_ref="category:tea")
    prediction = AnswerOnlyPrediction(
        selected_item_id="i3",
        response_text="Answer: i3",
        audit_graph=original_graph,
    )
    cf_case = CounterfactualAuditCase(
        intervention=InterventionType.CANDIDATE_FEATURE_SWAP,
        updated_graph=updated_graph,
        focus_candidate_id="i3",
    )

    result = verify_answer_only_prediction(
        example,
        prediction,
        counterfactual_cases=(cf_case,),
        weights=VerifierRewardWeights(cost_penalty_weight=0.0),
    )

    assert result.audit_graph_valid is True
    assert result.audit_graph_grounded is True
    assert result.graph_evidence_profile is not None
    assert result.graph_evidence_profile.selected_support_count == 1
    assert result.counterfactual_summary is not None
    assert result.counterfactual_summary.case_count == 1
    assert result.counterfactual_summary.targeted_update_accuracy == 1.0
    assert result.counterfactual_summary.non_target_stability == 0.0
    assert result.components.counterfactual == 1.0
    assert result.components.locality == 0.0
    assert result.reward > 0.0


def test_group_relative_advantages_normalize_and_clip() -> None:
    rewards = [0.0, 1.0, 2.0]

    advantages = group_relative_advantages(
        rewards,
        config=AdvantageConfig(clip_value=1.0),
    )

    assert advantages[0] == -1.0
    assert advantages[2] == 1.0
    assert round(mean_reward(rewards), 4) == 1.0


def _toy_example() -> NextItemExample:
    example = NextItemExample(
        example_id="ex1",
        user_id="u1",
        history=(
            InteractionEvent(item_id="h1", timestamp=1),
            InteractionEvent(item_id="h2", timestamp=2),
        ),
        target_item_id="i3",
        candidates=(
            CandidateItem(item_id="i1", label=0, source="neg", rank_prior=3.0),
            CandidateItem(item_id="i2", label=0, source="neg", rank_prior=2.0),
            CandidateItem(item_id="i3", label=1, source="target", rank_prior=1.0),
        ),
        split="validation",
        context={
            "available_evidence_refs": ("history:0", "history:1"),
            "available_feature_refs_by_candidate": {
                "i1": ("category:snack",),
                "i2": ("category:tea",),
                "i3": ("category:coffee",),
            },
        },
    )
    example.validate()
    return example


def _valid_graph_payload(*, selected_item_id: str, support_feature_ref: str) -> dict:
    return {
        "metadata": {
            "task": "next_item_recommendation",
            "user_id": "u1",
            "candidate_ids": ["i1", "i2", "i3"],
            "context": {
                "available_evidence_refs": ["history:0", "history:1"],
                "available_feature_refs_by_candidate": {
                    "i1": ["category:snack"],
                    "i2": ["category:tea"],
                    "i3": ["category:coffee"],
                },
            },
        },
        "nodes": [
            {
                "id": "pref-1",
                "type": "preference_state",
                "summary": "recent coffee repeat-buy tendency",
                "polarity": "positive",
                "horizon": "recent",
                "evidence_refs": ["history:0"],
                "source": "history",
            },
            {
                "id": "cand-selected",
                "type": "candidate_evidence",
                "candidate_id": selected_item_id,
                "feature_refs": [support_feature_ref],
                "summary": "selected candidate matches category",
            },
            {
                "id": "cand-other",
                "type": "candidate_evidence",
                "candidate_id": "i2" if selected_item_id != "i2" else "i1",
                "feature_refs": ["category:tea" if selected_item_id != "i2" else "category:snack"],
                "summary": "other candidate is a weaker fit",
            },
            {
                "id": "decision-1",
                "type": "decision",
                "selected_item_id": selected_item_id,
            },
        ],
        "edges": [
            {"source": "pref-1", "target": "cand-selected", "type": "supports"},
            {"source": "pref-1", "target": "cand-other", "type": "conflicts"},
            {"source": "cand-selected", "target": "decision-1", "type": "selected"},
        ],
    }


def _updated_graph_payload(*, selected_item_id: str, support_feature_ref: str) -> dict:
    return {
        "metadata": {
            "task": "next_item_recommendation",
            "user_id": "u1",
            "candidate_ids": ["i1", "i2", "i3"],
            "context": {
                "available_evidence_refs": ["history:0", "history:1"],
                "available_feature_refs_by_candidate": {
                    "i1": ["category:snack"],
                    "i2": ["category:tea"],
                    "i3": [support_feature_ref],
                },
            },
        },
        "nodes": [
            {
                "id": "pref-1",
                "type": "preference_state",
                "summary": "recent coffee support weakened",
                "polarity": "positive",
                "horizon": "recent",
                "evidence_refs": ["history:0"],
                "source": "history",
            },
            {
                "id": "cand-selected",
                "type": "candidate_evidence",
                "candidate_id": "i3",
                "feature_refs": [support_feature_ref],
                "summary": "selected candidate lost the old feature match",
            },
            {
                "id": "cand-other",
                "type": "candidate_evidence",
                "candidate_id": selected_item_id,
                "feature_refs": ["category:tea"],
                "summary": "alternative candidate now fits better",
            },
            {
                "id": "decision-1",
                "type": "decision",
                "selected_item_id": selected_item_id,
            },
        ],
        "edges": [
            {"source": "pref-1", "target": "cand-other", "type": "supports"},
            {"source": "pref-1", "target": "cand-selected", "type": "conflicts"},
            {"source": "cand-other", "target": "decision-1", "type": "selected"},
        ],
    }
