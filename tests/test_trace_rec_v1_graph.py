"""Unit tests for TRACE-Rec V1 graph parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.evaluation import summarize_graph_validation_results
from llm4rec.reasoning import (
    TraceRecGraphValidationError,
    load_trace_rec_v1_schema_config,
    parse_trace_rec_graph,
    validate_trace_rec_graph,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "prompt" / "rec_graph_cot.yaml"
)


def _valid_graph_payload() -> dict:
    return {
        "metadata": {
            "task": "next_item_recommendation",
            "user_id": "u-1",
            "candidate_ids": ["item-a", "item-b"],
            "context": {
                "available_evidence_refs": ["history:7", "history:9"],
                "available_feature_refs_by_candidate": {
                    "item-a": ["genre:sci-fi"],
                    "item-b": ["genre:romance"],
                },
            },
        },
        "nodes": [
            {
                "id": "pref-recent",
                "type": "preference_state",
                "summary": "recently watched science fiction",
                "polarity": "positive",
                "horizon": "recent",
                "evidence_refs": ["history:7"],
                "source": "history",
            },
            {
                "id": "cand-a",
                "type": "candidate_evidence",
                "candidate_id": "item-a",
                "feature_refs": ["genre:sci-fi"],
                "summary": "candidate matches science fiction genre",
            },
            {
                "id": "cand-b",
                "type": "candidate_evidence",
                "candidate_id": "item-b",
                "feature_refs": ["genre:romance"],
                "summary": "candidate has romance genre",
            },
            {
                "id": "decision-1",
                "type": "decision",
                "selected_item_id": "item-a",
            },
        ],
        "edges": [
            {"source": "pref-recent", "target": "cand-a", "type": "supports"},
            {"source": "cand-a", "target": "decision-1", "type": "selected"},
        ],
    }


def test_load_prompt_constraints_from_yaml() -> None:
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)
    assert config.name == "trace_rec_v1_minimal"
    assert config.constraints.max_preference_state_nodes == 3
    assert config.constraints.strict_single_pointer_mode is True


def test_parse_valid_trace_rec_graph() -> None:
    payload = _valid_graph_payload()
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)

    graph = parse_trace_rec_graph(payload, constraints=config.constraints)

    assert graph.metadata.user_id == "u-1"
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 2


def test_invalid_if_selected_edge_and_decision_disagree() -> None:
    payload = _valid_graph_payload()
    payload["nodes"][-1]["selected_item_id"] = "item-b"
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)

    result = validate_trace_rec_graph(payload, constraints=config.constraints)

    assert not result.valid
    assert any(issue.code == "selected_edge_mismatch" for issue in result.issues)


def test_invalid_if_same_preference_supports_and_conflicts_same_candidate() -> None:
    payload = _valid_graph_payload()
    payload["edges"].insert(
        1, {"source": "pref-recent", "target": "cand-a", "type": "conflicts"}
    )
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)

    result = validate_trace_rec_graph(payload, constraints=config.constraints)

    assert not result.valid
    assert any(issue.code == "contradictory_signed_edges" for issue in result.issues)


def test_invalid_if_strict_pointer_mode_is_violated() -> None:
    payload = _valid_graph_payload()
    payload["nodes"][0]["evidence_refs"] = ["history:7", "history:8"]
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)

    with pytest.raises(TraceRecGraphValidationError):
        parse_trace_rec_graph(payload, constraints=config.constraints, strict=True)


def test_invalid_if_candidate_not_in_candidate_set() -> None:
    payload = _valid_graph_payload()
    payload["nodes"][1]["candidate_id"] = "item-z"
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)

    result = validate_trace_rec_graph(payload, constraints=config.constraints)

    assert not result.valid
    assert any(issue.code == "candidate_not_in_set" for issue in result.issues)


def test_invalid_if_evidence_ref_is_not_in_registry() -> None:
    payload = _valid_graph_payload()
    payload["nodes"][0]["evidence_refs"] = ["history:404"]
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)

    result = validate_trace_rec_graph(payload, constraints=config.constraints)

    assert not result.valid
    assert any(issue.code == "unknown_evidence_ref" for issue in result.issues)


def test_structure_summary_counts_unknown_pointer_rate() -> None:
    config = load_trace_rec_v1_schema_config(CONFIG_PATH)
    valid_result = validate_trace_rec_graph(_valid_graph_payload(), constraints=config.constraints)

    invalid_payload = _valid_graph_payload()
    invalid_payload["nodes"][0]["evidence_refs"] = ["history:404"]
    invalid_result = validate_trace_rec_graph(invalid_payload, constraints=config.constraints)

    summary = summarize_graph_validation_results([valid_result, invalid_result])

    assert summary.parse_success_rate == 0.5
    assert summary.unknown_pointer_rate == 0.5
    assert summary.contradiction_rate == 0.0
