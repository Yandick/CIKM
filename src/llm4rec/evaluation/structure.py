"""Structure-metric aggregation helpers for TRACE-Rec validation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from llm4rec.reasoning.trace_rec_v1 import ValidationResult


@dataclass(frozen=True)
class GraphValidationSummary:
    """Compact aggregate for graph validation outcomes over a split or run."""

    parse_success_rate: float
    contradiction_rate: float
    unknown_pointer_rate: float


def summarize_graph_validation_results(
    results: Sequence[ValidationResult],
) -> GraphValidationSummary:
    """Aggregate graph validation outcomes into offline structure metrics."""

    if not results:
        raise ValueError("results must be non-empty")

    total = float(len(results))
    parse_success_rate = sum(1 for result in results if result.valid) / total
    contradiction_rate = (
        sum(
            1
            for result in results
            if any(issue.code == "contradictory_signed_edges" for issue in result.issues)
        )
        / total
    )
    unknown_pointer_rate = (
        sum(
            1
            for result in results
            if any(
                issue.code in {"unknown_evidence_ref", "unknown_feature_ref"}
                for issue in result.issues
            )
        )
        / total
    )

    return GraphValidationSummary(
        parse_success_rate=parse_success_rate,
        contradiction_rate=contradiction_rate,
        unknown_pointer_rate=unknown_pointer_rate,
    )
