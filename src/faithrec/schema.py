from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HistoryItem:
    evidence_id: str
    item_id: str
    title: str
    rating: float | None = None
    timestamp: int | None = None
    categories: list[str] = field(default_factory=list)
    store: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class CandidateItem:
    candidate_id: str
    item_id: str
    title: str
    categories: list[str] = field(default_factory=list)
    store: str | None = None
    text: str | None = None
    retriever_rank: int | None = None
    retriever_score: float | None = None


@dataclass(frozen=True)
class RerankInstance:
    instance_id: str
    user_id: str
    history: list[HistoryItem]
    candidates: list[CandidateItem]
    target_candidate_id: str | None = None
    dataset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_ids(self) -> list[str]:
        return [candidate.candidate_id for candidate in self.candidates]

    @property
    def evidence_ids(self) -> set[str]:
        history_ids = {item.evidence_id for item in self.history}
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        extra_ids = set(self.metadata.get("extra_evidence_ids", []))
        return history_ids | candidate_ids | extra_ids


@dataclass(frozen=True)
class RerankOutput:
    ranking: list[str]
    selected_candidate_id: str
    evidence_refs: list[str] = field(default_factory=list)
    raw_text: str | None = None
    scores: dict[str, float] | None = None


def validate_output(output: RerankOutput, instance: RerankInstance) -> list[str]:
    """Return validation errors for the final answer contract."""

    errors: list[str] = []
    candidate_ids = instance.candidate_ids
    candidate_set = set(candidate_ids)

    if output.selected_candidate_id not in candidate_set:
        errors.append("selected_candidate_not_in_pool")

    if not output.ranking:
        errors.append("empty_ranking")
    elif output.ranking[0] != output.selected_candidate_id:
        errors.append("selected_not_first")

    unknown_candidates = [cid for cid in output.ranking if cid not in candidate_set]
    if unknown_candidates:
        errors.append("ranking_contains_unknown_candidate")

    if len(output.ranking) != len(candidate_ids):
        errors.append("ranking_length_mismatch")

    if len(set(output.ranking)) != len(output.ranking):
        errors.append("ranking_contains_duplicates")

    if set(output.ranking) != candidate_set:
        errors.append("ranking_candidate_set_mismatch")

    unknown_refs = [ref for ref in output.evidence_refs if ref not in instance.evidence_ids]
    if unknown_refs:
        errors.append("invented_evidence_ref")

    return errors
