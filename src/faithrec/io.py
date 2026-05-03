from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from faithrec.schema import CandidateItem, HistoryItem, RerankInstance


def _categories(metadata: dict[str, Any]) -> list[str]:
    categories = metadata.get("categories") or []
    if isinstance(categories, list):
        return [str(category) for category in categories]
    return [str(categories)]


def _text(metadata: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for key in ("description", "features"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value[:5] if item)
    return " ".join(parts) if parts else None


def instance_from_record(record: dict[str, Any]) -> RerankInstance:
    history: list[HistoryItem] = []
    for item in record["history"]:
        metadata = item.get("metadata") or {}
        history.append(
            HistoryItem(
                evidence_id=item["evidence_id"],
                item_id=item["item_id"],
                title=metadata.get("title") or "Unknown item",
                rating=metadata.get("rating"),
                categories=_categories(metadata),
                store=metadata.get("store"),
                text=_text(metadata),
            )
        )

    candidates: list[CandidateItem] = []
    for item in record["candidates"]:
        metadata = item.get("metadata") or {}
        candidates.append(
            CandidateItem(
                candidate_id=item["candidate_id"],
                item_id=item["item_id"],
                title=metadata.get("title") or "Unknown item",
                categories=_categories(metadata),
                store=metadata.get("store"),
                text=_text(metadata),
                retriever_rank=item.get("retriever_rank"),
                retriever_score=item.get("retriever_score"),
            )
        )

    label = record.get("label") or {}
    return RerankInstance(
        instance_id=record["instance_id"],
        user_id=record["user_id"],
        history=history,
        candidates=candidates,
        target_candidate_id=label.get("target_candidate_id"),
        dataset=record.get("dataset"),
        metadata={"target_item_id": label.get("target_item_id")},
    )


def read_instances(path: Path) -> Iterable[RerankInstance]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield instance_from_record(json.loads(line))
