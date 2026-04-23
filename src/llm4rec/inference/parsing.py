"""Shared parsing and consistency checks for model prediction text."""

from __future__ import annotations

import re
from typing import Sequence


ANSWER_LINE_RE = re.compile(r"answer\s*:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)


def extract_candidate_id_from_text(
    text: str,
    candidate_ids: Sequence[str],
) -> str | None:
    """Extract one candidate id from free-form model output."""

    if not text:
        return None

    match = ANSWER_LINE_RE.search(text)
    if match is not None:
        candidate_id = match.group(1).strip()
        if candidate_id in candidate_ids:
            return candidate_id

    normalized = text.strip()
    if normalized in candidate_ids:
        return normalized

    mentioned = [candidate_id for candidate_id in candidate_ids if candidate_id in text]
    if len(mentioned) == 1:
        return mentioned[0]
    return None


def validate_prediction_consistency(
    *,
    selected_item_id: str | None,
    response_text: str | None,
    candidate_ids: Sequence[str],
) -> None:
    """Raise when normalized prediction fields disagree."""

    if not selected_item_id or not response_text:
        return

    parsed_item_id = extract_candidate_id_from_text(response_text, candidate_ids)
    if parsed_item_id is None:
        raise ValueError(
            "response_text is present but could not be normalized against the candidate set"
        )
    if parsed_item_id != selected_item_id:
        raise ValueError(
            "selected_item_id and response_text disagree: "
            f"{selected_item_id!r} vs {parsed_item_id!r}"
        )


def normalize_ranked_item_ids(
    ranked_item_ids: Sequence[str],
    *,
    selected_item_id: str | None = None,
) -> tuple[str, ...]:
    """De-duplicate a ranking list and keep the selected item first when present."""

    normalized: list[str] = []
    if selected_item_id:
        normalized.append(selected_item_id)
    for item_id in ranked_item_ids:
        if not item_id:
            continue
        if item_id in normalized:
            continue
        normalized.append(str(item_id))
    return tuple(normalized)
