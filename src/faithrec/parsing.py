from __future__ import annotations

import json
from typing import Any

from faithrec.schema import RerankOutput


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found")


def parse_final_answer(text: str) -> RerankOutput:
    """Parse the minimal final-answer JSON from an LLM response."""

    answer_start = text.lower().rfind("final answer")
    search_text = text[answer_start:] if answer_start >= 0 else text
    obj = _first_json_object(search_text)

    ranking = obj.get("ranking")
    selected = obj.get("selected_candidate_id")
    evidence_refs = obj.get("evidence_refs", [])
    scores = obj.get("scores")

    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
        raise ValueError("Final answer field 'ranking' must be a list of strings")
    if not isinstance(selected, str):
        raise ValueError("Final answer field 'selected_candidate_id' must be a string")
    if not isinstance(evidence_refs, list) or not all(
        isinstance(x, str) for x in evidence_refs
    ):
        raise ValueError("Final answer field 'evidence_refs' must be a list of strings")
    if scores is not None and not isinstance(scores, dict):
        raise ValueError("Final answer field 'scores' must be an object when present")

    return RerankOutput(
        ranking=ranking,
        selected_candidate_id=selected,
        evidence_refs=evidence_refs,
        raw_text=text,
        scores=scores,
    )
