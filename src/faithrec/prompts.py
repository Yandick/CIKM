from __future__ import annotations

from faithrec.schema import CandidateItem, HistoryItem, RerankInstance


def _category_text(categories: list[str]) -> str:
    return " > ".join(categories) if categories else "None"


def _history_line(item: HistoryItem) -> str:
    store = item.store or "None"
    return (
        f'{item.evidence_id}. item_id="{item.item_id}", title="{item.title}", '
        f"rating={item.rating}, categories=\"{_category_text(item.categories)}\", "
        f'brand="{store}"'
    )


def _candidate_line(item: CandidateItem) -> str:
    store = item.store or "None"
    return (
        f'{item.candidate_id}. item_id="{item.item_id}", title="{item.title}", '
        f'categories="{_category_text(item.categories)}", brand="{store}", '
        f"retriever_rank={item.retriever_rank}, retriever_score={item.retriever_score}"
    )


def render_evidence_rerank_prompt(instance: RerankInstance) -> str:
    history = "\n".join(_history_line(item) for item in instance.history)
    candidates = "\n".join(_candidate_line(item) for item in instance.candidates)
    return f"""Task:
Rank candidate items for the user's next interaction.
You may only rank items from the candidate set.
Use only the provided history, candidate metadata, and optional retriever prior.

User History:
{history}

Candidate Set:
{candidates}

Instruction:
First select evidence from the provided user history and candidate metadata.
Every evidence reference must use an existing evidence ID such as H01, H02, A,
B, or a listed profile/retriever evidence ID. Do not invent evidence.

Evidence Selection:
E1: [Hxx] short statement about what this evidence supports or conflicts with.

Candidate Reasoning:
A: support=[E1], conflict=[], brief_reason="..."

Ranking Decision:
Briefly explain the final ranking.

Final Answer:
{{"ranking": ["A", "B", "..."], "selected_candidate_id": "A", "evidence_refs": ["Hxx"]}}
"""


def render_direct_prompt(instance: RerankInstance) -> str:
    history = "\n".join(_history_line(item) for item in instance.history)
    candidates = "\n".join(_candidate_line(item) for item in instance.candidates)
    return f"""Task:
Rank candidate items for the user's next interaction.
You may only rank items from the candidate set.
Use only the provided history and candidate metadata.

User History:
{history}

Candidate Set:
{candidates}

Instruction:
Rank every candidate from most likely to least likely.
Do not explain your reasoning.
Return only:
Final Answer:
{{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}}
"""


def render_free_form_cot_prompt(instance: RerankInstance) -> str:
    history = "\n".join(_history_line(item) for item in instance.history)
    candidates = "\n".join(_candidate_line(item) for item in instance.candidates)
    return f"""Task:
Rank candidate items for the user's next interaction.
You may only rank items from the candidate set.
Use only the provided history and candidate metadata.

User History:
{history}

Candidate Set:
{candidates}

Instruction:
Think step by step about the user's likely preference, then rank every
candidate.

Reasoning:
<write concise reasoning>

Final Answer:
{{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}}
"""
