# FaithRec Prompt Contract

This document defines the LLM-facing prompt and output schema for
evidence-grounded candidate reranking.

## 1. Shared Input

All baselines should receive the same input boundary unless the experiment is an
explicit ablation.

```text
Task:
Rank candidate items for the user's next interaction.
You may only rank items from the candidate set.
Use only the provided history, candidate metadata, and optional retriever prior.

User History:
H01. item_id="<asin>", title="<title>", rating=<rating>, categories="<path>", brand="<brand>"
H02. ...

Candidate Set:
A. item_id="<asin>", title="<title>", categories="<path>", brand="<brand>", retriever_rank=<rank>
B. ...

Optional User Profile:
<profile or None>
```

The target label must not be included.

## 2. Direct Baseline

```text
Instruction:
Return only the final answer JSON.

Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}
```

## 3. Free-form CoT Baseline

```text
Instruction:
Think step by step about the user's likely preference, then rank every
candidate.

Reasoning:
<free-form reasoning>

Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}
```

## 4. Evidence Rerank V1

```text
Instruction:
First select evidence from the provided user history and candidate metadata.
Every evidence reference must use an existing evidence ID such as H01, H02, A,
B, or a listed metadata field. Do not invent evidence.

Evidence Selection:
E1: [Hxx] <short statement about what this history item supports or conflicts with>
E2: [Hyy] <short statement>

Candidate Reasoning:
A: support=[E1], conflict=[E2], brief_reason="<reason>"
B: support=[], conflict=[E1], brief_reason="<reason>"

Ranking Decision:
<one or two short sentences>

Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A", "evidence_refs": ["Hxx", "Hyy"]}
```

## 5. Output Validity

The parser should enforce:

```text
ranking contains every candidate exactly once
selected_candidate_id equals ranking[0]
all ranking IDs come from the candidate set
evidence_refs contains only provided evidence IDs
the final answer is valid JSON
```

Invalid outputs receive low format reward and are counted separately from wrong
recommendations.

## 6. Evidence Grounding Rules

Allowed evidence IDs:

```text
Hxx: user history item
A/B/C/...: candidate item
Pxx: optional derived profile statement
Rxx: optional retriever prior statement
```

The model may cite candidate IDs when explaining why a candidate matches the
user, but the main faithfulness tests focus on history evidence references.

## 7. Counterfactual Prompt Variants

For `targeted_drop`, remove the history evidence cited by the model:

```text
The following history items were removed for an audit: H03, H07.
Rank the same candidates again using the remaining evidence.
Return only Final Answer JSON.
```

For `irrelevant_drop`, remove uncited history evidence:

```text
Some uncited history items were removed for an audit.
Rank the same candidates again using the remaining evidence.
Return only Final Answer JSON.
```

The audit prompts should avoid revealing which behavior is expected.
