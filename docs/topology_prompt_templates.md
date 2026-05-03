# TopoRec-Family Prompt Templates

This document defines prompt templates for direct, linear, tree, and graph reasoning. The templates follow a lightweight design:

```text
natural-language reasoning trace + minimal final JSON answer
```

Only the final answer block is strictly parsed. The intermediate trace uses natural language section names to simulate chain, tree, or graph topology.

## 1. Shared Design Rules

All templates must follow the same information boundary:

```text
User history H_u
Candidate pool C_u
Candidate text X_C
Optional retriever rank or score S_C
Optional offline profile or item description A_u, only when enabled for all methods in the same comparison
```

No template may use information that another topology cannot access. In particular, graph reasoning must not use item-attribute graphs or verifier graphs as online prompt inputs unless the same information is also available to linear and tree modes in the same ablation.

The model may write natural-language reasoning, but the final answer must be valid JSON.

## 2. Shared Input Format

Use a compact and deterministic input format:

```text
Task:
You are ranking candidate items for sequential next-item recommendation.
Given a user's interaction history and a candidate pool, rank all candidates by how likely the user is to interact with them next.
Only use the information provided below.
Do not recommend any item outside the candidate pool.

User history, chronological:
H1. title="<title>", rating=<rating_or_null>, time="<relative_time_or_null>"
H2. title="<title>", rating=<rating_or_null>, time="<relative_time_or_null>"
...

Candidate pool:
A. item_id="<id>", title="<title>", retriever_rank=<rank_or_null>, retriever_score=<score_or_null>, text="<optional_text>"
B. item_id="<id>", title="<title>", retriever_rank=<rank_or_null>, retriever_score=<score_or_null>, text="<optional_text>"
...

Optional offline artifacts:
User profile: <profile_or_none>
Item descriptions:
A. <description_or_none>
B. <description_or_none>
...
```

When a field is unavailable, explicitly write `null` or `None`. Do not silently omit fields, because missing fields make prompt variants hard to compare.

## 3. Shared Final Answer Format

Every topology must end with:

```text
Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}
```

Constraints:

```text
ranking must include every candidate exactly once
selected_candidate_id must be the first candidate in ranking
candidate ids must be selected only from the candidate pool
do not put item titles in ranking
```

Optional scores can be added only if needed:

```text
Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A", "scores": {"A": 0.7, "B": 0.3}}
```

For the first implementation, the no-score version is recommended because it is shorter and easier to parse.

## 4. Direct Answer Baseline

Purpose: measure answer-only reranking without explicit reasoning.

```text
System:
You are a recommendation ranking assistant.

User:
<SHARED_INPUT>

Instruction:
Rank every candidate from most likely to least likely for the user's next interaction.
Do not explain your reasoning.
Return only the final answer block:

Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}
```

Direct mode should be used as a strong baseline. It tells us whether topology reasoning adds value beyond concise LLM reranking.

## 5. Linear Reasoning Template

Purpose: cheap ordered reasoning. It summarizes preference, matches candidates, then ranks.

```text
System:
You are a recommendation ranking assistant.

User:
<SHARED_INPUT>

Instruction:
Use the Linear protocol. Write a short reasoning trace with exactly three ordered steps:

Reasoning Trace:
Step 1 - Preference Summary:
Summarize the user's likely recent and long-term preference from the history.

Step 2 - Candidate Matching:
Compare candidates against that preference. Use candidate IDs.

Step 3 - Ranking Decision:
State the final ranking logic briefly.

Then return:
Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}
```

Constraints:

```text
use one ordered chain only
do not create branches
do not create graph factors or edges
rank every candidate exactly once
keep each step concise
```

## 6. Tree Reasoning Template

Purpose: decompose preference into parallel branches, then aggregate.

```text
System:
You are a recommendation ranking assistant.

User:
<SHARED_INPUT>

Instruction:
Use the Tree protocol. Write a short reasoning trace with branches and aggregation:

Reasoning Trace:
Branch 1 - Recent Interest:
Analyze what the latest interactions suggest.

Branch 2 - Long-Term Interest:
Analyze stable preference across the history.

Branch 3 - Candidate Text Match:
Compare candidate descriptions or titles with the inferred interests.

Aggregation:
Combine the branch conclusions and decide the final ranking.

Then return:
Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}
```

Constraints:

```text
use two or three branches
branches should be parallel evidence views, not a hidden chain
aggregation must explain how branch evidence is combined
rank every candidate exactly once
do not create support/conflict graph relations
```

Suggested branch policy:

```text
If history length >= 8:
  use recent interest and long-term interest.
If candidate text is informative:
  use candidate text match.
If retriever scores are available:
  mention retriever prior inside aggregation or as an additional branch in ablation.
```

For fair fixed-topology experiments, use the same branch policy for all samples rather than letting the model invent arbitrary branches.

## 7. Graph Reasoning Template

Purpose: simulate compact factor-candidate graph reasoning using natural language.

```text
System:
You are a recommendation ranking assistant.

User:
<SHARED_INPUT>

Instruction:
Use the Graph protocol. Write a compact reasoning trace with factors, support/conflict relations, and aggregation:

Reasoning Trace:
Preference Factors:
F1: State one evidence factor from the provided history or candidate information.
F2: State another evidence factor if useful.
F3: State a retriever-prior or negative-evidence factor if useful.

Support / Conflict Relations:
Explain which factors support or conflict with which candidate IDs.
Use only support and conflict relations.

Aggregation:
Combine the support and conflict evidence into a final ranking.

Then return:
Final Answer:
{"ranking": ["A", "B", "..."], "selected_candidate_id": "A"}
```

Constraints:

```text
use 3 to 6 factors for K = 20
use only support and conflict relation words
do not introduce candidate nodes as JSON
do not create an item-attribute graph
rank every candidate exactly once
do not use external knowledge not present in the input
```

Default factor policy:

```text
F_recent: evidence from the most recent history items
F_long: evidence from the full or older history
F_prior: retriever rank or score pattern, if available
F_text: salient candidate text pattern, if useful
F_negative: evidence against plausible but mismatched candidates, if useful
```

## 8. Prompt Variants for Controlled Ablation

Use these prompt variants to isolate effects:

```text
without_retriever_score:
  hide retriever_score but keep candidate set.

without_optional_profile:
  set optional_artifacts.user_profile = null.

same_token_budget:
  cap each trace to the same maximum number of sentences.

answer_only:
  direct answer baseline only.

minimal_trace:
  one sentence per step, branch, or factor.
```

The `same_token_budget` ablation is important because graph reasoning may otherwise win or lose due to verbosity rather than topology.

## 9. Parsing and Repair Prompt

The parser should extract the JSON object after `Final Answer:`.

If the final answer block is invalid JSON, allow at most one repair call in development experiments:

```text
System:
You repair final answer blocks. Return valid JSON only.

User:
The following final answer failed parsing or validation:
<RAW_FINAL_ANSWER>

Validation errors:
<ERRORS>

Repair it without changing the candidate ids or adding new candidates.
Return only the corrected JSON object.
```

For final evaluation, report:

```text
performance before repair
performance after one repair, if used
parse success rate
candidate hallucination rate
```

## 10. Recommended First Experiment

Run fixed-topology inference on a small validation slice:

```text
datasets: MovieLens-1M and Amazon Book
instances: 100 to 300 per dataset
candidates: 20
topologies: direct, linear, tree, graph
output: natural-language trace plus final answer JSON
storage: JSONL records following docs/reasoning_schema.md
```

Primary goal:

```text
verify final-answer parse stability
measure token cost
identify whether different topology modes win on different samples
```
