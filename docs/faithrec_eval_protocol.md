# FaithRec Evaluation Protocol

## 1. Evaluation Questions

RQ1:

```text
Do current LLM4Rec reasoning prompts produce faithful evidence-grounded traces?
```

RQ2:

```text
Does faithfulness-aware RL improve both recommendation quality and reasoning
faithfulness over rank-only RL?
```

RQ3:

```text
What is the accuracy/faithfulness/latency trade-off?
```

## 2. Baselines

Use the same model and candidate pool for all LLM baselines.

```text
retriever_prior_only
direct_llm_rerank
free_form_cot_rerank
evidence_prompt_rerank
evidence_sft
rank_only_grpo
faithfulness_grpo
```

Optional strong baselines:

```text
ReRec-style reward shaping, if query-based data is available
R2Rec-style interaction reasoning, if interaction-chain data is constructed
```

## 3. Ranking Metrics

```text
HR@1
HR@5
NDCG@3
NDCG@5
MRR
```

For 20-candidate reranking, HR@1 and NDCG@5 are the most important.

## 4. Format Metrics

```text
parse_success_rate
candidate_only_rate
complete_ranking_rate
selected_is_top_rate
evidence_ref_validity
```

Report these before ranking metrics for generated-output methods.

## 5. Faithfulness Metrics

### Evidence Ref Validity

```text
valid_refs / total_refs
```

Measures whether the model cites real evidence IDs.

### Evidence Precision

```text
refs with lexical/category/item-overlap support / total refs
```

This is a heuristic grounding score. Keep it separate from counterfactual
faithfulness.

### Targeted Evidence Sensitivity

Remove cited support evidence and rerun the model.

```text
TES = fraction of cases where selected candidate rank worsens
```

Alternative continuous version:

```text
TES_delta = mean(rank_after_drop - rank_before_drop)
```

Higher is better when the removed evidence was claimed as supportive.

### Irrelevant Evidence Stability

Remove uncited history evidence and rerun the model.

```text
IES = fraction of cases where selected candidate remains unchanged
```

Higher is better.

### Sufficiency

Keep only cited evidence and rerun the model.

```text
Sufficiency = fraction where selected candidate remains unchanged
```

This checks whether cited evidence is enough to support the decision.

### Comprehensiveness

Remove all cited evidence and rerun the model.

```text
Comprehensiveness = fraction where selected candidate changes or rank drops
```

This overlaps with targeted sensitivity but can remove all cited evidence at
once.

## 6. Efficiency Metrics

```text
input_tokens
completion_tokens
latency_ms
num_model_calls
counterfactual_audit_cost
```

Counterfactual evaluation is more expensive than normal reranking, so always
report the audit cost separately.

## 7. Failure Taxonomy

Log failures into:

```text
invalid_json
candidate_not_in_pool
missing_candidate_in_ranking
invented_evidence_ref
unsupported_evidence_claim
unstable_under_irrelevant_drop
insensitive_to_cited_evidence_drop
overlong_reasoning
```

This taxonomy is useful for paper analysis and debugging.

## 8. Pilot Evaluation

For the first milestone:

```text
dataset: Amazon Food
test size: 500
candidate size: 20
baselines: direct, free-form CoT, evidence prompt
audit: targeted_drop + irrelevant_drop
```

Success criteria:

```text
1. evidence prompt can be parsed in >90% of cases
2. free-form CoT has lower evidence validity than evidence prompt
3. at least one baseline shows plausible explanations with weak TES
4. evidence prompt has better validity but may still fail counterfactual tests
```

Only start RL after this phenomenon is measured.
