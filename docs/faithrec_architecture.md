# FaithRec-Rerank Architecture

## 1. Research Goal

This project studies **faithfulness-aware reasoning for LLM-based recommender
reranking**.

The key claim we want to test is:

> A reasoning trace is not useful enough if it only looks plausible. It should
> be grounded in observable recommendation evidence and should change in
> predictable ways when that evidence is removed or perturbed.

This keeps the project close to current LLM4Rec reasoning papers while adding a
clear missing piece: recommendation-specific reasoning faithfulness.

## 2. Problem Definition

We use candidate-aware reranking rather than full-catalog item generation.

For a user \(u\), let:

```text
H_u = [(i_1, r_1, t_1), ..., (i_T, r_T, t_T)]
C_u = {c_1, ..., c_K}
X_i = textual metadata for item i
S_C = optional retriever rank or score for candidates
```

The reranker receives:

```text
Input x_u = (H_u, C_u, X_H, X_C, optional S_C)
```

and outputs:

```text
Evidence set E_u
Reasoning trace T_u
Ranking pi_u over C_u
```

The prediction is:

```text
y_hat = pi_u[0]
```

The ground-truth next item \(y_u\) is used only for training and evaluation. It
must not appear in inference prompts.

### Candidate Set

The recommended main protocol is:

```text
C_u = {ground_truth} + hard negatives
```

Three candidate policies are supported:

```text
random_negative: R2Rec-style controlled candidate pool
popularity_negative: cheap Amazon Food pilot candidate pool
retriever_hard_negative: SASRec/LightGCN retrieved hard candidates
```

The project should start with `popularity_negative` on the local Amazon Food
pilot, then move to `retriever_hard_negative` once the data pipeline is stable.

## 3. Reference Positioning

The local `ref/` papers motivate the architecture:

```text
CoT-Rec      -> personalized preference and item perception can help reranking.
GOT4Rec      -> decomposed reasoning can better use short/long/collaborative evidence.
R2Rec        -> interaction structures can be transformed into reasoning traces.
ReRec        -> recommendation-specific reward shaping improves RFT.
ThinkRec     -> reasoning activation and expert personalization improve LLM4Rec.
Latent-R3    -> explicit CoT is expensive and CoT labels are hard to obtain.
SIREN        -> compact interest sketches and internalized reasoning reduce latency.
```

The gap:

```text
These methods optimize recommendation accuracy and often produce explanations,
but they do not fully verify whether the explanation is causally grounded in the
evidence the recommendation should depend on.
```

The general LLM literature on CoT faithfulness motivates the verification side:

```text
Language Models Don't Always Say What They Think
Measuring Faithfulness in Chain-of-Thought Reasoning
Faithful Chain-of-Thought Reasoning
The Probabilities Also Matter
Reasoning Models Don't Always Say What They Think
```

We adapt these ideas to recommendation, where evidence is structured as user
history, item metadata, candidate text, retriever priors, and optional derived
profiles.

## 4. Model Choice

Default model:

```text
Qwen2.5-1.5B-Instruct
```

Reasons:

```text
1. small enough for local LoRA/RL experiments
2. instruction-following is better than tiny base LMs
3. compatible with Hugging Face, PEFT, and TRL-style training
4. smaller than the 3B/4B/8B models used in many ref papers, so pilot costs stay low
```

Recommended model ladder:

```text
Smoke test: Qwen2.5-0.5B-Instruct
Pilot:      Qwen2.5-1.5B-Instruct
Main:       Qwen2.5-3B-Instruct if compute allows
```

Do not mix model sizes across baselines in the same comparison.

## 5. Dataset Choice

Use local Amazon Food first because it is already available:

```text
data/raw/amazon-food/Grocery_and_Gourmet_Food.train.csv
data/raw/amazon-food/Grocery_and_Gourmet_Food.valid.csv
data/raw/amazon-food/Grocery_and_Gourmet_Food.test.csv
data/raw/amazon-food/meta_Grocery_and_Gourmet_Food.jsonl
```

The CSV splits already provide:

```text
user_id,parent_asin,rating,timestamp,history
```

The metadata JSONL provides:

```text
parent_asin,title,categories,store,features,description,details,average_rating
```

Pilot subset:

```text
train: 2k to 10k instances
valid: 500 instances
test:  500 instances
history length: 5 to 30
candidate size: 20
```

The first pilot should not use the 5.9GB review JSONL. It is too expensive and
not required for evidence grounding.

Future datasets:

```text
MovieLens-1M: clean item attributes and strong interpretability.
Amazon Book: richer text metadata and closer to R2Rec/ReRec references.
Amazon Electronics: noisier e-commerce domain for robustness.
RecBench+: query-based transfer setting aligned with ReRec.
```

## 6. Evidence-grounded Reasoning Data

Each reasoning instance contains four evidence pools:

```text
history_evidence:    user history items and their metadata
candidate_evidence:  candidate item metadata
retriever_evidence:  candidate rank/score, if available
derived_evidence:    optional profile or item perception generated offline
```

The LLM output must cite evidence IDs, not raw hidden assumptions.

Example evidence unit:

```json
{
  "evidence_id": "H03",
  "source": "history",
  "item_id": "B01NAYX4S3",
  "text": "User rated a dark roast coffee product 5 stars.",
  "attributes": {
    "category": "Coffee",
    "brand": "example"
  }
}
```

Example output contract:

```text
Evidence Selection:
E1: [H03] supports candidates in coffee or beverage categories.
E2: [H07] conflicts with candy-like snacks.

Candidate Reasoning:
A: supported by E1, weak conflict with E2.
B: little support from selected evidence.

Final Answer:
{"ranking": ["A", "C", "B"], "selected_candidate_id": "A", "evidence_refs": ["H03", "H07"]}
```

The output is considered invalid if it cites evidence IDs that are not present
in the input.

## 7. Training Framework

Training has five stages.

### Stage 0: Data and Candidate Construction

Build deterministic JSONL instances:

```text
instance_id
history items with metadata
candidate items with metadata
target candidate id
candidate construction policy
```

### Stage 1: Prompt-only Baselines

Run without training:

```text
direct_answer
free_form_cot
evidence_rerank_v1
```

This tests whether the faithfulness issue exists before we train anything.

### Stage 2: Evidence Trace SFT

Use heuristic or teacher-generated traces to teach the model the output format.

SFT is not the main novelty. It is a warm start so that RL is not dominated by
format errors.

### Stage 3: Rank-only RL/RFT

Train with ranking reward only:

```text
R = R_rank + R_format
```

This is the key ablation against ReRec/R2Rec-style reward training.

### Stage 4: Faithfulness-aware RL/RFT

Train with:

```text
R = R_rank
  + alpha * R_grounding
  + beta  * R_counterfactual
  + gamma * R_format
  - delta * R_cost
```

Recommended algorithm:

```text
GRPO-style group sampling with LoRA adapters
```

If online counterfactual evaluation is too expensive, compute it on a subset of
rollouts or use a cached verifier for most updates.

## 8. Reward Design

### Ranking Reward

Use candidate ranking metrics:

```text
Hit@1, NDCG@3, NDCG@5
```

For RL, use a scalar:

```text
R_rank = 1.0 if selected_candidate_id == target else NDCG@K
```

### Grounding Reward

Reward valid evidence references:

```text
valid refs only
refs come from input
refs have plausible lexical/category overlap with cited candidate
no unsupported invented facts
```

### Counterfactual Reward

Run two perturbations:

```text
targeted_drop: remove cited support evidence
irrelevant_drop: remove uncited evidence
```

Desired behavior:

```text
targeted_drop should reduce confidence or ranking of the selected candidate
irrelevant_drop should keep the selected candidate stable
```

This adapts CoT faithfulness ideas from general LLM work to recommendation
evidence.

### Format Reward

Reward parseable final answers and candidate-only rankings:

```text
ranking contains every candidate exactly once
selected_candidate_id == ranking[0]
evidence_refs is a subset of provided evidence IDs
```

## 9. Evaluation Protocol

Main metrics:

```text
ranking: HR@1, HR@5, NDCG@5, MRR
format: parse success, candidate-only rate, evidence-ref validity
faithfulness: targeted evidence sensitivity, irrelevant evidence stability
efficiency: input tokens, reasoning tokens, latency
```

Baselines:

```text
Retriever prior only
Direct LLM reranker
Free-form CoT reranker
Evidence prompt reranker without training
Evidence SFT
Rank-only RL
Faithfulness-aware RL
```

Ablations:

```text
without counterfactual reward
without grounding reward
without evidence selection
random negatives vs popularity negatives vs retriever hard negatives
small model vs stronger model, if compute allows
```

## 10. Missing Design Pieces Added

The original high-level idea needs these extra controls:

```text
1. leakage control: target id must never appear in prompts as ground truth
2. candidate policy: report how candidates are constructed
3. metadata index: build only metadata needed by sampled ASINs
4. cost budget: counterfactual reward should be batched or sampled
5. label separation: evidence traces can use labels only during training data creation
6. baseline parity: all LLM baselines use the same model and candidate pool
7. failure taxonomy: hallucinated evidence, invalid candidate, unstable answer, overlong trace
```

These controls are important for reviewer credibility.

## 11. Recommended First Milestone

The first useful milestone is not full RL. It is:

```text
1. Build 500 valid + 500 test Amazon Food instances.
2. Run direct, free-form CoT, and evidence-rerank prompts.
3. Measure whether cited evidence passes counterfactual tests.
4. Show that plausible explanations can be unfaithful.
```

If this phenomenon is clear, move to SFT and faithfulness-aware GRPO.
