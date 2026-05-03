# FaithRec RL Training Design

## 1. Training Objective

The goal is to train an LLM reranker that is accurate and evidence-faithful.

The policy generates:

```text
Evidence Selection -> Candidate Reasoning -> Final Answer JSON
```

The scalar reward is:

```text
R = w_rank * R_rank
  + w_ground * R_grounding
  + w_cf * R_counterfactual
  + w_format * R_format
  - w_cost * R_cost
```

## 2. Algorithm

Use a GRPO-style RFT loop with LoRA adapters:

```text
1. For each prompt, sample G completions.
2. Parse final answers and evidence refs.
3. Compute reward components for each completion.
4. Normalize rewards within group or batch.
5. Update the LoRA policy with a KL penalty to the reference model.
```

Why GRPO:

```text
R2Rec uses SFT + RL for recommendation reasoning.
ReRec uses RFT and reward shaping for complex recommendation queries.
Latent-R3 modifies GRPO for efficient recommendation reasoning.
```

We use the same broad family but add evidence faithfulness as a reward signal.

## 3. Stages

### Stage A: Format SFT

Train on a small set of traces to reduce parse failures.

Sources:

```text
heuristic evidence traces
teacher LLM traces
manual spot-corrected traces
```

Do not rely on large SFT as the main claim.

### Stage B: Rank-only RL

Reward:

```text
R = R_rank + 0.2 * R_format
```

This is the main RL baseline.

### Stage C: Faithfulness-aware RL

Reward:

```text
R = R_rank
  + 0.25 * R_grounding
  + 0.25 * R_counterfactual
  + 0.20 * R_format
  - 0.05 * R_cost
```

Weights are validation hyperparameters, not fixed claims.

## 4. Reward Components

### R_rank

Recommended scalar:

```text
R_rank = 1.0 if selected == target
       = NDCG@5 otherwise
```

For a 20-candidate pool, this provides dense feedback when the target is near
the top but not selected.

### R_grounding

Checks:

```text
valid evidence IDs
no invented evidence
history refs are semantically related to the stated preference
candidate refs are used only for candidate properties
```

Initial implementation can use deterministic lexical/category overlap. Later,
replace or augment it with a learned verifier.

### R_counterfactual

Definitions:

```text
targeted_drop_score:
  Selected candidate should lose rank or confidence when cited support evidence is removed.

irrelevant_drop_score:
  Selected candidate should remain stable when uncited evidence is removed.
```

Since full counterfactual rollout is expensive, use:

```text
online reward on 20-30% of batches
cached audit completions for repeated prompts
offline full evaluation on validation/test
```

### R_format

Checks:

```text
valid JSON
complete ranking
candidate-only ranking
selected_candidate_id == ranking[0]
evidence_refs subset of input evidence IDs
```

### R_cost

Penalize excessive traces:

```text
completion_tokens / max_allowed_completion_tokens
```

This prevents the model from maximizing faithfulness by generating long,
unusable explanations.

## 5. Data Needed for Training

Each RL record should include:

```json
{
  "instance_id": "amazon_food_valid_000001",
  "prompt": "...",
  "candidate_ids": ["A", "B"],
  "evidence_ids": ["H01", "H02", "A", "B"],
  "target_candidate_id": "B",
  "counterfactual_plans": {
    "targeted_drop": ["H02"],
    "irrelevant_drop": ["H05"]
  }
}
```

Counterfactual plans can be generated after the first model output because the
targeted evidence depends on the cited refs. For offline evaluation, store the
actual removed refs.

## 6. Feasibility Risks

```text
Risk: Counterfactual reward is too slow.
Mitigation: use it sparsely online, fully offline.

Risk: Evidence grounding becomes lexical matching only.
Mitigation: report lexical verifier separately; later add embedding or LLM verifier.

Risk: Model learns to cite many history items to pass audits.
Mitigation: cap evidence refs and penalize length.

Risk: Popularity negatives make the task too easy.
Mitigation: move to SASRec/LightGCN hard negatives for main experiments.

Risk: Faithfulness improves but accuracy drops.
Mitigation: tune reward weights and report Pareto trade-off.
```

## 7. Minimum Viable RL Experiment

```text
dataset: Amazon Food pilot
train: 2k instances
valid: 500 instances
candidate size: 20
model: Qwen2.5-1.5B-Instruct + LoRA
baselines: evidence SFT, rank-only GRPO, faithfulness-GRPO
main outputs: HR@1, NDCG@5, evidence validity, targeted sensitivity, stability
```
