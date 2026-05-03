# FaithRec-Rerank: Faithfulness-aware LLM4Rec Reranking

This workspace is for a CIKM-oriented research project on **LLM-based
candidate reranking with faithful, evidence-grounded reasoning**.

The core question is not only whether LLM reasoning improves recommendation,
but whether the model's stated reasoning is actually grounded in the user
history, item metadata, and candidate evidence used for the decision.

## Problem

We follow the candidate-aware reranking setting used by recent LLM4Rec work:
a retriever first produces a small candidate set, then an LLM reranks the
candidates.

Given:

```text
H_u: user interaction history
C_u: candidate set from a frozen retriever or sampling policy
X_C: candidate text and metadata
S_C: optional retriever ranks or scores
```

the LLM reranker outputs:

```text
evidence -> reasoning -> ranking(C_u)
```

The model must not recommend items outside the candidate set.

## Current Research Direction

Existing reference papers in `ref/` cover:

- CoT-Rec: personalized information extraction and utilization.
- GOT4Rec: graph-of-thought decomposition for sequential recommendation.
- R2Rec: interaction-chain reasoning and SFT + RL.
- ReRec: reinforcement fine-tuning with recommendation-specific rewards.
- ThinkRec: reasoning activation and personalized expert fusion.
- Latent-R3 / SIREN: efficient latent or internalized reasoning.

Our positioning:

```text
From "make LLM4Rec generate reasoning" to
"make LLM4Rec reasoning verifiable and useful for RL training".
```

## Main Design

The project introduces a faithfulness-aware reranking framework:

1. Build evidence-grounded reasoning traces.
2. Evaluate whether cited evidence causally affects the recommendation.
3. Use the faithfulness signal as an auxiliary reward during RL/RFT.

The default reward is:

```text
R = R_rank
  + alpha * R_grounding
  + beta  * R_counterfactual
  + gamma * R_format
  - delta * R_cost
```

## Pilot Setup

The local `data/raw/amazon-food` dataset is large, so the first pilot uses a
small deterministic subset from the existing CSV splits:

```text
data/raw/amazon-food/Grocery_and_Gourmet_Food.train.csv
data/raw/amazon-food/Grocery_and_Gourmet_Food.valid.csv
data/raw/amazon-food/Grocery_and_Gourmet_Food.test.csv
data/raw/amazon-food/meta_Grocery_and_Gourmet_Food.jsonl
```

The pilot avoids the huge review JSONL unless review text becomes necessary.
Metadata is enough for the first evidence-grounding experiments.

## Default Small Model

The default local LLM is:

```text
Qwen2.5-1.5B-Instruct
```

Use `Qwen2.5-0.5B-Instruct` for smoke tests and `Qwen2.5-3B-Instruct` for a
stronger main run if local compute allows it.

## Key Files

- [Architecture](docs/faithrec_architecture.md)
- [Prompt Contract](docs/faithrec_prompt_contract.md)
- [RL Training Design](docs/faithrec_rl_training.md)
- [Evaluation Protocol](docs/faithrec_eval_protocol.md)
- [Amazon Food Pilot Config](configs/data/amazon_food_pilot.yaml)
- [Small Model Config](configs/model/qwen2_5_1_5b_instruct.yaml)
- [Evidence Rerank Prompt Config](configs/prompt/evidence_rerank_v1.yaml)
- [Faithfulness GRPO Config](configs/train/faithfulness_grpo.yaml)
- [Faithfulness Eval Config](configs/eval/faithfulness_counterfactual.yaml)

Earlier topology-oriented drafts remain in `docs/toporec_*` and can be reused
as ablations, but the current primary direction is faithfulness-aware reranking.
