# Reward and Verifier Spec

Date: `2026-04-22`

## Goal

Define a first practical reward and verifier contract for:

`answer-only candidate reranking policy + optional audit graph`

This spec follows the framing update in [2026-04-22-framing-update.md](2026-04-22-framing-update.md).

## Design Principles

1. Reward the decision, not the prose.
2. Treat schema validity as a hard gate.
3. Use a programmatic verifier first.
4. Keep counterfactual faithfulness central.
5. Treat compactness only as a weak regularizer.
6. Start with offline scoring before full RL.

## Minimal Verifier Contract

### Policy Input

- `NextItemExample`
- user history
- candidate set
- target item for supervision
- evidence / feature registries in `context`

### Policy Output

- `selected_item_id`

### Optional Audit Artifact

- a compact `TRACE-Rec V1` graph
- or a compact `eval_view`

The audit artifact is optional at inference and mainly used for training, offline checking, and faithfulness evaluation.

## Programmatic Checks

The first verifier should check:

1. selected item belongs to the candidate set
2. optional audit graph is valid under the current schema
3. audit graph decision matches the policy prediction
4. evidence refs and feature refs are grounded in the registries
5. selected candidate has at least one supporting path
6. at least one contrasted candidate receives conflict evidence when available
7. counterfactual graph updates move in the expected direction

## Reward Components

### Hard Gate

- `R_schema`
- if the answer is invalid, or an attached audit graph is invalid or inconsistent, total reward is `0`

### Positive Terms

- `R_utility`
  - main task utility
  - use reciprocal rank when a ranking is available
  - reduce to top-1 hit in answer-only mode
- `R_grounding`
  - whether attached evidence pointers are valid
- `R_consistency`
  - whether the audit graph decision matches the selected answer
- `R_support`
  - whether the selected candidate has grounded support
- `R_conflict`
  - whether at least one non-selected candidate is explicitly ruled out
- `R_cf`
  - targeted counterfactual update accuracy
- `R_locality`
  - non-target stability under interventions

### Weak Penalty

- `R_cost`
  - weak penalty for long outputs or oversized audit graphs
  - never strong enough to dominate utility or faithfulness

## Default Weighting

The first default weighting is:

```text
utility       = 1.00
counterfactual= 1.00
grounding     = 0.50
consistency   = 0.50
support       = 0.50
conflict      = 0.25
locality      = 0.25
cost_penalty  = 0.10
```

The positive terms are averaged over the available components, then the cost penalty is subtracted.

## Training Path

The first practical path is:

1. answer-only SFT
2. offline verifier scoring
3. verifier-guided reranking and calibration
4. only then consider GRPO-style sequence-level RL

Do not start with token-level or reasoning-segment-level credit assignment.

## Advantage Rule

Use query-wise group-relative advantage first:

```text
A_i = clip((R_i - mean(R_group)) / (std(R_group) + eps), -a, a)
```

This keeps the first RL loop simple and avoids premature complexity from segment-level penalties.

## Main Risks

1. evidence omission gaming
2. verifier proxy hacking
3. counterfactual overfitting
4. popularity shortcutting under sparse utility rewards
5. stability collapse if counterfactual pressure is too high
