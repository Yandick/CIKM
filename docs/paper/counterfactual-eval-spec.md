# TRACE-Rec V1 Counterfactual Evaluation Spec

Date: `2026-04-22`

## 🎯 Evaluation Goal

The goal is not to test whether the model is sensitive to input changes in general.

The goal is to test whether:

1. the graph changes for the right reasons
2. the recommendation changes in the right direction
3. irrelevant perturbations do not cause spurious reasoning updates

This evaluation is intentionally minimal. It is designed to be reliable on a first prototype.

## Why Keep It Small

A counterfactual benchmark can become more complicated than the method itself. That would be a mistake here.

TRACE-Rec V1 should use:

- `3` required intervention templates plus `1` optional extension
- `4` core metrics
- simple pass/fail behavior

That is enough to distinguish:

- free-form CoT that merely rephrases itself
- structured reasoning that updates in a controlled and checkable way

## 🧭 Principle

Every intervention should have:

1. a clearly identified affected component
2. a clear expected graph update
3. a clear expected recommendation direction

If an intervention does not have these three properties, do not include it in V1.

## Common Eval View

To compare TRACE-Rec and free-form CoT fairly, both models should emit a tiny normalized `eval_view` in addition to their normal output.

```json
{
  "recent_support": "up|same|down",
  "long_support": "up|same|down",
  "aversion": "up|same|down",
  "candidate_match": "up|same|down",
  "final_choice": "item_id"
}
```

For TRACE-Rec:

- `recent_support` is derived from `preference_state` nodes with `horizon=recent`
- `long_support` is derived from `preference_state` nodes with `horizon=persistent`
- `aversion` is derived from negative-polarity `preference_state` nodes
- `candidate_match` is derived from `candidate_evidence`
- in the current code path, this view should be projected automatically from the original/updated graph pair rather than handwritten

For free-form CoT:

- the model outputs the same compact view explicitly so both systems are scored on the same surface

## 🧪 Intervention Templates

## 1. Recent-Support Removal

Operation:

Remove one recent interaction that supports the predicted item.

Affected component:

- one `preference_state` node with `horizon=recent`

Expected graph update:

- that recent preference-state node is removed, weakened, or no longer connects to the same candidate-evidence node with `supports`

Expected decision update:

- the originally selected candidate should become less favored
- the decision may change, but does not have to change if persistent preferences still support it

Why this one matters:

It tests whether the model really uses recent behavior rather than merely mentioning it.

## 2. Long-Support Removal

Operation:

Remove a small cluster of older interactions that reflect a durable preference.

Affected component:

- one `preference_state` node with `horizon=persistent`

Expected graph update:

- long-support weakens while unrelated recent-support factors remain stable

Expected decision update:

- the originally selected candidate should become less favored if it depended on persistent support

Why this one matters:

It separates recent-behavior effects from durable-preference effects without requiring separate node types in the graph.

## 3. Candidate-Feature Swap

Operation:

Alter or swap a key feature of the selected candidate, or replace the selected candidate with a matched alternative that differs on one critical feature.

Affected component:

- one `candidate_evidence` node and the edges touching it

Expected graph update:

- support or opposition edges involving that candidate should update
- unrelated preference-state nodes should remain mostly stable

Expected decision update:

- the modified candidate should lose support if the removed feature was crucial
- its rank should move in the expected direction

Why this one matters:

It checks whether candidate-conditioned reasoning is real rather than generic preference summarization.

## Optional 4. Hard-Aversion Injection

Operation:

Add an explicit dislike or exclusion constraint tied to the candidate or its category.

Affected component:

- negative-polarity `preference_state`

Expected graph update:

- aversion appears or strengthens
- support for the affected candidate should weaken or flip to conflict

Expected decision update:

- the affected candidate should be demoted

Why this one matters:

It directly tests whether negative preference can be expressed as signed structure rather than only prose.

If the initial dataset does not support reliable negative constraints, keep this as an optional extension rather than a required V1 intervention.

## 📏 Core Metrics

## 1. Base Utility

Use:

- `HR@10`
- `NDCG@10`

Definition:

Recommendation quality on untouched examples.

This is only a guardrail so faithfulness gains are not achieved by collapsing utility.

## 2. Targeted Update Accuracy

Abbreviation:

`TUA`

Definition:

Did the factor touched by the intervention change in the expected direction in the `eval_view` and underlying graph?

Examples:

- recent-support removal should make `recent_support=down`
- long-support removal should make `long_support=down`
- hard-aversion injection should make `aversion=up`
- candidate-feature swap should make `candidate_match=down`

This is the main graph-faithfulness metric.

## 3. Decision Direction Consistency

Abbreviation:

`DDC`

Definition:

Did the affected candidate move in the expected ranking direction, or did the final choice update correctly when the intervention was decision-relevant?

Examples:

- removing key supporting evidence should not improve the candidate
- damaging a key candidate feature should not strengthen its rank

This is the main decision-faithfulness metric.

## 4. Non-target Stability

Abbreviation:

`NTS`

Definition:

Did the factors not targeted by the intervention stay mostly unchanged?

Examples:

- a recent-support removal should not also flip `long_support`
- a candidate-feature swap should not arbitrarily change unrelated support factors

Higher is better.

This is the main locality metric.

## ✅ Pass / Fail Behavior

TRACE-Rec should be considered healthy in V1 if:

1. the targeted factor updates correctly
2. the recommendation direction usually matches intervention semantics
3. non-target factors remain mostly stable
4. base utility remains competitive

TRACE-Rec should be considered weak if:

1. targeted factors often stay unchanged under relevant interventions
2. recommendation direction does not match intervention semantics
3. non-target factors change too often
4. gains vanish when compared against a carefully prompted free-form CoT baseline

## Suggested Pilot Thresholds

These are internal research thresholds, not final paper claims.

- graph parse success rate: `> 90%`
- `TUA` should beat free-form CoT by a clear margin
- `DDC` should beat free-form CoT on decision-relevant interventions
- `NTS` should not be worse than free-form CoT
- base utility should stay within a small margin of the stronger model

If those conditions are not met, simplify the method before scaling experiments.

## 🔗 Mapping Interventions to V1 Graph

| Intervention | Expected graph target | Expected decision behavior |
| --- | --- | --- |
| recent-support removal | recent `preference_state` or its signed edges | target candidate weakens or stays only if persistent support is strong |
| long-support removal | persistent `preference_state` or its signed edges | target candidate weakens if it depended on long-term support |
| candidate-feature swap | edges touching one `candidate_evidence` node | target candidate weakens if key feature is removed |
| hard-aversion injection | negative-polarity `preference_state` | target candidate is demoted |

## 📦 Evaluation Output Format

For each example, save:

- original input
- original graph
- original ranking
- intervention description
- counterfactual input
- counterfactual graph
- counterfactual ranking
- automatically computed metric flags

This makes both quantitative scoring and qualitative inspection easy.

## Recommendation

Do not expand the counterfactual benchmark until the model already shows signal on this minimal protocol.

The first paper only needs to prove:

`our structure updates more correctly and more stably than free-form CoT`

That is enough for V1.
