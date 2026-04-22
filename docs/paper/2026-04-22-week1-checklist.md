# TRACE-Rec Week 1 Checklist

> Status note (`2026-04-22`): this checklist predates the pivot away from a teacher-trace-first plan.
> Use [2026-04-22-framing-update.md](2026-04-22-framing-update.md) as the current decision record.

Date: `2026-04-22`

## Goal

Finish the minimum design work needed before coding the full method.

## Checklist

- [ ] Freeze node types
- [ ] Freeze edge types
- [ ] Define evidence-pointer format
- [ ] Define legal graph constraints
- [ ] Define recommendation output format
- [ ] Define 4-5 counterfactual interventions
- [ ] Select first dataset
- [ ] Select first candidate-construction strategy
- [ ] Freeze first three baselines
- [ ] Define main metrics
- [ ] Define pilot subset size
- [ ] Write the one-paragraph problem statement

## Expected Outputs

- `docs/paper/graph-schema-spec.md`
- `docs/paper/counterfactual-eval-spec.md`
- `configs/data/amazon_food.yaml`
- `configs/prompt/free_form_cot.yaml`
- `configs/prompt/rec_graph_cot.yaml`

## Non-Goals

Do not start:

- long training runs
- large-scale teacher trace generation
- multi-dataset comparison
- RL or RFT

Those come later.
