# TRACE-Rec Execution Roadmap

> Status note (`2026-04-22`): this roadmap was drafted before the latest baseline audit.
> Teacher-trace-heavy steps should now be treated as optional. The current primary framing is [2026-04-22-framing-update.md](2026-04-22-framing-update.md).

Date: `2026-04-22`

## 🎯 Objective

Move the project from a plausible paper idea to a defensible research program with:

1. a locked problem definition
2. a minimal but convincing experimental pipeline
3. early evidence on whether the idea is publishable

This roadmap assumes the current target is:

`typed, evidence-grounded, candidate-conditioned structured reasoning for generative next-item recommendation`

## 🧭 Guiding Principle

Do not start by scaling models or training long runs.

The critical path is:

1. lock the structure
2. lock the evaluation
3. beat trivial baselines on small data
4. only then scale

If those steps are skipped, later gains will be hard to interpret and the paper can collapse into "another prompt variant".

## 📌 Workstreams

The project should run as four parallel but coordinated workstreams.

### 1. Problem and Paper Framing

Deliverables:

- frozen task definition
- precise contribution claims
- intro narrative
- related-work positioning

### 2. Structure and Evaluation Design

Deliverables:

- final rationale schema
- parser and validator specification
- counterfactual intervention protocol
- metric definitions

### 3. Data and Experiment Pipeline

Deliverables:

- benchmark preprocessing
- candidate construction
- baseline runners
- experiment tracking

### 4. Method Implementation

Deliverables:

- teacher graph generation
- student training
- optional RFT or RL
- optional distilled inference

## 🗓️ Six-Week Plan

## Week 1: Freeze the Research Contract

Goal:

Make the paper testable before writing code that depends on unstable assumptions.

Deliverables:

- final node types, edge types, and evidence-pointer rules
- counterfactual intervention taxonomy
- primary datasets selected
- primary metrics selected
- baseline shortlist frozen
- one-page method summary

Exit criteria:

- you can state in one paragraph exactly what TRACE-Rec predicts and how it is evaluated
- two people reading the schema would annotate the same example similarly

## Week 2: Build Minimal Data + Evaluation Infrastructure

Goal:

Get a pipeline that can run end to end on one dataset.

Deliverables:

- preprocessing for `Amazon Food`
- candidate construction for next-item recommendation
- JSON schema for rationale graphs
- parser and validator
- counterfactual example builder
- offline metric runner

Exit criteria:

- one sample can flow from raw history to candidate set to parsed graph to metrics
- invalid graphs can be automatically rejected

## Week 3: Establish Minimal Baselines

Goal:

Know what "good enough to continue" means.

Deliverables:

- base non-reasoning LLM baseline
- free-form CoT baseline
- one structured baseline proxy if full reproduction is too heavy
- small benchmark table on `Amazon Food`

Priority order:

1. base LLM recommender
2. free-form CoT baseline
3. `GOT4Rec`-style structured prompt proxy
4. `OneRec-Think`-style explicit reasoning proxy

Exit criteria:

- you know whether structured reasoning helps at all on the small setup
- you can measure cost in tokens and latency, not only ranking quality

## Week 4: Teacher Trace and Student SFT

Goal:

Turn the idea into an actual trainable method.

Deliverables:

- teacher prompt for graph generation
- trace filtering pipeline
- student SFT objective
- first small-run TRACE-Rec model

Exit criteria:

- generated traces parse cleanly at a reasonable rate
- student can produce legal graphs on held-out samples

## Week 5: Faithfulness and Ablation Phase

Goal:

Test the paper's unique claim.

Deliverables:

- counterfactual-faithfulness benchmark
- node-ablation analysis
- graph-vs-linear-CoT ablation
- evidence-grounding ablation
- candidate-conditioning ablation

Exit criteria:

- TRACE-Rec must win on at least one faithfulness metric in a stable way
- otherwise the paper story is too weak

## Week 6: Scale or Pivot

Goal:

Decide whether to push for a paper-quality full study or pivot.

Deliverables:

- second and third datasets
- larger comparison table
- explicit vs distilled inference comparison
- draft paper figures
- abstract and intro draft

Decision:

- scale if the method shows utility plus faithfulness gains
- pivot if gains are only cosmetic or only prompt-format dependent

## 🔬 Experimental Order

The implementation order should be:

1. one dataset
2. one simple candidate constructor
3. one non-reasoning baseline
4. one free-form CoT baseline
5. TRACE-Rec without RL
6. faithfulness benchmark
7. only then add more baselines, more datasets, or RL

This order matters because the main risk is not weak performance. The main risk is an untestable contribution.

## ⚖️ Baseline Priority

Do not try to reproduce every recent paper immediately.

Use this order:

### Must have

- base LLM recommender
- free-form CoT baseline
- one structured-prompt baseline

### Should have

- `GOT4Rec`
- `OneRec-Think`
- `R2Rec`

### Nice to have

- `SCoTER`
- `SIREN`
- `LatentR3`
- `Reasoning to Rank`

If a baseline is too expensive or unavailable, build a close proxy first and document that decision.

## 🚪 Go / No-Go Criteria

By the end of Week 3, continue only if at least two of the following are true:

1. structured graphs parse reliably
2. graph outputs are more stable than free-form CoT under irrelevant perturbations
3. graph outputs update more correctly under relevant interventions
4. recommendation quality is at least competitive with the free-form CoT baseline

By the end of Week 5, continue to full scaling only if:

1. faithfulness gains are clear
2. the gains survive at least one ablation
3. the method is not just longer reasoning with more tokens

## 🚨 Pivot Triggers

Pivot away from the current formulation if any of these happen:

1. the graph is mostly a serialization of the answer with no measurable control benefit
2. free-form CoT matches faithfulness once prompted carefully
3. evidence grounding is too noisy to validate automatically
4. the method only works with expensive teacher traces and cannot transfer to a smaller student

## ✅ Immediate Next Tasks

The next concrete tasks should be:

1. finalize the JSON schema for rationale graphs
2. define 4-5 counterfactual intervention templates
3. choose `Amazon Food` as the first benchmark
4. implement a simple candidate constructor
5. implement graph parser + validator
6. implement two baselines:
   - base LLM
   - free-form CoT
7. build a 50-200 example pilot set for fast iteration

## 📄 Paper Writing Order

Write in this order:

1. problem definition
2. method section
3. experimental protocol
4. related work
5. abstract
6. intro

The intro should be written last enough that it matches the actual evidence.
