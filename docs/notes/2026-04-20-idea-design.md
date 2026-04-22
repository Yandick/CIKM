# Idea Design Note

Date: `2026-04-20`

Note:

This note captures early brainstorming. The executable repository target is now the smaller V1 schema in `docs/paper/graph-schema-spec.md`.

## Recommended Direction

Do not pursue:

- generic tree-of-thought for recommendation
- generic graph-of-thought for recommendation
- generic "structured CoT helps LLM4Rec"

Those directions are too close to existing work.

Pursue instead:

`candidate-conditioned preference graph reasoning for generative recommendation`

Strongest current variant:

`typed evidence-grounded structured CoT + counterfactual faithfulness`

## Core Thesis

Existing reasoning-based recommendation methods mostly do one of four things:

1. inject free-form CoT
2. use manually designed staged reasoning
3. transfer structured reasoning without tight candidate conditioning
4. replace explicit CoT with latent/internalized reasoning for efficiency

The gap worth attacking is:

`recommendation needs a structure that expresses preference factors, conflicts, temporal shifts, and candidate-specific evidence in a way that can be parsed, validated, rewarded, and distilled`

## Proposed Structure

### Preference Graph

Each example is converted into a small directed graph:

- `goal` node:
  what the user currently seems to want
- `short_term_interest` nodes:
  recent session or recency-sensitive evidence
- `long_term_interest` nodes:
  durable preference evidence
- `aversion` nodes:
  evidence for what should be avoided
- `temporal_shift` nodes:
  signals that recent behavior departs from historical preference
- `candidate_evidence` nodes:
  why a specific candidate matches or conflicts
- `decision` node:
  final recommendation or ranking decision

Edges:

- `supports`
- `conflicts`
- `dominates`
- `rejects`

Each non-decision node should be grounded by at least one explicit evidence pointer:

- history span
- item metadata field
- retrieved collaborative cue
- user constraint

## Why This Is More Defensible

This is more defensible than a generic graph/tree CoT because:

1. The structure is candidate-conditioned, not just a decomposition of user history.
2. The nodes are typed and evidence-grounded rather than free-form.
3. The nodes have recommendation semantics that can be checked.
4. The graph can be converted into rewards and ablations.
5. The same structure can support explicit reasoning during training and compact reasoning at inference.

## Strong Optional Pivot: Slate Recommendation

A single-item setting is simpler for the first pass.

But if novelty pressure remains high, the better paper angle may be:

`structured slate reasoning`

That means the reasoning graph is built over a candidate set or output slate with explicit constraints such as:

- relevance
- diversity
- novelty
- redundancy avoidance
- compatibility within the slate

This is more ambitious, but it opens cleaner space against single-output reasoning papers.

## Training Pipeline

### Stage 1: Task Alignment

- Choose a generative recommendation backbone.
- Align item representations with semantic IDs or textual item summaries.

### Stage 2: Teacher Graph Generation

- Use a stronger LLM to generate preference graphs from user history and candidate sets.
- Ask for explicit nodes and labeled edges instead of free-form rationale paragraphs.

### Stage 3: Graph Validation

Filter teacher graphs using:

- recommendation outcome correctness
- parse validity
- node coverage rules
- contradiction checks
- optional agreement with collaborative or retrieval signals
- evidence-pointer validity

### Stage 4: Student SFT

Train a smaller model to:

1. predict the structured graph
2. predict the final recommendation

### Stage 5: Structure-Aware RL or RFT

Reward components:

- recommendation utility
- graph validity
- candidate evidence consistency
- branch efficiency
- rationale length control
- counterfactual faithfulness

### Stage 6: Efficient Inference

Try one of two options:

1. answer-only inference after hidden alignment or distillation
2. compact structured rationale with fixed graph budget

This is the part where the project can learn from `SIREN` and `LatentR3` without copying them.

## Minimum Viable Paper

### Task

Start with sequential recommendation or next-item recommendation.

### Datasets

Pick 2-3 from:

- Amazon Beauty
- Amazon Sports
- Amazon Food
- Steam

Avoid starting with too many datasets before the pipeline is stable.

### Baselines

Core comparison set:

- non-reasoning LLM baseline
- `CoT-Rec`
- `ThinkRec`
- `GOT4Rec`
- `R2Rec`
- `OneRec-Think`
- `SCoTER` if reproducible
- one efficient reasoning baseline such as `SIREN` or `LatentR3`

If moving to slate recommendation, also compare against:

- `HiGR`

### Metrics

- `HR@5`, `HR@10`, `NDCG@5`, `NDCG@10`, `MRR`
- prompt tokens
- generated rationale tokens
- inference latency
- parse success rate
- contradiction rate
- faithfulness under node ablation
- faithfulness under counterfactual intervention

## Key Ablations

1. Replace the graph with linear CoT.
2. Remove candidate-conditioned nodes.
3. Remove aversion and conflict edges.
4. Remove temporal-shift nodes.
5. Distill to answer-only inference and compare quality vs latency.
6. Remove evidence grounding and keep only free-form structure.

## Paper Pitch Draft

We introduce a recommendation-specific reasoning structure for generative recommendation that represents compact preference states, conflicts, and candidate evidence as a typed, evidence-grounded graph. Unlike prior work on generic graph-of-thought or structured CoT for recommendation, our method is candidate-conditioned, structure-aware during training, and evaluated not only by recommendation accuracy but also by counterfactual faithfulness. The same structure is compatible with efficient inference through distillation or internalization.

## Main Risk

If the method ends up being only a prompt template plus a generic graph serialization, it will likely not survive comparison against `GOT4Rec` and `SCoTER`.

The contribution must therefore come from at least two of:

- a better recommendation-specific structure
- a stronger training objective
- a faithfulness evaluation story
- an efficiency transfer/internalization story

The strongest combination currently looks like:

- typed evidence-grounded structure
- counterfactual faithfulness evaluation
- optional slate-level reasoning
