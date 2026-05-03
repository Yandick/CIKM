# TopoRec-Family Training Design

This document defines the training plan after fixed-topology prompt experiments. The main principle is to separate topology effects from tuning effects.

The updated training target is:

```text
topology-specific natural-language trace + minimal final answer JSON
```

The model should not be trained to emit large nested JSON reasoning traces unless that is used as a specific ablation.

## 1. Training Goal

The model should learn to rerank candidates under a selected reasoning topology:

\[
s(c) = g_m(H_u, C_u, X_C, A_u), \quad m \in \{\text{direct}, \text{linear}, \text{tree}, \text{graph}\}.
\]

The training design has three goals:

```text
1. teach the model to follow topology-specific natural-language trace formats
2. improve candidate ranking accuracy
3. eventually train a router that selects a topology under cost constraints
```

Router training should not start until fixed-topology runs and oracle topology analysis have been completed.

## 2. Overall Schedule

Recommended order:

```text
Stage 0: data and retriever preparation
Stage 1: prompt-only fixed topology evaluation
Stage 2: oracle topology label construction
Stage 3: topology-trace SFT
Stage 4: ranking-oriented RFT or GRPO
Stage 5: router training and joint evaluation
```

This order is important. If router or RL training starts too early, it becomes hard to know whether gains come from topology, prompt wording, data leakage, or tuning.

## 3. Stage 0: Data and Retriever Preparation

Main datasets:

```text
MovieLens-1M
Amazon Book
Amazon Electronics
```

Main candidate construction:

\[
C_u = \{y_u\} \cup \operatorname{Top}_{19}(R(H_u) \setminus \{y_u\}).
\]

Default retriever:

```text
SASRec
```

Optional robustness retriever:

```text
LightGCN
```

Default history length:

```text
L = 10 recent interactions
```

Data artifact format:

```json
{
  "instance_id": "u123_t17",
  "dataset": "ml1m",
  "history": [],
  "candidates": [],
  "label": {"target_candidate_id": "C"},
  "retriever": {
    "name": "SASRec",
    "recall_k": 0.0,
    "scores_available": true
  }
}
```

The target candidate ID is used only for training and evaluation. It must not appear in inference prompts.

## 4. Stage 1: Prompt-Only Fixed Topology

Before tuning, run:

```text
direct
linear
tree
graph
```

on the same sample set.

Record:

```text
ranking metrics
final-answer parse success rate
token cost
latency
trace section compliance
topology-specific win rate
```

This stage answers whether natural-language topology traces are executable and whether topology modes produce meaningfully different rankings.

Recommended pilot size:

```text
100 to 300 validation instances per dataset
```

Recommended full fixed-topology run:

```text
1000 validation/test instances per dataset, matching R2Rec scale if possible
```

## 5. Stage 2: Oracle Topology Labels

Run all topology modes on the same validation instances and compute:

\[
U_m(u) = \operatorname{Metric}(m,u) - \lambda \operatorname{Cost}(m,u).
\]

Then define:

\[
m_u^* = \arg\max_m U_m(u).
\]

Suggested utility:

```text
Metric = Hit@1 or NDCG@3
Cost = normalized completion tokens or latency
lambda = chosen from validation sweep
```

Store oracle labels:

```json
{
  "instance_id": "u123_t17",
  "oracle_topology": "tree",
  "utilities": {
    "direct": 0.20,
    "linear": 0.43,
    "tree": 0.62,
    "graph": 0.58
  },
  "winner_margin": 0.04,
  "difficulty_signals": {
    "uncertainty": 0.71,
    "conflict": 0.52,
    "noise": 0.33
  }
}
```

Important analysis:

```text
oracle distribution over topology modes
mode-specific win rate by dataset
mode-specific win rate by uncertainty/conflict/noise bins
oracle gain over best fixed topology
oracle gain over direct answer
```

If oracle gain is small, the adaptive topology story is weak and should be revised before training a router.

## 6. Stage 3: Topology-Trace SFT

SFT should teach format following and topology-specific reasoning style. It should not be presented as the main novelty.

Preferred design:

```text
shared base model + topology instruction token/template
```

Example control tokens:

```text
<TOPO_DIRECT>
<TOPO_LINEAR>
<TOPO_TREE>
<TOPO_GRAPH>
```

Training output should look like:

```text
Reasoning Trace:
<natural-language trace following the selected topology>

Final Answer:
{"ranking": ["A", "C", "B"], "selected_candidate_id": "A"}
```

Direct mode can omit the reasoning trace and return only the final answer block.

### 6.1 SFT Data Sources

SFT targets can be built from:

```text
teacher LLM outputs following docs/topology_prompt_templates.md
high-quality parsed outputs from pilot runs
synthetic natural-language traces generated from known labels and candidate metadata
small manually edited seed traces for each topology
```

Filtering rules:

```text
final answer JSON parses successfully
complete ranking
no candidate hallucination
trace follows the requested topology sections
reasonable token length
target appears in top positions for positive teacher-filtered examples
```

Training sample format:

```json
{
  "instruction": "Use the Linear protocol for candidate reranking.",
  "input": "serialized H_u, C_u, X_C, S_C",
  "output": "natural-language topology trace plus minimal final answer JSON"
}
```

### 6.2 SFT Objective

Basic objective:

\[
\mathcal{L}_{SFT} = -\sum_t \log p_\theta(y_t \mid x, y_{<t}).
\]

Recommended masking:

```text
mask prompt tokens
train on trace and final answer tokens
optionally upweight final answer tokens
```

Optional field-aware loss:

```text
trace section headings: low weight
reasoning text: medium weight
final ranking and selected_candidate_id: high weight
```

This is lighter than full JSON-trace SFT and should be easier to train.

## 7. Stage 4: Ranking-Oriented RFT or GRPO

After SFT stabilizes final-answer parsing, use ranking reward to improve recommendation quality.

Reward components:

```text
answer_format_reward
ranking_reward
trace_compliance_reward
cost_penalty
optional verifier_reward
```

Suggested reward:

\[
R = R_{rank} + \alpha R_{answer} + \gamma R_{trace} - \beta R_{cost}.
\]

Where:

```text
R_rank = 1 if Hit@1 else NDCG@3 or MRR
R_answer = 1 if final answer JSON is valid else 0
R_trace = 1 if required topology sections are present else 0
R_cost = normalized completion tokens or latency
```

For GRPO-style training:

```text
sample multiple outputs per instance under the same topology
score each output with ranking, answer format, trace compliance, and cost reward
optimize relative advantage within the group
```

Topology-specific RFT order:

```text
first train direct or linear for stability
then train tree
then train graph
finally compare shared-topology model against separate topology adapters
```

Avoid using verifier or item-attribute graph as prompt input during RFT unless it is explicitly part of an ablation. If used, it should be reward-side only.

## 8. Stage 5: Router Training

Router training starts after oracle labels exist.

Router input features:

```text
retriever score entropy
top-1/top-2 retriever margin
candidate semantic similarity
history length
duplicate ratio
rating variance
recent-vs-long-term preference shift
direct LLM answer margin
estimated token budget
```

Router labels:

```text
oracle_topology from Stage 2
```

Router objective:

\[
\mathcal{L}_{route} = \operatorname{CE}(r_\theta(z_u), m_u^*).
\]

Cost-aware variant:

\[
m_u^* = \arg\max_m \operatorname{Metric}(m,u) - \lambda \operatorname{Cost}(m,u).
\]

Router baselines:

```text
random router
always direct
always linear
always tree
always graph
heuristic router
oracle router
learned router
```

Router evaluation:

```text
oracle agreement
routing regret
final HR@1 / NDCG@3
cost reduction relative to always graph
accuracy-cost Pareto curve
```

## 9. Heuristic Router Before Learned Router

A simple heuristic router is useful as a sanity check:

```text
if budget is very tight:
  choose direct
else if uncertainty is low and conflict is low:
  choose linear
else if conflict is moderate or candidate similarity is high:
  choose tree
else if uncertainty is high or conflict is high or noise is high:
  choose graph
else:
  choose tree
```

This router should be treated as a baseline. The learned router must show lower regret or better accuracy-cost tradeoff.

## 10. Main Comparisons

Main model variants:

```text
Direct
Vanilla CoT
Fixed Linear
Fixed Tree
Fixed Graph
TopoRec-Heuristic
TopoRec-Learned
TopoRec-Oracle
```

External baselines:

```text
R2Rec on MovieLens-1M, Amazon Book, Amazon Electronics
ReRec on RecBench+ Movie and Book as supplementary query-based comparison
SASRec retriever baseline
LightGCN or GRU4Rec if available
```

## 11. Required Ablations

Minimum ablations:

```text
without uncertainty signal
without conflict signal
without noise signal
without retriever score
without optional derived profile
same token budget across topology modes
natural-language trace vs full JSON trace
random negatives vs retrieved hard negatives
random router vs learned router
oracle router upper bound
```

Training ablations:

```text
without SFT
without RFT
SFT only
RFT only, if stable
shared LoRA vs topology-specific LoRA
answer-format reward only vs ranking reward
trace-compliance reward only vs ranking reward
```

## 12. Go/No-Go Criteria

Continue to router training only if:

```text
fixed topology modes have high final-answer parse success
tree or graph wins on a non-trivial subset of instances
oracle topology improves over best fixed topology or always graph under cost-aware utility
uncertainty/conflict/noise signals correlate with oracle topology labels
natural-language traces do not create excessive token cost
```

If these are not true, revise the schema or task setting before investing in tuning.

## 13. Recommended First Implementation Milestone

The first implementation milestone should be:

```text
Dataset: MovieLens-1M and Amazon Book
Candidates: 20, using SASRec hard negatives
Topologies: direct, linear, tree, graph
Samples: 100 to 300 validation examples per dataset
Output: natural-language trace plus final answer JSON
Analysis: final-answer parse rate, HR@1, NDCG@3, token cost, oracle topology distribution
```

Only after this milestone should SFT data generation start.
