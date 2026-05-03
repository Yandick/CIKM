# TopoRec-Family: Problem Formulation and Method Draft

## 1. Motivation

Existing LLM4Rec reasoning work covers several important directions, including explicit CoT, Graph-of-Thought, latent reasoning, verifiable reasoning, and reasoning-enhanced RL. A safer positioning for this project is therefore not to claim the first use of graph or CoT reasoning in recommendation. Instead, the intended contribution is to define a minimal-input, candidate-aware recommendation reasoning setting and study how different reasoning topologies can be selected or collapsed under accuracy and efficiency constraints.

The key design principle is that topology should change the organization of evidence, not the amount of privileged information available to the model. This is important because many related implementations use limited online inputs: user histories, candidate item titles, optional retrieved scores, and occasionally offline-generated user or item summaries. Rich item-attribute graphs or verifier graphs are often used on the reward/evaluation side rather than as default prompt inputs.

## 2. Problem Formulation

Let \(\mathcal{U}\) denote users, \(\mathcal{I}\) denote the item catalog, and \(\mathcal{E}\) denote timestamped user-item interactions. For a user \(u\), the historical interaction sequence is

\[
H_u = [(i_1, r_1, t_1), \ldots, (i_T, r_T, t_T)],
\]

where \(i_j \in \mathcal{I}\), \(r_j\) is an optional rating or feedback label, and \(t_j\) is the interaction time. Each item \(i\) may have textual information \(x_i\), such as title, category, brand, or description, depending on the dataset.

The underlying task is sequential next-item recommendation:

\[
y_u = i_{T+1}.
\]

However, the main experimental protocol is candidate-aware reranking. A frozen retriever \(R\), such as SASRec, LightGCN, or another collaborative recommendation model, produces a candidate set:

\[
C_u = R_K(H_u) \subset \mathcal{I}.
\]

The LLM-based reranker receives \(H_u\), \(C_u\), and candidate text \(X_C = \{x_c \mid c \in C_u\}\), and produces ranking scores:

\[
s_\phi(c \mid H_u, C_u, X_C), \quad c \in C_u.
\]

The final prediction is

\[
\hat{y}_u = \arg\max_{c \in C_u} s_\phi(c \mid H_u, C_u, X_C).
\]

For controlled evaluation, we recommend using a retriever-based hard candidate pool:

\[
C_u = \{y_u\} \cup \operatorname{Top}_{K-1}(R(H_u) \setminus \{y_u\}).
\]

This keeps the ground-truth item in the candidate set while ensuring that the negative candidates are difficult because they are retrieved by a recommender. An end-to-end variant can use \(C_u = R_K(H_u)\) directly and separately report retriever Recall@K, since the reranker cannot recover the target if it is absent from the retrieved set.

## 3. Input Boundary

The default online input should be intentionally narrow:

\[
\mathcal{X}_u = (H_u, C_u, X_C, S_C),
\]

where \(S_C\) denotes optional retriever ranks or scores. The following resources may be used as optional offline artifacts:

\[
A_u = (P_u, D_C),
\]

where \(P_u\) is a generated user profile summary and \(D_C\) contains generated item descriptions or item perceptions. These should be ablated explicitly because not all benchmarks provide them naturally.

Item-attribute graphs, verifier graphs, and alignment graphs should not be assumed as default online prompt inputs. They can be used for reward shaping, verifier training, or evaluation-side analysis, but the main method should remain valid with only \(H_u\), \(C_u\), and item text.

## 4. Method Overview

TopoRec-Family consists of three components:

1. A signal extractor that computes routing features from minimal online inputs.
2. A topology router that selects a reasoning protocol under a budget.
3. A topology executor that performs candidate reranking using linear, tree, or graph reasoning.

Formally, the routing signal is

\[
z_u = h(H_u, C_u, X_C, S_C),
\]

and the topology mode is selected by

\[
m_u = r_\theta(H_u, C_u, X_C, z_u, B),
\]

where \(m_u \in \{\text{linear}, \text{tree}, \text{graph}\}\), and \(B\) is the available cost budget. The selected executor then produces candidate scores:

\[
s(c) = g_{m_u}(H_u, C_u, X_C, A_u).
\]

The important constraint is that \(g_{\text{linear}}\), \(g_{\text{tree}}\), and \(g_{\text{graph}}\) operate on the same information boundary. Their difference lies in reasoning structure rather than extra evidence.

## 5. Routing Signals

The router should avoid relying on LLM self-reported uncertainty as the only signal. The following signals are computable from standard recommendation inputs.

Uncertainty can be measured using retriever score entropy, top-1/top-2 margin, score distribution flatness, candidate title similarity, or the answer-token logit margin from a cheap direct LLM pass. High uncertainty suggests that a direct linear protocol may be insufficient.

Conflict measures whether different evidence sources point to different candidates. Examples include disagreement between recent-history preference and long-term-history preference, disagreement between retriever ranking and direct LLM ranking, or contradictory support factors for the same candidate.

Noise measures whether the user history is unreliable or hard to interpret. Possible indicators include short history length, high duplicate ratio, low-rating interactions, bursty timestamps, high category entropy, or strong recency shift.

Budget measures practical constraints such as token limit, latency limit, and maximum number of LLM calls. When budget is tight, the router should collapse graph or tree reasoning toward a cheaper linear protocol.

## 6. Topology Family Schema

The schema should be compact. We use four source types:

```yaml
source:
  behavior: history titles, ratings, timestamps
  item_text: candidate titles and available text metadata
  candidate_prior: retriever rank or score
  derived: offline user profile or item perception
```

We use two relation types:

```yaml
relation:
  support: evidence increases the plausibility of a candidate
  conflict: evidence decreases plausibility or contradicts another factor
```

A reasoning instance is defined semantically rather than as a required LLM JSON output:

```text
task: candidate reranking
input: H_u, C_u, X_C, optional S_C
signals: uncertainty, conflict, noise, budget
topology: linear | tree | graph
reasoning units: factors, support/conflict relations, aggregation
final output: ranking over C_u
```

In implementation, the LLM should usually emit a lightweight natural-language topology trace plus a minimal final answer block:

```text
Reasoning Trace:
Step / Branch / Factor sections depending on the selected topology.

Final Answer:
{"ranking": ["candidate_A", "candidate_B"], "selected_candidate_id": "candidate_A"}
```

This avoids forcing the model to generate large nested JSON reasoning traces. JSON is used for data storage and final-answer parsing, not as the mandatory format for every intermediate CoT unit.

## 7. Linear Protocol

The linear protocol is the cheapest executor. It serializes reasoning into a fixed chain:

\[
H_u \rightarrow \text{preference summary} \rightarrow \text{candidate matching} \rightarrow \text{ranking}.
\]

It is suitable for low-uncertainty and low-conflict samples. Its role in the paper is not simply to be a baseline, but to be one member of the same topology family. It should use the same input fields and produce the same final score format as tree and graph protocols.

## 8. Tree Protocol

The tree protocol decomposes user preference into multiple branches. Branches may correspond to recent interest, long-term interest, category preference, price/style preference, or candidate groups. Each branch independently scores candidates, then an aggregation step combines the branch-level scores:

\[
s(c) = \operatorname{Agg}(s_{\text{recent}}(c), s_{\text{long}}(c), s_{\text{aspect}_1}(c), \ldots).
\]

The tree protocol is suitable when a user history contains multiple interests or when candidates are semantically similar. It provides a controlled expansion over linear reasoning without requiring arbitrary graph edges.

## 9. Graph Protocol

The graph protocol constructs a compact factor-candidate graph:

\[
G_u = (V_f \cup V_c, E),
\]

where \(V_f\) contains evidence factors, \(V_c\) contains candidate items, and \(E\) contains support/conflict relations. The candidate score can be computed by weighted aggregation:

\[
s(c) = \sum_{f \in V_f} w_{f,c} \cdot \operatorname{polarity}(f,c) \cdot \operatorname{conf}(f).
\]

In an LLM implementation, the graph should usually be represented as a constrained natural-language trace with factor, support/conflict, and aggregation sections. The key is that the graph is not an unconstrained prompt saying "think as a graph"; it has explicit factor units, support/conflict relations, and an aggregation rule. A structured JSON graph can be used as an ablation or storage format, but it is not the default intermediate CoT format.

The graph protocol is reserved for high-conflict, high-noise, or high-uncertainty samples. This avoids paying graph reasoning cost on easy cases.

## 10. Training Objective

For reranking, a standard candidate softmax objective can be used:

\[
\mathcal{L}_{rec} = -\log \frac{\exp s(y_u)}{\sum_{c \in C_u} \exp s(c)}.
\]

If oracle topology labels are available, the router can be trained by running all topology modes on a subset and selecting the mode with the best validation utility:

\[
m_u^* = \arg\max_m \left[\operatorname{Metric}(g_m(x_u)) - \lambda \operatorname{Cost}(m)\right].
\]

The router loss is:

\[
\mathcal{L}_{route} = \operatorname{CE}(r_\theta(x_u), m_u^*).
\]

A regret-style objective can also be used:

\[
\mathcal{L}_{regret} = \max(0, U(m_u^*) - U(m_u)),
\]

where \(U(m)\) is accuracy-cost utility. The full objective can be:

\[
\mathcal{L} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{route} + \beta \mathcal{L}_{regret} + \gamma \operatorname{Cost}(m_u).
\]

If a verifier or item-attribute graph is used, it should appear as an auxiliary reward or evaluation signal rather than a default online input.

## 11. Dataset and Baseline Configuration

The main experiments should be aligned with R2Rec because it is the closest direct baseline under a sequential recommendation setting. R2Rec evaluates on MovieLens-1M, Amazon Book, and Amazon Electronics, using a 20-item candidate set composed of one positive next item and nineteen negative items. We keep the same domains and candidate size, but replace random negatives with retriever-based hard negatives as the main protocol.

The recommended main datasets are:

```text
MovieLens-1M
Amazon Book
Amazon Electronics
```

MovieLens-1M provides a relatively clean movie-domain benchmark with ratings and item attributes. Amazon Book provides richer item text and a natural overlap with ReRec's book domain. Amazon Electronics provides a noisier e-commerce domain where candidate ambiguity, diverse user interests, and behavior noise are more likely to expose differences among linear, tree, and graph reasoning protocols.

The main candidate construction is:

\[
C_u = \{y_u\} \cup \operatorname{Top}_{19}(R(H_u) \setminus \{y_u\}),
\]

where \(R\) is a frozen SASRec retriever by default. This yields a 20-item candidate pool and matches the scale of R2Rec and ReRec while making the negative candidates harder than uniformly random negatives. If compute permits, LightGCN can be used as a second retriever for robustness analysis, but the primary setting should use a single fixed retriever to avoid confounding reranking quality with retrieval quality.

The default history length is:

```text
L = 10 recent interactions
```

This is consistent with many LLM4Rec prompt settings and keeps prompt length manageable. A shorter setting, such as \(L=5\), can be added to directly compare with R2Rec-style prompts, but it should not replace the main setting unless cost becomes a blocker.

The main direct baselines should include:

```text
Traditional retrievers/rankers:
  SASRec
  LightGCN
  GRU4Rec or Caser, if implementation cost is acceptable

LLM reranking baselines:
  Direct answer without reasoning
  Vanilla CoT
  Fixed Linear
  Fixed Tree
  Fixed Graph
  R2Rec

Adaptive baselines:
  Random router
  Heuristic router
  Oracle router
  Learned router, if enough oracle labels are collected
```

ReRec should be handled as a supplementary comparison rather than the main sequential-recommendation baseline. Its primary benchmark, RecBench+, is query-based and contains Movie and Book domains with condition-based and profile-based queries. This differs from the main next-item reranking task. We therefore recommend using RecBench+ only after the main pipeline is stable:

```text
Supplementary / generalization datasets:
  RecBench+ Movie
  RecBench+ Book
```

In this supplementary setting, the candidate construction should follow ReRec:

```text
1 positive item + 19 random negatives
or
1 positive item + hard negatives + simple negatives
```

The purpose of the RecBench+ experiment is to test whether the topology family transfers from sequential next-item reranking to query-based candidate selection. It should not be mixed with the main R2Rec-aligned experimental protocol.

The recommended experimental schedule is:

```text
Stage 1:
  MovieLens-1M + Amazon Book
  SASRec hard candidates
  fixed topology runs
  oracle topology analysis

Stage 2:
  Add Amazon Electronics
  test robustness under noisier e-commerce behavior
  add router variants

Stage 3:
  Add RecBench+ Movie / Book
  compare with ReRec under query-based candidate selection
  frame as cross-task or generalization experiment
```

## 12. Evaluation Protocol

The main metrics should include HR@K, NDCG@K, MRR, and candidate-level accuracy. Since this work focuses on adaptive reasoning, it should also report token cost, latency, number of LLM calls, and accuracy-cost Pareto curves.

The candidate pool protocol should be explicit:

1. Main controlled setting: \(\{y_u\} + K-1\) retrieved hard negatives.
2. End-to-end setting: actual retriever top-K, with retriever Recall@K reported.
3. Ablation setting: random negatives, mainly to compare with earlier LLM4Rec protocols.

Critical ablations include fixed linear, fixed tree, fixed graph, random router, heuristic router, learned router, oracle router, without uncertainty signal, without conflict signal, without noise signal, without derived profile, and with the same token budget across topologies.

The router itself should be evaluated by mode frequency, oracle agreement, routing regret, and performance under different sample difficulty groups. This is necessary to show that adaptive topology selection is doing useful work rather than acting as an arbitrary prompt selector.

For R2Rec-aligned experiments with 20 candidates, the primary metrics should include:

```text
HitRatio@1
HitRatio@3
NDCG@3
MRR
```

For RecBench+-aligned query experiments, report Accuracy as the primary metric to match ReRec, and optionally report HR@1 because it is equivalent to selecting the correct item from the candidate set.

## 13. Recommended Claim Boundary

The safe claim is:

> We propose a minimal-input, candidate-aware LLM reranking framework that adaptively selects among constrained linear, tree, and graph reasoning protocols under uncertainty, conflict, noise, and budget signals.

The unsafe claims to avoid are:

> first graph reasoning for recommendation;
> first CoT for LLM4Rec;
> first verifiable recommendation reasoning;
> graph topology always improves recommendation.

The contribution should be framed as topology control, routing, and cost-aware reasoning under a realistic reranking protocol.
