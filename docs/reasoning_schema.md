# TopoRec-Family Reasoning Schema

This document defines the reasoning schema for TopoRec-Family. The schema is a semantic definition of the reasoning topology, not a requirement that the LLM must output every intermediate step as JSON.

The recommended implementation is:

```text
natural-language topology trace + minimal machine-readable final answer
```

This keeps the design close to existing CoT-style recommendation work, where chain, tree, or graph structure is usually simulated by natural-language sections, while still allowing reliable evaluation through a small final answer block.

## 1. Scope

The task is candidate-aware sequential recommendation reranking. For each user instance, the model receives:

```text
H_u: user history
C_u: candidate set
X_C: candidate text
S_C: optional retriever ranks or scores
A_u: optional offline artifacts, such as a compact user profile or item descriptions
```

The model outputs a ranking over `C_u`. It must not recommend items outside the candidate set.

All topology modes share the same information boundary:

```text
direct
linear
tree
graph
```

The difference among topology modes is how evidence is organized during reasoning, not what information the model can access.

## 2. Input Record

Each training or evaluation instance can be stored as:

```json
{
  "instance_id": "u123_t17",
  "user_id": "u123",
  "task": "candidate_reranking",
  "history": [
    {
      "history_id": "H1",
      "item_id": "i01",
      "title": "item title",
      "rating": 5,
      "timestamp": 1700000000,
      "text": "optional metadata text"
    }
  ],
  "candidates": [
    {
      "candidate_id": "A",
      "item_id": "i98",
      "title": "candidate title",
      "retriever_rank": 1,
      "retriever_score": 0.81,
      "text": "optional metadata text"
    }
  ],
  "optional_artifacts": {
    "user_profile": null,
    "item_descriptions": {}
  },
  "label": {
    "target_candidate_id": "C",
    "target_item_id": "i77"
  }
}
```

The target label is used only by the evaluator and training data builder. It must not appear in the inference prompt.

## 3. Two-Level Output Design

TopoRec-Family uses two output levels.

### 3.1 LLM-Facing Output

The LLM-facing output should use natural-language trace sections that reflect the selected topology.

Example:

```text
Reasoning Trace:
Step 1 - Preference: ...
Step 2 - Candidate Match: ...
Step 3 - Final Ranking: ...

Final Answer:
{"ranking": ["C", "A", "B", "D"], "selected_candidate_id": "C"}
```

The trace can be natural language because it is mainly used for reasoning, training, and qualitative analysis. The final answer is minimal JSON because it is used for deterministic parsing and evaluation.

### 3.2 Storage-Level Output

After parsing, each run can be stored as a structured JSONL record:

```json
{
  "instance_id": "u123_t17",
  "topology": "linear",
  "trace_text": "raw natural-language reasoning trace",
  "ranking": ["C", "A", "B", "D"],
  "selected_candidate_id": "C",
  "scores": null,
  "parse_success": true,
  "latency_ms": 1200,
  "prompt_tokens": 900,
  "completion_tokens": 180
}
```

This storage schema is for logging and analysis. It should not be confused with the required LLM intermediate output.

## 4. Final Answer Contract

Every topology should end with the same minimal answer block:

```json
{
  "ranking": ["C", "A", "B", "D"],
  "selected_candidate_id": "C"
}
```

Optional:

```json
{
  "ranking": ["C", "A", "B", "D"],
  "selected_candidate_id": "C",
  "scores": {
    "A": 0.21,
    "B": 0.13,
    "C": 0.58,
    "D": 0.08
  }
}
```

Validity constraints:

```text
ranking must contain every candidate exactly once
selected_candidate_id must equal ranking[0]
all candidate ids must come from C_u
scores are optional
if scores are present, they should include every candidate and be finite
```

Evaluation should use `ranking`. Absolute score calibration is not required.

## 5. Semantic Units

The topology family is defined by three semantic units:

```text
factor: a compact preference or evidence statement
relation: support or conflict between evidence and candidates
aggregation: the final combination of evidence into a ranking
```

These units can be expressed in natural language. They do not need to be emitted as JSON fields.

Allowed factor sources:

```text
behavior: user history titles, ratings, timestamps
item_text: candidate or history item text
candidate_prior: retriever rank or score
derived: optional profile or summary derived from provided data
```

Allowed relation types:

```text
support: evidence increases candidate plausibility
conflict: evidence decreases candidate plausibility or conflicts with another factor
```

This compact design avoids a large hand-crafted heterogeneous graph. Genre, brand, author, director, category, and rating details should be written inside factor text rather than turned into many relation types.

## 6. Direct Mode

Direct mode is the answer-only baseline. It does not expose explicit CoT.

LLM-facing format:

```text
Final Answer:
{"ranking": ["C", "A", "B", "D"], "selected_candidate_id": "C"}
```

Use direct mode to measure whether explicit reasoning is necessary.

## 7. Linear Mode

Linear mode uses one ordered reasoning chain:

```text
preference summary -> candidate matching -> final ranking
```

LLM-facing format:

```text
Reasoning Trace:
Step 1 - Preference Summary:
The recent history suggests ...

Step 2 - Candidate Matching:
A matches ...; B is weaker because ...; C best matches ...

Step 3 - Ranking Decision:
The final ranking is C > A > B > D.

Final Answer:
{"ranking": ["C", "A", "B", "D"], "selected_candidate_id": "C"}
```

Linear mode is expected to be the cheapest explicit topology. It is preferred for low-uncertainty and low-conflict instances.

## 8. Tree Mode

Tree mode decomposes the recommendation decision into several parallel branches and then aggregates them.

Recommended branches:

```text
recent interest
long-term interest
candidate text match
retriever prior
negative evidence
```

LLM-facing format:

```text
Reasoning Trace:
Branch 1 - Recent Interest:
...

Branch 2 - Long-Term Interest:
...

Branch 3 - Candidate Text Match:
...

Aggregation:
Recent interest favors C, long-term interest also supports C, while A has weaker support.

Final Answer:
{"ranking": ["C", "A", "D", "B"], "selected_candidate_id": "C"}
```

Tree mode is useful when the user history has multiple interests or candidates should be compared from several independent aspects.

## 9. Graph Mode

Graph mode uses a compact factor-candidate reasoning graph, but the graph can be represented in natural language rather than JSON.

Recommended natural-language graph trace:

```text
Reasoning Trace:
Preference Factors:
F1: Recent items suggest ...
F2: Long-term history suggests ...
F3: Retriever prior favors ...

Support / Conflict Relations:
F1 supports C and A.
F2 supports C.
F3 supports A but conflicts with B.

Aggregation:
C receives support from both recent and long-term factors; A is plausible but less consistent.

Final Answer:
{"ranking": ["C", "A", "D", "B"], "selected_candidate_id": "C"}
```

Graph mode should stay compact:

```text
3 to 6 factors for K = 20
only support and conflict relations
no full item-attribute graph as default prompt input
no separate relation type for each attribute
```

Graph mode is reserved for high-conflict, high-noise, or high-uncertainty cases due to its higher cost.

## 10. Parser and Validation Rules

The parser only needs to extract the final answer block.

Required checks:

```text
final answer block is valid JSON
ranking contains known candidate ids only
ranking contains every candidate exactly once
selected_candidate_id equals ranking[0]
no candidate hallucination
```

Optional checks:

```text
trace contains the expected section names for the selected topology
linear trace has ordered steps
tree trace has branches and aggregation
graph trace has factors, support/conflict relations, and aggregation
scores are finite if present
```

For final experiments, invalid outputs should be reported rather than manually corrected:

```text
parse success rate
format error rate
candidate hallucination rate
missing candidate rate
```

One repair prompt can be used during development, but main evaluation should report performance before repair.

## 11. Training Implication

SFT should not force the model to generate large nested JSON traces. A better target format is:

```text
topology-specific natural-language trace
minimal final answer JSON
```

This is easier to generate, closer to related work, cheaper at inference time, and still supports deterministic evaluation.
