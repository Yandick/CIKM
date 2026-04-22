# TRACE-Rec V1 Data Schema Spec

Date: `2026-04-22`

## Goal

Define one canonical example format for:

- non-reasoning LLM baselines
- free-form CoT baselines
- TRACE-Rec graph generation
- counterfactual evaluation

The first target setting is:

`next-item recommendation with a fixed candidate set`

## Canonical Objects

### InteractionEvent

One user-item interaction:

- `item_id`
- `timestamp`
- optional `rating`
- optional `context`

### CandidateItem

One item in the fixed candidate set:

- `item_id`
- `label`
  `1` for the target and `0` for negatives
- `source`
  for example `target` or `popularity_negative`
- optional `rank_prior`

### NextItemExample

One training or evaluation sample:

- `example_id`
- `user_id`
- chronological `history`
- `target_item_id`
- fixed `candidates`
- `split`
- optional `context`

## JSON Shape

```json
{
  "example_id": "amazon-food-train-u42-1700000000-B00XYZ",
  "user_id": "u42",
  "history": [
    {"item_id": "B00AAA", "timestamp": 1699999998},
    {"item_id": "B00BBB", "timestamp": 1699999999}
  ],
  "target_item_id": "B00XYZ",
  "candidates": [
    {"item_id": "B00XYZ", "label": 1, "source": "target", "rank_prior": 287.0},
    {"item_id": "B00NEG", "label": 0, "source": "popularity_negative", "rank_prior": 521.0}
  ],
  "split": "train",
  "context": {
    "target_timestamp": 1700000000,
    "target_rating": 5.0,
    "raw_history_length": 12,
    "history_timestamp_mode": "synthetic_from_order",
    "available_evidence_refs": ["history:0", "history:1"],
    "available_feature_refs_by_candidate": {
      "B00XYZ": ["mean_rating:4.5+", "review_count:10-49"]
    }
  }
}
```

## Invariants

Every valid example must satisfy:

1. history is chronological
2. candidate ids are unique
3. exactly one positive candidate exists
4. the positive candidate matches `target_item_id`
5. candidate set size is fixed for a given experiment config

## V1 Candidate Constructor

The default constructor is:

`target + popularity negatives`

Rules:

1. always include the target item
2. sample `K-1` negatives from globally popular items
3. exclude seen history items by default
4. shuffle candidate order deterministically per example

This is intentionally simple. It is not the final research contribution; it is the stable input layer for the first experiments.

## Current Default Dataset

The active first dataset is `Amazon Food`, using:

- precomputed `train / validation / test` split CSV files
- one review JSONL file aggregated into weak item profiles

This is enough for the first structured-reasoning pipeline because the current target is candidate-conditioned reasoning over a fixed candidate set. A separate Amazon `meta` file becomes important later if we want to reproduce metadata-heavy baselines or ground evidence directly to product attributes.
