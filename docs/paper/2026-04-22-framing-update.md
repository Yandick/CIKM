# 2026-04-22 Framing Update

## Status

This note supersedes the earlier `teacher-trace-first` framing wherever they conflict.

The current baseline audit was rechecked against local references under `ref/`, including:

- `CoT-Rec`
- `ThinkRec`
- `R2Rec`
- `ReRec`
- `LatentR3`
- `SIREN`

## What The Audit Changed

The project should **not** be framed as:

- the first recommendation-specific structured CoT
- the first graph reasoning for recommendation
- the first answer-only reasoning recommender
- a teacher-trace generation paper

Those claims are too weak or too risky after comparing with recent work.

## Corrected Paper Task

The safer task is:

`candidate-set generative recommendation with answer-only inference, trained or post-trained with a structure-aware verifier/reward model, and evaluated with counterfactual faithfulness`

Concretely:

- The online policy outputs only the final candidate id.
- A recommendation-specific structured object is still defined, but it is primarily used for:
  - verifier checks
  - reward shaping
  - offline auditing
  - counterfactual evaluation
- The paper should focus on fast deployable recommendation, not on forcing the model to emit long explicit rationales at test time.

## Why This Direction Is Safer

### `CoT-Rec`

`CoT-Rec` already uses explicit personalized information extraction and utilization for recommendation, but does not study faithfulness under interventions.

### `ThinkRec`

`ThinkRec` already uses synthetic reasoning traces distilled from a strong reasoning model and evaluates generated reasons with `METEOR` and `BLEURT`. This makes a pure "better structured rationale generation" story less safe.

### `R2Rec`

`R2Rec` is especially important. It already does:

- interaction-of-thought reasoning
- annotated reasoning traces
- `SFT + RL`
- explicit reasoning outputs

This means a paper centered on "structured reasoning traces plus RL" will overlap directly with it.

### `ReRec`

`ReRec` already establishes that recommendation reasoning can be improved through reinforcement fine-tuning and reward shaping. So "we also use reasoning reward" is not a differentiator by itself.

### `LatentR3` and `SIREN`

These works are the main warning against over-claiming `answer-only` novelty:

- both push toward efficient inference
- both reduce or remove explicit CoT cost at serving time

So "answer-only inference" is useful, but not novel on its own.

## What We Should Claim Instead

The paper should claim the combination below, not any one component in isolation:

1. `Answer-only policy`
2. `Recommendation-native structure-aware verifier`
3. `Counterfactual-faithfulness reward / evaluation`
4. `Fast inference with optional audit mode`

The key differentiator is:

`we do not require the policy to emit explicit CoT, but we still constrain and evaluate reasoning through a structured verifier space`

## Corrected Contributions

### 1. New problem formulation

We study recommendation reasoning under a stricter deployment constraint:

- online inference must stay answer-only or near answer-only
- reasoning should still be structured enough to be checked
- evaluation should measure whether decisions update correctly under interventions

### 2. Structure-aware verifier space

We define a minimal recommendation-specific structured object that captures:

- user-side evidence
- candidate-conditioned support/conflict signals
- final decision consistency

This object is not the main user-facing output. It is the object the verifier/reward model uses.

### 3. Counterfactual-faithfulness objective

We explicitly evaluate and potentially reward whether:

- removing supporting history weakens the right support path
- changing candidate attributes updates candidate support correctly
- irrelevant perturbations keep predictions stable

This is stronger than explanation quality metrics such as `METEOR` or `BLEURT`.

### 4. Efficient inference story

The policy remains fast because:

- output length is constant-sized
- no verbose rationale is required online
- structured reasoning is moved into training, verification, or offline audit

## Experimental Protocol Decisions

To reduce mismatch with recent baselines:

- default to a fixed candidate-set reranking protocol
- use `K = 20` candidates by default
- treat `Amazon Food` as the first executable benchmark
- do not compare directly against all-ranking and binary-prediction numbers without protocol notes

## Immediate Next Steps

1. Keep the current graph schema, but reinterpret it as a verifier object rather than a mandatory policy output.
2. Build an answer-only policy baseline first.
3. Implement structure-aware verifier scores and reward contracts.
4. Keep counterfactual faithfulness as the main evaluation differentiator.
5. Treat explicit rationale generation as optional audit mode, not the default method.
