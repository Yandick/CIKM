# AGENT.md

## Mission

Build a research-grade workspace for studying recommendation-specific reasoning in LLM-based generative recommendation.

The near-term goal is not "do anything with CoT". The goal is to identify a paper framing that is still novel after accounting for 2025-2026 work.

## Project Status

As of `2026-04-20`, the workspace started empty and was scaffolded into a research template.

Current novelty assessment:

- Do not claim "the first structured CoT for recommendation".
- Do not claim "the first graph/tree reasoning for recommendation".
- Those claims are too likely to be false because of recent work such as `GOT4Rec`, `SCoTER`, `OneRec-Think`, `GREAM`, `Reasoning to Rank`, `ReRec`, `SIREN`, and `LatentR3`.

## Recommended Research Framing

Prefer one of these framings:

1. Answer-only policy with structure-aware verifier:
   The policy predicts only the final item, while a recommendation-specific structured object is used for checking and reward.
2. Counterfactual-faithfulness for recommendation:
   Reasoning quality is judged by whether the decision updates correctly under controlled interventions.
3. Candidate-conditioned verifier grammar:
   Keep a compact typed graph or decision object, but do not require it as the default online output.
4. Efficient reasoning without verbose rationale:
   The method should preserve fast inference and avoid large explicit CoT cost by default.

Avoid staying at the level of "we use tree-of-thoughts for recommendation". That is too close to existing work.

## What Agents Must Do

1. Verify latest literature before making novelty claims, especially papers from 2025 and 2026.
2. Use absolute dates in notes and summaries.
3. Write literature findings to `docs/literature/`.
4. Write idea and experiment notes to `docs/notes/`.
5. Keep configs under `configs/` instead of hiding settings in scripts.
6. Save experimental artifacts under `outputs/runs/` with dated folders.
7. Keep raw datasets immutable under `data/raw/`.
8. Treat `Amazon Food` as the default first executable dataset unless the user explicitly changes the benchmark.
9. Assume training and inference use a locally downloaded LLM directory; do not rely on on-demand checkpoint downloads.

## Agent Mode

Use agent mode for substantial multi-step work that benefits from parallel critique or sidecar execution.

Typical triggers:

1. literature sweeps across multiple recent papers
2. reward or verifier design where failure modes must be argued explicitly
3. data plumbing, JSONL contracts, and experiment runner design
4. regression-test expansion and fixture generation
5. experiment review, ablation comparison, and failure-case collection

Do not use agent mode for:

1. trivial single-file edits
2. cosmetic wording changes
3. pure brainstorming with no concrete deliverable

## Agent Execution Protocol

When agent mode is used, follow this operating sequence:

1. state the exact blocking question before delegating
2. keep agent roles disjoint and concrete
3. prefer `2-3` parallel agents, not a swarm
4. require each agent to return corrections, failure modes, or decisions
5. resolve disagreements by choosing the narrower claim and smaller executable change
6. integrate results locally and run verification before reporting completion

Default role split for this workspace when available:

1. `Feynman`:
   framing skeptic, reward exploit hunter, novelty critic
2. `Socrates`:
   interface reviewer, schema critic, workflow-contract reviewer
3. `Hubble`:
   experiment-order reviewer, evaluation critic, training-path skeptic

These names are conventions, not hard dependencies. If different subagents are available, assign the same responsibilities by role.

## How Agents Should Assist

Agents are not there to repeat the same work. They should help in distinct roles.

Preferred roles:

1. literature auditor:
   verify novelty and extract exact task/training/evaluation assumptions from recent work
2. skeptic:
   attack the current reward, verifier, or evaluation design and surface exploits
3. interface reviewer:
   check JSONL contracts, script boundaries, and config assumptions
4. experiment reviewer:
   sanity-check whether metrics, baselines, and protocols are actually comparable

The main agent should use them to create argument pressure, not just parallel summaries.

Each agent response should contain:

1. one or more concrete corrections
2. at least one expected exploit or failure mode
3. a clear statement of what should happen next

Agents should not stop at "looks reasonable".

## Reward / Verifier Iteration

When agents are used to improve reward or verifier design, they must follow these rules:

1. start from a concrete failure case, not an abstract preference
2. state exactly what behavior a reward term is supposed to encourage or suppress
3. change one important term at a time before proposing combined weighting changes
4. report the effect on:
   `ranking utility`, `schema validity`, and `counterfactual faithfulness`
5. treat schema validity as the only hard gate unless there is strong evidence for another hard constraint
6. remove soft terms that do not hold up on held-out examples

Agents should argue from:

1. local examples
2. expected exploits
3. measured deltas

Avoid abstract debate detached from concrete cases.

## Definition Of Done

Agent-assisted work is not done until the main agent has:

1. applied the accepted corrections locally
2. updated the relevant repo artifact:
   `AGENT.md`, `README.md`, `docs/paper/`, `configs/`, or code under `src/`
3. run at least one concrete verification step for code changes:
   tests, a script `--help`, or a small end-to-end dry run
4. reported remaining risks or known gaps explicitly

For literature-driven changes, also update the relevant note under `docs/literature/` or `docs/paper/` if the framing changed.

## Prediction Contract

For offline scoring and reranking, prediction files should stay minimal and join by `example_id`.

Required fields:

- `example_id`
- `selected_item_id`

Recommended fields:

- `group_id`
- `sample_index`
- `ranked_item_ids`
- `audit_graph`
- `counterfactual_audits`
- `metadata`

`response_text` is optional and only exists as a parsing fallback for baselines that do not emit `selected_item_id` directly. If both `selected_item_id` and `response_text` are present, they must agree after parsing.

`ranked_item_ids` should mean one ordered list over the presented candidate set, not an ambiguous top-k fragment.

If multiple predictions are stored for the same `example_id`, then `sample_index` becomes required.

Do not duplicate full history or candidate-set payloads into every prediction row unless a standalone scorer explicitly requires it.

## Audit Contract

Structured audit artifacts are optional for the answer-only policy, but when present they must be machine-checkable.

1. `audit_graph` should be serialized JSON matching the current graph schema version
2. invalid audit artifacts should never be silently accepted
3. counterfactual audits should stay tied to one original `example_id`
4. each intervention record should make the targeted factor and expected direction recoverable

If the audit object is too loose to validate, it should be dropped instead of treated as soft evidence.

## What Agents Must Not Do

1. Do not describe the idea as novel without checking recent papers.
2. Do not collapse the project into only prompt engineering unless the experiment explicitly studies prompting.
3. Do not add large binary files, datasets, or checkpoints to Git.
4. Do not mix exploratory notebook logic into `src/`.

## Baselines To Track

At minimum, keep the following in scope when comparing a new method:

- `CoT-Rec`
- `ThinkRec`
- `GOT4Rec`
- `RecLLM-R1`
- `OneRec-Think`
- `GREAM`
- `SCoTER`
- `Reasoning to Rank`
- `SIREN`
- `LatentR3`
- `ReRec`
- `GR2`

If a baseline is unavailable, write down why and choose the nearest executable alternative.

## Evaluation Checklist

Every serious experiment should try to report:

- Recommendation quality: `HR@K`, `NDCG@K`, `MRR`, or task-appropriate ranking metrics
- Efficiency: prompt length, generated reasoning tokens, latency, memory
- Structure quality: parse success, node coverage, branch usage, contradiction rate
- Faithfulness: whether rationale components actually affect the prediction
- Robustness: sensitivity to noisy history, long history, candidate set shifts

## Default Paper Direction

The current recommended default is:

`Fixed 20-way history-based candidate reranking on Amazon Food with an answer-only serving constraint, a compact candidate-conditioned verifier contract, and counterfactual diagnostic evaluation`

That direction is safer than generic tree/graph CoT because it can be differentiated by:

- answer-only online serving as a deployment constraint, not as a novelty claim
- compact candidate-conditioned verification instead of mandatory verbose rationale generation
- explicit evidence binding when audit artifacts are present
- rewardable structure constraints without making the graph the default output
- counterfactual diagnostics for targeted interventions
- optional audit mode instead of mandatory rationale generation

## File Map

- Literature landscape: [docs/literature/2026-04-20-llm4rec-structured-cot-landscape.md](docs/literature/2026-04-20-llm4rec-structured-cot-landscape.md)
- Current idea design: [docs/notes/2026-04-20-idea-design.md](docs/notes/2026-04-20-idea-design.md)
- Paper blueprint: [docs/paper/2026-04-20-paper-blueprint.md](docs/paper/2026-04-20-paper-blueprint.md)
- Framing update: [docs/paper/2026-04-22-framing-update.md](docs/paper/2026-04-22-framing-update.md)
- Graph schema spec: [docs/paper/graph-schema-spec.md](docs/paper/graph-schema-spec.md)
- Counterfactual eval spec: [docs/paper/counterfactual-eval-spec.md](docs/paper/counterfactual-eval-spec.md)
- Reward/verifier spec: [docs/paper/reward-verifier-spec.md](docs/paper/reward-verifier-spec.md)
- Data schema spec: [docs/paper/data-schema-spec.md](docs/paper/data-schema-spec.md)
- Baseline inference protocol: [docs/paper/2026-04-22-baseline-inference-protocol.md](docs/paper/2026-04-22-baseline-inference-protocol.md)
- Execution roadmap: [docs/paper/2026-04-22-execution-roadmap.md](docs/paper/2026-04-22-execution-roadmap.md)
- Week 1 checklist: [docs/paper/2026-04-22-week1-checklist.md](docs/paper/2026-04-22-week1-checklist.md)
- Config root: [configs/README.md](configs/README.md)

## Immediate TODO

1. Finalize a recommendation-specific verifier schema.
2. Keep the shared preprocessing pipeline centered on `Amazon Food` first, then expand to additional datasets.
3. Standardize a `20-way` candidate reranking protocol as the default first setting.
4. Implement an answer-only policy baseline and a structure-aware verifier baseline.
5. Add counterfactual-faithfulness and efficiency evaluation, not only accuracy.
6. Treat explicit rationale generation as optional audit mode, not the default path.
