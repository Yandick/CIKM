# Structured Reasoning for Generative Recommendation

This repository is a research workspace for a CIKM-style paper on recommendation-specific reasoning for LLM-based generative recommendation.

Last literature sweep: `2026-04-20`.

## Current Position

The initial idea was to introduce a special chain-of-thought (CoT) format for recommendation, especially tree-like or graph-like reasoning.

The current assessment is:

- A generic "graph/tree CoT for recommendation" claim is not safe.
- The closest overlap is already strong with `GOT4Rec`, `SCoTER`, and newer 2025-2026 reasoning-for-recommendation papers.
- A safer direction is to treat answer-only inference as a serving constraint, use a compact candidate-conditioned verifier contract, and study counterfactual diagnostics rather than generic rationale quality.

See:

- [docs/literature/2026-04-20-llm4rec-structured-cot-landscape.md](docs/literature/2026-04-20-llm4rec-structured-cot-landscape.md)
- [docs/notes/2026-04-20-idea-design.md](docs/notes/2026-04-20-idea-design.md)
- [docs/paper/2026-04-20-paper-blueprint.md](docs/paper/2026-04-20-paper-blueprint.md)
- [docs/paper/2026-04-22-framing-update.md](docs/paper/2026-04-22-framing-update.md)
- [docs/paper/graph-schema-spec.md](docs/paper/graph-schema-spec.md)
- [docs/paper/counterfactual-eval-spec.md](docs/paper/counterfactual-eval-spec.md)
- [docs/paper/reward-verifier-spec.md](docs/paper/reward-verifier-spec.md)
- [docs/paper/data-schema-spec.md](docs/paper/data-schema-spec.md)
- [docs/paper/2026-04-22-execution-roadmap.md](docs/paper/2026-04-22-execution-roadmap.md)
- [docs/paper/2026-04-22-week1-checklist.md](docs/paper/2026-04-22-week1-checklist.md)
- [AGENT.md](AGENT.md)

## Repository Layout

```text
configs/         Experiment configs for data, model, train, eval, and prompt schemas
data/            Local datasets and preprocessing outputs (not for Git-tracked large files)
docs/            Literature notes, idea notes, and project planning
notebooks/       Exploratory analysis
outputs/         Reports and experiment runs
scripts/         One-off scripts and launch helpers
src/llm4rec/     Core package
tests/           Unit and regression tests
```

## Suggested Workflow

1. Update the literature note before making novelty claims.
2. Turn each new hypothesis into a short note under `docs/notes/`.
3. Encode each experiment in `configs/` before implementation.
4. Save outputs under `outputs/runs/<date>-<tag>/`.
5. Record negative results. They are often where the paper idea becomes clear.

## Immediate Priorities

1. Finalize the problem framing away from generic graph/tree CoT claims.
2. Use `Amazon Food` as the default first benchmark and keep the data path reproducible.
3. Standardize the default protocol around `20-way` candidate reranking.
4. Build an answer-only policy plus structure-aware verifier/reward path.
5. Establish offline evaluation for accuracy, efficiency, and counterfactual intervention diagnostics.

## Default Dataset

The current default dataset pipeline targets `Amazon Food` under [data/raw/amazon-food](/D:/SCUT/26_spring/CIKM/data/raw/amazon-food).

The active config is [configs/data/amazon_food.yaml](/D:/SCUT/26_spring/CIKM/configs/data/amazon_food.yaml), and the processed export script is [scripts/process_amazon_food.py](/D:/SCUT/26_spring/CIKM/scripts/process_amazon_food.py).

## Baseline Scaffold

The current minimal baseline path is:

`NextItemExample -> prompt record -> weak SFT record`

Key entry points:

- prompt rendering: [src/llm4rec/prompts/baselines.py](/D:/SCUT/26_spring/CIKM/src/llm4rec/prompts/baselines.py)
- baseline data adapter: [src/llm4rec/training/baselines.py](/D:/SCUT/26_spring/CIKM/src/llm4rec/training/baselines.py)
- export script: [scripts/build_baseline_prompts.py](/D:/SCUT/26_spring/CIKM/scripts/build_baseline_prompts.py)
- deterministic prediction export: [scripts/build_baseline_predictions.py](/D:/SCUT/26_spring/CIKM/scripts/build_baseline_predictions.py)
- offline verifier scoring: [scripts/run_offline_verifier.py](/D:/SCUT/26_spring/CIKM/scripts/run_offline_verifier.py)

## Environment

The project is scaffolded with `pyproject.toml`.

Typical setup:

```bash
pip install -e ".[dev,research,llm]"
```

## Git

This workspace has been initialized as a Git repository, but no commit has been created yet.
