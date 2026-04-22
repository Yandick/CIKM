# Config Layout

Use configs to make experiments reproducible.

## Folders

- `data/`: dataset names, paths, sequence construction, candidate construction
- `model/`: base LLM, adapters, semantic IDs, reasoning modules
- `train/`: SFT, RL, distillation, curriculum, regularization
- `eval/`: ranking metrics, rationale metrics, latency, robustness
- `prompt/`: rationale schemas and prompt templates

## Rule

If an experiment cannot be reconstructed from files in `configs/`, it is not a finished experiment.

Current default data entry:

- `configs/data/amazon_food.yaml`
