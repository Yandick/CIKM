# Config Layout

Use configs to make experiments reproducible.

## Folders

- `data/`: dataset names, paths, sequence construction, candidate construction
- `inference/`: local-model inference runs, prompt style, split, and scoring switches
- `model/`: base LLM, adapters, semantic IDs, reasoning modules
- `train/`: SFT, RL, distillation, curriculum, regularization
- `eval/`: ranking metrics, rationale metrics, latency, robustness
- `prompt/`: rationale schemas and prompt templates

## Rule

If an experiment cannot be reconstructed from files in `configs/`, it is not a finished experiment.

Current FaithRec entries:

- `configs/data/amazon_food_pilot.yaml`: small local Amazon Food pilot for evidence-grounded reranking
- `configs/model/qwen2_5_1_5b_instruct.yaml`: default small local model
- `configs/prompt/evidence_rerank_v1.yaml`: evidence-selection reranking prompt contract
- `configs/train/faithfulness_grpo.yaml`: SFT + rank-only GRPO + faithfulness GRPO plan
- `configs/eval/faithfulness_counterfactual.yaml`: ranking and counterfactual-faithfulness evaluation

Legacy/earlier entries:

- `configs/data/amazon_food.yaml`
