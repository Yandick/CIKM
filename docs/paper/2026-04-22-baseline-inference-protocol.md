# Baseline Inference Protocol

Date: `2026-04-22`

## Goal

Define the first no-training baseline protocol for this repo without drifting into method-specific extras.

## Protocol

1. Use `Amazon Food` with fixed `20-way` candidate reranking.
2. Keep the same local backbone for:
   - `answer-only`
   - `free-form CoT`
3. Keep the same candidate set and candidate order across prompt styles.
4. Use only:
   - recent history
   - candidate item profiles
   - the same prompt renderer
5. Do not inject method-specific chains, personas, teacher rationales, or extra retrieval features into the vanilla baseline.
6. Force a normalized final answer line for free-form CoT:
   `Answer: <candidate_id>`
7. Start with deterministic decoding so prompt-style comparisons are not confounded by sampling noise.

## Why This Protocol

- `R2Rec` evaluates on `1 positive + 19 negatives`, so a fixed `20-way` candidate protocol is aligned with a recent reasoning-for-recommendation setup.
- `ReRec` also reports `20`-candidate selection settings in its recommendation evaluations.
- `ThinkRec` and `ReRec` both use short history windows around `10`, which supports keeping the first baseline concise.
- Local refs do not provide a clean first comparison of `same backbone + answer-only` versus `same backbone + free-form CoT` under one shared candidate-set protocol, so this repo should establish that pair explicitly.
- `CoT-Rec` highlights ranking-stage position bias, so candidate ordering must be controlled when comparing prompt styles.
- `LatentR3` emphasizes inference cost, so latency and token counts should be captured even for no-training baselines.

## Local Model Assumption

All training and inference in this workspace assume the user has already downloaded the LLM to a local Hugging Face-style directory.

Runners should therefore:

1. consume a local model path from `configs/model/`
2. use `local_files_only=True`
3. fail fast if the path does not exist

## Immediate Use

- model config: [configs/model/base_llm.yaml](/D:/SCUT/26_spring/CIKM/configs/model/base_llm.yaml)
- answer-only run config: [configs/inference/answer_only.yaml](/D:/SCUT/26_spring/CIKM/configs/inference/answer_only.yaml)
- free-form CoT run config: [configs/inference/free_form_cot.yaml](/D:/SCUT/26_spring/CIKM/configs/inference/free_form_cot.yaml)
- runner: [scripts/run_llm_inference.py](/D:/SCUT/26_spring/CIKM/scripts/run_llm_inference.py)
