# Scripts

Keep scripts thin.

Preferred pattern:

1. Put reusable logic in `src/llm4rec/`.
2. Use scripts only as launchers, data converters, or one-off utilities.
3. Make every script consume config files rather than hidden constants.

Current launchers:

- `process_amazon_food.py`: build canonical item/example JSONL artifacts
- `build_baseline_prompts.py`: export baseline prompt records and weak-SFT records
- `build_baseline_predictions.py`: export deterministic prediction JSONL for verifier dry runs
- `run_llm_inference.py`: run one local Hugging Face model over baseline prompts and export prediction JSONL
- `run_offline_verifier.py`: score answer-only predictions and select best-by-reward outputs
