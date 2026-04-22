"""Training and post-training utilities."""

from llm4rec.training.baselines import (
    PredictionScore,
    SFTRecord,
    extract_predicted_candidate_id,
    iter_baseline_prompt_records,
    prompt_record_to_sft_record,
    score_prediction_text,
    write_prompt_records_jsonl,
    write_sft_records_jsonl,
)
from llm4rec.training.rewards import (
    AdvantageConfig,
    group_relative_advantages,
    mean_reward,
)

__all__ = [
    "AdvantageConfig",
    "PredictionScore",
    "SFTRecord",
    "extract_predicted_candidate_id",
    "group_relative_advantages",
    "iter_baseline_prompt_records",
    "mean_reward",
    "prompt_record_to_sft_record",
    "score_prediction_text",
    "write_prompt_records_jsonl",
    "write_sft_records_jsonl",
]
