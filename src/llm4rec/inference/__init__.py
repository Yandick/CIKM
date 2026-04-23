"""Inference helpers with lazy exports to avoid cross-package import cycles."""

from llm4rec.inference.contracts import InferenceRunConfig, load_inference_run_config
from llm4rec.inference.parsing import (
    extract_candidate_id_from_text,
    normalize_ranked_item_ids,
    validate_prediction_consistency,
)

__all__ = [
    "HFLocalGenerationConfig",
    "HFLocalModelConfig",
    "InferenceRunConfig",
    "extract_candidate_id_from_text",
    "generate_prediction_records_with_local_hf",
    "load_inference_run_config",
    "load_local_hf_causal_lm",
    "load_local_hf_model_config",
    "normalize_ranked_item_ids",
    "prediction_record_from_response",
    "render_prompt_text",
    "validate_prediction_consistency",
]


def __getattr__(name: str):
    """Load local-HF helpers on demand to avoid import cycles."""

    if name in {
        "HFLocalGenerationConfig",
        "HFLocalModelConfig",
        "generate_prediction_records_with_local_hf",
        "load_local_hf_causal_lm",
        "load_local_hf_model_config",
        "prediction_record_from_response",
        "render_prompt_text",
    }:
        from importlib import import_module

        _local_hf = import_module("llm4rec.inference.local_hf")

        return getattr(_local_hf, name)
    raise AttributeError(name)
