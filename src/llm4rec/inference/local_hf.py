"""Local Hugging Face causal-LM inference for baseline recommendation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import yaml

from llm4rec.evaluation.offline import OfflinePredictionRecord
from llm4rec.inference.parsing import (
    extract_candidate_id_from_text,
    normalize_ranked_item_ids,
)
from llm4rec.prompts import PromptRecord


@dataclass(frozen=True)
class HFLocalGenerationConfig:
    """Generation controls for local Hugging Face baseline inference."""

    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class HFLocalModelConfig:
    """Config for a locally downloaded Hugging Face causal LM."""

    name: str
    backend: str
    model_path: str
    local_files_only: bool = True
    trust_remote_code: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"
    use_chat_template: bool = True
    max_input_tokens: int = 4096
    batch_size: int = 1
    generation: HFLocalGenerationConfig = field(default_factory=HFLocalGenerationConfig)


def load_local_hf_model_config(config_path: str | Path) -> HFLocalModelConfig:
    """Load one local-HF model config from YAML."""

    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    generation = data.get("generation", {})
    runtime = data.get("runtime", {})
    return HFLocalModelConfig(
        name=str(data.get("name", "")),
        backend=str(data.get("backend", "huggingface_local_causal_lm")),
        model_path=str(data.get("model_path", "")),
        local_files_only=bool(data.get("local_files_only", True)),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
        device_map=str(data.get("device_map", "auto")),
        torch_dtype=str(data.get("torch_dtype", "auto")),
        use_chat_template=bool(runtime.get("use_chat_template", True)),
        max_input_tokens=int(runtime.get("max_input_tokens", 4096)),
        batch_size=int(runtime.get("batch_size", 1)),
        generation=HFLocalGenerationConfig(
            max_new_tokens=int(generation.get("max_new_tokens", 128)),
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 0.0)),
            top_p=float(generation.get("top_p", 1.0)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
        ),
    )


def load_local_hf_causal_lm(model_config: HFLocalModelConfig) -> tuple[Any, Any]:
    """Load a local causal LM and tokenizer from disk only."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "local HF inference requires the optional llm dependencies "
            "(`transformers`, `torch`)."
        ) from exc

    model_path = Path(model_config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path}")

    common_kwargs = {
        "local_files_only": model_config.local_files_only,
        "trust_remote_code": model_config.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path, **common_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "device_map": model_config.device_map,
    }
    if model_config.torch_dtype != "auto":
        dtype = getattr(torch, model_config.torch_dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported torch_dtype: {model_config.torch_dtype}")
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        **common_kwargs,
        **model_kwargs,
    )
    model.eval()
    return model, tokenizer


def render_prompt_text(
    prompt_record: PromptRecord,
    *,
    tokenizer: Any | None = None,
    use_chat_template: bool = True,
) -> str:
    """Convert a chat-style prompt record into one text string for generation."""

    if (
        tokenizer is not None
        and use_chat_template
        and hasattr(tokenizer, "apply_chat_template")
    ):
        return str(
            tokenizer.apply_chat_template(
                list(prompt_record.prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    lines: list[str] = []
    for message in prompt_record.prompt:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}:\n{content}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def prediction_record_from_response(
    prompt_record: PromptRecord,
    response_text: str,
    *,
    model_config: HFLocalModelConfig,
    run_id: str,
    prompt_version: str,
    latency_ms: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    finish_reason: str | None = "stop",
    error: str | None = None,
) -> OfflinePredictionRecord:
    """Normalize one raw model response into the shared prediction contract."""

    selected_item_id = extract_candidate_id_from_text(
        response_text,
        prompt_record.candidate_ids,
    )
    return OfflinePredictionRecord(
        example_id=prompt_record.example_id,
        selected_item_id=selected_item_id,
        response_text=response_text,
        ranked_item_ids=normalize_ranked_item_ids((), selected_item_id=selected_item_id),
        run_id=run_id,
        model_name=model_config.name,
        prompt_style=prompt_record.style.value,
        prompt_version=prompt_version,
        generation_config={
            "max_new_tokens": model_config.generation.max_new_tokens,
            "do_sample": model_config.generation.do_sample,
            "temperature": model_config.generation.temperature,
            "top_p": model_config.generation.top_p,
            "repetition_penalty": model_config.generation.repetition_penalty,
        },
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        group_id=run_id,
        sample_index=0,
        error=error,
        metadata={
            "user_id": prompt_record.metadata.get("user_id"),
            "history_length": prompt_record.metadata.get("history_length"),
            "target_timestamp": prompt_record.metadata.get("target_timestamp"),
        },
    )


def generate_prediction_records_with_local_hf(
    prompt_records: Iterable[PromptRecord],
    *,
    model: Any,
    tokenizer: Any,
    model_config: HFLocalModelConfig,
    run_id: str,
    prompt_version: str,
) -> list[OfflinePredictionRecord]:
    """Run local-HF generation over prompt records and return prediction rows."""

    prediction_records: list[OfflinePredictionRecord] = []
    for prompt_record in prompt_records:
        prompt_text = render_prompt_text(
            prompt_record,
            tokenizer=tokenizer,
            use_chat_template=model_config.use_chat_template,
        )
        try:
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=model_config.max_input_tokens,
            )
            prompt_tokens = _prompt_token_count(inputs)
            inputs = _move_inputs_to_model_device(inputs, model)
            generate_kwargs = _generation_kwargs(model_config, tokenizer)

            start_time = perf_counter()
            output_ids = model.generate(**inputs, **generate_kwargs)
            latency_ms = (perf_counter() - start_time) * 1000.0

            generated_ids = output_ids[0][prompt_tokens:]
            completion_tokens = int(len(generated_ids))
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            prediction_records.append(
                prediction_record_from_response(
                    prompt_record,
                    response_text,
                    model_config=model_config,
                    run_id=run_id,
                    prompt_version=prompt_version,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        except Exception as exc:
            prediction_records.append(
                prediction_record_from_response(
                    prompt_record,
                    "",
                    model_config=model_config,
                    run_id=run_id,
                    prompt_version=prompt_version,
                    finish_reason="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return prediction_records


def _prompt_token_count(inputs: Any) -> int:
    """Compute the effective prompt token count from tokenizer outputs."""

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        mask_row = attention_mask[0]
        if hasattr(mask_row, "sum"):
            try:
                return int(mask_row.sum().item())
            except Exception:
                return int(mask_row.sum())
    input_ids = inputs["input_ids"]
    return int(input_ids.shape[-1])


def _move_inputs_to_model_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    """Move tensor-like inputs onto the model device when possible."""

    device = getattr(model, "device", None)
    if device is None:
        return inputs

    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _generation_kwargs(
    model_config: HFLocalModelConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    """Build one normalized set of generate kwargs."""

    kwargs: dict[str, Any] = {
        "max_new_tokens": model_config.generation.max_new_tokens,
        "do_sample": model_config.generation.do_sample,
        "repetition_penalty": model_config.generation.repetition_penalty,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if model_config.generation.do_sample:
        kwargs["temperature"] = model_config.generation.temperature
        kwargs["top_p"] = model_config.generation.top_p
    return kwargs
