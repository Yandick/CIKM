from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class GenerationResult:
    output_text: str
    latency_ms: int
    input_tokens: int
    output_tokens: int


def read_model_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_placeholder_model_path(model_path: str | None) -> bool:
    return not model_path or model_path.startswith("CHANGE_ME")


def build_prompt_text(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_hf_model(config: dict[str, Any]) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing local inference dependencies. Install the `llm` optional dependencies "
            "from pyproject.toml, including torch and transformers."
        ) from exc

    model_path = config.get("model_path")
    if is_placeholder_model_path(model_path):
        raise ValueError(
            "Model path is not configured. Set `model_path` in the model YAML or use --dry-run."
        )

    local_files_only = bool(config.get("local_files_only", True))
    trust_remote_code = bool(config.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = config.get("torch_dtype", "auto")
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if dtype:
        model_kwargs["torch_dtype"] = dtype

    device_map = config.get("device_map")
    if device_map:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not device_map:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    model.eval()
    return tokenizer, model


def _inputs_to_model_device(inputs: Any, model: Any) -> Any:
    return inputs.to(model.device)


def generate_batch(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    *,
    use_chat_template: bool,
    max_input_tokens: int,
    generation_config: dict[str, Any],
) -> list[GenerationResult]:
    import torch

    rendered_prompts = [
        build_prompt_text(tokenizer, prompt, use_chat_template=use_chat_template)
        for prompt in prompts
    ]

    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    input_width = int(inputs["input_ids"].shape[1])
    inputs = _inputs_to_model_device(inputs, model)

    gen_kwargs = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 256)),
        "do_sample": bool(generation_config.get("do_sample", False)),
        "repetition_penalty": float(generation_config.get("repetition_penalty", 1.0)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = float(generation_config.get("temperature", 0.7))
        gen_kwargs["top_p"] = float(generation_config.get("top_p", 0.95))

    start = time.perf_counter()
    with torch.no_grad():
        sequences = model.generate(**inputs, **gen_kwargs)
    latency_ms = int((time.perf_counter() - start) * 1000)

    results: list[GenerationResult] = []
    for row_idx, sequence in enumerate(sequences):
        prompt_len = int(input_lengths[row_idx])
        new_tokens = sequence[input_width:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(
            GenerationResult(
                output_text=output_text,
                latency_ms=latency_ms,
                input_tokens=prompt_len,
                output_tokens=int(len(new_tokens)),
            )
        )
    return results


def batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
