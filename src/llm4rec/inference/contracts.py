"""Typed configs for offline baseline inference runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from llm4rec.prompts import BaselinePromptStyle


@dataclass(frozen=True)
class InferenceRunConfig:
    """Config for one baseline inference run."""

    name: str
    prompt_style: BaselinePromptStyle
    prompt_version: str
    prompt_config_path: str | None = None
    split: str = "validation"
    max_examples: int = 100
    score_with_verifier: bool = True


def load_inference_run_config(config_path: str | Path) -> InferenceRunConfig:
    """Load one inference config YAML."""

    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return InferenceRunConfig(
        name=str(data.get("name", "")),
        prompt_style=BaselinePromptStyle(str(data.get("prompt_style", "answer_only"))),
        prompt_version=str(data.get("prompt_version", "builtin_v1")),
        prompt_config_path=(
            None if data.get("prompt_config_path") is None else str(data.get("prompt_config_path"))
        ),
        split=str(data.get("split", "validation")),
        max_examples=int(data.get("max_examples", 100)),
        score_with_verifier=bool(data.get("score_with_verifier", True)),
    )
