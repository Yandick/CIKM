"""Dataset and example schemas for TRACE-Rec V1 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InteractionEvent:
    """One user-item interaction in chronological history."""

    item_id: str
    timestamp: int
    rating: float | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemRecord:
    """Minimal item metadata record for candidate-aware prompting and tracing."""

    item_id: str
    title: str | None = None
    feature_refs: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateItem:
    """One item inside a fixed candidate set for next-item recommendation."""

    item_id: str
    label: int
    source: str
    rank_prior: float | None = None


@dataclass(frozen=True)
class NextItemExample:
    """Canonical TRACE-Rec input example used by models and evaluators."""

    example_id: str
    user_id: str
    history: tuple[InteractionEvent, ...]
    target_item_id: str
    candidates: tuple[CandidateItem, ...]
    split: str = "train"
    context: dict[str, Any] = field(default_factory=dict)

    def candidate_ids(self) -> tuple[str, ...]:
        """Return candidate ids in presentation order."""

        return tuple(candidate.item_id for candidate in self.candidates)

    def seen_item_ids(self) -> tuple[str, ...]:
        """Return history item ids in chronological order."""

        return tuple(event.item_id for event in self.history)

    def positive_candidates(self) -> tuple[CandidateItem, ...]:
        """Return positively labeled candidates."""

        return tuple(candidate for candidate in self.candidates if candidate.label == 1)

    def validate(self) -> None:
        """Validate basic invariants for a next-item example."""

        if not self.example_id:
            raise ValueError("example_id is required")
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.history:
            raise ValueError("history must contain at least one interaction")
        if not self.target_item_id:
            raise ValueError("target_item_id is required")
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        if len(set(self.candidate_ids())) != len(self.candidates):
            raise ValueError("candidate ids must be unique")

        positives = self.positive_candidates()
        if len(positives) != 1:
            raise ValueError("exactly one positive candidate is required")
        if positives[0].item_id != self.target_item_id:
            raise ValueError("positive candidate must match target_item_id")

        timestamps = [event.timestamp for event in self.history]
        if timestamps != sorted(timestamps):
            raise ValueError("history must be sorted by timestamp")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Dataset preprocessing settings for next-item example construction."""

    min_history_len: int = 5
    max_history_len: int = 50
    rating_threshold: float | None = None
    max_item_snippets: int = 3
    max_snippet_chars: int = 240


@dataclass(frozen=True)
class CandidateConstructorConfig:
    """Config for fixed-set candidate construction."""

    strategy: str = "popularity"
    num_candidates: int = 10
    exclude_history: bool = True
    allow_insufficient_candidates: bool = False
    shuffle_candidates: bool = True
    seed: int = 42


@dataclass(frozen=True)
class DatasetConfig:
    """Top-level dataset config loaded from YAML."""

    name: str
    raw_dir: str
    interactions_file: str | None
    validation_file: str | None
    test_file: str | None
    review_file: str | None
    items_file: str | None
    metadata_file: str | None
    separator: str
    columns: dict[str, str]
    preprocessing: PreprocessingConfig
    candidate_constructor: CandidateConstructorConfig


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """Load a dataset config YAML into typed dataclasses."""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    preprocessing = data.get("preprocessing", {})
    constructor = data.get("candidate_constructor", {})

    return DatasetConfig(
        name=str(data.get("name", "")),
        raw_dir=str(data.get("raw_dir", "")),
        interactions_file=(
            None
            if data.get("interactions_file") is None
            else str(data.get("interactions_file"))
        ),
        validation_file=(
            None if data.get("validation_file") is None else str(data.get("validation_file"))
        ),
        test_file=None if data.get("test_file") is None else str(data.get("test_file")),
        review_file=None if data.get("review_file") is None else str(data.get("review_file")),
        items_file=None if data.get("items_file") is None else str(data.get("items_file")),
        metadata_file=(
            None if data.get("metadata_file") is None else str(data.get("metadata_file"))
        ),
        separator=str(data.get("separator", "\t")),
        columns={str(k): str(v) for k, v in data.get("columns", {}).items()},
        preprocessing=PreprocessingConfig(
            min_history_len=int(preprocessing.get("min_history_len", 5)),
            max_history_len=int(preprocessing.get("max_history_len", 50)),
            rating_threshold=(
                None
                if preprocessing.get("rating_threshold") is None
                else float(preprocessing.get("rating_threshold"))
            ),
            max_item_snippets=int(preprocessing.get("max_item_snippets", 3)),
            max_snippet_chars=int(preprocessing.get("max_snippet_chars", 240)),
        ),
        candidate_constructor=CandidateConstructorConfig(
            strategy=str(constructor.get("strategy", "popularity")),
            num_candidates=int(constructor.get("num_candidates", 10)),
            exclude_history=bool(constructor.get("exclude_history", True)),
            allow_insufficient_candidates=bool(
                constructor.get("allow_insufficient_candidates", False)
            ),
            shuffle_candidates=bool(constructor.get("shuffle_candidates", True)),
            seed=int(constructor.get("seed", 42)),
        ),
    )
