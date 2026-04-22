"""Unit tests for TRACE-Rec data schema and candidate construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.data import (
    CandidateConstructorConfig,
    InteractionEvent,
    NextItemExample,
    PopularityCandidateConstructor,
    load_dataset_config,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "amazon_food.yaml"


def _toy_interactions() -> list[InteractionEvent]:
    return [
        InteractionEvent(item_id="i1", timestamp=1),
        InteractionEvent(item_id="i2", timestamp=2),
        InteractionEvent(item_id="i2", timestamp=3),
        InteractionEvent(item_id="i3", timestamp=4),
        InteractionEvent(item_id="i4", timestamp=5),
        InteractionEvent(item_id="i4", timestamp=6),
        InteractionEvent(item_id="i4", timestamp=7),
        InteractionEvent(item_id="i5", timestamp=8),
        InteractionEvent(item_id="i6", timestamp=9),
    ]


def test_load_amazon_food_dataset_config() -> None:
    config = load_dataset_config(CONFIG_PATH)

    assert config.name == "amazon_food"
    assert config.review_file == "Grocery_and_Gourmet_Food.jsonl"
    assert config.metadata_file == "meta_Grocery_and_Gourmet_Food.jsonl"
    assert config.preprocessing.min_history_len == 5
    assert config.candidate_constructor.strategy == "popularity"
    assert config.candidate_constructor.num_candidates == 20


def test_popularity_constructor_includes_target_and_excludes_history() -> None:
    interactions = _toy_interactions()
    config = CandidateConstructorConfig(num_candidates=4, seed=13)
    constructor = PopularityCandidateConstructor.from_interactions(
        interactions, config=config
    )

    candidates = constructor.construct(
        user_id="u1",
        history_item_ids=["i1", "i2"],
        target_item_id="i3",
        example_id="ex1",
    )

    candidate_ids = [candidate.item_id for candidate in candidates]
    labels = [candidate.label for candidate in candidates]

    assert "i3" in candidate_ids
    assert "i1" not in candidate_ids
    assert "i2" not in candidate_ids
    assert sum(labels) == 1


def test_popularity_constructor_is_deterministic() -> None:
    interactions = _toy_interactions()
    config = CandidateConstructorConfig(num_candidates=4, seed=7)
    constructor = PopularityCandidateConstructor.from_interactions(
        interactions, config=config
    )

    first = constructor.construct(
        user_id="u9",
        history_item_ids=["i1"],
        target_item_id="i5",
        example_id="same-example",
    )
    second = constructor.construct(
        user_id="u9",
        history_item_ids=["i1"],
        target_item_id="i5",
        example_id="same-example",
    )

    assert first == second


def test_build_example_validates_positive_target_alignment() -> None:
    interactions = _toy_interactions()
    config = CandidateConstructorConfig(num_candidates=4, seed=11)
    constructor = PopularityCandidateConstructor.from_interactions(
        interactions, config=config
    )

    example = constructor.build_example(
        example_id="ex2",
        user_id="u1",
        history=interactions[:3],
        target_item_id="i5",
        split="train",
    )

    assert isinstance(example, NextItemExample)
    assert len(example.candidates) == 4
    assert example.positive_candidates()[0].item_id == "i5"


def test_constructor_raises_when_negatives_are_insufficient() -> None:
    interactions = [
        InteractionEvent(item_id="i1", timestamp=1),
        InteractionEvent(item_id="i2", timestamp=2),
        InteractionEvent(item_id="i3", timestamp=3),
    ]
    config = CandidateConstructorConfig(num_candidates=4, seed=0)
    constructor = PopularityCandidateConstructor.from_interactions(
        interactions, config=config
    )

    with pytest.raises(ValueError):
        constructor.construct(
            user_id="u1",
            history_item_ids=["i1", "i2"],
            target_item_id="i3",
            example_id="ex3",
        )
