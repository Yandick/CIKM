"""Candidate construction utilities for TRACE-Rec next-item experiments."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from llm4rec.data.schema import (
    CandidateConstructorConfig,
    CandidateItem,
    InteractionEvent,
    NextItemExample,
)


@dataclass(frozen=True)
class CandidatePoolItem:
    """One item available for candidate sampling, with a popularity prior."""

    item_id: str
    rank_prior: float


class PopularityCandidateConstructor:
    """Deterministic target-plus-negatives constructor ordered by item popularity."""

    def __init__(
        self,
        candidate_pool: Sequence[CandidatePoolItem],
        *,
        num_candidates: int = 10,
        exclude_history: bool = True,
        allow_insufficient_candidates: bool = False,
        shuffle_candidates: bool = True,
        seed: int = 42,
    ) -> None:
        if num_candidates < 2:
            raise ValueError("num_candidates must be at least 2")

        self.candidate_pool = tuple(candidate_pool)
        self.num_candidates = num_candidates
        self.exclude_history = exclude_history
        self.allow_insufficient_candidates = allow_insufficient_candidates
        self.shuffle_candidates = shuffle_candidates
        self.seed = seed

    @classmethod
    def from_interactions(
        cls,
        interactions: Iterable[InteractionEvent],
        *,
        config: CandidateConstructorConfig,
    ) -> "PopularityCandidateConstructor":
        """Build a popularity constructor from observed interaction frequencies."""

        counts = Counter(event.item_id for event in interactions)
        candidate_pool = tuple(
            CandidatePoolItem(item_id=item_id, rank_prior=float(count))
            for item_id, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        return cls(
            candidate_pool,
            num_candidates=config.num_candidates,
            exclude_history=config.exclude_history,
            allow_insufficient_candidates=config.allow_insufficient_candidates,
            shuffle_candidates=config.shuffle_candidates,
            seed=config.seed,
        )

    def construct(
        self,
        *,
        user_id: str,
        history_item_ids: Sequence[str],
        target_item_id: str,
        example_id: str | None = None,
    ) -> tuple[CandidateItem, ...]:
        """Construct a fixed candidate set containing one target and K-1 negatives."""

        if not target_item_id:
            raise ValueError("target_item_id is required")

        excluded = {target_item_id}
        if self.exclude_history:
            excluded.update(history_item_ids)

        negatives: list[CandidateItem] = []
        for item in self.candidate_pool:
            if item.item_id in excluded:
                continue
            negatives.append(
                CandidateItem(
                    item_id=item.item_id,
                    label=0,
                    source="popularity_negative",
                    rank_prior=item.rank_prior,
                )
            )
            if len(negatives) >= self.num_candidates - 1:
                break

        if len(negatives) < self.num_candidates - 1 and not self.allow_insufficient_candidates:
            raise ValueError("not enough negatives available to build a full candidate set")

        candidates = [
            CandidateItem(
                item_id=target_item_id,
                label=1,
                source="target",
                rank_prior=self._rank_prior_for(target_item_id),
            ),
            *negatives,
        ]

        if self.shuffle_candidates:
            rng = random.Random(self._local_seed(user_id, target_item_id, example_id))
            rng.shuffle(candidates)

        return tuple(candidates)

    def build_example(
        self,
        *,
        example_id: str,
        user_id: str,
        history: Sequence[InteractionEvent],
        target_item_id: str,
        split: str = "train",
        context: dict | None = None,
    ) -> NextItemExample:
        """Build a full next-item example with candidates and validate it."""

        history_tuple = tuple(sorted(history, key=lambda event: event.timestamp))
        candidates = self.construct(
            user_id=user_id,
            history_item_ids=[event.item_id for event in history_tuple],
            target_item_id=target_item_id,
            example_id=example_id,
        )
        example = NextItemExample(
            example_id=example_id,
            user_id=user_id,
            history=history_tuple,
            target_item_id=target_item_id,
            candidates=candidates,
            split=split,
            context={} if context is None else dict(context),
        )
        example.validate()
        return example

    def _rank_prior_for(self, item_id: str) -> float | None:
        """Return the popularity prior for one item if it exists in the pool."""

        for item in self.candidate_pool:
            if item.item_id == item_id:
                return item.rank_prior
        return None

    def _local_seed(self, user_id: str, target_item_id: str, example_id: str | None) -> int:
        """Derive a stable per-example seed so candidate order is reproducible."""

        seed_text = "::".join(
            [str(self.seed), user_id, target_item_id, "" if example_id is None else example_id]
        )
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)
