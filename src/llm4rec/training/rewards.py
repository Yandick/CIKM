"""Reward aggregation and advantage utilities for verifier-guided training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AdvantageConfig:
    """Configuration for GRPO-style sequence-level advantage normalization."""

    epsilon: float = 1e-6
    clip_value: float = 5.0


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    config: AdvantageConfig | None = None,
) -> tuple[float, ...]:
    """Normalize rewards within a query group and clip extreme advantages."""

    if not rewards:
        raise ValueError("rewards must be non-empty")

    advantage_config = config or AdvantageConfig()
    reward_mean = sum(rewards) / float(len(rewards))
    variance = sum((reward - reward_mean) ** 2 for reward in rewards) / float(len(rewards))
    reward_std = max(variance**0.5, advantage_config.epsilon)

    normalized = tuple((reward - reward_mean) / reward_std for reward in rewards)
    return tuple(
        max(-advantage_config.clip_value, min(advantage_config.clip_value, value))
        for value in normalized
    )


def mean_reward(rewards: Sequence[float]) -> float:
    """Return the arithmetic mean reward over one query group."""

    if not rewards:
        raise ValueError("rewards must be non-empty")
    return sum(rewards) / float(len(rewards))
