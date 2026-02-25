"""
Reward signal processing for the contextual bandit router.

Converts execution outcomes into scalar reward signals that the router
uses to update its agent selection policy. Implements multiple reward
shaping strategies:

    1. Binary: 1.0 for success, 0.0 for failure.
    2. Continuous: Multi-factor reward based on latency, cost, and quality.
    3. Comparative: Reward based on relative performance vs. running average.

Reward normalization and windowed statistics ensure stable learning
under non-stationary reward distributions.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from lattice.core.router import RoutingHistory


class RewardStrategy(str, Enum):
    """Strategy for computing reward signals."""

    BINARY = "binary"
    CONTINUOUS = "continuous"
    COMPARATIVE = "comparative"


class RewardSignal(BaseModel):
    """Computed reward signal with provenance."""

    reward: float = Field(ge=0.0, le=1.0)
    strategy: RewardStrategy
    components: dict[str, float] = Field(default_factory=dict)
    agent_id: str
    task_id: str | None = None


class ExecutionFeedback(BaseModel):
    """Raw feedback from an execution step."""

    agent_id: str
    task_id: str | None = None
    success: bool
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    quality_score: float | None = None
    user_rating: float | None = None
    timestamp: float = Field(default_factory=time.time)


class FeedbackProcessor:
    """
    Processes execution feedback into reward signals for the router.

    Maintains running statistics per agent to normalize rewards and
    compute comparative scores. Supports configurable reward weights
    for different quality dimensions.

    Args:
        strategy: Reward computation strategy.
        latency_weight: Weight for latency in continuous reward.
        cost_weight: Weight for cost in continuous reward.
        quality_weight: Weight for quality in continuous reward.
        window_size: Size of the running statistics window.
        target_latency_ms: Target latency for normalization.
        target_cost_usd: Target cost for normalization.
    """

    def __init__(
        self,
        strategy: RewardStrategy = RewardStrategy.CONTINUOUS,
        latency_weight: float = 0.2,
        cost_weight: float = 0.2,
        quality_weight: float = 0.6,
        window_size: int = 100,
        target_latency_ms: float = 5000.0,
        target_cost_usd: float = 0.10,
    ) -> None:
        self._strategy = strategy
        self._latency_weight = latency_weight
        self._cost_weight = cost_weight
        self._quality_weight = quality_weight
        self._window_size = window_size
        self._target_latency = target_latency_ms
        self._target_cost = target_cost_usd

        # Per-agent windowed statistics
        self._agent_rewards: dict[str, deque[float]] = {}
        self._global_rewards: deque[float] = deque(maxlen=window_size)

    def process(
        self,
        feedback: ExecutionFeedback,
        task_embedding: list[float] | None = None,
    ) -> tuple[RewardSignal, RoutingHistory | None]:
        """
        Process execution feedback into a reward signal.

        Args:
            feedback: Raw execution feedback.
            task_embedding: Optional embedding for the task (for RoutingHistory).

        Returns:
            Tuple of (RewardSignal, optional RoutingHistory for router update).
        """
        if self._strategy == RewardStrategy.BINARY:
            signal = self._binary_reward(feedback)
        elif self._strategy == RewardStrategy.COMPARATIVE:
            signal = self._comparative_reward(feedback)
        else:
            signal = self._continuous_reward(feedback)

        # Track reward for statistics
        if feedback.agent_id not in self._agent_rewards:
            self._agent_rewards[feedback.agent_id] = deque(maxlen=self._window_size)
        self._agent_rewards[feedback.agent_id].append(signal.reward)
        self._global_rewards.append(signal.reward)

        # Build RoutingHistory if we have an embedding
        routing_history = None
        if task_embedding is not None:
            routing_history = RoutingHistory(
                task_embedding=task_embedding,
                agent_id=feedback.agent_id,
                reward=signal.reward,
            )

        return signal, routing_history

    def _binary_reward(self, feedback: ExecutionFeedback) -> RewardSignal:
        """Simple binary reward: 1.0 for success, 0.0 for failure."""
        reward = 1.0 if feedback.success else 0.0
        return RewardSignal(
            reward=reward,
            strategy=RewardStrategy.BINARY,
            components={"success": reward},
            agent_id=feedback.agent_id,
            task_id=feedback.task_id,
        )

    def _continuous_reward(self, feedback: ExecutionFeedback) -> RewardSignal:
        """
        Multi-factor continuous reward.

        Components:
            - success: 1.0 or 0.0
            - latency: exponential decay based on target
            - cost: exponential decay based on target
            - quality: direct quality score or default 0.7
        """
        if not feedback.success:
            return RewardSignal(
                reward=0.0,
                strategy=RewardStrategy.CONTINUOUS,
                components={"success": 0.0},
                agent_id=feedback.agent_id,
                task_id=feedback.task_id,
            )

        # Latency reward: exponential decay
        latency_reward = np.exp(-feedback.latency_ms / self._target_latency)

        # Cost reward: exponential decay
        cost_reward = np.exp(-feedback.cost_usd / self._target_cost) if self._target_cost > 0 else 1.0

        # Quality reward
        quality_reward = feedback.quality_score if feedback.quality_score is not None else 0.7
        if feedback.user_rating is not None:
            quality_reward = 0.5 * quality_reward + 0.5 * feedback.user_rating

        # Weighted combination
        reward = (
            self._latency_weight * latency_reward
            + self._cost_weight * cost_reward
            + self._quality_weight * quality_reward
        )

        # Clamp to [0, 1]
        reward = float(np.clip(reward, 0.0, 1.0))

        return RewardSignal(
            reward=reward,
            strategy=RewardStrategy.CONTINUOUS,
            components={
                "latency": float(latency_reward),
                "cost": float(cost_reward),
                "quality": float(quality_reward),
                "success": 1.0,
            },
            agent_id=feedback.agent_id,
            task_id=feedback.task_id,
        )

    def _comparative_reward(self, feedback: ExecutionFeedback) -> RewardSignal:
        """
        Reward based on performance relative to running average.

        Computes a continuous reward, then adjusts based on how the agent
        performs vs. the global average.
        """
        base = self._continuous_reward(feedback)

        # Compare to global average
        if self._global_rewards:
            global_avg = float(np.mean(list(self._global_rewards)))
            relative = base.reward - global_avg
            # Sigmoid mapping of relative performance to [0, 1]
            adjusted = 1.0 / (1.0 + np.exp(-5.0 * relative))
            reward = float(np.clip(adjusted, 0.0, 1.0))
        else:
            reward = base.reward

        components = dict(base.components)
        components["comparative_adjustment"] = reward - base.reward

        return RewardSignal(
            reward=reward,
            strategy=RewardStrategy.COMPARATIVE,
            components=components,
            agent_id=feedback.agent_id,
            task_id=feedback.task_id,
        )

    def get_agent_stats(self, agent_id: str) -> dict[str, float]:
        """Get running reward statistics for an agent."""
        rewards = self._agent_rewards.get(agent_id, deque())
        if not rewards:
            return {"mean": 0.0, "std": 0.0, "count": 0, "min": 0.0, "max": 0.0}

        arr = list(rewards)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "count": len(arr),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def get_global_stats(self) -> dict[str, Any]:
        """Get global reward statistics."""
        if not self._global_rewards:
            return {"mean": 0.0, "std": 0.0, "count": 0}

        arr = list(self._global_rewards)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "count": len(arr),
            "agents_tracked": len(self._agent_rewards),
        }
