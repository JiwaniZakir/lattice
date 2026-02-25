"""
Cost attribution and metrics collection.

Tracks resource consumption across the orchestration pipeline with
per-agent and per-model cost attribution. Provides:

    - Token counting and cost estimation per model
    - Per-agent cost breakdown
    - Latency percentile tracking
    - Throughput metrics
    - Budget monitoring with alerts
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


# Cost per 1M tokens for common models (approximate, as of 2025).
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
}


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class AgentMetrics(BaseModel):
    """Aggregated metrics for a single agent."""

    agent_id: str
    total_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class CostTracker:
    """
    Tracks token usage and cost attribution per agent and model.

    Estimates costs based on published pricing for common models and
    supports custom cost tables for other providers.

    Args:
        custom_costs: Optional custom cost table overriding defaults.
        budget_usd: Optional budget limit for alerts.
    """

    def __init__(
        self,
        custom_costs: dict[str, dict[str, float]] | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self._costs = {**MODEL_COSTS, **(custom_costs or {})}
        self._budget = budget_usd
        self._total_cost = 0.0
        self._agent_costs: dict[str, float] = defaultdict(float)
        self._model_costs: dict[str, float] = defaultdict(float)
        self._history: list[TokenUsage] = []

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost for a given model and token counts.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        # Try exact match first, then prefix match
        cost_entry = self._costs.get(model)
        if cost_entry is None:
            for key in self._costs:
                if model.startswith(key) or key.startswith(model):
                    cost_entry = self._costs[key]
                    break

        if cost_entry is None:
            # Default to GPT-4o pricing
            cost_entry = self._costs.get("gpt-4o", {"input": 2.50, "output": 10.00})

        input_cost = (input_tokens / 1_000_000) * cost_entry["input"]
        output_cost = (output_tokens / 1_000_000) * cost_entry["output"]
        return input_cost + output_cost

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: str | None = None,
    ) -> TokenUsage:
        """
        Record token usage and compute cost.

        Args:
            model: Model used.
            input_tokens: Input tokens consumed.
            output_tokens: Output tokens consumed.
            agent_id: Optional agent that made the call.

        Returns:
            TokenUsage record.
        """
        cost = self.estimate_cost(model, input_tokens, output_tokens)

        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=cost,
        )

        self._total_cost += cost
        self._model_costs[model] += cost
        if agent_id:
            self._agent_costs[agent_id] += cost
        self._history.append(usage)

        return usage

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def is_over_budget(self) -> bool:
        if self._budget is None:
            return False
        return self._total_cost > self._budget

    @property
    def budget_remaining(self) -> float | None:
        if self._budget is None:
            return None
        return max(self._budget - self._total_cost, 0.0)

    def get_cost_breakdown(self) -> dict[str, Any]:
        """Get a cost breakdown by model and agent."""
        return {
            "total_usd": self._total_cost,
            "by_model": dict(self._model_costs),
            "by_agent": dict(self._agent_costs),
            "budget_usd": self._budget,
            "budget_remaining_usd": self.budget_remaining,
            "total_calls": len(self._history),
        }


class MetricsCollector:
    """
    Collects and aggregates execution metrics across all agents.

    Tracks latency distributions, throughput, success rates, and
    other operational metrics.

    Args:
        cost_tracker: Optional cost tracker for cost attribution.
    """

    def __init__(self, cost_tracker: CostTracker | None = None) -> None:
        self._cost_tracker = cost_tracker or CostTracker()
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._call_counts: dict[str, int] = defaultdict(int)
        self._success_counts: dict[str, int] = defaultdict(int)
        self._error_counts: dict[str, int] = defaultdict(int)
        self._start_time = time.time()

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def record_call(
        self,
        agent_id: str,
        latency_ms: float,
        success: bool,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a single agent call."""
        self._call_counts[agent_id] += 1
        self._latencies[agent_id].append(latency_ms)

        if success:
            self._success_counts[agent_id] += 1
        else:
            self._error_counts[agent_id] += 1

    def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get aggregated metrics for a specific agent."""
        lats = self._latencies.get(agent_id, [])
        calls = self._call_counts.get(agent_id, 0)

        if lats:
            arr = np.array(lats)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            avg = float(np.mean(arr))
            total_lat = float(np.sum(arr))
        else:
            p50 = p95 = p99 = avg = total_lat = 0.0

        return AgentMetrics(
            agent_id=agent_id,
            total_calls=calls,
            total_tokens=0,
            total_cost_usd=self._cost_tracker._agent_costs.get(agent_id, 0.0),
            total_latency_ms=total_lat,
            success_count=self._success_counts.get(agent_id, 0),
            error_count=self._error_counts.get(agent_id, 0),
            avg_latency_ms=avg,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
        )

    def get_all_metrics(self) -> dict[str, AgentMetrics]:
        """Get metrics for all tracked agents."""
        agents = set(self._call_counts.keys())
        return {aid: self.get_agent_metrics(aid) for aid in agents}

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all metrics."""
        elapsed = time.time() - self._start_time
        total_calls = sum(self._call_counts.values())
        total_success = sum(self._success_counts.values())
        total_errors = sum(self._error_counts.values())

        return {
            "elapsed_s": elapsed,
            "total_calls": total_calls,
            "total_success": total_success,
            "total_errors": total_errors,
            "success_rate": total_success / total_calls if total_calls > 0 else 0.0,
            "throughput_calls_per_s": total_calls / elapsed if elapsed > 0 else 0.0,
            "cost": self._cost_tracker.get_cost_breakdown(),
            "num_agents": len(self._call_counts),
        }
