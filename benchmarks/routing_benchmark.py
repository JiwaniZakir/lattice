"""
Routing benchmark: measures decision latency and reward convergence.

Simulates routing decisions across multiple agents and tracks:
  - Decision latency (p50, p95, p99)
  - Reward convergence over training steps
  - Exploration vs. exploitation ratio
  - Per-strategy comparison
"""

import asyncio
import time

import numpy as np

from lattice.agents.base import AgentResult, BaseAgent
from lattice.core.router import ExplorationStrategy, Router, RoutingHistory


class BenchmarkAgent(BaseAgent):
    """Agent with configurable quality for benchmarking."""

    def __init__(self, agent_id: str, quality: float) -> None:
        super().__init__(agent_id=agent_id, name=f"Agent-{agent_id}")
        self._quality = quality

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id, task=task, output="bench", cost_usd=0.001,
        )

    @property
    def quality(self) -> float:
        return self._quality


async def benchmark_strategy(
    strategy: ExplorationStrategy,
    num_agents: int = 5,
    embedding_dim: int = 128,
    num_rounds: int = 500,
) -> dict:
    """Run a routing benchmark for a given strategy."""
    # Create agents with varying quality
    agents = [
        BenchmarkAgent(f"agent_{i}", quality=0.3 + 0.6 * (i / (num_agents - 1)))
        for i in range(num_agents)
    ]
    best_agent_id = agents[-1].agent_id  # Highest quality

    router = Router(
        agents=agents,
        embedding_dim=embedding_dim,
        strategy=strategy,
        epsilon=0.5,
        epsilon_decay=0.995,
    )

    latencies: list[float] = []
    rewards: list[float] = []
    best_selections = 0
    explore_count = 0

    for step in range(num_rounds):
        embedding = np.random.randn(embedding_dim).astype(np.float32)

        start = time.perf_counter()
        decision = await router.route(embedding)
        latency_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(latency_us)

        if decision.explored:
            explore_count += 1
        if decision.agent_id == best_agent_id:
            best_selections += 1

        # Simulate reward based on agent quality
        agent = next(a for a in agents if a.agent_id == decision.agent_id)
        noise = np.random.normal(0, 0.1)
        reward = np.clip(agent.quality + noise, 0.0, 1.0)
        rewards.append(reward)

        await router.update(RoutingHistory(
            task_embedding=embedding.tolist(),
            agent_id=decision.agent_id,
            reward=float(reward),
        ))

    lat_arr = np.array(latencies)
    return {
        "strategy": strategy.value,
        "rounds": num_rounds,
        "p50_latency_us": float(np.percentile(lat_arr, 50)),
        "p95_latency_us": float(np.percentile(lat_arr, 95)),
        "p99_latency_us": float(np.percentile(lat_arr, 99)),
        "mean_reward": float(np.mean(rewards)),
        "final_50_reward": float(np.mean(rewards[-50:])),
        "best_agent_rate": best_selections / num_rounds,
        "exploration_rate": explore_count / num_rounds,
        "final_epsilon": router.epsilon,
    }


async def main() -> None:
    print("=" * 70)
    print("Lattice Routing Benchmark")
    print("=" * 70)

    strategies = [
        ExplorationStrategy.EPSILON_GREEDY,
        ExplorationStrategy.UCB,
        ExplorationStrategy.THOMPSON,
    ]

    for strategy in strategies:
        print(f"\nBenchmarking {strategy.value}...")
        result = await benchmark_strategy(strategy)
        print(f"  Decision latency  p50={result['p50_latency_us']:.1f}us  "
              f"p95={result['p95_latency_us']:.1f}us  "
              f"p99={result['p99_latency_us']:.1f}us")
        print(f"  Mean reward:      {result['mean_reward']:.4f}")
        print(f"  Final-50 reward:  {result['final_50_reward']:.4f}")
        print(f"  Best agent rate:  {result['best_agent_rate']:.2%}")
        print(f"  Exploration rate: {result['exploration_rate']:.2%}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())
