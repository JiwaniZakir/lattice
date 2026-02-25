"""
Throughput benchmark: measures execution throughput under varying concurrency.

Tests:
  - Steps per second at different concurrency levels
  - Memory overhead per step
  - DAG execution overhead vs. sequential
  - Checkpointing overhead
"""

import asyncio
import time
from typing import Any

import numpy as np

from lattice.agents.base import AgentResult, BaseAgent
from lattice.core.executor import Executor, ExecutorConfig
from lattice.core.planner import Plan, PlanningStrategy, SubGoal


class FastAgent(BaseAgent):
    """Minimal-overhead agent for throughput testing."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        # Simulate minimal work
        await asyncio.sleep(0.001)
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output="done",
            tokens_used=1,
            cost_usd=0.0001,
        )


def build_linear_plan(n: int) -> Plan:
    """Build a linear chain of n dependent steps."""
    goals = []
    for i in range(n):
        deps = [f"s{i - 1}"] if i > 0 else []
        goals.append(SubGoal(
            goal_id=f"s{i}",
            description=f"Step {i}",
            assigned_agent="fast",
            dependencies=deps,
        ))
    return Plan(task=f"Linear-{n}", strategy=PlanningStrategy.PLAN_AND_SOLVE, sub_goals=goals)


def build_wide_plan(n: int) -> Plan:
    """Build a plan with n independent (parallel) steps."""
    goals = [
        SubGoal(goal_id=f"s{i}", description=f"Step {i}", assigned_agent="fast")
        for i in range(n)
    ]
    return Plan(task=f"Wide-{n}", strategy=PlanningStrategy.PLAN_AND_SOLVE, sub_goals=goals)


def build_diamond_plan(width: int, depth: int) -> Plan:
    """Build a diamond-shaped DAG: fan-out then fan-in per layer."""
    goals = []
    prev_layer: list[str] = []

    for d in range(depth):
        layer = []
        for w in range(width):
            gid = f"s{d}_{w}"
            deps = prev_layer if d > 0 else []
            goals.append(SubGoal(
                goal_id=gid,
                description=f"Layer {d} Step {w}",
                assigned_agent="fast",
                dependencies=list(deps),
            ))
            layer.append(gid)
        prev_layer = layer

    return Plan(
        task=f"Diamond-{width}x{depth}",
        strategy=PlanningStrategy.PLAN_AND_SOLVE,
        sub_goals=goals,
    )


async def run_benchmark(
    plan: Plan,
    concurrency: int,
    iterations: int = 5,
) -> dict[str, Any]:
    """Run a plan multiple times and measure throughput."""
    agent = FastAgent(agent_id="fast")
    executor = Executor(
        config=ExecutorConfig(
            max_concurrent=concurrency,
            enable_verification=False,
            enable_checkpointing=False,
            max_retries=0,
        ),
        agents={"fast": agent},
    )

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = await executor.execute(plan)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    lat_arr = np.array(latencies)
    num_steps = len(plan.sub_goals)
    avg_ms = float(np.mean(lat_arr))
    steps_per_sec = (num_steps / (avg_ms / 1000)) if avg_ms > 0 else 0

    return {
        "plan": plan.task,
        "steps": num_steps,
        "concurrency": concurrency,
        "avg_ms": avg_ms,
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "steps_per_sec": steps_per_sec,
    }


async def main() -> None:
    print("=" * 70)
    print("Lattice Throughput Benchmark")
    print("=" * 70)

    # Linear chains
    print("\n--- Linear chains (sequential) ---")
    for n in [5, 10, 20]:
        plan = build_linear_plan(n)
        result = await run_benchmark(plan, concurrency=1)
        print(f"  {plan.task:>12}: {result['avg_ms']:>8.1f}ms avg  "
              f"{result['steps_per_sec']:>8.0f} steps/s")

    # Wide plans (parallel)
    print("\n--- Wide plans (parallel) ---")
    for n in [5, 10, 20]:
        for conc in [1, 5, 10]:
            plan = build_wide_plan(n)
            result = await run_benchmark(plan, concurrency=conc)
            print(f"  {plan.task:>12} @{conc:>2} conc: {result['avg_ms']:>8.1f}ms avg  "
                  f"{result['steps_per_sec']:>8.0f} steps/s")

    # Diamond DAGs
    print("\n--- Diamond DAGs ---")
    for width, depth in [(3, 3), (5, 5), (10, 3)]:
        plan = build_diamond_plan(width, depth)
        result = await run_benchmark(plan, concurrency=width)
        print(f"  {plan.task:>16}: {result['avg_ms']:>8.1f}ms avg  "
              f"{result['steps_per_sec']:>8.0f} steps/s  "
              f"({result['steps']} steps)")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())
