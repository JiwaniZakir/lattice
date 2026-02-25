"""
Multi-Agent Research Pipeline Example

Demonstrates a multi-agent workflow where:
  1. A research agent gathers information
  2. An analysis agent synthesizes findings
  3. A critic agent evaluates quality
  4. Results are stored in scoped memory

Uses the full Lattice pipeline: Router -> Planner -> Verifier -> Executor.
"""

import asyncio
from typing import Any

from lattice.agents.base import AgentCapabilities, AgentResult, BaseAgent
from lattice.agents.critic import CriticAgent
from lattice.core.executor import Executor, ExecutorConfig
from lattice.core.memory import ScopedMemory
from lattice.core.planner import Planner, PlannerConfig, PlanningStrategy, SubGoal
from lattice.core.verifier import Verifier, VerifierConfig
from lattice.observability.metrics import CostTracker, MetricsCollector


class ResearchAgent(BaseAgent):
    """Simulates gathering research on a topic."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        # In production, this would call an LLM with search tools
        findings = (
            f"Research findings on '{task}':\n"
            "1. Multi-agent systems show 40% improvement over single-agent approaches.\n"
            "2. Contextual bandits reduce routing latency by 60%.\n"
            "3. Z3 verification catches 95% of safety violations pre-execution."
        )
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=findings,
            tokens_used=150,
            cost_usd=0.003,
        )


class AnalysisAgent(BaseAgent):
    """Synthesizes research findings into a structured analysis."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        # Build on previous research
        research_output = ""
        if context:
            for key, val in context.items():
                if isinstance(val, str):
                    research_output += val + "\n"

        analysis = (
            f"Analysis of: {task}\n"
            f"Based on research: {research_output[:200]}...\n\n"
            "Key Insights:\n"
            "- Multi-agent orchestration is a rapidly growing field.\n"
            "- Learned routing significantly outperforms static assignment.\n"
            "- Formal verification is essential for production deployments."
        )
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=analysis,
            tokens_used=200,
            cost_usd=0.004,
        )


async def main() -> None:
    # -- Set up agents -------------------------------------------------------
    researcher = ResearchAgent(
        agent_id="researcher",
        name="Research Agent",
        capabilities=AgentCapabilities(supported_task_types=["research"]),
    )
    analyst = AnalysisAgent(
        agent_id="analyst",
        name="Analysis Agent",
        capabilities=AgentCapabilities(supported_task_types=["analysis"]),
    )
    critic = CriticAgent(
        agent_id="critic",
        name="Quality Critic",
    )

    # -- Set up infrastructure -----------------------------------------------
    memory = ScopedMemory(default_scope="research_pipeline")
    memory.create_scope("research", parent="research_pipeline")
    memory.create_scope("analysis", parent="research_pipeline")

    metrics = MetricsCollector(cost_tracker=CostTracker(budget_usd=1.0))
    verifier = Verifier(config=VerifierConfig())

    executor = Executor(
        config=ExecutorConfig(max_concurrent=3),
        agents={
            "researcher": researcher,
            "analyst": analyst,
            "critic": critic,
        },
        verifier=verifier,
        memory=memory,
    )

    planner = Planner(
        config=PlannerConfig(strategy=PlanningStrategy.PLAN_AND_SOLVE),
    )

    # -- Build a multi-step plan manually ------------------------------------
    # In production, the planner would generate this from a task description
    from lattice.core.planner import Plan

    plan = Plan(
        task="Research and analyze multi-agent orchestration",
        strategy=PlanningStrategy.PLAN_AND_SOLVE,
        sub_goals=[
            SubGoal(
                goal_id="step_1",
                description="Research multi-agent orchestration techniques",
                assigned_agent="researcher",
                dependencies=[],
            ),
            SubGoal(
                goal_id="step_2",
                description="Research learned routing approaches",
                assigned_agent="researcher",
                dependencies=[],
            ),
            SubGoal(
                goal_id="step_3",
                description="Synthesize research findings",
                assigned_agent="analyst",
                dependencies=["step_1", "step_2"],
            ),
            SubGoal(
                goal_id="step_4",
                description="Evaluate analysis quality",
                assigned_agent="critic",
                dependencies=["step_3"],
            ),
        ],
    )

    # -- Verify the plan -----------------------------------------------------
    print("Verifying plan safety...")
    verification = await verifier.verify_plan(plan)
    print(f"  Safe: {verification.is_safe}")
    if verification.warnings:
        for w in verification.warnings:
            print(f"  Warning: {w}")

    # -- Execute the plan ----------------------------------------------------
    print("\nExecuting plan...")
    result = await executor.execute(plan)

    print(f"\nExecution status: {result.status.value}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total cost: ${result.total_cost_usd:.4f}")
    print(f"Latency: {result.total_latency_ms:.1f}ms")

    print("\nStep results:")
    for sr in result.step_results:
        print(f"  [{sr.status.value}] {sr.goal_id} ({sr.agent_id})")
        if sr.output:
            output_preview = str(sr.output)[:100]
            print(f"    Output: {output_preview}...")

    # -- Store results in memory ---------------------------------------------
    for sr in result.step_results:
        await memory.set(
            sr.goal_id,
            sr.output,
            scope="research" if sr.agent_id == "researcher" else "analysis",
        )

    stats = memory.get_stats()
    print(f"\nMemory: {stats['total_entries']} entries across {stats['num_scopes']} scopes")


if __name__ == "__main__":
    asyncio.run(main())
