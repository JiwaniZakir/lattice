"""
Lattice Quick Start Example

Demonstrates the basic orchestration loop:
  1. Create agents
  2. Route a task
  3. Plan execution
  4. Verify safety
  5. Execute
"""

import asyncio

import numpy as np

from lattice.agents.base import AgentCapabilities, AgentResult, BaseAgent
from lattice.core.executor import Executor, ExecutorConfig
from lattice.core.memory import ScopedMemory
from lattice.core.planner import Planner, PlannerConfig, PlanningStrategy
from lattice.core.router import ExplorationStrategy, Router
from lattice.core.verifier import Verifier, VerifierConfig
from lattice.routing.embedder import TaskEmbedder


# -- Define a simple agent --------------------------------------------------

class EchoAgent(BaseAgent):
    """A simple agent that echoes its input (for demonstration)."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=f"Completed: {task}",
            tokens_used=len(task.split()),
            cost_usd=0.001,
        )


class ResearchAgent(BaseAgent):
    """A mock research agent."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=f"Research findings for: {task}",
            tokens_used=50,
            cost_usd=0.005,
        )


# -- Main orchestration loop ------------------------------------------------

async def main() -> None:
    # 1. Create agents
    echo = EchoAgent(agent_id="echo", name="Echo Agent")
    research = ResearchAgent(
        agent_id="research",
        name="Research Agent",
        capabilities=AgentCapabilities(
            supported_task_types=["research"],
        ),
    )

    # 2. Set up the router with contextual bandits
    router = Router(
        agents=[echo, research],
        embedding_dim=384,
        strategy=ExplorationStrategy.EPSILON_GREEDY,
        epsilon=0.3,
    )

    # 3. Set up the planner (no LLM -- uses heuristic mode)
    planner = Planner(
        config=PlannerConfig(strategy=PlanningStrategy.REACT),
    )

    # 4. Set up the verifier
    verifier = Verifier(config=VerifierConfig())

    # 5. Set up shared memory
    memory = ScopedMemory(default_scope="quickstart")

    # 6. Set up the executor
    executor = Executor(
        config=ExecutorConfig(max_concurrent=3, enable_verification=False),
        agents={"echo": echo, "research": research},
        verifier=verifier,
        memory=memory,
    )

    # 7. Embed a task
    embedder = TaskEmbedder(dimension=384)
    task = "Research the latest advances in multi-agent systems"
    embedding_result = await embedder.embed(task)
    embedding = np.array(embedding_result.embedding)

    # 8. Route the task
    decision = await router.route(embedding)
    print(f"Routed to: {decision.agent_id} (confidence: {decision.confidence:.3f})")

    # 9. Plan the task
    plan = await planner.plan(
        task=task,
        available_agents=[decision.agent_id],
    )
    print(f"Plan created: {len(plan.sub_goals)} sub-goals, strategy: {plan.strategy}")

    # 10. Execute the plan
    result = await executor.execute(plan)
    print(f"Execution: {result.status.value}")
    for sr in result.step_results:
        print(f"  Step {sr.goal_id}: {sr.status.value} -> {sr.output}")
    print(f"Total tokens: {result.total_tokens}, Cost: ${result.total_cost_usd:.4f}")

    # 11. Store result in memory
    await memory.set("last_result", result.model_dump(), scope="quickstart")
    stored = await memory.get("last_result", scope="quickstart")
    print(f"Stored in memory: {stored is not None}")

    # 12. Send reward feedback to the router
    from lattice.core.router import RoutingHistory
    await router.update(RoutingHistory(
        task_embedding=embedding_result.embedding,
        agent_id=decision.agent_id,
        reward=0.9,
    ))
    print(f"Router epsilon after update: {router.epsilon:.4f}")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
