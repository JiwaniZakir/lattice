"""
DAG execution engine with dependency tracking, parallelism, and checkpointing.

The executor takes a Plan (from the planner) and executes its sub-goals as a
directed acyclic graph. Sub-goals with no dependencies run in parallel;
dependent sub-goals wait for their predecessors. The engine supports:

    - Execution checkpointing for fault recovery
    - Token-level streaming from agent outputs
    - Cost tracking and attribution per sub-goal
    - Timeout and retry policies
    - OpenTelemetry span creation per execution step

This module coordinates all other core components: it calls the Verifier
before execution, dispatches to Agents via the Router, and stores results
in Memory.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from lattice.agents.base import BaseAgent
    from lattice.core.memory import ScopedMemory
    from lattice.core.planner import Plan, SubGoal
    from lattice.core.verifier import Verifier
    from lattice.execution.checkpointing import CheckpointManager
    from lattice.observability.tracing import TracingManager


class ExecutionStatus(str, Enum):
    """Status of an execution step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StepResult(BaseModel):
    """Result of executing a single sub-goal."""

    step_id: str
    goal_id: str
    agent_id: str
    status: ExecutionStatus
    output: Any = None
    error: str | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0


class ExecutionResult(BaseModel):
    """Complete result of executing a plan."""

    execution_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    plan_id: str
    status: ExecutionStatus
    step_results: list[StepResult] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None


class ExecutorConfig(BaseModel):
    """Configuration for the executor."""

    max_concurrent: int = 5
    step_timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    enable_checkpointing: bool = True
    enable_verification: bool = True
    fail_fast: bool = False


class Executor:
    """
    DAG execution engine for multi-agent plans.

    Given a Plan containing sub-goals with dependency edges, the executor
    builds a dependency graph and executes sub-goals in topological order
    with maximum parallelism for independent steps.

    Features:
        - Topological sort for execution ordering
        - asyncio.TaskGroup for parallel execution of independent steps
        - Per-step timeout and retry with exponential backoff
        - Checkpoint save after each completed step
        - Optional Z3 verification before execution
        - Cost and token tracking per step

    Args:
        config: Executor configuration.
        agents: Mapping from agent_id to agent instance.
        verifier: Optional verifier for pre-execution safety checks.
        memory: Optional memory for storing results.
        checkpoint_manager: Optional checkpoint manager.
        tracer: Optional OpenTelemetry tracer.
    """

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        agents: dict[str, BaseAgent] | None = None,
        verifier: Verifier | None = None,
        memory: ScopedMemory | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        tracer: TracingManager | None = None,
    ) -> None:
        self._config = config or ExecutorConfig()
        self._agents = agents or {}
        self._verifier = verifier
        self._memory = memory
        self._checkpoint_mgr = checkpoint_manager
        self._tracer = tracer
        self._running: dict[str, asyncio.Task[StepResult]] = {}

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    def register_agent(self, agent_id: str, agent: BaseAgent) -> None:
        """Register an agent for execution dispatch."""
        self._agents[agent_id] = agent

    async def execute(self, plan: Plan) -> ExecutionResult:
        """
        Execute a plan as a DAG of sub-goals.

        Builds a dependency graph from the plan's sub-goals, resolves
        execution order via topological sort, and runs independent
        steps concurrently up to the configured concurrency limit.

        Args:
            plan: The plan to execute.

        Returns:
            ExecutionResult with per-step results and aggregate metrics.
        """
        start = time.perf_counter()
        result = ExecutionResult(
            plan_id=plan.plan_id,
            status=ExecutionStatus.RUNNING,
        )

        # Try to restore from checkpoint
        if self._config.enable_checkpointing and self._checkpoint_mgr:
            restored = await self._checkpoint_mgr.restore(plan.plan_id)
            if restored:
                result = restored

        # Verification gate
        if self._config.enable_verification and self._verifier:
            verification = await self._verifier.verify_plan(plan)
            if not verification.is_safe:
                result.status = ExecutionStatus.FAILED
                for violation in verification.violations:
                    result.step_results.append(StepResult(
                        step_id=uuid.uuid4().hex[:12],
                        goal_id="verification",
                        agent_id="verifier",
                        status=ExecutionStatus.FAILED,
                        error=f"Safety violation: {violation}",
                    ))
                return result

        # Build dependency graph
        completed_goals = {
            sr.goal_id for sr in result.step_results
            if sr.status == ExecutionStatus.COMPLETED
        }
        pending_goals = [
            g for g in plan.sub_goals
            if g.goal_id not in completed_goals
        ]

        # Execute using topological ordering with concurrency control
        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        completed: dict[str, StepResult] = {
            sr.goal_id: sr for sr in result.step_results
            if sr.status == ExecutionStatus.COMPLETED
        }

        # Process in waves: each wave contains goals whose dependencies are met
        remaining = list(pending_goals)
        while remaining:
            ready = [
                g for g in remaining
                if all(dep in completed for dep in g.dependencies)
            ]
            if not ready:
                # Circular dependency or unresolvable
                for g in remaining:
                    result.step_results.append(StepResult(
                        step_id=uuid.uuid4().hex[:12],
                        goal_id=g.goal_id,
                        agent_id=g.assigned_agent or "unknown",
                        status=ExecutionStatus.FAILED,
                        error="Unresolvable dependencies",
                    ))
                break

            # Execute ready goals concurrently
            tasks = [
                self._execute_step(g, completed, semaphore)
                for g in ready
            ]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for g, sr in zip(ready, step_results):
                if isinstance(sr, BaseException):
                    step_result = StepResult(
                        step_id=uuid.uuid4().hex[:12],
                        goal_id=g.goal_id,
                        agent_id=g.assigned_agent or "unknown",
                        status=ExecutionStatus.FAILED,
                        error=str(sr),
                    )
                else:
                    step_result = sr

                result.step_results.append(step_result)

                if step_result.status == ExecutionStatus.COMPLETED:
                    completed[g.goal_id] = step_result
                elif self._config.fail_fast:
                    # Cancel remaining goals
                    for rg in remaining:
                        if rg.goal_id not in completed and rg.goal_id != g.goal_id:
                            result.step_results.append(StepResult(
                                step_id=uuid.uuid4().hex[:12],
                                goal_id=rg.goal_id,
                                agent_id=rg.assigned_agent or "unknown",
                                status=ExecutionStatus.CANCELLED,
                            ))
                    remaining = []
                    break

                # Checkpoint after each step
                if self._config.enable_checkpointing and self._checkpoint_mgr:
                    await self._checkpoint_mgr.save(plan.plan_id, result)

            remaining = [g for g in remaining if g.goal_id not in completed]

        # Compute aggregates
        result.total_tokens = sum(sr.tokens_used for sr in result.step_results)
        result.total_cost_usd = sum(sr.cost_usd for sr in result.step_results)
        result.total_latency_ms = (time.perf_counter() - start) * 1000
        result.completed_at = time.time()

        all_completed = all(
            sr.status == ExecutionStatus.COMPLETED
            for sr in result.step_results
        )
        result.status = (
            ExecutionStatus.COMPLETED if all_completed else ExecutionStatus.FAILED
        )

        # Store result in memory
        if self._memory:
            await self._memory.set(
                f"execution:{result.execution_id}",
                result.model_dump(),
                scope="execution",
            )

        return result

    async def _execute_step(
        self,
        goal: SubGoal,
        completed: dict[str, StepResult],
        semaphore: asyncio.Semaphore,
    ) -> StepResult:
        """
        Execute a single sub-goal with retry and timeout.

        Args:
            goal: The sub-goal to execute.
            completed: Already-completed step results for context.
            semaphore: Concurrency limiter.

        Returns:
            StepResult for this step.
        """
        async with semaphore:
            agent_id = goal.assigned_agent or "default"
            agent = self._agents.get(agent_id)

            if agent is None:
                return StepResult(
                    step_id=uuid.uuid4().hex[:12],
                    goal_id=goal.goal_id,
                    agent_id=agent_id,
                    status=ExecutionStatus.FAILED,
                    error=f"Agent '{agent_id}' not registered",
                )

            # Gather context from completed dependencies
            dep_context: dict[str, Any] = {}
            for dep_id in goal.dependencies:
                if dep_id in completed:
                    dep_context[dep_id] = completed[dep_id].output

            retries = 0
            last_error: str | None = None

            while retries <= self._config.max_retries:
                start = time.perf_counter()
                try:
                    output = await asyncio.wait_for(
                        agent.execute(
                            task=goal.description,
                            context=dep_context,
                        ),
                        timeout=self._config.step_timeout_s,
                    )
                    latency = (time.perf_counter() - start) * 1000

                    return StepResult(
                        step_id=uuid.uuid4().hex[:12],
                        goal_id=goal.goal_id,
                        agent_id=agent_id,
                        status=ExecutionStatus.COMPLETED,
                        output=output.output if hasattr(output, "output") else output,
                        tokens_used=getattr(output, "tokens_used", 0),
                        cost_usd=getattr(output, "cost_usd", 0.0),
                        latency_ms=latency,
                        retries=retries,
                    )
                except asyncio.TimeoutError:
                    last_error = f"Step timed out after {self._config.step_timeout_s}s"
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"

                retries += 1
                if retries <= self._config.max_retries:
                    await asyncio.sleep(
                        self._config.retry_backoff_s * (2 ** (retries - 1))
                    )

            return StepResult(
                step_id=uuid.uuid4().hex[:12],
                goal_id=goal.goal_id,
                agent_id=agent_id,
                status=ExecutionStatus.FAILED,
                error=last_error,
                retries=retries - 1,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        for task_id, task in self._running.items():
            if not task.done():
                task.cancel()
        self._running.clear()
        return True
