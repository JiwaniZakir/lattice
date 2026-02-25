"""Tests for the DAG execution engine."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lattice.agents.base import AgentResult, BaseAgent
from lattice.core.executor import Executor, ExecutorConfig, ExecutionStatus
from lattice.core.planner import Plan, PlanningStrategy, SubGoal


# -- Mock agents -------------------------------------------------------------

class SuccessAgent(BaseAgent):
    """Agent that always succeeds."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=f"Done: {task}",
            tokens_used=10,
            cost_usd=0.001,
        )


class FailAgent(BaseAgent):
    """Agent that always fails."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        raise RuntimeError("Intentional failure")


class SlowAgent(BaseAgent):
    """Agent that takes time to respond."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        await asyncio.sleep(0.1)
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=f"Slow done: {task}",
            tokens_used=5,
            cost_usd=0.0005,
        )


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def success_agent() -> SuccessAgent:
    return SuccessAgent(agent_id="success")


@pytest.fixture
def fail_agent() -> FailAgent:
    return FailAgent(agent_id="fail")


@pytest.fixture
def executor(success_agent: SuccessAgent) -> Executor:
    return Executor(
        config=ExecutorConfig(
            max_concurrent=3,
            enable_verification=False,
            enable_checkpointing=False,
            max_retries=0,
        ),
        agents={"success": success_agent},
    )


# -- Tests -------------------------------------------------------------------

class TestExecutorBasic:
    @pytest.mark.asyncio
    async def test_execute_single_step(self, executor: Executor) -> None:
        plan = Plan(
            task="Simple task",
            strategy=PlanningStrategy.REACT,
            sub_goals=[
                SubGoal(goal_id="s1", description="Do it", assigned_agent="success"),
            ],
        )
        result = await executor.execute(plan)
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.step_results) == 1
        assert result.step_results[0].status == ExecutionStatus.COMPLETED
        assert "Done" in str(result.step_results[0].output)

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self, executor: Executor) -> None:
        plan = Plan(task="Empty", strategy=PlanningStrategy.REACT, sub_goals=[])
        result = await executor.execute(plan)
        assert result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cost_tracking(self, executor: Executor) -> None:
        plan = Plan(
            task="Cost test",
            strategy=PlanningStrategy.REACT,
            sub_goals=[
                SubGoal(goal_id="s1", description="Step 1", assigned_agent="success"),
                SubGoal(goal_id="s2", description="Step 2", assigned_agent="success"),
            ],
        )
        result = await executor.execute(plan)
        assert result.total_tokens == 20
        assert result.total_cost_usd > 0


class TestExecutorDependencies:
    @pytest.mark.asyncio
    async def test_sequential_dependencies(self, executor: Executor) -> None:
        plan = Plan(
            task="Sequential",
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=[
                SubGoal(goal_id="s1", description="First", assigned_agent="success"),
                SubGoal(goal_id="s2", description="Second", assigned_agent="success", dependencies=["s1"]),
                SubGoal(goal_id="s3", description="Third", assigned_agent="success", dependencies=["s2"]),
            ],
        )
        result = await executor.execute(plan)
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.step_results) == 3

    @pytest.mark.asyncio
    async def test_parallel_execution(self) -> None:
        """Independent steps should execute in parallel."""
        slow = SlowAgent(agent_id="slow")
        executor = Executor(
            config=ExecutorConfig(
                max_concurrent=5,
                enable_verification=False,
                enable_checkpointing=False,
                max_retries=0,
            ),
            agents={"slow": slow},
        )
        plan = Plan(
            task="Parallel",
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=[
                SubGoal(goal_id=f"s{i}", description=f"Step {i}", assigned_agent="slow")
                for i in range(3)
            ],
        )
        result = await executor.execute(plan)
        assert result.status == ExecutionStatus.COMPLETED
        # Parallel execution should be much faster than sequential
        # 3 x 100ms sequential = 300ms; parallel should be ~100ms
        assert result.total_latency_ms < 500


class TestExecutorErrors:
    @pytest.mark.asyncio
    async def test_agent_failure(self, fail_agent: FailAgent) -> None:
        executor = Executor(
            config=ExecutorConfig(
                enable_verification=False,
                enable_checkpointing=False,
                max_retries=0,
            ),
            agents={"fail": fail_agent},
        )
        plan = Plan(
            task="Failing task",
            strategy=PlanningStrategy.REACT,
            sub_goals=[
                SubGoal(goal_id="s1", description="Will fail", assigned_agent="fail"),
            ],
        )
        result = await executor.execute(plan)
        assert result.status == ExecutionStatus.FAILED
        assert result.step_results[0].error is not None

    @pytest.mark.asyncio
    async def test_missing_agent(self, executor: Executor) -> None:
        plan = Plan(
            task="Missing agent",
            strategy=PlanningStrategy.REACT,
            sub_goals=[
                SubGoal(goal_id="s1", description="No agent", assigned_agent="nonexistent"),
            ],
        )
        result = await executor.execute(plan)
        assert result.status == ExecutionStatus.FAILED
        assert "not registered" in result.step_results[0].error

    @pytest.mark.asyncio
    async def test_fail_fast_mode(
        self, success_agent: SuccessAgent, fail_agent: FailAgent
    ) -> None:
        executor = Executor(
            config=ExecutorConfig(
                enable_verification=False,
                enable_checkpointing=False,
                fail_fast=True,
                max_retries=0,
            ),
            agents={"success": success_agent, "fail": fail_agent},
        )
        plan = Plan(
            task="Fail fast",
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=[
                SubGoal(goal_id="s1", description="Fails", assigned_agent="fail"),
                SubGoal(goal_id="s2", description="Never runs", assigned_agent="success"),
            ],
        )
        result = await executor.execute(plan)
        assert result.status == ExecutionStatus.FAILED


class TestExecutorRetries:
    @pytest.mark.asyncio
    async def test_retries_on_failure(self, fail_agent: FailAgent) -> None:
        executor = Executor(
            config=ExecutorConfig(
                enable_verification=False,
                enable_checkpointing=False,
                max_retries=2,
                retry_backoff_s=0.01,
            ),
            agents={"fail": fail_agent},
        )
        plan = Plan(
            task="Retry test",
            strategy=PlanningStrategy.REACT,
            sub_goals=[
                SubGoal(goal_id="s1", description="Will retry", assigned_agent="fail"),
            ],
        )
        result = await executor.execute(plan)
        assert result.step_results[0].retries == 2


class TestExecutorRegistration:
    def test_register_agent(self) -> None:
        executor = Executor(config=ExecutorConfig())
        agent = SuccessAgent(agent_id="new")
        executor.register_agent("new", agent)
        assert "new" in executor._agents
