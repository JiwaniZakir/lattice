"""Tests for the hybrid planner (ReAct + Plan-and-Solve)."""

from __future__ import annotations

import pytest

from lattice.core.planner import (
    Plan,
    Planner,
    PlannerConfig,
    PlanningStrategy,
    SubGoal,
    ThoughtType,
)


@pytest.fixture
def planner() -> Planner:
    return Planner(config=PlannerConfig(
        strategy=PlanningStrategy.HYBRID,
        enable_skill_cache=True,
    ))


class TestPlannerConfig:
    def test_default_config(self) -> None:
        config = PlannerConfig()
        assert config.strategy == PlanningStrategy.HYBRID
        assert config.max_react_steps == 10
        assert config.enable_skill_cache is True

    def test_custom_config(self) -> None:
        config = PlannerConfig(
            strategy=PlanningStrategy.REACT,
            max_react_steps=5,
        )
        assert config.strategy == PlanningStrategy.REACT
        assert config.max_react_steps == 5


class TestPlannerPlanning:
    @pytest.mark.asyncio
    async def test_simple_task_uses_react(self, planner: Planner) -> None:
        """Simple tasks (low complexity) should use ReAct."""
        plan = await planner.plan(
            task="Summarize this text",
            available_agents=["agent_a"],
        )
        # Low complexity => should use ReAct
        assert plan.strategy == PlanningStrategy.REACT
        assert len(plan.sub_goals) >= 1
        assert plan.task == "Summarize this text"

    @pytest.mark.asyncio
    async def test_complex_task_uses_plan_and_solve(self, planner: Planner) -> None:
        """Complex tasks should use Plan-and-Solve decomposition."""
        plan = await planner.plan(
            task=(
                "Research and analyze multiple papers on multi-agent systems, "
                "then synthesize the findings and additionally create a "
                "comparison table, then also write a summary and furthermore "
                "evaluate each approach"
            ),
            available_agents=["researcher", "analyst"],
        )
        assert plan.strategy == PlanningStrategy.PLAN_AND_SOLVE
        assert len(plan.sub_goals) >= 1

    @pytest.mark.asyncio
    async def test_react_plan_has_trace(self, planner: Planner) -> None:
        plan = await planner.plan(
            task="Hello",
            available_agents=["agent_a"],
        )
        assert len(plan.react_trace) >= 1
        assert plan.react_trace[0].step_type == ThoughtType.THOUGHT

    @pytest.mark.asyncio
    async def test_plan_has_timing(self, planner: Planner) -> None:
        plan = await planner.plan(task="Test task", available_agents=["a"])
        assert plan.planning_time_ms > 0


class TestPlannerCaching:
    @pytest.mark.asyncio
    async def test_skill_cache_hit(self, planner: Planner) -> None:
        """Same task should be served from cache on second call."""
        task = "identical task for caching"
        plan1 = await planner.plan(task=task, available_agents=["a"])
        assert plan1.cached is False

        plan2 = await planner.plan(task=task, available_agents=["a"])
        assert plan2.cached is True

    @pytest.mark.asyncio
    async def test_cache_ignores_case(self, planner: Planner) -> None:
        task1 = "Cache Test Task"
        task2 = "cache test task"
        await planner.plan(task=task1, available_agents=["a"])
        plan2 = await planner.plan(task=task2, available_agents=["a"])
        assert plan2.cached is True

    def test_clear_cache(self, planner: Planner) -> None:
        count = planner.clear_cache()
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_cache_after_planning(self, planner: Planner) -> None:
        await planner.plan(task="task 1", available_agents=["a"])
        await planner.plan(task="task 2", available_agents=["a"])
        count = planner.clear_cache()
        assert count == 2


class TestComplexityEstimation:
    @pytest.mark.asyncio
    async def test_short_task_low_complexity(self, planner: Planner) -> None:
        complexity = await planner._estimate_complexity("Summarize", None)
        assert complexity < 0.6

    @pytest.mark.asyncio
    async def test_long_task_with_indicators(self, planner: Planner) -> None:
        task = (
            "First research this and then analyze that and also "
            "compare every approach furthermore additionally"
        )
        complexity = await planner._estimate_complexity(task, None)
        assert complexity > 0.3


class TestSubGoalParsing:
    def test_parse_numbered_list(self, planner: Planner) -> None:
        text = "1. Research papers\n2. Analyze findings\n3. Write report"
        goals = planner._parse_sub_goals(text)
        assert len(goals) == 3
        assert goals[0].description == "Research papers"

    def test_parse_empty_text(self, planner: Planner) -> None:
        goals = planner._parse_sub_goals("")
        assert len(goals) == 1  # Falls back to single goal

    def test_parse_react_trace(self, planner: Planner) -> None:
        text = (
            "Thought: I need to research this topic\n"
            "Action: Search for papers\n"
            "Observation: Found 5 relevant papers\n"
        )
        steps = planner._parse_react_trace(text)
        assert len(steps) == 3
        assert steps[0].step_type == ThoughtType.THOUGHT
        assert steps[1].step_type == ThoughtType.ACTION
        assert steps[2].step_type == ThoughtType.OBSERVATION


class TestPlanModel:
    def test_plan_serialization(self) -> None:
        plan = Plan(
            task="test",
            strategy=PlanningStrategy.REACT,
            sub_goals=[SubGoal(description="step 1")],
        )
        data = plan.model_dump()
        assert data["task"] == "test"
        assert len(data["sub_goals"]) == 1

    def test_sub_goal_defaults(self) -> None:
        goal = SubGoal(description="test goal")
        assert goal.status == "pending"
        assert goal.dependencies == []
        assert goal.assigned_agent is None
        assert goal.estimated_complexity == 0.5
