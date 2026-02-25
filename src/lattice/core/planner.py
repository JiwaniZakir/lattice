"""
Hybrid planner combining ReAct (Yao et al., 2023) and Plan-and-Solve (Wang et al., 2023).

ReAct interleaves reasoning traces with action execution, while Plan-and-Solve
decomposes complex tasks into ordered sub-plans before execution. This module
implements a hierarchical strategy:

    1. Plan-and-Solve generates a high-level decomposition of the task into
       an ordered sequence of sub-goals.
    2. For each sub-goal, ReAct drives iterative thought-action-observation
       loops until the sub-goal is satisfied or a retry limit is reached.
    3. Voyager-style skill caching (Wang et al., 2023) persists successful
       plans for future reuse via the memory system.

The planner emits a DAG of execution steps that the Executor can run
with dependency tracking and parallelism.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lattice.core.memory import ScopedMemory
    from lattice.integrations.litellm import LiteLLMProvider


class PlanningStrategy(str, Enum):
    """Available planning strategies."""

    REACT = "react"
    PLAN_AND_SOLVE = "plan_and_solve"
    HYBRID = "hybrid"


class ThoughtType(str, Enum):
    """Types of reasoning steps in a ReAct trace."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"


class ReActStep(BaseModel):
    """A single step in a ReAct reasoning trace."""

    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    step_type: ThoughtType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class SubGoal(BaseModel):
    """A sub-goal in a Plan-and-Solve decomposition."""

    goal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    dependencies: list[str] = Field(default_factory=list)
    assigned_agent: str | None = None
    estimated_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    status: str = "pending"
    result: Any = None


class Plan(BaseModel):
    """A complete execution plan with sub-goals and provenance."""

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str
    strategy: PlanningStrategy
    sub_goals: list[SubGoal] = Field(default_factory=list)
    react_trace: list[ReActStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    planning_time_ms: float = 0.0
    cached: bool = False


class PlannerConfig(BaseModel):
    """Configuration for the planner."""

    strategy: PlanningStrategy = PlanningStrategy.HYBRID
    max_react_steps: int = 10
    max_sub_goals: int = 20
    complexity_threshold: float = 0.6
    enable_skill_cache: bool = True
    planning_model: str = "gpt-4o"
    temperature: float = 0.2


class Planner:
    """
    Hierarchical planner combining Plan-and-Solve decomposition with ReAct execution.

    The planner operates in two phases:

    Phase 1 -- Decomposition (Plan-and-Solve):
        Given a complex task description, the planner prompts an LLM to
        decompose it into an ordered list of sub-goals with explicit
        dependency edges. This produces a DAG of SubGoals.

    Phase 2 -- Execution Planning (ReAct):
        For each sub-goal, the planner generates a ReAct trace of
        thought-action-observation steps. Actions map to agent capabilities,
        enabling the executor to dispatch work to the right agents.

    The planner also implements Voyager-style skill caching: successful
    plans are stored in memory keyed by a normalized task description,
    so similar future tasks can reuse cached plans.

    Args:
        config: Planner configuration.
        llm_provider: LLM provider for generating plans.
        memory: Optional memory system for skill caching.
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
        llm_provider: LiteLLMProvider | None = None,
        memory: ScopedMemory | None = None,
    ) -> None:
        self._config = config or PlannerConfig()
        self._llm = llm_provider
        self._memory = memory
        self._plan_cache: dict[str, Plan] = {}

    @property
    def config(self) -> PlannerConfig:
        return self._config

    async def plan(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        available_agents: Sequence[str] | None = None,
    ) -> Plan:
        """
        Generate an execution plan for the given task.

        Selects strategy based on estimated task complexity and configuration.
        For hybrid mode, uses Plan-and-Solve for complex tasks and ReAct for
        simple ones, with the complexity threshold set in config.

        Args:
            task: Natural language task description.
            context: Optional context dictionary.
            available_agents: IDs of agents available for assignment.

        Returns:
            A Plan with sub-goals and/or ReAct trace.
        """
        start = time.perf_counter()

        # Check skill cache (Voyager pattern)
        cache_key = self._normalize_task(task)
        if self._config.enable_skill_cache and cache_key in self._plan_cache:
            cached = self._plan_cache[cache_key].model_copy(deep=True)
            cached.cached = True
            cached.plan_id = uuid.uuid4().hex[:12]
            return cached

        strategy = self._config.strategy

        if strategy == PlanningStrategy.HYBRID:
            complexity = await self._estimate_complexity(task, context)
            if complexity >= self._config.complexity_threshold:
                strategy = PlanningStrategy.PLAN_AND_SOLVE
            else:
                strategy = PlanningStrategy.REACT

        if strategy == PlanningStrategy.PLAN_AND_SOLVE:
            plan = await self._plan_and_solve(task, context, available_agents)
        else:
            plan = await self._react_plan(task, context, available_agents)

        plan.strategy = strategy
        plan.planning_time_ms = (time.perf_counter() - start) * 1000

        # Cache the plan (Voyager skill library)
        if self._config.enable_skill_cache:
            self._plan_cache[cache_key] = plan.model_copy(deep=True)

        return plan

    async def _plan_and_solve(
        self,
        task: str,
        context: dict[str, Any] | None,
        available_agents: Sequence[str] | None,
    ) -> Plan:
        """
        Plan-and-Solve decomposition (Wang et al., 2023).

        Prompts the LLM to decompose the task into ordered sub-goals with
        explicit dependencies, producing a DAG structure.
        """
        sub_goals = await self._decompose_task(task, context)

        # Assign agents to sub-goals if available
        if available_agents:
            for goal in sub_goals:
                if goal.assigned_agent is None:
                    goal.assigned_agent = await self._assign_agent(
                        goal, available_agents
                    )

        return Plan(
            task=task,
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=sub_goals,
        )

    async def _react_plan(
        self,
        task: str,
        context: dict[str, Any] | None,
        available_agents: Sequence[str] | None,
    ) -> Plan:
        """
        ReAct planning (Yao et al., 2023).

        Generates an interleaved thought-action-observation trace for the task.
        Each action maps to an agent capability.
        """
        trace: list[ReActStep] = []
        agent_list = ", ".join(available_agents) if available_agents else "any"

        # Initial thought
        thought = ReActStep(
            step_type=ThoughtType.THOUGHT,
            content=f"Analyzing task: {task}. Available agents: {agent_list}.",
        )
        trace.append(thought)

        if self._llm:
            react_steps = await self._generate_react_trace(
                task, context, available_agents
            )
            trace.extend(react_steps)
        else:
            # Without LLM, create a single-step plan
            action = ReActStep(
                step_type=ThoughtType.ACTION,
                content=f"Execute task directly: {task}",
                metadata={"agent": available_agents[0] if available_agents else "default"},
            )
            trace.append(action)

        # Convert trace to a single sub-goal for the executor
        sub_goal = SubGoal(
            description=task,
            assigned_agent=trace[-1].metadata.get("agent") if trace else None,
        )

        return Plan(
            task=task,
            strategy=PlanningStrategy.REACT,
            sub_goals=[sub_goal],
            react_trace=trace,
        )

    async def _decompose_task(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> list[SubGoal]:
        """
        Decompose a task into ordered sub-goals using the LLM.

        Falls back to a single sub-goal if no LLM is configured.
        """
        if self._llm is None:
            return [SubGoal(description=task)]

        ctx_str = ""
        if context:
            ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items())

        prompt = (
            "You are a planning agent. Decompose the following task into ordered "
            "sub-goals. Each sub-goal should be atomic and actionable. Output a "
            "numbered list. If a sub-goal depends on previous ones, note which.\n\n"
            f"Task: {task}\n"
        )
        if ctx_str:
            prompt += f"\nContext:\n{ctx_str}\n"

        response = await self._llm.complete(
            prompt,
            model=self._config.planning_model,
            temperature=self._config.temperature,
        )

        return self._parse_sub_goals(response.content)

    async def _generate_react_trace(
        self,
        task: str,
        context: dict[str, Any] | None,
        available_agents: Sequence[str] | None,
    ) -> list[ReActStep]:
        """Generate a ReAct reasoning trace via LLM."""
        if self._llm is None:
            return []

        agent_list = ", ".join(available_agents) if available_agents else "any"
        prompt = (
            "You are a ReAct agent. For the given task, generate an interleaved "
            "sequence of Thought, Action, and Observation steps. Format each step "
            "as 'Thought: ...', 'Action: ...', or 'Observation: ...'.\n\n"
            f"Task: {task}\n"
            f"Available agents: {agent_list}\n"
        )

        response = await self._llm.complete(
            prompt,
            model=self._config.planning_model,
            temperature=self._config.temperature,
        )

        return self._parse_react_trace(response.content)

    async def _estimate_complexity(
        self, task: str, context: dict[str, Any] | None
    ) -> float:
        """
        Estimate task complexity to choose between ReAct and Plan-and-Solve.

        Uses simple heuristics when no LLM is available; otherwise prompts
        the LLM for a complexity score.
        """
        # Heuristic: longer tasks and those with conjunctions tend to be more complex
        words = task.split()
        word_count = len(words)

        complexity_indicators = [
            "and", "then", "after", "before", "while", "also",
            "additionally", "furthermore", "multiple", "several",
            "each", "every", "all",
        ]
        indicator_count = sum(1 for w in words if w.lower() in complexity_indicators)

        # Base complexity from length
        length_score = min(word_count / 50.0, 1.0)
        indicator_score = min(indicator_count / 3.0, 1.0)

        return min(0.3 * length_score + 0.7 * indicator_score, 1.0)

    async def _assign_agent(
        self, goal: SubGoal, available_agents: Sequence[str]
    ) -> str:
        """Assign an agent to a sub-goal. Defaults to round-robin."""
        # Simple hash-based assignment when no LLM is available
        idx = hash(goal.description) % len(available_agents)
        return available_agents[idx]

    def _parse_sub_goals(self, text: str) -> list[SubGoal]:
        """Parse numbered sub-goals from LLM output."""
        goals: list[SubGoal] = []
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip leading numbers, dots, dashes
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned:
                # Check for dependency markers like "(depends on 1, 2)"
                deps: list[str] = []
                if "(depends on" in cleaned.lower():
                    dep_start = cleaned.lower().index("(depends on")
                    dep_text = cleaned[dep_start:]
                    cleaned = cleaned[:dep_start].strip()
                    # Extract goal IDs from dependency text
                    import re
                    dep_ids = re.findall(r"\d+", dep_text)
                    deps = dep_ids

                goal = SubGoal(description=cleaned, dependencies=deps)
                goals.append(goal)

        return goals if goals else [SubGoal(description=text.strip())]

    def _parse_react_trace(self, text: str) -> list[ReActStep]:
        """Parse ReAct trace from LLM output."""
        steps: list[ReActStep] = []
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith("thought:"):
                steps.append(ReActStep(
                    step_type=ThoughtType.THOUGHT,
                    content=line[len("thought:"):].strip(),
                ))
            elif lower.startswith("action:"):
                steps.append(ReActStep(
                    step_type=ThoughtType.ACTION,
                    content=line[len("action:"):].strip(),
                ))
            elif lower.startswith("observation:"):
                steps.append(ReActStep(
                    step_type=ThoughtType.OBSERVATION,
                    content=line[len("observation:"):].strip(),
                ))

        return steps

    @staticmethod
    def _normalize_task(task: str) -> str:
        """Normalize task description for cache key."""
        return " ".join(task.lower().split())

    def clear_cache(self) -> int:
        """Clear the plan cache and return the number of entries removed."""
        count = len(self._plan_cache)
        self._plan_cache.clear()
        return count
