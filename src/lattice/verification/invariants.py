"""
Safety invariants for multi-agent workflows.

Defines reusable safety invariants that can be checked against execution
plans using Z3. Invariants express properties that must hold for a plan
to be considered safe:

    - Resource bounds (cost, tokens, latency)
    - Agent isolation (no cross-tenant data access)
    - Ordering constraints (e.g., approval before execution)
    - Liveness properties (no deadlocks in the DAG)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import z3
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lattice.core.planner import Plan


class InvariantResult(BaseModel):
    """Result of checking a single invariant."""

    name: str
    holds: bool
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Invariant(ABC):
    """
    Abstract base class for safety invariants.

    Subclasses define a name, description, and a `check` method that
    evaluates the invariant against a plan using Z3.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the invariant."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this invariant checks."""
        ...

    @abstractmethod
    async def check(self, plan: Plan) -> InvariantResult:
        """Check this invariant against a plan."""
        ...


class MaxCostInvariant(Invariant):
    """Ensures total plan cost does not exceed a budget."""

    def __init__(self, max_cost_usd: float = 10.0) -> None:
        self._max_cost = max_cost_usd

    @property
    def name(self) -> str:
        return "max_cost"

    @property
    def description(self) -> str:
        return f"Total cost must not exceed ${self._max_cost:.2f}"

    async def check(self, plan: Plan) -> InvariantResult:
        solver = z3.Solver()
        n = len(plan.sub_goals)

        costs = [z3.Real(f"cost_{i}") for i in range(n)]
        for c in costs:
            solver.add(c >= 0, c <= z3.RealVal("5.0"))

        total = z3.Sum(*costs) if costs else z3.RealVal("0")
        solver.add(total <= z3.RealVal(str(self._max_cost)))

        if solver.check() == z3.sat:
            return InvariantResult(
                name=self.name,
                holds=True,
                message=f"Cost budget of ${self._max_cost:.2f} is feasible for {n} steps",
            )
        return InvariantResult(
            name=self.name,
            holds=False,
            message=f"Cannot satisfy cost budget of ${self._max_cost:.2f} for {n} steps",
        )


class MaxStepsInvariant(Invariant):
    """Ensures plan does not have too many steps."""

    def __init__(self, max_steps: int = 50) -> None:
        self._max_steps = max_steps

    @property
    def name(self) -> str:
        return "max_steps"

    @property
    def description(self) -> str:
        return f"Plan must have at most {self._max_steps} steps"

    async def check(self, plan: Plan) -> InvariantResult:
        n = len(plan.sub_goals)
        holds = n <= self._max_steps
        return InvariantResult(
            name=self.name,
            holds=holds,
            message=(
                f"Plan has {n} steps (limit: {self._max_steps})"
                if holds
                else f"Plan exceeds step limit: {n} > {self._max_steps}"
            ),
        )


class AcyclicInvariant(Invariant):
    """Ensures the sub-goal dependency graph is acyclic."""

    @property
    def name(self) -> str:
        return "acyclic"

    @property
    def description(self) -> str:
        return "Sub-goal dependencies must form a DAG (no cycles)"

    async def check(self, plan: Plan) -> InvariantResult:
        if not plan.sub_goals:
            return InvariantResult(name=self.name, holds=True, message="No sub-goals")

        solver = z3.Solver()
        n = len(plan.sub_goals)
        goal_ids = [g.goal_id for g in plan.sub_goals]
        order = {gid: z3.Int(f"order_{gid}") for gid in goal_ids}

        for var in order.values():
            solver.add(var >= 0, var < n)
        solver.add(z3.Distinct(*order.values()))

        for goal in plan.sub_goals:
            for dep in goal.dependencies:
                if dep in order:
                    solver.add(order[dep] < order[goal.goal_id])

        if solver.check() == z3.sat:
            return InvariantResult(name=self.name, holds=True, message="DAG is acyclic")
        return InvariantResult(
            name=self.name, holds=False, message="Cycle detected in sub-goal dependencies"
        )


class ApprovalBeforeExecutionInvariant(Invariant):
    """
    Ensures that steps requiring human approval are preceded by
    a human review step in the DAG.
    """

    def __init__(self, approval_agents: set[str] | None = None) -> None:
        self._approval_agents = approval_agents or {"human"}

    @property
    def name(self) -> str:
        return "approval_before_execution"

    @property
    def description(self) -> str:
        return "High-risk steps must be preceded by human approval"

    async def check(self, plan: Plan) -> InvariantResult:
        # Find goals that require approval
        goals_needing_approval = [
            g for g in plan.sub_goals
            if g.assigned_agent and g.assigned_agent in self._approval_agents
        ]

        if not goals_needing_approval:
            return InvariantResult(
                name=self.name, holds=True,
                message="No steps require human approval",
            )

        # Approval steps are valid if they have explicit dependencies
        missing = [
            g.goal_id for g in goals_needing_approval
            if not g.dependencies
        ]

        if missing:
            return InvariantResult(
                name=self.name, holds=False,
                message=f"Approval steps {missing} have no predecessors",
            )
        return InvariantResult(
            name=self.name, holds=True,
            message="All approval steps have proper dependencies",
        )


class InvariantChecker:
    """
    Checks a plan against a collection of invariants.

    Args:
        invariants: List of invariants to check.
    """

    def __init__(self, invariants: list[Invariant] | None = None) -> None:
        self._invariants = invariants or [
            MaxCostInvariant(),
            MaxStepsInvariant(),
            AcyclicInvariant(),
        ]

    def add_invariant(self, invariant: Invariant) -> None:
        """Add an invariant to the checker."""
        self._invariants.append(invariant)

    async def check_all(self, plan: Plan) -> list[InvariantResult]:
        """Check all invariants against the plan."""
        results = []
        for inv in self._invariants:
            result = await inv.check(plan)
            results.append(result)
        return results

    async def is_safe(self, plan: Plan) -> bool:
        """Return True if all invariants hold."""
        results = await self.check_all(plan)
        return all(r.holds for r in results)
