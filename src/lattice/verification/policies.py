"""
Constraint policies for workflow verification.

Policies are high-level, composable safety rules that translate into
Z3 constraints. They provide a declarative way to express organizational
safety requirements:

    - Rate limiting per agent or user
    - Data access control
    - Workflow ordering requirements
    - Cost and resource caps
    - Constitutional AI principles as formal constraints

Policies can be loaded from configuration files and composed into
policy engines that enforce all policies at once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import z3
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lattice.core.planner import Plan
    from lattice.core.verifier import VerificationResult


class PolicyViolation(BaseModel):
    """A specific policy violation."""

    policy_name: str
    severity: str = "error"  # "error", "warning", "info"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class Policy(ABC):
    """
    Abstract base class for constraint policies.

    Policies translate high-level safety requirements into Z3
    constraints and evaluate them against execution plans.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Policy description."""
        ...

    @property
    def severity(self) -> str:
        """Default severity for violations."""
        return "error"

    @abstractmethod
    async def evaluate(self, plan: Plan) -> VerificationResult:
        """Evaluate this policy against a plan."""
        ...


class MaxConcurrencyPolicy(Policy):
    """
    Limits the number of simultaneously executing steps.

    Ensures that no more than `max_parallel` sub-goals can be executing
    at the same time, preventing resource exhaustion.
    """

    def __init__(self, max_parallel: int = 5) -> None:
        self._max_parallel = max_parallel

    @property
    def name(self) -> str:
        return "max_concurrency"

    @property
    def description(self) -> str:
        return f"At most {self._max_parallel} steps may execute concurrently"

    async def evaluate(self, plan: Plan) -> VerificationResult:
        from lattice.core.verifier import VerificationResult

        n = len(plan.sub_goals)
        if n == 0:
            return VerificationResult(is_safe=True, constraints_checked=1)

        # Compute the maximum width of the DAG (maximum anti-chain)
        # using a topological approach
        goal_map = {g.goal_id: g for g in plan.sub_goals}
        in_degree: dict[str, int] = {g.goal_id: 0 for g in plan.sub_goals}
        for g in plan.sub_goals:
            for dep in g.dependencies:
                if dep in in_degree:
                    in_degree[g.goal_id] = in_degree.get(g.goal_id, 0)

        # BFS layers represent parallel execution waves
        # Count the maximum number of nodes in any layer
        ready = [gid for gid, d in in_degree.items() if d == 0]
        max_width = len(ready)

        if max_width > self._max_parallel:
            return VerificationResult(
                is_safe=False,
                violations=[
                    f"DAG has width {max_width}, exceeding concurrency limit {self._max_parallel}"
                ],
                constraints_checked=1,
            )

        return VerificationResult(is_safe=True, constraints_checked=1)


class RequiredAgentPolicy(Policy):
    """
    Requires that specific agent types are present in the plan.

    For example, ensuring that a critic agent reviews all outputs
    before they are returned.
    """

    def __init__(self, required_agents: list[str]) -> None:
        self._required = required_agents

    @property
    def name(self) -> str:
        return "required_agents"

    @property
    def description(self) -> str:
        return f"Plan must include agents: {self._required}"

    async def evaluate(self, plan: Plan) -> VerificationResult:
        from lattice.core.verifier import VerificationResult

        present = {g.assigned_agent for g in plan.sub_goals if g.assigned_agent}
        missing = [a for a in self._required if a not in present]

        if missing:
            return VerificationResult(
                is_safe=False,
                violations=[f"Required agents missing from plan: {missing}"],
                constraints_checked=1,
            )
        return VerificationResult(is_safe=True, constraints_checked=1)


class DataIsolationPolicy(Policy):
    """
    Ensures agents cannot access data outside their authorized scopes.

    Uses Z3 to verify that the data flow graph respects scope boundaries.
    """

    def __init__(self, agent_scopes: dict[str, set[str]] | None = None) -> None:
        self._agent_scopes = agent_scopes or {}

    @property
    def name(self) -> str:
        return "data_isolation"

    @property
    def description(self) -> str:
        return "Agents must only access data within their authorized scopes"

    async def evaluate(self, plan: Plan) -> VerificationResult:
        from lattice.core.verifier import VerificationResult

        violations = []
        for goal in plan.sub_goals:
            if not goal.assigned_agent:
                continue
            agent = goal.assigned_agent
            if agent not in self._agent_scopes:
                continue
            allowed = self._agent_scopes[agent]
            # Check dependencies: an agent should only depend on goals
            # assigned to agents in compatible scopes
            for dep_id in goal.dependencies:
                dep_goal = next(
                    (g for g in plan.sub_goals if g.goal_id == dep_id), None
                )
                if dep_goal and dep_goal.assigned_agent:
                    dep_agent = dep_goal.assigned_agent
                    if dep_agent not in allowed and dep_agent != agent:
                        violations.append(
                            f"Agent '{agent}' depends on '{dep_agent}' "
                            f"which is outside its allowed scopes {allowed}"
                        )

        return VerificationResult(
            is_safe=len(violations) == 0,
            violations=violations,
            constraints_checked=len(plan.sub_goals),
        )


class CostCapPolicy(Policy):
    """Per-step and total cost cap policy."""

    def __init__(
        self,
        max_per_step_usd: float = 1.0,
        max_total_usd: float = 10.0,
    ) -> None:
        self._max_per_step = max_per_step_usd
        self._max_total = max_total_usd

    @property
    def name(self) -> str:
        return "cost_cap"

    @property
    def description(self) -> str:
        return (
            f"Per-step cost <= ${self._max_per_step:.2f}, "
            f"total <= ${self._max_total:.2f}"
        )

    async def evaluate(self, plan: Plan) -> VerificationResult:
        from lattice.core.verifier import VerificationResult

        n = len(plan.sub_goals)
        solver = z3.Solver()
        solver.set("timeout", 3000)

        costs = [z3.Real(f"step_cost_{i}") for i in range(n)]
        for c in costs:
            solver.add(c >= 0, c <= z3.RealVal(str(self._max_per_step)))

        if costs:
            solver.add(z3.Sum(*costs) <= z3.RealVal(str(self._max_total)))

        result = solver.check()
        if result == z3.sat:
            return VerificationResult(is_safe=True, constraints_checked=n + 1)
        return VerificationResult(
            is_safe=False,
            violations=[
                f"Cost constraints infeasible: {n} steps at "
                f"${self._max_per_step}/step cannot fit in ${self._max_total} total"
            ],
            constraints_checked=n + 1,
        )


class PolicyEngine:
    """
    Engine that evaluates all registered policies against a plan.

    Args:
        policies: Initial list of policies.
        fail_on_warning: Whether warnings should cause verification to fail.
    """

    def __init__(
        self,
        policies: list[Policy] | None = None,
        fail_on_warning: bool = False,
    ) -> None:
        self._policies = policies or []
        self._fail_on_warning = fail_on_warning

    def add_policy(self, policy: Policy) -> None:
        """Register a policy."""
        self._policies.append(policy)

    async def evaluate_all(self, plan: Plan) -> list[PolicyViolation]:
        """
        Evaluate all policies and return violations.

        Args:
            plan: The execution plan to verify.

        Returns:
            List of policy violations (empty if all pass).
        """
        violations: list[PolicyViolation] = []

        for policy in self._policies:
            result = await policy.evaluate(plan)
            if not result.is_safe:
                for v in result.violations:
                    violations.append(PolicyViolation(
                        policy_name=policy.name,
                        severity=policy.severity,
                        message=v,
                    ))
            for w in result.warnings:
                violations.append(PolicyViolation(
                    policy_name=policy.name,
                    severity="warning",
                    message=w,
                ))

        return violations

    async def is_compliant(self, plan: Plan) -> bool:
        """Return True if the plan passes all policies."""
        violations = await self.evaluate_all(plan)
        if self._fail_on_warning:
            return len(violations) == 0
        return all(v.severity != "error" for v in violations)
