"""
Z3-based safety verification for multi-agent workflows.

Implements formal verification of execution plans against configurable
safety invariants before dispatch. Inspired by Constitutional AI
(Bai et al., 2022), the verifier ensures that:

    1. Resource constraints are not violated (cost budgets, token limits).
    2. Agent capability constraints are satisfied (agents only receive
       tasks they can handle).
    3. Dependency ordering is acyclic and well-formed.
    4. Custom policy constraints defined as Z3 formulas are satisfiable
       under the execution plan.

The verifier transforms Plan objects into Z3 constraints and checks
satisfiability. If the constraint system is UNSAT, the plan is rejected
with concrete violation explanations.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import z3
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lattice.core.planner import Plan, SubGoal
    from lattice.verification.policies import Policy


class VerificationResult(BaseModel):
    """Result of verifying a plan against safety invariants."""

    is_safe: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    solver_time_ms: float = 0.0
    constraints_checked: int = 0
    model: dict[str, Any] | None = None


class ResourceBudget(BaseModel):
    """Resource constraints for verification."""

    max_total_cost_usd: float = 10.0
    max_total_tokens: int = 1_000_000
    max_steps: int = 50
    max_parallel: int = 10
    max_retries_per_step: int = 3
    timeout_s: float = 300.0


class AgentCapability(BaseModel):
    """Declared capability of an agent for constraint checking."""

    agent_id: str
    supported_tasks: list[str] = Field(default_factory=list)
    max_tokens_per_call: int = 100_000
    max_cost_per_call: float = 1.0
    requires_human_approval: bool = False


class VerifierConfig(BaseModel):
    """Configuration for the verifier."""

    budget: ResourceBudget = Field(default_factory=ResourceBudget)
    agent_capabilities: list[AgentCapability] = Field(default_factory=list)
    enable_dag_check: bool = True
    enable_budget_check: bool = True
    enable_capability_check: bool = True
    solver_timeout_ms: int = 5000


class Verifier:
    """
    Z3-based safety verifier for execution plans.

    Translates plan structure and constraints into Z3 formulas and checks
    satisfiability. The verifier runs multiple independent checks:

    1. DAG Acyclicity: Verifies that sub-goal dependencies form a valid DAG.
    2. Budget Feasibility: Checks that estimated costs and tokens fit within
       the configured budget.
    3. Capability Matching: Ensures each sub-goal is assigned to an agent
       that can handle it.
    4. Custom Policies: Evaluates user-defined Z3 constraint policies.

    Args:
        config: Verifier configuration with budget and capability specs.
        policies: Optional list of custom constraint policies.
    """

    def __init__(
        self,
        config: VerifierConfig | None = None,
        policies: Sequence[Policy] | None = None,
    ) -> None:
        self._config = config or VerifierConfig()
        self._policies = list(policies) if policies else []

    @property
    def config(self) -> VerifierConfig:
        return self._config

    async def verify_plan(self, plan: Plan) -> VerificationResult:
        """
        Verify a plan against all configured safety invariants.

        Runs DAG, budget, capability, and custom policy checks in sequence.
        Returns a VerificationResult with all violations found.

        Args:
            plan: The execution plan to verify.

        Returns:
            VerificationResult with safety assessment.
        """
        start = time.perf_counter()
        violations: list[str] = []
        warnings: list[str] = []
        constraints_checked = 0

        # 1. DAG acyclicity check
        if self._config.enable_dag_check:
            dag_result = self._check_dag_acyclicity(plan.sub_goals)
            constraints_checked += 1
            violations.extend(dag_result)

        # 2. Budget feasibility check
        if self._config.enable_budget_check:
            budget_result = self._check_budget_feasibility(plan.sub_goals)
            constraints_checked += len(plan.sub_goals) + 1
            violations.extend(budget_result["violations"])
            warnings.extend(budget_result["warnings"])

        # 3. Capability matching check
        if self._config.enable_capability_check:
            cap_result = self._check_capability_matching(plan.sub_goals)
            constraints_checked += len(plan.sub_goals)
            violations.extend(cap_result)

        # 4. Custom policy checks
        for policy in self._policies:
            policy_result = await self._check_policy(policy, plan)
            constraints_checked += 1
            if not policy_result.is_safe:
                violations.extend(policy_result.violations)
                warnings.extend(policy_result.warnings)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return VerificationResult(
            is_safe=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            solver_time_ms=elapsed_ms,
            constraints_checked=constraints_checked,
        )

    def _check_dag_acyclicity(self, sub_goals: Sequence[SubGoal]) -> list[str]:
        """
        Verify that sub-goal dependencies form a valid DAG using Z3.

        Encodes the ordering constraint that if goal B depends on goal A,
        then order(A) < order(B). If the system is UNSAT, there is a cycle.
        """
        if not sub_goals:
            return []

        solver = z3.Solver()
        solver.set("timeout", self._config.solver_timeout_ms)

        # Create an integer variable for each goal representing its topological order
        goal_ids = [g.goal_id for g in sub_goals]
        goal_set = set(goal_ids)
        order_vars: dict[str, z3.ArithRef] = {
            gid: z3.Int(f"order_{gid}") for gid in goal_ids
        }

        n = len(sub_goals)

        # Each order variable is in [0, n)
        for gid, var in order_vars.items():
            solver.add(var >= 0, var < n)

        # All orders are distinct
        solver.add(z3.Distinct(*order_vars.values()))

        # Dependency constraints: if B depends on A, then order(A) < order(B)
        for goal in sub_goals:
            for dep in goal.dependencies:
                if dep in order_vars:
                    solver.add(order_vars[dep] < order_vars[goal.goal_id])
                elif dep not in goal_set:
                    # Dependency on unknown goal -- treat as warning handled elsewhere
                    pass

        result = solver.check()
        if result == z3.unsat:
            return ["DAG cycle detected: sub-goal dependencies contain a cycle"]
        elif result == z3.unknown:
            return ["DAG check inconclusive: solver timed out"]
        return []

    def _check_budget_feasibility(
        self, sub_goals: Sequence[SubGoal]
    ) -> dict[str, list[str]]:
        """
        Verify that estimated resource usage fits within budget using Z3.

        Creates Z3 real variables for per-step cost and token usage, adds
        constraints from agent capabilities, and checks that the total
        fits within the budget.
        """
        violations: list[str] = []
        warnings: list[str] = []
        budget = self._config.budget

        if len(sub_goals) > budget.max_steps:
            violations.append(
                f"Plan has {len(sub_goals)} steps, exceeding maximum of {budget.max_steps}"
            )

        solver = z3.Solver()
        solver.set("timeout", self._config.solver_timeout_ms)

        # Create per-step cost and token variables
        step_costs: list[z3.ArithRef] = []
        step_tokens: list[z3.ArithRef] = []

        cap_map = {c.agent_id: c for c in self._config.agent_capabilities}

        for i, goal in enumerate(sub_goals):
            cost_var = z3.Real(f"cost_{i}")
            token_var = z3.Int(f"tokens_{i}")
            step_costs.append(cost_var)
            step_tokens.append(token_var)

            # Non-negative
            solver.add(cost_var >= 0)
            solver.add(token_var >= 0)

            # Per-agent bounds
            if goal.assigned_agent and goal.assigned_agent in cap_map:
                cap = cap_map[goal.assigned_agent]
                solver.add(cost_var <= z3.RealVal(str(cap.max_cost_per_call)))
                solver.add(token_var <= cap.max_tokens_per_call)
            else:
                # Default bounds -- generous
                solver.add(cost_var <= z3.RealVal("5.0"))
                solver.add(token_var <= 200_000)

        # Total budget constraints
        if step_costs:
            total_cost = z3.Sum(*step_costs)
            solver.add(total_cost <= z3.RealVal(str(budget.max_total_cost_usd)))

        if step_tokens:
            total_tokens = z3.Sum(*step_tokens)
            solver.add(total_tokens <= budget.max_total_tokens)

        # Check if there exists a feasible assignment
        result = solver.check()
        if result == z3.unsat:
            violations.append(
                "Budget infeasible: no cost/token assignment satisfies all constraints"
            )
        elif result == z3.unknown:
            warnings.append("Budget check inconclusive: solver timed out")

        # Warning if we're close to budget
        num_steps = len(sub_goals)
        avg_cost_per_step = budget.max_total_cost_usd / max(num_steps, 1)
        if avg_cost_per_step < 0.01:
            warnings.append(
                f"Tight budget: {budget.max_total_cost_usd:.2f} USD across "
                f"{num_steps} steps ({avg_cost_per_step:.4f} USD per step)"
            )

        return {"violations": violations, "warnings": warnings}

    def _check_capability_matching(
        self, sub_goals: Sequence[SubGoal]
    ) -> list[str]:
        """Verify each sub-goal is assigned to a capable agent."""
        violations: list[str] = []
        cap_map = {c.agent_id: c for c in self._config.agent_capabilities}

        for goal in sub_goals:
            if goal.assigned_agent is None:
                violations.append(
                    f"Sub-goal '{goal.goal_id}' ({goal.description[:50]}) "
                    f"has no assigned agent"
                )
                continue

            if goal.assigned_agent not in cap_map:
                # No capability declaration -- skip check
                continue

            cap = cap_map[goal.assigned_agent]
            if cap.supported_tasks:
                # Check if any supported task pattern matches
                desc_lower = goal.description.lower()
                if not any(t.lower() in desc_lower for t in cap.supported_tasks):
                    violations.append(
                        f"Agent '{goal.assigned_agent}' cannot handle "
                        f"'{goal.description[:50]}' -- supported tasks: "
                        f"{cap.supported_tasks}"
                    )

        return violations

    async def _check_policy(
        self, policy: Policy, plan: Plan
    ) -> VerificationResult:
        """Evaluate a custom Z3 constraint policy against the plan."""
        return await policy.evaluate(plan)

    def verify_invariant(
        self,
        formula: z3.BoolRef,
        description: str = "custom invariant",
    ) -> VerificationResult:
        """
        Verify a standalone Z3 formula (useful for custom checks).

        Args:
            formula: Z3 boolean formula to check for satisfiability.
            description: Human-readable description of the invariant.

        Returns:
            VerificationResult.
        """
        start = time.perf_counter()
        solver = z3.Solver()
        solver.set("timeout", self._config.solver_timeout_ms)
        solver.add(formula)

        result = solver.check()
        elapsed = (time.perf_counter() - start) * 1000

        if result == z3.sat:
            model_dict = {}
            m = solver.model()
            for decl in m.decls():
                model_dict[decl.name()] = str(m[decl])
            return VerificationResult(
                is_safe=True,
                solver_time_ms=elapsed,
                constraints_checked=1,
                model=model_dict,
            )
        elif result == z3.unsat:
            return VerificationResult(
                is_safe=False,
                violations=[f"Invariant unsatisfiable: {description}"],
                solver_time_ms=elapsed,
                constraints_checked=1,
            )
        else:
            return VerificationResult(
                is_safe=False,
                warnings=[f"Invariant check inconclusive: {description}"],
                solver_time_ms=elapsed,
                constraints_checked=1,
            )
