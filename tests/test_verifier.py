"""Tests for Z3-based safety verification."""

from __future__ import annotations

import z3
import pytest

from lattice.core.planner import Plan, PlanningStrategy, SubGoal
from lattice.core.verifier import (
    AgentCapability,
    ResourceBudget,
    Verifier,
    VerifierConfig,
)
from lattice.verification.invariants import (
    AcyclicInvariant,
    InvariantChecker,
    MaxCostInvariant,
    MaxStepsInvariant,
)
from lattice.verification.solver import Z3SolverWrapper


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def verifier() -> Verifier:
    return Verifier(config=VerifierConfig(
        budget=ResourceBudget(max_total_cost_usd=5.0, max_steps=10),
        agent_capabilities=[
            AgentCapability(
                agent_id="research",
                supported_tasks=["research", "search"],
                max_cost_per_call=1.0,
            ),
        ],
    ))


# -- Verifier tests ----------------------------------------------------------

class TestVerifierPlan:
    @pytest.mark.asyncio
    async def test_valid_plan(self, verifier: Verifier) -> None:
        plan = Plan(
            task="Valid",
            strategy=PlanningStrategy.REACT,
            sub_goals=[
                SubGoal(goal_id="1", description="Research topic", assigned_agent="research"),
                SubGoal(goal_id="2", description="Analyze", assigned_agent="research", dependencies=["1"]),
            ],
        )
        result = await verifier.verify_plan(plan)
        assert result.is_safe
        assert result.constraints_checked > 0

    @pytest.mark.asyncio
    async def test_cyclic_plan_detected(self, verifier: Verifier) -> None:
        plan = Plan(
            task="Cyclic",
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=[
                SubGoal(goal_id="a", description="A", assigned_agent="research", dependencies=["c"]),
                SubGoal(goal_id="b", description="B", assigned_agent="research", dependencies=["a"]),
                SubGoal(goal_id="c", description="C", assigned_agent="research", dependencies=["b"]),
            ],
        )
        result = await verifier.verify_plan(plan)
        assert not result.is_safe
        assert any("cycle" in v.lower() for v in result.violations)

    @pytest.mark.asyncio
    async def test_too_many_steps(self) -> None:
        verifier = Verifier(config=VerifierConfig(
            budget=ResourceBudget(max_steps=2),
        ))
        plan = Plan(
            task="Too many",
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=[
                SubGoal(goal_id=str(i), description=f"Step {i}", assigned_agent="a")
                for i in range(5)
            ],
        )
        result = await verifier.verify_plan(plan)
        assert not result.is_safe
        assert any("steps" in v.lower() for v in result.violations)

    @pytest.mark.asyncio
    async def test_empty_plan_is_safe(self, verifier: Verifier) -> None:
        plan = Plan(task="Empty", strategy=PlanningStrategy.REACT, sub_goals=[])
        result = await verifier.verify_plan(plan)
        assert result.is_safe


class TestVerifierInvariant:
    def test_satisfiable_invariant(self, verifier: Verifier) -> None:
        x = z3.Int("x")
        result = verifier.verify_invariant(z3.And(x > 0, x < 10), "x in range")
        assert result.is_safe
        assert result.model is not None

    def test_unsatisfiable_invariant(self, verifier: Verifier) -> None:
        x = z3.Int("x")
        result = verifier.verify_invariant(z3.And(x > 10, x < 5), "impossible")
        assert not result.is_safe
        assert len(result.violations) > 0


# -- Z3 Solver Wrapper tests ------------------------------------------------

class TestZ3Solver:
    def test_basic_satisfaction(self) -> None:
        solver = Z3SolverWrapper()
        x = solver.int_var("x")
        y = solver.int_var("y")
        solver.add(x + y == 10)
        solver.add(x > 0, y > 0)

        result = solver.check()
        assert result.satisfiable is True
        assert result.model is not None
        assert "x" in result.model
        assert "y" in result.model

    def test_unsatisfiable(self) -> None:
        solver = Z3SolverWrapper()
        x = solver.int_var("x")
        solver.add(x > 10)
        solver.add(x < 5)

        result = solver.check()
        assert result.satisfiable is False

    def test_all_different(self) -> None:
        solver = Z3SolverWrapper()
        a, b, c = solver.int_vars(["a", "b", "c"])
        solver.all_different(a, b, c)
        solver.bounded(a, 1, 3)
        solver.bounded(b, 1, 3)
        solver.bounded(c, 1, 3)

        result = solver.check()
        assert result.satisfiable is True
        vals = set(result.model.values())
        assert len(vals) == 3

    def test_implies(self) -> None:
        solver = Z3SolverWrapper()
        p = solver.bool_var("p")
        q = solver.bool_var("q")
        solver.add(p == True)
        solver.implies(p, q)

        result = solver.check()
        assert result.satisfiable is True
        assert result.model["q"] == "True"

    def test_exactly_one(self) -> None:
        solver = Z3SolverWrapper()
        a, b, c = solver.bool_vars(["a", "b", "c"])
        solver.exactly_one(a, b, c)

        result = solver.check()
        assert result.satisfiable is True
        true_count = sum(1 for v in result.model.values() if v == "True")
        assert true_count == 1

    def test_push_pop(self) -> None:
        solver = Z3SolverWrapper()
        x = solver.int_var("x")
        solver.add(x > 0)

        solver.push()
        solver.add(x < 0)
        result = solver.check()
        assert result.satisfiable is False

        solver.pop()
        result = solver.check()
        assert result.satisfiable is True

    def test_real_variables(self) -> None:
        solver = Z3SolverWrapper()
        x = solver.real_var("x")
        solver.add(x > z3.RealVal("0.5"))
        solver.add(x < z3.RealVal("0.6"))

        result = solver.check()
        assert result.satisfiable is True

    def test_reset(self) -> None:
        solver = Z3SolverWrapper()
        solver.int_var("x")
        solver.add(z3.Int("x") > 0)
        solver.reset()
        assert solver.num_constraints == 0
        assert len(solver.variables) == 0


# -- Invariant tests ---------------------------------------------------------

class TestInvariants:
    @pytest.mark.asyncio
    async def test_max_cost_passes(self) -> None:
        inv = MaxCostInvariant(max_cost_usd=10.0)
        plan = Plan(
            task="test",
            strategy=PlanningStrategy.REACT,
            sub_goals=[SubGoal(goal_id="1", description="s1")],
        )
        result = await inv.check(plan)
        assert result.holds

    @pytest.mark.asyncio
    async def test_max_steps_fails(self) -> None:
        inv = MaxStepsInvariant(max_steps=2)
        plan = Plan(
            task="test",
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=[
                SubGoal(goal_id=str(i), description=f"s{i}")
                for i in range(5)
            ],
        )
        result = await inv.check(plan)
        assert not result.holds

    @pytest.mark.asyncio
    async def test_acyclic_valid(self) -> None:
        inv = AcyclicInvariant()
        plan = Plan(
            task="test",
            strategy=PlanningStrategy.PLAN_AND_SOLVE,
            sub_goals=[
                SubGoal(goal_id="1", description="s1"),
                SubGoal(goal_id="2", description="s2", dependencies=["1"]),
            ],
        )
        result = await inv.check(plan)
        assert result.holds

    @pytest.mark.asyncio
    async def test_invariant_checker(self) -> None:
        checker = InvariantChecker()
        plan = Plan(
            task="test",
            strategy=PlanningStrategy.REACT,
            sub_goals=[SubGoal(goal_id="1", description="simple")],
        )
        safe = await checker.is_safe(plan)
        assert safe
