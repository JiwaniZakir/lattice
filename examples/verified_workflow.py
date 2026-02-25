"""
Verified Workflow Example

Demonstrates Z3-based safety verification including:
  1. DAG acyclicity verification
  2. Budget feasibility checking
  3. Custom policy enforcement
  4. Invariant checking

Shows how verification catches unsafe plans before execution.
"""

import asyncio

import z3

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
from lattice.verification.policies import CostCapPolicy, PolicyEngine, RequiredAgentPolicy
from lattice.verification.solver import Z3SolverWrapper


async def demo_z3_solver() -> None:
    """Demonstrate the Z3 solver wrapper."""
    print("=== Z3 Solver Demo ===\n")

    solver = Z3SolverWrapper(timeout_ms=5000)

    # Create variables for a scheduling problem
    task_a = solver.int_var("task_a_time")
    task_b = solver.int_var("task_b_time")
    task_c = solver.int_var("task_c_time")

    # All tasks start at different times
    solver.all_different(task_a, task_b, task_c)

    # Tasks must start in the first 10 time slots
    solver.bounded(task_a, low=0, high=9)
    solver.bounded(task_b, low=0, high=9)
    solver.bounded(task_c, low=0, high=9)

    # Task B must start after Task A
    solver.add(task_b > task_a)

    # Task C must start after Task B
    solver.add(task_c > task_b)

    result = solver.check()
    print(f"Satisfiable: {result.satisfiable}")
    print(f"Model: {result.model}")
    print(f"Solver time: {result.time_ms:.2f}ms")
    print(f"Constraints: {result.num_constraints}")


async def demo_plan_verification() -> None:
    """Demonstrate plan verification with safety checks."""
    print("\n=== Plan Verification Demo ===\n")

    # Configure verifier with budget and agent capabilities
    config = VerifierConfig(
        budget=ResourceBudget(
            max_total_cost_usd=5.0,
            max_total_tokens=500_000,
            max_steps=20,
        ),
        agent_capabilities=[
            AgentCapability(
                agent_id="researcher",
                supported_tasks=["research", "search", "gather"],
                max_cost_per_call=1.0,
            ),
            AgentCapability(
                agent_id="analyst",
                supported_tasks=["analyze", "synthesize", "compare"],
                max_cost_per_call=0.5,
            ),
        ],
    )
    verifier = Verifier(config=config)

    # -- Test 1: Valid plan --------------------------------------------------
    valid_plan = Plan(
        task="Research and analyze AI safety",
        strategy=PlanningStrategy.PLAN_AND_SOLVE,
        sub_goals=[
            SubGoal(goal_id="s1", description="Research AI safety papers", assigned_agent="researcher"),
            SubGoal(goal_id="s2", description="Analyze findings", assigned_agent="analyst", dependencies=["s1"]),
        ],
    )

    result = await verifier.verify_plan(valid_plan)
    print(f"Valid plan: safe={result.is_safe}, violations={len(result.violations)}")
    print(f"  Solver time: {result.solver_time_ms:.2f}ms")
    print(f"  Constraints checked: {result.constraints_checked}")

    # -- Test 2: Plan with cycle ---------------------------------------------
    cyclic_plan = Plan(
        task="Cyclic dependencies",
        strategy=PlanningStrategy.PLAN_AND_SOLVE,
        sub_goals=[
            SubGoal(goal_id="a", description="Task A", assigned_agent="researcher", dependencies=["c"]),
            SubGoal(goal_id="b", description="Task B", assigned_agent="analyst", dependencies=["a"]),
            SubGoal(goal_id="c", description="Task C", assigned_agent="researcher", dependencies=["b"]),
        ],
    )

    result = await verifier.verify_plan(cyclic_plan)
    print(f"\nCyclic plan: safe={result.is_safe}")
    for v in result.violations:
        print(f"  Violation: {v}")

    # -- Test 3: Standalone Z3 invariant -------------------------------------
    print("\n--- Standalone invariant ---")
    x = z3.Int("agents_available")
    y = z3.Int("tasks_pending")
    formula = z3.And(x >= 3, y <= 10, x * 5 >= y)
    inv_result = verifier.verify_invariant(formula, "enough agents for tasks")
    print(f"Invariant: safe={inv_result.is_safe}, model={inv_result.model}")


async def demo_invariant_checker() -> None:
    """Demonstrate the invariant checker."""
    print("\n=== Invariant Checker Demo ===\n")

    checker = InvariantChecker([
        MaxCostInvariant(max_cost_usd=2.0),
        MaxStepsInvariant(max_steps=5),
        AcyclicInvariant(),
    ])

    plan = Plan(
        task="Small workflow",
        strategy=PlanningStrategy.REACT,
        sub_goals=[
            SubGoal(goal_id="1", description="Step 1", assigned_agent="a"),
            SubGoal(goal_id="2", description="Step 2", assigned_agent="b", dependencies=["1"]),
            SubGoal(goal_id="3", description="Step 3", assigned_agent="a", dependencies=["2"]),
        ],
    )

    results = await checker.check_all(plan)
    for r in results:
        status = "PASS" if r.holds else "FAIL"
        print(f"  [{status}] {r.name}: {r.message}")

    is_safe = await checker.is_safe(plan)
    print(f"\n  Overall safe: {is_safe}")


async def demo_policy_engine() -> None:
    """Demonstrate the policy engine."""
    print("\n=== Policy Engine Demo ===\n")

    engine = PolicyEngine(policies=[
        CostCapPolicy(max_per_step_usd=0.50, max_total_usd=2.0),
        RequiredAgentPolicy(required_agents=["researcher", "critic"]),
    ])

    # Plan missing the critic agent
    plan = Plan(
        task="Incomplete workflow",
        strategy=PlanningStrategy.PLAN_AND_SOLVE,
        sub_goals=[
            SubGoal(goal_id="1", description="Research", assigned_agent="researcher"),
            SubGoal(goal_id="2", description="Analyze", assigned_agent="analyst"),
        ],
    )

    violations = await engine.evaluate_all(plan)
    compliant = await engine.is_compliant(plan)
    print(f"Compliant: {compliant}")
    for v in violations:
        print(f"  [{v.severity}] {v.policy_name}: {v.message}")


async def main() -> None:
    await demo_z3_solver()
    await demo_plan_verification()
    await demo_invariant_checker()
    await demo_policy_engine()
    print("\nAll verification demos complete!")


if __name__ == "__main__":
    asyncio.run(main())
