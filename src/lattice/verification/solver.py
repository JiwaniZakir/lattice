"""
Z3 solver wrapper with ergonomic API for workflow verification.

Provides a higher-level interface over the raw Z3 solver that simplifies
common verification patterns for multi-agent workflows:

    - Typed variable creation (Int, Real, Bool, String)
    - Constraint builder DSL
    - Timeout management
    - Model extraction into Python dicts
    - Incremental solving with push/pop
"""

from __future__ import annotations

import time
from typing import Any

import z3
from pydantic import BaseModel, Field


class SolverResult(BaseModel):
    """Result of a Z3 solver check."""

    satisfiable: bool | None  # None means unknown
    model: dict[str, Any] | None = None
    unsat_core: list[str] | None = None
    time_ms: float = 0.0
    num_constraints: int = 0


class Z3SolverWrapper:
    """
    Ergonomic wrapper around Z3 for workflow verification.

    Provides typed variable creation, constraint management, and
    model extraction with timeout support.

    Example::

        solver = Z3SolverWrapper(timeout_ms=5000)
        x = solver.int_var("x")
        y = solver.int_var("y")
        solver.add(x + y == 10)
        solver.add(x > 0, y > 0)
        result = solver.check()
        # result.model == {"x": "3", "y": "7"} (or similar)

    Args:
        timeout_ms: Solver timeout in milliseconds.
        track_unsat_core: Whether to track UNSAT cores for debugging.
    """

    def __init__(
        self,
        timeout_ms: int = 5000,
        track_unsat_core: bool = False,
    ) -> None:
        self._timeout_ms = timeout_ms
        self._track_core = track_unsat_core

        if track_unsat_core:
            self._solver = z3.Solver()
            self._solver.set("unsat_core", True)
        else:
            self._solver = z3.Solver()

        self._solver.set("timeout", timeout_ms)
        self._variables: dict[str, z3.ExprRef] = {}
        self._constraint_count = 0

    # ── Variable creation ──────────────────────────────────────────

    def int_var(self, name: str) -> z3.ArithRef:
        """Create an integer variable."""
        var = z3.Int(name)
        self._variables[name] = var
        return var

    def real_var(self, name: str) -> z3.ArithRef:
        """Create a real-valued variable."""
        var = z3.Real(name)
        self._variables[name] = var
        return var

    def bool_var(self, name: str) -> z3.BoolRef:
        """Create a boolean variable."""
        var = z3.Bool(name)
        self._variables[name] = var
        return var

    def int_vars(self, names: list[str]) -> list[z3.ArithRef]:
        """Create multiple integer variables."""
        return [self.int_var(n) for n in names]

    def real_vars(self, names: list[str]) -> list[z3.ArithRef]:
        """Create multiple real-valued variables."""
        return [self.real_var(n) for n in names]

    def bool_vars(self, names: list[str]) -> list[z3.BoolRef]:
        """Create multiple boolean variables."""
        return [self.bool_var(n) for n in names]

    # ── Constraint management ──────────────────────────────────────

    def add(self, *constraints: z3.BoolRef) -> None:
        """Add constraints to the solver."""
        for c in constraints:
            if self._track_core:
                label = z3.Bool(f"c_{self._constraint_count}")
                self._solver.assert_and_track(c, label)
            else:
                self._solver.add(c)
            self._constraint_count += 1

    def add_soft(self, constraint: z3.BoolRef, weight: int = 1) -> None:
        """Add a soft constraint (for optimization)."""
        # Z3's Optimize solver supports soft constraints;
        # for regular Solver we add it as a normal constraint.
        self._solver.add(constraint)
        self._constraint_count += 1

    # ── Solving ────────────────────────────────────────────────────

    def check(self) -> SolverResult:
        """
        Check satisfiability of all constraints.

        Returns:
            SolverResult with model if satisfiable.
        """
        start = time.perf_counter()
        result = self._solver.check()
        elapsed = (time.perf_counter() - start) * 1000

        if result == z3.sat:
            model = self._extract_model()
            return SolverResult(
                satisfiable=True,
                model=model,
                time_ms=elapsed,
                num_constraints=self._constraint_count,
            )
        elif result == z3.unsat:
            core = None
            if self._track_core:
                core = [str(c) for c in self._solver.unsat_core()]
            return SolverResult(
                satisfiable=False,
                unsat_core=core,
                time_ms=elapsed,
                num_constraints=self._constraint_count,
            )
        else:
            return SolverResult(
                satisfiable=None,
                time_ms=elapsed,
                num_constraints=self._constraint_count,
            )

    def push(self) -> None:
        """Push solver state (for incremental solving)."""
        self._solver.push()

    def pop(self) -> None:
        """Pop solver state."""
        self._solver.pop()

    def reset(self) -> None:
        """Reset the solver, clearing all constraints."""
        self._solver.reset()
        self._solver.set("timeout", self._timeout_ms)
        self._variables.clear()
        self._constraint_count = 0

    # ── Convenience methods ────────────────────────────────────────

    def all_different(self, *vars: z3.ExprRef) -> None:
        """Add a constraint that all variables must be distinct."""
        self.add(z3.Distinct(*vars))

    def bounded(
        self,
        var: z3.ArithRef,
        low: int | float | None = None,
        high: int | float | None = None,
    ) -> None:
        """Add bounds constraints on a variable."""
        if low is not None:
            self.add(var >= low)
        if high is not None:
            self.add(var <= high)

    def implies(self, antecedent: z3.BoolRef, consequent: z3.BoolRef) -> None:
        """Add an implication constraint."""
        self.add(z3.Implies(antecedent, consequent))

    def exactly_one(self, *bools: z3.BoolRef) -> None:
        """Exactly one of the boolean variables must be True."""
        # At least one
        self.add(z3.Or(*bools))
        # At most one: pairwise exclusion
        for i in range(len(bools)):
            for j in range(i + 1, len(bools)):
                self.add(z3.Not(z3.And(bools[i], bools[j])))

    def at_most_k(self, bools: list[z3.BoolRef], k: int) -> None:
        """At most k of the boolean variables can be True."""
        # Use Z3's PbLe (pseudo-boolean <=)
        self.add(z3.PbLe([(b, 1) for b in bools], k))

    # ── Internal ───────────────────────────────────────────────────

    def _extract_model(self) -> dict[str, Any]:
        """Extract the satisfying model as a Python dict."""
        model: dict[str, Any] = {}
        z3_model = self._solver.model()
        for decl in z3_model.decls():
            val = z3_model[decl]
            model[decl.name()] = str(val)
        return model

    @property
    def num_constraints(self) -> int:
        return self._constraint_count

    @property
    def variables(self) -> dict[str, z3.ExprRef]:
        return dict(self._variables)
