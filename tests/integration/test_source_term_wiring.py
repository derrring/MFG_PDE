"""Integration tests for source_term wiring through FixedPointIterator.

Verifies that source_term_hjb and source_term_fp on MFGProblem
flow through the iterator to the HJB and FP solvers (#921).

Test strategy:
- Solve same problem with and without source_term
- Verify source_term changes the solution (not silently ignored)
- Verify source_term=None produces baseline solution
"""

from __future__ import annotations

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents


def _make_problem(**extra_kwargs):
    """Create a minimal 1D MFG problem with optional source terms."""
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=hamiltonian,
    )
    return MFGProblem(
        Nx=[30],
        Nt=10,
        T=1.0,
        components=components,
        **extra_kwargs,
    )


class TestSourceTermHJBWiring:
    """Verify source_term_hjb flows through to HJB solver."""

    def test_source_term_changes_solution(self):
        """A non-zero HJB source_term should change the value function."""
        # Baseline: no source term
        problem_base = _make_problem()
        result_base = problem_base.solve(max_iterations=3, verbose=False)

        # With source term: S_hjb(x, m, v, t) = 1.0 (constant forcing)
        problem_src = _make_problem(
            source_term_hjb=lambda x, m, v, t: np.ones(x.shape[0]),
        )
        result_src = problem_src.solve(max_iterations=3, verbose=False)

        # Solutions must differ
        assert result_base is not None
        assert result_src is not None
        diff = np.max(np.abs(result_src.U - result_base.U))
        assert diff > 1e-6, f"source_term_hjb had no effect: max diff = {diff:.2e}"

    def test_zero_source_term_matches_baseline(self):
        """A zero source_term should produce the same result as None."""
        problem_base = _make_problem()
        result_base = problem_base.solve(max_iterations=3, verbose=False)

        problem_zero = _make_problem(
            source_term_hjb=lambda x, m, v, t: np.zeros(x.shape[0]),
        )
        result_zero = problem_zero.solve(max_iterations=3, verbose=False)

        # Should match (within floating point)
        np.testing.assert_allclose(
            result_zero.U,
            result_base.U,
            atol=1e-10,
            err_msg="Zero source_term_hjb should match no source_term",
        )

    def test_source_term_field_stored(self):
        """Verify source_term_hjb is stored on MFGProblem."""

        def src(x, m, v, t):
            return np.ones(x.shape[0])

        problem = _make_problem(source_term_hjb=src)
        assert problem.source_term_hjb is src


class TestSourceTermFPWiring:
    """Verify source_term_fp flows through to FP solver."""

    def test_fp_source_changes_density(self):
        """A non-zero FP source_term should change the density evolution."""
        problem_base = _make_problem()
        result_base = problem_base.solve(max_iterations=3, verbose=False)

        # Small positive source: births everywhere
        problem_src = _make_problem(
            source_term_fp=lambda x, m, v, t: 0.01 * np.ones(x.shape[0]),
        )
        result_src = problem_src.solve(max_iterations=3, verbose=False)

        assert result_base is not None
        assert result_src is not None
        diff = np.max(np.abs(result_src.M - result_base.M))
        assert diff > 1e-6, f"source_term_fp had no effect: max diff = {diff:.2e}"


class TestExtendedPDEFields:
    """Test that MFGProblem stores all extended PDE fields."""

    def test_fields_default_none(self):
        problem = _make_problem()
        assert problem.source_term_hjb is None
        assert problem.source_term_fp is None
        assert problem.nonlocal_operator is None
        assert problem.obstacle is None

    def test_obstacle_field_stored(self):
        def obstacle(x):
            return x - 0.5

        problem = _make_problem(obstacle=obstacle)
        assert problem.obstacle is obstacle

    def test_nonlocal_operator_field_stored(self):
        problem = _make_problem(nonlocal_operator="placeholder")
        assert problem.nonlocal_operator == "placeholder"
