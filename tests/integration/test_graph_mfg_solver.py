"""Integration tests for GraphMFGSolver (Type 2 network MFG)."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.graph_coupling import (
    AdjacencyCoupling,
    LaplacianCoupling,
)
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import (
    GraphMFGResult,
    GraphMFGSolver,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem


def _make_node_problem(coupling_strength: float = 1.0, sigma: float = 0.3) -> MFGProblem:
    """Create a 1D MFG problem for one graph node."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_strength * m,
        coupling_dm=lambda m: coupling_strength,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        Nx=21,
        xmin=0.0,
        xmax=1.0,
        T=0.5,
        Nt=10,
        sigma=sigma,
        components=components,
    )


def _make_3node_system():
    """Create a 3-node ring graph with per-node FDM solvers."""
    problems = [_make_node_problem(c) for c in [1.0, 0.5, 0.8]]
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    coupling = AdjacencyCoupling(A, alpha=0.05, beta=0.02)
    hjbs = [HJBFDMSolver(p) for p in problems]
    fps = [FPFDMSolver(p) for p in problems]
    return problems, coupling, hjbs, fps


class TestGraphMFGSolverInstantiation:
    def test_basic_3node(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        assert solver._N == 3

    def test_mismatched_counts_raises(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        with pytest.raises(ValueError, match="Need 3 problems"):
            GraphMFGSolver(
                problems=problems[:2],
                coupling=coupling,
                hjb_solvers=hjbs,
                fp_solvers=fps,
            )

    def test_custom_params(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=20,
            tolerance=1e-3,
            damping=0.3,
        )
        assert solver._max_iter == 20
        assert solver._tol == 1e-3


class TestGraphMFGSolverSolve:
    def test_returns_result(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert isinstance(result, GraphMFGResult)

    def test_result_shapes(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert len(result.values) == 3
        assert len(result.densities) == 3
        assert result.n_nodes == 3
        Nt = problems[0].Nt
        Nx = problems[0].geometry.get_grid_shape()[0]
        for k in range(3):
            assert result.values[k].shape == (Nt + 1, Nx)

    def test_solutions_finite(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = solver.solve()
        for k in range(3):
            assert np.all(np.isfinite(result.values[k]))
            assert np.all(np.isfinite(result.densities[k]))

    def test_error_history(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = solver.solve()
        assert len(result.error_history) > 0
        assert len(result.error_history) <= 5

    def test_iterations_reported(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert 0 < result.iterations <= 3


class TestGraphMFGSolverCouplingEffect:
    def test_coupling_affects_solution(self):
        """With coupling, node solutions should differ from uncoupled."""
        # Heterogeneous nodes so coupling has something to mix
        problems = [_make_node_problem(0.5), _make_node_problem(2.0)]
        A = np.array([[0, 1], [1, 0]], dtype=float)

        # Strong coupling
        coupling_strong = AdjacencyCoupling(A, alpha=0.5, beta=0.1)
        hjbs1 = [HJBFDMSolver(p) for p in problems]
        fps1 = [FPFDMSolver(p) for p in problems]
        solver_coupled = GraphMFGSolver(
            problems=problems,
            coupling=coupling_strong,
            hjb_solvers=hjbs1,
            fp_solvers=fps1,
            max_iterations=5,
            damping=0.5,
        )

        # Zero coupling
        coupling_zero = AdjacencyCoupling(A, alpha=0.0, beta=0.0)
        hjbs2 = [HJBFDMSolver(p) for p in problems]
        fps2 = [FPFDMSolver(p) for p in problems]
        solver_uncoupled = GraphMFGSolver(
            problems=problems,
            coupling=coupling_zero,
            hjb_solvers=hjbs2,
            fp_solvers=fps2,
            max_iterations=5,
            damping=0.5,
        )

        r1 = solver_coupled.solve()
        r2 = solver_uncoupled.solve()

        # Coupled solutions should differ
        assert not np.allclose(r1.values[0], r2.values[0], atol=1e-6)

    def test_laplacian_coupling(self):
        """GraphMFGSolver should work with LaplacianCoupling."""
        problems = [_make_node_problem(1.0) for _ in range(3)]
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        coupling = LaplacianCoupling(A, kappa=0.05)
        hjbs = [HJBFDMSolver(p) for p in problems]
        fps = [FPFDMSolver(p) for p in problems]
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = solver.solve()
        assert np.all(np.isfinite(result.values[0]))


class TestGraphMFGSolverGetResults:
    def test_get_results_after_solve(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        solver.solve()
        U, _M = solver.get_results()
        assert U.shape[0] == problems[0].Nt + 1

    def test_get_results_before_solve_raises(self):
        problems, coupling, hjbs, fps = _make_3node_system()
        solver = GraphMFGSolver(
            problems=problems,
            coupling=coupling,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        with pytest.raises(RuntimeError, match="No solution"):
            solver.get_results()
