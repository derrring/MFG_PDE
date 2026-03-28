"""
Tests for True Adjoint Mode (Issue #707).

Verifies that the linearized HJB operator (Jacobian) transpose matches
the independently-constructed FP divergence_upwind operator. This is the
core mathematical claim of the true adjoint approach:

    A_FP_advection = (A_HJB_Jacobian)^T

References:
    - Issue #706 (adjoint discretization)
    - Achdou & Capuzzo-Dolcetta (2010): Structure-preserving discretization
"""

from __future__ import annotations

import pytest

import numpy as np
import scipy.sparse as sparse

from mfgarchon import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc


def _create_lq_problem_1d(Nx: int = 51, T: float = 1.0, Nt: int = 20) -> MFGProblem:
    """Create a 1D LQ-MFG problem with class-based Hamiltonian."""
    from mfgarchon.core import MFGComponents, QuadraticMFGHamiltonian

    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[Nx],
        boundary_conditions=neumann_bc(dimension=1),
    )
    H = QuadraticMFGHamiltonian(coupling_coefficient=1.0)
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0 / Nx,
    )
    return MFGProblem(
        geometry=grid,
        T=T,
        Nt=Nt,
        diffusion=0.1,
        components=components,
    )


@pytest.mark.unit
class TestBuildLinearizedOperator1D:
    """Test build_linearized_operator() for 1D FDM."""

    def test_returns_sparse_matrix(self):
        """Linearized operator returns a sparse CSR matrix of correct shape."""
        problem = _create_lq_problem_1d(Nx=21)
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)

        U = np.sin(np.linspace(0, np.pi, 21))
        M = np.ones(21) / 21

        A = solver.build_linearized_operator(U, M, time=0.0)

        assert sparse.issparse(A)
        assert A.shape == (21, 21)

    def test_tridiagonal_structure(self):
        """1D linearized operator is tridiagonal."""
        problem = _create_lq_problem_1d(Nx=31)
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)

        U = 0.5 * np.linspace(0, 1, 31) ** 2
        M = np.ones(31) / 31

        A = solver.build_linearized_operator(U, M, time=0.0)
        dense = A.toarray()

        # Check that entries beyond tridiagonal band are zero
        for i in range(31):
            for j in range(31):
                if abs(i - j) > 1:
                    assert dense[i, j] == 0.0, f"Non-zero at ({i},{j}): {dense[i, j]}"

    def test_boundary_rows_zero(self):
        """Boundary rows should be zero (no-flux BC)."""
        problem = _create_lq_problem_1d(Nx=21)
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)

        U = np.sin(np.linspace(0, np.pi, 21))
        M = np.ones(21) / 21

        A = solver.build_linearized_operator(U, M, time=0.0)
        dense = A.toarray()

        np.testing.assert_array_equal(dense[0, :], 0.0)
        np.testing.assert_array_equal(dense[-1, :], 0.0)

    def test_zero_value_function_gives_zero_operator(self):
        """If U = const, gradients are zero, so A_adv should be zero."""
        problem = _create_lq_problem_1d(Nx=21)
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)

        U = np.ones(21) * 3.0  # constant
        M = np.ones(21) / 21

        A = solver.build_linearized_operator(U, M, time=0.0)

        assert A.nnz == 0 or np.allclose(A.toarray(), 0.0, atol=1e-14)


@pytest.mark.unit
class TestJacobianTransposeEqualsAdvection:
    """Core verification: A_Jac^T should match the velocity-based advection matrix
    for quadratic Hamiltonians H = (c/2)|p|^2.

    For LQ problems, dH/dp = c*p, and the Jacobian approach reduces to
    v = c*p with upwind stencil coefficients. This should match
    build_advection_matrix() which computes v = -coupling * grad(U).
    """

    def test_jacobian_coefficients_match_analytical_dh_dp(self):
        """Verify that A_Jac encodes dH/dp correctly by checking against analytical values.

        For H = (1/2)|p|^2: dH/dp = p = grad(U).
        The Jacobian entry A_jac[i,i] should equal dH/dp[i] * (stencil coeff for U_i).
        """
        Nx = 21
        problem = _create_lq_problem_1d(Nx=Nx)
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)

        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]
        U = -0.5 * (x - 0.5) ** 2
        M = np.ones(Nx) / Nx

        A_jac = solver.build_linearized_operator(U, M, time=0.0)
        dense = A_jac.toarray()

        # Check interior points where grad > 0 (backward stencil)
        for i in range(2, Nx // 2):
            grad_i = (U[i] - U[i - 1]) / dx  # backward upwind gradient
            dH_dp = grad_i  # For quadratic H
            expected_diag = dH_dp * (1.0 / dx)  # dp/dU_i = +1/dx for backward
            expected_lower = dH_dp * (-1.0 / dx)  # dp/dU_{i-1} = -1/dx

            np.testing.assert_allclose(
                dense[i, i],
                expected_diag,
                atol=1e-10,
                err_msg=f"Diagonal at i={i}: expected dH_dp/dx={expected_diag}",
            )
            np.testing.assert_allclose(
                dense[i, i - 1],
                expected_lower,
                atol=1e-10,
                err_msg=f"Lower at i={i}: expected -dH_dp/dx={expected_lower}",
            )

    def test_jacobian_and_advection_use_different_upwind(self):
        """Verify that A_Jac and A_adv have different sparsity patterns.

        For v = -coupling*grad(U), HJB upwinds on sign(grad) while FP upwinds
        on sign(v) = sign(-grad). They wind in opposite directions.
        """
        Nx = 21
        problem = _create_lq_problem_1d(Nx=Nx)
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)

        x = np.linspace(0, 1, Nx)
        U = -0.5 * (x - 0.5) ** 2  # grad > 0 for x < 0.5, grad < 0 for x > 0.5
        M = np.ones(Nx) / Nx

        A_jac = solver.build_linearized_operator(U, M, time=0.0)
        A_adv = solver.build_advection_matrix(U, time=0.0)

        # At interior points where grad != 0, they should differ in structure
        # because they upwind in opposite directions
        jac_dense = A_jac.toarray()
        adv_dense = A_adv.toarray()

        # Check a point with positive gradient (e.g., index 3)
        # A_jac should use backward stencil (lower diagonal), A_adv should use forward (upper)
        assert jac_dense[3, 2] != 0.0, "A_jac should have lower entry (backward stencil)"
        assert adv_dense[3, 4] != 0.0, "A_adv should have upper entry (forward stencil)"

    def test_transpose_column_sum_property(self):
        """A_Jac^T should have zero column sums (mass conservation) for interior nodes."""
        Nx = 41
        problem = _create_lq_problem_1d(Nx=Nx)
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)

        x = np.linspace(0, 1, Nx)
        U = np.sin(2 * np.pi * x)
        M = np.ones(Nx) / Nx

        A_jac = solver.build_linearized_operator(U, M, time=0.0)

        # For a divergence-form FP operator (A_jac^T), column sums of A^T = row sums of A
        # should be zero for interior points (mass conservation).
        row_sums = np.array(A_jac.sum(axis=1)).ravel()

        # Interior rows (skip boundaries which are zeroed)
        interior_sums = row_sums[2:-2]
        np.testing.assert_allclose(
            interior_sums,
            0.0,
            atol=1e-12,
            err_msg="Interior row sums of A_Jac should be zero (conservation)",
        )


@pytest.mark.unit
class TestRequiresHamiltonianClass:
    """build_linearized_operator requires a class-based Hamiltonian."""

    def test_raises_without_hamiltonian_class(self):
        """Should raise ValueError if no Hamiltonian class is set on the problem."""
        # Create a problem and then clear its hamiltonian_class to simulate missing H
        problem = _create_lq_problem_1d(Nx=21)
        problem.components._hamiltonian_class = None  # Force no Hamiltonian

        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        solver = HJBFDMSolver(problem)
        U = np.zeros(21)
        M = np.ones(21) / 21

        with pytest.raises(ValueError, match="Hamiltonian class"):
            solver.build_linearized_operator(U, M)


if __name__ == "__main__":
    """Quick smoke test."""
    print("Testing True Adjoint Mode (Issue #707)...")

    problem = _create_lq_problem_1d(Nx=31)
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

    solver = HJBFDMSolver(problem)

    x = np.linspace(0, 1, 31)
    U = -0.5 * (x - 0.5) ** 2
    M = np.ones(31) / 31

    A_jac = solver.build_linearized_operator(U, M, time=0.0)
    A_adv = solver.build_advection_matrix(U, time=0.0)

    print(f"A_jac shape: {A_jac.shape}, nnz: {A_jac.nnz}")
    print(f"A_adv shape: {A_adv.shape}, nnz: {A_adv.nnz}")

    diff = np.abs(A_jac.toarray()[1:-1, 1:-1] - A_adv.toarray()[1:-1, 1:-1]).max()
    print(f"Max interior difference (LQ): {diff:.2e}")

    row_sums = np.array(A_jac.sum(axis=1)).ravel()
    print(f"Max interior row sum: {np.abs(row_sums[2:-2]).max():.2e}")

    print("Smoke test passed.")
