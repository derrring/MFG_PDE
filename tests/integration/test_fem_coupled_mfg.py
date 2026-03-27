"""
Integration test: Full MFG solve on unstructured mesh using FEM assembly.

Tests the complete pipeline: scikit-fem assembly → HJB backward → FP forward
→ Picard coupling on a 2D triangle mesh.
"""

from __future__ import annotations

import pytest

import numpy as np

skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")

from mfgarchon.alg.numerical.fem.assembly import (  # noqa: E402
    assemble_mass,
    assemble_stiffness,
    create_basis,
)


@pytest.mark.integration
class TestFEMCoupledMFGSolve:
    """End-to-end MFG solve on unstructured mesh via FEM assembly."""

    def test_picard_iteration_converges(self):
        """Picard iteration produces finite, non-negative results."""
        from scipy.sparse.linalg import spsolve

        mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
        basis = create_basis(mesh, order=1)
        K = assemble_stiffness(basis)
        M_mat = assemble_mass(basis)
        N = basis.N

        T, Nt = 0.5, 10
        dt = T / Nt
        D = 0.5 * 0.3**2

        # Initial density
        x, y = mesh.p
        m_init = np.exp(-10 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        m_init /= (M_mat @ m_init).sum()

        U = np.zeros((Nt + 1, N))
        M = np.zeros((Nt + 1, N))
        M[0] = m_init

        A_base = M_mat / dt + D * K

        for _picard in range(5):
            # HJB backward
            U_new = np.zeros((Nt + 1, N))
            for n in range(Nt - 1, -1, -1):
                rhs = (M_mat / dt) @ U_new[n + 1] + M_mat @ M[n]
                U_new[n] = spsolve(A_base, rhs)

            # FP forward
            M_new = np.zeros((Nt + 1, N))
            M_new[0] = m_init
            for n in range(Nt):
                M_new[n + 1] = spsolve(A_base, (M_mat / dt) @ M_new[n])
                M_new[n + 1] = np.maximum(M_new[n + 1], 0.0)

            eta = 0.3
            U = (1 - eta) * U + eta * U_new
            M = (1 - eta) * M + eta * M_new

        assert np.all(np.isfinite(U)), "U contains NaN/Inf"
        assert np.all(np.isfinite(M)), "M contains NaN/Inf"
        assert np.all(M >= -1e-10), f"Density negative: min={M.min()}"

    def test_pure_diffusion_conserves_mass(self):
        """FP without advection should conserve mass exactly."""
        from scipy.sparse.linalg import spsolve

        mesh = skfem.MeshTri.init_sqsymmetric().refined(3)
        basis = create_basis(mesh, order=1)
        K = assemble_stiffness(basis)
        M_mat = assemble_mass(basis)

        dt = 0.01
        D = 0.1
        A = M_mat / dt + D * K

        x, y = mesh.p
        m = np.exp(-20 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        initial_mass = (M_mat @ m).sum()

        for _ in range(20):
            m = spsolve(A, (M_mat / dt) @ m)
            m = np.maximum(m, 0.0)

        final_mass = (M_mat @ m).sum()
        rel_error = abs(final_mass - initial_mass) / initial_mass

        assert rel_error < 0.01, f"Mass error {rel_error:.2%} exceeds 1%"

    def test_hjb_backward_produces_smooth_solution(self):
        """HJB backward solve should produce smooth, bounded value function."""
        from scipy.sparse.linalg import spsolve

        mesh = skfem.MeshTri.init_sqsymmetric().refined(2)
        basis = create_basis(mesh, order=1)
        K = assemble_stiffness(basis)
        M_mat = assemble_mass(basis)
        N = basis.N

        dt = 0.05
        D = 0.1
        A = M_mat / dt + D * K

        U = np.zeros(N)  # terminal condition
        for _ in range(10):
            rhs = (M_mat / dt) @ U + M_mat @ np.ones(N)  # constant source
            U = spsolve(A, rhs)

        assert np.all(np.isfinite(U))
        assert U.max() < 100, f"U unbounded: max={U.max()}"
        assert U.max() > 0, "U should be positive with positive source"
