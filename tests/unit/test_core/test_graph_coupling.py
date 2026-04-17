"""Tests for GraphCouplingOperator Protocol and implementations."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.graph_coupling import (
    AdjacencyCoupling,
    GraphCouplingOperator,
    LaplacianCoupling,
    _get_time_slice,
)


def _ring_adjacency(n: int) -> np.ndarray:
    """Create adjacency matrix for n-node ring graph."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[i, (i - 1) % n] = 1.0
    return A


def _mock_values(n_nodes: int, Nt: int = 6, Nx: int = 11) -> list[np.ndarray]:
    """Create mock value functions for n nodes."""
    rng = np.random.RandomState(42)
    return [rng.randn(Nt, Nx) for _ in range(n_nodes)]


def _mock_densities(n_nodes: int, Nt: int = 6, Nx: int = 11) -> list[np.ndarray]:
    """Create mock density arrays (non-negative)."""
    rng = np.random.RandomState(7)
    return [np.abs(rng.randn(Nt, Nx)) + 0.1 for _ in range(n_nodes)]


class TestGraphCouplingProtocol:
    def test_adjacency_satisfies_protocol(self):
        A = _ring_adjacency(3)
        coupling = AdjacencyCoupling(A)
        assert isinstance(coupling, GraphCouplingOperator)

    def test_laplacian_satisfies_protocol(self):
        A = _ring_adjacency(3)
        coupling = LaplacianCoupling(A)
        assert isinstance(coupling, GraphCouplingOperator)


class TestAdjacencyCoupling:
    def test_n_nodes(self):
        A = _ring_adjacency(4)
        coupling = AdjacencyCoupling(A)
        assert coupling.n_nodes == 4

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            AdjacencyCoupling(np.zeros((3, 4)))

    def test_hjb_source_returns_callable(self):
        A = _ring_adjacency(3)
        coupling = AdjacencyCoupling(A, alpha=0.1)
        values = _mock_values(3)
        densities = _mock_densities(3)
        source = coupling.compute_hjb_source(0, values, densities, dt=0.05)
        assert callable(source)

    def test_hjb_source_shape(self):
        A = _ring_adjacency(3)
        coupling = AdjacencyCoupling(A, alpha=0.1)
        values = _mock_values(3, Nx=21)
        densities = _mock_densities(3, Nx=21)
        source = coupling.compute_hjb_source(0, values, densities, dt=0.05)
        x = np.linspace(0, 1, 21)
        result = source(0.0, x)
        assert result.shape == (21,)

    def test_hjb_source_zero_for_identical_values(self):
        """Default coupling is value difference — zero if all nodes same."""
        A = _ring_adjacency(3)
        coupling = AdjacencyCoupling(A, alpha=0.1)
        V = np.ones((6, 11))
        values = [V.copy(), V.copy(), V.copy()]
        densities = _mock_densities(3)
        source = coupling.compute_hjb_source(0, values, densities, dt=0.05)
        result = source(0.0, np.linspace(0, 1, 11))
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_fp_source_shape(self):
        A = _ring_adjacency(3)
        coupling = AdjacencyCoupling(A, beta=0.05)
        values = _mock_values(3, Nx=21)
        densities = _mock_densities(3, Nx=21)
        source = coupling.compute_fp_source(1, values, densities, dt=0.05)
        x = np.linspace(0, 1, 21)
        result = source(0.0, x)
        assert result.shape == (21,)

    def test_fp_source_mass_transfer(self):
        """Mass transfer from high-density neighbor to low-density node."""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        coupling = AdjacencyCoupling(A, beta=1.0)
        values = [np.zeros((6, 11)), np.zeros((6, 11))]
        # Node 0: low density, Node 1: high density
        m0 = np.ones((6, 11)) * 0.1
        m1 = np.ones((6, 11)) * 0.9
        densities = [m0, m1]
        source = coupling.compute_fp_source(0, values, densities, dt=0.05)
        result = source(0.0, np.linspace(0, 1, 11))
        # Inflow to node 0: beta * (m1 - m0) = 1.0 * (0.9 - 0.1) = 0.8
        np.testing.assert_allclose(result, 0.8, atol=1e-10)

    def test_no_self_coupling(self):
        """Node should not couple with itself even if A_{ii} != 0."""
        A = np.eye(3)  # Only self-loops
        coupling = AdjacencyCoupling(A, alpha=1.0)
        values = _mock_values(3)
        densities = _mock_densities(3)
        source = coupling.compute_hjb_source(0, values, densities, dt=0.05)
        result = source(0.0, np.linspace(0, 1, 11))
        # Self-loop skipped (j == i check)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_custom_coupling_function(self):
        A = np.array([[0, 1], [1, 0]], dtype=float)

        def custom_hjb(i, j, v_j, m_j, t):
            return m_j * 2.0  # density-based coupling

        coupling = AdjacencyCoupling(A, coupling_hjb=custom_hjb)
        values = [np.zeros((6, 11)), np.zeros((6, 11))]
        densities = [np.ones((6, 11)), np.ones((6, 11)) * 3.0]
        source = coupling.compute_hjb_source(0, values, densities, dt=0.05)
        result = source(0.0, np.linspace(0, 1, 11))
        # A[0,1] * custom(0, 1, v_1, m_1=3, t=0) = 1 * 3 * 2 = 6
        np.testing.assert_allclose(result, 6.0, atol=1e-10)


class TestLaplacianCoupling:
    def test_n_nodes(self):
        A = _ring_adjacency(5)
        coupling = LaplacianCoupling(A)
        assert coupling.n_nodes == 5

    def test_laplacian_structure(self):
        """L = D - A should have zero row sums."""
        A = _ring_adjacency(3)
        coupling = LaplacianCoupling(A)
        row_sums = coupling._L.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_hjb_source_zero_for_uniform(self):
        """Laplacian of constant function is zero."""
        A = _ring_adjacency(3)
        coupling = LaplacianCoupling(A, kappa=1.0)
        V = np.ones((6, 11)) * 5.0
        values = [V.copy(), V.copy(), V.copy()]
        densities = _mock_densities(3)
        source = coupling.compute_hjb_source(0, values, densities, dt=0.05)
        result = source(0.0, np.linspace(0, 1, 11))
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_fp_source_conservation(self):
        """Total FP source across all nodes should sum to zero (mass conservation)."""
        A = _ring_adjacency(3)
        coupling = LaplacianCoupling(A, kappa=0.5)
        values = _mock_values(3, Nx=11)
        densities = _mock_densities(3, Nx=11)
        x = np.linspace(0, 1, 11)

        total = np.zeros(11)
        for i in range(3):
            source = coupling.compute_fp_source(i, values, densities, dt=0.05)
            total += source(0.0, x)

        # L has zero row sums -> sum of all sources = kappa * sum_i L_i @ m = 0
        # (when all nodes have same Nx)
        np.testing.assert_allclose(total, 0.0, atol=1e-10)

    def test_1d_density_input(self):
        """Should handle (Nx,) density arrays (no time dimension)."""
        A = _ring_adjacency(2)
        coupling = LaplacianCoupling(A, kappa=1.0)
        values = [np.zeros(11), np.ones(11)]
        densities = [np.ones(11) * 0.5, np.ones(11) * 1.5]
        source = coupling.compute_fp_source(0, values, densities, dt=0.05)
        result = source(0.0, np.linspace(0, 1, 11))
        assert result.shape == (11,)
        assert np.all(np.isfinite(result))


class TestGetTimeSliceRegression:
    """Regression tests for Issue #1006 Bug 1: hardcoded dt=0.05.

    Before the fix, `_get_time_slice` had dt=0.05 as a default and was called
    without dt argument in all coupling implementations, giving wrong array
    indices for any problem with non-0.05 time step.
    """

    def test_uses_provided_dt_not_hardcoded(self):
        """With dt=0.01, t=0.25 must index row 25 (not row 5 as before the fix)."""
        arr = np.arange(60).reshape(30, 2).astype(float)  # 30 time rows
        sl = _get_time_slice(arr, t=0.25, dt=0.01)
        np.testing.assert_array_equal(sl, arr[25])

    def test_different_dt_gives_different_index(self):
        """Same t, different dt must give different rows."""
        arr = np.arange(200).reshape(20, 10).astype(float)
        sl_05 = _get_time_slice(arr, t=0.5, dt=0.05)  # index 10
        sl_10 = _get_time_slice(arr, t=0.5, dt=0.10)  # index 5
        assert not np.array_equal(sl_05, sl_10)
        np.testing.assert_array_equal(sl_05, arr[10])
        np.testing.assert_array_equal(sl_10, arr[5])

    def test_index_clamped_to_last_row(self):
        """t beyond array horizon clamps to last row, not out of bounds."""
        arr = np.arange(20).reshape(5, 4).astype(float)
        sl = _get_time_slice(arr, t=100.0, dt=0.05)
        np.testing.assert_array_equal(sl, arr[-1])

    def test_negative_time_clamps_to_zero(self):
        """Negative t clamps to index 0 (round(-0.01/0.05) = 0)."""
        arr = np.arange(20).reshape(5, 4).astype(float)
        sl = _get_time_slice(arr, t=-0.01, dt=0.05)
        np.testing.assert_array_equal(sl, arr[0])

    def test_zero_dt_returns_first_row(self):
        """dt <= 0 returns arr[0] as safe fallback."""
        arr = np.arange(20).reshape(5, 4).astype(float)
        sl = _get_time_slice(arr, t=0.5, dt=0.0)
        np.testing.assert_array_equal(sl, arr[0])

    def test_1d_passthrough_ignores_dt(self):
        """1D array returns itself regardless of dt."""
        arr = np.arange(10).astype(float)
        sl = _get_time_slice(arr, t=0.5, dt=0.01)
        np.testing.assert_array_equal(sl, arr)
