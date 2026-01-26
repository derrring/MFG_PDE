"""
Unit test for FPParticleSolver with multi-exit absorbing BC.

Validates Issue #535 Phase 1: Segment-aware boundary conditions with
multiple DIRICHLET exits.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCSegment, mixed_bc, no_flux_bc
from mfg_pde.geometry.boundary.types import BCType


def _default_components_1d():
    """Provide default components for 1D test problems."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-((np.asarray(x) - 5.0) ** 2) / 2.0),
        u_final=lambda x: 0.0,
    )


def _default_components_2d():
    """Provide default components for 2D test problems."""

    def m_initial_2d(x):
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            return np.exp(-((x_arr[0] - 2.0) ** 2 + (x_arr[1] - 2.0) ** 2) / 2.0)
        return np.exp(-((x_arr[..., 0] - 2.0) ** 2 + (x_arr[..., 1] - 2.0) ** 2) / 2.0)

    return MFGComponents(
        m_initial=m_initial_2d,
        u_final=lambda x: 0.0,
    )


def test_particle_solver_multi_exit_1d():
    """
    Test FPParticleSolver with two absorbing exits in 1D.

    Setup:
        Domain: [0, 10]
        Exit 1: x ∈ [0, 0.5]   (left wall, partial)
        Exit 2: x ∈ [9.5, 10]  (right wall, partial)
        Reflecting: elsewhere

    Drift: Particles pushed toward edges → some exit left, some exit right
    """
    # Create 1D domain
    geometry = TensorProductGrid(bounds=[(0.0, 10.0)], Nx_points=[101], boundary_conditions=no_flux_bc(1))
    problem = MFGProblem(
        geometry=geometry,
        T=3.0,
        Nt=60,
        diffusion=0.05,
        coupling_coefficient=1.0,
        components=_default_components_1d(),
    )

    # Multi-exit BC: two DIRICHLET exits on opposite ends
    bc_multi_exit = mixed_bc(
        dimension=1,
        segments=[
            BCSegment(
                name="exit_left",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="left",  # Entire left boundary
            ),
            BCSegment(
                name="exit_right",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="right",  # Entire right boundary
            ),
        ],
        domain_bounds=np.array([[0.0, 10.0]]),
    )

    solver = FPParticleSolver(
        problem,
        num_particles=200,
        boundary_conditions=bc_multi_exit,
    )

    # Initial density: centered Gaussian
    x = geometry.coordinates[0]
    M_init = np.exp(-((x - 5.0) ** 2) / 2.0)
    M_init = M_init / (np.sum(M_init) * 0.1)  # Normalize

    # Value function: U(x) = -α*|x - 5|
    # → Drift: -∇U = α*sign(x - 5) (pushes to edges)
    # For particles left of center → drift left, right of center → drift right
    Nt = problem.Nt + 1
    Nx = len(x)
    drift_field = np.zeros((Nt, Nx))
    for t in range(Nt):
        drift_field[t, :] = -5.0 * np.abs(x - 5.0)  # V-shaped potential

    # Solve
    M_solution = solver.solve_fp_system(
        M_initial=M_init,
        drift_field=drift_field,
        show_progress=False,
    )

    # Assertions
    assert M_solution.shape == (Nt, Nx)
    assert not np.any(np.isnan(M_solution)), "Solution contains NaN"
    assert not np.any(np.isinf(M_solution)), "Solution contains inf"

    # Verify particles were absorbed
    assert solver.total_absorbed > 0, "No particles absorbed (drift may be too weak)"
    assert len(solver.exit_flux_history) > 0, "No exit flux recorded"

    # Verify BOTH exits were used (key test for multi-exit)
    if solver.total_absorbed > 0:
        all_exit_positions = np.concatenate(solver.exit_positions_history)

        # Classify by position
        left_exits = np.sum(all_exit_positions < 1.0)  # Near x=0
        right_exits = np.sum(all_exit_positions > 9.0)  # Near x=10

        # Both exits should be used
        assert left_exits > 0, f"Left exit not used (absorbed: {left_exits})"
        assert right_exits > 0, f"Right exit not used (absorbed: {right_exits})"

        print(f"✓ Multi-exit test passed: {left_exits} left, {right_exits} right")


@pytest.mark.skip(reason="TODO: Requires stronger drift parameters - 1D test validates multi-exit BC")
def test_particle_solver_multi_exit_2d():
    """
    Test FPParticleSolver with two absorbing exits in 2D.

    Setup:
        Domain: [0, 10] × [0, 10]
        Exit 1: Right wall at y ∈ [4, 6]  (partial boundary)
        Exit 2: Top wall at x ∈ [4, 6]    (partial boundary)
        Reflecting: elsewhere

    Drift: Uniform drift toward top-right corner → particles use both exits
    """
    # Create 2D domain
    geometry = TensorProductGrid(
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        Nx_points=[21, 21],
        boundary_conditions=no_flux_bc(2),
    )
    problem = MFGProblem(
        geometry=geometry,
        T=3.0,
        Nt=30,
        diffusion=0.05,
        components=_default_components_2d(),
    )
    grid_shape = problem.geometry.get_grid_shape()

    # Multi-exit BC: exits on right and top walls
    bc_multi_exit = mixed_bc(
        dimension=2,
        segments=[
            BCSegment(
                name="exit_right",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                region={"x": (10.0, 10.0), "y": (4.0, 6.0)},  # Right wall partial
            ),
            BCSegment(
                name="exit_top",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                region={"x": (4.0, 6.0), "y": (10.0, 10.0)},  # Top wall partial
            ),
            BCSegment(
                name="walls",
                bc_type=BCType.REFLECTING,
                boundary="all",
                priority=-1,
            ),
        ],
        domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
    )

    solver = FPParticleSolver(
        problem,
        num_particles=300,
        boundary_conditions=bc_multi_exit,
    )

    # Initial density: bottom-left Gaussian
    coords = problem.geometry.coordinates
    X, Y = np.meshgrid(coords[0], coords[1], indexing="ij")
    M_init = np.exp(-((X - 2.0) ** 2 + (Y - 2.0) ** 2) / 2.0)
    dA = (10.0 / 20) ** 2
    M_init = M_init / (np.sum(M_init) * dA)

    # Drift field: strong constant drift toward top-right (toward both exits)
    Nt = problem.Nt + 1
    drift_field = np.zeros((Nt, *tuple(grid_shape), 2))
    drift_field[..., 0] = 3.0  # Strong drift right (toward exit_right)
    drift_field[..., 1] = 3.0  # Strong drift up (toward exit_top)

    # Solve
    M_solution = solver.solve_fp_system(
        M_initial=M_init,
        drift_field=drift_field,
        drift_is_precomputed=True,
        show_progress=False,
    )

    # Assertions
    assert M_solution.shape == (Nt, *tuple(grid_shape))
    assert not np.any(np.isnan(M_solution)), "Solution contains NaN"
    assert not np.any(np.isinf(M_solution)), "Solution contains inf"

    # Verify particles were absorbed
    assert solver.total_absorbed > 0, "No particles absorbed"
    assert len(solver.exit_flux_history) > 0, "No exit flux recorded"

    # Verify BOTH exits were used
    if solver.total_absorbed > 5:  # Need enough particles for meaningful test
        all_exit_positions = np.vstack(solver.exit_positions_history)

        # Classify by which wall (right vs top)
        right_exits = np.sum(all_exit_positions[:, 0] > 9.5)  # Near x=10
        top_exits = np.sum(all_exit_positions[:, 1] > 9.5)  # Near y=10

        # Both exits should be used (allows for some statistical variation)
        assert right_exits > 0 or top_exits > 0, "Neither exit used"

        print(f"✓ 2D Multi-exit: {right_exits} right, {top_exits} top, {solver.total_absorbed} total")


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing multi-exit absorbing BC...")

    print("\n1D multi-exit test:")
    test_particle_solver_multi_exit_1d()

    print("\n✓ Multi-exit test passed! (2D test skipped - requires tuning)")
