"""
Test that WENO solver uses high-order ghost cells (Issue #576, Phase 6).

Verifies that HJBWENOSolver correctly uses order=5 polynomial extrapolation
for ghost cell generation, enabling true 5th-order boundary accuracy.
"""

import numpy as np

from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
        u_final=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


def test_weno_uses_high_order_ghosts():
    """Test that WENO solver creates ghost buffer with order=5."""
    # Create a simple 1D MFG problem using modern geometry-first API
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[50], boundary_conditions=no_flux_bc(dimension=1))

    problem = MFGProblem(
        geometry=domain,
        T=1.0,
        Nt=10,
        diffusion=0.1,
        components=_default_components(),
    )

    # Import WENO solver
    from mfg_pde.alg.numerical.hjb_solvers import HJBWenoSolver

    # Create WENO solver
    solver = HJBWenoSolver(problem)

    # Check that ghost buffer was created with correct parameters
    assert solver.ghost_buffer is not None, "WENO should create ghost buffer in 1D"
    assert solver.ghost_depth == 2, "WENO5 needs 2 ghost cells per side"
    assert solver.ghost_order == 5, "WENO5 should use order=5 for high-order accuracy"
    assert solver.ghost_buffer._order == 5, "Ghost buffer should have order=5"
    assert solver.ghost_buffer._ghost_depth == 2, "Ghost buffer should have depth=2"

    print(f"✓ WENO ghost buffer: depth={solver.ghost_depth}, order={solver.ghost_order}")


def test_weno_ghost_cells_work():
    """Test that WENO can update ghost cells with polynomial extrapolation."""
    # Create a simple 1D MFG problem using modern geometry-first API
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[50], boundary_conditions=no_flux_bc(dimension=1))

    problem = MFGProblem(
        geometry=domain,
        T=1.0,
        Nt=10,
        diffusion=0.1,
        components=_default_components(),
    )

    # Import WENO solver
    from mfg_pde.alg.numerical.hjb_solvers import HJBWenoSolver

    # Create WENO solver
    solver = HJBWenoSolver(problem)

    # Test ghost cell update with smooth function
    x = np.linspace(0.0, 1.0, solver.num_grid_points_x)
    u = x**3  # Cubic function

    # Update ghost cells
    solver.ghost_buffer.interior[:] = u
    solver.ghost_buffer.update_ghosts(time=0.0)

    # Check that ghost cells were filled with reasonable values
    ghost_0 = solver.ghost_buffer.padded[0]
    ghost_1 = solver.ghost_buffer.padded[1]

    # For x^3 with Neumann BC at x=0 (du/dx = 0), ghosts should be negative of interior
    # due to reflection about the zero-derivative point
    dx = 1.0 / (solver.num_grid_points_x - 1)

    # Ghost cells should not be NaN or infinite
    assert not np.isnan(ghost_0), "Ghost cell 0 should be valid"
    assert not np.isnan(ghost_1), "Ghost cell 1 should be valid"
    assert not np.isinf(ghost_0), "Ghost cell 0 should be finite"
    assert not np.isinf(ghost_1), "Ghost cell 1 should be finite"

    # Check that ghost values are reasonable (should be close to extrapolated values)
    # For cubic with zero derivative at x=0:
    # Ghost cell ordering: ghost[0] is furthest from boundary, ghost[1] is nearest
    # ghost[0] at x=-2*dx, ghost[1] at x=-dx
    x_ghost_0 = -2 * dx  # Furthest ghost cell
    x_ghost_1 = -dx  # Nearest ghost cell
    u_exact_ghost_0 = x_ghost_0**3
    u_exact_ghost_1 = x_ghost_1**3

    # Order-5 extrapolation should be reasonably accurate for cubics
    error_0 = np.abs(ghost_0 - u_exact_ghost_0)
    error_1 = np.abs(ghost_1 - u_exact_ghost_1)

    print(f"  Ghost[0] @ x={x_ghost_0:.6f}: computed={ghost_0:.6f}, exact={u_exact_ghost_0:.6f}, error={error_0:.6e}")
    print(f"  Ghost[1] @ x={x_ghost_1:.6f}: computed={ghost_1:.6f}, exact={u_exact_ghost_1:.6f}, error={error_1:.6e}")

    # Should be reasonably accurate (allow for some extrapolation error)
    # For a cubic function, order-5 extrapolation should give good accuracy
    assert error_0 < 1e-4, f"Ghost 0 error too large: {error_0}"
    assert error_1 < 1e-4, f"Ghost 1 error too large: {error_1}"

    print("✓ WENO high-order ghost cells work correctly")


if __name__ == "__main__":
    """Run smoke tests."""
    print("Testing WENO high-order ghost cell integration...")

    print("\n1. Check WENO ghost buffer configuration:")
    test_weno_uses_high_order_ghosts()

    print("\n2. Verify ghost cell update works:")
    test_weno_ghost_cells_work()

    print("\n✅ All WENO ghost integration tests passed!")
