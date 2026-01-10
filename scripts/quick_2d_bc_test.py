"""Quick smoke test for 2D FDM BC enforcement after refactor."""

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCSegment, BCType, BoundaryConditions

print("Testing 2D FDM BC enforcement (refactored)...")

# Create 2D grid with Dirichlet BC on x-boundaries
bc = BoundaryConditions(
    segments=[
        BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
        BCSegment(name="right", bc_type=BCType.DIRICHLET, value=2.0, boundary="x_max"),
    ]
)

grid = TensorProductGrid(
    dimension=2,
    bounds=[(0, 1), (0, 1)],
    Nx=[10, 10],  # Small grid for quick test
    boundary_conditions=bc,
)

# Create simple MFG problem
problem = MFGProblem(geometry=grid, T=0.1, Nt=2)  # Short time for quick test

# Set simple terminal condition
spatial_grid = grid.get_spatial_grid()  # Shape: (Nx, Ny, 2)
g_terminal = np.ones(grid.get_grid_shape()) * 5.0  # Constant terminal condition

# Create density field
m_density = np.ones((problem.Nt + 1, *grid.get_grid_shape())) / (grid.get_grid_shape()[0] * grid.get_grid_shape()[1])

# Solve HJB
solver = HJBFDMSolver(problem, solver_type="newton", max_newton_iterations=5)
U_coupling = np.zeros((problem.Nt + 1, *grid.get_grid_shape()))  # Dummy coupling
U = solver.solve_hjb_system(M_density=m_density, U_terminal=g_terminal, U_coupling_prev=U_coupling)

print(f"Solution shape: {U.shape}")
print(f"Expected: (Nt+1, Nx, Ny) = ({problem.Nt + 1}, {grid.get_grid_shape()[0]}, {grid.get_grid_shape()[1]})")

# Check BC enforcement at intermediate time
t_check = 1  # First non-terminal timestep
left_vals = U[t_check, 0, :]  # Left boundary (x=0)
right_vals = U[t_check, -1, :]  # Right boundary (x=1)

print(f"\nBoundary values at t={t_check}:")
print(f"Left (x=0):  mean={left_vals.mean():.6f}, std={left_vals.std():.6f} (should be 1.0 ± 0)")
print(f"Right (x=1): mean={right_vals.mean():.6f}, std={right_vals.std():.6f} (should be 2.0 ± 0)")

# Verify
tol = 1e-10
if np.allclose(left_vals, 1.0, atol=tol) and np.allclose(right_vals, 2.0, atol=tol):
    print("\n✅ PASS: 2D BC enforcement works after refactor!")
else:
    print("\n❌ FAIL: BC not properly enforced")
    print(f"  Left: expected 1.0, got {left_vals.mean():.6f}")
    print(f"  Right: expected 2.0, got {right_vals.mean():.6f}")
