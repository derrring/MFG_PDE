"""
Quick verification that mass conservation fix works.
"""

import numpy as np

from benchmarks.validation.test_2d_crowd_motion import CrowdMotion2D, run_fdm_solver
from benchmarks.validation.utils import compute_mass_conservation

print("\n" + "=" * 70)
print("  VERIFICATION: Mass Conservation Fix")
print("=" * 70 + "\n")

# Test with 8×8 grid (where we observed 99.96% error)
print("Testing 8×8 grid (original failure case)...")
problem = CrowdMotion2D(grid_resolution=8, time_horizon=0.4, num_timesteps=10)

# Get grid spacing and volume element
dx, dy = problem.geometry.grid.spacing
dV = dx * dy
print(f"Grid spacing: dx = {dx:.6f}, dy = {dy:.6f}")
print(f"Volume element: dV = {dV:.6f}")

# Get initial density at all grid points
points = problem.geometry.grid.flatten()
initial_m = problem.initial_density(points)

# Reshape to 2D grid
nx, ny = problem.geometry.grid.num_points
m_grid = initial_m.reshape((nx, ny))

# Compute initial mass
initial_mass = float(np.sum(m_grid) * dV)
print("\nInitial density:")
print(f"  sum(m) = {np.sum(m_grid):.6f}")
print(f"  ∫∫ m dx dy = {initial_mass:.6f}")
print("  Expected: 1.0")
print(f"  Error: {abs(initial_mass - 1.0) * 100:.2f}%")

if abs(initial_mass - 1.0) < 0.01:  # Less than 1% error
    print("\n✅ Initial mass conservation: PASS")
else:
    print(f"\n❌ Initial mass conservation: FAIL (error = {abs(initial_mass - 1.0) * 100:.2f}%)")

# Now run a short solve to check final mass
print("\n" + "-" * 70)
print("Running short solve to check final mass conservation...")
result = run_fdm_solver(grid_resolution=8, num_timesteps=10, max_iterations=10)

# Check final mass
M_final = result["M"][-1]
dx, dy = result["grid_spacing"]
final_mass_error = compute_mass_conservation(M_final, dx=(dx, dy), target_mass=1.0)

print("\nAfter solve:")
print(f"  Converged: {result['info']['converged']}")
print(f"  Iterations: {result['info']['num_iterations']}")
print(f"  Final mass error: {final_mass_error * 100:.2f}%")
print("  Expected: < 5% for FDM dimensional splitting")

if final_mass_error < 0.05:
    print("\n✅ Final mass conservation: PASS")
else:
    print(f"\n❌ Final mass conservation: FAIL (error = {final_mass_error * 100:.2f}%)")

print("\n" + "=" * 70)
print("  VERIFICATION COMPLETE")
print("=" * 70)
