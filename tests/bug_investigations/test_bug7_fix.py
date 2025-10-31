"""
Test Bug #7 Fix: Time-Index Mismatch in MFG Coupling

Expected outcome: Picard iterations should converge now that M(t_n) is used
instead of M(t_{n+1}) when solving HJB at time t_n.
"""

import numpy as np

from benchmarks.validation.test_2d_crowd_motion import CrowdMotion2D
from mfg_pde.factory import create_basic_solver

print("=" * 70)
print("  Testing Bug #7 Fix: Time-Index Correction in HJB-MFG Coupling")
print("=" * 70)
print()
print("Fix: Changed base_hjb.py:763 from M[n+1] to M[n]")
print()
print("Test problem: 16×16 grid, T=0.4, 20 timesteps")
print("Expected: Monotonic error decrease, convergence in < 50 iterations")
print()

problem = CrowdMotion2D(
    grid_resolution=16,
    time_horizon=0.4,
    num_timesteps=20,
)

solver = create_basic_solver(
    problem,
    damping=0.6,
    max_iterations=50,  # Should converge well before this
    tolerance=1e-4,
)

print("Running solver with Bug #7 fix...")
print()

result = solver.solve()

print()
print("=" * 70)
print("  Results")
print("=" * 70)
print()
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final U error: {result.metadata['l2distu_rel'][-1]:.6e}")
print(f"Final M error: {result.metadata['l2distm_rel'][-1]:.6e}")

# Check convergence pattern
errors_u = result.metadata["l2distu_rel"]
errors_m = result.metadata["l2distm_rel"]

print()
print("Convergence history (showing every 5th iteration):")
print("Iter    U_err       M_err")
print("-" * 40)
for i in range(0, len(errors_u), 5):
    print(f"{i + 1:4d}  {errors_u[i]:.6e}  {errors_m[i]:.6e}")
if len(errors_u) % 5 != 1:  # Show last iteration if not already shown
    i = len(errors_u) - 1
    print(f"{i + 1:4d}  {errors_u[i]:.6e}  {errors_m[i]:.6e}")

# Check mass conservation
dx, dy = problem.geometry.grid.spacing
dV = dx * dy
initial_mass = np.sum(result.M[0]) * dV
final_mass = np.sum(result.M[-1]) * dV
mass_loss_percent = abs(final_mass - initial_mass) / initial_mass * 100

print()
print("Mass conservation:")
print(f"  Initial: {initial_mass:.6f}")
print(f"  Final: {final_mass:.6f}")
print(f"  Loss: {mass_loss_percent:.2f}%")

print()
print("=" * 70)
print("  Analysis")
print("=" * 70)
print()

if result.converged:
    print(f"✓ SUCCESS: Solver converged in {result.iterations} iterations")
    print("  Bug #7 fix appears to have resolved the oscillation issue")
else:
    print(f"✗ FAILURE: Solver did not converge after {result.iterations} iterations")

    # Check if errors are decreasing
    if len(errors_u) > 10:
        early_u_err = np.mean(errors_u[0:5])
        late_u_err = np.mean(errors_u[-5:])

        if late_u_err < early_u_err * 0.5:
            print("  Errors ARE decreasing - may need more iterations")
        else:
            print("  Errors NOT decreasing consistently - bug may not be fully fixed")

print()
print("=" * 70)
