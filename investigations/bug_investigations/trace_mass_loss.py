"""
Diagnostic Script for Bug #8: Mass Loss Investigation

Traces mass evolution timestep by timestep to identify where mass is lost.
"""

import numpy as np

from benchmarks.validation.test_2d_crowd_motion import CrowdMotion2D
from mfg_pde.factory import create_basic_solver

print("=" * 70)
print("  Bug #8 Investigation: Tracing Mass Evolution")
print("=" * 70)
print()

# Small problem for quick diagnosis
problem = CrowdMotion2D(
    grid_resolution=8,  # 8×8 grid
    time_horizon=0.4,
    num_timesteps=10,  # Fewer timesteps for detailed analysis
)

solver = create_basic_solver(
    problem,
    damping=0.6,
    max_iterations=30,
    tolerance=1e-4,
)

print(f"Problem: {8}×{8} grid, T={0.4}, {10} timesteps")
print(f"Grid spacing: dx={problem.geometry.grid.spacing[0]:.4f}, dy={problem.geometry.grid.spacing[1]:.4f}")
print()
print("Running single Picard iteration to trace FP solver...")
print()

# Run solver
result = solver.solve()

# Analyze mass evolution
dx, dy = problem.geometry.grid.spacing
dV = dx * dy

print("Mass Evolution Analysis:")
print("-" * 60)
print("Time      Mass          Loss(%)      M_min        M_max")
print("-" * 60)

initial_mass = np.sum(result.M[0]) * dV

for t_idx in range(len(result.M)):
    mass_t = np.sum(result.M[t_idx]) * dV
    loss_percent = (initial_mass - mass_t) / initial_mass * 100
    m_min = np.min(result.M[t_idx])
    m_max = np.max(result.M[t_idx])

    t_value = t_idx * problem.dt
    print(f"{t_value:5.3f}    {mass_t:10.6f}    {loss_percent:6.2f}%      {m_min:8.4f}     {m_max:8.4f}")

print("-" * 60)
print()

# Check for negative densities
has_negative = np.any(result.M < 0)
if has_negative:
    print("⚠ WARNING: Negative densities detected!")
    print(f"   Min density: {np.min(result.M):.6e}")
    print(f"   Number of negative values: {np.sum(result.M < 0)}")
else:
    print("✓ All densities are non-negative")

print()

# Compute mass loss rate
if len(result.M) > 1:
    mass_losses = []
    for t_idx in range(len(result.M) - 1):
        mass_before = np.sum(result.M[t_idx]) * dV
        mass_after = np.sum(result.M[t_idx + 1]) * dV
        loss_per_step = (mass_before - mass_after) / mass_before * 100
        mass_losses.append(loss_per_step)

    print("Mass Loss Per Timestep:")
    print(f"  Average: {np.mean(mass_losses):.3f}%")
    print(f"  Max: {np.max(mass_losses):.3f}%")
    print(f"  Min: {np.min(mass_losses):.3f}%")
    print()

    # Check if loss is constant or increasing
    if len(mass_losses) > 5:
        early_avg = np.mean(mass_losses[:3])
        late_avg = np.mean(mass_losses[-3:])
        print(f"  Early timesteps (avg): {early_avg:.3f}%")
        print(f"  Late timesteps (avg): {late_avg:.3f}%")

        if late_avg > early_avg * 1.2:
            print("  → Loss INCREASES over time (accumulates)")
        elif late_avg < early_avg * 0.8:
            print("  → Loss DECREASES over time (stabilizes)")
        else:
            print("  → Loss is CONSTANT (systematic)")

print()
print("=" * 70)
print("  Analysis Summary")
print("=" * 70)
print()

total_loss = (initial_mass - np.sum(result.M[-1]) * dV) / initial_mass * 100
print(f"Total mass loss: {total_loss:.2f}%")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print()

# Hypothesis testing
if total_loss > 80:
    print("HYPOTHESIS: Severe mass loss (>80%)")
    print("  Most likely cause: Incorrect boundary flux formulation")
    print("  Check: FP solver boundary matrix coefficients")
elif total_loss > 20:
    print("HYPOTHESIS: Moderate mass loss (>20%)")
    print("  Possible causes:")
    print("  1. Numerical diffusion from upwind scheme")
    print("  2. Boundary flux imbalance")
    print("  3. Dimensional splitting error")
elif total_loss > 1:
    print("HYPOTHESIS: Small mass loss (>1%)")
    print("  Possible causes:")
    print("  1. Time integration error")
    print("  2. Iterative solver tolerance")
else:
    print("RESULT: Mass is well conserved!")
    print("  No action needed")

print()
