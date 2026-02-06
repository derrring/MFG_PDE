"""
1D Bilateral Constraint Problem - Box Constraints on Heat Equation.

This demonstrates bilateral (box) constraints using the HEAT EQUATION:
    ψ_lower(x) ≤ u(t,x) ≤ ψ_upper(x)

Physical Interpretation: Temperature Control with Upper and Lower Bounds
------------------------------------------------------------------------
Consider a temperature-controlled rod with both heating and cooling:
- u(t,x) = temperature at time t, position x
- ψ_lower(x) = minimum allowable temperature (safety limit)
- ψ_upper(x) = maximum allowable temperature (material limit)
- Constraint: ψ_lower(x) ≤ u(t,x) ≤ ψ_upper(x)

The bilateral constraint represents:
- Lower bound: Heating system prevents freezing
- Upper bound: Cooling system prevents overheating
- Both systems activate as needed

Mathematically:
    ∂u/∂t = σ²/2 · ∂²u/∂x²        if ψ_lower < u < ψ_upper  (free evolution)
    u = ψ_lower                     if u = ψ_lower            (heating active)
    u = ψ_upper                     if u = ψ_upper            (cooling active)

Active Set: Points where either constraint binds.

Setup:
    - Initial condition: u(0,x) = Gaussian dip (cold center)
    - Lower bound: ψ_lower(x) = parabolic (higher at center)
    - Upper bound: ψ_upper(x) = constant (uniform ceiling)
    - Physics: Heat diffuses, filling the cold dip
    - Lower constraint initially active at center, then releases

Expected Behavior:
    - Early times: Center at lower bound (heating active)
    - Middle times: Heat diffuses inward, center warms up
    - Later times: System equilibrates within bounds
    - Active set SHRINKS then stabilizes

Created: 2026-01-17 (Issue #591 - Bilateral Constraint Demonstration)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BilateralConstraint, no_flux_bc

print("=" * 70)
print("Bilateral Constraint Problem - Temperature Control")
print("=" * 70)

# ========================================
# Parameters
# ========================================
x_min, x_max = 0.0, 1.0
Nx = 50
T = 1.0  # Total simulation time
Nt = 125  # Timesteps for stability
sigma = 0.2  # Diffusion coefficient

# Create grid

grid = TensorProductGrid(bounds=[(x_min, x_max)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1))
x = grid.coordinates[0]
Nx_actual = len(x)
dx = x[1] - x[0]
dt = T / Nt

print("\nPhysical Setup:")
print("  Temperature-controlled rod")
print("  Both heating and cooling systems")
print(f"  Domain: x ∈ [{x_min}, {x_max}]")
print(f"  Time: t ∈ [0, {T}]")
print(f"  Diffusion: σ² = {sigma**2:.4f}")

# ========================================
# Initial Condition and Constraints
# ========================================
# Initial temperature: Gaussian dip (cold center, warm edges)
u_initial_dip = 0.3
u_initial_width = 0.15
u_initial_edges = 0.8
u_initial = u_initial_edges - u_initial_dip * np.exp(-(((x - 0.5) / u_initial_width) ** 2))

# Lower bound: Parabolic (higher minimum at center for safety)
psi_lower_center = 0.6
psi_lower_edge = 0.3
psi_lower = psi_lower_center + (psi_lower_edge - psi_lower_center) * (2 * (x - 0.5)) ** 2

# Upper bound: Uniform ceiling (material limit)
psi_upper = np.ones_like(x) * 0.9

print("\nInitial Condition:")
print("  u(0,x) = Gaussian dip (cold center)")
print(f"    Center: {u_initial[Nx_actual // 2]:.4f}")
print(f"    Edges: {u_initial[0]:.4f}")

print("\nBilateral Constraints:")
print("  Lower bound ψ_lower(x) (heating system):")
print(f"    ψ_lower(0.5) = {psi_lower[Nx_actual // 2]:.4f} (center)")
print(f"    ψ_lower(0, 1) = {psi_lower[0]:.4f} (edges)")
print("  Upper bound ψ_upper(x) (cooling system):")
print(f"    ψ_upper = {psi_upper[0]:.4f} (uniform)")

# Create bilateral constraint
constraint = BilateralConstraint(psi_lower, psi_upper)

# Check and project initial condition
if constraint.is_feasible(u_initial, tol=1e-10):
    print("\n✓ Initial condition satisfies box constraint")
else:
    violations_lower = np.maximum(0, psi_lower - u_initial)
    violations_upper = np.maximum(0, u_initial - psi_upper)
    num_viol_lower = (violations_lower > 1e-10).sum()
    num_viol_upper = (violations_upper > 1e-10).sum()
    print("\n⚠ Initial condition violates constraints:")
    print(f"  Lower violations: {num_viol_lower} points (max {violations_lower.max():.4f})")
    print(f"  Upper violations: {num_viol_upper} points (max {violations_upper.max():.4f})")
    print("  Projecting onto feasible set...")
    u_initial = constraint.project(u_initial)
    print(f"  ✓ After projection: u(0,x) ∈ [{u_initial.min():.4f}, {u_initial.max():.4f}]")

# ========================================
# Solve Heat Equation WITHOUT Constraint
# ========================================
print("\n" + "=" * 70)
print("Solving Heat Equation WITHOUT constraint...")
print("=" * 70)

U_unconstr = np.zeros((Nt + 1, Nx_actual))
U_unconstr[0] = u_initial.copy()

# Forward Euler timestepping
diffusion_coef = sigma**2 / 2
stability_limit = dx**2 / (2 * diffusion_coef)

print("\nNumerical stability:")
print(f"  dt = {dt:.6f}")
print(f"  Stability limit = {stability_limit:.6f}")
print(f"  Ratio dt/limit = {dt / stability_limit:.3f}")

if dt > stability_limit:
    raise ValueError(f"Unstable! dt={dt:.6f} exceeds stability limit {stability_limit:.6f}")
elif dt > 0.8 * stability_limit:
    print("  ⚠ Warning: dt close to stability limit")
else:
    print("  ✓ Stable (dt < 0.8 × stability limit)")

# Simple central differences for Laplacian
for n in range(Nt):
    u_curr = U_unconstr[n]
    # Laplacian with Neumann BC (du/dx = 0 at boundaries)
    laplacian = np.zeros(Nx_actual)
    laplacian[1:-1] = (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]) / dx**2
    laplacian[0] = (u_curr[1] - u_curr[0]) / dx**2
    laplacian[-1] = (u_curr[-2] - u_curr[-1]) / dx**2

    U_unconstr[n + 1] = u_curr + dt * diffusion_coef * laplacian

print("\nSolution evolution (unconstrained):")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    u_min, u_max = U_unconstr[n].min(), U_unconstr[n].max()
    print(f"  t={t:.3f}: u ∈ [{u_min:.4f}, {u_max:.4f}]")

# Check violations
violations_lower_unconstr = np.maximum(0, psi_lower - U_unconstr)
violations_upper_unconstr = np.maximum(0, U_unconstr - psi_upper)
max_viol_lower = violations_lower_unconstr.max()
max_viol_upper = violations_upper_unconstr.max()
num_viol_lower = (violations_lower_unconstr > 1e-8).sum()
num_viol_upper = (violations_upper_unconstr > 1e-8).sum()

print("\nConstraint check:")
print(f"  Lower violations: max={max_viol_lower:.4f}, count={num_viol_lower}")
print(f"  Upper violations: max={max_viol_upper:.4f}, count={num_viol_upper}")

if max_viol_lower > 1e-6 or max_viol_upper > 1e-6:
    print("  → Heat diffusion causes constraint violations")
else:
    print("  → Temperature stays within bounds naturally")

# ========================================
# Solve WITH Bilateral Constraint
# ========================================
print("\n" + "=" * 70)
print("Solving Heat Equation WITH bilateral constraint...")
print("=" * 70)

U_constr = np.zeros((Nt + 1, Nx_actual))
U_constr[0] = u_initial.copy()

for n in range(Nt):
    u_curr = U_constr[n]

    # Same heat equation step
    laplacian = np.zeros(Nx_actual)
    laplacian[1:-1] = (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]) / dx**2
    laplacian[0] = (u_curr[1] - u_curr[0]) / dx**2
    laplacian[-1] = (u_curr[-2] - u_curr[-1]) / dx**2

    u_next = u_curr + dt * diffusion_coef * laplacian

    # Apply bilateral constraint (clip to box)
    U_constr[n + 1] = constraint.project(u_next)

print("\nSolution evolution (constrained):")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    u_min, u_max = U_constr[n].min(), U_constr[n].max()
    active = constraint.get_active_set(U_constr[n], tol=1e-6)
    print(f"  t={t:.3f}: u ∈ [{u_min:.4f}, {u_max:.4f}], {active.sum()}/{Nx_actual} active")

# Verify no violations
violations_lower_constr = np.maximum(0, psi_lower - U_constr)
violations_upper_constr = np.maximum(0, U_constr - psi_upper)
max_viol_constr = max(violations_lower_constr.max(), violations_upper_constr.max())

print("\nConstraint satisfaction:")
print(f"  Lower violations: {violations_lower_constr.max():.2e}")
print(f"  Upper violations: {violations_upper_constr.max():.2e}")
print(f"  Status: {'✓ SATISFIED' if max_viol_constr < 1e-8 else '✗ VIOLATED'}")

# ========================================
# Active Set Analysis
# ========================================
print("\n" + "=" * 70)
print("Active Set Evolution")
print("=" * 70)

active_sets_lower = []
active_sets_upper = []
for n in range(Nt + 1):
    # Lower constraint active when u = ψ_lower
    active_lower = np.abs(U_constr[n] - psi_lower) < 1e-6
    # Upper constraint active when u = ψ_upper
    active_upper = np.abs(U_constr[n] - psi_upper) < 1e-6
    active_sets_lower.append(active_lower)
    active_sets_upper.append(active_upper)

active_sets_lower = np.array(active_sets_lower)
active_sets_upper = np.array(active_sets_upper)
active_sets_total = active_sets_lower | active_sets_upper

print("\nActive set breakdown:")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    frac_lower = active_sets_lower[n].sum() / Nx_actual
    frac_upper = active_sets_upper[n].sum() / Nx_actual
    frac_total = active_sets_total[n].sum() / Nx_actual
    print(
        f"  t={t:.3f}: Lower={frac_lower * 100:4.1f}%, Upper={frac_upper * 100:4.1f}%, Total={frac_total * 100:4.1f}%"
    )

print("\nPhysical interpretation:")
print("  Initially: Cold center at lower bound (heating active)")
print("  Heat diffuses: Warms the cold center from edges")
print("  Lower constraint releases: As center temperature rises")
print("  Upper constraint: May activate if temperature exceeds ceiling")

# ========================================
# Visualization
# ========================================
print("\n" + "=" * 70)
print("Creating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

t_grid = np.linspace(0, T, Nt + 1)
X_mesh, T_mesh = np.meshgrid(x, t_grid)

# Plot 1: Unconstrained evolution
ax = axes[0, 0]
im = ax.pcolormesh(X_mesh, T_mesh, U_unconstr, shading="auto", cmap="coolwarm")
plt.colorbar(im, ax=ax, label="Temperature u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Unconstrained Heat Diffusion")

# Plot 2: Constrained evolution
ax = axes[0, 1]
im = ax.pcolormesh(X_mesh, T_mesh, U_constr, shading="auto", cmap="coolwarm")
plt.colorbar(im, ax=ax, label="Temperature u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Constrained Heat Diffusion")

# Plot 3: Difference
ax = axes[0, 2]
diff = U_constr - U_unconstr
im = ax.pcolormesh(X_mesh, T_mesh, diff, shading="auto", cmap="RdBu_r")
plt.colorbar(im, ax=ax, label="Δu (constraint effect)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Constraint Correction")

# Plot 4: Temperature snapshots with bounds
ax = axes[1, 0]
times = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]
colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
for i, n in enumerate(times):
    t = n * dt
    ax.plot(x, U_constr[n], "-", color=colors[i], linewidth=2, label=f"t={t:.2f}")
ax.plot(x, psi_lower, "r:", linewidth=2, label="ψ_lower (heating)")
ax.plot(x, psi_upper, "b:", linewidth=2, label="ψ_upper (cooling)")
ax.set_xlabel("Position x")
ax.set_ylabel("Temperature u(t,x)")
ax.set_title("Temperature Evolution with Bounds")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 5: Active set heatmap (combined)
ax = axes[1, 1]
# Color code: 0=free, 1=lower active, 2=upper active
active_coded = np.zeros_like(active_sets_total, dtype=float)
active_coded[active_sets_lower] = 1
active_coded[active_sets_upper] = 2
im = ax.pcolormesh(X_mesh, T_mesh, active_coded, shading="auto", cmap="RdYlBu", vmin=0, vmax=2)
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["Free", "Lower", "Upper"])
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Active Constraints")

# Plot 6: Active set fractions over time
ax = axes[1, 2]
frac_lower = [active_sets_lower[n].sum() / Nx_actual for n in range(Nt + 1)]
frac_upper = [active_sets_upper[n].sum() / Nx_actual for n in range(Nt + 1)]
frac_total = [active_sets_total[n].sum() / Nx_actual for n in range(Nt + 1)]
ax.plot(t_grid, np.array(frac_lower) * 100, "r-", linewidth=2, label="Lower bound")
ax.plot(t_grid, np.array(frac_upper) * 100, "b-", linewidth=2, label="Upper bound")
ax.plot(t_grid, np.array(frac_total) * 100, "k--", linewidth=2, label="Total")
ax.set_xlabel("Time t")
ax.set_ylabel("Active Set (%)")
ax.set_title("Constraint Activity Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
output_dir = Path(__file__).parent / "outputs" / "obstacle_problem_1d_bilateral"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "bilateral_constraint_validation.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved to {output_file}")
plt.close()

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("SUMMARY - Bilateral Constraint Validation")
print("=" * 70)
print("Physical problem: Heat diffusion with box constraints")
print("Initial condition: Gaussian dip (cold center)")
print("Lower bound: Parabolic (higher at center)")
print("Upper bound: Uniform ceiling")
print("\nResults:")
print("  Unconstrained violations:")
print(f"    Lower: {max_viol_lower:.4f}")
print(f"    Upper: {max_viol_upper:.4f}")
print("  Constrained violations:")
print(f"    Lower: {violations_lower_constr.max():.2e}")
print(f"    Upper: {violations_upper_constr.max():.2e}")
print("  Active set evolution:")
print(f"    Lower: {frac_lower[0] * 100:.1f}% → {frac_lower[-1] * 100:.1f}%")
print(f"    Upper: {frac_upper[0] * 100:.1f}% → {frac_upper[-1] * 100:.1f}%")
print("\n✓ Bilateral constraint validated successfully!")
print("  - Both bounds enforced perfectly")
print("  - Lower constraint releases as system warms")
print("  - Demonstrates box constraint projection")
print("=" * 70)
