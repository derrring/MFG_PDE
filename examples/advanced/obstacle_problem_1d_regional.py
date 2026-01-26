"""
1D Regional Constraint Problem - Spatially Localized Obstacles.

This demonstrates regional (spatially localized) constraints on the heat equation:
    u(t,x) ≥ ψ(x)  for x ∈ Ω_constraint
    u(t,x) free    for x ∉ Ω_constraint

Physical Interpretation: Protected Zone with Minimum Temperature
-----------------------------------------------------------------
Consider a rod where only a central region must maintain minimum temperature:
- u(t,x) = temperature at time t, position x
- ψ(x) = minimum temperature (active only in protected zone)
- Ω_constraint = protected region (e.g., x ∈ [0.3, 0.7])
- Outside region: temperature can drop freely

The regional constraint represents:
- Protected zone: Critical region requiring temperature control (e.g., sensor location)
- Free zones: Edges can cool without restriction
- Physical example: Thermostat only controls central area

Mathematically:
    ∂u/∂t = σ²/2 · ∂²u/∂x² - λu                  (heat equation with cooling)

    Constraint:
        u(t,x) ≥ ψ(x)  if x ∈ [x_min_protected, x_max_protected]
        u(t,x) free    otherwise

Active Set: Points in protected region where constraint binds.

Setup:
    - Initial condition: u(0,x) = 1.0 (uniform hot)
    - Protected region: x ∈ [0.3, 0.7]
    - Minimum in protected zone: ψ = 0.4
    - Cooling: rod loses heat exponentially
    - Outside protected zone: can cool below ψ freely

Expected Behavior:
    - Early times: Hot everywhere, no constraint active
    - Later times: Rod cools
    - In protected zone: Temperature maintained at ψ = 0.4
    - Outside zone: Temperature drops below ψ freely
    - Active set appears only in protected region

Created: 2026-01-17 (Issue #591 - Regional Constraint Demonstration)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import ObstacleConstraint, no_flux_bc

print("=" * 70)
print("Regional Constraint Problem - Protected Temperature Zone")
print("=" * 70)

# ========================================
# Parameters
# ========================================
x_min, x_max = 0.0, 1.0
Nx = 50
T = 3.0  # Extended to see constraint activate
Nt = 185  # Timesteps for stability
sigma = 0.15
cooling_rate = 0.6  # Faster cooling to see effect

# Protected region boundaries
x_protected_min = 0.3
x_protected_max = 0.7

# Create grid

grid = TensorProductGrid(bounds=[(x_min, x_max)], Nx=[Nx], boundary_conditions=no_flux_bc(dimension=1))
x = grid.coordinates[0]
Nx_actual = len(x)
dx = x[1] - x[0]
dt = T / Nt

print("\nPhysical Setup:")
print("  Rod with protected central zone")
print(f"  Domain: x ∈ [{x_min}, {x_max}]")
print(f"  Protected zone: x ∈ [{x_protected_min}, {x_protected_max}]")
print(f"  Time: t ∈ [0, {T}]")
print(f"  Diffusion: σ² = {sigma**2:.4f}")
print(f"  Cooling rate: λ = {cooling_rate:.2f}")

# ========================================
# Initial Condition and Regional Constraint
# ========================================
# Initial temperature: Uniform hot
u_initial = np.ones_like(x) * 1.0

# Obstacle: Constant minimum in protected zone
psi = np.ones_like(x) * 0.4

# Region mask: constraint only applies in protected zone
region = (x >= x_protected_min) & (x <= x_protected_max)

print("\nInitial Condition:")
print(f"  u(0,x) = {u_initial[0]:.4f} (uniform)")

print("\nRegional Constraint:")
print(f"  Minimum temperature: ψ = {psi[0]:.4f}")
print(f"  Active in region: x ∈ [{x_protected_min}, {x_protected_max}]")
print(f"  Protected points: {region.sum()}/{Nx_actual}")
print(f"  Free points: {(~region).sum()}/{Nx_actual}")

# Create regional obstacle constraint
constraint = ObstacleConstraint(psi, constraint_type="lower", region=region)

# Verify initial condition
if constraint.is_feasible(u_initial, tol=1e-10):
    print("\n✓ Initial condition satisfies regional constraint")
else:
    print("\n⚠ Initial condition violates constraint, projecting...")
    u_initial = constraint.project(u_initial)

# ========================================
# Solve Heat Equation WITHOUT Constraint
# ========================================
print("\n" + "=" * 70)
print("Solving Heat Equation WITHOUT regional constraint...")
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

# Heat equation with cooling
for n in range(Nt):
    u_curr = U_unconstr[n]
    laplacian = np.zeros(Nx_actual)
    laplacian[1:-1] = (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]) / dx**2
    laplacian[0] = (u_curr[1] - u_curr[0]) / dx**2
    laplacian[-1] = (u_curr[-2] - u_curr[-1]) / dx**2

    U_unconstr[n + 1] = u_curr + dt * (diffusion_coef * laplacian - cooling_rate * u_curr)

print("\nSolution evolution (unconstrained):")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    u_min, u_max = U_unconstr[n].min(), U_unconstr[n].max()
    # Check violations in protected zone only
    violations_in_zone = np.maximum(0, psi[region] - U_unconstr[n][region])
    max_viol_zone = violations_in_zone.max() if violations_in_zone.size > 0 else 0
    print(f"  t={t:.3f}: u ∈ [{u_min:.4f}, {u_max:.4f}], zone violation={max_viol_zone:.4f}")

# Check violations in protected zone
violations_unconstr = np.maximum(0, psi - U_unconstr)
violations_in_protected = violations_unconstr[:, region]
max_viol = violations_in_protected.max()
num_viol = (violations_in_protected > 1e-8).sum()

print("\nConstraint check (protected zone only):")
print(f"  Max violation: {max_viol:.4f}")
print(f"  Violating space-time points: {num_viol} / {(Nt + 1) * region.sum()}")

if max_viol > 1e-6:
    print("  → Cooling causes temperature drop below minimum in protected zone")
else:
    print("  → Temperature stays above minimum naturally")

# ========================================
# Solve WITH Regional Constraint
# ========================================
print("\n" + "=" * 70)
print("Solving Heat Equation WITH regional constraint...")
print("=" * 70)

U_constr = np.zeros((Nt + 1, Nx_actual))
U_constr[0] = u_initial.copy()

for n in range(Nt):
    u_curr = U_constr[n]

    laplacian = np.zeros(Nx_actual)
    laplacian[1:-1] = (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]) / dx**2
    laplacian[0] = (u_curr[1] - u_curr[0]) / dx**2
    laplacian[-1] = (u_curr[-2] - u_curr[-1]) / dx**2

    u_next = u_curr + dt * (diffusion_coef * laplacian - cooling_rate * u_curr)

    # Apply regional constraint (only enforced in protected zone)
    U_constr[n + 1] = constraint.project(u_next)

print("\nSolution evolution (constrained):")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    u_min, u_max = U_constr[n].min(), U_constr[n].max()
    active = constraint.get_active_set(U_constr[n], tol=1e-6)
    active_in_zone = active[region].sum()
    print(f"  t={t:.3f}: u ∈ [{u_min:.4f}, {u_max:.4f}], {active_in_zone}/{region.sum()} active in zone")

# Verify no violations in protected zone
violations_constr = np.maximum(0, psi - U_constr)
violations_in_protected_constr = violations_constr[:, region]
max_viol_constr = violations_in_protected_constr.max()

print("\nConstraint satisfaction (protected zone):")
print(f"  Max violation: {max_viol_constr:.2e}")
print(f"  Status: {'✓ SATISFIED' if max_viol_constr < 1e-8 else '✗ VIOLATED'}")

# Check that outside zone can drop below ψ
outside_region = ~region
u_final_outside = U_constr[-1][outside_region]
min_outside = u_final_outside.min()
print("\nOutside protected zone:")
print(f"  Final temperature: min={min_outside:.4f}")
print(f"  Constraint ψ: {psi[0]:.4f}")
if min_outside < psi[0] - 1e-6:
    print("  ✓ Temperature drops below ψ freely (as expected)")
else:
    print("  Temperature stays above ψ (might need longer time)")

# ========================================
# Active Set Analysis
# ========================================
print("\n" + "=" * 70)
print("Active Set Evolution")
print("=" * 70)

active_sets = []
for n in range(Nt + 1):
    active = constraint.get_active_set(U_constr[n], tol=1e-6)
    active_sets.append(active)
active_sets = np.array(active_sets)

# Analyze active set in protected zone vs outside
active_in_zone = active_sets[:, region]
active_outside = active_sets[:, ~region]

print("\nActive set in protected zone:")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    frac_in_zone = active_in_zone[n].sum() / region.sum() if region.sum() > 0 else 0
    frac_outside = active_outside[n].sum() / (~region).sum() if (~region).sum() > 0 else 0
    print(f"  t={t:.3f}: Protected={frac_in_zone * 100:5.1f}%, Outside={frac_outside * 100:5.1f}%")

print("\nPhysical interpretation:")
print("  As rod cools:")
print(f"  - Protected zone: Constraint activates, maintaining ψ = {psi[0]:.1f}")
print("  - Outside zone: Temperature drops below ψ freely")
print("  - Demonstrates spatial selectivity of constraint")

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
im = ax.pcolormesh(X_mesh, T_mesh, U_unconstr, shading="auto", cmap="hot")
ax.axvline(x_protected_min, color="cyan", linestyle="--", linewidth=2, alpha=0.7)
ax.axvline(x_protected_max, color="cyan", linestyle="--", linewidth=2, alpha=0.7)
plt.colorbar(im, ax=ax, label="Temperature u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Unconstrained (Protected Zone: cyan lines)")

# Plot 2: Constrained evolution
ax = axes[0, 1]
im = ax.pcolormesh(X_mesh, T_mesh, U_constr, shading="auto", cmap="hot")
ax.axvline(x_protected_min, color="cyan", linestyle="--", linewidth=2, alpha=0.7)
ax.axvline(x_protected_max, color="cyan", linestyle="--", linewidth=2, alpha=0.7)
plt.colorbar(im, ax=ax, label="Temperature u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Constrained (Protected Zone: cyan lines)")

# Plot 3: Difference
ax = axes[0, 2]
diff = U_constr - U_unconstr
im = ax.pcolormesh(X_mesh, T_mesh, diff, shading="auto", cmap="Greens")
ax.axvline(x_protected_min, color="red", linestyle="--", linewidth=2, alpha=0.7)
ax.axvline(x_protected_max, color="red", linestyle="--", linewidth=2, alpha=0.7)
plt.colorbar(im, ax=ax, label="Δu (constraint effect)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Constraint Effect (Protected Zone: red lines)")

# Plot 4: Temperature snapshots
ax = axes[1, 0]
times = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(times)))
for i, n in enumerate(times):
    t = n * dt
    ax.plot(x, U_constr[n], "-", color=colors[i], linewidth=2, label=f"t={t:.2f}")
ax.axhline(psi[0], color="k", linestyle=":", linewidth=2, label=f"ψ = {psi[0]:.1f}")
ax.axvspan(x_protected_min, x_protected_max, alpha=0.2, color="yellow", label="Protected zone")
ax.set_xlabel("Position x")
ax.set_ylabel("Temperature u(t,x)")
ax.set_title("Temperature Evolution")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# Plot 5: Active set heatmap
ax = axes[1, 1]
im = ax.pcolormesh(X_mesh, T_mesh, active_sets.astype(float), shading="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax.axvline(x_protected_min, color="white", linestyle="--", linewidth=2)
ax.axvline(x_protected_max, color="white", linestyle="--", linewidth=2)
plt.colorbar(im, ax=ax, label="Constraint Active (1) / Free (0)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Active Set (Protected Zone: white lines)")

# Plot 6: Compare protected vs outside zones
ax = axes[1, 2]
# Average temperature in each zone over time
u_in_zone_avg = [U_constr[n][region].mean() for n in range(Nt + 1)]
u_outside_avg = [U_constr[n][~region].mean() for n in range(Nt + 1)]
active_frac_zone = [active_in_zone[n].sum() / region.sum() for n in range(Nt + 1)]

ax.plot(t_grid, u_in_zone_avg, "b-", linewidth=2, label="Protected zone (avg)")
ax.plot(t_grid, u_outside_avg, "r--", linewidth=2, label="Outside zone (avg)")
ax.axhline(psi[0], color="k", linestyle=":", linewidth=2, label=f"ψ = {psi[0]:.1f}")
ax.set_xlabel("Time t")
ax.set_ylabel("Average Temperature")
ax.set_title("Protected vs Outside Zones")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
output_dir = Path(__file__).parent / "outputs" / "obstacle_problem_1d_regional"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "regional_constraint_validation.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved to {output_file}")
plt.close()

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("SUMMARY - Regional Constraint Validation")
print("=" * 70)
print("Physical problem: Cooling rod with protected central zone")
print(f"Protected region: x ∈ [{x_protected_min}, {x_protected_max}]")
print(f"Constraint: u ≥ {psi[0]:.1f} (only in protected zone)")
print("\nResults:")
print(f"  Unconstrained violations (protected zone): {max_viol:.4f}")
print(f"  Constrained violations (protected zone): {max_viol_constr:.2e}")
print(f"  Final temperature outside zone: {min_outside:.4f} < ψ = {psi[0]:.1f}")
print(f"  Active set in protected zone: {active_frac_zone[0] * 100:.1f}% → {active_frac_zone[-1] * 100:.1f}%")
print("\n✓ Regional constraint validated successfully!")
print("  - Constraint enforced only in protected zone")
print("  - Outside zone cools freely below ψ")
print("  - Demonstrates spatial selectivity")
print("=" * 70)
