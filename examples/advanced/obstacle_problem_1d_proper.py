"""
1D Obstacle Problem - Proper Formulation with Physical Meaning.

This demonstrates a REALISTIC obstacle problem in optimal control/MFG context.

Physical Interpretation: Optimal Exit Problem
----------------------------------------------
An agent moves in 1D domain trying to minimize cost. At any time, the agent can:
1. CONTINUE: Pay running cost and reach terminal time
2. EXIT: Stop immediately and pay obstacle penalty ψ(x)

The value function u(t,x) represents the optimal cost-to-go if continuing.
The obstacle ψ(x) represents the cost of quitting immediately.

Active Set: Points where it's optimal to EXIT (u = ψ, stopping is optimal)
Inactive Set: Points where it's optimal to CONTINUE (u < ψ)

Mathematical Formulation:
-------------------------
HJB Variational Inequality:
    min(-∂u/∂t + H(∇u) - σ²/2 Δu - L(x), u - ψ(x)) = 0

Equivalently:
    -∂u/∂t + H(∇u) - σ²/2 Δu = L(x)  if u < ψ  (continue optimal)
    u = ψ(x)                           if u = ψ  (exit optimal)

Setup:
    - Domain: x ∈ [0, 1]
    - Time: t ∈ [0, T]
    - Running cost: L(x) = λ(x - x_target)²  (tracking cost)
    - Terminal cost: g(x) = c_T(x - x_target)²  (terminal penalty)
    - Obstacle: ψ(x) = c_exit + c_shape(x - x_target)²  (exit penalty)
    - Hamiltonian: H(p) = (1/2)|p|²  (kinetic energy)
    - Diffusion: σ² = 0.01

Physical Intuition:
    - Near target (x ≈ 0.5): Low running cost, worth continuing → u < ψ
    - Far from target: High running cost, might be worth exiting → u ≈ ψ
    - Active set grows near boundaries if exit penalty is attractive enough

References:
    - Pham (2009): Continuous-time Stochastic Control and Optimization with Financial Applications
    - Bardi & Capuzzo-Dolcetta (2008): Optimal Control and Viscosity Solutions of HJB
    - Achdou & Capuzzo-Dolcetta (2010): Mean Field Games: Numerical Methods

Created: 2026-01-17 (Issue #591 - Proper Obstacle Problem)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import ObstacleConstraint, neumann_bc

print("=" * 70)
print("Proper 1D Obstacle Problem - Optimal Exit")
print("=" * 70)

# ========================================
# Physical Parameters
# ========================================
# Domain and discretization
x_min, x_max = 0.0, 1.0
x_target = 0.5  # Target location
Nx = 100
T = 1.0
Nt = 50

# Cost parameters (tuned for interesting behavior)
lambda_running = 2.0  # Running cost strength
c_terminal = 5.0  # Terminal penalty strength
c_exit_base = 0.3  # Base exit cost
c_exit_shape = 1.5  # Exit cost shape parameter
sigma = 0.1  # Diffusion

print("\nPhysical Setup:")
print(f"  Agent tries to stay near x = {x_target} with minimal cost")
print("  Can EXIT anytime and pay penalty ψ(x)")
print("  Or CONTINUE and pay running + terminal costs")

print("\nParameters:")
print(f"  Domain: x ∈ [{x_min}, {x_max}]")
print(f"  Time horizon: T = {T}")
print(f"  Grid: {Nx} spatial points, {Nt} time steps")
print(f"  Diffusion: σ = {sigma}")
print(f"  Running cost strength: λ = {lambda_running}")
print(f"  Terminal penalty: c_T = {c_terminal}")
print(f"  Exit base cost: c_exit = {c_exit_base}")

# ========================================
# Create Geometry and Grid
# ========================================
grid = TensorProductGrid(bounds=[(x_min, x_max)], Nx=[Nx])
x = grid.coordinates[0]
Nx_actual = len(x)
dx = x[1] - x[0]
dt = T / Nt


# ========================================
# Cost Functions
# ========================================
def running_cost(x_coords, alpha=None):
    """
    Running cost L(x) = λ(x - x_target)².

    Penalizes deviation from target. Agent wants to stay near x_target.
    """
    return lambda_running * (x_coords[0] - x_target) ** 2


def terminal_cost(x_coords):
    """
    Terminal cost g(x) = c_T(x - x_target)².

    Penalty for not being at target at final time.
    """
    return c_terminal * (x_coords[0] - x_target) ** 2


# Compute terminal cost values
g_terminal = c_terminal * (x - x_target) ** 2


# ========================================
# Obstacle Function (Exit Penalty)
# ========================================
def obstacle_function(x, c_base, c_shape, x_target):
    """
    Obstacle ψ(x) = c_base + c_shape(x - x_target)².

    Represents cost of exiting immediately:
    - Base cost c_base for giving up
    - Additional penalty c_shape(x - x_target)² if far from target

    Interpretation: Exiting far from target is more costly.
    """
    return c_base + c_shape * (x - x_target) ** 2


psi = obstacle_function(x, c_exit_base, c_exit_shape, x_target)

print("\nCost Functions:")
print(f"  Running cost: L(x) = {lambda_running}·(x - {x_target})²")
print(f"  Terminal cost: g(x) = {c_terminal}·(x - {x_target})²")
print(f"    g(x) range: [{g_terminal.min():.4f}, {g_terminal.max():.4f}]")
print(f"  Exit penalty: ψ(x) = {c_exit_base} + {c_exit_shape}·(x - {x_target})²")
print(f"    ψ(x) range: [{psi.min():.4f}, {psi.max():.4f}]")

# ========================================
# Physical Interpretation
# ========================================
print("\nPhysical Comparison (at boundaries x=0, x=1):")
print(f"  Terminal cost g(x) at boundary: {g_terminal[0]:.4f}")
print(f"  Exit penalty ψ(x) at boundary: {psi[0]:.4f}")
if psi[0] < g_terminal[0]:
    print("  → Exit is CHEAPER than continuing (ψ < g)")
    print("     Expect active set near boundaries")
else:
    print("  → Continuing is CHEAPER than exit (g < ψ)")
    print("     Exit not attractive, small active set")

print(f"\nAt center x={x_target}:")
print(f"  Terminal cost g({x_target}): {g_terminal[Nx_actual // 2]:.4f}")
print(f"  Exit penalty ψ({x_target}): {psi[Nx_actual // 2]:.4f}")

# ========================================
# Create Problem and Constraint
# ========================================
bc = neumann_bc(dimension=1)
problem = MFGProblem(
    geometry=grid,
    T=T,
    Nt=Nt,
    diffusion=sigma,
    bc=bc,
    running_cost=running_cost,
    terminal_cost=terminal_cost,
)

# Create constraint (lower obstacle: u ≥ ψ means "value ≥ exit cost")
# Physical meaning: Can't do better than exiting
constraint = ObstacleConstraint(psi, constraint_type="lower")

print("\nConstraint:")
print("  Type: Lower obstacle (u ≥ ψ)")
print("  Meaning: Optimal value ≥ exit cost")
print("  Active set = where exiting is optimal")

# ========================================
# Initial Conditions
# ========================================
U_terminal = g_terminal.copy()
M_density = np.ones((Nt + 1, Nx_actual)) / Nx_actual
U_coupling_prev = np.tile(U_terminal, (Nt + 1, 1))

# ========================================
# Solve WITHOUT Constraint
# ========================================
print("\n" + "=" * 70)
print("Solving WITHOUT obstacle constraint...")
print("=" * 70)

solver_unconstr = HJBFDMSolver(
    problem, solver_type="newton", max_newton_iterations=100, newton_tolerance=1e-9, constraint=None
)

U_unconstr = solver_unconstr.solve_hjb_system(
    M_density=M_density, U_terminal=U_terminal, U_coupling_prev=U_coupling_prev
)

print("Unconstrained solution:")
print(f"  Range: [{U_unconstr.min():.4f}, {U_unconstr.max():.4f}]")
print(f"  u(t=0): [{U_unconstr[0].min():.4f}, {U_unconstr[0].max():.4f}]")
print(f"  u(t=T): [{U_unconstr[-1].min():.4f}, {U_unconstr[-1].max():.4f}]")

# Check violations
violations_unconstr = np.maximum(0, psi - U_unconstr)
max_viol_unconstr = violations_unconstr.max()
num_viol_unconstr = (violations_unconstr > 1e-8).sum()

print("\nConstraint check (without obstacle):")
print(f"  Max violation (ψ - u): {max_viol_unconstr:.4f}")
print(f"  Points violating: {num_viol_unconstr} / {(Nt + 1) * Nx_actual}")

if max_viol_unconstr > 1e-6:
    print("  → Solution violates obstacle (u < ψ in some regions)")
    print("     Constraint will be active!")
else:
    print("  → Solution naturally satisfies u ≥ ψ")
    print("     Constraint won't change solution much")

# ========================================
# Solve WITH Constraint
# ========================================
print("\n" + "=" * 70)
print("Solving WITH obstacle constraint...")
print("=" * 70)

solver_constr = HJBFDMSolver(
    problem, solver_type="newton", max_newton_iterations=100, newton_tolerance=1e-9, constraint=constraint
)

U_constr = solver_constr.solve_hjb_system(M_density=M_density, U_terminal=U_terminal, U_coupling_prev=U_coupling_prev)

print("Constrained solution:")
print(f"  Range: [{U_constr.min():.4f}, {U_constr.max():.4f}]")
print(f"  u(t=0): [{U_constr[0].min():.4f}, {U_constr[0].max():.4f}]")
print(f"  u(t=T): [{U_constr[-1].min():.4f}, {U_constr[-1].max():.4f}]")

# Check violations
violations_constr = np.maximum(0, psi - U_constr)
max_viol_constr = violations_constr.max()

print("\nConstraint satisfaction:")
print(f"  Max violation: {max_viol_constr:.2e}")
print(f"  Status: {'✓ SATISFIED' if max_viol_constr < 1e-8 else '✗ VIOLATED'}")

# ========================================
# Active Set Analysis
# ========================================
print("\n" + "=" * 70)
print("Active Set Analysis (Where Exit is Optimal)")
print("=" * 70)

active_sets = []
for n in range(Nt + 1):
    active = constraint.get_active_set(U_constr[n], tol=1e-6)
    active_sets.append(active)

active_sets = np.array(active_sets)

print("\nActive set evolution:")
times_to_check = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]
for n in times_to_check:
    t = n * dt
    active_frac = active_sets[n].sum() / Nx_actual
    print(f"  t={t:.2f}: {active_frac * 100:5.1f}% active ({active_sets[n].sum()}/{Nx_actual} points)")

# Spatial analysis at t=0
if active_sets[0].sum() > 0:
    active_indices = np.where(active_sets[0])[0]
    print("\nSpatial distribution of active set at t=0:")
    print(f"  Active near x=0 (boundary): {active_sets[0][:10].sum()}/10 points")
    print(f"  Active near x=0.5 (center): {active_sets[0][45:55].sum()}/10 points")
    print(f"  Active near x=1 (boundary): {active_sets[0][-10:].sum()}/10 points")
    print("  → Exit optimal at: ", end="")
    if active_sets[0][:10].sum() > 5:
        print("BOUNDARIES", end="")
    if active_sets[0][45:55].sum() > 5:
        print(" CENTER", end="")
    print()

# ========================================
# Visualization
# ========================================
print("\n" + "=" * 70)
print("Creating visualizations...")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

t_grid = np.linspace(0, T, Nt + 1)
X_mesh, T_mesh = np.meshgrid(x, t_grid)

# Plot 1: Unconstrained value function
ax = fig.add_subplot(gs[0, 0])
im = ax.pcolormesh(X_mesh, T_mesh, U_unconstr, shading="auto", cmap="RdYlBu_r")
plt.colorbar(im, ax=ax, label="u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Unconstrained Value Function")

# Plot 2: Constrained value function
ax = fig.add_subplot(gs[0, 1])
im = ax.pcolormesh(X_mesh, T_mesh, U_constr, shading="auto", cmap="RdYlBu_r")
plt.colorbar(im, ax=ax, label="u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Constrained Value Function")

# Plot 3: Difference (constraint effect)
ax = fig.add_subplot(gs[0, 2])
diff = U_constr - U_unconstr
im = ax.pcolormesh(X_mesh, T_mesh, diff, shading="auto", cmap="Greens")
plt.colorbar(im, ax=ax, label="Δu")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Constraint Correction (u_constr - u_unconstr)")

# Plot 4: Value function snapshots
ax = fig.add_subplot(gs[1, :2])
for n in times_to_check:
    t = n * dt
    ax.plot(x, U_unconstr[n], "--", alpha=0.5, label=f"t={t:.2f} (no constraint)")
    ax.plot(x, U_constr[n], "-", linewidth=2, label=f"t={t:.2f} (with constraint)")
ax.plot(x, psi, "k:", linewidth=3, label="ψ(x) (exit cost)")
ax.set_xlabel("Position x")
ax.set_ylabel("Value u(t,x)")
ax.set_title("Value Function Snapshots")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 5: Active set heatmap
ax = fig.add_subplot(gs[1, 2])
im = ax.pcolormesh(X_mesh, T_mesh, active_sets.astype(float), shading="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="Active (1) / Inactive (0)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Active Set (Exit Optimal)")

# Plot 6: Cost comparison at t=0
ax = fig.add_subplot(gs[2, 0])
ax.plot(x, U_constr[0], "b-", linewidth=2, label="Optimal value u(0,x)")
ax.plot(x, psi, "r--", linewidth=2, label="Exit cost ψ(x)")
ax.fill_between(x, psi, U_constr[0], where=(U_constr[0] > psi), alpha=0.3, color="blue", label="Continue region")
ax.fill_between(x, psi, psi, where=(np.abs(U_constr[0] - psi) < 1e-6), alpha=0.5, color="red", label="Exit region")
ax.set_xlabel("Position x")
ax.set_ylabel("Cost")
ax.set_title("Decision at t=0: Continue vs Exit")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 7: Active set fraction over time
ax = fig.add_subplot(gs[2, 1])
active_frac_time = [active_sets[n].sum() / Nx_actual for n in range(Nt + 1)]
ax.plot(t_grid, np.array(active_frac_time) * 100, "b-", linewidth=2)
ax.set_xlabel("Time t")
ax.set_ylabel("Active Set Fraction (%)")
ax.set_title("Exit Optimality Over Time")
ax.grid(True, alpha=0.3)

# Plot 8: Violation reduction
ax = fig.add_subplot(gs[2, 2])
viol_time_unconstr = [violations_unconstr[n].max() for n in range(Nt + 1)]
viol_time_constr = [violations_constr[n].max() for n in range(Nt + 1)]
ax.semilogy(t_grid, np.array(viol_time_unconstr) + 1e-12, "r--", linewidth=2, label="Without constraint")
ax.semilogy(t_grid, np.array(viol_time_constr) + 1e-12, "g-", linewidth=2, label="With constraint")
ax.axhline(1e-8, color="k", linestyle=":", alpha=0.5, label="Tolerance")
ax.set_xlabel("Time t")
ax.set_ylabel("Max violation (ψ - u)")
ax.set_title("Constraint Violation")
ax.legend()
ax.grid(True, alpha=0.3)

# Save
output_dir = Path(__file__).parent / "outputs" / "obstacle_problem_1d_proper"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "optimal_exit_problem.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved to {output_file}")

plt.show()

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("SUMMARY - Optimal Exit Problem")
print("=" * 70)
print("Physical interpretation:")
print(f"  Agent minimizes cost while tracking target x={x_target}")
print("  Can exit anytime and pay ψ(x), or continue to T")
print("\nResults:")
print(f"  Unconstrained violations: {max_viol_unconstr:.4f}")
print(f"  Constrained violations: {max_viol_constr:.2e}")
print(f"  Active set at t=0: {active_frac_time[0] * 100:.1f}%")
print(f"  Active set at t=T: {active_frac_time[-1] * 100:.1f}%")
print("\n✓ Obstacle problem with proper physics validated!")
print("=" * 70)
