"""
Plot and save mass loss data for Bug #8 investigation
"""

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.validation.test_2d_crowd_motion import CrowdMotion2D
from mfg_pde.factory import create_basic_solver

print("Running simulation and saving data...")

# Run simulation
problem = CrowdMotion2D(
    grid_resolution=8,
    time_horizon=0.4,
    num_timesteps=10,
)

solver = create_basic_solver(
    problem,
    damping=0.6,
    max_iterations=30,
    tolerance=1e-4,
)

result = solver.solve()

# Calculate mass evolution
dx, dy = problem.geometry.grid.spacing
dV = dx * dy

times = []
masses = []
loss_percents = []

initial_mass = np.sum(result.M[0]) * dV

for t_idx in range(len(result.M)):
    t_value = t_idx * problem.dt
    mass_t = np.sum(result.M[t_idx]) * dV
    loss_percent = (initial_mass - mass_t) / initial_mass * 100

    times.append(t_value)
    masses.append(mass_t)
    loss_percents.append(loss_percent)

# Save data
np.savez(
    "bug8_mass_loss_data.npz",
    times=np.array(times),
    masses=np.array(masses),
    loss_percents=np.array(loss_percents),
    M=result.M,
    U=result.U,
    grid_spacing=np.array([dx, dy]),
    problem_params={
        "grid_resolution": 8,
        "time_horizon": 0.4,
        "num_timesteps": 10,
    },
)

print("Data saved to bug8_mass_loss_data.npz")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Mass vs Time
ax = axes[0, 0]
ax.plot(times, masses, "b-o", linewidth=2, markersize=6)
ax.axhline(y=initial_mass, color="r", linestyle="--", label="Initial mass")
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Total Mass", fontsize=12)
ax.set_title("Mass Evolution Over Time", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Mass Loss Percentage vs Time
ax = axes[0, 1]
ax.plot(times, loss_percents, "r-o", linewidth=2, markersize=6)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Mass Loss (%)", fontsize=12)
ax.set_title("Cumulative Mass Loss", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)

# Plot 3: Mass Loss Rate (per timestep)
ax = axes[1, 0]
if len(masses) > 1:
    loss_rates = []
    for i in range(len(masses) - 1):
        loss_rate = (masses[i] - masses[i + 1]) / masses[i] * 100
        loss_rates.append(loss_rate)

    timestep_centers = [(times[i] + times[i + 1]) / 2 for i in range(len(times) - 1)]
    ax.plot(timestep_centers, loss_rates, "g-o", linewidth=2, markersize=6)
    ax.axhline(y=np.mean(loss_rates), color="k", linestyle="--", label=f"Average: {np.mean(loss_rates):.2f}%")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Loss Rate (% per timestep)", fontsize=12)
    ax.set_title("Mass Loss Rate", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

# Plot 4: Density heatmap at final time
ax = axes[1, 1]
im = ax.imshow(result.M[-1].T, origin="lower", cmap="viridis", aspect="auto")
ax.set_xlabel("x index", fontsize=12)
ax.set_ylabel("y index", fontsize=12)
ax.set_title(f"Density at t={times[-1]:.3f}", fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax, label="Density")

plt.tight_layout()
plt.savefig("bug8_mass_loss_plots.png", dpi=150, bbox_inches="tight")
print("Plots saved to bug8_mass_loss_plots.png")

# Print summary
print()
print("Summary:")
print(f"  Initial mass: {initial_mass:.6f}")
print(f"  Final mass: {masses[-1]:.6f}")
print(f"  Total loss: {loss_percents[-1]:.2f}%")
if len(masses) > 1:
    print(f"  Avg loss rate: {np.mean(loss_rates):.2f}% per timestep")

plt.show()
