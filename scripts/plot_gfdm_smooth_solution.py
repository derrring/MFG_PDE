#!/usr/bin/env python3
"""
Plot GFDM-LCR smooth BC solution results.

Visualizes the solution from solution_gfdm_lcr_smooth_fdm.h5 to assess
the smooth terminal cost BC compatibility approach.
"""

from pathlib import Path

import h5py

import matplotlib.pyplot as plt
import numpy as np

# Results directory
results_dir = Path(
    "/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/mfg-research/experiments/crowd_evacuation_2d/results/exp1_baseline"
)

# Load GFDM smooth solution
with h5py.File(results_dir / "solution_gfdm_lcr_smooth_fdm.h5", "r") as f:
    U_gfdm = np.array(f["U"])
    M_gfdm = np.array(f["M"])
    x_grid = np.array(f["x_grid"])
    y_grid = np.array(f["y_grid"])

# Reconstruct time grid (backward in time from T to 0)
Nt = U_gfdm.shape[0] - 1
T = 2.0  # From config
t_grid = np.linspace(T, 0, Nt + 1)

# Load FDM baseline for comparison
with h5py.File(results_dir / "solution_fdm_fdm.h5", "r") as f:
    U_fdm = np.array(f["U"])
    M_fdm = np.array(f["M"])

X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

# Time snapshots to visualize
time_indices = [0, len(t_grid) // 4, len(t_grid) // 2, 3 * len(t_grid) // 4, -1]
time_labels = ["t=0 (T)", "t=T/4", "t=T/2", "t=3T/4", "t=final (0)"]

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)

# Row 1: Density evolution (GFDM smooth)
for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels, strict=False)):
    ax = fig.add_subplot(gs[0, i])
    im = ax.pcolormesh(X, Y, M_gfdm[t_idx], shading="auto", cmap="hot", vmin=0)
    ax.set_title(f"M (GFDM-smooth) {t_label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Density")

    # Mark exit
    ax.plot(20, 5, "g*", markersize=15, label="Exit")

    # Compute and show center of mass
    com_x = np.sum(X * M_gfdm[t_idx]) / np.sum(M_gfdm[t_idx])
    com_y = np.sum(Y * M_gfdm[t_idx]) / np.sum(M_gfdm[t_idx])
    ax.plot(com_x, com_y, "wo", markersize=8, label="CoM")

# Row 2: Value function U (GFDM smooth)
for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels, strict=False)):
    ax = fig.add_subplot(gs[1, i])
    im = ax.pcolormesh(X, Y, U_gfdm[t_idx], shading="auto", cmap="viridis")
    ax.set_title(f"U (GFDM-smooth) {t_label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Value")

    # Overlay contours
    ax.contour(X, Y, U_gfdm[t_idx], levels=10, colors="white", alpha=0.3, linewidths=0.5)

# Row 3: Comparison with FDM at final time
ax = fig.add_subplot(gs[2, 0])
im = ax.pcolormesh(X, Y, M_fdm[-1], shading="auto", cmap="hot")
ax.set_title("M (FDM baseline) t=final")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax, label="Density")

ax = fig.add_subplot(gs[2, 1])
im = ax.pcolormesh(X, Y, M_gfdm[-1], shading="auto", cmap="hot")
ax.set_title("M (GFDM-smooth) t=final")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax, label="Density")

# Difference in density
ax = fig.add_subplot(gs[2, 2])
M_diff = M_gfdm[-1] - M_fdm[-1]
vmax = max(abs(M_diff.min()), abs(M_diff.max()))
im = ax.pcolormesh(X, Y, M_diff, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title("Density Difference\n(GFDM - FDM)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
plt.colorbar(im, ax=ax, label="ΔM")

# Velocity field comparison at t=0
ax = fig.add_subplot(gs[2, 3])
dU_dx_fdm = np.gradient(U_fdm[0], x_grid, axis=0)
dU_dy_fdm = np.gradient(U_fdm[0], y_grid, axis=1)
speed_fdm = np.sqrt(dU_dx_fdm**2 + dU_dy_fdm**2)
# Subsample for quiver
skip = 3
ax.quiver(
    X[::skip, ::skip],
    Y[::skip, ::skip],
    -dU_dx_fdm[::skip, ::skip],
    -dU_dy_fdm[::skip, ::skip],
    speed_fdm[::skip, ::skip],
    cmap="coolwarm",
    scale=100,
    alpha=0.7,
)
ax.set_title("Velocity Field (FDM)\nt=0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[2, 4])
dU_dx_gfdm = np.gradient(U_gfdm[0], x_grid, axis=0)
dU_dy_gfdm = np.gradient(U_gfdm[0], y_grid, axis=1)
speed_gfdm = np.sqrt(dU_dx_gfdm**2 + dU_dy_gfdm**2)
ax.quiver(
    X[::skip, ::skip],
    Y[::skip, ::skip],
    -dU_dx_gfdm[::skip, ::skip],
    -dU_dy_gfdm[::skip, ::skip],
    speed_gfdm[::skip, ::skip],
    cmap="coolwarm",
    scale=100,
    alpha=0.7,
)
ax.set_title("Velocity Field (GFDM-smooth)\nt=0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")

fig.suptitle("GFDM-LCR with Smooth Terminal Cost BC Compatibility", fontsize=16, fontweight="bold")
plt.savefig(results_dir / "gfdm_smooth_comprehensive.png", dpi=150, bbox_inches="tight")
print(f"Saved: {results_dir / 'gfdm_smooth_comprehensive.png'}")

# Create detailed metrics comparison
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Center of mass trajectory
com_x_fdm = np.zeros(len(t_grid))
com_y_fdm = np.zeros(len(t_grid))
com_x_gfdm = np.zeros(len(t_grid))
com_y_gfdm = np.zeros(len(t_grid))

for i in range(len(t_grid)):
    com_x_fdm[i] = np.sum(X * M_fdm[i]) / np.sum(M_fdm[i])
    com_y_fdm[i] = np.sum(Y * M_fdm[i]) / np.sum(M_fdm[i])
    com_x_gfdm[i] = np.sum(X * M_gfdm[i]) / np.sum(M_gfdm[i])
    com_y_gfdm[i] = np.sum(Y * M_gfdm[i]) / np.sum(M_gfdm[i])

axes[0, 0].plot(t_grid, com_x_fdm, "b-", linewidth=2, label="FDM")
axes[0, 0].plot(t_grid, com_x_gfdm, "r--", linewidth=2, label="GFDM-smooth")
axes[0, 0].axhline(y=20, color="g", linestyle=":", label="Exit x=20")
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Center of Mass X")
axes[0, 0].set_title("X-Displacement Over Time")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t_grid, com_y_fdm, "b-", linewidth=2, label="FDM")
axes[0, 1].plot(t_grid, com_y_gfdm, "r--", linewidth=2, label="GFDM-smooth")
axes[0, 1].axhline(y=5, color="g", linestyle=":", label="Exit y=5")
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Center of Mass Y")
axes[0, 1].set_title("Y-Displacement Over Time")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gradient statistics over time
grad_mean_fdm = np.zeros(len(t_grid))
grad_std_fdm = np.zeros(len(t_grid))
grad_mean_gfdm = np.zeros(len(t_grid))
grad_std_gfdm = np.zeros(len(t_grid))

for i in range(len(t_grid)):
    dU_dx = np.gradient(U_fdm[i], x_grid, axis=0)
    grad_mean_fdm[i] = dU_dx.mean()
    grad_std_fdm[i] = dU_dx.std()

    dU_dx = np.gradient(U_gfdm[i], x_grid, axis=0)
    grad_mean_gfdm[i] = dU_dx.mean()
    grad_std_gfdm[i] = dU_dx.std()

axes[1, 0].plot(t_grid, grad_mean_fdm, "b-", linewidth=2, label="FDM")
axes[1, 0].plot(t_grid, grad_mean_gfdm, "r--", linewidth=2, label="GFDM-smooth")
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Mean dU/dx")
axes[1, 0].set_title("Gradient Mean Over Time")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t_grid, grad_std_fdm, "b-", linewidth=2, label="FDM")
axes[1, 1].plot(t_grid, grad_std_gfdm, "r--", linewidth=2, label="GFDM-smooth")
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Std dU/dx")
axes[1, 1].set_title("Gradient Std Deviation Over Time")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig2.suptitle("Quantitative Comparison: GFDM-smooth vs FDM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(results_dir / "gfdm_smooth_metrics.png", dpi=150, bbox_inches="tight")
print(f"Saved: {results_dir / 'gfdm_smooth_metrics.png'}")

# Print quantitative summary
print("\n" + "=" * 70)
print("GFDM-LCR Smooth Terminal Cost - Quantitative Summary")
print("=" * 70)
print("\nDisplacement (x-direction):")
print(f"  FDM:         {com_x_fdm[-1] - com_x_fdm[0]:.2f} units")
print(f"  GFDM-smooth: {com_x_gfdm[-1] - com_x_gfdm[0]:.2f} units")
print(f"  Ratio:       {100 * (com_x_gfdm[-1] - com_x_gfdm[0]) / (com_x_fdm[-1] - com_x_fdm[0]):.1f}% of FDM")

print("\nGradient quality at t=0 (terminal time):")
print(f"  FDM  dU/dx:  mean={grad_mean_fdm[0]:.2f}, std={grad_std_fdm[0]:.2f}")
print(f"  GFDM dU/dx:  mean={grad_mean_gfdm[0]:.2f}, std={grad_std_gfdm[0]:.2f}")
print(f"  Std ratio:   {grad_std_gfdm[0] / grad_std_fdm[0]:.1f}x worse")

# Count wrong-sign gradients
dU_dx_fdm_t0 = np.gradient(U_fdm[0], x_grid, axis=0)
dU_dx_gfdm_t0 = np.gradient(U_gfdm[0], x_grid, axis=0)
wrong_fdm = np.sum(dU_dx_fdm_t0 > 0)
wrong_gfdm = np.sum(dU_dx_gfdm_t0 > 0)
total = dU_dx_fdm_t0.size

print("\nWrong-sign gradients (dU/dx > 0):")
print(f"  FDM:         {wrong_fdm}/{total} = {100 * wrong_fdm / total:.1f}%")
print(f"  GFDM-smooth: {wrong_gfdm}/{total} = {100 * wrong_gfdm / total:.1f}%")

print("\n" + "=" * 70)

plt.show()
