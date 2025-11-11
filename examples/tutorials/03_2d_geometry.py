"""
Tutorial 03: 2D Geometry

Learn how to work with 2D spatial domains in MFG_PDE.

What you'll learn:
- How to use GridBasedMFGProblem for nD problems
- How to define 2D Hamiltonians with gradient vectors
- How to visualize 2D density evolution
- How mass conservation works in 2D

Mathematical Problem:
    2D crowd navigation with target attraction:
    - Domain: [0,10] × [0,10] (square room)
    - Target: Center at (5,5) - agents want to reach it
    - Hamiltonian: H = (1/2)|∇u|² + λm|∇u|² (congestion)
    - Initial: Agents start at corners
"""

import numpy as np

from mfg_pde import GridBasedMFGProblem, solve_mfg

# ==============================================================================
# Step 1: Create 2D Problem
# ==============================================================================

print("=" * 70)
print("TUTORIAL 03: 2D Geometry")
print("=" * 70)
print()


class TargetAttraction2D(GridBasedMFGProblem):
    """
    2D MFG: Agents navigate toward a target in a square room.

    Key differences from 1D:
    - domain_bounds is a tuple of (xmin, xmax, ymin, ymax)
    - p is now a 2D gradient vector [px, py]
    - Density m(t,x,y) is a 3D array
    """

    def __init__(self, target_location=(5.0, 5.0), congestion_weight=1.0):
        # Initialize 2D grid-based problem
        super().__init__(
            domain_bounds=(0.0, 10.0, 0.0, 10.0),  # (xmin, xmax, ymin, ymax)
            grid_resolution=30,  # 30×30 spatial grid
            time_domain=(4.0, 40),  # (T, Nt) - 4 seconds, 40 timesteps
            diffusion_coeff=0.2,  # σ
        )

        self.target = np.array(target_location)
        self.congestion_weight = congestion_weight

    def hamiltonian(self, x, m, p, t):
        """
        Hamiltonian for 2D problem.

        Args:
            x: Spatial positions - shape (N, 2) where N is number of points
            m: Density values - shape (N,)
            p: Gradient of value function - shape (N, 2) for [px, py]
            t: Time (scalar or array)

        Returns:
            Hamiltonian values - shape (N,)
        """
        # Compute |∇u|² = px² + py²
        # p.shape is (N, 2), so we sum over last axis
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else np.sum(p**2)

        # H = (1/2)|∇u|² + λ·m·|∇u|²
        return 0.5 * p_squared + self.congestion_weight * m * p_squared

    def terminal_cost(self, x):
        """
        Terminal cost: Squared distance to target.

        Args:
            x: Spatial positions - shape (N, 2)

        Returns:
            Cost values - shape (N,)
        """
        # Distance to target: |x - target|²
        return np.sum((x - self.target) ** 2, axis=1)

    def initial_density(self, x):
        """
        Initial density: 4 Gaussian blobs at corners.

        Args:
            x: Spatial positions - shape (N, 2)

        Returns:
            Density values - shape (N,)
        """
        # Define corner locations
        corners = [
            np.array([2.0, 2.0]),  # Bottom-left
            np.array([8.0, 2.0]),  # Bottom-right
            np.array([2.0, 8.0]),  # Top-left
            np.array([8.0, 8.0]),  # Top-right
        ]

        # Sum of Gaussians centered at corners
        density = np.zeros(x.shape[0])
        for corner in corners:
            dist_squared = np.sum((x - corner) ** 2, axis=1)
            density += np.exp(-5 * dist_squared)

        # Normalize (2D integration using trapezoidal rule)
        # For 2D, we need to integrate over the grid
        grid_shape = (self.grid_resolution, self.grid_resolution)
        density_2d = density.reshape(grid_shape)

        dx = (self.spatial_bounds[0, 1] - self.spatial_bounds[0, 0]) / (self.grid_resolution - 1)
        dy = (self.spatial_bounds[1, 1] - self.spatial_bounds[1, 0]) / (self.grid_resolution - 1)

        total_mass = np.trapz(np.trapz(density_2d, dx=dy, axis=0), dx=dx)

        return density / total_mass


# ==============================================================================
# Step 2: Solve the Problem
# ==============================================================================

print("Creating 2D target attraction problem...")
problem = TargetAttraction2D(target_location=(5.0, 5.0), congestion_weight=1.0)

print(f"  Domain: {problem.spatial_bounds}")
print(f"  Grid: {problem.grid_resolution} × {problem.grid_resolution}")
print(f"  Target: {problem.target}")
print()

print("Solving 2D MFG system...")
print("(This may take a moment - 2D problems are more computationally intensive)")
print()

result = solve_mfg(problem, verbose=True)

print()
print(f"Converged: {result.converged} (iterations: {result.iterations})")
print()

# ==============================================================================
# Step 3: Mass Conservation Check
# ==============================================================================

print("=" * 70)
print("MASS CONSERVATION (2D)")
print("=" * 70)
print()

# In 2D, mass conservation requires 2D integration
dx = (problem.spatial_bounds[0, 1] - problem.spatial_bounds[0, 0]) / (problem.grid_resolution - 1)
dy = (problem.spatial_bounds[1, 1] - problem.spatial_bounds[1, 0]) / (problem.grid_resolution - 1)

# Check mass at each timestep
grid_shape = (problem.grid_resolution, problem.grid_resolution)
masses = []
for t_idx in range(len(problem.time_grid)):
    m_2d = result.M[t_idx].reshape(grid_shape)
    mass = np.trapz(np.trapz(m_2d, dx=dy, axis=0), dx=dx)
    masses.append(mass)

print(f"Initial mass: {masses[0]:.6f}")
print(f"Final mass:   {masses[-1]:.6f}")
print(f"Mass drift:   {abs(masses[-1] - masses[0]):.6e}")
print()

# ==============================================================================
# Step 4: Visualize 2D Solution
# ==============================================================================

try:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10))

    # Reshape density to 2D grid for plotting
    grid_shape = (problem.grid_resolution, problem.grid_resolution)

    # Create spatial grids
    x_1d = np.linspace(problem.spatial_bounds[0, 0], problem.spatial_bounds[0, 1], problem.grid_resolution)
    y_1d = np.linspace(problem.spatial_bounds[1, 0], problem.spatial_bounds[1, 1], problem.grid_resolution)
    X, Y = np.meshgrid(x_1d, y_1d)

    # Plot snapshots at different times
    times_to_plot = [0, len(problem.time_grid) // 3, 2 * len(problem.time_grid) // 3, -1]

    for idx, t_idx in enumerate(times_to_plot, 1):
        ax = fig.add_subplot(2, 2, idx)

        m_2d = result.M[t_idx].reshape(grid_shape)

        # Density heatmap
        contour = ax.contourf(X, Y, m_2d, levels=20, cmap="viridis")
        ax.plot(*problem.target, "r*", markersize=20, label="Target")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Density at t={problem.time_grid[t_idx]:.2f}s")
        ax.set_aspect("equal")
        plt.colorbar(contour, ax=ax)
        ax.legend()

    plt.tight_layout()
    plt.savefig("examples/outputs/tutorials/03_2d_geometry.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: examples/outputs/tutorials/03_2d_geometry.png")
    print()

except ImportError:
    print("Matplotlib not available - skipping visualization")
    print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
print()
print("What you learned:")
print("  1. How to use GridBasedMFGProblem for nD problems")
print("  2. How to work with 2D gradients p = [px, py]")
print("  3. How to compute 2D integrals for mass conservation")
print("  4. How to visualize 2D density evolution")
print()
print("Key differences from 1D:")
print("  - domain_bounds: (xmin, xmax, ymin, ymax) instead of xmin/xmax")
print("  - p is a vector: shape (N, 2) instead of (N,)")
print("  - Integration: Use nested trapz() for 2D")
print()
print("Next: Tutorial 04 - Particle Methods")
print("=" * 70)
