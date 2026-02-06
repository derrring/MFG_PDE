"""
2D Crowd Motion with FDM Solvers

Demonstrates dimension-agnostic FDM solvers (Phase 2) on a simple 2D crowd
motion problem where agents navigate from starting region to goal region
while minimizing congestion.

Key Features:
- Automatic dimension detection (no 1D vs 2D distinction needed)
- MFGProblem for rectangular domains with nD support
- create_basic_solver() uses HJB-FDM + FP-FDM (both support 2D via dimensional splitting)
- Simple factory interface - dimension handled automatically

Problem Setup:
- Domain: [0,1] × [0,1]
- Initial density: Concentrated at (0.2, 0.2) [bottom-left]
- Goal: (0.8, 0.8) [top-right]
- Agents minimize: travel cost + congestion cost
"""

import numpy as np

from mfg_pde import MFGComponents, MFGProblem
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.factory import create_basic_solver


class CrowdMotion2D(MFGProblem):
    """
    2D crowd motion MFG problem.

    Agents navigate from start to goal while avoiding congestion.
    """

    def __init__(
        self,
        grid_resolution=15,
        time_horizon=0.5,
        num_timesteps=20,
        diffusion=0.05,
        congestion_weight=0.5,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    ):
        """
        Initialize 2D crowd motion problem.

        Parameters
        ----------
        grid_resolution : int
            Points per dimension (grid_resolution × grid_resolution)
        time_horizon : float
            Final time T
        num_timesteps : int
            Number of time steps
        diffusion : float
            Diffusion coefficient σ
        congestion_weight : float
            Weight κ for congestion cost
        goal : tuple
            Goal position (x, y)
        start : tuple
            Initial density center (x, y)
        """
        super().__init__(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[grid_resolution, grid_resolution],
            T=time_horizon,
            Nt=num_timesteps,
            diffusion=diffusion,
        )
        self.grid_resolution = grid_resolution

        self.congestion_weight = congestion_weight
        self.goal = np.array(goal)
        self.start = np.array(start)

    def initial_density(self, x):
        """Gaussian blob centered at start position."""
        dist_sq = np.sum((x - self.start) ** 2, axis=1)
        density = np.exp(-100 * dist_sq)
        return density / (np.sum(density) + 1e-10)

    def terminal_cost(self, x):
        """Quadratic cost: distance to goal."""
        dist_sq = np.sum((x - self.goal) ** 2, axis=1)
        return 5.0 * dist_sq

    def running_cost(self, x, t):
        """Small running cost encourages fast movement."""
        return 0.1 * np.ones(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """H = (1/2)|p|² + κ·m (isotropic control + congestion)."""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        h = 0.5 * p_squared
        h += self.congestion_weight * m
        return h

    def setup_components(self):
        """Setup MFG components for numerical solvers."""
        kappa = self.congestion_weight

        def initial_density_func(x):
            """Gaussian initial density."""
            if np.isscalar(x):
                x = np.array([x])
            x = np.atleast_1d(x)
            if x.shape[0] >= 2:
                dist_sq = np.sum((x[:2] - self.start) ** 2)
            else:
                dist_sq = (x[0] - self.start[0]) ** 2
            return np.exp(-100 * dist_sq)

        def terminal_cost_func(x):
            """Quadratic terminal cost."""
            if np.isscalar(x):
                x = np.array([x])
            x = np.atleast_1d(x)
            if x.shape[0] >= 2:
                dist_sq = np.sum((x[:2] - self.goal) ** 2)
            else:
                dist_sq = (x[0] - self.goal[0]) ** 2
            return 5.0 * dist_sq

        # Class-based Hamiltonian: H = (1/2)|p|² + κ·m
        hamiltonian = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: kappa * m,
            coupling_dm=lambda m: kappa,
        )

        return MFGComponents(
            hamiltonian=hamiltonian,
            m_initial=initial_density_func,
            u_terminal=terminal_cost_func,
        )


def main():
    """Run 2D crowd motion example."""
    print("\n" + "=" * 70)
    print("  2D Crowd Motion with Dimension-Agnostic FDM Solvers")
    print("=" * 70 + "\n")

    # Create problem
    print("Creating 2D MFG problem...")
    problem = CrowdMotion2D(
        grid_resolution=12,  # 12×12 grid
        time_horizon=0.4,
        num_timesteps=15,
        diffusion=0.05,
        congestion_weight=0.3,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    )

    print("  Domain: [0,1] × [0,1]")
    print(f"  Grid: {problem.geometry.grid.num_points[0] - 1} × {problem.geometry.grid.num_points[1] - 1}")
    print(f"  Timesteps: {problem.Nt + 1}")
    print(f"  Start: {problem.start}")
    print(f"  Goal: {problem.goal}")
    print(f"  Congestion weight κ: {problem.congestion_weight}")

    # Create solver using factory
    # Note: Factory automatically detects 2D and uses dimensional splitting!
    print("\nCreating solver with factory (automatic 2D detection)...")
    solver = create_basic_solver(
        problem,
        damping=0.6,  # Damping for Picard iteration
        max_iterations=20,  # Reduced for faster demo (2D is slow with coarse grid)
        tolerance=1e-3,  # Relaxed tolerance for demonstration
    )

    print(f"  HJB solver: {solver.hjb_solver.__class__.__name__} (dimension={solver.hjb_solver.dimension})")
    print(f"  FP solver: {solver.fp_solver.__class__.__name__} (dimension={solver.fp_solver.dimension})")
    print("  Method: Dimensional splitting (Strang)")

    # Solve
    print("\nSolving MFG system...")
    result = solver.solve()

    # Print results
    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.num_iterations}")
    print(f"  Final error: {result.residual:.6e}")

    # Check mass conservation
    U, M = result.U, result.M
    dx, dy = problem.geometry.grid.spacing
    cell_volume = dx * dy
    initial_mass = np.sum(M[0]) * cell_volume
    final_mass = np.sum(M[-1]) * cell_volume
    mass_error = abs(final_mass - initial_mass) / (initial_mass + 1e-10)

    print("\nMass conservation:")
    print(f"  Initial mass: {initial_mass:.6f}")
    print(f"  Final mass: {final_mass:.6f}")
    print(f"  Error: {mass_error:.2%}")

    # Analyze density evolution
    print("\nDensity statistics:")
    print(f"  Initial max: {np.max(M[0]):.6f}")
    print(f"  Final max: {np.max(M[-1]):.6f}")
    print(f"  Initial entropy: {-np.sum(M[0] * np.log(M[0] + 1e-10)) * cell_volume:.4f}")
    print(f"  Final entropy: {-np.sum(M[-1] * np.log(M[-1] + 1e-10)) * cell_volume:.4f}")

    # Analyze value function
    print("\nValue function statistics:")
    print(f"  Initial (at t=0): min={np.min(U[0]):.4f}, max={np.max(U[0]):.4f}")
    print(f"  Final (at t=T): min={np.min(U[-1]):.4f}, max={np.max(U[-1]):.4f}")

    print("\n" + "=" * 70)
    print("  [OK] 2D FDM example complete!")
    print("=" * 70 + "\n")

    print("Key observations:")
    print("  • Factory automatically detected 2D problem")
    print("  • Dimensional splitting used for both HJB and FP")
    print(f"  • Mass error {mass_error:.2%} is typical for FDM splitting ({result.num_iterations} iterations)")
    print("  • No code changes needed vs 1D - dimension-agnostic design works!")

    return result


if __name__ == "__main__":
    main()
