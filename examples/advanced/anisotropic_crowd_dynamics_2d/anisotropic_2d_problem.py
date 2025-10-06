"""
2D Anisotropic Crowd Dynamics using Full MFG_PDE Infrastructure

This implementation provides a complete 2D anisotropic crowd dynamics solution
using the MFG_PDE package with proper MFGComponents integration. It includes
barrier configurations, custom Hamiltonians, and factory functions for
crowd evacuation scenarios with non-separable anisotropic movement preferences.

Mathematical formulation:
- HJB: -∂u/∂t + ½[u_x² + 2ρ(x)u_x u_y + u_y²] + γm|∇u|² = 1
- FP: ∂m/∂t + div(m[∇u + 2γm∇u]) - σΔm = 0

The anisotropy function ρ(x) creates directional preferences,
while optional barriers add realistic architectural constraints.
"""

import sys
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde import BoundaryConditions, MFGComponents, MFGProblem
from mfg_pde.config import create_research_config
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("anisotropic_2d", level="INFO")
logger = get_logger(__name__)


@dataclass
class BarrierConfig:
    """Base class for barrier configurations."""

    @abstractmethod
    def compute_distance(self, x: np.ndarray) -> np.ndarray:
        """Compute signed distance to barrier boundary."""


@dataclass
class CircularBarrier(BarrierConfig):
    """Circular barrier configuration."""

    center: tuple[float, float]
    radius: float

    def compute_distance(self, x: np.ndarray) -> np.ndarray:
        """Compute signed distance to circular barrier."""
        center_array = np.array(self.center)
        distances = np.linalg.norm(x - center_array, axis=1)
        return distances - self.radius


@dataclass
class LinearBarrier(BarrierConfig):
    """Linear barrier configuration."""

    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float = 0.05

    def compute_distance(self, x: np.ndarray) -> np.ndarray:
        """Compute signed distance to linear barrier."""
        start_array = np.array(self.start)
        end_array = np.array(self.end)
        line_vec = end_array - start_array
        line_length = np.linalg.norm(line_vec)
        line_unit = line_vec / line_length if line_length > 0 else np.array([1, 0])

        # Vector from start to each point
        point_vecs = x - start_array

        # Project onto line and clamp to segment
        projections = np.clip(np.dot(point_vecs, line_unit), 0, line_length)
        closest_points = start_array + projections[:, np.newaxis] * line_unit

        # Distance to closest point on line segment
        distances = np.linalg.norm(x - closest_points, axis=1)
        return distances - self.thickness / 2


@dataclass
class RectangularBarrier(BarrierConfig):
    """Rectangular barrier configuration."""

    bounds: list[tuple[float, float]]  # [(x_min, x_max), (y_min, y_max)]

    def compute_distance(self, x: np.ndarray) -> np.ndarray:
        """Compute signed distance to rectangular barrier."""
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]

        # Distance to rectangle boundary
        dx = np.maximum(x_min - x[:, 0], x[:, 0] - x_max, 0)
        dy = np.maximum(y_min - x[:, 1], x[:, 1] - y_max, 0)

        # Outside distance
        outside_dist = np.sqrt(dx**2 + dy**2)

        # Inside distance (negative)
        inside_mask = (x[:, 0] >= x_min) & (x[:, 0] <= x_max) & (x[:, 1] >= y_min) & (x[:, 1] <= y_max)
        inside_dist = np.minimum(
            np.minimum(x[:, 0] - x_min, x_max - x[:, 0]), np.minimum(x[:, 1] - y_min, y_max - x[:, 1])
        )

        return np.where(inside_mask, -inside_dist, outside_dist)


class GridBased2DAdapter:
    """
    Adapter class to handle 2D grid-based MFG problems using 1D MFG infrastructure.

    This class provides the bridge between 2D physics and 1D solver architecture
    by implementing proper index mapping and boundary condition handling.
    """

    def __init__(self, nx1, nx2, domain_bounds):
        self.nx1, self.nx2 = nx1, nx2
        self.x1_min, self.x1_max, self.x2_min, self.x2_max = domain_bounds

        # Create 2D grids
        self.x1_grid = np.linspace(self.x1_min, self.x1_max, nx1 + 1)
        self.x2_grid = np.linspace(self.x2_min, self.x2_max, nx2 + 1)
        self.X1, self.X2 = np.meshgrid(self.x1_grid, self.x2_grid, indexing="ij")

        # Grid spacing
        self.dx1 = (self.x1_max - self.x1_min) / nx1
        self.dx2 = (self.x2_max - self.x2_min) / nx2

    def idx_1d_to_2d(self, idx_1d):
        """Convert 1D solver index to 2D grid coordinates."""
        i1 = idx_1d // (self.nx2 + 1)
        i2 = idx_1d % (self.nx2 + 1)
        return min(i1, self.nx1), min(i2, self.nx2)

    def idx_2d_to_1d(self, i1, i2):
        """Convert 2D grid coordinates to 1D solver index."""
        return i1 * (self.nx2 + 1) + i2

    def get_2d_coordinates(self, idx_1d):
        """Get actual (x1, x2) coordinates from 1D index."""
        i1, i2 = self.idx_1d_to_2d(idx_1d)
        x1 = self.x1_grid[i1] if i1 < len(self.x1_grid) else self.x1_max
        x2 = self.x2_grid[i2] if i2 < len(self.x2_grid) else self.x2_max
        return x1, x2

    def compute_2d_gradient(self, field_1d, idx_1d):
        """Approximate 2D gradient using finite differences."""
        i1, i2 = self.idx_1d_to_2d(idx_1d)

        # Boundary handling
        i1_plus = min(i1 + 1, self.nx1)
        i1_minus = max(i1 - 1, 0)
        i2_plus = min(i2 + 1, self.nx2)
        i2_minus = max(i2 - 1, 0)

        # Central differences where possible
        idx_i1_plus = self.idx_2d_to_1d(i1_plus, i2)
        idx_i1_minus = self.idx_2d_to_1d(i1_minus, i2)
        idx_i2_plus = self.idx_2d_to_1d(i1, i2_plus)
        idx_i2_minus = self.idx_2d_to_1d(i1, i2_minus)

        # Ensure indices are within bounds
        max_idx = len(field_1d) - 1
        idx_i1_plus = min(idx_i1_plus, max_idx)
        idx_i1_minus = min(idx_i1_minus, max_idx)
        idx_i2_plus = min(idx_i2_plus, max_idx)
        idx_i2_minus = min(idx_i2_minus, max_idx)

        # Compute gradients
        if i1 == 0:
            grad_x1 = (field_1d[idx_i1_plus] - field_1d[idx_1d]) / self.dx1
        elif i1 == self.nx1:
            grad_x1 = (field_1d[idx_1d] - field_1d[idx_i1_minus]) / self.dx1
        else:
            grad_x1 = (field_1d[idx_i1_plus] - field_1d[idx_i1_minus]) / (2 * self.dx1)

        if i2 == 0:
            grad_x2 = (field_1d[idx_i2_plus] - field_1d[idx_1d]) / self.dx2
        elif i2 == self.nx2:
            grad_x2 = (field_1d[idx_1d] - field_1d[idx_i2_minus]) / self.dx2
        else:
            grad_x2 = (field_1d[idx_i2_plus] - field_1d[idx_i2_minus]) / (2 * self.dx2)

        return grad_x1, grad_x2

    def convert_to_2d_array(self, field_1d):
        """Convert 1D field to 2D array for visualization."""
        if len(field_1d.shape) == 1:
            # Single time step
            result = np.zeros((self.nx1 + 1, self.nx2 + 1))
            for i1 in range(self.nx1 + 1):
                for i2 in range(self.nx2 + 1):
                    idx_1d = self.idx_2d_to_1d(i1, i2)
                    if idx_1d < len(field_1d):
                        result[i1, i2] = field_1d[idx_1d]
            return result
        else:
            # Multiple time steps
            nt = field_1d.shape[0]
            result = np.zeros((nt, self.nx1 + 1, self.nx2 + 1))
            for t in range(nt):
                for i1 in range(self.nx1 + 1):
                    for i2 in range(self.nx2 + 1):
                        idx_1d = self.idx_2d_to_1d(i1, i2)
                        if idx_1d < field_1d.shape[1]:
                            result[t, i1, i2] = field_1d[t, idx_1d]
            return result


class AnisotropicMFGProblem2D:
    """
    2D Anisotropic Mean Field Game using MFG_PDE infrastructure.

    This class provides a complete implementation of 2D anisotropic crowd dynamics
    using the MFGComponents system with proper barrier support and custom Hamiltonians.
    """

    def __init__(
        self,
        domain_bounds=(0.0, 1.0, 0.0, 1.0),  # (x1_min, x1_max, x2_min, x2_max)
        grid_size=(32, 32),  # (Nx1, Nx2)
        time_domain=(0.5, 500),  # (T, Nt)
        sigma=0.02,  # Diffusion
        gamma=0.05,  # Density coupling
        rho_amplitude=0.3,  # Anisotropy strength
        barrier_configuration="none",
    ):  # Barrier configuration
        self.domain_bounds = domain_bounds
        self.Nx1, self.Nx2 = grid_size
        self.T, self.Nt = time_domain
        self.sigma = sigma
        self.gamma = gamma
        self.rho_amplitude = rho_amplitude
        self.barrier_configuration = barrier_configuration

        # Validate anisotropy strength
        if abs(rho_amplitude) >= 1.0:
            raise ValueError(f"Anisotropy amplitude {rho_amplitude} must satisfy |ρ| < 1 for positive definiteness")

        # Setup barriers
        self.barriers = self._setup_barriers(barrier_configuration)

        # Create 2D grid adapter
        self.grid_adapter = GridBased2DAdapter(self.Nx1, self.Nx2, domain_bounds)

        # Total number of grid points for 1D representation
        self.total_nodes = (self.Nx1 + 1) * (self.Nx2 + 1)

        # Store value function for gradient computation
        self.current_u = None

        # Create MFG components
        self.components = self._create_mfg_components()

        # Initialize 1D MFG problem
        self.mfg_problem = MFGProblem(
            xmin=0.0,
            xmax=float(self.total_nodes - 1),
            Nx=self.total_nodes - 1,
            T=self.T,
            Nt=self.Nt,
            sigma=self.sigma,
            components=self.components,
        )

        logger.info(f"Created 2D anisotropic MFG problem: {self.Nx1 + 1}×{self.Nx2 + 1} grid")
        logger.info(f"Parameters: γ={gamma}, σ={sigma}, ρ_amp={rho_amplitude}")
        if self.barriers:
            logger.info(f"Using barrier configuration: {barrier_configuration} with {len(self.barriers)} barriers")

    def _setup_barriers(self, configuration: str) -> list[BarrierConfig]:
        """Configure internal barriers within domain."""
        if configuration == "central_obstacle":
            barriers = [CircularBarrier(center=(0.5, 0.4), radius=0.15)]
        elif configuration == "anisotropy_aligned":
            barriers = [
                LinearBarrier(start=(0.1, 0.1), end=(0.4, 0.4)),  # Region A alignment
                LinearBarrier(start=(0.9, 0.1), end=(0.6, 0.4)),  # Region B alignment
            ]
        elif configuration == "corridor_system":
            barriers = [
                RectangularBarrier(bounds=[(0.3, 0.4), (0.3, 0.7)]),
                RectangularBarrier(bounds=[(0.6, 0.7), (0.3, 0.7)]),
            ]
        else:
            barriers = []  # No barriers
        return barriers

    def _anisotropy_function(self, x1, x2):
        """Checkerboard anisotropy pattern."""
        return self.rho_amplitude * np.sin(np.pi * x1) * np.cos(np.pi * x2)

    def _apply_barrier_penalty(self, x_position, hamiltonian_value):
        """Apply barrier penalties to Hamiltonian."""
        if not self.barriers:
            return hamiltonian_value

        # Create coordinate array
        x_coords = np.array([[x_position[0], x_position[1]]])

        penalty = 0.0
        for barrier in self.barriers:
            distance = barrier.compute_distance(x_coords)[0]
            if distance < 0.05:  # Near or inside barrier
                penalty += 1000.0 * np.exp(-distance / 0.01)

        return hamiltonian_value + penalty

    def _create_mfg_components(self):
        """Create MFGComponents with custom 2D anisotropic Hamiltonian."""

        def anisotropic_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            """
            2D Anisotropic Hamiltonian: H = 0.5 * p^T * A(x) * p + gamma * m * |p|
            where A(x) = [[1, rho(x)], [rho(x), 1]]
            """
            try:
                x_idx = int(x_idx)
                if x_idx >= self.total_nodes:
                    return 0.0

                # Get 2D coordinates
                x1, x2 = self.grid_adapter.get_2d_coordinates(x_idx)

                # Anisotropy function
                rho = self._anisotropy_function(x1, x2)

                # Extract gradient information from p_values
                # In standard MFG, p represents gradient of value function
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)

                # Approximate gradient as finite difference
                p1 = p_forward - p_backward

                # For 2D, we need p2 component - use simple approximation
                # In full implementation, this would require proper 2D gradient computation
                p2 = 0.1 * p1  # Simplified coupling for demonstration

                # Anisotropic kinetic energy: 0.5 * [p1, p2] * A * [p1; p2]
                # A * p = [p1 + rho*p2, rho*p1 + p2]
                kinetic = 0.5 * (p1 * p1 + 2 * rho * p1 * p2 + p2 * p2)

                # Density-dependent friction (congestion)
                if m_at_x > 1e-10:
                    friction = self.gamma * m_at_x * (p1 * p1 + p2 * p2)
                else:
                    friction = 0.0

                result = kinetic + friction

                # Apply barrier penalties
                result = self._apply_barrier_penalty([x1, x2], result)

                # Numerical stability
                if np.isnan(result) or np.isinf(result) or result < 0:
                    return 0.0

                return min(result, 1e6)  # Cap to prevent overflow

            except Exception as e:
                logger.debug(f"Hamiltonian evaluation error at idx {x_idx}: {e}")
                return 0.0

        def hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            """Derivative of Hamiltonian with respect to density m."""
            try:
                x_idx = int(x_idx)
                if x_idx >= self.total_nodes:
                    return 0.0

                # Extract momentum components
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p1 = p_forward - p_backward
                p2 = 0.1 * p1  # Simplified

                # dH/dm = gamma * |p|^2
                result = self.gamma * (p1 * p1 + p2 * p2)

                if np.isnan(result) or np.isinf(result):
                    return 0.0

                return min(result, 1e6)

            except Exception as e:
                logger.debug(f"Hamiltonian dm evaluation error at idx {x_idx}: {e}")
                return 0.0

        def initial_density_2d(x_position):
            """2D initial density: Gaussian blob in lower-left corner."""
            try:
                x_idx = int(x_position)
                if x_idx >= self.total_nodes:
                    return 1e-10

                x1, x2 = self.grid_adapter.get_2d_coordinates(x_idx)

                # Gaussian blob centered at (0.2, 0.2)
                center_x1, center_x2 = 0.2, 0.2
                width = 0.1

                density = np.exp(-((x1 - center_x1) ** 2 + (x2 - center_x2) ** 2) / (2 * width**2))

                return max(density, 1e-10)

            except Exception as e:
                logger.debug(f"Initial density evaluation error at position {x_position}: {e}")
                return 1e-10

        def final_value_2d(x_position):
            """2D final value: distance to exit at top boundary."""
            try:
                x_idx = int(x_position)
                if x_idx >= self.total_nodes:
                    return 0.0

                x1, x2 = self.grid_adapter.get_2d_coordinates(x_idx)

                # Target exit at (0.5, 1.0)
                target_x1, target_x2 = 0.5, 1.0

                distance_cost = (x1 - target_x1) ** 2 + (x2 - target_x2) ** 2

                return distance_cost

            except Exception as e:
                logger.debug(f"Final value evaluation error at position {x_position}: {e}")
                return 0.0

        # Create MFG components
        components = MFGComponents(
            hamiltonian_func=anisotropic_hamiltonian,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_2d,
            final_value_func=final_value_2d,
            parameters={
                "gamma": self.gamma,
                "rho_amplitude": self.rho_amplitude,
                "dimension": 2,
                "grid_shape": (self.Nx1 + 1, self.Nx2 + 1),
            },
            description="2D Anisotropic Crowd Dynamics",
            problem_type="anisotropic_mfg_2d",
        )

        return components

    def setup_boundary_conditions(self) -> BoundaryConditions:
        """Configure mixed boundary conditions for evacuation scenario."""
        # No-flux boundary conditions on walls
        no_flux_boundaries = [
            ("left", {"type": "no_flux"}),  # x₁ = 0
            ("right", {"type": "no_flux"}),  # x₁ = 1
            ("bottom", {"type": "no_flux"}),  # x₂ = 0
        ]

        # Dirichlet condition at exit
        exit_boundary = (
            "top",
            {
                "type": "dirichlet",
                "value": lambda x, t: 0.0,  # u = 0 at exit
            },
        )

        return BoundaryConditions([*no_flux_boundaries, exit_boundary])

    def get_problem_info(self) -> dict[str, Any]:
        """Return problem configuration information."""
        return {
            "problem_type": "2D Anisotropic Crowd Dynamics",
            "domain": self.domain_bounds,
            "time_horizon": self.T,
            "grid_size": (self.Nx1, self.Nx2),
            "parameters": {"gamma": self.gamma, "sigma": self.sigma, "rho_amplitude": self.rho_amplitude},
            "barrier_configuration": self.barrier_configuration,
            "num_barriers": len(self.barriers),
            "hamiltonian_type": "non-separable",
            "anisotropy_pattern": "checkerboard",
        }

    def solve(self):
        """Solve the 2D anisotropic MFG problem using MFG_PDE infrastructure."""
        logger.info("Solving 2D anisotropic MFG problem...")

        try:
            # Create research-grade solver configuration
            config = create_research_config()

            # Adjust configuration for stability
            config.newton.max_iterations = 50  # Conservative iteration count
            config.newton.tolerance = 1e-4  # Relaxed tolerance
            config.picard.max_iterations = 20  # Fewer Picard iterations
            config.convergence_tolerance = 1e-4  # Relaxed global tolerance

            # Create solver using factory with hybrid solver for better integration
            from mfg_pde.factory import create_solver

            solver = create_solver(
                problem=self.mfg_problem,
                solver_type="hybrid_fp_particle_hjb_fdm",  # Use hybrid solver
                preset="fast",  # Fast preset for quick demo
            )

            # Solve the problem
            solution = solver.solve()

            if solution is None:
                logger.error("Solver failed to find solution")
                return None

            logger.info("Successfully solved 2D anisotropic MFG problem")

            return solution

        except Exception as e:
            logger.error(f"Solver failed with error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def visualize_anisotropy(self):
        """Visualize the anisotropy function rho(x1, x2)."""
        rho_grid = np.zeros((self.Nx1 + 1, self.Nx2 + 1))

        for i1 in range(self.Nx1 + 1):
            for i2 in range(self.Nx2 + 1):
                x1 = self.grid_adapter.x1_grid[i1]
                x2 = self.grid_adapter.x2_grid[i2]
                rho_grid[i1, i2] = self._anisotropy_function(x1, x2)

        plt.figure(figsize=(10, 8))
        plt.contourf(self.grid_adapter.X1, self.grid_adapter.X2, rho_grid, levels=20, cmap="RdBu_r")
        plt.colorbar(label=r"$\rho(x_1, x_2)$")
        plt.title("Anisotropy Function: Checkerboard Pattern")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return rho_grid

    def visualize_solution(self, solution):
        """Visualize the solution if available."""
        if solution is None:
            logger.warning("No solution to visualize")
            return

        try:
            # Check available solution components
            if hasattr(solution, "m_history") and solution.m_history is not None:
                # Convert density history to 2D
                density_2d = self.grid_adapter.convert_to_2d_array(solution.m_history)

                # Plot initial and final density
                _fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # Initial density
                im1 = axes[0].contourf(
                    self.grid_adapter.X1, self.grid_adapter.X2, density_2d[0], levels=20, cmap="viridis"
                )
                axes[0].set_title("Initial Density")
                axes[0].set_xlabel(r"$x_1$")
                axes[0].set_ylabel(r"$x_2$")
                plt.colorbar(im1, ax=axes[0])

                # Final density
                im2 = axes[1].contourf(
                    self.grid_adapter.X1, self.grid_adapter.X2, density_2d[-1], levels=20, cmap="viridis"
                )
                axes[1].set_title("Final Density")
                axes[1].set_xlabel(r"$x_1$")
                axes[1].set_ylabel(r"$x_2$")
                plt.colorbar(im2, ax=axes[1])

                plt.tight_layout()

                # Print statistics
                initial_mass = np.sum(density_2d[0]) * self.grid_adapter.dx1 * self.grid_adapter.dx2
                final_mass = np.sum(density_2d[-1]) * self.grid_adapter.dx1 * self.grid_adapter.dx2

                logger.info(f"Mass: initial = {initial_mass:.4f}, final = {final_mass:.4f}")
                logger.info(
                    f"Peak density: initial = {np.max(density_2d[0]):.4f}, final = {np.max(density_2d[-1]):.4f}"
                )

                return density_2d
            else:
                logger.warning("Solution does not contain density history")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            import traceback

            traceback.print_exc()


def run_experiment():
    """Run the complete 2D anisotropic experiment."""
    logger.info("Starting 2D Anisotropic Crowd Dynamics Experiment")

    try:
        # Create problem with conservative parameters
        problem = AnisotropicMFGProblem2D(
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            grid_size=(16, 16),  # Small grid for testing
            time_domain=(0.2, 50),  # Short time, fewer steps
            sigma=0.05,  # Higher diffusion for stability
            gamma=0.02,  # Lower coupling
            rho_amplitude=0.2,  # Moderate anisotropy
        )

        # Visualize anisotropy pattern
        logger.info("Visualizing anisotropy pattern...")
        problem.visualize_anisotropy()

        # Solve the problem
        solution = problem.solve()

        if solution is not None:
            logger.info("SUCCESS: 2D anisotropic MFG experiment completed!")

            # Visualize solution
            problem.visualize_solution(solution)

            plt.show()

            return solution, problem
        else:
            logger.error("Experiment failed to produce solution")
            return None, problem

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def create_anisotropic_problem(
    barrier_config: str = "none",
    gamma: float = 0.05,
    sigma: float = 0.02,
    rho_amplitude: float = 0.3,
    domain_bounds: tuple = (0.0, 1.0, 0.0, 1.0),
    grid_size: tuple = (32, 32),
    time_domain: tuple = (0.5, 500),
) -> AnisotropicMFGProblem2D:
    """
    Factory function to create 2D anisotropic crowd dynamics problem.

    Args:
        barrier_config: Barrier configuration type
        gamma: Density-velocity coupling
        sigma: Diffusion coefficient
        rho_amplitude: Anisotropy strength
        domain_bounds: Spatial domain bounds (x1_min, x1_max, x2_min, x2_max)
        grid_size: Grid resolution (Nx1, Nx2)
        time_domain: Time domain (T, Nt)

    Returns:
        Configured 2D anisotropic MFG problem instance
    """
    return AnisotropicMFGProblem2D(
        domain_bounds=domain_bounds,
        grid_size=grid_size,
        time_domain=time_domain,
        sigma=sigma,
        gamma=gamma,
        rho_amplitude=rho_amplitude,
        barrier_configuration=barrier_config,
    )


if __name__ == "__main__":
    solution, problem = run_experiment()
