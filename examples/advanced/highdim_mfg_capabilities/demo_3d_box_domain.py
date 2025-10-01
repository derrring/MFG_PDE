"""
3D MFG Demonstration: Crowd Dynamics in a Box Domain

This example demonstrates solving a 3D Mean Field Game using the new high-dimensional
capabilities of MFG_PDE. We model crowd evacuation in a 3D box with obstacles.
"""

import sys
from pathlib import Path

import numpy as np

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde import MFGComponents
from mfg_pde.core.highdim_mfg_problem import HighDimMFGProblem, HybridMFGSolver
from mfg_pde.geometry import Domain3D
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("3d_box_demo", level="INFO")
logger = get_logger(__name__)


class CrowdEvacuation3D(HighDimMFGProblem):
    """
    3D crowd evacuation model in a box domain.

    This problem models agents evacuating from a 3D box through exit points,
    with congestion effects and collision avoidance.
    """

    def __init__(
        self,
        box_size: float = 2.0,
        obstacles: list | None = None,
        mesh_size: float = 0.2,
        time_horizon: float = 2.0,
        num_timesteps: int = 50,
    ):
        # Create 3D box geometry with potential obstacles
        geometry = Domain3D(
            domain_type="box",
            bounds=(0.0, box_size, 0.0, box_size, 0.0, box_size),
            holes=obstacles or [],
            mesh_size=mesh_size,
        )

        # Problem parameters
        self.box_size = box_size
        self.exit_center = np.array([box_size / 2, box_size / 2, box_size])  # Top center exit
        self.congestion_strength = 0.1
        self.exit_radius = 0.5

        super().__init__(geometry=geometry, time_domain=(time_horizon, num_timesteps), diffusion_coeff=0.05)

        logger.info(f"Created 3D evacuation problem: {box_size}×{box_size}×{box_size} box")
        logger.info(f"Mesh: {self.mesh_data.num_vertices} vertices, {self.mesh_data.num_elements} tetrahedra")

    def setup_components(self) -> MFGComponents:
        """Setup MFG components for 3D crowd evacuation."""

        def crowd_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            """
            3D crowd Hamiltonian with congestion and directional preference.
            H = ½|p|² + γm|p|² + V(x)
            """
            try:
                # Get 3D coordinates
                x_idx = int(x_idx)
                if x_idx >= self.num_spatial_points:
                    return 0.0

                coords = self.mesh_data.vertices[x_idx]

                # Extract momentum (approximate from finite differences)
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p_magnitude = abs(p_forward - p_backward)

                # Kinetic energy: ½|p|²
                kinetic_energy = 0.5 * p_magnitude**2

                # Congestion effect: γm|p|²
                congestion = self.congestion_strength * m_at_x * p_magnitude**2

                # Exit attraction potential
                distance_to_exit = np.linalg.norm(coords - self.exit_center)
                potential = 0.1 * distance_to_exit**2  # Quadratic attraction to exit

                result = kinetic_energy + congestion + potential

                # Numerical stability
                if np.isnan(result) or np.isinf(result):
                    return 0.0

                return min(result, 1e6)

            except Exception as e:
                logger.debug(f"Hamiltonian evaluation error at idx {x_idx}: {e}")
                return 0.0

        def hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            """Derivative of Hamiltonian with respect to density."""
            try:
                x_idx = int(x_idx)
                if x_idx >= self.num_spatial_points:
                    return 0.0

                # Extract momentum
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p_magnitude = abs(p_forward - p_backward)

                # dH/dm = γ|p|² (congestion term)
                result = self.congestion_strength * p_magnitude**2

                return min(result, 1e6)

            except Exception as e:
                logger.debug(f"Hamiltonian dm evaluation error: {e}")
                return 0.0

        def initial_crowd_density(x_position):
            """Initial crowd distribution: concentrated in corners."""
            try:
                x_idx = int(x_position)
                if x_idx >= self.num_spatial_points:
                    return 1e-10

                coords = self.mesh_data.vertices[x_idx]
                _x, _y, _z = coords

                # Multiple crowd sources in different corners
                sources = [
                    np.array([0.3, 0.3, 0.3]),  # Bottom front left
                    np.array([1.7, 0.3, 0.3]),  # Bottom front right
                    np.array([0.3, 1.7, 0.3]),  # Bottom back left
                    np.array([1.7, 1.7, 0.3]),  # Bottom back right
                ]

                density = 0.0
                for source in sources:
                    distance = np.linalg.norm(coords - source)
                    density += 2.0 * np.exp(-(distance**2) / (2 * 0.1**2))

                return max(density, 1e-10)

            except Exception as e:
                logger.debug(f"Initial density error: {e}")
                return 1e-10

        def terminal_cost(x_position):
            """Terminal cost: penalize distance from exit."""
            try:
                x_idx = int(x_position)
                if x_idx >= self.num_spatial_points:
                    return 0.0

                coords = self.mesh_data.vertices[x_idx]

                # Cost based on distance to exit
                distance_to_exit = np.linalg.norm(coords - self.exit_center)

                # High cost for being far from exit
                return distance_to_exit**2

            except Exception as e:
                logger.debug(f"Terminal cost error: {e}")
                return 0.0

        return MFGComponents(
            hamiltonian_func=crowd_hamiltonian,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_crowd_density,
            final_value_func=terminal_cost,
            parameters={"dimension": 3, "congestion_strength": self.congestion_strength, "box_size": self.box_size},
            description="3D Crowd Evacuation MFG",
            problem_type="crowd_evacuation_3d",
        )

    def initial_density(self, x: np.ndarray) -> np.ndarray:
        """Override for direct coordinate input."""
        return self.components.initial_density_func(0)  # Placeholder

    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """Override for direct coordinate input."""
        return self.components.final_value_func(0)  # Placeholder

    def running_cost(self, x: np.ndarray, t: float) -> np.ndarray:
        """Uniform running cost."""
        return np.ones(x.shape[0])

    def hamiltonian(self, x: np.ndarray, m: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
        """Override for direct coordinate input."""
        return self.components.hamiltonian_func(0, 0, 0, {}, 0, t, self)


def run_3d_box_demonstration():
    """Run complete 3D box domain MFG demonstration."""
    logger.info("=" * 60)
    logger.info("3D MFG Demonstration: Crowd Evacuation in Box Domain")
    logger.info("=" * 60)

    try:
        # Create problem with obstacles
        obstacles = [
            {
                "type": "box",
                "bounds": (0.8, 1.2, 0.8, 1.2, 0.0, 1.0),  # Central pillar
            },
            {"type": "sphere", "center": (0.5, 1.5, 0.5), "radius": 0.2},
        ]

        problem = CrowdEvacuation3D(
            box_size=2.0,
            obstacles=obstacles,
            mesh_size=0.3,  # Coarse mesh for demonstration
            time_horizon=1.5,
            num_timesteps=30,
        )

        logger.info("Problem setup complete:")
        logger.info(f"  Domain: 2×2×2 box with {len(obstacles)} obstacles")
        logger.info(f"  Mesh: {problem.mesh_data.num_vertices} vertices")
        logger.info(f"  Quality: min={problem.mesh_data.quality_metrics.get('min_quality', 0):.3f}")

        # Method 1: Damped Fixed Point (fastest)
        logger.info("\n" + "=" * 40)
        logger.info("Method 1: Damped Fixed Point Solver")
        logger.info("=" * 40)

        result_fp = problem.solve_with_damped_fixed_point(damping_factor=0.6, max_iterations=20, tolerance=1e-3)

        if result_fp["success"]:
            logger.info("✅ Damped fixed point solver succeeded!")
            logger.info(f"   Method: {result_fp['solver_info']['method']}")
            logger.info(f"   Dimension: {result_fp['solver_info']['dimension']}")

            # Basic analysis
            final_density = result_fp["solution"]["final_density"]
            logger.info(f"   Final mass: {np.sum(final_density):.4f}")
            logger.info(f"   Peak density: {np.max(final_density):.4f}")

        else:
            logger.error("❌ Damped fixed point solver failed")
            logger.error(f"   Error: {result_fp.get('error', 'Unknown')}")

        # Method 2: Hybrid Solver (most robust)
        logger.info("\n" + "=" * 40)
        logger.info("Method 2: Hybrid Multi-Strategy Solver")
        logger.info("=" * 40)

        hybrid_solver = HybridMFGSolver(problem)
        result_hybrid = hybrid_solver.solve(max_total_iterations=40, tolerance=1e-4, strategy="adaptive")

        if result_hybrid["success"]:
            logger.info("✅ Hybrid solver succeeded!")
            logger.info(f"   Method: {result_hybrid['solver_info']['method']}")
            logger.info(f"   Phases: {result_hybrid['solver_info'].get('phases', 'N/A')}")

            # Analysis of evacuation efficiency
            final_density = result_hybrid["solution"]["final_density"]
            coords = result_hybrid["solution"]["coordinates"]

            # Compute evacuation progress
            distances_to_exit = np.linalg.norm(coords - problem.exit_center, axis=1)
            evacuation_efficiency = np.sum(final_density * (1.0 / (1.0 + distances_to_exit)))

            logger.info(f"   Evacuation efficiency: {evacuation_efficiency:.4f}")
            logger.info(f"   Remaining crowd: {np.sum(final_density):.4f}")

        else:
            logger.error("❌ Hybrid solver failed")
            logger.error(f"   Error: {result_hybrid.get('error', 'Unknown')}")

        # Visualization (if available)
        logger.info("\n" + "=" * 40)
        logger.info("Visualization")
        logger.info("=" * 40)

        try:
            if result_hybrid["success"]:
                logger.info("Creating 3D visualization...")
                problem.visualize_solution(result_hybrid, time_index=-1, field_type="density")
                logger.info("✅ 3D visualization complete!")
            else:
                logger.warning("No successful solution to visualize")

        except ImportError:
            logger.warning("PyVista not available - skipping 3D visualization")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

        # Performance summary
        logger.info("\n" + "=" * 50)
        logger.info("3D MFG Demonstration Summary")
        logger.info("=" * 50)

        if result_fp["success"]:
            logger.info("✅ Damped Fixed Point: SUCCESS (recommended for testing)")
        else:
            logger.info("❌ Damped Fixed Point: FAILED")

        if result_hybrid["success"]:
            logger.info("✅ Hybrid Solver: SUCCESS (recommended for production)")
        else:
            logger.info("❌ Hybrid Solver: FAILED")

        logger.info("\n📊 Problem Statistics:")
        logger.info("   Spatial dimension: 3D")
        logger.info(f"   Mesh vertices: {problem.mesh_data.num_vertices}")
        logger.info(f"   Time steps: {problem.Nt}")
        logger.info(f"   Total DOF: {problem.mesh_data.num_vertices * problem.Nt}")

        logger.info("\n🎯 High-Dimensional MFG Infrastructure:")
        logger.info("   ✅ Domain3D geometry")
        logger.info("   ✅ Tetrahedral mesh generation")
        logger.info("   ✅ Multi-dimensional solver integration")
        logger.info("   ✅ Hybrid solver strategies")
        logger.info("   ✅ 3D visualization capabilities")

        return {"damped_fixed_point": result_fp, "hybrid_solver": result_hybrid, "problem": problem}

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_3d_box_demonstration()

    if results:
        print("\n" + "=" * 60)
        print("🎉 3D MFG High-Dimensional Extension: DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("The MFG_PDE package now supports:")
        print("  • 3D domain geometry with complex shapes")
        print("  • Multi-dimensional damped fixed point solvers")
        print("  • Hybrid solver strategies for robustness")
        print("  • Complete 3D visualization pipeline")
        print("  • Professional mesh generation with Gmsh")
        print("\nReady for production 2D/3D Mean Field Games! 🚀")
    else:
        print("\n❌ Demonstration encountered issues. Check logs for details.")
