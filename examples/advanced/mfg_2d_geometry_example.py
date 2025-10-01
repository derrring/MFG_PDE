"""
Advanced example: MFG problem on 2D complex geometry using new mesh pipeline.

This example demonstrates solving a Mean Field Game on a 2D domain with
complex geometry using the Gmsh → Meshio → PyVista pipeline integrated
with MFG_PDE's existing solver architecture.
"""

import sys
from pathlib import Path

import numpy as np

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde.config import OmegaConfManager
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import BoundaryManager, Domain2D, MeshPipeline
from mfg_pde.utils.logging import configure_research_logging, get_logger
from mfg_pde.utils.polars_integration import create_analysis_dataframe

# Configure logging
configure_research_logging("mfg_2d_geometry", level="INFO")
logger = get_logger(__name__)


class MFGProblem2D(MFGProblem):
    """
    Extended MFG problem class for 2D domains with complex geometry.

    This class bridges the new geometry system with the existing MFG solver framework,
    providing a template for future 2D/3D MFG implementations.
    """

    def __init__(self, geometry_config: dict, time_domain: tuple = (1.0, 51), sigma: float = 1.0, **kwargs):
        """
        Initialize 2D MFG problem.

        Args:
            geometry_config: Configuration for 2D domain geometry
            time_domain: (T_final, N_time_steps)
            sigma: Diffusion coefficient
            **kwargs: Additional MFG problem parameters
        """

        # Create 2D geometry
        self.geometry = Domain2D(**geometry_config)
        self.mesh_data = self.geometry.generate_mesh()

        # Setup boundary conditions
        self.boundary_manager = BoundaryManager(self.mesh_data)
        self._setup_boundary_conditions()

        # Initialize parent with approximated 1D parameters for compatibility
        # This is a temporary bridge - full 2D implementation would override parent methods
        bounds_min, bounds_max = self.mesh_data.bounds
        super().__init__(
            xmin=bounds_min[0],
            xmax=bounds_max[0],
            Nx=int(np.sqrt(self.mesh_data.num_vertices)),  # Approximate 1D grid size
            T=time_domain[0],
            Nt=time_domain[1],
            sigma=sigma,
            **kwargs,
        )

        # 2D-specific attributes
        self.dimension = 2
        self.num_nodes_2d = self.mesh_data.num_vertices
        self.num_elements_2d = self.mesh_data.num_elements

        logger.info(f"Initialized 2D MFG problem: {self.num_nodes_2d} nodes, {self.num_elements_2d} elements")

    def _setup_boundary_conditions(self):
        """Setup boundary conditions for the 2D domain."""
        # This is a placeholder - specific boundary conditions depend on the problem

        # Example: Dirichlet boundary conditions on all boundaries
        # In a real implementation, this would be problem-specific
        for region_id in [1, 2, 3, 4]:  # Assuming 4 boundary regions for rectangle
            try:
                self.boundary_manager.add_boundary_condition(
                    region_id=region_id,
                    bc_type="dirichlet",
                    value=0.0,
                    description=f"Homogeneous Dirichlet on region {region_id}",
                )
            except:
                # Skip if region doesn't exist
                pass

    def get_2d_mesh_info(self) -> dict:
        """Get detailed 2D mesh information."""
        return {
            "num_vertices": self.mesh_data.num_vertices,
            "num_elements": self.mesh_data.num_elements,
            "element_type": self.mesh_data.element_type,
            "dimension": self.mesh_data.dimension,
            "bounds": self.mesh_data.bounds,
            "quality_metrics": self.geometry.compute_mesh_quality(),
        }

    def visualize_geometry(self):
        """Visualize the 2D geometry and mesh."""
        pipeline = MeshPipeline(self.geometry)
        pipeline.visualize_interactive(show_quality=True)

    def export_mesh(self, output_dir: str = "./output"):
        """Export mesh in multiple formats for external solvers."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        formats = ["msh", "vtk", "xdmf"]  # Common formats for FEM solvers
        for fmt in formats:
            filename = output_path / f"mfg_mesh.{fmt}"
            self.geometry.export_mesh(fmt, str(filename))
            logger.info(f"Exported mesh: {filename}")


def create_rectangle_with_holes_problem():
    """Create MFG problem on rectangular domain with circular holes."""
    logger.info("Creating MFG problem on rectangle with holes")

    # Define geometry with holes
    geometry_config = {
        "domain_type": "rectangle",
        "bounds": (0.0, 2.0, 0.0, 1.0),
        "holes": [
            {"type": "circle", "center": (0.5, 0.5), "radius": 0.2},
            {"type": "circle", "center": (1.5, 0.5), "radius": 0.15},
        ],
        "mesh_size": 0.08,
    }

    # Create MFG problem
    problem = MFGProblem2D(geometry_config=geometry_config, time_domain=(1.0, 51), sigma=0.5, coefCT=1.0)

    return problem


def create_l_shaped_domain_problem():
    """Create MFG problem on L-shaped domain."""
    logger.info("Creating MFG problem on L-shaped domain")

    # Define L-shaped geometry
    l_vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)]

    geometry_config = {"domain_type": "polygon", "vertices": l_vertices, "mesh_size": 0.06}

    # Create MFG problem
    problem = MFGProblem2D(geometry_config=geometry_config, time_domain=(1.0, 51), sigma=0.8)

    return problem


def analyze_mesh_convergence():
    """Analyze mesh convergence by comparing different mesh resolutions."""
    logger.info("Analyzing mesh convergence")

    mesh_sizes = [0.2, 0.1, 0.05, 0.025]
    results = []

    for mesh_size in mesh_sizes:
        geometry_config = {"domain_type": "circle", "center": (0.0, 0.0), "radius": 1.0, "mesh_size": mesh_size}

        problem = MFGProblem2D(geometry_config=geometry_config)
        mesh_info = problem.get_2d_mesh_info()

        results.append(
            {
                "mesh_size": mesh_size,
                "num_vertices": mesh_info["num_vertices"],
                "num_elements": mesh_info["num_elements"],
                "min_area": mesh_info["quality_metrics"]["min_area"],
                "max_area": mesh_info["quality_metrics"]["max_area"],
                "mean_aspect_ratio": mesh_info["quality_metrics"]["mean_aspect_ratio"],
            }
        )

        logger.info(f"Mesh size {mesh_size}: {mesh_info['num_vertices']} vertices")

    # Create analysis dataframe
    df = create_analysis_dataframe(results)
    logger.info("Mesh convergence analysis:")
    logger.info(df.to_pandas())

    return df


def demonstrate_advanced_boundary_conditions():
    """Demonstrate advanced boundary conditions with spatial variation."""
    logger.info("Demonstrating advanced boundary conditions")

    # Create simple domain
    geometry_config = {"domain_type": "rectangle", "bounds": (0.0, 1.0, 0.0, 1.0), "mesh_size": 0.1}

    problem = MFGProblem2D(geometry_config=geometry_config)
    boundary_manager = problem.boundary_manager

    # Add spatially-varying boundary conditions

    # Bottom: u = x(1-x) (parabolic profile)
    boundary_manager.add_boundary_condition(
        region_id=1,
        bc_type="dirichlet",
        value=lambda coords: coords[:, 0] * (1 - coords[:, 0]),
        description="Bottom: parabolic profile u = x(1-x)",
    )

    # Right: Neumann with du/dn = sin(2πy)
    boundary_manager.add_boundary_condition(
        region_id=2,
        bc_type="neumann",
        gradient_value=lambda coords: np.sin(2 * np.pi * coords[:, 1]),
        description="Right: du/dn = sin(2πy)",
    )

    # Top: Time-dependent Dirichlet u = t*exp(-x²)
    def time_dependent_top(coords, time):
        return time * np.exp(-(coords[:, 0] ** 2))

    boundary_manager.add_boundary_condition(
        region_id=3,
        bc_type="dirichlet",
        time_dependent=True,
        spatial_function=time_dependent_top,
        description="Top: time-dependent u = t*exp(-x²)",
    )

    # Left: Robin boundary condition αu + β(du/dn) = g
    boundary_manager.add_boundary_condition(
        region_id=4,
        bc_type="robin",
        alpha=1.0,
        beta=0.5,
        value=lambda coords: np.ones(len(coords)),
        description="Left: Robin BC with α=1, β=0.5",
    )

    # Print summary
    summary = boundary_manager.get_summary()
    logger.info("Advanced boundary conditions summary:")
    for region_id, info in summary["regions"].items():
        logger.info(f"  Region {region_id}: {info['description']}")

    return boundary_manager


def integration_with_existing_config():
    """Demonstrate integration with existing OmegaConf configuration system."""
    logger.info("Demonstrating OmegaConf integration")

    # Create configuration that includes geometry
    config_dict = {
        "problem": {"type": "ExampleMFGProblem", "T": 1.0, "Nt": 51, "sigma": 0.5, "coefCT": 1.0},
        "geometry": {
            "dimension": 2,
            "domain_type": "rectangle",
            "bounds": [0.0, 2.0, 0.0, 1.0],
            "holes": [{"type": "circle", "center": [1.0, 0.5], "radius": 0.3}],
            "mesh_size": 0.08,
        },
        "solver": {"type": "hybrid_fem", "max_iterations": 100, "tolerance": 1e-6},
        "visualization": {"export_formats": ["vtk", "xdmf"], "quality_analysis": True, "interactive_plots": True},
    }

    # Use OmegaConf to manage configuration
    config_manager = OmegaConfManager(config_dict)
    config = config_manager.get_config()

    # Create problem from configuration
    problem = MFGProblem2D(
        geometry_config=dict(config.geometry),
        time_domain=(config.problem.T, config.problem.Nt),
        sigma=config.problem.sigma,
        coefCT=config.problem.coefCT,
    )

    logger.info("Created MFG problem from OmegaConf configuration")
    logger.info(f"Configuration: {config_manager.to_yaml()}")

    return problem, config


def main():
    """Run all 2D geometry MFG demonstrations."""
    logger.info(" Starting 2D MFG geometry demonstrations")

    try:
        # Demo 1: Rectangle with holes
        problem1 = create_rectangle_with_holes_problem()
        mesh_info1 = problem1.get_2d_mesh_info()
        logger.info(f"Rectangle with holes: {mesh_info1['num_vertices']} vertices")

        # Demo 2: L-shaped domain
        problem2 = create_l_shaped_domain_problem()
        mesh_info2 = problem2.get_2d_mesh_info()
        logger.info(f"L-shaped domain: {mesh_info2['num_vertices']} vertices")

        # Demo 3: Mesh convergence analysis
        analyze_mesh_convergence()

        # Demo 4: Advanced boundary conditions
        demonstrate_advanced_boundary_conditions()

        # Demo 5: OmegaConf integration
        _problem_config, _config = integration_with_existing_config()

        # Export meshes for external use
        problem1.export_mesh("./output/rectangle_holes")
        problem2.export_mesh("./output/l_shaped")

        logger.info("SUCCESS: All 2D MFG geometry demonstrations completed!")

        # Visualization (uncomment to show interactive plots)
        # problem1.visualize_geometry()
        # problem2.visualize_geometry()
        # advanced_bc.visualize_boundary_conditions()

    except ImportError as e:
        logger.error(f"ERROR: Missing dependency: {e}")
        logger.info("To run this demo, install: pip install gmsh meshio pyvista")

    except Exception as e:
        logger.error(f"ERROR: Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
