"""
High-Dimensional MFG Problem Framework

This module provides base classes and infrastructure for solving Mean Field Games
in 2D, 3D, and higher dimensions using particle-collocation methods and
advanced geometry handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .mfg_problem import MFGComponents, MFGProblem

if TYPE_CHECKING:
    from mfg_pde.geometry.base_geometry import BaseGeometry


class HighDimMFGProblem(ABC):
    """
    Abstract base class for high-dimensional MFG problems.

    This class provides the foundation for 2D, 3D, and nD MFG problems using
    modern computational approaches including particle-collocation methods,
    adaptive mesh refinement, and complex geometry handling.
    """

    def __init__(
        self,
        geometry: BaseGeometry,
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float = 0.1,
        dimension: int | None = None,
    ):
        """
        Initialize high-dimensional MFG problem.

        Args:
            geometry: Geometric domain (Domain2D, Domain3D, etc.)
            time_domain: (T_final, N_timesteps)
            diffusion_coeff: Diffusion coefficient σ
            dimension: Spatial dimension (inferred from geometry if not provided)
        """
        self.geometry = geometry
        self.T, self.Nt = time_domain
        self.sigma = diffusion_coeff
        self.dimension = dimension or geometry.dimension

        # Generate mesh
        self.mesh_data = self.geometry.generate_mesh()
        self.num_spatial_points = self.mesh_data.num_vertices

        # Time discretization
        self.dt = self.T / self.Nt
        self.time_grid = np.linspace(0, self.T, self.Nt + 1)

        # Collocation points for solvers (vertices of mesh by default)
        self.collocation_points = self.mesh_data.vertices

        # MFG components (to be defined by subclasses)
        self.components = None

    @abstractmethod
    def setup_components(self) -> MFGComponents:
        """Setup MFG components for the specific problem."""

    @abstractmethod
    def initial_density(self, x: np.ndarray) -> np.ndarray:
        """Define initial density distribution m₀(x)."""

    @abstractmethod
    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        """Define terminal cost function g(x)."""

    @abstractmethod
    def running_cost(self, x: np.ndarray, t: float) -> np.ndarray:
        """Define running cost function f(x,t)."""

    @abstractmethod
    def hamiltonian(self, x: np.ndarray, m: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
        """Define Hamiltonian H(x, m, p, t)."""

    def create_1d_adapter_problem(self) -> MFGProblem:
        """
        Create a 1D adapter MFG problem for use with existing solvers.

        This method maps the high-dimensional problem to a 1D representation
        that can be used with the existing MFG_PDE solver infrastructure.
        """
        if self.components is None:
            self.components = self.setup_components()  # type: ignore[assignment]

        # Create 1D problem using linearized indexing of spatial points
        adapter_problem = MFGProblem(
            xmin=0.0,
            xmax=float(self.num_spatial_points - 1),
            Nx=self.num_spatial_points - 1,
            T=self.T,
            Nt=self.Nt,
            sigma=self.sigma,
            components=self.components,
        )

        # Store reference to original high-dimensional problem
        adapter_problem._highdim_problem = self  # type: ignore[attr-defined]

        return adapter_problem

    def solve_with_damped_fixed_point(
        self,
        damping_factor: float = 0.5,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        solver_config: dict | None = None,
    ) -> dict:
        """
        Solve using damped fixed point iteration optimized for high dimensions.
        """
        from mfg_pde.config import create_fast_config
        from mfg_pde.factory import create_fast_solver

        # Create 1D adapter problem
        adapter_problem = self.create_1d_adapter_problem()

        # Setup solver configuration
        config = create_fast_config()
        config.newton.max_iterations = max_iterations
        config.newton.tolerance = tolerance
        config.picard.max_iterations = max_iterations // 2
        config.convergence_tolerance = tolerance

        if solver_config:
            for key, value in solver_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create damped fixed point solver
        solver = create_fast_solver(problem=adapter_problem, solver_type="fixed_point")

        # Solve the system
        solution = solver.solve()

        # Convert solution back to high-dimensional format
        if solution is not None:
            highdim_solution = self._convert_solution_to_highdim(solution)
            return {
                "success": True,
                "solution": highdim_solution,
                "solver_info": {
                    "method": "damped_fixed_point",
                    "dimension": self.dimension,
                    "num_spatial_points": self.num_spatial_points,
                    "time_steps": self.Nt,
                    "damping_factor": damping_factor,
                },
            }
        else:
            return {
                "success": False,
                "error": "Solver failed to converge",
                "solver_info": {"method": "damped_fixed_point", "dimension": self.dimension},
            }

    # Note: solve_with_particle_collocation method removed
    # Particle-collocation methods have been removed from core package

    def _convert_solution_to_highdim(self, solution_1d: Any) -> dict:
        """Convert 1D solver solution back to high-dimensional arrays."""
        # Extract solution components
        if hasattr(solution_1d, "m_history") and solution_1d.m_history is not None:
            m_history = solution_1d.m_history
            u_history = getattr(solution_1d, "u_history", None)
        else:
            m_history = getattr(solution_1d, "M", None)
            u_history = getattr(solution_1d, "U", None)

        result = {}

        if m_history is not None:
            result["density_history"] = self._reshape_to_spatial_grid(m_history)
            result["final_density"] = result["density_history"][-1]

        if u_history is not None:
            result["value_history"] = self._reshape_to_spatial_grid(u_history)
            result["final_value"] = result["value_history"][-1]

        # Add spatial coordinates for visualization
        result["coordinates"] = self.mesh_data.vertices
        result["mesh_data"] = self.mesh_data  # type: ignore[assignment]
        result["time_grid"] = self.time_grid

        return result

    def _reshape_to_spatial_grid(self, field_1d: np.ndarray) -> np.ndarray:
        """Reshape 1D field array to match spatial point structure."""
        if len(field_1d.shape) == 1:
            # Single time step
            return field_1d[: self.num_spatial_points]
        else:
            # Multiple time steps
            return field_1d[:, : self.num_spatial_points]

    def visualize_solution(
        self, solution: dict, time_index: int = -1, field_type: str = "density", **kwargs: Any
    ) -> None:
        """Visualize solution at specified time index."""
        if not solution.get("success", False):
            print("No solution to visualize")
            return

        if self.dimension == 2:
            self._visualize_2d(solution, time_index, field_type, **kwargs)
        elif self.dimension == 3:
            self._visualize_3d(solution, time_index, field_type, **kwargs)
        else:
            print(f"Visualization not implemented for {self.dimension}D")

    def _visualize_2d(self, solution: dict, time_index: int, field_type: str, **kwargs: Any) -> None:
        """2D visualization using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.tri as tri
        except ImportError:
            print("matplotlib required for 2D visualization")
            return

        # Get field data
        if field_type == "density" and "density_history" in solution:
            field_data = solution["density_history"][time_index]
        elif field_type == "value" and "value_history" in solution:
            field_data = solution["value_history"][time_index]
        else:
            print(f"Field type '{field_type}' not available")
            return

        # Get coordinates
        coords = solution["coordinates"]

        # Create triangulation for visualization
        if self.mesh_data.element_type == "triangle":
            triangles = self.mesh_data.elements
        else:
            # Create Delaunay triangulation
            from scipy.spatial import Delaunay

            triangulation = Delaunay(coords)
            triangles = triangulation.simplices

        # Plot
        _fig, ax = plt.subplots(figsize=(10, 8))

        # Contour plot
        triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        contour = ax.tricontourf(triang, field_data, levels=20, cmap="viridis")

        plt.colorbar(contour, ax=ax, label=field_type.capitalize())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{field_type.capitalize()} at t = {self.time_grid[time_index]:.3f}")
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

    def _visualize_3d(self, solution: dict, time_index: int, field_type: str, **kwargs: Any) -> None:
        """3D visualization using PyVista."""
        try:
            import pyvista as pv
        except ImportError:
            print("pyvista required for 3D visualization")
            return

        # Get field data
        if field_type == "density" and "density_history" in solution:
            field_data = solution["density_history"][time_index]
        elif field_type == "value" and "value_history" in solution:
            field_data = solution["value_history"][time_index]
        else:
            print(f"Field type '{field_type}' not available")
            return

        # Create PyVista mesh
        coords = solution["coordinates"]
        elements = self.mesh_data.elements

        if self.mesh_data.element_type == "tetrahedron":
            # Create unstructured grid
            cells = np.hstack(
                [
                    np.full((len(elements), 1), 4),  # Tetrahedron cell type
                    elements,
                ]
            )
            mesh = pv.UnstructuredGrid(cells, np.full(len(elements), 10), coords)
        else:
            # Create point cloud
            mesh = pv.PolyData(coords)  # type: ignore[assignment]

        # Add field data
        mesh[field_type] = field_data  # type: ignore[index]

        # Plot
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars=field_type, cmap="viridis", show_edges=False)
        plotter.add_axes()
        plotter.show_grid()
        plotter.add_text(f"{field_type.capitalize()} at t = {self.time_grid[time_index]:.3f}", position="upper_left")
        plotter.show()


class GridBasedMFGProblem(HighDimMFGProblem):
    """
    High-dimensional MFG problem on regular grids.

    This class provides a simpler interface for problems on rectangular domains
    with regular grid discretization, while still supporting the full
    high-dimensional solver infrastructure.
    """

    def __init__(
        self,
        domain_bounds: tuple,  # e.g., (0, 1, 0, 1) for 2D, (0, 1, 0, 1, 0, 1) for 3D
        grid_resolution: int | tuple[int, ...],
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float = 0.1,
    ):
        """
        Initialize grid-based MFG problem.

        Args:
            domain_bounds: Domain boundaries (2*dim values: min/max for each dimension)
            grid_resolution: Grid points per dimension (int for uniform, tuple for per-dimension)
            time_domain: (T_final, N_timesteps)
            diffusion_coeff: Diffusion coefficient
        """
        # Determine dimension from bounds
        dimension = len(domain_bounds) // 2

        # Create appropriate geometry (prefer simple grid for regular domains)
        try:
            if dimension == 2:
                from mfg_pde.geometry.simple_grid import SimpleGrid2D

                if isinstance(grid_resolution, int):
                    res = (grid_resolution, grid_resolution)
                else:
                    if len(grid_resolution) != 2:
                        raise ValueError(
                            f"For 2D problems, grid_resolution must be int or 2-tuple, got {len(grid_resolution)}-tuple"
                        )
                    res = tuple(grid_resolution[:2])  # type: ignore[assignment]  # Ensure exactly 2 elements
                geometry = SimpleGrid2D(bounds=domain_bounds, resolution=res)
            elif dimension == 3:
                from mfg_pde.geometry.simple_grid import SimpleGrid3D

                if isinstance(grid_resolution, int):
                    res = (grid_resolution, grid_resolution, grid_resolution)  # type: ignore[assignment]
                else:
                    if len(grid_resolution) != 3:
                        raise ValueError(
                            f"For 3D problems, grid_resolution must be int or 3-tuple, got {len(grid_resolution)}-tuple"
                        )
                    res = tuple(grid_resolution[:3])  # type: ignore[assignment]  # Ensure exactly 3 elements
                geometry = SimpleGrid3D(bounds=domain_bounds, resolution=res)  # type: ignore[assignment,arg-type]
            else:
                raise ValueError(f"Grid-based problems only support 2D and 3D, got {dimension}D")

        except ImportError:
            # Fallback to Domain2D/Domain3D if simple grid not available
            if dimension == 2:
                from mfg_pde.geometry.domain_2d import Domain2D

                geometry = Domain2D(  # type: ignore[assignment]
                    domain_type="rectangle",
                    bounds=domain_bounds,
                    mesh_size=1.0 / (grid_resolution if isinstance(grid_resolution, int) else min(grid_resolution)),
                )
            elif dimension == 3:
                # Domain3D is not fully implemented yet - abstract class cannot be instantiated
                raise NotImplementedError(
                    "3D grid-based problems are not yet supported. Domain3D is an abstract class that requires concrete implementation."
                ) from None
            else:
                raise ValueError(f"Grid-based problems only support 2D and 3D, got {dimension}D") from None

        super().__init__(geometry, time_domain, diffusion_coeff, dimension)

        self.domain_bounds = domain_bounds
        self.grid_resolution = grid_resolution


class HybridMFGSolver:
    """
    Hybrid solver that combines multiple approaches for robust high-dimensional MFG solving.

    This solver implements a multi-strategy approach:
    1. Start with damped fixed point for rapid initial convergence
    2. Switch to particle collocation for higher accuracy
    3. Apply adaptive refinement in problematic regions
    """

    def __init__(self, problem: HighDimMFGProblem):
        self.problem = problem
        self.solution_history: list[dict] = []

    def solve(self, max_total_iterations: int = 100, tolerance: float = 1e-6, strategy: str = "adaptive") -> dict:
        """
        Solve using hybrid multi-strategy approach.

        Args:
            max_total_iterations: Total iteration budget
            tolerance: Target accuracy
            strategy: adaptive, "sequential", or "parallel"
        """

        if strategy == "adaptive":
            return self._solve_adaptive(max_total_iterations, tolerance)
        elif strategy == "sequential":
            return self._solve_sequential(max_total_iterations, tolerance)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _solve_adaptive(self, max_iterations: int, tolerance: float) -> dict:
        """Adaptive strategy: switch methods based on convergence behavior.

        Note: Simplified to use only damped fixed point method.
        Particle-collocation methods have been removed from core package.
        """

        # Phase 1: Quick damped fixed point for initialization
        print("Phase 1: Damped fixed point initialization...")
        result1 = self.problem.solve_with_damped_fixed_point(
            damping_factor=0.7,  # Higher damping for stability
            max_iterations=max_iterations // 2,
            tolerance=tolerance * 5,  # Relaxed tolerance for initialization
        )

        if not result1["success"]:
            return result1

        # Phase 2: Refinement with tighter tolerance
        print("Phase 2: Refined fixed point...")
        final_result = self.problem.solve_with_damped_fixed_point(
            damping_factor=0.3,  # Lower damping for accuracy
            max_iterations=max_iterations // 2,
            tolerance=tolerance,
        )

        final_result["solver_info"]["method"] = "adaptive"
        final_result["solver_info"]["phases"] = ["damped_fixed_point_init", "damped_fixed_point_refined"]

        return final_result

    def _solve_sequential(self, max_iterations: int, tolerance: float) -> dict:
        """Sequential strategy: run damped fixed point method.

        Note: Simplified to use only damped fixed point method.
        Particle-collocation methods have been removed from core package.
        """

        # Use damped fixed point method
        result = self.problem.solve_with_damped_fixed_point(max_iterations=max_iterations, tolerance=tolerance)

        if result["success"]:
            result["solver_info"]["method"] = "sequential"
            return result
        else:
            return {"success": False, "error": "Damped fixed point failed"}
