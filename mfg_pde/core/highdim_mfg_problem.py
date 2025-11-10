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


class _TensorGridGeometry:
    """
    Dimension-agnostic geometry wrapper for TensorProductGrid.

    This internal helper class wraps TensorProductGrid to provide a
    BaseGeometry-compatible interface for GridBasedMFGProblem.
    """

    def __init__(self, grid):
        """Initialize with TensorProductGrid."""
        from mfg_pde.geometry.tensor_product_grid import TensorProductGrid

        if not isinstance(grid, TensorProductGrid):
            raise TypeError(f"Expected TensorProductGrid, got {type(grid)}")

        self.grid = grid
        self.dimension = grid.dimension

    def generate_mesh(self):
        """Generate MeshData from grid points."""
        from mfg_pde.geometry.base_geometry import MeshData

        # Get all grid points as (N, d) array
        points = self.grid.flatten()

        # Create MeshData with empty elements (structured grid doesn't need element connectivity)
        return MeshData(
            vertices=points,
            elements=np.array([], dtype=np.int32).reshape(0, 0),
            element_type="point",  # Point cloud representation
            boundary_tags=np.array([], dtype=np.int32),
            element_tags=np.array([], dtype=np.int32),
            boundary_faces=np.array([], dtype=np.int32).reshape(0, 0),
            dimension=self.dimension,
        )


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


class GridBasedMFGProblem(MFGProblem):
    """
    Legacy base class for grid-based MFG problems (backward compatible).

    DEPRECATED: Use MFGProblem directly instead of inheriting from this class.

    This class exists only to support existing code that inherits from
    GridBasedMFGProblem. It will be removed in v2.0.0.

    Deprecation Timeline:
    - v0.9.0 (2025-Q1): Added deprecation warning
    - v1.0.0 (2025-Q2): Warning becomes more prominent
    - v2.0.0 (2026-Q1): Class removed entirely

    Migration:
        # Old (deprecated):
        class MyProblem(GridBasedMFGProblem):
            def __init__(self, ...):
                super().__init__(domain_bounds=(...), ...)

        # New (recommended):
        class MyProblem(MFGProblem):
            def __init__(self, ...):
                super().__init__(spatial_bounds=[...], ...)
    """

    def __init__(
        self,
        domain_bounds: tuple,
        grid_resolution: int | tuple[int, ...],
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float = 0.1,
    ):
        import warnings

        warnings.warn(
            "GridBasedMFGProblem class is deprecated and will be removed in v2.0.0. "
            "Use MFGProblem() directly. See docs/migration/unified_problem_migration.md",
            DeprecationWarning,
            stacklevel=2,
        )

        # Determine dimension
        dimension = len(domain_bounds) // 2
        spatial_bounds = [(domain_bounds[2 * i], domain_bounds[2 * i + 1]) for i in range(dimension)]

        # Convert grid_resolution
        if isinstance(grid_resolution, int):
            spatial_discretization = [grid_resolution] * dimension
        else:
            spatial_discretization = list(grid_resolution)

        # Extract time parameters
        T, Nt = time_domain

        # Initialize parent MFGProblem
        super().__init__(
            spatial_bounds=spatial_bounds,
            spatial_discretization=spatial_discretization,
            T=T,
            Nt=Nt,
            sigma=diffusion_coeff,
        )

        # Store legacy attributes for backward compatibility
        self.domain_bounds = domain_bounds
        self.grid_resolution = grid_resolution

        # Create geometry wrapper for compatibility with dimension-agnostic solvers
        # The HJBSemiLagrangianSolver expects problem.geometry.grid
        if dimension > 1:
            self.geometry = _TensorGridGeometry(self._grid)

            # Remove 1D attributes for nD problems to ensure solvers use geometry path
            # FixedPointIterator checks hasattr(problem, "Nx"), which returns True even when Nx=None
            del self.Nx, self.xmin, self.xmax, self.dx, self.Lx, self.xSpace


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
