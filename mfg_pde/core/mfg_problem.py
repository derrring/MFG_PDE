#!/usr/bin/env python3
"""
Dimension-agnostic MFG problem implementation.

This module provides a single MFGProblem class that works for 1D, 2D,
3D, and nD problems with a consistent API.

This replaces:
    - Old MFGProblem (1D-only)
    - GridBasedMFGProblem (deprecated 2D+)
    - ExampleMFGProblem (legacy)

Mathematical Notation:
    - m(t,x): Density function
    - u(t,x): Value function
    - H(x, m, p, t): Hamiltonian
    - g(x): Terminal cost
    - m₀(x): Initial density

Created: 2025-11-05
Part of: Issue #245 - Radical Architecture Renovation
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import numpy as np

from mfg_pde.utils.logging import get_logger

from .base_problem import BaseMFGProblem

logger = get_logger(__name__)


class MFGProblem(BaseMFGProblem):
    """
    MFG problem class for 1D, 2D, 3D, nD Cartesian grids.

    This single class provides dimension-agnostic functionality with a
    consistent API regardless of spatial dimension.

    Mathematical Components:
        - Hamiltonian: H(x, m, p, t)
        - Terminal cost: g(x)
        - Initial density: m₀(x)
        - Running cost: f(x, m, t)

    Parameters:
        spatial_bounds: [(x₀_min, x₀_max), (x₁_min, x₁_max), ...]
            Domain bounds per dimension
        spatial_discretization: [N₀, N₁, ...]
            Grid points per dimension
        time_domain: (T_final, Nt)
            Final time and number of time steps
        diffusion_coeff: σ
            Diffusion coefficient (noise intensity)
        hamiltonian_func: Optional custom Hamiltonian H(x, m, p, t)
        terminal_cost_func: Optional custom terminal cost g(x)
        initial_density_func: Optional custom initial density m₀(x)
        running_cost_func: Optional custom running cost f(x, m, t)

    Examples:
        >>> # 1D problem
        >>> problem_1d = MFGProblem(
        ...     spatial_bounds=[(0.0, 1.0)],
        ...     spatial_discretization=[100],
        ...     time_domain=(1.0, 50),
        ...     diffusion_coeff=0.1
        ... )
        >>> print(f"Dimension: {problem_1d.dimension}")  # 1
        >>> print(f"Grid shape: {problem_1d.grid_shape}")  # (100,)

        >>> # 2D problem - same API!
        >>> problem_2d = MFGProblem(
        ...     spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
        ...     spatial_discretization=[50, 50],
        ...     time_domain=(1.0, 100),
        ...     diffusion_coeff=0.1
        ... )
        >>> print(f"Dimension: {problem_2d.dimension}")  # 2
        >>> print(f"Grid shape: {problem_2d.grid_shape}")  # (50, 50)

        >>> # 3D problem - same API!
        >>> problem_3d = MFGProblem(
        ...     spatial_bounds=[(0, 1), (0, 1), (0, 1)],
        ...     spatial_discretization=[20, 20, 20],
        ...     time_domain=(1.0, 50),
        ...     diffusion_coeff=0.1
        ... )
        >>> print(f"Dimension: {problem_3d.dimension}")  # 3

        >>> # Custom components
        >>> def custom_hamiltonian(x, m, p, t):
        ...     # Quadratic with congestion
        ...     p_arr = np.array(p) if hasattr(p, '__iter__') else p
        ...     return 0.5 * np.sum(p_arr**2) + 0.1 * m
        >>>
        >>> problem_custom = MFGProblem(
        ...     spatial_bounds=[(0.0, 1.0)],
        ...     spatial_discretization=[100],
        ...     time_domain=(1.0, 50),
        ...     diffusion_coeff=0.1,
        ...     hamiltonian_func=custom_hamiltonian
        ... )
    """

    def __init__(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int],
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float | Callable = 0.1,
        hamiltonian_func: Callable | None = None,
        terminal_cost_func: Callable | None = None,
        initial_density_func: Callable | None = None,
        running_cost_func: Callable | None = None,
    ):
        """
        Initialize unified MFG problem.

        Args:
            spatial_bounds: [(x₀_min, x₀_max), ...] for each dimension
            spatial_discretization: [N₀, ...] grid points per dimension
            time_domain: (T_final, Nt)
            diffusion_coeff: Diffusion coefficient σ (float or callable σ(x))
            hamiltonian_func: Custom Hamiltonian H(x, m, p, t) or None for default
            terminal_cost_func: Custom terminal cost g(x) or None for default
            initial_density_func: Custom initial density m₀(x) or None for default
            running_cost_func: Custom running cost f(x, m, t) or None for default
        """
        # Initialize base class
        super().__init__(spatial_bounds, spatial_discretization, time_domain, diffusion_coeff)

        # Store custom component functions
        self._hamiltonian_func = hamiltonian_func
        self._terminal_cost_func = terminal_cost_func
        self._initial_density_func = initial_density_func
        self._running_cost_func = running_cost_func

        # Warn about default functions
        if hamiltonian_func is None:
            logger.warning("Using default Hamiltonian H = 0.5·|p|² (LQ-type with no interaction)")
        if terminal_cost_func is None:
            logger.warning("Using default terminal cost g(x) = 0.5·|x - x_center|² (quadratic)")
        if initial_density_func is None:
            logger.warning("Using default initial density m₀(x) = Gaussian centered at domain center")
        if running_cost_func is None:
            logger.info("Using default running cost f = 0 (no running cost)")

        # Build spatial grid
        self._build_spatial_grid()

        # Setup initial/terminal conditions
        self._setup_conditions()

    def _build_spatial_grid(self) -> None:
        """
        Build nD tensor product Cartesian grid.

        Creates coordinate arrays and computes grid spacing for arbitrary dimensions.

        For 1D: spatial_grid is a 1D array [x₀, x₁, ..., x_N]
        For nD: spatial_grid is a list of nD meshgrid arrays [X₀, X₁, ..., X_{n-1}]

        Grid spacing: Δx_i = (x_i_max - x_i_min) / (N_i - 1)
        """
        # Create 1D grids for each dimension
        grids_1d = [
            np.linspace(bounds[0], bounds[1], n_points)
            for bounds, n_points in zip(self.spatial_bounds, self.spatial_discretization, strict=False)
        ]

        # Compute grid spacing per dimension
        self.grid_spacing = [
            (bounds[1] - bounds[0]) / (n_points - 1) if n_points > 1 else 0.0
            for bounds, n_points in zip(self.spatial_bounds, self.spatial_discretization, strict=False)
        ]

        # Create spatial grid
        if self.dimension == 1:
            # For 1D, just use the 1D array directly
            self.spatial_grid = grids_1d[0]
        else:
            # For nD, create meshgrid with indexing='ij' (matrix indexing)
            # This matches the standard mathematical convention for multi-dimensional arrays
            self.spatial_grid = np.meshgrid(*grids_1d, indexing="ij")

        # Compute total number of spatial points
        self.num_spatial_points = int(np.prod(self.spatial_discretization))

    def _setup_conditions(self) -> None:
        """
        Setup initial density m₀(x) and terminal cost g(x) arrays.

        Evaluates the initial density and terminal cost functions on the grid
        and stores them as arrays for efficient access by solvers.

        Initial density is automatically normalized: ∫ m₀(x) dx = 1
        """
        # Evaluate initial density m₀(x)
        if self.dimension == 1:
            # 1D: Direct vectorized evaluation
            x_vals = self.spatial_grid
            self.m_init = np.array([self.initial_density(x) for x in x_vals])
        else:
            # nD: Evaluate at each grid point
            points_shape = self.grid_shape
            self.m_init = np.zeros(points_shape)

            # TODO: Vectorize this for performance (future optimization)
            for idx in np.ndindex(points_shape):
                x_point = tuple(grid[idx] for grid in self.spatial_grid)
                self.m_init[idx] = self.initial_density(x_point)

        # Normalize initial density: ∫ m₀(x) dx = 1
        dx_volume = np.prod(self.grid_spacing)
        total_mass = np.sum(self.m_init) * dx_volume
        if total_mass > 0:
            self.m_init = self.m_init / total_mass
        else:
            # Fallback to uniform if all zeros
            self.m_init = np.ones_like(self.m_init) / self.num_spatial_points

        # Evaluate terminal cost g(x)
        if self.dimension == 1:
            # 1D: Direct vectorized evaluation
            self.u_terminal = np.array([self.terminal_cost(x) for x in self.spatial_grid])
        else:
            # nD: Evaluate at each grid point
            self.u_terminal = np.zeros(self.grid_shape)
            for idx in np.ndindex(self.grid_shape):
                x_point = tuple(grid[idx] for grid in self.spatial_grid)
                self.u_terminal[idx] = self.terminal_cost(x_point)

    # ========================================================================
    # MFG Component Methods
    # ========================================================================

    def hamiltonian(self, x, m, p, t) -> float:
        """
        Hamiltonian H(x, m, p, t).

        Default: H = 0.5·|p|² (LQ-type with no interaction)

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)
            m: Density value m(t,x)
            p: Momentum/co-state ∂u/∂x (scalar for 1D, tuple/array for nD)
            t: Time

        Returns:
            Hamiltonian value H(x, m, p, t)
        """
        if self._hamiltonian_func is not None:
            return self._hamiltonian_func(x, m, p, t)

        # Default LQ Hamiltonian: H = 0.5·|p|²
        if isinstance(p, (list, tuple, np.ndarray)):
            p_arr = np.array(p)
            return 0.5 * np.sum(p_arr**2)
        return 0.5 * p**2

    def terminal_cost(self, x) -> float:
        """
        Terminal cost g(x).

        Default: g(x) = 0.5·|x - x_center|² where x_center is domain center

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)

        Returns:
            Terminal cost value g(x)
        """
        if self._terminal_cost_func is not None:
            return self._terminal_cost_func(x)

        # Default: Quadratic cost relative to domain center
        # Center point: (x_min + x_max) / 2 for each dimension
        if isinstance(x, (list, tuple, np.ndarray)):
            x_arr = np.array(x)
            center = np.array([(bounds[0] + bounds[1]) / 2.0 for bounds in self.spatial_bounds])
            return 0.5 * np.sum((x_arr - center) ** 2)
        else:
            # 1D scalar case
            center = (self.spatial_bounds[0][0] + self.spatial_bounds[0][1]) / 2.0
            return 0.5 * (x - center) ** 2

    def initial_density(self, x) -> float:
        """
        Initial density m₀(x).

        Default: Gaussian centered at domain center

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)

        Returns:
            Initial density value m₀(x) (unnormalized)
        """
        if self._initial_density_func is not None:
            return self._initial_density_func(x)

        # Default: Gaussian centered at domain center
        # σ_gaussian = 0.1 * domain_width (narrow Gaussian)
        if isinstance(x, (list, tuple, np.ndarray)):
            x_arr = np.array(x)
            center = np.array([(bounds[0] + bounds[1]) / 2.0 for bounds in self.spatial_bounds])
            widths = np.array([bounds[1] - bounds[0] for bounds in self.spatial_bounds])
            # Normalize by domain width for dimension-independent behavior
            normalized_dist = np.sum(((x_arr - center) / widths) ** 2)
            return np.exp(-5.0 * normalized_dist)
        else:
            # 1D scalar case
            center = (self.spatial_bounds[0][0] + self.spatial_bounds[0][1]) / 2.0
            width = self.spatial_bounds[0][1] - self.spatial_bounds[0][0]
            normalized_dist = ((x - center) / width) ** 2
            return np.exp(-5.0 * normalized_dist)

    def running_cost(self, x, m, t) -> float:
        """
        Running cost f(x, m, t).

        Default: f = 0 (no running cost)

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)
            m: Density value m(t,x)
            t: Time

        Returns:
            Running cost value f(x, m, t)
        """
        if self._running_cost_func is not None:
            return self._running_cost_func(x, m, t)

        # Default: zero running cost
        return 0.0

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_spatial_point(self, indices: tuple[int, ...]) -> tuple | float:
        """
        Get spatial coordinates at given grid indices.

        Args:
            indices: Tuple of grid indices (i₀, i₁, ...)

        Returns:
            Spatial coordinates (x₀, x₁, ...) or scalar x for 1D

        Examples:
            >>> problem_2d = MFGProblem(
            ...     spatial_bounds=[(0, 1), (0, 1)],
            ...     spatial_discretization=[11, 11],
            ...     time_domain=(1.0, 10),
            ...     diffusion_coeff=0.1
            ... )
            >>> x, y = problem_2d.get_spatial_point((5, 5))
            >>> print(f"Center point: ({x:.2f}, {y:.2f})")  # (0.50, 0.50)
        """
        if self.dimension == 1:
            return self.spatial_grid[indices[0]]
        else:
            return tuple(grid[indices] for grid in self.spatial_grid)

    def compute_grid_volume(self) -> float:
        """
        Compute volume element dx = Δx₀ · Δx₁ · ... · Δx_{n-1}.

        Returns:
            Volume element for integration

        Examples:
            >>> problem = MFGProblem(
            ...     spatial_bounds=[(0, 1), (0, 2)],
            ...     spatial_discretization=[10, 20],
            ...     time_domain=(1.0, 10),
            ...     diffusion_coeff=0.1
            ... )
            >>> dx = problem.compute_grid_volume()
            >>> # Can be used for integration: ∫ f(x) dx ≈ Σ f(x_i) · dx
        """
        return float(np.prod(self.grid_spacing))

    def is_inside_domain(self, x) -> bool:
        """
        Check if point x is inside the domain.

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)

        Returns:
            True if x is inside domain bounds

        Examples:
            >>> problem = MFGProblem(
            ...     spatial_bounds=[(0, 1)],
            ...     spatial_discretization=[100],
            ...     time_domain=(1.0, 10),
            ...     diffusion_coeff=0.1
            ... )
            >>> problem.is_inside_domain(0.5)  # True
            >>> problem.is_inside_domain(1.5)  # False
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            x_arr = np.array(x)
            return all(
                bounds[0] <= x_val <= bounds[1] for x_val, bounds in zip(x_arr, self.spatial_bounds, strict=False)
            )
        else:
            # 1D scalar case
            return self.spatial_bounds[0][0] <= x <= self.spatial_bounds[0][1]

    def __repr__(self) -> str:
        """String representation with dimension and grid info."""
        return (
            f"MFGProblem("
            f"dimension={self.dimension}, "
            f"grid_shape={self.grid_shape}, "
            f"T={self.T}, Nt={self.Nt}, "
            f"σ={self.sigma})"
        )
