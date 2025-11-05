#!/usr/bin/env python3
"""
Base problem infrastructure for dimension-agnostic MFG problems.

This module defines the core protocol and abstract base class that ALL
MFG problems must implement, regardless of dimension (1D, 2D, 3D, nD).

Mathematical Notation:
    - m(t,x): Density function
    - u(t,x): Value function
    - ∂u/∂x: Spatial gradient
    - H(x, m, p, t): Hamiltonian
    - g(x): Terminal cost
    - f(x, m, t): Running cost

Created: 2025-11-05
Part of: Issue #245 - Radical Architecture Renovation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class MFGProblemProtocol(Protocol):
    """
    Protocol that ALL MFG problems must satisfy.

    This defines the minimal interface for dimension-agnostic MFG solvers.
    Any class implementing these attributes and methods can be used with
    our solver infrastructure.

    Spatial Notation:
        - dimension: Number of spatial dimensions
        - spatial_bounds: [(x₀_min, x₀_max), (x₁_min, x₁_max), ...]
        - spatial_discretization: [N₀, N₁, ...] grid points per dimension
        - spatial_grid: Coordinate arrays (1D array for 1D, list of nD arrays for nD)
        - grid_shape: (N₀, N₁, ...) shape of spatial grid
        - grid_spacing: [Δx₀, Δx₁, ...] spacing per dimension

    Temporal Notation:
        - T: Final time
        - Nt: Number of time steps
        - dt: Time step size Δt = T/Nt
        - time_grid: Array of time points [t₀, t₁, ..., t_Nt]

    Physical Parameters:
        - sigma: Diffusion coefficient σ (constant or callable for matrix diffusion)

    MFG Components:
        - hamiltonian(x, m, p, t): H(x, m, p, t)
        - terminal_cost(x): g(x)
        - initial_density(x): m₀(x)
        - running_cost(x, m, t): f(x, m, t)
    """

    # Spatial properties
    dimension: int
    spatial_bounds: list[tuple[float, float]]
    spatial_discretization: list[int]
    spatial_grid: NDArray | list[NDArray]
    grid_shape: tuple[int, ...]
    grid_spacing: list[float]

    # Temporal properties
    T: float
    Nt: int
    dt: float
    time_grid: NDArray

    # Physical properties
    sigma: float | Callable  # Constant, position-dependent scalar, or matrix diffusion

    # MFG components
    def hamiltonian(self, x, m, p, t) -> float:
        """
        Hamiltonian H(x, m, p, t).

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)
            m: Density value m(t,x)
            p: Momentum/co-state (scalar for 1D, tuple/array for nD)
            t: Time

        Returns:
            Hamiltonian value H(x, m, p, t)
        """
        ...

    def terminal_cost(self, x) -> float:
        """
        Terminal cost g(x).

        Args:
            x: Spatial position

        Returns:
            Terminal cost value g(x)
        """
        ...

    def initial_density(self, x) -> float:
        """
        Initial density m₀(x).

        Args:
            x: Spatial position

        Returns:
            Initial density value m₀(x)
        """
        ...

    def running_cost(self, x, m, t) -> float:
        """
        Running cost f(x, m, t).

        Args:
            x: Spatial position
            m: Density value m(t,x)
            t: Time

        Returns:
            Running cost value f(x, m, t)
        """
        ...


class BaseMFGProblem(ABC):
    """
    Abstract base class for ALL MFG problems (1D, 2D, 3D, nD).

    This class defines the universal interface and common initialization
    that all MFG problems must implement, regardless of dimension or
    domain type (Cartesian grid, network, manifold, etc.).

    The design follows modern mathematical notation:
        - m(t,x) for density
        - u(t,x) for value function
        - ∂u/∂x for spatial derivatives
        - H(x, m, p, t) for Hamiltonian

    Parameters:
        spatial_bounds: [(x₀_min, x₀_max), (x₁_min, x₁_max), ...]
            Bounds for each spatial dimension
        spatial_discretization: [N₀, N₁, ...]
            Number of grid points per dimension
        time_domain: (T_final, Nt)
            Final time T and number of time steps
        diffusion_coeff: σ
            Diffusion coefficient (noise intensity)

    Attributes:
        dimension: Number of spatial dimensions
        spatial_bounds: Bounds per dimension
        spatial_discretization: Grid points per dimension
        grid_shape: Tuple of grid dimensions
        grid_spacing: List of spacing per dimension [Δx₀, Δx₁, ...]
        spatial_grid: Coordinate arrays
        T: Final time
        Nt: Number of time steps
        dt: Time step size Δt
        time_grid: Time points array
        sigma: Diffusion coefficient σ

    Example:
        >>> # 1D problem
        >>> problem_1d = ConcreteProblem(
        ...     spatial_bounds=[(0.0, 1.0)],
        ...     spatial_discretization=[100],
        ...     time_domain=(1.0, 50),
        ...     diffusion_coeff=0.1
        ... )
        >>> print(f"Dimension: {problem_1d.dimension}")  # 1
        >>> print(f"Grid spacing: {problem_1d.grid_spacing}")  # [0.01]

        >>> # 2D problem - same API!
        >>> problem_2d = ConcreteProblem(
        ...     spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
        ...     spatial_discretization=[50, 50],
        ...     time_domain=(1.0, 100),
        ...     diffusion_coeff=0.1
        ... )
        >>> print(f"Dimension: {problem_2d.dimension}")  # 2
        >>> print(f"Grid shape: {problem_2d.grid_shape}")  # (50, 50)
    """

    def __init__(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int],
        time_domain: tuple[float, int],
        diffusion_coeff: float | Callable,
    ):
        """
        Initialize dimension-agnostic MFG problem.

        Args:
            spatial_bounds: [(x₀_min, x₀_max), (x₁_min, x₁_max), ...]
                Bounds for each spatial dimension
            spatial_discretization: [N₀, N₁, ...]
                Number of grid points per dimension
            time_domain: (T_final, Nt)
                Final time and number of time steps
            diffusion_coeff: σ or σ(x) or D(x)
                Diffusion coefficient (noise intensity)
                - float: Constant scalar diffusion σ (isotropic, σI in matrix form)
                - Callable returning float: Position-dependent scalar σ(x) → float
                - Callable returning NDArray: Full matrix diffusion D(x) → ℝ^{d×d}
                  For anisotropic diffusion with direction-dependent noise

        Raises:
            ValueError: If dimensions are inconsistent or parameters invalid
        """
        # Validate inputs
        if len(spatial_bounds) != len(spatial_discretization):
            raise ValueError(
                f"Inconsistent dimensions: {len(spatial_bounds)} bounds "
                f"vs {len(spatial_discretization)} discretization values"
            )

        if any(n <= 0 for n in spatial_discretization):
            raise ValueError(f"All discretization values must be positive: {spatial_discretization}")

        if any(x_max <= x_min for x_min, x_max in spatial_bounds):
            raise ValueError(f"All bounds must satisfy x_max > x_min: {spatial_bounds}")

        if time_domain[0] <= 0 or time_domain[1] <= 0:
            raise ValueError(f"Time domain values must be positive: {time_domain}")

        # Validate diffusion coefficient
        if callable(diffusion_coeff):
            # For callable sigma, validation happens at runtime
            pass
        elif diffusion_coeff < 0:
            raise ValueError(f"Diffusion coefficient must be non-negative: {diffusion_coeff}")

        # Spatial setup
        self.spatial_bounds = spatial_bounds
        self.spatial_discretization = spatial_discretization
        self.dimension = len(spatial_bounds)

        # Temporal setup
        self.T, self.Nt = time_domain
        self.dt = self.T / self.Nt
        self.time_grid = np.linspace(0, self.T, self.Nt + 1)

        # Physical parameters
        self.sigma = diffusion_coeff

        # Grid properties (initialized by subclass)
        self.grid_shape: tuple[int, ...] = tuple(spatial_discretization)
        self.grid_spacing: list[float] = []
        self.spatial_grid: NDArray | list[NDArray] | None = None

        # Initial and terminal conditions (initialized by subclass)
        self.m_init: NDArray | None = None  # m₀(x)
        self.u_terminal: NDArray | None = None  # g(x)

    @abstractmethod
    def _build_spatial_grid(self) -> None:
        """
        Build dimension-specific spatial grid.

        This method must be implemented by concrete subclasses to construct
        the appropriate spatial discretization (Cartesian grid, network, etc.).

        The implementation should:
        1. Create coordinate arrays (self.spatial_grid)
        2. Compute grid spacing (self.grid_spacing)
        3. Store any additional grid metadata

        For Cartesian grids:
            - 1D: spatial_grid is a 1D array
            - nD: spatial_grid is a list of nD meshgrid arrays
        """

    @abstractmethod
    def hamiltonian(self, x, m, p, t) -> float:
        """
        Hamiltonian H(x, m, p, t).

        The Hamiltonian defines the optimal control problem and determines
        the dynamics through the HJB equation:
            ∂u/∂t + H(x, m, ∂u/∂x, t) = 0

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)
            m: Density value m(t,x)
            p: Momentum/co-state ∂u/∂x (scalar for 1D, tuple/array for nD)
            t: Time

        Returns:
            Hamiltonian value H(x, m, p, t)

        Examples:
            >>> # LQ Hamiltonian: H = 0.5·|p|²
            >>> def hamiltonian(self, x, m, p, t):
            ...     if isinstance(p, (list, tuple, np.ndarray)):
            ...         return 0.5 * np.sum(np.array(p)**2)
            ...     return 0.5 * p**2

            >>> # With congestion: H = 0.5·|p|² + α·m
            >>> def hamiltonian(self, x, m, p, t):
            ...     p_arr = np.array(p) if hasattr(p, '__iter__') else p
            ...     return 0.5 * np.sum(p_arr**2) + self.alpha * m
        """

    @abstractmethod
    def terminal_cost(self, x) -> float:
        """
        Terminal cost g(x).

        Defines the boundary condition for the HJB equation at final time:
            u(T, x) = g(x)

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)

        Returns:
            Terminal cost value g(x)

        Examples:
            >>> # Quadratic terminal cost: g(x) = 0.5·|x - x_target|²
            >>> def terminal_cost(self, x):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else x
            ...     x_target = np.array([0.5, 0.5])  # For 2D
            ...     return 0.5 * np.sum((x_arr - x_target)**2)

            >>> # Distance to boundary
            >>> def terminal_cost(self, x):
            ...     # Minimum distance to domain boundary
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else np.array([x])
            ...     dist = min(
            ...         min(x_arr[i] - bounds[i][0], bounds[i][1] - x_arr[i])
            ...         for i, bounds in enumerate(self.spatial_bounds)
            ...     )
            ...     return -dist  # Negative distance (reward for staying inside)
        """

    @abstractmethod
    def initial_density(self, x) -> float:
        """
        Initial density m₀(x).

        Defines the initial condition for the Fokker-Planck equation:
            m(0, x) = m₀(x)

        The implementation should return unnormalized values; normalization
        will be handled automatically.

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)

        Returns:
            Initial density value m₀(x) (unnormalized)

        Examples:
            >>> # Gaussian centered at origin
            >>> def initial_density(self, x):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else x
            ...     return np.exp(-5 * np.sum(x_arr**2))

            >>> # Uniform density
            >>> def initial_density(self, x):
            ...     return 1.0  # Will be normalized automatically

            >>> # Mixture of Gaussians
            >>> def initial_density(self, x):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else x
            ...     centers = [np.array([0.3, 0.3]), np.array([0.7, 0.7])]
            ...     return sum(np.exp(-10 * np.sum((x_arr - c)**2)) for c in centers)
        """

    def running_cost(self, x, m, t) -> float:
        """
        Running cost f(x, m, t).

        Optional cost integrated over the time horizon. Default is zero.
        The value function satisfies:
            u(t,x) = E[ ∫_t^T f(X_s, m_s, s)ds + g(X_T) ]

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)
            m: Density value m(t,x)
            t: Time

        Returns:
            Running cost value f(x, m, t)

        Examples:
            >>> # Quadratic state cost: f = 0.5·|x|²
            >>> def running_cost(self, x, m, t):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else x
            ...     return 0.5 * np.sum(x_arr**2)

            >>> # Congestion cost: f = α·m²
            >>> def running_cost(self, x, m, t):
            ...     return self.alpha * m**2
        """
        return 0.0

    def get_diffusion_matrix(self, x) -> NDArray:
        """
        Get diffusion matrix D(x) at position x.

        This method provides a unified interface for diffusion coefficients:
        - Constant scalar σ → σI (scalar × identity matrix)
        - Callable returning scalar σ(x) → σ(x)I
        - Callable returning matrix D(x) → D(x) directly

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)

        Returns:
            Diffusion matrix D(x) ∈ ℝ^{d×d}

        Raises:
            ValueError: If diffusion matrix is not symmetric positive-definite

        Examples:
            >>> # Constant scalar diffusion σ = 0.1
            >>> problem = MFGProblem(..., diffusion_coeff=0.1)
            >>> D = problem.get_diffusion_matrix(x)
            >>> # Returns 0.1 * I for any dimension

            >>> # Position-dependent scalar diffusion
            >>> def sigma_func(x):
            ...     return 0.1 + 0.01 * np.linalg.norm(x)**2
            >>> problem = MFGProblem(..., diffusion_coeff=sigma_func)
            >>> D = problem.get_diffusion_matrix(x)
            >>> # Returns σ(x) * I

            >>> # Anisotropic matrix diffusion
            >>> def D_func(x):
            ...     if x[0] < 0.5:  # Left half: easy horizontal movement
            ...         return np.array([[1.0, 0.0], [0.0, 0.1]])
            ...     else:  # Right half: easy vertical movement
            ...         return np.array([[0.1, 0.0], [0.0, 1.0]])
            >>> problem = MFGProblem(..., diffusion_coeff=D_func)
            >>> D = problem.get_diffusion_matrix(x)
            >>> # Returns position-dependent diffusion matrix
        """
        if callable(self.sigma):
            # Evaluate callable diffusion coefficient
            result = self.sigma(x)

            if isinstance(result, (int, float, np.number)):
                # Scalar diffusion: return σ(x) * I
                return result * np.eye(self.dimension)
            elif isinstance(result, np.ndarray):
                # Matrix diffusion: validate and return
                if result.shape != (self.dimension, self.dimension):
                    raise ValueError(
                        f"Diffusion matrix must be ({self.dimension}, {self.dimension}), "
                        f"got {result.shape} at position {x}"
                    )

                # Validate symmetric positive-definite
                if not self._is_symmetric_positive_definite(result):
                    raise ValueError(f"Diffusion matrix at {x} is not symmetric positive-definite: {result}")

                return result
            else:
                raise ValueError(
                    f"Callable sigma must return float or NDArray, got {type(result).__name__} at position {x}"
                )
        else:
            # Constant scalar diffusion: return σ * I
            return float(self.sigma) * np.eye(self.dimension)

    def _is_symmetric_positive_definite(self, matrix: NDArray, tol: float = 1e-10) -> bool:
        """
        Check if matrix is symmetric positive-definite.

        Args:
            matrix: Matrix to check
            tol: Tolerance for symmetry check

        Returns:
            True if matrix is SPD, False otherwise
        """
        # Check symmetric
        if not np.allclose(matrix, matrix.T, atol=tol):
            return False

        # Check positive-definite via eigenvalues
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues > 0)

    def __repr__(self) -> str:
        """String representation of the problem."""
        return (
            f"{self.__class__.__name__}("
            f"dimension={self.dimension}, "
            f"grid_shape={self.grid_shape}, "
            f"T={self.T}, Nt={self.Nt}, "
            f"σ={self.sigma})"
        )
