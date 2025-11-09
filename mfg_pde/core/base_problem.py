#!/usr/bin/env python3
"""
Base problem infrastructure for dimension-agnostic MFG problems.

This module defines the core protocol that ALL MFG problems must implement,
regardless of dimension (1D, 2D, 3D, nD) or domain type (Cartesian grid,
network, manifold, implicit domain, etc.).

Mathematical Notation:
    - m(t,x): Density function
    - u(t,x): Value function
    - ∂u/∂x: Spatial gradient
    - H(x, m, p, t): Hamiltonian
    - g(x): Terminal cost
    - f(x, m, t): Running cost

Part of: Issue #245 - Incremental Evolution toward Unified nD Architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@runtime_checkable
class MFGProblemProtocol(Protocol):
    """
    Protocol that ALL MFG problems must satisfy for dimension-agnostic solvers.

    This defines the minimal interface required by solvers. Any class
    implementing these attributes and methods can be used with our
    solver infrastructure, enabling:
    - Uniform solver code for 1D, 2D, 3D, nD
    - Type-safe solver implementations
    - Clear separation between problem definition and solver logic

    Spatial Properties:
        dimension: int
            Number of spatial dimensions (1, 2, 3, ...)
        spatial_bounds: list[tuple[float, float]]
            [(x₀_min, x₀_max), (x₁_min, x₁_max), ...]
        spatial_discretization: list[int]
            [N₀, N₁, ...] grid points per dimension
        spatial_grid: NDArray | list[NDArray]
            Coordinate arrays (format depends on dimension/domain)
        grid_shape: tuple[int, ...]
            Shape of spatial grid (N₀, N₁, ...)
        grid_spacing: list[float]
            [Δx₀, Δx₁, ...] spacing per dimension

    Temporal Properties:
        T: float
            Final time
        Nt: int
            Number of time steps
        dt: float
            Time step size Δt = T/Nt
        time_grid: NDArray
            Array of time points [t₀, t₁, ..., t_Nt]

    Physical Properties:
        sigma: float | Callable
            Diffusion coefficient σ
            - float: Constant scalar diffusion
            - Callable → float: Position-dependent scalar σ(x)
            - Callable → NDArray: Matrix diffusion D(x) ∈ ℝ^{d×d}

    MFG Components:
        All problems must provide these mathematical components:
        - hamiltonian(x, m, p, t): H(x, m, p, t)
        - terminal_cost(x): g(x)
        - initial_density(x): m₀(x)
        - running_cost(x, m, t): f(x, m, t)

    Examples:
        >>> # Type checking with Protocol
        >>> def solve_hjb(problem: MFGProblemProtocol) -> NDArray:
        ...     # Solver works for ANY dimension
        ...     dim = problem.dimension
        ...     dx = problem.grid_spacing
        ...     # ... dimension-agnostic solver code ...
        ...     return u

        >>> # Runtime validation
        >>> from mfg_pde.core.base_problem import MFGProblemProtocol
        >>> problem = MFGProblem(...)
        >>> assert isinstance(problem, MFGProblemProtocol)
    """

    # ====================
    # Spatial Properties
    # ====================

    dimension: int
    spatial_bounds: list[tuple[float, float]]
    spatial_discretization: list[int]
    spatial_grid: NDArray | list[NDArray]
    grid_shape: tuple[int, ...]
    grid_spacing: list[float]

    # ====================
    # Temporal Properties
    # ====================

    T: float
    Nt: int
    dt: float
    time_grid: NDArray

    # ====================
    # Physical Properties
    # ====================

    sigma: float | Callable

    # ====================
    # MFG Components
    # ====================

    def hamiltonian(self, x, m, p, t) -> float:
        """
        Hamiltonian H(x, m, p, t).

        Args:
            x: Spatial position
                - 1D: float
                - nD: tuple/array of length d
            m: Density value m(t,x) at this position
            p: Momentum/co-state ∂u/∂x
                - 1D: float
                - nD: tuple/array of length d
            t: Time

        Returns:
            Hamiltonian value H(x, m, p, t)

        Example:
            >>> # Quadratic Hamiltonian with congestion
            >>> def hamiltonian(self, x, m, p, t):
            ...     p_arr = np.array(p) if hasattr(p, '__iter__') else p
            ...     return 0.5 * np.sum(p_arr**2) + 0.1 * m
        """
        ...

    def terminal_cost(self, x) -> float:
        """
        Terminal cost g(x).

        Args:
            x: Spatial position

        Returns:
            Terminal cost value g(x)

        Example:
            >>> # Quadratic terminal cost
            >>> def terminal_cost(self, x):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else np.array([x])
            ...     return 0.5 * np.sum((x_arr - 0.5)**2)
        """
        ...

    def initial_density(self, x) -> float:
        """
        Initial density m₀(x).

        Args:
            x: Spatial position

        Returns:
            Initial density value m₀(x) ≥ 0

        Example:
            >>> # Gaussian initial density
            >>> def initial_density(self, x):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else np.array([x])
            ...     return np.exp(-10 * np.sum((x_arr - 0.5)**2))
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

        Example:
            >>> # Congestion cost
            >>> def running_cost(self, x, m, t):
            ...     return 0.1 * m  # Penalize high density
        """
        ...
