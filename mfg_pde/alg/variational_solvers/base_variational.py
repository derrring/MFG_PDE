#!/usr/bin/env python3
"""
Base Variational Solver for Lagrangian MFG Problems

This module provides the abstract base class for all variational solvers
that directly optimize the Lagrangian formulation of MFG problems.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.utils.integration import trapezoid

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.core.lagrangian_mfg_problem import LagrangianMFGProblem

logger = logging.getLogger(__name__)


@dataclass
class VariationalSolverResult:
    """
    Result container for variational MFG solvers.

    Unlike HJB-FP results that return value function and density,
    variational results focus on optimal trajectories and flows.
    """

    # Optimal solution
    optimal_flow: NDArray | None = None  # m(t,x) population flow
    representative_trajectory: NDArray | None = None  # x(t) typical trajectory
    representative_velocity: NDArray | None = None  # ẋ(t) typical velocity

    # Cost and convergence information
    final_cost: float = np.inf  # J[x,m] final cost value
    cost_history: list | None = None  # Cost evolution during optimization
    converged: bool = False  # Convergence flag
    num_iterations: int = 0  # Number of iterations

    # Constraint satisfaction
    constraint_violations: dict[str, float] | None = None

    # Timing and performance
    solve_time: float = 0.0  # Total solve time

    # Additional solver-specific information
    solver_info: dict[str, Any] | None = None

    def __post_init__(self):
        if self.solver_info is None:
            self.solver_info = {}
        if self.cost_history is None:
            self.cost_history = []
        if self.constraint_violations is None:
            self.constraint_violations = {}


class BaseVariationalSolver(ABC):
    """
    Abstract base class for variational MFG solvers.

    Variational solvers directly optimize the cost functional:
    min J[m] = ∫₀ᵀ ∫ L(t,x,v(t,x),m(t,x)) dxdt + ∫ g(x)m(T,x) dx

    subject to the continuity equation:
    ∂m/∂t + ∇·(m v) = σ²/2 Δm
    """

    def __init__(self, problem: LagrangianMFGProblem):
        """
        Initialize variational solver.

        Args:
            problem: Lagrangian MFG problem to solve
        """
        self.problem = problem
        self.solver_name = "BaseVariational"

        # Grid information
        self.x_grid = problem.x
        self.t_grid = problem.t
        self.Nx = problem.Nx
        self.Nt = problem.Nt
        self.dx = problem.Dx
        self.dt = problem.Dt

        logger.info(f"Initialized {self.solver_name} solver")
        logger.info(f"  Problem: {problem.components.description}")
        logger.info(f"  Grid: {self.Nx+1} × {self.Nt} (space × time)")

    @abstractmethod
    def solve(
        self,
        initial_guess: NDArray | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> VariationalSolverResult:
        """
        Solve the variational MFG problem.

        Args:
            initial_guess: Initial guess for density evolution
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            **kwargs: Solver-specific parameters

        Returns:
            VariationalSolverResult with optimal solution
        """

    def evaluate_cost_functional(self, density_evolution: NDArray, velocity_field: NDArray | None = None) -> float:
        """
        Evaluate the cost functional J[m,v].

        Args:
            density_evolution: m(t,x) density field shape (Nt, Nx+1)
            velocity_field: v(t,x) velocity field shape (Nt, Nx+1)

        Returns:
            Total cost value
        """
        if velocity_field is None:
            velocity_field = self._compute_velocity_from_density(density_evolution)

        # Running cost integral
        running_cost = 0.0
        for i, t in enumerate(self.t_grid):
            for j, x in enumerate(self.x_grid):
                m = density_evolution[i, j]
                v = velocity_field[i, j]

                if m > 1e-12:  # Avoid numerical issues with zero density
                    L_value = self.problem.evaluate_lagrangian(t, x, v, m)
                    running_cost += L_value * m * self.dx * self.dt

        # Terminal cost
        terminal_cost = 0.0
        for j, x in enumerate(self.x_grid):
            m_final = density_evolution[-1, j]
            g_value = self.problem.evaluate_terminal_cost(x)
            terminal_cost += g_value * m_final * self.dx

        total_cost = running_cost + terminal_cost
        return total_cost

    def _compute_velocity_from_density(self, density_evolution: NDArray) -> NDArray:
        """
        Compute velocity field from density evolution using continuity equation.

        The continuity equation: ∂m/∂t + ∇·(mv) = σ²/2 Δm
        can be solved for v: mv = -∫(∂m/∂t - σ²/2 Δm) dx

        Args:
            density_evolution: m(t,x) shape (Nt, Nx+1)

        Returns:
            velocity_field: v(t,x) shape (Nt, Nx+1)
        """
        velocity_field = np.zeros_like(density_evolution)

        for i in range(1, self.Nt):  # Skip initial time
            for j in range(1, self.Nx):  # Skip boundaries
                m = density_evolution[i, j]

                if m > 1e-12:  # Avoid division by zero
                    # Time derivative ∂m/∂t
                    dm_dt = (density_evolution[i, j] - density_evolution[i - 1, j]) / self.dt

                    # Diffusion term σ²/2 Δm
                    d2m_dx2 = (
                        density_evolution[i, j + 1] - 2 * density_evolution[i, j] + density_evolution[i, j - 1]
                    ) / self.dx**2
                    diffusion = 0.5 * self.problem.sigma**2 * d2m_dx2

                    # Solve for velocity (simplified 1D case)
                    # This is an approximation - full implementation would solve the system properly
                    velocity_field[i, j] = -(dm_dt - diffusion) / m
                else:
                    velocity_field[i, j] = 0.0

        return velocity_field

    def check_continuity_equation(self, density_evolution: NDArray, velocity_field: NDArray) -> float:
        """
        Check how well the continuity equation is satisfied.

        Returns the L2 norm of the residual:
        ||∂m/∂t + ∇·(mv) - σ²/2 Δm||₂

        Args:
            density_evolution: m(t,x) shape (Nt, Nx+1)
            velocity_field: v(t,x) shape (Nt, Nx+1)

        Returns:
            L2 norm of continuity equation residual
        """
        residual = np.zeros((self.Nt - 1, self.Nx - 1))

        for i in range(1, self.Nt):
            for j in range(1, self.Nx):
                # Time derivative
                dm_dt = (density_evolution[i, j] - density_evolution[i - 1, j]) / self.dt

                # Divergence term ∇·(mv)
                mv_left = density_evolution[i, j - 1] * velocity_field[i, j - 1]
                mv_right = density_evolution[i, j + 1] * velocity_field[i, j + 1]
                div_mv = (mv_right - mv_left) / (2 * self.dx)

                # Diffusion term
                d2m_dx2 = (
                    density_evolution[i, j + 1] - 2 * density_evolution[i, j] + density_evolution[i, j - 1]
                ) / self.dx**2
                diffusion = 0.5 * self.problem.sigma**2 * d2m_dx2

                # Continuity equation residual
                residual[i - 1, j - 1] = dm_dt + div_mv - diffusion

        return np.linalg.norm(residual) * np.sqrt(self.dx * self.dt)

    def compute_mass_conservation_error(self, density_evolution: NDArray) -> float:
        """
        Check mass conservation: ∫m(t,x)dx = constant

        Args:
            density_evolution: m(t,x) shape (Nt, Nx+1)

        Returns:
            Maximum mass conservation error over time
        """
        total_mass = trapezoid(density_evolution, x=self.x_grid, axis=1)
        initial_mass = total_mass[0]
        mass_errors = np.abs(total_mass - initial_mass)
        return np.max(mass_errors)

    def create_initial_guess(self, strategy: str = "uniform") -> NDArray:
        """
        Create initial guess for density evolution.

        Args:
            strategy: Strategy for initial guess
                     "uniform" - constant density
                     "gaussian" - Gaussian initial condition evolving
                     "random" - random perturbation of uniform

        Returns:
            Initial density evolution shape (Nt, Nx+1)
        """
        density_guess = np.zeros((self.Nt, self.Nx + 1))

        if strategy == "uniform":
            # Uniform density maintained over time
            for i in range(self.Nt):
                if self.problem.components.initial_density_func:
                    density_guess[i, :] = [self.problem.components.initial_density_func(x) for x in self.x_grid]
                else:
                    density_guess[i, :] = 1.0 / (self.problem.xmax - self.problem.xmin)

        elif strategy == "gaussian":
            # Gaussian spreading with diffusion
            center = 0.5 * (self.problem.xmin + self.problem.xmax)
            initial_width = 0.1 * (self.problem.xmax - self.problem.xmin)

            for i, t in enumerate(self.t_grid):
                # Width increases with time due to diffusion
                width = initial_width + self.problem.sigma * np.sqrt(t)

                gaussian = np.exp(-0.5 * ((self.x_grid - center) / width) ** 2)
                gaussian = gaussian / trapezoid(gaussian, x=self.x_grid)  # Normalize
                density_guess[i, :] = gaussian

        elif strategy == "random":
            # Random perturbation of uniform
            base_density = 1.0 / (self.problem.xmax - self.problem.xmin)

            for i in range(self.Nt):
                perturbation = 0.1 * np.random.randn(self.Nx + 1)
                density_guess[i, :] = base_density * (1 + perturbation)

                # Ensure positivity and normalization
                density_guess[i, :] = np.maximum(density_guess[i, :], 1e-6)
                density_guess[i, :] = density_guess[i, :] / trapezoid(density_guess[i, :], x=self.x_grid)

        else:
            raise ValueError(f"Unknown initial guess strategy: {strategy}")

        return density_guess

    def get_solver_info(self) -> dict[str, Any]:
        """Return solver information."""
        return {
            "solver_type": "Variational",
            "solver_name": self.solver_name,
            "problem_type": "Lagrangian MFG",
            "grid_size": {"Nx": self.Nx, "Nt": self.Nt},
            "domain": {
                "spatial": [self.problem.xmin, self.problem.xmax],
                "temporal": [0.0, self.problem.T],
            },
        }
