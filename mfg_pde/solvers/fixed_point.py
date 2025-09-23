"""
Fixed Point Solver with Clean Interface

A modern implementation of the fixed-point iteration solver that
demonstrates the new clean API design with hooks support.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mfg_pde.types import ConvergenceInfo, MFGProblem, SolutionArray, SpatialTemporalState

from .base import BaseSolver


class FixedPointSolver(BaseSolver):
    """
    Fixed-point iteration solver for MFG problems.

    This solver alternates between solving the HJB equation and the
    Fokker-Planck equation until convergence.

    Example:
        solver = FixedPointSolver(max_iterations=200, tolerance=1e-8)
        result = solver.solve(problem)

    Example with hooks:
        from mfg_pde.hooks import DebugHook

        solver = FixedPointSolver()
        result = solver.solve(problem, hooks=DebugHook())
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping_factor: float = 1.0,
        hjb_method: str = "semi_lagrangian",
        fp_method: str = "upwind",
        **config,
    ):
        """
        Initialize fixed-point solver.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            damping_factor: Damping factor for updates (0 < damping <= 1)
            hjb_method: Method for HJB equation ("semi_lagrangian", "finite_difference")
            fp_method: Method for FP equation ("upwind", "central_difference")
            **config: Additional configuration options
        """
        super().__init__(max_iterations, tolerance, **config)

        self.damping_factor = damping_factor
        self.hjb_method = hjb_method
        self.fp_method = fp_method

        # Validate parameters
        if not 0 < damping_factor <= 1:
            raise ValueError("damping_factor must be in (0, 1]")

    def _initialize_state(self, problem: MFGProblem) -> SpatialTemporalState:
        """Initialize solver state from problem."""
        # Get domain and time information
        xmin, xmax = problem.get_domain_bounds()
        T = problem.get_time_horizon()

        # Create spatial and temporal grids
        # (In a real implementation, these would come from problem or config)
        Nx = self.config.get("Nx", 100)
        Nt = self.config.get("Nt", 100)

        x_grid = np.linspace(xmin, xmax, Nx + 1)
        t_grid = np.linspace(0, T, Nt + 1)

        # Initialize u and m
        try:
            # Try to get initial conditions from problem
            u_init = problem.get_initial_value_function()
            m_init = problem.get_initial_density()

            # Ensure correct shape
            if u_init.shape != (Nt + 1, Nx + 1):
                # Reshape or interpolate as needed
                u_init = np.zeros((Nt + 1, Nx + 1))
                u_init[-1, :] = problem.get_initial_value_function()  # Terminal condition

            if m_init.shape != (Nt + 1, Nx + 1):
                m_init = np.ones((Nt + 1, Nx + 1))
                m_init[0, :] = problem.get_initial_density()  # Initial condition
                # Normalize
                m_init[0, :] /= np.trapz(m_init[0, :], x_grid)

        except (AttributeError, NotImplementedError):
            # Fallback to default initialization
            u_init = np.zeros((Nt + 1, Nx + 1))
            m_init = np.ones((Nt + 1, Nx + 1)) / (Nx + 1)

        # Compute initial residual
        initial_residual = self._compute_residual(u_init, m_init, problem, x_grid, t_grid)

        return SpatialTemporalState(
            u=u_init,
            m=m_init,
            iteration=0,
            residual=initial_residual,
            metadata={"x_grid": x_grid, "t_grid": t_grid, "hjb_method": self.hjb_method, "fp_method": self.fp_method},
        )

    def _iteration_step(self, state: SpatialTemporalState, problem: MFGProblem) -> SpatialTemporalState:
        """Perform one fixed-point iteration."""
        x_grid = state.metadata["x_grid"]
        t_grid = state.metadata["t_grid"]

        # Step 1: Solve HJB equation with current density
        u_new = self._solve_hjb_step(state.u, state.m, problem, x_grid, t_grid)

        # Step 2: Solve Fokker-Planck equation with new value function
        m_new = self._solve_fp_step(u_new, state.m, problem, x_grid, t_grid)

        # Step 3: Apply damping
        if self.damping_factor < 1.0:
            u_new = self.damping_factor * u_new + (1 - self.damping_factor) * state.u
            m_new = self.damping_factor * m_new + (1 - self.damping_factor) * state.m

        # Step 4: Compute new residual
        residual = self._compute_residual(u_new, m_new, problem, x_grid, t_grid)

        return SpatialTemporalState(
            u=u_new, m=m_new, iteration=state.iteration + 1, residual=residual, metadata=state.metadata
        )

    def _solve_hjb_step(
        self,
        u_current: SolutionArray,
        m_current: SolutionArray,
        problem: MFGProblem,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
    ) -> SolutionArray:
        """
        Solve HJB equation for one step.

        This is a simplified implementation. In a real solver, this would
        call the appropriate HJB solver based on self.hjb_method.
        """
        # Simplified HJB step - just a placeholder
        # Real implementation would use semi-Lagrangian or finite difference methods
        u_new = u_current.copy()

        # Simple update rule (placeholder)
        dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.01
        x_grid[1] - x_grid[0] if len(x_grid) > 1 else 0.01

        # Backward time iteration
        for n in range(len(t_grid) - 2, -1, -1):
            for i in range(1, len(x_grid) - 1):
                # Simplified HJB update
                x, t = x_grid[i], t_grid[n]
                m_val = m_current[n, i]

                # Simple finite difference approximation
                hamiltonian_val = problem.evaluate_hamiltonian(x, 0.0, m_val, t)
                u_new[n, i] = u_new[n + 1, i] - dt * hamiltonian_val

        return u_new

    def _solve_fp_step(
        self,
        u_current: SolutionArray,
        m_current: SolutionArray,
        problem: MFGProblem,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
    ) -> SolutionArray:
        """
        Solve Fokker-Planck equation for one step.

        This is a simplified implementation. In a real solver, this would
        call the appropriate FP solver based on self.fp_method.
        """
        # Simplified FP step - just a placeholder
        m_new = m_current.copy()

        # Simple update rule (placeholder)
        t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.01
        x_grid[1] - x_grid[0] if len(x_grid) > 1 else 0.01

        # Forward time iteration
        for n in range(len(t_grid) - 1):
            # Simple conservation of mass constraint
            total_mass = np.trapz(m_new[n, :], x_grid)
            if total_mass > 0:
                m_new[n, :] /= total_mass

        return m_new

    def _compute_residual(
        self, u: SolutionArray, m: SolutionArray, problem: MFGProblem, x_grid: np.ndarray, t_grid: np.ndarray
    ) -> float:
        """Compute residual for convergence checking."""
        # Simple L2 residual - in practice would check equation residuals
        return float(np.sqrt(np.sum(u**2) + np.sum(m**2))) / (u.size + m.size)

    def _create_result(
        self,
        state: SpatialTemporalState,
        problem: MFGProblem,
        convergence_info: ConvergenceInfo,
        total_time: float,
        avg_iteration_time: float,
    ) -> FixedPointResult:
        """Create result object with rich analysis capabilities."""
        return FixedPointResult(
            u=state.u,
            m=state.m,
            problem=problem,
            convergence_info=convergence_info,
            solver_info=self.get_solver_info(),
            x_grid=state.metadata["x_grid"],
            t_grid=state.metadata["t_grid"],
            total_time=total_time,
            avg_iteration_time=avg_iteration_time,
        )


class FixedPointResult:
    """
    Result object for FixedPointSolver with rich analysis capabilities.

    This class implements the MFGResult protocol and provides additional
    methods specific to fixed-point iteration results.
    """

    def __init__(
        self,
        u: SolutionArray,
        m: SolutionArray,
        problem: MFGProblem,
        convergence_info: ConvergenceInfo,
        solver_info: dict[str, Any],
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        total_time: float,
        avg_iteration_time: float,
    ):
        self._u = u
        self._m = m
        self._problem = problem
        self._convergence_info = convergence_info
        self._solver_info = solver_info
        self._x_grid = x_grid
        self._t_grid = t_grid
        self._total_time = total_time
        self._avg_iteration_time = avg_iteration_time

    # MFGResult protocol implementation
    @property
    def u(self) -> SolutionArray:
        """Value function u(t, x)."""
        return self._u

    @property
    def m(self) -> SolutionArray:
        """Density function m(t, x)."""
        return self._m

    @property
    def converged(self) -> bool:
        """Whether the solver converged."""
        return self._convergence_info.converged

    @property
    def iterations(self) -> int:
        """Number of iterations performed."""
        return self._convergence_info.iterations

    def plot_solution(self, **kwargs) -> None:
        """Plot the solution (u and m)."""
        try:
            import matplotlib.pyplot as plt

            _fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Value function
            im1 = axes[0, 0].imshow(self._u, aspect="auto", cmap="viridis", **kwargs)
            axes[0, 0].set_title("Value Function u(t,x)")
            axes[0, 0].set_xlabel("Space")
            axes[0, 0].set_ylabel("Time")
            plt.colorbar(im1, ax=axes[0, 0])

            # Density function
            im2 = axes[0, 1].imshow(self._m, aspect="auto", cmap="plasma", **kwargs)
            axes[0, 1].set_title("Density m(t,x)")
            axes[0, 1].set_xlabel("Space")
            axes[0, 1].set_ylabel("Time")
            plt.colorbar(im2, ax=axes[0, 1])

            # Convergence history
            axes[1, 0].semilogy(self._convergence_info.residual_history)
            axes[1, 0].set_title("Convergence History")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Residual")
            axes[1, 0].grid(True)

            # Final time profiles
            axes[1, 1].plot(self._x_grid, self._u[-1, :], label="u(T,x)", alpha=0.7)
            axes[1, 1].plot(self._x_grid, self._m[-1, :], label="m(T,x)", alpha=0.7)
            axes[1, 1].set_title("Final Time Profiles")
            axes[1, 1].set_xlabel("Space")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def export_data(self, filename: str) -> None:
        """Export solution data to file."""
        if filename.endswith(".npz"):
            np.savez(
                filename,
                u=self._u,
                m=self._m,
                x_grid=self._x_grid,
                t_grid=self._t_grid,
                converged=self.converged,
                iterations=self.iterations,
            )
        elif filename.endswith(".h5"):
            try:
                import h5py

                with h5py.File(filename, "w") as f:
                    f.create_dataset("u", data=self._u)
                    f.create_dataset("m", data=self._m)
                    f.create_dataset("x_grid", data=self._x_grid)
                    f.create_dataset("t_grid", data=self._t_grid)
                    f.attrs["converged"] = self.converged
                    f.attrs["iterations"] = self.iterations
            except ImportError:
                raise ImportError("h5py required for HDF5 export")
        else:
            raise ValueError("Unsupported file format. Use .npz or .h5")

    # Additional methods specific to FixedPointResult
    @property
    def convergence_info(self) -> ConvergenceInfo:
        """Detailed convergence information."""
        return self._convergence_info

    @property
    def solver_info(self) -> dict[str, Any]:
        """Information about the solver used."""
        return self._solver_info.copy()

    @property
    def total_time(self) -> float:
        """Total solving time in seconds."""
        return self._total_time

    @property
    def x_grid(self) -> np.ndarray:
        """Spatial grid."""
        return self._x_grid

    @property
    def t_grid(self) -> np.ndarray:
        """Temporal grid."""
        return self._t_grid

    def get_solution_at_time(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Get solution at specific time."""
        # Find closest time index
        t_idx = np.argmin(np.abs(self._t_grid - t))
        return self._u[t_idx, :], self._m[t_idx, :]

    def __str__(self) -> str:
        return (
            f"FixedPointResult(converged={self.converged}, "
            f"iterations={self.iterations}, "
            f"final_residual={self._convergence_info.final_residual:.2e})"
        )

    def __repr__(self) -> str:
        return self.__str__()
