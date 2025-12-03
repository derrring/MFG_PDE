"""
Hybrid FP-Particle + HJB-FDM Solver for Mean Field Games.

.. deprecated:: 0.9.0
    This class is deprecated and will be removed in v1.0.0.
    Use :class:`FixedPointIterator` with modular HJB/FP solvers instead.

    Migration example::

        # OLD (deprecated)
        from mfg_pde.alg.numerical.coupling import HybridFPParticleHJBFDM
        solver = HybridFPParticleHJBFDM(problem, num_particles=5000)

        # NEW (recommended)
        from mfg_pde import solve_mfg
        from mfg_pde.config import ConfigBuilder

        config = (
            ConfigBuilder()
            .solver_hjb(method="fdm")
            .solver_fp_particle(num_particles=5000)
            .picard(max_iterations=20, damping_factor=0.5)
            .build()
        )
        result = solve_mfg(problem, config=config)

This module implements a hybrid solver that combines:
- Fokker-Planck equation: Particle-based solution
- Hamilton-Jacobi-Bellman equation: Finite Difference Method (FDM)

This hybrid approach leverages the strengths of both methods:
- Particle methods handle complex geometries and mass conservation naturally
- FDM provides stable and accurate solution of the HJB equation
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_mfg import BaseMFGSolver

if TYPE_CHECKING:
    from mfg_pde.config.solver_config import MFGSolverConfig
    from mfg_pde.core.mfg_problem import MFGProblem


class HybridFPParticleHJBFDM(BaseMFGSolver):
    """
    Hybrid solver combining Particle FP and FDM HJB methods.

    This solver implements the specific combination:
    - Fokker-Planck equation solved using particle methods
    - HJB equation solved using finite difference methods
    - Coupled through fixed point iteration with damping
    """

    def __init__(
        self,
        problem: MFGProblem | None = None,
        num_particles: int = 5000,
        kde_bandwidth: str | float = "scott",
        max_newton_iterations: int = 30,
        newton_tolerance: float = 1e-7,
        damping_parameter: float = 0.5,
        config: MFGSolverConfig | None = None,
        # Deprecated parameters for backward compatibility
        mfg_problem: MFGProblem | None = None,
        hjb_newton_iterations: int | None = None,
        hjb_newton_tolerance: float | None = None,
        hjb_fd_scheme: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the hybrid FP-Particle + HJB-FDM solver.

        .. deprecated:: 0.9.0
            This class is deprecated. Use FixedPointIterator with ConfigBuilder instead.

        Args:
            problem: MFG problem to solve
            num_particles: Number of particles for FP solver
            kde_bandwidth: Bandwidth for kernel density estimation
            max_newton_iterations: Max Newton iterations for HJB solver
            newton_tolerance: Newton tolerance for HJB solver
            damping_parameter: Damping parameter for fixed point iteration
            config: Optional configuration object
            mfg_problem: (Deprecated) Use 'problem' instead
            hjb_newton_iterations: (Deprecated) Use 'max_newton_iterations' instead
            hjb_newton_tolerance: (Deprecated) Use 'newton_tolerance' instead
            hjb_fd_scheme: (Deprecated) No longer used, HJBFDMSolver handles scheme internally
            **kwargs: Additional parameters
        """
        import warnings

        # Handle deprecated 'mfg_problem' parameter
        if mfg_problem is not None:
            warnings.warn(
                "Parameter 'mfg_problem' is deprecated. Use 'problem' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if problem is None:
                problem = mfg_problem

        # Ensure problem is provided
        if problem is None:
            raise ValueError("Parameter 'problem' is required")

        # Handle deprecated 'hjb_newton_iterations' parameter
        if hjb_newton_iterations is not None:
            warnings.warn(
                "Parameter 'hjb_newton_iterations' is deprecated. Use 'max_newton_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_newton_iterations = hjb_newton_iterations

        # Handle deprecated 'hjb_newton_tolerance' parameter
        if hjb_newton_tolerance is not None:
            warnings.warn(
                "Parameter 'hjb_newton_tolerance' is deprecated. Use 'newton_tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            newton_tolerance = hjb_newton_tolerance

        # Warn about hjb_fd_scheme being removed
        if hjb_fd_scheme is not None:
            warnings.warn(
                "Parameter 'hjb_fd_scheme' is deprecated and ignored. "
                "HJBFDMSolver handles finite difference scheme internally.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(problem)

        # Store solver parameters
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.max_newton_iterations = max_newton_iterations
        self.newton_tolerance = newton_tolerance
        self.damping_parameter = damping_parameter
        self.config = config

        # Initialize individual solvers
        self._initialize_solvers()

        # Solver identification
        self.name = f"Hybrid(FP-Particle[{num_particles}]_HJB-FDM)"

        # Solution storage
        self.U_solution: np.ndarray
        self.M_solution: np.ndarray
        self.convergence_history: list[dict[str, Any]] = []

    def _initialize_solvers(self) -> None:
        """Initialize the FP particle and HJB FDM solvers."""
        # Initialize FP Particle solver
        from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

        self.fp_solver = FPParticleSolver(
            problem=self.problem,
            num_particles=self.num_particles,
            kde_bandwidth=self.kde_bandwidth,
            normalize_kde_output=True,
        )

        # Initialize HJB FDM solver
        from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        self.hjb_solver = HJBFDMSolver(
            problem=self.problem,
            max_newton_iterations=self.max_newton_iterations,
            newton_tolerance=self.newton_tolerance,
        )

    def solve(
        self,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        # Deprecated parameters for backward compatibility
        max_picard_iterations: int | None = None,
        picard_tolerance: float | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Solve the MFG system using hybrid FP-Particle + HJB-FDM approach.

        Args:
            max_iterations: Maximum number of Picard iterations
            tolerance: Convergence tolerance
            max_picard_iterations: (Deprecated) Use max_iterations instead
            picard_tolerance: (Deprecated) Use tolerance instead
            verbose: Whether to print progress information
            **kwargs: Additional parameters

        Returns:
            Tuple of (U_solution, M_solution, convergence_info)
        """
        import warnings

        # Handle parameter precedence: standardized > deprecated
        if max_iterations is not None:
            final_max_iterations = max_iterations
        elif max_picard_iterations is not None:
            warnings.warn(
                "Parameter 'max_picard_iterations' is deprecated. Use 'max_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_max_iterations = max_picard_iterations
        else:
            final_max_iterations = 50  # Default for hybrid solver

        if tolerance is not None:
            final_tolerance = tolerance
        elif picard_tolerance is not None:
            warnings.warn(
                "Parameter 'picard_tolerance' is deprecated. Use 'tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_tolerance = picard_tolerance
        else:
            final_tolerance = 1e-6  # Default for hybrid solver

        if verbose:
            print(f"Starting {self.name}:")
            print(f"  - FP Particles: {self.num_particles}")
            print("  - HJB Method: Finite Difference (Newton)")
            print(f"  - Max Picard iterations: {final_max_iterations}")
            print(f"  - Convergence tolerance: {final_tolerance}")
            print(f"  - Damping parameter: {self.damping_parameter}")

        # Get problem dimensions
        Nt = self.problem.Nt + 1
        Nx = self.problem.Nx + 1

        # Initialize solutions
        warm_start_init = self._get_warm_start_initialization()
        if warm_start_init is not None:
            U_current, M_current = warm_start_init
            if verbose:
                print("   Using warm start initialization")
        else:
            U_current = np.zeros((Nt, Nx))
            M_current = np.zeros((Nt, Nx))

        # Get boundary conditions
        if hasattr(self.problem, "get_final_u"):
            terminal_condition = self.problem.get_final_u()
        else:
            terminal_condition = np.zeros(Nx)

        if hasattr(self.problem, "get_initial_m"):
            initial_density = self.problem.get_initial_m()
        else:
            initial_density = np.ones(Nx) / self.problem.Lx if self.problem.dx > 1e-14 else np.ones(Nx)

        # Initialize with boundary conditions
        if warm_start_init is None:
            for t in range(Nt):
                U_current[t, :] = terminal_condition
            for t in range(Nt):
                M_current[t, :] = initial_density

        # Always enforce boundary conditions
        U_current[Nt - 1, :] = terminal_condition
        M_current[0, :] = initial_density

        # Hybrid Picard iteration
        self.convergence_history = []
        solve_start_time = time.time()

        for picard_iter in range(final_max_iterations):
            iter_start_time = time.time()

            if verbose and picard_iter % 10 == 0:
                print(f"  Hybrid Picard iteration {picard_iter}")

            # Store previous iteration
            U_prev = U_current.copy()
            M_prev = M_current.copy()

            # Step 1: Solve HJB equation using FDM with current density
            try:
                U_new = self.hjb_solver.solve_hjb_system(
                    M_density=M_current,
                    U_terminal=terminal_condition,
                    U_coupling_prev=U_current,
                )

                # Apply damping
                U_current = self.damping_parameter * U_new + (1 - self.damping_parameter) * U_prev

            except Exception as e:
                if verbose:
                    print(f"  Warning: HJB-FDM solver failed at iteration {picard_iter}: {e}")
                U_current = U_prev.copy()

            # Step 2: Solve FP equation using particles with updated control
            try:
                M_new = self.fp_solver.solve_fp_system(
                    M_initial=initial_density,
                    drift_field=U_current,
                )

                # Apply damping
                M_current = self.damping_parameter * M_new + (1 - self.damping_parameter) * M_prev

            except Exception as e:
                if verbose:
                    print(f"  Warning: FP-Particle solver failed at iteration {picard_iter}: {e}")
                M_current = M_prev.copy()

            # Compute convergence metrics
            U_error = np.linalg.norm(U_current - U_prev) / max(float(np.linalg.norm(U_prev)), 1e-10)
            M_error = np.linalg.norm(M_current - M_prev) / max(float(np.linalg.norm(M_prev)), 1e-10)
            total_error = max(float(U_error), float(M_error))

            iter_time = time.time() - iter_start_time

            convergence_info = {
                "iteration": picard_iter,
                "U_error": U_error,
                "M_error": M_error,
                "total_error": total_error,
                "iteration_time": iter_time,
            }
            self.convergence_history.append(convergence_info)

            if verbose and picard_iter % 10 == 0:
                print(f"    Hybrid errors: U={U_error:.2e}, M={M_error:.2e}, Time={iter_time:.3f}s")

            # Check convergence
            if total_error < final_tolerance:
                if verbose:
                    print(f"  Hybrid convergence achieved at iteration {picard_iter}")
                    print(f"    Final U error: {U_error:.2e}")
                    print(f"    Final M error: {M_error:.2e}")
                break

        # Store results
        self.U_solution = U_current
        self.M_solution = M_current

        # Prepare convergence info
        execution_time = time.time() - solve_start_time
        final_convergence_info = {
            "converged": total_error < final_tolerance,
            "final_error": total_error,
            "iterations": len(self.convergence_history),
            "execution_time": execution_time,
            "convergence_history": self.convergence_history,
            "solver_info": {
                "name": self.name,
                "fp_method": "particle",
                "hjb_method": "fdm",
                "num_particles": self.num_particles,
                "damping_parameter": self.damping_parameter,
            },
        }

        if verbose:
            print(f"Hybrid solver completed in {execution_time:.3f}s:")
            print(f"  - Converged: {final_convergence_info['converged']}")
            print(f"  - Final error: {final_convergence_info['final_error']:.2e}")
            print(f"  - Total iterations: {final_convergence_info['iterations']}")

        # Mark solution as computed for warm start capability
        self._solution_computed = True

        return U_current, M_current, final_convergence_info

    def _get_warm_start_initialization(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get warm start initialization data."""
        return self.get_warm_start_data()

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the computed U and M solutions."""
        if not hasattr(self, "U_solution") or not hasattr(self, "M_solution"):
            raise ValueError("No solution available. Call solve() first.")
        return self.U_solution, self.M_solution

    def get_convergence_history(self) -> list[dict[str, Any]]:
        """Get the convergence history."""
        return self.convergence_history

    def get_solver_info(self) -> dict[str, Any]:
        """Get comprehensive information about the hybrid solver."""
        return {
            "name": self.name,
            "type": "hybrid",
            "fp_solver": {
                "method": "particle",
                "num_particles": self.num_particles,
                "kde_bandwidth": self.kde_bandwidth,
            },
            "hjb_solver": {
                "method": "fdm",
                "max_newton_iterations": self.max_newton_iterations,
                "newton_tolerance": self.newton_tolerance,
            },
            "coupling": {
                "method": "fixed_point_iteration",
                "damping_parameter": self.damping_parameter,
            },
        }


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HybridFPParticleHJBFDM...")

    # Test class availability
    assert HybridFPParticleHJBFDM is not None
    print("  HybridFPParticleHJBFDM class available")

    # Full smoke test requires complete solver setup
    # See examples/advanced/ for usage examples

    print("Smoke tests passed!")
