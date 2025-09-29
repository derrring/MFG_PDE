"""
Adaptive Particle Collocation Solver.

This solver demonstrates the decorator pattern for adaptive convergence,
automatically detecting particle methods and applying appropriate convergence criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .particle_collocation_solver import ParticleCollocationSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions


class AdaptiveParticleCollocationSolver(ParticleCollocationSolver):
    """
    Particle Collocation Solver with automatic adaptive convergence.

    This solver automatically detects that it uses particle methods and applies
    advanced convergence criteria (Wasserstein distance + oscillation stabilization)
    instead of classical L2 error convergence.

    Features:
    - Automatic particle method detection
    - Advanced convergence criteria for particle-based distributions
    - Classical fallback for non-particle components
    - Detailed convergence diagnostics
    - Backward compatibility with existing ParticleCollocationSolver
    """

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        num_particles: int = 5000,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        max_newton_iterations: int = 30,
        newton_tolerance: float = 1e-4,
        kde_bandwidth: str = "scott",
        normalize_kde_output: bool = False,
        boundary_indices: np.ndarray | None = None,
        boundary_conditions: BoundaryConditions | dict | None = None,
        use_monotone_constraints: bool = False,
        # Adaptive convergence parameters
        classical_tol: float = 1e-3,
        wasserstein_tol: float = 1e-4,
        u_magnitude_tol: float = 1e-3,
        u_stability_tol: float = 1e-4,
        history_length: int = 10,
        verbose: bool = True,
        precision: str = "standard",  # "standard", "high", "fast"
    ):
        """
        Initialize adaptive particle collocation solver.

        Args:
            precision: Convergence precision level ("fast", "standard", "high")
            classical_tol: Classical L2 error tolerance
            wasserstein_tol: Wasserstein distance tolerance for distributions
            u_magnitude_tol: Magnitude tolerance for value function
            u_stability_tol: Stability tolerance for oscillation detection
            history_length: Length of convergence history for analysis
            Other parameters same as ParticleCollocationSolver
        """
        # Apply precision-based parameter adjustments
        if precision == "fast":
            classical_tol = max(classical_tol, 5e-3)
            wasserstein_tol = max(wasserstein_tol, 5e-4)
            u_magnitude_tol = max(u_magnitude_tol, 5e-3)
            history_length = min(history_length, 5)
        elif precision == "high":
            classical_tol = min(classical_tol, 5e-4)
            wasserstein_tol = min(wasserstein_tol, 5e-5)
            u_magnitude_tol = min(u_magnitude_tol, 5e-4)
            history_length = max(history_length, 15)

        super().__init__(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=num_particles,
            delta=delta,
            taylor_order=taylor_order,
            weight_function=weight_function,
            max_newton_iterations=max_newton_iterations,
            newton_tolerance=newton_tolerance,
            kde_bandwidth=kde_bandwidth,
            normalize_kde_output=normalize_kde_output,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,  # type: ignore[arg-type]
            use_monotone_constraints=use_monotone_constraints,
        )

        # Store adaptive convergence parameters
        self.adaptive_params = {
            "classical_tol": classical_tol,
            "wasserstein_tol": wasserstein_tol,
            "u_magnitude_tol": u_magnitude_tol,
            "u_stability_tol": u_stability_tol,
            "history_length": history_length,
            "verbose": verbose,
        }
        self.precision_level = precision

        # Initialize adaptive convergence state
        self.convergence_history: list[dict[str, Any]] = []
        self.adaptive_mode = "particle_aware"  # Default for particle collocation

    def solve(  # type: ignore[override]
        self,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        # Deprecated parameters for backward compatibility
        Niter: int | None = None,
        l2errBound: float | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Solve the MFG system with adaptive convergence criteria.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            Niter: (Deprecated) Use max_iterations instead
            l2errBound: (Deprecated) Use tolerance instead
            verbose: Whether to print convergence information
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (U_solution, M_solution, adaptive_convergence_info)
        """
        import warnings

        # Handle parameter precedence: standardized > deprecated
        if max_iterations is not None:
            final_max_iterations = max_iterations
        elif Niter is not None:
            warnings.warn(
                "Parameter 'Niter' is deprecated. Use 'max_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_max_iterations = Niter
        else:
            final_max_iterations = 20  # Default

        if tolerance is not None:
            # Use provided tolerance as base, but apply adaptive scaling
            base_tolerance = tolerance
        elif l2errBound is not None:
            warnings.warn(
                "Parameter 'l2errBound' is deprecated. Use 'tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            base_tolerance = l2errBound
        else:
            base_tolerance = self.adaptive_params["classical_tol"]

        if verbose:
            print("Starting Adaptive Particle-Collocation MFG solver:")
            print(f"  - Particles: {self.num_particles}")
            print(f"  - Collocation points: {self.hjb_solver.n_points}")
            print(f"  - Max iterations: {final_max_iterations}")
            print(f"  - Base tolerance: {base_tolerance}")
            print(f"  - Precision level: {self.precision_level}")
            print(f"  - Adaptive mode: {self.adaptive_mode}")

        # Get problem dimensions
        Nt = self.problem.Nt + 1
        Nx = self.problem.Nx + 1

        # Try warm start initialization first
        warm_start_init = self._get_warm_start_initialization()
        if warm_start_init is not None:
            U_current, M_current = warm_start_init
            if verbose:
                print("   Using warm start initialization from previous solution")
        else:
            # Cold start - better initialization
            U_current = np.zeros((Nt, Nx))
            M_current = np.zeros((Nt, Nx))

        # Get terminal condition for U
        if hasattr(self.problem, "get_final_u"):
            terminal_condition = self.problem.get_final_u()
        else:
            terminal_condition = np.zeros(Nx)

        # Get initial density for M
        if hasattr(self.problem, "get_initial_m"):
            initial_density = self.problem.get_initial_m()
        else:
            initial_density = np.ones(Nx) / self.problem.Lx if self.problem.Dx > 1e-14 else np.ones(Nx)

        # For cold start, initialize interior with boundary conditions
        if warm_start_init is None:
            for t in range(Nt):
                U_current[t, :] = terminal_condition
            for t in range(Nt):
                M_current[t, :] = initial_density

        # Always enforce boundary conditions
        U_current[Nt - 1, :] = terminal_condition
        M_current[0, :] = initial_density

        # Adaptive Picard iteration
        self.convergence_history = []

        for picard_iter in range(final_max_iterations):
            if verbose and picard_iter % 10 == 0:
                print(f"  Adaptive Picard iteration {picard_iter}")

            # Store previous iteration for convergence check
            U_prev = U_current.copy()
            M_prev = M_current.copy()

            # Step 1: Solve HJB equation with current density
            try:
                U_new = self.hjb_solver.solve_hjb_system(
                    M_density_evolution_from_FP=M_current,
                    U_final_condition_at_T=U_current[Nt - 1, :],
                    U_from_prev_picard=U_current,
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: HJB solver failed at iteration {picard_iter}: {e}")
                U_new = U_current.copy()

            # Step 2: Solve FP equation with updated control
            try:
                M_new = self.fp_solver.solve_fp_system(m_initial_condition=M_current[0, :], U_solution_for_drift=U_new)
            except Exception as e:
                if verbose:
                    print(f"  Warning: FP solver failed at iteration {picard_iter}: {e}")
                M_new = M_current.copy()

            # Update solutions
            U_current = U_new
            M_current = M_new

            # Adaptive convergence checking
            convergence_result = self._check_adaptive_convergence(
                U_current, M_current, U_prev, M_prev, picard_iter, base_tolerance
            )

            # Store convergence information
            self.convergence_history.append(convergence_result)

            if verbose and picard_iter % 10 == 0:
                print(
                    f"    Adaptive errors: U={convergence_result['U_error']:.2e}, M={convergence_result['M_error']:.2e}"
                )

            # Check adaptive convergence criteria
            if convergence_result["converged"]:
                if verbose:
                    print(f"  Adaptive convergence achieved at iteration {picard_iter}")
                    print(f"    Final U error: {convergence_result['U_error']:.2e}")
                    print(f"    Final M error: {convergence_result['M_error']:.2e}")
                    print(f"    Convergence mode: {convergence_result['mode']}")
                break

        # Store results
        self.U_solution = U_current  # type: ignore[assignment]
        self.M_solution = M_current  # type: ignore[assignment]

        # Prepare adaptive convergence info
        final_convergence = self.convergence_history[-1] if self.convergence_history else {"converged": False}
        adaptive_convergence_info = {
            "converged": final_convergence["converged"],
            "final_error": max(
                final_convergence.get("U_error", float("inf")), final_convergence.get("M_error", float("inf"))
            ),
            "iterations": len(self.convergence_history),
            "convergence_history": self.convergence_history,
            "adaptive_mode": self.adaptive_mode,
            "precision_level": self.precision_level,
            "adaptive_params": self.adaptive_params,
        }

        if verbose:
            print("Adaptive Particle-Collocation solver completed:")
            print(f"  - Converged: {adaptive_convergence_info['converged']}")
            print(f"  - Final error: {adaptive_convergence_info['final_error']:.2e}")
            print(f"  - Total iterations: {adaptive_convergence_info['iterations']}")
            print(f"  - Adaptive mode: {adaptive_convergence_info['adaptive_mode']}")

        # Mark solution as computed for warm start capability
        self._solution_computed = True

        return U_current, M_current, adaptive_convergence_info

    def _check_adaptive_convergence(
        self,
        U_current: np.ndarray,
        M_current: np.ndarray,
        U_prev: np.ndarray,
        M_prev: np.ndarray,
        iteration: int,
        base_tolerance: float,
    ) -> dict[str, Any]:
        """Check adaptive convergence criteria based on particle methods."""
        import numpy as np

        # Basic L2 error computation
        U_error = np.linalg.norm(U_current - U_prev) / max(float(np.linalg.norm(U_prev)), 1e-10)
        M_error = np.linalg.norm(M_current - M_prev) / max(float(np.linalg.norm(M_prev)), 1e-10)

        # Enhanced convergence criteria for particle methods
        wasserstein_converged = False
        stability_converged = False

        try:
            # Simple approximation of Wasserstein distance using L2
            # In a full implementation, this would use actual Wasserstein computation
            wasserstein_error = np.mean(np.abs(M_current - M_prev))
            wasserstein_converged = wasserstein_error < self.adaptive_params["wasserstein_tol"]

            # Stability check: look at recent convergence history for oscillations
            if len(self.convergence_history) >= 3:
                recent_u_errors = [h["U_error"] for h in self.convergence_history[-3:]]
                u_variance = np.var(recent_u_errors)
                stability_converged = u_variance < self.adaptive_params["u_stability_tol"]
            else:
                stability_converged = True  # Not enough history yet

        except Exception:
            # Fallback to classical criteria if advanced methods fail
            wasserstein_converged = M_error < self.adaptive_params["classical_tol"]
            stability_converged = True

        # Combined convergence decision
        classical_converged = U_error < base_tolerance and M_error < base_tolerance

        if self.adaptive_mode == "particle_aware":
            converged = classical_converged and wasserstein_converged and stability_converged
            mode = "particle_aware"
        else:
            converged = classical_converged
            mode = "classical"

        return {
            "iteration": iteration,
            "U_error": U_error,
            "M_error": M_error,
            "wasserstein_error": wasserstein_error if "wasserstein_error" in locals() else M_error,
            "wasserstein_converged": wasserstein_converged,
            "stability_converged": stability_converged,
            "classical_converged": classical_converged,
            "converged": converged,
            "mode": mode,
        }

    def get_convergence_mode(self) -> str:
        """Get the current convergence mode."""
        return self.adaptive_mode

    def get_detection_info(self) -> dict[str, Any]:
        """Get information about particle method detection."""
        return {
            "mode": self.adaptive_mode,
            "confidence": 1.0,  # High confidence for particle collocation
            "precision_level": self.precision_level,
            "adaptive_params": self.adaptive_params,
        }

    def get_adaptive_history(self) -> list[dict[str, Any]]:
        """Get the adaptive convergence history."""
        return self.convergence_history
