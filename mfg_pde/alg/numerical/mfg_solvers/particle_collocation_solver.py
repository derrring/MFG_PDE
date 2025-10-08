"""
Unified Particle-Collocation solver for Mean Field Games.

This solver combines particle methods for Fokker-Planck equations with
generalized finite difference (GFDM) collocation for Hamilton-Jacobi-Bellman equations.

Supports optional advanced convergence monitoring for robust particle-based MFG methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base_mfg import BaseMFGSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class ParticleCollocationSolver(BaseMFGSolver):
    """
    Unified Particle-Collocation solver for Mean Field Games.

    This solver combines:
    - Particle method for Fokker-Planck equations
    - GFDM collocation method for Hamilton-Jacobi-Bellman equations

    The particle-collocation framework provides:
    1. Natural handling of irregular particle distributions
    2. Meshfree spatial discretization
    3. Flexibility in choosing collocation points
    4. Excellent conservation properties from particles

    Optional Features:
    - Advanced convergence monitoring (use_advanced_convergence=True)
    - Oscillation stabilization detection
    - Wasserstein distance-based distribution comparison
    - Multi-criteria convergence validation
    """

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        num_particles: int = 5000,
        delta: float = 0.1,
        taylor_order: int = 2,
        weight_function: str = "gaussian",
        weight_scale: float = 1.0,
        max_newton_iterations: int = 30,
        newton_tolerance: float = 1e-6,
        kde_bandwidth: str = "scott",
        normalize_kde_output: bool = True,
        boundary_indices: np.ndarray | None = None,
        boundary_conditions: dict | None = None,
        use_monotone_constraints: bool = False,
        use_advanced_convergence: bool = False,
        convergence_monitor: Any | None = None,
        **convergence_kwargs: Any,
    ):
        """
        Initialize the Particle-Collocation solver.

        Args:
            problem: MFG problem instance
            collocation_points: (N_points, d) array of collocation points for HJB
            num_particles: Number of particles for FP solver
            delta: Neighborhood radius for GFDM collocation
            taylor_order: Order of Taylor expansion for GFDM
            weight_function: Weight function for GFDM ("gaussian", "inverse_distance", "uniform")
            weight_scale: Scale parameter for weight function
            max_newton_iterations: Maximum Newton iterations for HJB
            newton_tolerance: Newton convergence tolerance for HJB
            kde_bandwidth: Bandwidth method for KDE in particle method
            normalize_kde_output: Whether to normalize KDE output
            boundary_indices: Indices of boundary collocation points
            boundary_conditions: Dictionary specifying boundary conditions
            use_monotone_constraints: Enable constrained QP for HJB monotonicity
            use_advanced_convergence: Enable advanced convergence monitoring
            convergence_monitor: Pre-configured convergence monitor (optional)
            **convergence_kwargs: Parameters for default monitor if none provided
        """
        super().__init__(problem)

        # Store solver parameters
        self.collocation_points = collocation_points
        self.num_particles = num_particles
        self.use_advanced_convergence = use_advanced_convergence

        # Initialize FP solver (Particle method)
        # Use same boundary conditions for particles as for HJB
        from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

        self.fp_solver = FPParticleSolver(
            problem=problem,
            num_particles=num_particles,
            kde_bandwidth=kde_bandwidth,
            normalize_kde_output=normalize_kde_output,
            boundary_conditions=boundary_conditions,  # type: ignore[arg-type]
        )

        # Initialize HJB solver (GFDM collocation)
        from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

        self.hjb_solver = HJBGFDMSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=delta,
            taylor_order=taylor_order,
            weight_function=weight_function,
            weight_scale=weight_scale,
            max_newton_iterations=max_newton_iterations,
            newton_tolerance=newton_tolerance,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
            use_monotone_constraints=use_monotone_constraints,
        )

        # Initialize convergence monitoring (if enabled)
        if self.use_advanced_convergence:
            if convergence_monitor is not None:
                self.convergence_monitor = convergence_monitor
            else:
                from mfg_pde.utils.numerical.convergence import create_default_monitor

                self.convergence_monitor = create_default_monitor(**convergence_kwargs)
            self.detailed_convergence_history: list[dict[str, Any]] = []
        else:
            self.convergence_monitor = None
            self.detailed_convergence_history = []

        # Storage for results
        self.U_solution = None
        self.M_solution = None
        self.convergence_history: list[dict[str, float]] = []
        self.particles_trajectory = None

    def solve(
        self,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        # Deprecated parameters for backward compatibility
        Niter: int | None = None,
        l2errBound: float | None = None,
        verbose: bool = True,
        plot_convergence: bool = False,
        save_convergence_plot: str | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Solve the MFG system using particle-collocation method.

        Args:
            max_iterations: Maximum number of Picard iterations
            tolerance: Convergence tolerance for Picard iteration
            Niter: (Deprecated) Use max_iterations instead
            l2errBound: (Deprecated) Use tolerance instead
            verbose: Whether to print convergence information
            plot_convergence: Whether to plot convergence diagnostics (requires use_advanced_convergence=True)
            save_convergence_plot: Optional file path to save convergence plot
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (U_solution, M_solution, convergence_info)
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
            final_tolerance = tolerance
        elif l2errBound is not None:
            warnings.warn(
                "Parameter 'l2errBound' is deprecated. Use 'tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_tolerance = l2errBound
        else:
            final_tolerance = 1e-6  # Default
        if verbose:
            solver_mode = "Enhanced" if self.use_advanced_convergence else "Standard"
            print(f"Starting {solver_mode} Particle-Collocation MFG solver:")
            print(f"  - Particles: {self.num_particles}")
            print(f"  - Collocation points: {self.hjb_solver.n_points}")
            print(f"  - Max Picard iterations: {final_max_iterations}")
            print(f"  - Convergence tolerance: {final_tolerance}")
            if self.use_advanced_convergence:
                print("  - Advanced monitoring: enabled")

        # Get problem dimensions
        Nt = self.problem.Nt + 1
        Nx = self.problem.Nx + 1

        # Try warm start initialization first
        warm_start_init = self._get_warm_start_initialization()
        if warm_start_init is not None:
            U_current, M_current = warm_start_init
            print("   Using warm start initialization from previous solution")
        else:
            # Cold start - better initialization: set initial guess everywhere (like other MFG solvers)
            U_current = np.zeros((Nt, Nx))
            M_current = np.zeros((Nt, Nx))

        # Get terminal condition for U
        if hasattr(self.problem, "get_final_u"):
            terminal_condition = self.problem.get_final_u()
        else:
            # Default terminal condition
            terminal_condition = np.zeros(Nx)

        # Get initial density for M
        if hasattr(self.problem, "get_initial_m"):
            initial_density = self.problem.get_initial_m()
        else:
            # Default uniform initial density
            initial_density = np.ones(Nx) / self.problem.Lx if self.problem.Dx > 1e-14 else np.ones(Nx)

        # For cold start, initialize interior with boundary conditions
        if warm_start_init is None:
            # Initialize U everywhere with terminal condition (better initial guess)
            for t in range(Nt):
                U_current[t, :] = terminal_condition

            # Initialize M everywhere with initial density (better initial guess)
            for t in range(Nt):
                M_current[t, :] = initial_density

        # Always enforce boundary conditions (even with warm start)
        U_current[Nt - 1, :] = terminal_condition  # Terminal condition
        M_current[0, :] = initial_density  # Initial condition

        # Picard iteration
        convergence_history = []

        for picard_iter in range(final_max_iterations):
            if verbose and picard_iter % 10 == 0:
                print(f"  Picard iteration {picard_iter}")

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

            # Compute convergence metrics (basic or advanced)
            if self.use_advanced_convergence:
                # Enhanced convergence monitoring
                convergence_data = self.convergence_monitor.check_convergence(
                    U_current, M_current, U_prev, M_prev, picard_iter
                )

                # Store detailed convergence information
                detailed_info = {
                    "iteration": picard_iter,
                    "convergence_data": convergence_data,
                    "U_norm": float(np.linalg.norm(U_current)),
                    "M_norm": float(np.linalg.norm(M_current)),
                    "U_error": convergence_data.get("U_error", 0.0),
                    "M_error": convergence_data.get("M_error", 0.0),
                }
                self.detailed_convergence_history.append(detailed_info)

                # Also maintain basic history for compatibility
                convergence_info = {
                    "iteration": picard_iter,
                    "U_error": detailed_info["U_error"],
                    "M_error": detailed_info["M_error"],
                    "total_error": max(detailed_info["U_error"], detailed_info["M_error"]),
                }
                convergence_history.append(convergence_info)

                U_error = detailed_info["U_error"]
                M_error = detailed_info["M_error"]
                total_error = convergence_info["total_error"]
                converged = convergence_data.get("converged", False)

                if verbose and picard_iter % 10 == 0:
                    print(f"    Enhanced errors: U={U_error:.2e}, M={M_error:.2e}")
            else:
                # Basic convergence metrics
                U_error = np.linalg.norm(U_current - U_prev) / max(float(np.linalg.norm(U_prev)), 1e-10)
                M_error = np.linalg.norm(M_current - M_prev) / max(float(np.linalg.norm(M_prev)), 1e-10)
                total_error = max(float(U_error), float(M_error))

                convergence_info = {
                    "iteration": picard_iter,
                    "U_error": U_error,
                    "M_error": M_error,
                    "total_error": total_error,
                }
                convergence_history.append(convergence_info)

                converged = total_error < final_tolerance

                if verbose and picard_iter % 10 == 0:
                    print(f"    U error: {U_error:.2e}, M error: {M_error:.2e}")

            # Check convergence
            if converged:
                if verbose:
                    mode_str = "Enhanced convergence" if self.use_advanced_convergence else "Converged"
                    print(f"  {mode_str} achieved at iteration {picard_iter}")
                    print(f"    Final U error: {U_error:.2e}")
                    print(f"    Final M error: {M_error:.2e}")
                break

        # Store results
        self.U_solution = U_current  # type: ignore[assignment]
        self.M_solution = M_current  # type: ignore[assignment]
        self.convergence_history = convergence_history  # type: ignore[assignment]

        # Store particle trajectory if available
        if hasattr(self.fp_solver, "M_particles_trajectory"):
            self.particles_trajectory = self.fp_solver.M_particles_trajectory  # type: ignore[assignment]

        # Prepare convergence info (basic or enhanced)
        if self.use_advanced_convergence:
            final_convergence_info = {
                "converged": converged,
                "final_error": total_error,
                "iterations": len(convergence_history),
                "history": convergence_history,
                "detailed_history": self.detailed_convergence_history,
                "convergence_monitor": self.convergence_monitor.get_summary()
                if hasattr(self.convergence_monitor, "get_summary")
                else {},
            }
        else:
            final_convergence_info = {
                "converged": total_error < final_tolerance,
                "final_error": total_error,
                "iterations": len(convergence_history),
                "history": convergence_history,
            }

        # Optional convergence plotting (only for advanced mode)
        if plot_convergence and self.use_advanced_convergence:
            self._plot_enhanced_convergence(save_convergence_plot)

        if verbose:
            mode_str = "Enhanced " if self.use_advanced_convergence else ""
            print(f"{mode_str}Particle-Collocation solver completed:")
            print(f"  - Converged: {final_convergence_info['converged']}")
            print(f"  - Final error: {final_convergence_info['final_error']:.2e}")
            print(f"  - Total iterations: {final_convergence_info['iterations']}")

        # Store solutions and mark as computed for warm start capability
        self.U_solution = U_current  # type: ignore[assignment]
        self.M_solution = M_current  # type: ignore[assignment]
        self._solution_computed = True

        return U_current, M_current, final_convergence_info

    def _get_warm_start_initialization(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get warm start initialization data."""
        return self.get_warm_start_data()

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the computed U and M solutions.

        Returns:
            Tuple of (U_solution, M_solution)
        """
        if self.U_solution is None or self.M_solution is None:
            raise ValueError("No solution available. Call solve() first.")

        return self.U_solution, self.M_solution

    def get_particles_trajectory(self) -> np.ndarray | None:
        """
        Get the particle trajectory from the FP solver.

        Returns:
            Particle trajectory array (Nt, num_particles) or None
        """
        return self.particles_trajectory

    def get_convergence_history(self) -> list:
        """
        Get the convergence history.

        Returns:
            List of convergence information dictionaries
        """
        return self.convergence_history

    def get_collocation_info(self) -> dict:
        """
        Get information about the collocation structure.

        Returns:
            Dictionary with collocation information
        """
        hjb_solver = self.hjb_solver

        # Count valid Taylor matrices
        valid_matrices = sum(1 for i in range(hjb_solver.n_points) if hjb_solver.taylor_matrices[i] is not None)

        # Compute neighborhood statistics
        neighborhood_sizes = [hjb_solver.neighborhoods[i]["size"] for i in range(hjb_solver.n_points)]

        info = {
            "n_collocation_points": hjb_solver.n_points,
            "dimension": hjb_solver.dimension,
            "delta": hjb_solver.delta,
            "taylor_order": hjb_solver.taylor_order,
            "weight_function": hjb_solver.weight_function,
            "valid_taylor_matrices": valid_matrices,
            "min_neighborhood_size": min(neighborhood_sizes),  # type: ignore[type-var]
            "max_neighborhood_size": max(neighborhood_sizes),  # type: ignore[type-var]
            "avg_neighborhood_size": np.mean(neighborhood_sizes),  # type: ignore[arg-type]
            "multi_indices": hjb_solver.multi_indices,
            "n_derivatives": hjb_solver.n_derivatives,
        }

        return info

    def get_solver_info(self) -> dict:
        """
        Get comprehensive information about both solvers.

        Returns:
            Dictionary with solver information
        """
        info = {
            "method": "Particle-Collocation",
            "advanced_convergence": self.use_advanced_convergence,
            "fp_solver": {
                "method": self.fp_solver.fp_method_name,
                "num_particles": self.fp_solver.num_particles,
                "kde_bandwidth": self.fp_solver.kde_bandwidth,
                "normalize_kde": self.fp_solver.normalize_kde_output,
            },
            "hjb_solver": {
                "method": self.hjb_solver.hjb_method_name,
                **self.get_collocation_info(),
            },
        }

        return info

    def get_enhanced_convergence_history(self) -> list[dict[str, Any]]:
        """
        Get the detailed convergence history (only available if use_advanced_convergence=True).

        Returns:
            List of detailed convergence information dictionaries
        """
        if not self.use_advanced_convergence:
            raise ValueError("Enhanced convergence history only available with use_advanced_convergence=True")
        return self.detailed_convergence_history

    def get_convergence_monitor(self) -> Any:
        """
        Get the convergence monitor object (only available if use_advanced_convergence=True).

        Returns:
            Convergence monitor instance
        """
        if not self.use_advanced_convergence:
            raise ValueError("Convergence monitor only available with use_advanced_convergence=True")
        return self.convergence_monitor

    def _plot_enhanced_convergence(self, save_path: str | None = None) -> None:
        """Plot enhanced convergence diagnostics (only for advanced mode)."""
        if not self.use_advanced_convergence:
            print("  Warning: Enhanced convergence plotting requires use_advanced_convergence=True")
            return

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Enhanced Convergence Monitoring", fontsize=14)

            # Extract convergence data
            iterations = [h["iteration"] for h in self.detailed_convergence_history]
            u_errors = [h["U_error"] for h in self.detailed_convergence_history]
            m_errors = [h["M_error"] for h in self.detailed_convergence_history]
            u_norms = [h["U_norm"] for h in self.detailed_convergence_history]
            m_norms = [h["M_norm"] for h in self.detailed_convergence_history]

            # Plot 1: Error evolution
            axes[0, 0].semilogy(iterations, u_errors, "b-", label="U error")
            axes[0, 0].semilogy(iterations, m_errors, "r-", label="M error")
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].set_ylabel("Error")
            axes[0, 0].set_title("Convergence Error Evolution")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Plot 2: Solution norm evolution
            axes[0, 1].plot(iterations, u_norms, "b-", label="||U||")
            axes[0, 1].plot(iterations, m_norms, "r-", label="||M||")
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Norm")
            axes[0, 1].set_title("Solution Norm Evolution")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # Plot 3: Error ratio (relative change)
            if len(u_errors) > 1:
                u_ratios = [
                    u_errors[i] / u_errors[i - 1] if u_errors[i - 1] > 1e-15 else 1 for i in range(1, len(u_errors))
                ]
                m_ratios = [
                    m_errors[i] / m_errors[i - 1] if m_errors[i - 1] > 1e-15 else 1 for i in range(1, len(m_errors))
                ]
                axes[1, 0].plot(iterations[1:], u_ratios, "b-", label="U ratio")
                axes[1, 0].plot(iterations[1:], m_ratios, "r-", label="M ratio")
                axes[1, 0].axhline(y=1, color="k", linestyle="--", alpha=0.5)
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Error Ratio")
            axes[1, 0].set_title("Convergence Rate Analysis")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # Plot 4: Combined error
            combined_errors = [max(u_errors[i], m_errors[i]) for i in range(len(u_errors))]
            axes[1, 1].semilogy(iterations, combined_errors, "g-", linewidth=2)
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Max Error")
            axes[1, 1].set_title("Combined Error Evolution")
            axes[1, 1].grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"  Convergence plot saved to: {save_path}")

            plt.show()

        except ImportError:
            print("  Warning: matplotlib not available for convergence plotting")
        except Exception as e:
            print(f"  Warning: Failed to plot convergence: {e}")
