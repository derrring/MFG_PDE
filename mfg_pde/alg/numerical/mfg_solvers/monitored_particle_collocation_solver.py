"""
Enhanced Particle Collocation Solver with Advanced Convergence Monitoring.

This solver extends the standard ParticleCollocationSolver with robust convergence
criteria specifically designed for particle-based MFG methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .particle_collocation_solver import ParticleCollocationSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions


class MonitoredParticleCollocationSolver(ParticleCollocationSolver):
    """
    Particle Collocation Solver with advanced convergence monitoring.

    Features:
    - Robust convergence criteria for noisy particle distributions
    - Oscillation stabilization detection for value functions
    - Wasserstein distance-based distribution comparison
    - Multi-criteria convergence validation
    - Detailed convergence diagnostics and visualization
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
        boundary_conditions: BoundaryConditions | None = None,
        use_monotone_constraints: bool = False,
        convergence_monitor: Any | None = None,
        **convergence_kwargs: Any,
    ):
        """
        Initialize enhanced particle collocation solver.

        Args:
            convergence_monitor: Pre-configured convergence monitor
            **convergence_kwargs: Parameters for default monitor if none provided
            (All other args same as ParticleCollocationSolver)
        """
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

        # Initialize convergence monitoring
        if convergence_monitor is not None:
            self.convergence_monitor = convergence_monitor
        else:
            from mfg_pde.utils.convergence import create_default_monitor

            self.convergence_monitor = create_default_monitor(**convergence_kwargs)

        # Enhanced convergence tracking
        self.detailed_convergence_history: list[dict[str, Any]] = []
        self.use_advanced_convergence = True

    def solve(  # type: ignore[override]
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
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Solve the MFG system with enhanced convergence monitoring.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            Niter: (Deprecated) Use max_iterations instead
            l2errBound: (Deprecated) Use tolerance instead
            verbose: Whether to print convergence information
            plot_convergence: Whether to plot convergence diagnostics
            save_convergence_plot: Optional file path to save convergence plot
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (U_solution, M_solution, enhanced_convergence_info)
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
            final_tolerance = 1e-3  # Default

        if verbose:
            print("Starting Enhanced Particle-Collocation MFG solver with monitoring:")
            print(f"  - Particles: {self.num_particles}")
            print(f"  - Collocation points: {self.hjb_solver.n_points}")
            print(f"  - Max iterations: {final_max_iterations}")
            print(f"  - Convergence tolerance: {final_tolerance}")
            print(f"  - Enhanced monitoring: {self.use_advanced_convergence}")

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

        # Enhanced Picard iteration with monitoring
        self.detailed_convergence_history = []

        for picard_iter in range(final_max_iterations):
            if verbose and picard_iter % 10 == 0:
                print(f"  Enhanced Picard iteration {picard_iter}")

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

            if verbose and picard_iter % 10 == 0:
                print(f"    Enhanced errors: U={detailed_info['U_error']:.2e}, M={detailed_info['M_error']:.2e}")

            # Check enhanced convergence criteria
            if convergence_data.get("converged", False):
                if verbose:
                    print(f"  Enhanced convergence achieved at iteration {picard_iter}")
                    print(f"    Final U error: {detailed_info['U_error']:.2e}")
                    print(f"    Final M error: {detailed_info['M_error']:.2e}")
                break

        # Store results
        self.U_solution = U_current  # type: ignore[assignment]
        self.M_solution = M_current  # type: ignore[assignment]

        # Prepare enhanced convergence info
        enhanced_convergence_info = {
            "converged": convergence_data.get("converged", False),
            "final_error": max(detailed_info["U_error"], detailed_info["M_error"]),
            "iterations": len(self.detailed_convergence_history),
            "detailed_history": self.detailed_convergence_history,
            "convergence_monitor": self.convergence_monitor.get_summary()
            if hasattr(self.convergence_monitor, "get_summary")
            else {},
        }

        # Optional convergence plotting
        if plot_convergence:
            self._plot_enhanced_convergence(save_convergence_plot)

        if verbose:
            print("Enhanced Particle-Collocation solver completed:")
            print(f"  - Converged: {enhanced_convergence_info['converged']}")
            print(f"  - Final error: {enhanced_convergence_info['final_error']:.2e}")
            print(f"  - Total iterations: {enhanced_convergence_info['iterations']}")

        # Mark solution as computed for warm start capability
        self._solution_computed = True

        return U_current, M_current, enhanced_convergence_info

    def _plot_enhanced_convergence(self, save_path: str | None = None) -> None:
        """Plot enhanced convergence diagnostics."""
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

    def get_enhanced_convergence_history(self) -> list[dict[str, Any]]:
        """Get the detailed convergence history."""
        return self.detailed_convergence_history

    def get_convergence_monitor(self) -> Any:
        """Get the convergence monitor object."""
        return self.convergence_monitor
