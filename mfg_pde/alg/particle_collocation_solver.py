from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .base_mfg_solver import MFGSolver
from .fp_solvers.fp_particle import FPParticleSolver
from .hjb_solvers.hjb_gfdm import HJBGFDMSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class ParticleCollocationSolver(MFGSolver):
    """
    Particle-Collocation solver for Mean Field Games.

    This solver combines:
    - Particle method for Fokker-Planck equations
    - GFDM collocation method for Hamilton-Jacobi-Bellman equations

    The particle-collocation framework provides:
    1. Natural handling of irregular particle distributions
    2. Meshfree spatial discretization
    3. Flexibility in choosing collocation points
    4. Excellent conservation properties from particles
    """

    def __init__(
        self,
        problem: "MFGProblem",
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
        boundary_indices: Optional[np.ndarray] = None,
        boundary_conditions: Optional[Dict] = None,
        use_monotone_constraints: bool = False,
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
        """
        super().__init__(problem)

        # Store solver parameters
        self.collocation_points = collocation_points
        self.num_particles = num_particles

        # Initialize FP solver (Particle method)
        # Use same boundary conditions for particles as for HJB
        self.fp_solver = FPParticleSolver(
            problem=problem,
            num_particles=num_particles,
            kde_bandwidth=kde_bandwidth,
            normalize_kde_output=normalize_kde_output,
            boundary_conditions=boundary_conditions,
        )

        # Initialize HJB solver (GFDM collocation)
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

        # Storage for results
        self.U_solution = None
        self.M_solution = None
        self.convergence_history = []
        self.particles_trajectory = None

    def solve(
        self,
        max_iterations: int = None,
        tolerance: float = None,
        # Deprecated parameters for backward compatibility
        Niter: int = None,
        l2errBound: float = None,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve the MFG system using particle-collocation method.

        Args:
            max_iterations: Maximum number of Picard iterations
            tolerance: Convergence tolerance for Picard iteration
            Niter: (Deprecated) Use max_iterations instead
            l2errBound: (Deprecated) Use tolerance instead
            verbose: Whether to print convergence information
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
            print(f"Starting Particle-Collocation MFG solver:")
            print(f"  - Particles: {self.num_particles}")
            print(f"  - Collocation points: {self.hjb_solver.n_points}")
            print(f"  - Max Picard iterations: {final_max_iterations}")
            print(f"  - Convergence tolerance: {final_tolerance}")

        # Get problem dimensions
        Nt = self.problem.Nt + 1
        Nx = self.problem.Nx + 1

        # Try warm start initialization first
        warm_start_init = self._get_warm_start_initialization()
        if warm_start_init is not None:
            U_current, M_current = warm_start_init
            print(f"   🚀 Using warm start initialization from previous solution")
        else:
            # Cold start - better initialization: set initial guess everywhere (like other MFG solvers)
            U_current = np.zeros((Nt, Nx))
            M_current = np.zeros((Nt, Nx))

        # Get terminal condition for U
        if hasattr(self.problem, "get_terminal_condition"):
            terminal_condition = self.problem.get_terminal_condition()
        else:
            # Default terminal condition
            terminal_condition = np.zeros(Nx)

        # Get initial density for M
        if hasattr(self.problem, "get_initial_density"):
            initial_density = self.problem.get_initial_density()
        else:
            # Default uniform initial density
            if self.problem.Dx > 1e-14:
                initial_density = np.ones(Nx) / self.problem.Lx
            else:
                initial_density = np.ones(Nx)

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
                    print(
                        f"  Warning: HJB solver failed at iteration {picard_iter}: {e}"
                    )
                U_new = U_current.copy()

            # Step 2: Solve FP equation with updated control
            try:
                M_new = self.fp_solver.solve_fp_system(
                    m_initial_condition=M_current[0, :], U_solution_for_drift=U_new
                )
            except Exception as e:
                if verbose:
                    print(
                        f"  Warning: FP solver failed at iteration {picard_iter}: {e}"
                    )
                M_new = M_current.copy()

            # Update solutions
            U_current = U_new
            M_current = M_new

            # Compute convergence metrics
            U_error = np.linalg.norm(U_current - U_prev) / max(
                np.linalg.norm(U_prev), 1e-10
            )
            M_error = np.linalg.norm(M_current - M_prev) / max(
                np.linalg.norm(M_prev), 1e-10
            )
            total_error = max(U_error, M_error)

            convergence_info = {
                "iteration": picard_iter,
                "U_error": U_error,
                "M_error": M_error,
                "total_error": total_error,
            }
            convergence_history.append(convergence_info)

            if verbose and picard_iter % 10 == 0:
                print(f"    U error: {U_error:.2e}, M error: {M_error:.2e}")

            # Check convergence
            if total_error < final_tolerance:
                if verbose:
                    print(f"  Converged at iteration {picard_iter}")
                    print(f"    Final U error: {U_error:.2e}")
                    print(f"    Final M error: {M_error:.2e}")
                break

        # Store results
        self.U_solution = U_current
        self.M_solution = M_current
        self.convergence_history = convergence_history

        # Store particle trajectory if available
        if hasattr(self.fp_solver, "M_particles_trajectory"):
            self.particles_trajectory = self.fp_solver.M_particles_trajectory

        # Prepare convergence info
        final_convergence_info = {
            "converged": total_error < final_tolerance,
            "final_error": total_error,
            "iterations": len(convergence_history),
            "history": convergence_history,
        }

        if verbose:
            print(f"Particle-Collocation solver completed:")
            print(f"  - Converged: {final_convergence_info['converged']}")
            print(f"  - Final error: {final_convergence_info['final_error']:.2e}")
            print(f"  - Total iterations: {final_convergence_info['iterations']}")

        # Store solutions and mark as computed for warm start capability
        self.U_solution = U_current
        self.M_solution = M_current
        self._solution_computed = True

        return U_current, M_current, final_convergence_info

    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the computed U and M solutions.

        Returns:
            Tuple of (U_solution, M_solution)
        """
        if self.U_solution is None or self.M_solution is None:
            raise ValueError("No solution available. Call solve() first.")

        return self.U_solution, self.M_solution

    def get_particles_trajectory(self) -> Optional[np.ndarray]:
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

    def get_collocation_info(self) -> Dict:
        """
        Get information about the collocation structure.

        Returns:
            Dictionary with collocation information
        """
        hjb_solver = self.hjb_solver

        # Count valid Taylor matrices
        valid_matrices = sum(
            1
            for i in range(hjb_solver.n_points)
            if hjb_solver.taylor_matrices[i] is not None
        )

        # Compute neighborhood statistics
        neighborhood_sizes = [
            hjb_solver.neighborhoods[i]["size"] for i in range(hjb_solver.n_points)
        ]

        info = {
            "n_collocation_points": hjb_solver.n_points,
            "dimension": hjb_solver.dimension,
            "delta": hjb_solver.delta,
            "taylor_order": hjb_solver.taylor_order,
            "weight_function": hjb_solver.weight_function,
            "valid_taylor_matrices": valid_matrices,
            "min_neighborhood_size": min(neighborhood_sizes),
            "max_neighborhood_size": max(neighborhood_sizes),
            "avg_neighborhood_size": np.mean(neighborhood_sizes),
            "multi_indices": hjb_solver.multi_indices,
            "n_derivatives": hjb_solver.n_derivatives,
        }

        return info

    def get_solver_info(self) -> Dict:
        """
        Get comprehensive information about both solvers.

        Returns:
            Dictionary with solver information
        """
        info = {
            "method": "Particle-Collocation",
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
