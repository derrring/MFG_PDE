"""
Finite Difference Method (FDM) for HJB Equation - All Dimensions.

Supports:
    - 1D: Optimized Newton solver from base_hjb
    - 2D/3D/nD: Uses centralized nonlinear solvers

References:
    - Evans (2010): Partial Differential Equations, Ch. 10
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: numerical methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfg_pde.utils.numerical import FixedPointSolver, NewtonSolver

from . import base_hjb
from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.core.mfg_problem import MFGProblem


class HJBFDMSolver(BaseHJBSolver):
    """
    Finite Difference Method solver for HJB equation (all dimensions).

    Automatically handles 1D, 2D, 3D, and higher-dimensional problems:
        - 1D: Uses optimized Newton solver from base_hjb
        - nD: Uses centralized FixedPointSolver or NewtonSolver

    Recommended: d ≤ 3 due to O(N^d) complexity
    """

    def __init__(
        self,
        problem: MFGProblem,
        solver_type: Literal["fixed_point", "newton"] = "newton",
        damping_factor: float = 1.0,
        max_newton_iterations: int | None = None,
        newton_tolerance: float | None = None,
        # Deprecated parameters
        NiterNewton: int | None = None,
        l2errBoundNewton: float | None = None,
        backend: str | None = None,
    ):
        """
        Initialize FDM solver.

        Args:
            problem: MFG problem (1D or GridBasedMFGProblem for nD)
            solver_type: 'fixed_point' or 'newton' (nD only, 1D always uses Newton)
            damping_factor: Damping ω ∈ (0,1] for fixed-point (recommend 0.5-0.8)
            max_newton_iterations: Max iterations per timestep
            newton_tolerance: Convergence tolerance
            backend: 'numpy', 'torch', or None
        """
        import warnings

        super().__init__(problem)

        # Initialize backend
        from mfg_pde.backends import create_backend

        self.backend = create_backend(backend or "numpy")

        # Handle deprecated parameters
        if NiterNewton is not None:
            warnings.warn(
                "Parameter 'NiterNewton' is deprecated. Use 'max_newton_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_newton_iterations = max_newton_iterations or NiterNewton

        if l2errBoundNewton is not None:
            warnings.warn(
                "Parameter 'l2errBoundNewton' is deprecated. Use 'newton_tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            newton_tolerance = newton_tolerance or l2errBoundNewton

        # Set defaults (use None check to avoid treating 0 as falsy)
        self.max_newton_iterations = max_newton_iterations if max_newton_iterations is not None else 30
        self.newton_tolerance = newton_tolerance if newton_tolerance is not None else 1e-6
        self.solver_type = solver_type
        self.damping_factor = damping_factor

        # Validate
        if self.max_newton_iterations < 1:
            raise ValueError(f"max_newton_iterations must be >= 1, got {self.max_newton_iterations}")
        if self.newton_tolerance <= 0:
            raise ValueError(f"newton_tolerance must be > 0, got {self.newton_tolerance}")
        if not 0 < damping_factor <= 1.0:
            raise ValueError(f"damping_factor must be in (0,1], got {damping_factor}")

        # Backward compatibility: Store Newton config
        self._newton_config = {
            "max_iterations": self.max_newton_iterations,
            "tolerance": self.newton_tolerance,
        }

        # Detect dimension
        self.dimension = self._detect_dimension(problem)
        # Backward compatibility: 1D uses "FDM", nD uses "FDM-{d}D-{solver_type}"
        if self.dimension == 1:
            self.hjb_method_name = "FDM"
        else:
            self.hjb_method_name = f"FDM-{self.dimension}D-{solver_type}"

        # For nD, extract grid info and create nonlinear solver
        if self.dimension > 1:
            if not (hasattr(problem, "geometry") and hasattr(problem.geometry, "grid")):
                raise ValueError("nD FDM requires GridBasedMFGProblem with TensorProductGrid")

            self.grid = problem.geometry.grid
            self.shape = tuple(self.grid.num_points)
            self.spacing = self.grid.spacing
            self.N_total = int(np.prod(self.shape))
            self.dt = problem.dt

            if self.dimension > 3:
                warnings.warn(
                    f"FDM solver in {self.dimension}D requires {self.N_total:,} grid points. "
                    f"Consider GFDM or sparse methods for d>3.",
                    UserWarning,
                    stacklevel=2,
                )

            # Create nonlinear solver
            if solver_type == "fixed_point":
                self.nonlinear_solver = FixedPointSolver(
                    damping_factor=damping_factor,
                    max_iterations=self.max_newton_iterations,
                    tolerance=self.newton_tolerance,
                )
            else:  # newton
                self.nonlinear_solver = NewtonSolver(
                    max_iterations=self.max_newton_iterations,
                    tolerance=self.newton_tolerance,
                    sparse=True,
                    jacobian=None,  # Use automatic finite differences
                )

    def _detect_dimension(self, problem) -> int:
        """Detect spatial dimension."""
        if hasattr(problem, "geometry") and hasattr(problem.geometry, "grid"):
            return getattr(problem.geometry.grid, "dimension", getattr(problem.geometry.grid, "ndim", 1))
        if hasattr(problem, "dimension"):
            return problem.dimension
        if hasattr(problem, "Nx") and not hasattr(problem, "Ny"):
            return 1
        raise ValueError("Cannot determine problem dimension")

    def solve_hjb_system(
        self,
        M_density_evolution: NDArray,
        U_final_condition: NDArray,
        U_from_prev_picard: NDArray,
    ) -> NDArray:
        """
        Solve HJB system backward in time.

        Automatically routes to 1D or nD solver based on dimension.
        """
        if self.dimension == 1:
            # Use optimized 1D solver
            return base_hjb.solve_hjb_system_backward(
                M_density_from_prev_picard=M_density_evolution,
                U_final_condition_at_T=U_final_condition,
                U_from_prev_picard=U_from_prev_picard,
                problem=self.problem,
                max_newton_iterations=self.max_newton_iterations,
                newton_tolerance=self.newton_tolerance,
                backend=self.backend,
            )
        else:
            # Use nD solver with centralized nonlinear solver
            return self._solve_hjb_nd(M_density_evolution, U_final_condition, U_from_prev_picard)

    def _solve_hjb_nd(
        self,
        M_density: NDArray,
        U_final: NDArray,
        U_prev: NDArray,
    ) -> NDArray:
        """Solve nD HJB using centralized nonlinear solvers."""
        # Validate shapes
        Nt = self.problem.Nt + 1
        expected_shape = (Nt, *self.shape)
        if M_density.shape != expected_shape:
            raise ValueError(f"M_density shape {M_density.shape} != {expected_shape}")
        if U_final.shape != self.shape:
            raise ValueError(f"U_final shape {U_final.shape} != {self.shape}")

        # Allocate solution
        U_solution = np.zeros(expected_shape, dtype=np.float64)
        U_solution[Nt - 1] = U_final.copy()

        if Nt <= 1:
            return U_solution

        # Progress bar
        from mfg_pde.utils.progress import tqdm

        timestep_range = tqdm(
            range(Nt - 2, -1, -1),
            desc=f"HJB {self.dimension}D-FDM ({self.solver_type})",
            unit="step",
        )

        # Backward time loop
        for n in timestep_range:
            U_next = U_solution[n + 1]
            M_next = M_density[n + 1]
            U_guess = U_prev[n]

            # Solve nonlinear system using centralized solver
            U_solution[n] = self._solve_single_timestep(U_next, M_next, U_guess)

        return U_solution

    def _solve_single_timestep(self, U_next: NDArray, M_next: NDArray, U_guess: NDArray) -> NDArray:
        """
        Solve single HJB timestep using centralized nonlinear solver.

        For fixed-point: Solves u = G(u) where G(u) = u_next - dt·H(∇u, m)
        For Newton: Solves F(u) = 0 where F(u) = (u - u_next)/dt + H(∇u, m)
        """
        if self.solver_type == "fixed_point":
            # Define fixed-point map G: u → u
            def G(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients)
                return U_next - self.dt * H_values

            U_solution, info = self.nonlinear_solver.solve(G, U_guess)

        else:  # newton
            # Define residual F: u → residual
            def F(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients)
                return (U - U_next) / self.dt + H_values

            U_solution, info = self.nonlinear_solver.solve(F, U_guess)

        # Warn if not converged
        if not info.converged:
            import warnings

            warnings.warn(
                f"{self.solver_type} did not converge (residual: {info.residual:.2e})",
                UserWarning,
                stacklevel=2,
            )

        return U_solution

    def _compute_gradients_nd(self, U: NDArray) -> dict[tuple, NDArray]:
        """Compute gradients using central differences."""
        gradients = {(0,) * self.dimension: U}

        for d in range(self.dimension):
            h = self.spacing[d]
            slices = [slice(None)] * self.dimension

            # Interior: central difference
            slices[d] = slice(1, -1)
            slices_fwd = slices.copy()
            slices_fwd[d] = slice(2, None)
            slices_bwd = slices.copy()
            slices_bwd[d] = slice(None, -2)

            grad_interior = (U[tuple(slices_fwd)] - U[tuple(slices_bwd)]) / (2 * h)

            # Boundaries: one-sided differences
            slices[d] = slice(0, 1)
            slices_fwd[d] = slice(1, 2)
            grad_left = (U[tuple(slices_fwd)] - U[tuple(slices)]) / h

            slices[d] = slice(-1, None)
            slices_bwd[d] = slice(-2, -1)
            grad_right = (U[tuple(slices)] - U[tuple(slices_bwd)]) / h

            # Concatenate
            grad_d = np.concatenate([grad_left, grad_interior, grad_right], axis=d)

            # Store with multi-index key
            multi_idx = tuple(1 if i == d else 0 for i in range(self.dimension))
            gradients[multi_idx] = grad_d

        return gradients

    def _evaluate_hamiltonian_nd(self, U: NDArray, M: NDArray, gradients: dict) -> NDArray:
        """Evaluate Hamiltonian at all grid points."""
        H_values = np.zeros(self.shape, dtype=np.float64)

        for multi_idx in np.ndindex(self.shape):
            x_coords = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
            m_at_point = M[multi_idx]
            derivs_at_point = {key: grad_array[multi_idx] for key, grad_array in gradients.items()}

            # Call problem Hamiltonian (try both interfaces)
            if hasattr(self.problem, "hamiltonian"):
                p = np.array(
                    [
                        derivs_at_point.get(tuple(1 if i == d else 0 for i in range(self.dimension)), 0.0)
                        for d in range(self.dimension)
                    ]
                )
                H_values[multi_idx] = self.problem.hamiltonian(x_coords, m_at_point, p, t=0.0)
            elif hasattr(self.problem, "H"):
                H_values[multi_idx] = self.problem.H(multi_idx, m_at_point, derivs=derivs_at_point)
            else:
                raise AttributeError("Problem must have 'hamiltonian' or 'H' method")

        return H_values
