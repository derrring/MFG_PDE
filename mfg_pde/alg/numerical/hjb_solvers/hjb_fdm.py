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
from mfg_pde.utils.pde_coefficients import CoefficientField

from . import base_hjb
from .base_hjb import BaseHJBSolver


def is_diagonal_tensor(Sigma: NDArray, rtol: float = 1e-10) -> bool:
    """
    Check if tensor is diagonal (off-diagonal elements near zero).

    Args:
        Sigma: Tensor array, either (d, d) or (*shape, d, d)
        rtol: Relative tolerance for off-diagonal elements

    Returns:
        True if diagonal, False otherwise
    """
    # Handle both single tensor and spatially-varying tensors
    if Sigma.ndim == 2:
        # Single (d, d) tensor
        d = Sigma.shape[0]
        off_diag_sum = np.sum(np.abs(Sigma)) - np.sum(np.abs(np.diag(Sigma)))
        diag_sum = np.sum(np.abs(np.diag(Sigma)))
        return off_diag_sum < rtol * diag_sum if diag_sum > 0 else off_diag_sum < rtol
    else:
        # Spatially-varying (*shape, d, d)
        d = Sigma.shape[-1]
        diag_mask = np.eye(d, dtype=bool)
        off_diag_elements = Sigma[..., ~diag_mask]
        diag_elements = Sigma[..., diag_mask]
        off_diag_norm = np.linalg.norm(off_diag_elements)
        diag_norm = np.linalg.norm(diag_elements)
        return off_diag_norm < rtol * diag_norm if diag_norm > 0 else off_diag_norm < rtol


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
            # Import at runtime to avoid circular dependency
            from mfg_pde.geometry.base import CartesianGrid

            if not isinstance(problem.geometry, CartesianGrid):
                raise ValueError(
                    "nD FDM requires problem with CartesianGrid geometry (SimpleGrid2D/3D or TensorProductGrid)"
                )

            self.grid = problem.geometry  # Geometry IS the grid
            self.shape = tuple(self.grid.get_grid_shape())
            self.spacing = self.grid.get_grid_spacing()
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
        # Try geometry.dimension first (unified interface)
        if hasattr(problem, "geometry") and hasattr(problem.geometry, "dimension"):
            return problem.geometry.dimension
        # Fall back to problem.dimension
        if hasattr(problem, "dimension"):
            return problem.dimension
        # Legacy 1D detection
        if getattr(problem, "Nx", None) is not None and getattr(problem, "Ny", None) is None:
            return 1
        raise ValueError("Cannot determine problem dimension")

    def solve_hjb_system(
        self,
        M_density_evolution: NDArray,
        U_final_condition: NDArray,
        U_from_prev_picard: NDArray,
        diffusion_field: float | NDArray | None = None,
        tensor_diffusion_field: NDArray | None = None,
    ) -> NDArray:
        """
        Solve HJB system backward in time.

        Automatically routes to 1D or nD solver based on dimension.

        Args:
            M_density_evolution: Density field from FP solver
            U_final_condition: Terminal condition at T
            U_from_prev_picard: Previous Picard iterate
            diffusion_field: Diffusion coefficient (None uses problem.sigma)
            tensor_diffusion_field: Tensor diffusion (Phase 3.0, not yet fully implemented)
                Note: Currently a placeholder. Full tensor diffusion support in HJB
                requires problem Hamiltonians to handle tensor-valued diffusion.
        """
        # Validate mutual exclusivity
        if diffusion_field is not None and tensor_diffusion_field is not None:
            raise ValueError(
                "Cannot specify both diffusion_field and tensor_diffusion_field. "
                "Use diffusion_field for scalar or tensor_diffusion_field for anisotropic."
            )

        # Check tensor type and warn if non-diagonal (not fully implemented yet)
        if tensor_diffusion_field is not None:
            import warnings

            # Check if diagonal
            if callable(tensor_diffusion_field):
                # Cannot easily check callable tensors without evaluation
                warnings.warn(
                    "Callable tensor_diffusion_field in HJB solver is not yet fully implemented. "
                    "If the tensor is diagonal, it will be handled correctly. "
                    "Full tensor (non-diagonal) support requires Hamiltonian refactoring.",
                    UserWarning,
                    stacklevel=2,
                )
            elif not is_diagonal_tensor(tensor_diffusion_field):
                warnings.warn(
                    "Full tensor (non-diagonal) diffusion in HJB solver is not yet implemented. "
                    "The parameter is accepted for API compatibility but only diagonal tensors "
                    "are currently supported. Full tensor support requires Hamiltonian refactoring.",
                    UserWarning,
                    stacklevel=2,
                )

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
                diffusion_field=diffusion_field,
            )
        else:
            # Use nD solver with centralized nonlinear solver
            return self._solve_hjb_nd(
                M_density_evolution, U_final_condition, U_from_prev_picard, diffusion_field, tensor_diffusion_field
            )

    def _solve_hjb_nd(
        self,
        M_density: NDArray,
        U_final: NDArray,
        U_prev: NDArray,
        diffusion_field: float | NDArray | None = None,
        tensor_diffusion_field: NDArray | None = None,
    ) -> NDArray:
        """Solve nD HJB using centralized nonlinear solvers with variable diffusion support.

        Supports scalar, array, and callable diffusion coefficients.

        Note: tensor_diffusion_field is accepted but not yet fully implemented.
        """
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

            # Extract or evaluate diffusion using CoefficientField abstraction
            if tensor_diffusion_field is not None:
                # Evaluate tensor at current timestep
                if callable(tensor_diffusion_field):
                    # Callable: Σ(t, x, m)
                    t = n * self.problem.dt
                    Sigma_at_n = np.zeros((*self.shape, self.dimension, self.dimension))
                    for idx in np.ndindex(self.shape):
                        x_coords = np.array([self.grid.coordinates[d][idx[d]] for d in range(self.dimension)])
                        m_at_point = M_next[idx]
                        Sigma_at_n[idx] = tensor_diffusion_field(t, x_coords, m_at_point)
                else:
                    # Constant or spatially-varying
                    Sigma_at_n = tensor_diffusion_field
            else:
                Sigma_at_n = None

            # Handle scalar diffusion_field
            if diffusion_field is not None or Sigma_at_n is None:
                diffusion = CoefficientField(
                    diffusion_field, self.problem.sigma, "diffusion_field", dimension=self.dimension
                )
                sigma_at_n = diffusion.evaluate_at(
                    timestep_idx=n, grid=self.grid.coordinates, density=M_next, dt=self.problem.dt
                )
            else:
                sigma_at_n = None

            # Solve nonlinear system using centralized solver
            U_solution[n] = self._solve_single_timestep(U_next, M_next, U_guess, sigma_at_n, Sigma_at_n)

        return U_solution

    def _solve_single_timestep(
        self,
        U_next: NDArray,
        M_next: NDArray,
        U_guess: NDArray,
        sigma_at_n: float | NDArray | None = None,
        Sigma_at_n: NDArray | None = None,
    ) -> NDArray:
        """
        Solve single HJB timestep using centralized nonlinear solver.

        For fixed-point: Solves u = G(u) where G(u) = u_next - dt·H(∇u, m)
        For Newton: Solves F(u) = 0 where F(u) = (u - u_next)/dt + H(∇u, m)

        Args:
            U_next: Value function at next timestep
            M_next: Density at next timestep
            U_guess: Initial guess for current timestep
            sigma_at_n: Scalar diffusion coefficient at current timestep (None, float, or array)
            Sigma_at_n: Tensor diffusion coefficient at current timestep (None or tensor array)
        """
        if self.solver_type == "fixed_point":
            # Define fixed-point map G: u → u
            def G(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients, sigma_at_n, Sigma_at_n)
                return U_next - self.dt * H_values

            U_solution, info = self.nonlinear_solver.solve(G, U_guess)

        else:  # newton
            # Define residual F: u → residual
            def F(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients, sigma_at_n, Sigma_at_n)
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

    def _evaluate_hamiltonian_nd(
        self,
        U: NDArray,
        M: NDArray,
        gradients: dict,
        sigma_at_n: float | NDArray | None = None,
        Sigma_at_n: NDArray | None = None,
    ) -> NDArray:
        """Evaluate Hamiltonian at all grid points with variable diffusion support.

        Supports scalar and diagonal tensor diffusion.

        Args:
            U: Value function at current timestep
            M: Density at current timestep
            gradients: Dictionary of gradient arrays
            sigma_at_n: Scalar diffusion coefficient (None uses problem.sigma, float is constant, array is spatially varying)
            Sigma_at_n: Tensor diffusion coefficient (None or tensor array). If provided and diagonal, computes
                        H_viscosity = (1/2) Σᵢ σᵢ² pᵢ² separately and adds to running cost.
        """
        H_values = np.zeros(self.shape, dtype=np.float64)

        # Determine which diffusion mode to use
        use_tensor_diffusion = Sigma_at_n is not None

        if not use_tensor_diffusion:
            # Scalar diffusion mode
            if sigma_at_n is None:
                sigma_base = self.problem.sigma
            else:
                sigma_base = sigma_at_n

        for multi_idx in np.ndindex(self.shape):
            x_coords = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
            m_at_point = M[multi_idx]
            derivs_at_point = {key: grad_array[multi_idx] for key, grad_array in gradients.items()}

            # Extract gradient p
            p = np.array(
                [
                    derivs_at_point.get(tuple(1 if i == d else 0 for i in range(self.dimension)), 0.0)
                    for d in range(self.dimension)
                ]
            )

            if use_tensor_diffusion:
                # Tensor diffusion mode
                # Get tensor at this point
                if Sigma_at_n.ndim == 2:
                    # Constant tensor
                    Sigma_point = Sigma_at_n
                else:
                    # Spatially-varying tensor
                    Sigma_point = Sigma_at_n[multi_idx]

                # Check if diagonal (should be, given warning in solve_hjb_system)
                if is_diagonal_tensor(Sigma_point):
                    # Extract diagonal elements: σᵢ²
                    sigma_squared = np.diag(Sigma_point)

                    # Compute viscosity term: H_viscosity = (1/2) Σᵢ σᵢ² pᵢ²
                    H_viscosity = 0.5 * np.sum(sigma_squared * p**2)

                    # Compute running cost manually without viscosity term
                    # Standard MFG: H = (coupling/2)|p|² + (sigma²/2)|p|² + V(x) + F(m)
                    # We want:      H_running = (coupling/2)|p|² + V(x) + F(m)
                    #               H_total = H_viscosity + H_running

                    # Control cost (always present)
                    p_squared_norm = np.sum(p**2)
                    H_control = 0.5 * self.problem.coupling_coefficient * p_squared_norm

                    # Coupling term F(m) - typically G(m) where G'(m) = g(m)
                    # For standard MFG: F(m) = 0 (coupling is only through g(m) in FP)
                    # If custom components have coupling, it would be included
                    H_coupling_m = 0.0

                    # Potential V(x) if present
                    # For now, assume V=0 unless custom components specify
                    H_potential = 0.0

                    H_running = H_control + H_coupling_m + H_potential
                    H_values[multi_idx] = H_viscosity + H_running
                else:
                    # Non-diagonal tensor - not fully supported yet
                    # Fall back to treating as diagonal (ignoring off-diagonal terms)
                    import warnings

                    warnings.warn(
                        "Non-diagonal tensor detected during HJB evaluation. "
                        "Full tensor support not implemented. Using diagonal approximation (ignoring off-diagonal terms).",
                        UserWarning,
                        stacklevel=2,
                    )
                    # Extract diagonal elements and ignore off-diagonal
                    sigma_squared = np.diag(Sigma_point)

                    # Compute viscosity with diagonal only: H_viscosity = (1/2) Σᵢ σᵢ² pᵢ²
                    H_viscosity = 0.5 * np.sum(sigma_squared * p**2)

                    # Compute running cost manually (same as diagonal case)
                    p_squared_norm = np.sum(p**2)
                    H_control = 0.5 * self.problem.coupling_coefficient * p_squared_norm
                    H_coupling_m = 0.0
                    H_potential = 0.0

                    H_running = H_control + H_coupling_m + H_potential
                    H_values[multi_idx] = H_viscosity + H_running

            else:
                # Scalar diffusion mode (original code)
                # Get diffusion at this point
                if isinstance(sigma_base, (int, float)):
                    sigma_at_point = sigma_base
                elif isinstance(sigma_base, np.ndarray):
                    sigma_at_point = sigma_base[multi_idx]
                else:
                    sigma_at_point = self.problem.sigma

                # Temporarily override problem.sigma for Hamiltonian evaluation
                original_sigma = self.problem.sigma
                self.problem.sigma = sigma_at_point

                try:
                    # Call problem Hamiltonian (try both interfaces)
                    if hasattr(self.problem, "hamiltonian"):
                        H_values[multi_idx] = self.problem.hamiltonian(x_coords, m_at_point, p, t=0.0)
                    elif hasattr(self.problem, "H"):
                        H_values[multi_idx] = self.problem.H(multi_idx, m_at_point, derivs=derivs_at_point)
                    else:
                        raise AttributeError("Problem must have 'hamiltonian' or 'H' method")
                finally:
                    # Restore original sigma
                    self.problem.sigma = original_sigma

        return H_values


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBFDMSolver...")

    # Test 1D problem
    from mfg_pde import ExampleMFGProblem

    problem_1d = ExampleMFGProblem(Nx=30, Nt=20, T=1.0, sigma=0.1)
    solver_1d = HJBFDMSolver(problem_1d, solver_type="newton")

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.solver_type == "newton"
    assert solver_1d.hjb_method_name == "FDM"

    # Test solve_hjb_system
    import numpy as np

    M_test = np.ones((problem_1d.Nt + 1, problem_1d.Nx + 1)) * 0.5
    U_final = np.zeros(problem_1d.Nx + 1)
    U_prev = np.zeros((problem_1d.Nt + 1, problem_1d.Nx + 1))

    U_solution = solver_1d.solve_hjb_system(M_test, U_final, U_prev)

    assert U_solution.shape == (problem_1d.Nt + 1, problem_1d.Nx + 1)
    assert not np.any(np.isnan(U_solution))
    assert not np.any(np.isinf(U_solution))

    print("  1D solver converged")
    print(f"  U range: [{U_solution.min():.3f}, {U_solution.max():.3f}]")

    print("All smoke tests passed!")
