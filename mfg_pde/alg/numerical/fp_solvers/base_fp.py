from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from mfg_pde.core.mfg_problem import MFGProblem


class BaseFPSolver(ABC):
    """
    Abstract Base Class for Fokker-Planck (FP) equation solvers.

    The FP equation describes density evolution:
        ∂m/∂t + ∇·(α m) = σ²/2 Δm

    where:
        m(t,x): probability density
        α(t,x): drift field (from various sources)
        σ²: diffusion coefficient (set via problem.sigma)

    Drift Sources:
        The drift α can come from multiple sources:
        - Optimal control (MFG): α = -∇U (most common)
        - Zero (heat equation): α = 0
        - Prescribed field: α = v(t,x) (wind, currents)
        - Custom: User-provided function

    Diffusion Control:
        Set problem.sigma to control diffusion strength:
        - σ > 0: Standard advection-diffusion (MFG typical)
        - σ = 0: Pure advection (requires specialized schemes like WENO, SL)

    This base class provides a general, powerful interface supporting all drift types
    while maintaining backward compatibility with MFG-centric usage.

    Design Philosophy:
        Make simple cases simple (MFG with U), make complex cases possible
        (custom drift sources, pure advection). The solver handles drift computation
        internally based on provided parameters.

    Note for Implementers:
        When implementing concrete solvers, ensure σ=0 (pure advection) is handled
        correctly. FDM with upwind may be unstable; consider WENO, Semi-Lagrangian,
        or flux-limiting schemes for advection-dominated flows.
    """

    def __init__(self, problem: MFGProblem):
        """
        Initializes the FP solver with the MFG problem definition.

        Args:
            problem (MFGProblem): An instance of an MFGProblem (or its subclass)
                                containing all problem-specific parameters and functions.
        """
        self.problem = problem
        self.fp_method_name: str = "BaseFP"  # Concrete solvers should override this
        self.backend = None  # Backend for array operations (NumPy, PyTorch, JAX)

        # Validate solver compatibility if problem supports it (Phase 3.1.5)
        self._validate_problem_compatibility()

    def _validate_problem_compatibility(self) -> None:
        """
        Validate that this solver is compatible with the problem.

        This method checks if the problem has solver compatibility detection
        (Phase 3.1 unified interface) and validates compatibility if available.
        For older problems without this feature, validation is skipped.
        """
        # Only validate if problem has the new unified interface
        if not hasattr(self.problem, "validate_solver_type"):
            return  # Backward compatibility: skip validation for old problems

        # Get solver type identifier from subclass
        solver_type = self._get_solver_type_id()
        if solver_type is None:
            return  # Solver doesn't specify type, skip validation

        # Validate compatibility
        try:
            self.problem.validate_solver_type(solver_type)
        except ValueError as e:
            # Re-raise with solver class information
            raise ValueError(f"Cannot use {self.__class__.__name__} with this problem.\n\n{e!s}") from e

    def _get_solver_type_id(self) -> str | None:
        """
        Get solver type identifier for compatibility checking.

        Subclasses should override this to return their type identifier.
        Returns None if solver type cannot be determined (skips validation).
        """
        # Map class names to solver type IDs
        class_name = self.__class__.__name__
        type_mapping = {
            "FPFDMSolver": "fdm",
            "FPParticleSolver": "particle",
            "FPNetworkSolver": "network_solver",
            "FPGFDMSolver": "gfdm",
        }
        return type_mapping.get(class_name)

    @abstractmethod
    def solve_fp_system(
        self,
        m_initial_condition: np.ndarray,
        drift_field: np.ndarray | Callable | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Solves the full Fokker-Planck (FP) system forward in time.

        This method computes density evolution M(t,x) from t=0 to t=T under
        the specified drift field.

        Equation Types Supported:
            The general FP equation is: ∂m/∂t + ∇·(α m) = σ²/2 Δm

            Three equation types are accessible via parameters:
            1. Advection-diffusion (σ>0, α≠0): Standard MFG, transport with diffusion
            2. Pure diffusion (σ>0, α=0): Heat equation
            3. Pure advection (σ=0, α≠0): Transport equation (WENO/SL recommended)

        Drift Specification:
            drift_field can be:
            - None: Zero drift α = 0 (pure diffusion)
            - np.ndarray: Precomputed drift field α(t,x)
              Shape: (Nt, Nx) for 1D scalar drift, (Nt, Nx, d) for d-dimensional vector drift
            - Callable: Function α(t, x, m) -> drift
              Signature: (float, ndarray, ndarray) -> ndarray

        Diffusion Control:
            Set problem.sigma to control diffusion:
            - σ > 0: Diffusion active (standard MFG)
            - σ = 0: Pure advection (ensure solver supports this!)

        Args:
            m_initial_condition: Initial density M(0,x) at t=0
                Shape: (Nx,) for 1D, (Nx, Ny) for 2D, etc.

            drift_field: Drift field specification (optional):
                - None: Zero drift (pure diffusion)
                - np.ndarray: Precomputed drift α(t,x)
                - Callable: Function α(t, x, m) -> drift
                Default: None

            show_progress: Display progress bar for timesteps
                Default: True

        Returns:
            Density evolution M(t,x) over time
            Shape: (Nt, Nx) or (Nt, Nx, Ny) etc.

        Examples:
            # Pure diffusion (heat equation, σ>0, α=0)
            >>> M = solver.solve_fp_system(m0)

            # MFG optimal control (σ>0, α=-∇U)
            >>> drift = -problem.compute_gradient(U_hjb) / problem.control_cost
            >>> M = solver.solve_fp_system(m0, drift_field=drift)

            # Pure advection (transport equation, σ=0, α≠0)
            >>> problem = MFGProblem(..., sigma=0.0)
            >>> solver = FPWENOSolver(problem)
            >>> drift = compute_advection_field(...)
            >>> M = solver.solve_fp_system(m0, drift_field=drift)

            # Prescribed wind field (σ>0, α=v(t,x))
            >>> wind = lambda t, x, m: np.array([1.0, 0.5])
            >>> M = solver.solve_fp_system(m0, drift_field=wind)

            # Custom drift function
            >>> def custom_drift(t, x, m):
            ...     return -np.gradient(potential(x)) + wind_field(t, x)
            >>> M = solver.solve_fp_system(m0, drift_field=custom_drift)

        Note:
            For MFG problems, users compute drift from value function externally:
            drift = -∇U / λ, where U is from HJB solver and λ is control cost.
            This gives full control over gradient computation method.
        """
