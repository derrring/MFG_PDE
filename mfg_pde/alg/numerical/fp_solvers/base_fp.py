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
        σ²: diffusion coefficient

    Drift Sources:
        The drift α can come from multiple sources:
        - Optimal control (MFG): α = -∇U (most common)
        - Zero (heat equation): α = 0
        - Prescribed field: α = v(t,x) (wind, currents)
        - Custom: User-provided function

    This base class provides a general, powerful interface supporting all drift types
    while maintaining backward compatibility with MFG-centric usage.

    Design Philosophy:
        Make simple cases simple (MFG with U), make complex cases possible
        (custom drift sources). The solver handles drift computation internally
        based on provided parameters.
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
        U_solution_for_drift: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Solves the full Fokker-Planck (FP) system forward in time.

        This method computes density evolution M(t,x) from t=0 to t=T under
        the specified drift. Supports multiple drift sources for generality.

        Drift Specification (in priority order):
            1. U_solution_for_drift: MFG optimal control drift α = -∇U
            2. drift_field: Prescribed drift (array or callable)
            3. None: Pure diffusion (heat equation) α = 0

        Args:
            m_initial_condition: Initial density M(0,x) at t=0
                Shape: (Nx,) for 1D, (Nx, Ny) for 2D, etc.

            U_solution_for_drift: Value function for MFG optimal control drift.
                If provided: drift α = -∇U / λ (standard MFG)
                Shape: (Nt, Nx) or (Nt, Nx, Ny) etc.
                Default: None

            drift_field: Alternative drift specification (if U not provided):
                - np.ndarray: Precomputed drift field α(t,x)
                  Shape: (Nt, Nx, d) where d is spatial dimension
                - Callable: Function α(t, x, m) -> drift vector
                  Signature: (float, ndarray, ndarray) -> ndarray
                - None: Zero drift (pure diffusion)
                Default: None

            show_progress: Display progress bar for timesteps
                Default: True

        Returns:
            Density evolution M(t,x) over time
            Shape: (Nt, Nx) or (Nt, Nx, Ny) etc.

        Examples:
            # MFG optimal control (backward compatible)
            >>> M = solver.solve_fp_system(m0, U_solution_for_drift=U_hjb)

            # Pure diffusion (heat equation)
            >>> M = solver.solve_fp_system(m0)

            # Prescribed wind field
            >>> wind = lambda t, x, m: np.array([1.0, 0.5])
            >>> M = solver.solve_fp_system(m0, drift_field=wind)

            # Precomputed drift
            >>> alpha = compute_custom_drift(...)  # (Nt, Nx, 2)
            >>> M = solver.solve_fp_system(m0, drift_field=alpha)

        Note:
            Subclasses must implement drift computation logic. If both U and
            drift_field are provided, U takes precedence (MFG is primary use case).
        """
