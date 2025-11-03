from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from mfg_pde.core.mfg_problem import MFGProblem


class BaseFPSolver(ABC):
    """
    Abstract Base Class for Fokker-Planck (FP) equation solvers.

    The FP equation describes the evolution of the density M(t,x).
    It typically takes the form:
    M_t + div(drift_term * M) - div(diffusion_term * grad(M)) = 0
    or for constant diffusion sigma:
    M_t + div(drift_term * M) - (sigma^2/2) * M_xx = 0

    The drift_term often depends on the gradient of the value function U(t,x)
    obtained from the HJB equation, e.g., drift = -coefCT * grad(U) or a more
    complex optimal control.

    Note: This class maintains backward compatibility with the original interface
    while being part of the new numerical methods paradigm.
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
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """
        Solves the full Fokker-Planck (FP) system forward in time.

        This method computes the evolution of the density M(t,x) from t=0 to t=T,
        given the initial density M(0,x) and the value function U(t,x) which
        is used to determine the drift term in the FP equation.

        Args:
            m_initial_condition (np.ndarray): A 1D array of shape (Nx,) representing
                                            the initial density M(0,x) at t=0.
            U_solution_for_drift (np.ndarray): A 2D array of shape (Nt, Nx) representing
                                            the value function U(t,x) over the entire
                                            time-space grid. This is used to compute
                                            the drift term for the FP equation.
            show_progress (bool): Whether to display progress bar for timesteps.
                                Default: True

        Returns:
            np.ndarray: A 2D array of shape (Nt, Nx) representing the computed
                        density M(t,x) over the time-space grid.
        """
