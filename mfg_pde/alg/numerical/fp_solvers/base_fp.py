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

    @abstractmethod
    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
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

        Returns:
            np.ndarray: A 2D array of shape (Nt, Nx) representing the computed
                        density M(t,x) over the time-space grid.
        """
