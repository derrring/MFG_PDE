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
        ∂m/∂t + ∇·(α m) = ∇·(D ∇m)

    where:
        m(t,x): probability density
        α(t,x,m): drift field (from various sources)
        D(t,x,m): diffusion tensor (isotropic, anisotropic, or state-dependent)

    Drift Sources (controlled by drift_field parameter):
        - Zero: α = 0 (pure diffusion)
        - Optimal control (MFG): α = -∇U (most common)
        - Prescribed field: α = v(t,x) (wind, currents)
        - Custom/state-dependent: α = f(t,x,m)

    Diffusion Types (controlled by diffusion_field parameter):
        - Constant isotropic: D = σ²/2 (scalar, same in all directions)
        - Anisotropic: D = diag(σ₁², σ₂², ...) (different per direction)
        - Spatially varying: D(t,x) (depends on location)
        - State-dependent: D(t,x,m) (nonlinear, depends on density)
        - Zero: D = 0 (pure advection, requires WENO/SL)

    This base class provides a general, powerful interface supporting all drift and
    diffusion types while maintaining backward compatibility with MFG-centric usage.

    Design Philosophy:
        Make simple cases simple (MFG with constant diffusion), make complex cases
        possible (anisotropic, state-dependent, spatially varying). The solver
        handles drift and diffusion computation based on provided parameters.

    Note for Implementers:
        When implementing concrete solvers:
        - Handle D=0 (pure advection) carefully - FDM may be unstable, use WENO/SL
        - Support anisotropic diffusion tensors for realistic applications
        - Enable state-dependent coefficients for nonlinear PDEs
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
        diffusion_field: float | np.ndarray | Callable | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Solves the full Fokker-Planck (FP) system forward in time.

        This method computes density evolution M(t,x) from t=0 to t=T under
        the specified drift and diffusion.

        General FP Equation:
            ∂m/∂t + ∇·(α m) = ∇·(D ∇m)

            where:
            - α(t,x,m): drift field (controlled by drift_field)
            - D(t,x,m): diffusion tensor (controlled by diffusion_field)

        Equation Types Supported:
            1. Advection-diffusion (D>0, α≠0): Standard MFG, transport with diffusion
            2. Pure diffusion (D>0, α=0): Heat equation
            3. Pure advection (D=0, α≠0): Transport equation (WENO/SL recommended)
            4. Anisotropic diffusion (D is tensor): Different diffusion in each direction
            5. State-dependent (D or α depend on m): Nonlinear PDEs

        Drift Specification:
            drift_field can be:
            - None: Zero drift α = 0 (pure diffusion)
            - np.ndarray: Precomputed drift field α(t,x)
              Shape: (Nt, Nx) for 1D scalar, (Nt, Nx, d) for d-dim vector
            - Callable: Function α(t, x, m) -> drift

        Diffusion Specification:
            diffusion_field can be:
            - None: Use problem.sigma (backward compatible)
            - float: Constant isotropic diffusion D = σ²/2
            - np.ndarray: Spatially varying diffusion D(t,x)
              Shape: (Nt, Nx) for scalar, (Nt, Nx, d, d) for tensor
            - Callable: Function D(t, x, m) -> diffusion

        Args:
            m_initial_condition: Initial density M(0,x) at t=0
                Shape: (Nx,) for 1D, (Nx, Ny) for 2D, etc.

            drift_field: Drift field specification (optional):
                - None: Zero drift
                - np.ndarray: Precomputed drift α(t,x)
                - Callable: Function α(t, x, m) -> drift
                Default: None

            diffusion_field: Diffusion specification (optional):
                - None: Use problem.sigma
                - float: Constant isotropic diffusion
                - np.ndarray: Spatially varying diffusion
                - Callable: Function D(t, x, m) -> diffusion
                Default: None

            show_progress: Display progress bar for timesteps
                Default: True

        Returns:
            Density evolution M(t,x) over time
            Shape: (Nt, Nx) or (Nt, Nx, Ny) etc.

        Examples:
            # Pure diffusion (heat equation, D>0, α=0)
            >>> M = solver.solve_fp_system(m0)

            # MFG optimal control (isotropic diffusion)
            >>> drift = -problem.compute_gradient(U_hjb) / problem.control_cost
            >>> M = solver.solve_fp_system(m0, drift_field=drift)

            # Anisotropic diffusion
            >>> D = np.diag([0.1, 0.5])  # Different in x,y directions
            >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=D)

            # State-dependent diffusion
            >>> D_func = lambda t, x, m: 0.1 * (1 + m)  # Increases with density
            >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=D_func)

            # Pure advection (D=0, α≠0)
            >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=0.0)

            # Spatially varying diffusion
            >>> D_field = create_spatially_varying_diffusion(...)  # (Nt, Nx)
            >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=D_field)

        Note:
            For MFG problems:
            - drift = -∇U / λ (user computes externally)
            - diffusion = problem.sigma (default) or custom
            This gives full control over both drift and diffusion.
        """


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing BaseFPSolver...")

    # Test base class availability
    assert BaseFPSolver is not None
    print("  BaseFPSolver class available")

    # Test that BaseFPSolver is abstract (cannot be instantiated)
    from mfg_pde import ExampleMFGProblem

    problem = ExampleMFGProblem(Nx=20, Nt=10, T=1.0, sigma=0.1)

    try:
        # This should fail because BaseFPSolver is abstract
        base_solver = BaseFPSolver(problem)
        raise AssertionError("Should have raised TypeError for abstract class")
    except TypeError:
        # Expected - abstract class cannot be instantiated
        print("  BaseFPSolver correctly abstract")

    print("Smoke tests passed!")
