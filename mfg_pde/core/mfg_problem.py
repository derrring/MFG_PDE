from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union
import inspect
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Import npart and ppart from the utils module
from mfg_pde.utils.aux_func import npart, ppart
from ..geometry import BoundaryConditions

# Define a limit for values before squaring to prevent overflow within H
VALUE_BEFORE_SQUARE_LIMIT = 1e150


# ============================================================================
# MFG Components for Custom Problem Definition
# ============================================================================


@dataclass
class MFGComponents:
    """
    Container for all components that define a custom MFG problem.

    This class holds all the mathematical components needed to fully specify
    an MFG problem, allowing users to provide custom implementations.
    """

    # Core Hamiltonian components
    hamiltonian_func: Optional[Callable] = None  # H(x, m, p, t) -> float
    hamiltonian_dm_func: Optional[Callable] = None  # dH/dm(x, m, p, t) -> float

    # Optional Jacobian for advanced solvers
    hamiltonian_jacobian_func: Optional[Callable] = None  # Jacobian contribution

    # Potential function V(x, t)
    potential_func: Optional[Callable] = None  # V(x, t) -> float

    # Initial and final conditions
    initial_density_func: Optional[Callable] = None  # m_0(x) -> float
    final_value_func: Optional[Callable] = None  # u_T(x) -> float

    # Boundary conditions
    boundary_conditions: Optional[BoundaryConditions] = None

    # Coupling terms (for advanced MFG formulations)
    coupling_func: Optional[Callable] = None  # Additional coupling terms

    # Problem parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: str = "MFG Problem"
    problem_type: str = "mfg"


# ============================================================================
# Unified MFG Problem Class
# ============================================================================


class MFGProblem:
    """
    Unified MFG problem class that can handle both predefined and custom formulations.

    This class serves as the single constructor for all MFG problems:
    - Default usage: Uses built-in Hamiltonian (equivalent to old ExampleMFGProblem)
    - Custom usage: Accepts MFGComponents for full mathematical control
    """

    def __init__(
        self,
        xmin: float = 0.0,
        xmax: float = 1.0,
        Nx: int = 51,
        T: float = 1.0,
        Nt: int = 51,
        sigma: float = 1.0,
        coefCT: float = 0.5,
        components: Optional[MFGComponents] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MFG problem with optional custom components.

        Args:
            xmin, xmax, Nx: Spatial domain parameters
            T, Nt: Time domain parameters
            sigma: Diffusion coefficient
            coefCT: Control cost coefficient
            components: Optional MFGComponents for custom problem definition
            **kwargs: Additional parameters
        """

        self.xmin: float = xmin
        self.xmax: float = xmax
        self.Lx: float = xmax - xmin
        self.Nx: int = Nx
        self.Dx: float = (xmax - xmin) / Nx if Nx > 0 else 0.0

        self.T: float = T
        self.Nt: int = Nt
        self.Dt: float = T / Nt if Nt > 0 else 0.0

        self.xSpace: np.ndarray = np.linspace(xmin, xmax, Nx + 1, endpoint=True)
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        self.sigma: float = sigma
        self.coefCT: float = coefCT

        # Store custom components if provided
        self.components = components
        self.is_custom = components is not None

        # Merge parameters
        if self.is_custom:
            all_params = {**components.parameters, **kwargs}
        else:
            all_params = kwargs

        # Initialize arrays
        self.f_potential: NDArray[np.float64]
        self.u_fin: NDArray[np.float64]
        self.m_init: NDArray[np.float64]

        # Initialize functions
        self._initialize_functions(**all_params)

        # Validate custom components if provided
        if self.is_custom:
            self._validate_components()

    def _potential(self, x: float) -> float:
        """Default potential function."""
        return 50 * (
            0.1 * np.cos(x * 2 * np.pi / self.Lx)
            + 0.25 * np.sin(x * 2 * np.pi / self.Lx)
            + 0.1 * np.sin(x * 4 * np.pi / self.Lx)
        )

    def _u_final(self, x: float) -> float:
        """Default final value function."""
        return 5 * (
            np.cos(x * 2 * np.pi / self.Lx) + 0.4 * np.sin(x * 4 * np.pi / self.Lx)
        )

    def _m_initial(self, x: float) -> float:
        """Default initial density function."""
        return 2 * np.exp(-200 * (x - 0.2) ** 2) + np.exp(-200 * (x - 0.8) ** 2)

    def _initialize_functions(self, **kwargs: Any) -> None:
        """Initialize potential, initial density, and final value functions."""

        # Initialize potential
        self.f_potential = np.zeros(self.Nx + 1)
        if self.is_custom and self.components.potential_func is not None:
            self._setup_custom_potential()
        else:
            for i in range(self.Nx + 1):
                self.f_potential[i] = self._potential(self.xSpace[i])

        # Initialize final value
        self.u_fin = np.zeros(self.Nx + 1)
        if self.is_custom and self.components.final_value_func is not None:
            self._setup_custom_final_value()
        else:
            for i in range(self.Nx + 1):
                self.u_fin[i] = self._u_final(self.xSpace[i])

        # Initialize initial density
        self.m_init = np.zeros(self.Nx + 1)
        if self.is_custom and self.components.initial_density_func is not None:
            self._setup_custom_initial_density()
        else:
            for i in range(self.Nx + 1):
                self.m_init[i] = self._m_initial(self.xSpace[i])

        # Normalize initial density
        integral_m_init = np.sum(self.m_init) * self.Dx
        if integral_m_init > 1e-10:
            self.m_init /= integral_m_init

    def _validate_components(self):
        """Validate that required components are provided."""
        if self.components.hamiltonian_func is None:
            raise ValueError("hamiltonian_func is required in MFGComponents")

        if self.components.hamiltonian_dm_func is None:
            raise ValueError("hamiltonian_dm_func is required in MFGComponents")

        # Validate function signatures
        self._validate_function_signature(
            self.components.hamiltonian_func,
            "hamiltonian_func",
            ["x_idx", "m_at_x", "p_values", "t_idx"],
        )

        self._validate_function_signature(
            self.components.hamiltonian_dm_func,
            "hamiltonian_dm_func",
            ["x_idx", "m_at_x", "p_values", "t_idx"],
        )

    def _validate_function_signature(
        self, func: Callable, name: str, expected_params: list
    ):
        """Validate function signature has expected parameters."""
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Allow extra parameters, but require the expected ones
        missing = [p for p in expected_params if p not in params]
        if missing:
            raise ValueError(
                f"{name} must accept parameters: {expected_params}. Missing: {missing}"
            )

    def _setup_custom_potential(self):
        """Setup custom potential function."""
        potential_func = self.components.potential_func

        for i in range(self.Nx + 1):
            x_i = self.xSpace[i]

            # Check if potential depends on time
            sig = inspect.signature(potential_func)
            if "t" in sig.parameters or "time" in sig.parameters:
                # Time-dependent potential - use t=0 for initialization
                self.f_potential[i] = potential_func(x_i, 0.0)
            else:
                # Time-independent potential
                self.f_potential[i] = potential_func(x_i)

    def _setup_custom_initial_density(self):
        """Setup custom initial density function."""
        initial_func = self.components.initial_density_func

        for i in range(self.Nx + 1):
            x_i = self.xSpace[i]
            self.m_init[i] = max(initial_func(x_i), 0.0)

    def _setup_custom_final_value(self):
        """Setup custom final value function."""
        final_func = self.components.final_value_func

        for i in range(self.Nx + 1):
            x_i = self.xSpace[i]
            self.u_fin[i] = final_func(x_i)

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Hamiltonian function - uses custom implementation if provided, otherwise default.
        """

        # Use custom Hamiltonian if provided
        if self.is_custom:
            try:
                # Get current position and time
                x_position = self.xSpace[x_idx]
                current_time = (
                    self.tSpace[t_idx]
                    if t_idx is not None and t_idx < len(self.tSpace)
                    else 0.0
                )

                # Call user-provided Hamiltonian
                result = self.components.hamiltonian_func(
                    x_idx=x_idx,
                    x_position=x_position,
                    m_at_x=m_at_x,
                    p_values=p_values,
                    t_idx=t_idx,
                    current_time=current_time,
                    problem=self,  # Pass reference to problem for accessing parameters
                )

                return result

            except Exception as e:
                # Log error but return NaN to maintain solver stability
                import logging

                logging.getLogger(__name__).warning(
                    f"Custom Hamiltonian evaluation failed at x_idx={x_idx}: {e}"
                )
                return np.nan

        # Default Hamiltonian implementation
        p_forward = p_values.get("forward")
        p_backward = p_values.get("backward")

        if p_forward is None or p_backward is None:
            return np.nan
        if (
            np.isnan(p_forward)
            or np.isinf(p_forward)
            or np.isnan(p_backward)
            or np.isinf(p_backward)
            or np.isnan(m_at_x)
            or np.isinf(m_at_x)
        ):
            return np.nan

        npart_val_fwd = npart(p_forward)
        ppart_val_bwd = ppart(p_backward)

        if (
            abs(npart_val_fwd) > VALUE_BEFORE_SQUARE_LIMIT
            or abs(ppart_val_bwd) > VALUE_BEFORE_SQUARE_LIMIT
        ):
            return np.nan

        try:
            term_npart_sq = npart_val_fwd**2
            term_ppart_sq = ppart_val_bwd**2
        except OverflowError:
            return np.nan

        if (
            np.isinf(term_npart_sq)
            or np.isnan(term_npart_sq)
            or np.isinf(term_ppart_sq)
            or np.isnan(term_ppart_sq)
        ):
            return np.nan

        hamiltonian_control_part = 0.5 * self.coefCT * (term_npart_sq + term_ppart_sq)

        if np.isinf(hamiltonian_control_part) or np.isnan(hamiltonian_control_part):
            return np.nan

        potential_cost_V_x = self.f_potential[x_idx]
        coupling_density_m_x = m_at_x**2

        if (
            np.isinf(potential_cost_V_x)
            or np.isnan(potential_cost_V_x)
            or np.isinf(coupling_density_m_x)
            or np.isnan(coupling_density_m_x)
        ):
            return np.nan

        result = hamiltonian_control_part - potential_cost_V_x - coupling_density_m_x

        if np.isinf(result) or np.isnan(result):
            return np.nan

        return result

    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        p_values: Dict[str, float],
        t_idx: Optional[int] = None,
    ) -> float:
        """
        Hamiltonian derivative with respect to density - uses custom if provided, otherwise default.
        """

        # Use custom derivative if provided
        if self.is_custom:
            try:
                # Get current position and time
                x_position = self.xSpace[x_idx]
                current_time = (
                    self.tSpace[t_idx]
                    if t_idx is not None and t_idx < len(self.tSpace)
                    else 0.0
                )

                # Call user-provided derivative
                result = self.components.hamiltonian_dm_func(
                    x_idx=x_idx,
                    x_position=x_position,
                    m_at_x=m_at_x,
                    p_values=p_values,
                    t_idx=t_idx,
                    current_time=current_time,
                    problem=self,  # Pass reference to problem
                )

                return result

            except Exception as e:
                # Log error but return NaN
                import logging

                logging.getLogger(__name__).warning(
                    f"Custom Hamiltonian derivative failed at x_idx={x_idx}: {e}"
                )
                return np.nan

        # Default derivative implementation
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan
        return 2 * m_at_x

    def get_hjb_hamiltonian_jacobian_contrib(
        self,
        U_for_jacobian_terms: np.ndarray,
        t_idx_n: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Optional Jacobian contribution for advanced solvers.
        """
        # Use custom Jacobian if provided
        if self.is_custom and self.components.hamiltonian_jacobian_func is not None:
            try:
                return self.components.hamiltonian_jacobian_func(
                    U_for_jacobian_terms=U_for_jacobian_terms,
                    t_idx_n=t_idx_n,
                    problem=self,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Jacobian computation failed: {e}")

        # Default Jacobian implementation (only for non-custom problems)
        if not self.is_custom:
            Nx = self.Nx + 1
            Dx = self.Dx
            coefCT = self.coefCT

            J_D_H = np.zeros(Nx)
            J_L_H = np.zeros(Nx)
            J_U_H = np.zeros(Nx)

            if abs(Dx) < 1e-14 or Nx <= 1:
                return J_D_H, J_L_H, J_U_H

            U_curr = U_for_jacobian_terms

            for i in range(Nx):
                ip1 = (i + 1) % Nx
                im1 = (i - 1 + Nx) % Nx

                # Derivatives of U_curr
                p1_i = (U_curr[ip1] - U_curr[i]) / Dx
                p2_i = (U_curr[i] - U_curr[im1]) / Dx

                J_D_H[i] = coefCT * (npart(p1_i) + ppart(p2_i)) / (Dx**2)
                J_L_H[i] = -coefCT * ppart(p2_i) / (Dx**2)
                J_U_H[i] = -coefCT * npart(p1_i) / (Dx**2)

            return J_D_H, J_L_H, J_U_H

        return None

    def get_hjb_residual_m_coupling_term(
        self,
        M_density_at_n_plus_1: np.ndarray,
        U_n_current_guess_derivatives: Dict[str, np.ndarray],
        x_idx: int,
        t_idx_n: int,
    ) -> Optional[float]:
        """
        Optional coupling term for residual computation.
        """
        # Use custom coupling if provided
        if self.is_custom and self.components.coupling_func is not None:
            try:
                return self.components.coupling_func(
                    M_density_at_n_plus_1=M_density_at_n_plus_1,
                    U_n_current_guess_derivatives=U_n_current_guess_derivatives,
                    x_idx=x_idx,
                    t_idx_n=t_idx_n,
                    problem=self,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Coupling term computation failed: {e}"
                )

        # Default coupling implementation (only for non-custom problems)
        if not self.is_custom:
            m_val = M_density_at_n_plus_1[x_idx]
            if np.isnan(m_val) or np.isinf(m_val):
                return np.nan
            try:
                term = -2 * (m_val**2)
            except OverflowError:
                return np.nan
            if np.isinf(term) or np.isnan(term):
                return np.nan
            return term

        return None

    def get_boundary_conditions(self) -> BoundaryConditions:
        """Get boundary conditions for the problem."""
        if self.is_custom and self.components.boundary_conditions is not None:
            return self.components.boundary_conditions
        else:
            # Default periodic boundary conditions
            return BoundaryConditions(type="periodic")

    def get_potential_at_time(self, t_idx: int) -> np.ndarray:
        """Get potential function at specific time (for time-dependent potentials)."""
        if self.is_custom and self.components.potential_func is not None:
            # Check if potential is time-dependent
            sig = inspect.signature(self.components.potential_func)
            if "t" in sig.parameters or "time" in sig.parameters:
                # Recompute potential at current time
                current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0
                potential_at_t = np.zeros_like(self.f_potential)

                for i in range(self.Nx + 1):
                    x_i = self.xSpace[i]
                    potential_at_t[i] = self.components.potential_func(
                        x_i, current_time
                    )

                return potential_at_t

        return self.f_potential.copy()

    def get_final_u(self) -> np.ndarray:
        return self.u_fin.copy()

    def get_initial_m(self) -> np.ndarray:
        return self.m_init.copy()

    def get_problem_info(self) -> Dict[str, Any]:
        """Get information about the problem."""
        if self.is_custom:
            return {
                "description": self.components.description,
                "problem_type": self.components.problem_type,
                "is_custom": True,
                "has_custom_hamiltonian": True,
                "has_custom_potential": self.components.potential_func is not None,
                "has_custom_initial": self.components.initial_density_func is not None,
                "has_custom_final": self.components.final_value_func is not None,
                "has_jacobian": self.components.hamiltonian_jacobian_func is not None,
                "has_coupling": self.components.coupling_func is not None,
                "parameters": self.components.parameters,
                "domain": {"xmin": self.xmin, "xmax": self.xmax, "Nx": self.Nx},
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"sigma": self.sigma, "coefCT": self.coefCT},
            }
        else:
            return {
                "description": "Default MFG Problem",
                "problem_type": "example",
                "is_custom": False,
                "has_custom_hamiltonian": False,
                "has_custom_potential": False,
                "has_custom_initial": False,
                "has_custom_final": False,
                "has_jacobian": False,
                "has_coupling": False,
                "parameters": {},
                "domain": {"xmin": self.xmin, "xmax": self.xmax, "Nx": self.Nx},
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"sigma": self.sigma, "coefCT": self.coefCT},
            }


# ============================================================================
# Builder Pattern for Easy Problem Construction
# ============================================================================


class MFGProblemBuilder:
    """
    Builder class for constructing MFG problems step by step.

    This class provides a fluent interface for building custom MFG problems.
    """

    def __init__(self):
        """Initialize empty builder."""
        self.components = MFGComponents()
        self.domain_params = {}
        self.time_params = {}
        self.solver_params = {}

    def hamiltonian(
        self, hamiltonian_func: Callable, hamiltonian_dm_func: Callable
    ) -> "MFGProblemBuilder":
        """
        Set custom Hamiltonian and its derivative.

        Args:
            hamiltonian_func: H(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem) -> float
            hamiltonian_dm_func: dH/dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem) -> float
        """
        self.components.hamiltonian_func = hamiltonian_func
        self.components.hamiltonian_dm_func = hamiltonian_dm_func
        return self

    def potential(self, potential_func: Callable) -> "MFGProblemBuilder":
        """
        Set custom potential function.

        Args:
            potential_func: V(x, t=None) -> float
        """
        self.components.potential_func = potential_func
        return self

    def initial_density(self, initial_func: Callable) -> "MFGProblemBuilder":
        """
        Set custom initial density function.

        Args:
            initial_func: m_0(x) -> float
        """
        self.components.initial_density_func = initial_func
        return self

    def final_value(self, final_func: Callable) -> "MFGProblemBuilder":
        """
        Set custom final value function.

        Args:
            final_func: u_T(x) -> float
        """
        self.components.final_value_func = final_func
        return self

    def boundary_conditions(self, bc: BoundaryConditions) -> "MFGProblemBuilder":
        """Set boundary conditions."""
        self.components.boundary_conditions = bc
        return self

    def jacobian(self, jacobian_func: Callable) -> "MFGProblemBuilder":
        """Set optional Jacobian function for advanced solvers."""
        self.components.hamiltonian_jacobian_func = jacobian_func
        return self

    def coupling(self, coupling_func: Callable) -> "MFGProblemBuilder":
        """Set optional coupling function."""
        self.components.coupling_func = coupling_func
        return self

    def domain(self, xmin: float, xmax: float, Nx: int) -> "MFGProblemBuilder":
        """Set spatial domain parameters."""
        self.domain_params = {"xmin": xmin, "xmax": xmax, "Nx": Nx}
        return self

    def time(self, T: float, Nt: int) -> "MFGProblemBuilder":
        """Set time domain parameters."""
        self.time_params = {"T": T, "Nt": Nt}
        return self

    def coefficients(
        self, sigma: float = 1.0, coefCT: float = 0.5
    ) -> "MFGProblemBuilder":
        """Set solver coefficients."""
        self.solver_params.update({"sigma": sigma, "coefCT": coefCT})
        return self

    def parameters(self, **params) -> "MFGProblemBuilder":
        """Set additional problem parameters."""
        self.components.parameters.update(params)
        return self

    def description(
        self, desc: str, problem_type: str = "custom"
    ) -> "MFGProblemBuilder":
        """Set problem description and type."""
        self.components.description = desc
        self.components.problem_type = problem_type
        return self

    def build(self) -> MFGProblem:
        """Build the MFG problem."""
        # Validate required components
        if self.components.hamiltonian_func is None:
            raise ValueError("Hamiltonian function is required")
        if self.components.hamiltonian_dm_func is None:
            raise ValueError("Hamiltonian derivative function is required")

        # Set default domain if not specified
        if not self.domain_params:
            self.domain_params = {"xmin": 0.0, "xmax": 1.0, "Nx": 51}

        # Set default time domain if not specified
        if not self.time_params:
            self.time_params = {"T": 1.0, "Nt": 51}

        # Combine all parameters
        all_params = {**self.domain_params, **self.time_params, **self.solver_params}

        # Create and return problem with custom components
        return MFGProblem(components=self.components, **all_params)


# ============================================================================
# Convenience Functions and Backward Compatibility
# ============================================================================


def ExampleMFGProblem(**kwargs) -> MFGProblem:
    """
    Create an MFG problem with default Hamiltonian (backward compatibility).

    This function provides backward compatibility for code that used
    the old ExampleMFGProblem class.
    """
    return MFGProblem(**kwargs)


def create_mfg_problem(
    hamiltonian_func: Callable, hamiltonian_dm_func: Callable, **kwargs
) -> MFGProblem:
    """
    Convenience function to create custom MFG problem.

    Args:
        hamiltonian_func: Custom Hamiltonian function
        hamiltonian_dm_func: Hamiltonian derivative function
        **kwargs: Domain, time, and solver parameters
    """
    # Extract configurations
    domain_config = {
        "xmin": kwargs.pop("xmin", 0.0),
        "xmax": kwargs.pop("xmax", 1.0),
        "Nx": kwargs.pop("Nx", 51),
    }

    time_config = {"T": kwargs.pop("T", 1.0), "Nt": kwargs.pop("Nt", 51)}

    solver_config = {
        "sigma": kwargs.pop("sigma", 1.0),
        "coefCT": kwargs.pop("coefCT", 0.5),
    }

    # Create components
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,
        **{k: v for k, v in kwargs.items() if k.endswith("_func")},
        parameters={k: v for k, v in kwargs.items() if not k.endswith("_func")},
    )

    return MFGProblem(
        components=components, **domain_config, **time_config, **solver_config
    )
