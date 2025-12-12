"""MFG Components and evaluation mixins.

This module contains:
- MFGComponents: Dataclass for custom MFG problem definition
- HamiltonianMixin: Mixin providing Hamiltonian evaluation methods (H, dH_dm, potential)
- ConditionsMixin: Mixin providing initial/final/boundary condition methods

The mixin pattern allows MFGProblem to inherit these methods while keeping
the logic in separate, focused modules.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.types import HamiltonianJacobians
from mfg_pde.utils.aux_func import npart, ppart

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.geometry.boundary.conditions import BoundaryConditions

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
    hamiltonian_func: Callable | None = None  # H(x, m, p, t) -> float
    hamiltonian_dm_func: Callable | None = None  # dH/dm(x, m, p, t) -> float

    # Optional Jacobian for advanced solvers
    hamiltonian_jacobian_func: Callable | None = None  # Jacobian contribution

    # Potential function V(x, t) - part of Hamiltonian
    potential_func: Callable | None = None  # V(x, t) -> float

    # Coupling terms (for advanced MFG formulations)
    coupling_func: Callable | None = None  # Additional coupling terms

    # Initial and final conditions
    initial_density_func: Callable | None = None  # m_0(x) -> float
    final_value_func: Callable | None = None  # u_T(x) -> float

    # Boundary conditions
    boundary_conditions: BoundaryConditions | None = None

    # Problem parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: str = "MFG Problem"
    problem_type: str = "mfg"


# ============================================================================
# Hamiltonian Mixin - H, dH_dm, Jacobian, Coupling, Potential
# ============================================================================


class HamiltonianMixin:
    """
    Mixin class providing Hamiltonian evaluation methods.

    Provides:
    - H(): Hamiltonian evaluation
    - dH_dm(): Hamiltonian derivative w.r.t. density
    - get_hjb_hamiltonian_jacobian_contrib(): Jacobian for Newton methods
    - get_hjb_residual_m_coupling_term(): Coupling terms
    - _setup_custom_potential(): Potential initialization
    - get_potential_at_time(): Time-dependent potential access

    Required attributes from the inheriting class:
    - components: MFGComponents | None
    - is_custom: bool
    - coupling_coefficient: float
    - tSpace: np.ndarray
    - f_potential: np.ndarray
    - spatial_shape: tuple
    - dimension: int
    - _get_spatial_grid_internal(): method
    - _get_num_intervals(): method
    - _get_spacing(): method
    """

    # Type hints for attributes expected from MFGProblem
    components: MFGComponents | None
    is_custom: bool
    coupling_coefficient: float
    tSpace: np.ndarray
    f_potential: np.ndarray
    spatial_shape: tuple
    dimension: int

    def _validate_hamiltonian_components(self) -> None:
        """Validate Hamiltonian-related components."""
        if self.components is None:
            return

        has_hamiltonian = self.components.hamiltonian_func is not None
        has_hamiltonian_dm = self.components.hamiltonian_dm_func is not None

        if has_hamiltonian and not has_hamiltonian_dm:
            raise ValueError("hamiltonian_dm_func is required when hamiltonian_func is provided")

        if has_hamiltonian_dm and not has_hamiltonian:
            raise ValueError("hamiltonian_func is required when hamiltonian_dm_func is provided")

        # Validate function signatures only if Hamiltonians are provided
        if has_hamiltonian:
            self._validate_function_signature(
                self.components.hamiltonian_func,
                "hamiltonian_func",
                ["x_idx", "m_at_x"],
                gradient_param_required=True,
            )

        if has_hamiltonian_dm:
            self._validate_function_signature(
                self.components.hamiltonian_dm_func,
                "hamiltonian_dm_func",
                ["x_idx", "m_at_x"],
                gradient_param_required=True,
            )

    def _validate_function_signature(
        self, func: Callable, name: str, expected_params: list, gradient_param_required: bool = False
    ) -> None:
        """
        Validate function signature has expected parameters.

        Args:
            func: Function to validate
            name: Name of the function (for error messages)
            expected_params: List of required parameter names
            gradient_param_required: If True, requires EITHER 'derivs' OR 'p_values'
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if gradient_param_required:
            has_derivs = "derivs" in params
            has_p_values = "p_values" in params

            if not (has_derivs or has_p_values):
                raise ValueError(
                    f"{name} must accept either 'derivs' (tuple notation, preferred) "
                    f"or 'p_values' (legacy string-key format) parameter. "
                    f"Current parameters: {params}"
                )

        missing = [p for p in expected_params if p not in params]
        if missing:
            raise ValueError(f"{name} must accept parameters: {expected_params}. Missing: {missing}")

    def _setup_custom_potential(self) -> None:
        """Setup custom potential function (part of Hamiltonian)."""
        if self.components is None or self.components.potential_func is None:
            return

        potential_func = self.components.potential_func
        spatial_grid = self._get_spatial_grid_internal()
        num_intervals = self._get_num_intervals() or 0

        for i in range(num_intervals + 1):
            x_i = spatial_grid[i]

            sig = inspect.signature(potential_func)
            if "t" in sig.parameters or "time" in sig.parameters:
                self.f_potential[i] = potential_func(x_i, 0.0)
            else:
                self.f_potential[i] = potential_func(x_i)

    def get_potential_at_time(self, t_idx: int) -> np.ndarray:
        """Get potential function at specific time (for time-dependent potentials)."""
        if self.is_custom and self.components is not None and self.components.potential_func is not None:
            sig = inspect.signature(self.components.potential_func)
            if "t" in sig.parameters or "time" in sig.parameters:
                current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0
                potential_at_t = np.zeros_like(self.f_potential)

                num_intervals = self._get_num_intervals() or 0
                spatial_grid = self._get_spatial_grid_internal()
                for i in range(num_intervals + 1):
                    x_i = spatial_grid[i]
                    potential_at_t[i] = self.components.potential_func(x_i, current_time)

                return potential_at_t

        return self.f_potential.copy()

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,
        p_values: dict[str, float] | None = None,
        t_idx: int | None = None,
        x_position: float | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian function H(x, m, p, t).

        Supports both tuple notation (derivs) and legacy string-key (p_values) formats.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives in tuple notation (NEW, preferred):
                    - 1D: {(0,): u, (1,): du/dx}
                    - 2D: {(0,0): u, (1,0): du/dx, (0,1): du/dy}
            p_values: Momentum dictionary (LEGACY, deprecated):
                      {"forward": p_forward, "backward": p_backward}
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Hamiltonian value H(x, m, p, t)

        Note:
            Provide EITHER derivs OR p_values. If both provided, derivs takes precedence.
            p_values is deprecated and will be removed in a future version.
        """
        import warnings

        if derivs is None and p_values is None:
            raise ValueError("Must provide either 'derivs' or 'p_values' to H()")

        if derivs is None:
            warnings.warn(
                "p_values parameter is deprecated. Use derivs instead. "
                "See docs/gradient_notation_standard.md for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            from mfg_pde.compat.gradient_notation import ensure_tuple_notation

            derivs = ensure_tuple_notation(p_values, dimension=1, u_value=0.0)

        # Compute x_position and current_time if not provided
        if x_position is None:
            if isinstance(x_idx, tuple):
                if hasattr(self, "geometry") and self.geometry is not None:
                    if hasattr(self, "spatial_shape") and len(self.spatial_shape) > 1:
                        flat_idx = np.ravel_multi_index(x_idx, self.spatial_shape)
                        spatial_grid = self.geometry.get_spatial_grid()
                        x_position = spatial_grid[flat_idx]
                    else:
                        x_position = None
                else:
                    x_position = None
            else:
                spatial_grid = self._get_spatial_grid_internal()
                if spatial_grid is not None:
                    x_position = spatial_grid[x_idx]
                else:
                    x_position = None

        if current_time is None and t_idx is not None:
            current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0

        # Use custom Hamiltonian if provided
        if self.is_custom and self.components is not None and self.components.hamiltonian_func is not None:
            sig = inspect.signature(self.components.hamiltonian_func)
            params = list(sig.parameters.keys())

            if "derivs" in params:
                return self.components.hamiltonian_func(
                    x_idx=x_idx,
                    x_position=x_position,
                    m_at_x=m_at_x,
                    derivs=derivs,
                    t_idx=t_idx,
                    current_time=current_time,
                    problem=self,
                )
            else:
                from mfg_pde.compat.gradient_notation import derivs_to_p_values_1d

                p_values_legacy = derivs_to_p_values_1d(derivs)

                return self.components.hamiltonian_func(
                    x_idx=x_idx,
                    x_position=x_position,
                    m_at_x=m_at_x,
                    p_values=p_values_legacy,
                    t_idx=t_idx,
                    current_time=current_time,
                    problem=self,
                )

        # Default Hamiltonian: H = 0.5*c*|p|^2 - V(x) - m^2
        p = derivs.get((1,), 0.0)

        if np.isnan(p) or np.isinf(p) or np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan

        npart_val = float(npart(p))
        ppart_val = float(ppart(p))

        if abs(npart_val) > VALUE_BEFORE_SQUARE_LIMIT or abs(ppart_val) > VALUE_BEFORE_SQUARE_LIMIT:
            return np.nan

        try:
            term_npart_sq = npart_val**2
            term_ppart_sq = ppart_val**2
        except OverflowError:
            return np.nan

        if np.isinf(term_npart_sq) or np.isnan(term_npart_sq) or np.isinf(term_ppart_sq) or np.isnan(term_ppart_sq):
            return np.nan

        hamiltonian_control_part = 0.5 * self.coupling_coefficient * (term_npart_sq + term_ppart_sq)

        if np.isinf(hamiltonian_control_part) or np.isnan(hamiltonian_control_part):
            return np.nan

        # Get potential value (handle both 1D and nD indexing)
        if isinstance(x_idx, tuple) and hasattr(self, "spatial_shape") and len(self.spatial_shape) > 1:
            flat_idx = np.ravel_multi_index(x_idx, self.spatial_shape)
            potential_cost_V_x = float(self.f_potential.flat[flat_idx])
        elif hasattr(self, "spatial_shape") and len(self.spatial_shape) > 1:
            potential_cost_V_x = float(self.f_potential.flat[x_idx])
        else:
            potential_cost_V_x = float(self.f_potential[x_idx])

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
        derivs: dict[tuple, float] | None = None,
        p_values: dict[str, float] | None = None,
        t_idx: int | None = None,
        x_position: float | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian derivative with respect to density dH/dm.

        Supports both tuple notation (derivs) and legacy string-key (p_values) formats.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives in tuple notation (NEW, preferred)
            p_values: Momentum dictionary (LEGACY, deprecated)
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Derivative dH/dm at (x, m, p, t)
        """
        import warnings

        if derivs is None and p_values is None:
            raise ValueError("Must provide either 'derivs' or 'p_values' to dH_dm()")

        if derivs is None:
            warnings.warn(
                "p_values parameter is deprecated. Use derivs instead. "
                "See docs/gradient_notation_standard.md for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            from mfg_pde.compat.gradient_notation import ensure_tuple_notation

            derivs = ensure_tuple_notation(p_values, dimension=1, u_value=0.0)

        # Compute x_position and current_time if not provided
        if x_position is None:
            if isinstance(x_idx, tuple):
                if hasattr(self, "geometry") and self.geometry is not None:
                    if hasattr(self, "spatial_shape") and len(self.spatial_shape) > 1:
                        flat_idx = np.ravel_multi_index(x_idx, self.spatial_shape)
                        spatial_grid = self.geometry.get_spatial_grid()
                        x_position = spatial_grid[flat_idx]
                    else:
                        x_position = None
                else:
                    x_position = None
            else:
                spatial_grid = self._get_spatial_grid_internal()
                if spatial_grid is not None:
                    x_position = spatial_grid[x_idx]
                else:
                    x_position = None

        if current_time is None and t_idx is not None:
            current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0

        # Use custom derivative if provided
        if self.is_custom and self.components is not None and self.components.hamiltonian_dm_func is not None:
            sig = inspect.signature(self.components.hamiltonian_dm_func)
            params = list(sig.parameters.keys())

            if "derivs" in params:
                return self.components.hamiltonian_dm_func(
                    x_idx=x_idx,
                    x_position=x_position,
                    m_at_x=m_at_x,
                    derivs=derivs,
                    t_idx=t_idx,
                    current_time=current_time,
                    problem=self,
                )
            else:
                from mfg_pde.compat.gradient_notation import derivs_to_p_values_1d

                p_values_legacy = derivs_to_p_values_1d(derivs)

                return self.components.hamiltonian_dm_func(
                    x_idx=x_idx,
                    x_position=x_position,
                    m_at_x=m_at_x,
                    p_values=p_values_legacy,
                    t_idx=t_idx,
                    current_time=current_time,
                    problem=self,
                )

        # Default: dH/dm = 2m
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan
        return 2 * m_at_x

    def get_hjb_hamiltonian_jacobian_contrib(
        self,
        U_for_jacobian_terms: np.ndarray,
        t_idx_n: int,
    ) -> HamiltonianJacobians | None:
        """
        Compute Hamiltonian Jacobian components for advanced HJB solvers.

        Returns structured Jacobian coefficients (diagonal, lower, upper) that form
        a tridiagonal matrix for Newton/policy iteration schemes.

        Args:
            U_for_jacobian_terms: Current value function estimate, shape (Nx,)
            t_idx_n: Time index for evaluation

        Returns:
            HamiltonianJacobians dataclass with diagonal, lower, upper components,
            or None if not applicable.
        """
        if self.is_custom and self.components is not None and self.components.hamiltonian_jacobian_func is not None:
            try:
                return self.components.hamiltonian_jacobian_func(
                    U_for_jacobian_terms=U_for_jacobian_terms,
                    t_idx_n=t_idx_n,
                    problem=self,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Jacobian computation failed: {e}")

        if not self.is_custom:
            num_intervals = self._get_num_intervals() or 0
            Nx = num_intervals + 1
            dx = self._get_spacing() or 1.0
            coupling_coefficient = self.coupling_coefficient

            J_D_H = np.zeros(Nx)
            J_L_H = np.zeros(Nx)
            J_U_H = np.zeros(Nx)

            if abs(dx) < 1e-14 or Nx <= 1:
                return HamiltonianJacobians(diagonal=J_D_H, lower=J_L_H, upper=J_U_H)

            U_curr = U_for_jacobian_terms

            for i in range(Nx):
                ip1 = (i + 1) % Nx
                im1 = (i - 1 + Nx) % Nx

                p1_i = (U_curr[ip1] - U_curr[i]) / dx
                p2_i = (U_curr[i] - U_curr[im1]) / dx

                J_D_H[i] = coupling_coefficient * (npart(p1_i) + ppart(p2_i)) / (dx**2)
                J_L_H[i] = -coupling_coefficient * ppart(p2_i) / (dx**2)
                J_U_H[i] = -coupling_coefficient * npart(p1_i) / (dx**2)

            return HamiltonianJacobians(diagonal=J_D_H, lower=J_L_H, upper=J_U_H)

        return None

    def get_hjb_residual_m_coupling_term(
        self,
        M_density_at_n_plus_1: np.ndarray,
        U_n_current_guess_derivatives: dict[str, np.ndarray],
        x_idx: int,
        t_idx_n: int,
    ) -> float | None:
        """Optional coupling term for residual computation."""
        if self.is_custom and self.components is not None and self.components.coupling_func is not None:
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

                logging.getLogger(__name__).warning(f"Coupling term computation failed: {e}")

        if not self.is_custom:
            m_val = M_density_at_n_plus_1[x_idx]
            m_val = m_val.item() if hasattr(m_val, "item") else float(m_val)
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


# ============================================================================
# Conditions Mixin - Initial, Final, Boundary Conditions
# ============================================================================


class ConditionsMixin:
    """
    Mixin class providing initial/final/boundary condition methods.

    Provides:
    - _setup_custom_initial_density(): Initial density setup
    - _setup_custom_final_value(): Final value setup
    - get_boundary_conditions(): Boundary condition access

    Required attributes from the inheriting class:
    - components: MFGComponents | None
    - is_custom: bool
    - m_init: np.ndarray
    - u_fin: np.ndarray
    - dimension: int
    - _get_spatial_grid_internal(): method
    - _get_num_intervals(): method
    """

    # Type hints for attributes expected from MFGProblem
    components: MFGComponents | None
    is_custom: bool
    m_init: np.ndarray
    u_fin: np.ndarray
    dimension: int

    def _setup_custom_initial_density(self) -> None:
        """Setup custom initial density function m_0(x)."""
        if self.components is None or self.components.initial_density_func is None:
            return

        initial_func = self.components.initial_density_func
        spatial_grid = self._get_spatial_grid_internal()
        num_intervals = self._get_num_intervals() or 0

        for i in range(num_intervals + 1):
            x_i = spatial_grid[i]
            self.m_init[i] = max(initial_func(x_i), 0.0)

    def _setup_custom_final_value(self) -> None:
        """Setup custom final value function u_T(x)."""
        if self.components is None or self.components.final_value_func is None:
            return

        final_func = self.components.final_value_func

        num_intervals = self._get_num_intervals()
        if self.dimension == 1 and num_intervals is not None:
            spatial_grid = self._get_spatial_grid_internal()
            for i in range(num_intervals + 1):
                x_i = spatial_grid[i]
                self.u_fin[i] = final_func(x_i)
        elif hasattr(self, "geometry") and self.geometry is not None:
            spatial_grid = self.geometry.get_spatial_grid()
            num_points = spatial_grid.shape[0]

            for i in range(num_points):
                x_i = spatial_grid[i]
                self.u_fin.flat[i] = final_func(x_i)
        else:
            import warnings

            warnings.warn(
                "Cannot setup custom final value: dimension not 1D and no geometry available. "
                "Using default u_fin initialization.",
                UserWarning,
                stacklevel=2,
            )

    def get_boundary_conditions(self) -> BoundaryConditions:
        """Get boundary conditions for the problem."""
        from mfg_pde.geometry.boundary.conditions import periodic_bc

        if self.is_custom and self.components is not None and self.components.boundary_conditions is not None:
            return self.components.boundary_conditions
        else:
            return periodic_bc(dimension=self.dimension)


# ============================================================================
# Backward Compatibility - Keep MFGComponentsMixin as alias
# ============================================================================


class MFGComponentsMixin(HamiltonianMixin, ConditionsMixin):
    """
    Combined mixin for backward compatibility.

    Deprecated: Use HamiltonianMixin and ConditionsMixin separately.
    """

    def _validate_components(self) -> None:
        """Validate all components (delegates to HamiltonianMixin)."""
        self._validate_hamiltonian_components()
