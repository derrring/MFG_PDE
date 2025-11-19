"""
Type protocols for callable signatures in MFG_PDE.

This module defines Protocol classes for callable objects used throughout the framework,
enabling precise type checking and better IDE support.

Usage:
    from mfg_pde.types.callable_protocols import DriftFieldCallable, DiffusionFieldCallable

    def solve_fp_system(
        self,
        drift_field: np.ndarray | DriftFieldCallable | None = None,
        diffusion_field: float | np.ndarray | DiffusionFieldCallable | None = None,
    ) -> np.ndarray:
        ...
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class DriftFieldCallable(Protocol):
    """
    Protocol for callable drift field α(t, x, m).

    A drift field specifies the velocity field for Fokker-Planck evolution:
        ∂m/∂t + ∇·(α m) = (σ²/2) Δm

    Signature:
        α(t, x, m) -> drift vector

    Args:
        t: Current time (float)
        x: Spatial position (ndarray of shape (d,) or (N, d))
        m: Density field (ndarray)

    Returns:
        Drift vector (ndarray of shape (d,) or (N, d))

    Examples:
        >>> # Constant wind
        >>> def wind_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return np.array([1.0, 0.5])

        >>> # State-dependent drift
        >>> def density_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return -grad(m) / (m + 1e-10)  # Diffusion approximation

        >>> # Optimal control drift (MFG)
        >>> def control_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return -grad_U / sigma_sq
    """

    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate drift field at (t, x, m)."""
        ...


@runtime_checkable
class DiffusionFieldCallable(Protocol):
    """
    Protocol for callable diffusion field σ(t, x, m).

    A diffusion field specifies spatially/temporally/state-varying diffusion coefficient:
        ∂m/∂t + ∇·(α m) = (1/2) ∇·(σ(t,x,m)² ∇m)

    Signature:
        σ(t, x, m) -> diffusion coefficient(s)

    Args:
        t: Current time (float)
        x: Spatial position (ndarray of shape (d,) or (N, d))
        m: Density field (ndarray)

    Returns:
        - Scalar: Isotropic diffusion σ²
        - Array (N,): Spatially varying isotropic diffusion σ²(x)
        - Array (N, d): Diagonal tensor diffusion [σ_x², σ_y², ...]
        - Array (N, d, d): Full tensor diffusion Σ(x)

    Examples:
        >>> # Spatially varying diffusion
        >>> def varying_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> float:
        ...     return 0.1 + 0.05 * np.linalg.norm(x)

        >>> # Density-dependent diffusion
        >>> def density_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return 0.1 * (1 + m)  # Higher diffusion in dense regions

        >>> # Anisotropic diffusion (diagonal)
        >>> def anisotropic_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        ...     return np.array([0.1, 0.05])  # Different diffusion in x and y
    """

    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Evaluate diffusion field at (t, x, m)."""
        ...


@runtime_checkable
class HamiltonianCallable(Protocol):
    """
    Protocol for custom Hamiltonian function H(x, m, p, t).

    The Hamiltonian appears in the HJB equation:
        -∂u/∂t + H(x, m, ∇u, t) - (σ²/2) Δu = 0

    Signature:
        H(x_idx, m_at_x, derivs, **kwargs) -> float

    Args:
        x_idx: Grid index (int)
        m_at_x: Density at x (float)
        derivs: Derivatives in tuple notation:
                - 1D: {(0,): u, (1,): ∂u/∂x}
                - 2D: {(0,0): u, (1,0): ∂u/∂x, (0,1): ∂u/∂y}
        t_idx: Time index (optional, int)
        x_position: Actual spatial coordinate (optional, ndarray)
        current_time: Actual time value (optional, float)
        problem: Problem instance (optional)

    Returns:
        Hamiltonian value H(x, m, p, t)

    Examples:
        >>> # Quadratic Hamiltonian (LQ control)
        >>> def lq_hamiltonian(x_idx, m_at_x, derivs, **kwargs):
        ...     p = derivs[(1,)]
        ...     return 0.5 * p**2 - V[x_idx] - m_at_x**2

        >>> # Non-quadratic Hamiltonian
        >>> def custom_hamiltonian(x_idx, m_at_x, derivs, **kwargs):
        ...     p = derivs[(1,)]
        ...     return np.sqrt(1 + p**2) - np.log(1 + m_at_x)
    """

    def __call__(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple[int, ...], float],
        t_idx: int | None = None,
        x_position: NDArray[np.floating] | None = None,
        current_time: float | None = None,
        problem: object | None = None,
    ) -> float:
        """Evaluate Hamiltonian at (x, m, p, t)."""
        ...


@runtime_checkable
class HamiltonianDerivativeCallable(Protocol):
    """
    Protocol for Hamiltonian derivative dH/dm(x, m, p, t).

    The derivative dH/dm appears in the Fokker-Planck coupling term:
        ∂m/∂t + ∇·(α m) = (σ²/2) Δm - ∇·(m ∇(dH/dm))

    Signature:
        dH/dm(x_idx, m_at_x, derivs, **kwargs) -> float

    Args:
        (Same as HamiltonianCallable)

    Returns:
        Derivative dH/dm at (x, m, p, t)

    Examples:
        >>> # For H = 0.5 p² - V - m²
        >>> def hamiltonian_dm(x_idx, m_at_x, derivs, **kwargs):
        ...     return -2 * m_at_x

        >>> # For H = 0.5 p² - V - log(m)
        >>> def hamiltonian_dm(x_idx, m_at_x, derivs, **kwargs):
        ...     return -1 / (m_at_x + 1e-10)
    """

    def __call__(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple[int, ...], float],
        t_idx: int | None = None,
        x_position: NDArray[np.floating] | None = None,
        current_time: float | None = None,
        problem: object | None = None,
    ) -> float:
        """Evaluate dH/dm at (x, m, p, t)."""
        ...


@runtime_checkable
class PotentialCallable(Protocol):
    """
    Protocol for potential function V(x, t).

    The potential appears in the running cost:
        Running cost = ∫ V(x, t) m(t, x) dx

    Signature:
        V(x, t=None) -> float

    Args:
        x: Spatial position (float for 1D, ndarray for nD)
        t: Time (optional, float)

    Returns:
        Potential value V(x, t)

    Examples:
        >>> # Time-independent potential
        >>> def quadratic_potential(x: float) -> float:
        ...     return 0.5 * x**2

        >>> # Time-dependent potential
        >>> def moving_well(x: float, t: float) -> float:
        ...     center = np.sin(t)
        ...     return 0.5 * (x - center)**2
    """

    def __call__(self, x: float | NDArray[np.floating], t: float | None = None) -> float:
        """Evaluate potential at (x, t)."""
        ...


@runtime_checkable
class InitialDensityCallable(Protocol):
    """
    Protocol for initial density function m₀(x).

    Signature:
        m₀(x) -> density

    Args:
        x: Spatial position (float for 1D, ndarray for nD)

    Returns:
        Initial density m₀(x) ≥ 0

    Examples:
        >>> # Gaussian density
        >>> def gaussian_initial(x: float) -> float:
        ...     return np.exp(-100 * (x - 0.5)**2)

        >>> # Multi-modal density
        >>> def bimodal_initial(x: float) -> float:
        ...     return np.exp(-200 * (x - 0.2)**2) + np.exp(-200 * (x - 0.8)**2)
    """

    def __call__(self, x: float | NDArray[np.floating]) -> float:
        """Evaluate initial density at x."""
        ...


@runtime_checkable
class FinalValueCallable(Protocol):
    """
    Protocol for terminal value function u_T(x).

    The terminal condition for the HJB equation:
        u(T, x) = u_T(x)

    Signature:
        u_T(x) -> value

    Args:
        x: Spatial position (float for 1D, ndarray for nD)

    Returns:
        Terminal value u_T(x)

    Examples:
        >>> # Quadratic terminal cost
        >>> def quadratic_terminal(x: float) -> float:
        ...     return 0.5 * (x - 0.5)**2

        >>> # Indicator terminal cost
        >>> def indicator_terminal(x: float) -> float:
        ...     return 0 if 0.4 <= x <= 0.6 else 1000
    """

    def __call__(self, x: float | NDArray[np.floating]) -> float:
        """Evaluate terminal value at x."""
        ...


# Type aliases for convenience
CoefficientFieldType = float | NDArray[np.floating] | DriftFieldCallable | DiffusionFieldCallable | None
"""Type alias for coefficient fields (scalar | array | callable | None)."""
