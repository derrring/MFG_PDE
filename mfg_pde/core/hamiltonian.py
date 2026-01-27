"""
Hamiltonian and Lagrangian abstractions for MFG optimal control.

Issues: #623 (original), #651 (duality), #667 (auto-diff), #673 (class-based API)

This module provides the mathematical foundation for coupling HJB and FP solvers
through a formal optimal control interface.

Mathematical Background
-----------------------
In MFG, agents minimize a cost functional:

    J[α] = E[∫₀ᵀ L(x, α, m) dt + g(x(T), m(T))]

where:
    - L(x, α, m): Running cost (Lagrangian)
    - α: Control (velocity/drift)
    - m: Population density

The HJB equation uses the Hamiltonian H, related to L via Legendre transform:

    H(x, p, m) = sup_α { p·α - L(x, α, m) }

The optimal control satisfies:

    α* = argmax_α { p·α - L(x, α, m) } = -∂H/∂p

Architecture (v0.17.2+)
-----------------------
Two complementary class hierarchies:

1. **Hamiltonian** (Issue #673): Full MFG Hamiltonian H(x, m, p, t)
   - Clean callable API: `H(x, m, p, t)` or `H(x, m, derivs, t)`
   - Auto-computed derivatives via `dp()` and `dm()` (Issue #667)
   - Supports state-dependent terms (congestion, potential)

2. **ControlCostBase** (original): Pure control cost L(α) or H(p)
   - Simpler interface for control-only Hamiltonians
   - `optimal_control(p)`, `lagrangian(α)`, `hamiltonian(p)`
   - Can be composed into Hamiltonian

3. **Lagrangian** (Issue #651): Running cost L(x, α, m, t)
   - Legendre transform to Hamiltonian
   - Duality: L ↔ H via `to_hamiltonian()` and `to_lagrangian()`

Design Philosophy
-----------------
Users typically think in terms of **running cost** (Lagrangian), but solvers
need the **Hamiltonian** and **optimal control formula**. This module:

1. Accepts either Lagrangian or Hamiltonian specification
2. Provides `optimal_control(p)` - the single source of truth for drift
3. Handles sign conventions via `OptimizationSense`
4. Auto-computes derivatives when not provided (Issue #667)

For common cases (quadratic, L1), closed-form formulas are provided.
For general cases, numerical Legendre transform is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

# Issue #700: Import HamiltonianJacobians for jacobian_fd() method
from mfg_pde.types import HamiltonianJacobians

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OptimizationSense(Enum):
    """
    Optimization direction for MFG problems.

    This enum captures the fundamental difference between:
    - Control theory: Agents MINIMIZE cost (α moves downhill on U)
    - Economics: Agents MAXIMIZE utility (α moves uphill on U)

    The sign convention in `optimal_control(p)` depends on this choice.

    Examples
    --------
    Cost minimization (standard MFG):
    >>> hamiltonian = QuadraticHamiltonian(sense=OptimizationSense.MINIMIZE)
    >>> alpha = hamiltonian.optimal_control(grad_U)  # Returns -grad_U/lambda

    Utility maximization (economics):
    >>> hamiltonian = QuadraticHamiltonian(sense=OptimizationSense.MAXIMIZE)
    >>> alpha = hamiltonian.optimal_control(grad_U)  # Returns +grad_U/lambda
    """

    MINIMIZE = "minimize"  # Control theory: min ∫L dt, α = -∂H/∂p
    MAXIMIZE = "maximize"  # Economics: max ∫U dt, α = +∂H/∂p


class ControlCostBase(ABC):
    """
    Abstract base for control cost specifications.

    This is the primary interface for defining how control effort translates
    to cost. Subclasses implement specific cost structures (quadratic, L1, etc.).

    The key method is `optimal_control(p)` which computes α* given momentum p.
    This is THE SINGLE SOURCE OF TRUTH for drift direction in MFG coupling.

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Control cost weight λ (always positive)

    Attributes
    ----------
    sense : OptimizationSense
        Optimization direction
    control_cost : float
        Control cost weight λ
    sign : int
        +1 for MINIMIZE, -1 for MAXIMIZE (internal use)
    """

    def __init__(
        self,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        control_cost: float = 1.0,
    ):
        if control_cost <= 0:
            raise ValueError(f"control_cost must be positive, got {control_cost}")

        self.sense = sense
        self.control_cost = control_cost
        # Sign convention: MINIMIZE -> α = -∂H/∂p, MAXIMIZE -> α = +∂H/∂p
        self.sign = 1 if sense == OptimizationSense.MINIMIZE else -1

    @abstractmethod
    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """
        Compute optimal control α* given momentum p.

        This is the PRIMARY interface for HJB-FP coupling. The FP solver
        uses this to compute drift from the HJB solution gradient.

        For cost minimization: α* = -∂H/∂p (gradient descent on value)
        For utility maximization: α* = +∂H/∂p (gradient ascent on utility)

        Parameters
        ----------
        p : ndarray
            Momentum field (typically ∇U from HJB solution)
            Shape: (Nx,) for 1D, (Ny, Nx) for 2D, etc.

        Returns
        -------
        ndarray
            Optimal control/velocity field α*
            Same shape as p
        """
        ...

    @abstractmethod
    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """
        Evaluate running cost L(α) for given control.

        Parameters
        ----------
        alpha : ndarray
            Control field

        Returns
        -------
        ndarray
            Running cost at each point
        """
        ...

    @abstractmethod
    def hamiltonian(self, p: np.ndarray) -> np.ndarray:
        """
        Evaluate Hamiltonian H(p) for given momentum.

        For MINIMIZE: H(p) = sup_α { p·α - L(α) }
        For MAXIMIZE: H(p) = inf_α { p·α - L(α) }

        Parameters
        ----------
        p : ndarray
            Momentum field

        Returns
        -------
        ndarray
            Hamiltonian value at each point
        """
        ...


class QuadraticControlCost(ControlCostBase):
    """
    Quadratic control cost: L(α) = ½λ|α|².

    This is the most common choice in MFG, corresponding to:
    - Lagrangian: L(α) = ½λ|α|²
    - Hamiltonian: H(p) = ½|p|²/λ
    - Optimal control: α* = ∓p/λ (sign depends on sense)

    The quadratic structure gives closed-form Legendre transform.

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Control cost weight λ (higher = more costly to move)

    Examples
    --------
    Standard MFG (minimize cost):
    >>> cost = QuadraticControlCost(sense=OptimizationSense.MINIMIZE, control_cost=1.0)
    >>> grad_U = np.array([1.0, 2.0, 3.0])
    >>> alpha = cost.optimal_control(grad_U)  # Returns [-1, -2, -3]

    Economics (maximize utility):
    >>> cost = QuadraticControlCost(sense=OptimizationSense.MAXIMIZE, control_cost=1.0)
    >>> alpha = cost.optimal_control(grad_U)  # Returns [1, 2, 3]
    """

    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """
        Compute α* = -p/λ for MINIMIZE, +p/λ for MAXIMIZE.

        Mathematical derivation (MINIMIZE case):
            H(p) = sup_α { p·α - ½λ|α|² }
            ∂/∂α = p - λα = 0  →  α* = p/λ
            But we want -∂H/∂p for drift: α* = -p/λ
        """
        return -self.sign * p / self.control_cost

    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """Evaluate L(α) = ½λ|α|²."""
        return 0.5 * self.control_cost * np.sum(alpha**2, axis=-1)

    def hamiltonian(self, p: np.ndarray) -> np.ndarray:
        """Evaluate H(p) = ½|p|²/λ."""
        return 0.5 * np.sum(p**2, axis=-1) / self.control_cost


class L1ControlCost(ControlCostBase):
    """
    L1 (bang-bang) control cost: L(α) = λ|α|.

    This models minimum fuel/effort problems where the cost is proportional
    to the magnitude of control, not its square. Results in bang-bang control.

    - Lagrangian: L(α) = λ|α|
    - Hamiltonian: H(p) = 0 if |p| ≤ λ, else ∞
    - Optimal control: α* = -sign(p) if |p| > λ, else 0

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Control cost weight λ (threshold for activation)

    Examples
    --------
    Minimum fuel problem:
    >>> cost = L1ControlCost(control_cost=0.5)
    >>> grad_U = np.array([0.3, 0.7, -0.8])
    >>> alpha = cost.optimal_control(grad_U)
    >>> # Returns [0, -1, 1] (bang-bang: no control below threshold)
    """

    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """
        Compute bang-bang control.

        α* = -sign(p) where |p| > λ, else 0
        """
        # Bang-bang: control is ±1 or 0
        alpha = np.zeros_like(p)
        active = np.abs(p) > self.control_cost
        alpha[active] = -self.sign * np.sign(p[active])
        return alpha

    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """Evaluate L(α) = λ|α|."""
        return self.control_cost * np.sum(np.abs(alpha), axis=-1)

    def hamiltonian(self, p: np.ndarray) -> np.ndarray:
        """
        Evaluate H(p).

        For L1 cost, H(p) = 0 if |p| ≤ λ, else the Legendre transform
        is not well-defined (would be +∞).
        """
        # Return 0 in feasible region, large value otherwise
        h = np.zeros_like(p)
        infeasible = np.abs(p) > self.control_cost
        h[infeasible] = np.inf
        return h


class BoundedControlCost(ControlCostBase):
    """
    Bounded control with quadratic cost: L(α) = ½λ|α|², |α| ≤ α_max.

    This models situations where control effort is limited (e.g., max speed).

    - Lagrangian: L(α) = ½λ|α|² for |α| ≤ α_max, else ∞
    - Optimal control: α* = clip(-p/λ, -α_max, α_max)

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Control cost weight λ
    max_control : float
        Maximum allowed control magnitude α_max

    Examples
    --------
    Speed-limited agents:
    >>> cost = BoundedControlCost(control_cost=1.0, max_control=2.0)
    >>> grad_U = np.array([1.0, 3.0, 5.0])
    >>> alpha = cost.optimal_control(grad_U)
    >>> # Returns [-1, -2, -2] (clipped at max_control)
    """

    def __init__(
        self,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        control_cost: float = 1.0,
        max_control: float = 1.0,
    ):
        super().__init__(sense, control_cost)
        if max_control <= 0:
            raise ValueError(f"max_control must be positive, got {max_control}")
        self.max_control = max_control

    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """Compute clipped quadratic optimal control."""
        # Unconstrained optimum
        alpha_unconstrained = -self.sign * p / self.control_cost
        # Clip to bounds
        return np.clip(alpha_unconstrained, -self.max_control, self.max_control)

    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """Evaluate L(α) = ½λ|α|² (assumes α is feasible)."""
        return 0.5 * self.control_cost * np.sum(alpha**2, axis=-1)

    def hamiltonian(self, p: np.ndarray) -> np.ndarray:
        """Evaluate H(p) with control bounds."""
        # Optimal control
        alpha_star = self.optimal_control(p)
        # H = p·α* - L(α*)
        p_dot_alpha = np.sum(p * alpha_star, axis=-1)
        return p_dot_alpha - self.lagrangian(alpha_star)


# Convenience aliases
QuadraticHamiltonian = QuadraticControlCost
L1Hamiltonian = L1ControlCost
BoundedHamiltonian = BoundedControlCost


# ============================================================================
# MFGOperator: Common base for Hamiltonian and Lagrangian (Issue #651)
# ============================================================================


class MFGOperatorBase(ABC):
    """
    Abstract base class for MFG operators (Hamiltonian and Lagrangian).

    This provides a common interface for both H(x, m, p, t) and L(x, α, m, t),
    enabling symmetric treatment via Legendre transform duality.

    Mathematical Background (Issue #651)
    -------------------------------------
    In optimal control, Hamiltonian and Lagrangian are Legendre duals:

        H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }   (Legendre transform)
        L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }   (Inverse transform)

    This duality means:
    - Users can specify either H or L (whichever is more natural)
    - The other is automatically derived via Legendre transform
    - Both have equal mathematical status

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost (MINIMIZE) or maximize utility (MAXIMIZE)
    finite_diff_eps : float
        Step size for finite difference derivatives (default: 1e-6)
    """

    def __init__(
        self,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        finite_diff_eps: float = 1e-6,
    ):
        self.sense = sense
        self.finite_diff_eps = finite_diff_eps
        # Sign convention: MINIMIZE -> α = -∂H/∂p, MAXIMIZE -> α = +∂H/∂p
        self._sign = 1 if sense == OptimizationSense.MINIMIZE else -1

    @property
    @abstractmethod
    def is_hamiltonian(self) -> bool:
        """Return True if this is a Hamiltonian operator."""
        ...

    @property
    @abstractmethod
    def is_lagrangian(self) -> bool:
        """Return True if this is a Lagrangian operator."""
        ...


# ============================================================================
# Hamiltonian: Full MFG Hamiltonian H(x, m, p, t) - Issue #673
# ============================================================================


@dataclass
class HamiltonianState:
    """
    State container for Hamiltonian evaluation.

    Encapsulates all information needed to evaluate H(x, m, p, t).
    This allows clean separation between state and computation.

    Attributes
    ----------
    x : NDArray
        Position(s), shape (d,) for single point or (N, d) for N points
    m : float | NDArray
        Density at x, scalar or array of shape (N,)
    p : NDArray
        Momentum ∇u at x, shape (d,) or (N, d)
    t : float
        Time
    x_idx : int | None
        Grid index if on a grid (for grid-based methods)
    """

    x: NDArray
    m: float | NDArray
    p: NDArray
    t: float = 0.0
    x_idx: int | None = None


class HamiltonianBase(MFGOperatorBase):
    """
    Abstract base for full MFG Hamiltonians H(x, m, p, t).

    This is the primary interface for class-based Hamiltonians in MFG.
    Unlike ControlCostBase (which handles only H(p)), Hamiltonian
    supports full state dependence including position x, density m, and time t.

    Key Features (Issue #673)
    -------------------------
    - Clean callable API: `H(x, m, p, t)` returns Hamiltonian value
    - Auto-differentiation: `dp()` and `dm()` computed automatically (#667)
    - Composable: Can wrap ControlCostBase for control cost component
    - Extensible: Subclass for custom state-dependent Hamiltonians
    - Symmetric duality: `to_lagrangian()` converts to Lagrangian (#651)

    Mathematical Background
    -----------------------
    The HJB equation in MFG is:

        -∂u/∂t + H(x, m, ∇u, t) = 0

    Where H typically has the form:

        H(x, m, p, t) = H_control(p) + V(x, t) + f(m)

    With:
    - H_control(p): Control cost term (e.g., ½|p|²/λ for quadratic)
    - V(x, t): Potential/running cost
    - f(m): Density coupling (e.g., congestion)

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility.
        Affects sign of optimal control: α* = ∓∂H/∂p
    finite_diff_eps : float
        Step size for finite difference derivatives (default: 1e-6)

    Examples
    --------
    Basic usage with separable Hamiltonian:

    >>> H = SeparableHamiltonian(
    ...     control_cost=QuadraticControlCost(control_cost=2.0),
    ...     potential=lambda x, t: np.sin(x),
    ...     coupling=lambda m: m**2
    ... )
    >>> x, m, p, t = np.array([0.5]), 0.3, np.array([1.0]), 0.0
    >>> H(x, m, p, t)  # Evaluate Hamiltonian
    >>> H.dp(x, m, p, t)  # Get ∂H/∂p (auto-computed)
    >>> H.dm(x, m, p, t)  # Get ∂H/∂m (auto-computed)
    >>> L = H.legendre_transform()  # Convert to Lagrangian (Issue #651)

    See Also
    --------
    LagrangianBase : Running cost L(x, α, m, t)
    DualHamiltonian : Hamiltonian from Lagrangian via Legendre transform
    DualLagrangian : Lagrangian from Hamiltonian via inverse Legendre
    """

    @property
    def is_hamiltonian(self) -> bool:
        """Return True - this is a Hamiltonian operator."""
        return True

    @property
    def is_lagrangian(self) -> bool:
        """Return False - this is not a Lagrangian operator."""
        return False

    @abstractmethod
    def __call__(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Evaluate Hamiltonian H(x, m, p, t).

        Parameters
        ----------
        x : NDArray
            Position, shape (d,) for d-dimensional problem
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x, shape (d,)
        t : float
            Time (default: 0.0)

        Returns
        -------
        float | NDArray
            Hamiltonian value(s)
        """
        ...

    def dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute ∂H/∂p (gradient w.r.t. momentum).

        This is used for:
        - Optimal control: α* = -sign * ∂H/∂p
        - Jacobian computation in Newton methods
        - Characteristic curves in semi-Lagrangian methods

        Default implementation uses finite differences (Issue #667).
        Override for analytic derivatives.

        Parameters
        ----------
        x : NDArray
            Position, shape (d,)
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x, shape (d,)
        t : float
            Time

        Returns
        -------
        NDArray
            Gradient ∂H/∂p, shape (d,)
        """
        return self._finite_diff_dp(x, m, p, t)

    def dm(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute ∂H/∂m (derivative w.r.t. density).

        This appears in the FP equation source term and is needed
        for coupling the HJB-FP system correctly.

        Default implementation uses finite differences (Issue #667).
        Override for analytic derivatives.

        Parameters
        ----------
        x : NDArray
            Position, shape (d,)
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x, shape (d,)
        t : float
            Time

        Returns
        -------
        float | NDArray
            Derivative ∂H/∂m
        """
        return self._finite_diff_dm(x, m, p, t)

    def optimal_control(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute optimal control α* given state and momentum.

        The optimal control satisfies the first-order condition:
        - MINIMIZE: α* = -∂H/∂p (gradient descent on value)
        - MAXIMIZE: α* = +∂H/∂p (gradient ascent on utility)

        Parameters
        ----------
        x : NDArray
            Position
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x
        t : float
            Time

        Returns
        -------
        NDArray
            Optimal control α*, same shape as p
        """
        dH_dp = self.dp(x, m, p, t)
        return -self._sign * dH_dp

    def jacobian_fd(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        dx: float,
        t: float = 0.0,
        scheme: str = "central",
    ) -> HamiltonianJacobians:
        """
        Compute FD Jacobian components for Newton/policy iteration.

        Uses chain rule: ∂H/∂U_j = ∂H/∂p · ∂p/∂U_j

        This method connects the continuous derivative dp() with the discrete
        finite difference Jacobian used by HJB solvers for Newton iteration.

        Issue #700: Unifies Hamiltonian class with HamiltonianJacobians.

        Parameters
        ----------
        x : NDArray
            Position
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x (current gradient estimate)
        dx : float
            Grid spacing
        t : float
            Time (default: 0.0)
        scheme : str
            FD scheme: "central", "upwind_forward", "upwind_backward"
            - "central": p ≈ (U[i+1] - U[i-1])/(2dx)
            - "upwind_forward": p ≈ (U[i+1] - U[i])/dx
            - "upwind_backward": p ≈ (U[i] - U[i-1])/dx

        Returns
        -------
        HamiltonianJacobians
            Dataclass with diagonal, lower, upper tridiagonal components

        Example
        -------
        >>> H = SeparableHamiltonian(control_cost=QuadraticControlCost(1.0))
        >>> jac = H.jacobian_fd(x, m, p, dx=0.01, scheme="central")
        >>> # Use in Newton iteration:
        >>> A_diag = diffusion_diag + jac.diagonal
        >>> A_lower = diffusion_lower + jac.lower
        >>> A_upper = diffusion_upper + jac.upper
        """
        # Get ∂H/∂p from the class method
        dH_dp = self.dp(x, m, p, t)
        dH_dp_scalar = float(dH_dp[0]) if hasattr(dH_dp, "__len__") else float(dH_dp)

        if scheme == "central":
            # p ≈ (U[i+1] - U[i-1]) / (2dx)
            # ∂p/∂U[i+1] = +1/(2dx)
            # ∂p/∂U[i-1] = -1/(2dx)
            # ∂p/∂U[i] = 0
            coeff = dH_dp_scalar / (2 * dx)
            return HamiltonianJacobians(
                diagonal=np.array([0.0]),
                lower=np.array([-coeff]),  # ∂H/∂U[i-1]
                upper=np.array([coeff]),  # ∂H/∂U[i+1]
            )
        elif scheme == "upwind_forward":
            # p ≈ (U[i+1] - U[i]) / dx
            # ∂p/∂U[i+1] = +1/dx
            # ∂p/∂U[i] = -1/dx
            coeff = dH_dp_scalar / dx
            return HamiltonianJacobians(
                diagonal=np.array([-coeff]),
                lower=np.array([0.0]),
                upper=np.array([coeff]),
            )
        elif scheme == "upwind_backward":
            # p ≈ (U[i] - U[i-1]) / dx
            # ∂p/∂U[i] = +1/dx
            # ∂p/∂U[i-1] = -1/dx
            coeff = dH_dp_scalar / dx
            return HamiltonianJacobians(
                diagonal=np.array([coeff]),
                lower=np.array([-coeff]),
                upper=np.array([0.0]),
            )
        else:
            raise ValueError(f"Unknown FD scheme: {scheme}. Supported: 'central', 'upwind_forward', 'upwind_backward'")

    def _finite_diff_dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float,
    ) -> NDArray:
        """Compute ∂H/∂p using central finite differences."""
        eps = self.finite_diff_eps
        d = p.shape[0] if p.ndim > 0 else 1
        grad = np.zeros(d)

        p_flat = np.atleast_1d(p).astype(float)

        for i in range(d):
            p_plus = p_flat.copy()
            p_minus = p_flat.copy()
            p_plus[i] += eps
            p_minus[i] -= eps

            H_plus = self(x, m, p_plus, t)
            H_minus = self(x, m, p_minus, t)
            grad[i] = (H_plus - H_minus) / (2 * eps)

        return grad

    def _finite_diff_dm(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float,
    ) -> float:
        """Compute ∂H/∂m using central finite differences."""
        eps = self.finite_diff_eps
        m_scalar = float(m) if np.isscalar(m) else float(np.mean(m))

        H_plus = self(x, m_scalar + eps, p, t)
        H_minus = self(x, m_scalar - eps, p, t)

        return float((H_plus - H_minus) / (2 * eps))

    # Issue #673: to_legacy_func() removed - use class-based API directly

    def legendre_transform(
        self,
        p_bounds: tuple[float, float] | None = None,
        n_search: int = 100,
    ) -> LagrangianBase:
        """
        Convert Hamiltonian to Lagrangian via Legendre transform.

        Computes L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }

        The Legendre transform is involutive: applying it twice recovers
        the original (up to convexification). This provides symmetric
        duality between H and L (Issue #651).

        Parameters
        ----------
        p_bounds : tuple[float, float] | None
            Bounds on momentum for numerical optimization.
            If None, uses (-10, 10) as default.
        n_search : int
            Number of points for grid search (default: 100)

        Returns
        -------
        LagrangianBase
            The Legendre-transformed Lagrangian (DualLagrangian)

        Examples
        --------
        >>> H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
        >>> L = H.legendre_transform()
        >>> # L(α) = ½λ|α|² (recovered from H = ½|p|²/λ)
        >>> H_back = L.legendre_transform()  # Involutive: back to Hamiltonian
        """
        return DualLagrangian(
            hamiltonian=self,
            sense=self.sense,
            p_bounds=p_bounds or (-10.0, 10.0),
            n_search=n_search,
        )


# ============================================================================
# Lagrangian: Running cost L(x, α, m, t) with Legendre transform - Issue #651
# ============================================================================


class LagrangianBase(MFGOperatorBase):
    """
    Abstract base for Lagrangian (running cost) L(x, α, m, t).

    The Lagrangian represents the instantaneous cost of being at position x
    with control α in density m at time t. It is related to the Hamiltonian
    via Legendre transform:

        H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }   (for MINIMIZE)
        L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }   (inverse transform)

    This symmetric duality (Issue #651) allows users to specify either:
    - The Hamiltonian directly (what solvers need)
    - The Lagrangian (often more intuitive for cost specification)

    Both can be converted to the other via Legendre transform.

    Mathematical Structure
    ----------------------
    Common Lagrangian forms:

    1. Separable: L(x, α, m, t) = L_control(α) + V(x, t) + f(m)
       - L_control(α) = ½λ|α|² (quadratic)
       - V(x, t): Potential energy / running state cost
       - f(m): Density penalty (congestion)

    2. Non-separable: L depends on (x, α, m, t) in coupled way
       - E.g., state-dependent control cost
       - Requires numerical Legendre transform

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost (MINIMIZE) or maximize utility (MAXIMIZE)

    Examples
    --------
    Define a quadratic running cost:

    >>> class QuadraticLagrangian(LagrangianBase):
    ...     def __init__(self, control_cost=1.0):
    ...         super().__init__()
    ...         self.lam = control_cost
    ...
    ...     def __call__(self, x, alpha, m, t=0.0):
    ...         return 0.5 * self.lam * np.sum(alpha**2)  # L = ½λ|α|²

    Convert to Hamiltonian and back:

    >>> L = QuadraticLagrangian(control_cost=2.0)
    >>> H = L.legendre_transform()  # L → H via Legendre transform
    >>> L2 = H.legendre_transform()  # H → L via inverse Legendre transform
    """

    @property
    def is_hamiltonian(self) -> bool:
        """Return False - this is not a Hamiltonian operator."""
        return False

    @property
    def is_lagrangian(self) -> bool:
        """Return True - this is a Lagrangian operator."""
        return True

    @abstractmethod
    def __call__(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Evaluate Lagrangian L(x, α, m, t).

        Parameters
        ----------
        x : NDArray
            Position, shape (d,)
        alpha : NDArray
            Control, shape (d,)
        m : float | NDArray
            Density at x
        t : float
            Time

        Returns
        -------
        float | NDArray
            Running cost value
        """
        ...

    def legendre_transform(
        self,
        alpha_bounds: tuple[float, float] | None = None,
        n_search: int = 100,
    ) -> HamiltonianBase:
        """
        Convert Lagrangian to Hamiltonian via Legendre transform.

        Computes H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }

        The Legendre transform is involutive: applying it twice recovers
        the original (up to convexification). This provides symmetric
        duality between H and L (Issue #651).

        For separable quadratic Lagrangians, this is done analytically.
        For general Lagrangians, numerical optimization is used.

        Parameters
        ----------
        alpha_bounds : tuple[float, float] | None
            Bounds on control for numerical optimization.
            If None, uses (-10, 10) as default.
        n_search : int
            Number of points for grid search (default: 100)

        Returns
        -------
        HamiltonianBase
            The Legendre-transformed Hamiltonian (DualHamiltonian)

        Examples
        --------
        >>> L = MyLagrangian()
        >>> H = L.legendre_transform()  # L → H
        >>> L_back = H.legendre_transform()  # Involutive: back to Lagrangian
        """
        return DualHamiltonian(
            lagrangian=self,
            sense=self.sense,
            alpha_bounds=alpha_bounds or (-10.0, 10.0),
            n_search=n_search,
        )


class DualHamiltonian(HamiltonianBase):
    """
    Hamiltonian defined via Legendre transform of a Lagrangian.

    This class computes H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }
    numerically for general Lagrangians. It is the "dual" of a given
    Lagrangian in the sense of Legendre/convex duality.

    For separable quadratic Lagrangians, use SeparableHamiltonian instead
    which has analytic formulas.

    Parameters
    ----------
    lagrangian : LagrangianBase
        The Lagrangian to transform
    sense : OptimizationSense
        Optimization direction
    alpha_bounds : tuple[float, float]
        Bounds on control for optimization
    n_search : int
        Number of points for initial grid search

    Notes
    -----
    The numerical Legendre transform uses a two-stage approach:
    1. Grid search to find approximate optimum
    2. Local refinement using scipy.optimize (if available)

    See Also
    --------
    DualLagrangian : Lagrangian created from Hamiltonian via Legendre transform
    LagrangianBase.legendre_transform : Symmetric operation (L → H)
    """

    def __init__(
        self,
        lagrangian: LagrangianBase,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        alpha_bounds: tuple[float, float] = (-10.0, 10.0),
        n_search: int = 100,
    ):
        super().__init__(sense=sense)
        self.lagrangian = lagrangian
        self.alpha_bounds = alpha_bounds
        self.n_search = n_search

    def __call__(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute H via numerical Legendre transform.

        H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }
        """
        d = p.shape[0] if p.ndim > 0 else 1
        p_flat = np.atleast_1d(p)

        # For 1D, use simple grid search + refinement
        if d == 1:
            alpha_grid = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], self.n_search)
            values = np.array(
                [float(p_flat[0]) * a - float(self.lagrangian(x, np.array([a]), m, t)) for a in alpha_grid]
            )

            if self.sense == OptimizationSense.MINIMIZE:
                return float(np.max(values))  # sup for minimization
            else:
                return float(np.min(values))  # inf for maximization

        # For higher dimensions, use scipy if available
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(alpha):
                # Minimize negative of (p·α - L)
                return -(np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t)))

            # Initial guess: project p onto bounds
            x0 = np.clip(p_flat, self.alpha_bounds[0], self.alpha_bounds[1])
            bounds = [self.alpha_bounds] * d

            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return float(-result.fun)

        except ImportError:
            # Fallback: grid search in each dimension
            from itertools import product

            alpha_1d = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], 20)
            best_val = -np.inf

            for alpha_tuple in product(alpha_1d, repeat=d):
                alpha = np.array(alpha_tuple)
                val = np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t))
                best_val = max(best_val, val)

            return float(best_val)

    def dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute ∂H/∂p = α* (optimal control).

        By envelope theorem, ∂H/∂p equals the optimal control α*.
        """
        return self._find_optimal_alpha(x, m, p, t)

    def _find_optimal_alpha(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float,
    ) -> NDArray:
        """Find α* = argmax_α { p·α - L(x, α, m, t) }."""
        d = p.shape[0] if p.ndim > 0 else 1
        p_flat = np.atleast_1d(p)

        if d == 1:
            alpha_grid = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], self.n_search)
            values = np.array(
                [float(p_flat[0]) * a - float(self.lagrangian(x, np.array([a]), m, t)) for a in alpha_grid]
            )
            best_idx = np.argmax(values)
            return np.array([alpha_grid[best_idx]])

        # Higher dimensions: scipy or grid
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(alpha):
                return -(np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t)))

            x0 = np.clip(p_flat, self.alpha_bounds[0], self.alpha_bounds[1])
            bounds = [self.alpha_bounds] * d
            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return result.x

        except ImportError:
            from itertools import product

            alpha_1d = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], 20)
            best_val = -np.inf
            best_alpha = np.zeros(d)

            for alpha_tuple in product(alpha_1d, repeat=d):
                alpha = np.array(alpha_tuple)
                val = np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t))
                if val > best_val:
                    best_val = val
                    best_alpha = alpha

            return best_alpha


class DualLagrangian(LagrangianBase):
    """
    Lagrangian defined via inverse Legendre transform of a Hamiltonian.

    This class computes L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }
    numerically for general Hamiltonians. It is the "dual" of a given
    Hamiltonian in the sense of Legendre/convex duality.

    For separable quadratic Hamiltonians, the inverse transform gives
    back the quadratic Lagrangian analytically.

    Parameters
    ----------
    hamiltonian : HamiltonianBase
        The Hamiltonian to inverse-transform
    sense : OptimizationSense
        Optimization direction
    p_bounds : tuple[float, float]
        Bounds on momentum for optimization
    n_search : int
        Number of points for initial grid search

    See Also
    --------
    DualHamiltonian : Hamiltonian created from Lagrangian via Legendre transform
    HamiltonianBase.legendre_transform : Symmetric operation (H → L)
    """

    def __init__(
        self,
        hamiltonian: HamiltonianBase,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        p_bounds: tuple[float, float] = (-10.0, 10.0),
        n_search: int = 100,
    ):
        super().__init__(sense=sense)
        self.hamiltonian = hamiltonian
        self.p_bounds = p_bounds
        self.n_search = n_search

    def __call__(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute L via inverse Legendre transform.

        L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }
        """
        d = alpha.shape[0] if alpha.ndim > 0 else 1
        alpha_flat = np.atleast_1d(alpha)

        # For 1D, use simple grid search
        if d == 1:
            p_grid = np.linspace(self.p_bounds[0], self.p_bounds[1], self.n_search)
            values = np.array(
                [float(p) * float(alpha_flat[0]) - float(self.hamiltonian(x, m, np.array([p]), t)) for p in p_grid]
            )

            if self.sense == OptimizationSense.MINIMIZE:
                return float(np.max(values))
            else:
                return float(np.min(values))

        # For higher dimensions, use scipy if available
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(p):
                return -(np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t)))

            x0 = np.clip(alpha_flat, self.p_bounds[0], self.p_bounds[1])
            bounds = [self.p_bounds] * d
            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return float(-result.fun)

        except ImportError:
            from itertools import product

            p_1d = np.linspace(self.p_bounds[0], self.p_bounds[1], 20)
            best_val = -np.inf

            for p_tuple in product(p_1d, repeat=d):
                p = np.array(p_tuple)
                val = np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t))
                best_val = max(best_val, val)

            return float(best_val)

    def d_alpha(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute ∂L/∂α = p* (optimal momentum).

        By envelope theorem, ∂L/∂α equals the optimal momentum p*.
        """
        return self._find_optimal_p(x, alpha, m, t)

    def dm(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> float:
        """Compute ∂L/∂m using finite differences."""
        eps = self.finite_diff_eps
        m_scalar = float(m) if np.isscalar(m) else float(np.mean(m))

        L_plus = self(x, alpha, m_scalar + eps, t)
        L_minus = self(x, alpha, m_scalar - eps, t)

        return float((L_plus - L_minus) / (2 * eps))

    def _find_optimal_p(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float,
    ) -> NDArray:
        """Find p* = argmax_p { p·α - H(x, p, m, t) }."""
        d = alpha.shape[0] if alpha.ndim > 0 else 1
        alpha_flat = np.atleast_1d(alpha)

        if d == 1:
            p_grid = np.linspace(self.p_bounds[0], self.p_bounds[1], self.n_search)
            values = np.array(
                [float(p) * float(alpha_flat[0]) - float(self.hamiltonian(x, m, np.array([p]), t)) for p in p_grid]
            )
            best_idx = np.argmax(values)
            return np.array([p_grid[best_idx]])

        # Higher dimensions: scipy or grid
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(p):
                return -(np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t)))

            x0 = np.clip(alpha_flat, self.p_bounds[0], self.p_bounds[1])
            bounds = [self.p_bounds] * d
            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return result.x

        except ImportError:
            from itertools import product

            p_1d = np.linspace(self.p_bounds[0], self.p_bounds[1], 20)
            best_val = -np.inf
            best_p = np.zeros(d)

            for p_tuple in product(p_1d, repeat=d):
                p = np.array(p_tuple)
                val = np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t))
                if val > best_val:
                    best_val = val
                    best_p = p

            return best_p


# ============================================================================
# Concrete Implementations
# ============================================================================


class SeparableHamiltonian(HamiltonianBase):
    """
    Separable Hamiltonian: H(x, m, p, t) = H_control(p) + V(x, t) + f(m).

    This is the most common form in MFG, where:
    - H_control(p): Control cost (from ControlCostBase)
    - V(x, t): Potential energy / state cost
    - f(m): Density coupling term

    The separability allows efficient computation and analytic derivatives.

    Parameters
    ----------
    control_cost : ControlCostBase
        Control cost specification (quadratic, L1, bounded, etc.)
    potential : Callable[[NDArray, float], float] | None
        Potential function V(x, t). If None, V = 0.
    coupling : Callable[[float | NDArray], float | NDArray] | None
        Density coupling f(m). If None, f = 0.
    coupling_dm : Callable[[float | NDArray], float | NDArray] | None
        Derivative df/dm. If None, computed via finite differences.

    Examples
    --------
    Standard MFG with quadratic control, no potential, m² coupling:

    >>> H = SeparableHamiltonian(
    ...     control_cost=QuadraticControlCost(control_cost=1.0),
    ...     coupling=lambda m: -m**2,
    ...     coupling_dm=lambda m: -2*m,  # Analytic derivative
    ... )
    >>> H(x=np.array([0.5]), m=0.3, p=np.array([1.0]), t=0.0)

    With potential field:

    >>> def potential(x, t):
    ...     return np.sin(2 * np.pi * x[0])  # Periodic potential
    >>>
    >>> H = SeparableHamiltonian(
    ...     control_cost=QuadraticControlCost(),
    ...     potential=potential,
    ... )
    """

    def __init__(
        self,
        control_cost: ControlCostBase,
        potential: callable | None = None,
        coupling: callable | None = None,
        coupling_dm: callable | None = None,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
    ):
        super().__init__(sense=sense)
        self.control_cost = control_cost
        self._potential = potential
        self._coupling = coupling
        self._coupling_dm = coupling_dm

    def __call__(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Evaluate H = H_control(p) + V(x, t) + f(m).
        """
        # Control cost term
        H_control = self.control_cost.hamiltonian(np.atleast_1d(p))
        if isinstance(H_control, np.ndarray):
            H_control = float(H_control.sum())

        # Potential term
        V = 0.0
        if self._potential is not None:
            V = float(self._potential(x, t))

        # Coupling term
        f_m = 0.0
        if self._coupling is not None:
            f_m = float(self._coupling(m))

        return H_control + V + f_m

    def dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Analytic ∂H/∂p from control cost.

        For separable H, ∂H/∂p = ∂H_control/∂p = α* (the unconstrained optimum).
        """
        # For quadratic: ∂H/∂p = p/λ
        # We derive from optimal_control: α* = -sign * ∂H/∂p
        # So ∂H/∂p = -sign * α*

        # Actually, for control_cost.optimal_control(p) = -sign * p/λ
        # We have ∂H/∂p = p/λ (for quadratic)
        # Let's compute directly

        p_flat = np.atleast_1d(p)

        if isinstance(self.control_cost, QuadraticControlCost):
            # H_control = ½|p|²/λ → ∂H/∂p = p/λ
            return p_flat / self.control_cost.control_cost

        elif isinstance(self.control_cost, BoundedControlCost):
            # Same as quadratic in interior, saturates at bounds
            unconstrained = p_flat / self.control_cost.control_cost
            # The derivative is still p/λ but limited by constraint activity
            return unconstrained

        else:
            # Fallback to finite differences
            return self._finite_diff_dp(x, m, p, t)

    def dm(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute ∂H/∂m = df/dm (only coupling term depends on m).
        """
        if self._coupling_dm is not None:
            return float(self._coupling_dm(m))

        if self._coupling is None:
            return 0.0

        # Finite difference fallback
        return self._finite_diff_dm(x, m, p, t)

    def optimal_control(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Optimal control from the control cost specification.

        For separable Hamiltonians, optimal control depends only on p,
        not on x, m, or t.
        """
        return self.control_cost.optimal_control(np.atleast_1d(p))


class QuadraticMFGHamiltonian(SeparableHamiltonian):
    """
    Standard quadratic MFG Hamiltonian: H = ½c|p|² - V(x) - m².

    This is the Hamiltonian used when no custom hamiltonian_func is provided
    in MFGComponents. It represents the most common form in MFG literature:

    - Quadratic control cost: H_control = ½c|p|²
    - Optional potential: V(x, t)
    - Quadratic density coupling: f(m) = -m²

    Parameters
    ----------
    coupling_coefficient : float
        Coefficient c in ½c|p|² (default: 1.0)
    potential : Callable | None
        Potential V(x, t) (default: None, meaning V=0)
    sense : OptimizationSense
        Optimization direction

    Notes
    -----
    The default coupling f(m) = -m² gives ∂H/∂m = -2m.
    This Hamiltonian leads to the classical optimal control:
    α* = -c·p (for MINIMIZE sense).
    """

    def __init__(
        self,
        coupling_coefficient: float = 1.0,
        potential: callable | None = None,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
    ):
        super().__init__(
            control_cost=QuadraticControlCost(
                sense=sense,
                control_cost=1.0 / coupling_coefficient if coupling_coefficient > 0 else 1.0,
            ),
            potential=potential,
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
            sense=sense,
        )
        self.coupling_coefficient = coupling_coefficient


# ============================================================================
# Factory and Utilities
# ============================================================================


def create_hamiltonian(
    hamiltonian_type: str = "quadratic",
    **kwargs,
) -> HamiltonianBase:
    """
    Factory function to create Hamiltonians by type name.

    Parameters
    ----------
    hamiltonian_type : str
        One of: "quadratic", "l1", "bounded", "default", "separable"
    **kwargs
        Type-specific parameters

    Returns
    -------
    HamiltonianBase
        The created Hamiltonian

    Examples
    --------
    >>> H = create_hamiltonian("quadratic", control_cost=2.0)
    >>> H = create_hamiltonian("default", coupling_coefficient=0.5)
    """
    sense = kwargs.pop("sense", OptimizationSense.MINIMIZE)

    if hamiltonian_type == "quadratic":
        control_cost = kwargs.get("control_cost", 1.0)
        return SeparableHamiltonian(
            control_cost=QuadraticControlCost(sense=sense, control_cost=control_cost),
            sense=sense,
        )

    elif hamiltonian_type == "l1":
        control_cost = kwargs.get("control_cost", 1.0)
        return SeparableHamiltonian(
            control_cost=L1ControlCost(sense=sense, control_cost=control_cost),
            sense=sense,
        )

    elif hamiltonian_type == "bounded":
        control_cost = kwargs.get("control_cost", 1.0)
        max_control = kwargs.get("max_control", 1.0)
        return SeparableHamiltonian(
            control_cost=BoundedControlCost(sense=sense, control_cost=control_cost, max_control=max_control),
            sense=sense,
        )

    elif hamiltonian_type == "default":
        coupling_coefficient = kwargs.get("coupling_coefficient", 1.0)
        potential = kwargs.get("potential")
        return QuadraticMFGHamiltonian(
            coupling_coefficient=coupling_coefficient,
            potential=potential,
            sense=sense,
        )

    elif hamiltonian_type == "separable":
        control_cost = kwargs.get(
            "control_cost",
            QuadraticControlCost(sense=sense),
        )
        potential = kwargs.get("potential")
        coupling = kwargs.get("coupling")
        coupling_dm = kwargs.get("coupling_dm")
        return SeparableHamiltonian(
            control_cost=control_cost,
            potential=potential,
            coupling=coupling,
            coupling_dm=coupling_dm,
            sense=sense,
        )

    else:
        raise ValueError(
            f"Unknown hamiltonian_type: {hamiltonian_type}. Valid types: quadratic, l1, bounded, default, separable"
        )


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing Hamiltonian abstractions...")
    print("=" * 60)

    # Test QuadraticControlCost
    print("\n1. QuadraticControlCost (MINIMIZE):")
    cost = QuadraticControlCost(sense=OptimizationSense.MINIMIZE, control_cost=2.0)
    p = np.array([1.0, 2.0, -3.0])
    alpha = cost.optimal_control(p)
    print(f"   p = {p}")
    print(f"   α* = {alpha}  (expected: [-0.5, -1.0, 1.5])")
    assert np.allclose(alpha, [-0.5, -1.0, 1.5]), "QuadraticControlCost MINIMIZE failed"

    print("\n2. QuadraticControlCost (MAXIMIZE):")
    cost_max = QuadraticControlCost(sense=OptimizationSense.MAXIMIZE, control_cost=2.0)
    alpha_max = cost_max.optimal_control(p)
    print(f"   α* = {alpha_max}  (expected: [0.5, 1.0, -1.5])")
    assert np.allclose(alpha_max, [0.5, 1.0, -1.5]), "QuadraticControlCost MAXIMIZE failed"

    # Test L1ControlCost
    print("\n3. L1ControlCost (bang-bang):")
    cost_l1 = L1ControlCost(control_cost=1.5)
    p_l1 = np.array([0.5, 2.0, -3.0])
    alpha_l1 = cost_l1.optimal_control(p_l1)
    print(f"   p = {p_l1}, threshold = 1.5")
    print(f"   α* = {alpha_l1}  (expected: [0, -1, 1])")
    assert np.allclose(alpha_l1, [0, -1, 1]), "L1ControlCost failed"

    # Test BoundedControlCost
    print("\n4. BoundedControlCost:")
    cost_bounded = BoundedControlCost(control_cost=1.0, max_control=1.5)
    p_bounded = np.array([1.0, 2.0, 3.0])
    alpha_bounded = cost_bounded.optimal_control(p_bounded)
    print(f"   p = {p_bounded}, max_control = 1.5")
    print(f"   α* = {alpha_bounded}  (expected: [-1, -1.5, -1.5])")
    assert np.allclose(alpha_bounded, [-1, -1.5, -1.5]), "BoundedControlCost failed"

    print("\n" + "=" * 60)
    print("Testing Hamiltonian classes (Issue #673)...")
    print("=" * 60)

    # Test SeparableHamiltonian
    print("\n5. SeparableHamiltonian (quadratic control):")
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=2.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    x = np.array([0.5])
    m_val = 0.3
    p_val = np.array([1.0])
    t_val = 0.0

    H_val = H(x, m_val, p_val, t_val)
    # H = ½|p|²/λ + f(m) = 0.5 * 1.0 / 2.0 + (-0.09) = 0.25 - 0.09 = 0.16
    print(f"   H(x={x}, m={m_val}, p={p_val}) = {H_val:.4f}")
    print("   Expected: 0.5 * 1.0² / 2.0 - 0.3² = 0.25 - 0.09 = 0.16")
    assert abs(H_val - 0.16) < 1e-10, f"SeparableHamiltonian value failed: {H_val}"

    # Test dp (analytic)
    dp_val = H.dp(x, m_val, p_val, t_val)
    print(f"   ∂H/∂p = {dp_val}  (expected: p/λ = [0.5])")
    assert np.allclose(dp_val, [0.5]), f"SeparableHamiltonian dp failed: {dp_val}"

    # Test dm (analytic)
    dm_val = H.dm(x, m_val, p_val, t_val)
    print(f"   ∂H/∂m = {dm_val:.4f}  (expected: -2m = -0.6)")
    assert abs(dm_val - (-0.6)) < 1e-10, f"SeparableHamiltonian dm failed: {dm_val}"

    # Test optimal control
    alpha_opt = H.optimal_control(x, m_val, p_val, t_val)
    print(f"   α* = {alpha_opt}  (expected: -p/λ = [-0.5])")
    assert np.allclose(alpha_opt, [-0.5]), "SeparableHamiltonian optimal_control failed"

    # Test QuadraticMFGHamiltonian (and backward-compat alias DefaultMFGHamiltonian)
    print("\n6. QuadraticMFGHamiltonian:")
    H_default = QuadraticMFGHamiltonian(coupling_coefficient=1.0)
    H_default_val = H_default(x, m_val, p_val, t_val)
    # H = ½c|p|² - m² = 0.5 * 1.0 * 1.0 - 0.09 = 0.5 - 0.09 = 0.41
    print(f"   H(x={x}, m={m_val}, p={p_val}) = {H_default_val:.4f}")
    print("   Expected: 0.5 * 1.0 * 1.0² - 0.3² = 0.5 - 0.09 = 0.41")
    assert abs(H_default_val - 0.41) < 1e-10, f"QuadraticMFGHamiltonian failed: {H_default_val}"

    # Test factory function
    print("\n7. create_hamiltonian factory:")
    H_factory = create_hamiltonian("quadratic", control_cost=2.0)
    H_factory_val = H_factory(x, m_val, p_val, t_val)
    print("   create_hamiltonian('quadratic', control_cost=2.0)")
    print(f"   H = {H_factory_val:.4f}  (expected: 0.25)")
    # H = ½|p|²/λ = 0.5 * 1.0 / 2.0 = 0.25
    assert abs(H_factory_val - 0.25) < 1e-10, "Factory Hamiltonian failed"

    # Issue #673: to_legacy_func() removed - test class-based API directly
    print("\n8. Class-based Hamiltonian direct calls:")
    H_class = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )

    # Call class-based API directly: H(x, m, p, t)
    p_test = np.array([2.0])  # p = 2.0 in 1D
    m_test = 0.3
    H_direct = H_class(x, m_test, p_test, t_val)
    # H = ½|p|²/λ - m² = 0.5 * 4.0 / 1.0 - 0.09 = 2.0 - 0.09 = 1.91
    print(f"   H(x, m=0.3, p=2.0, t) = {H_direct:.4f}")
    print("   Expected: 0.5 * 2² - 0.3² = 2.0 - 0.09 = 1.91")
    assert abs(H_direct - 1.91) < 1e-10, f"Class-based H failed: {H_direct}"

    dm_direct = H_class.dm(x, m_test, p_test, t_val)
    print(f"   H.dm(x, m=0.3, p, t) = {dm_direct:.4f}  (expected: -0.6)")
    assert abs(dm_direct - (-0.6)) < 1e-10, "Class-based dm failed"

    print("\n" + "=" * 60)
    print("Testing Lagrangian and Legendre transform (Issue #651)...")
    print("=" * 60)

    # Create a simple quadratic Lagrangian
    class TestQuadraticLagrangian(LagrangianBase):
        def __init__(self, lam=1.0):
            super().__init__()
            self.lam = lam

        def __call__(self, x, alpha, m, t=0.0):
            return 0.5 * self.lam * np.sum(alpha**2)

    print("\n9. Lagrangian -> Hamiltonian via Legendre transform:")
    L = TestQuadraticLagrangian(lam=2.0)
    H_legendre = L.legendre_transform()

    # For L = ½λ|α|², the Legendre transform gives H = ½|p|²/λ
    # With λ=2 and p=1: H = 0.5 * 1 / 2 = 0.25
    H_legendre_val = H_legendre(x, m_val, p_val, t_val)
    print("   L(α) = ½ * 2 * |α|² -> H(p) = ½|p|²/2")
    print(f"   H(p=1) = {H_legendre_val:.4f}  (expected: ~0.25)")
    # Allow some tolerance for numerical Legendre transform
    assert abs(H_legendre_val - 0.25) < 0.05, f"Legendre transform failed: {H_legendre_val}"

    print("\n10. Hamiltonian -> Lagrangian via inverse Legendre transform:")
    # Test symmetric duality: H -> L -> H should recover original
    H_orig = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=2.0),
    )
    L_from_H = H_orig.legendre_transform()

    # For H = ½|p|²/λ, the inverse Legendre transform gives L = ½λ|α|²
    # With λ=2 (control_cost=2): L(α=1) = 0.5 * 2 * 1 = 1.0
    alpha_test = np.array([1.0])
    L_val = L_from_H(x, alpha_test, m_val, t_val)
    print("   H(p) = ½|p|²/2 -> L(α) = ½ * 2 * |α|²")
    print(f"   L(α=1) = {L_val:.4f}  (expected: ~1.0)")
    assert abs(L_val - 1.0) < 0.1, f"Inverse Legendre transform failed: {L_val}"

    print("\n11. Symmetric duality: L -> H -> L should recover original:")
    L_orig = TestQuadraticLagrangian(lam=2.0)
    H_from_L = L_orig.legendre_transform()
    L_recovered = H_from_L.legendre_transform()

    # L_recovered(α=1) should ≈ L_orig(α=1) = 0.5 * 2 * 1 = 1.0
    L_orig_val = L_orig(x, alpha_test, m_val, t_val)
    L_recovered_val = L_recovered(x, alpha_test, m_val, t_val)
    print(f"   L_orig(α=1) = {L_orig_val:.4f}")
    print(f"   L_recovered(α=1) = {L_recovered_val:.4f}")
    assert abs(L_recovered_val - L_orig_val) < 0.2, f"Duality cycle failed: {L_recovered_val} vs {L_orig_val}"
    print("   Duality cycle: L -> H -> L verified!")

    print("\n12. MFGOperator properties:")
    print(f"   H.is_hamiltonian = {H_orig.is_hamiltonian}  (expected: True)")
    print(f"   H.is_lagrangian = {H_orig.is_lagrangian}    (expected: False)")
    print(f"   L.is_hamiltonian = {L_orig.is_hamiltonian}  (expected: False)")
    print(f"   L.is_lagrangian = {L_orig.is_lagrangian}    (expected: True)")
    assert H_orig.is_hamiltonian is True
    assert H_orig.is_lagrangian is False
    assert L_orig.is_hamiltonian is False
    assert L_orig.is_lagrangian is True

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
