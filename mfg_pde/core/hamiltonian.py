"""
Hamiltonian and Lagrangian abstractions for MFG optimal control (Issue #623).

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

Design Philosophy
-----------------
Users typically think in terms of **running cost** (Lagrangian), but solvers
need the **Hamiltonian** and **optimal control formula**. This module:

1. Accepts either Lagrangian or Hamiltonian specification
2. Provides `optimal_control(p)` - the single source of truth for drift
3. Handles sign conventions via `OptimizationSense`

For common cases (quadratic, L1), closed-form formulas are provided.
For general cases, numerical Legendre transform is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


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
    print("All smoke tests passed!")
