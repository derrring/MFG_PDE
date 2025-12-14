"""
Protocol-based convergence checkers for MFG solvers.

This module provides a Strategy pattern implementation for convergence checking,
allowing different solver types to use appropriate convergence criteria.

The design decouples convergence logic from solver implementation, enabling:
- Consistent convergence checking across all solver paradigms
- Easy swapping of convergence strategies
- Unified diagnostics and metrics collection

Usage:
    from mfg_pde.utils.convergence import (
        ConvergenceConfig,
        MFGConvergenceChecker,
    )

    config = ConvergenceConfig(tolerance=1e-6, require_both=True)
    checker = MFGConvergenceChecker(config)

    # In solver loop
    converged, metrics = checker.check(old_state, new_state)
    if converged:
        break
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .convergence_metrics import ConvergenceConfig

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_norm(
    diff: NDArray[np.floating],
    method: Literal["l1", "l2", "linf"] = "l2",
) -> float:
    """
    Compute norm of difference array.

    Args:
        diff: Difference array (new - old)
        method: Norm type ('l1', 'l2', 'linf')

    Returns:
        Scalar norm value
    """
    if method == "l1":
        return float(np.sum(np.abs(diff)))
    elif method == "linf":
        return float(np.max(np.abs(diff)))
    else:  # l2 default
        return float(np.sqrt(np.sum(diff**2)))


def _check_divergence(error: float, threshold: float = 1e10) -> tuple[bool, str]:
    """
    Check for divergence conditions.

    Args:
        error: Computed error value
        threshold: Safety threshold for divergence detection

    Returns:
        Tuple of (is_diverged, status_string)
    """
    if np.isnan(error):
        return True, "DIVERGED_NAN"
    if np.isinf(error):
        return True, "DIVERGED_INF"
    if error > threshold:
        return True, "DIVERGED_THRESHOLD"
    return False, "OK"


# =============================================================================
# CONVERGENCE CHECKER PROTOCOL
# =============================================================================


class ConvergenceChecker(Protocol):
    """
    Protocol for convergence checkers.

    All convergence checkers must implement this interface, allowing
    solvers to work with any checker type polymorphically.

    The check method receives explicit old and new states (not a single
    combined dict) to keep solver memory management explicit and enable
    stateless checkers.
    """

    def check(
        self,
        old_state: dict[str, NDArray[np.floating]],
        new_state: dict[str, NDArray[np.floating]],
    ) -> tuple[bool, dict[str, float]]:
        """
        Check convergence between old and new states.

        Args:
            old_state: Previous iteration state {'U': array, 'M': array, ...}
            new_state: Current iteration state {'U': array, 'M': array, ...}

        Returns:
            Tuple of:
                - bool: True if converged, False otherwise
                - dict: Diagnostic metrics (e.g., {'error_U': 0.001, 'status': 'OK'})
        """
        ...


# =============================================================================
# HJB CONVERGENCE CHECKER
# =============================================================================


class HJBConvergenceChecker:
    """
    Convergence checker for standalone HJB (Hamilton-Jacobi-Bellman) solvers.

    Checks convergence based on value function U:
        ||U_new - U_old|| < tolerance

    Args:
        config: ConvergenceConfig with tolerance settings
    """

    def __init__(self, config: ConvergenceConfig) -> None:
        self.config = config
        self.tolerance = config.tolerance_U if config.tolerance_U is not None else config.tolerance

    def check(
        self,
        old_state: dict[str, NDArray[np.floating]],
        new_state: dict[str, NDArray[np.floating]],
    ) -> tuple[bool, dict[str, float]]:
        """
        Check HJB convergence.

        Args:
            old_state: Must contain 'U' key
            new_state: Must contain 'U' key

        Returns:
            (converged, {'error_U': float, 'status': str})
        """
        U_old = old_state["U"]
        U_new = new_state["U"]

        diff = U_new - U_old
        error = compute_norm(diff, self.config.norm)

        # Relative error if requested
        if self.config.relative:
            norm_new = compute_norm(U_new, self.config.norm)
            if norm_new > 1e-15:  # Avoid division by zero
                error = error / norm_new

        # Check for divergence
        is_diverged, status = _check_divergence(error)
        if is_diverged:
            return False, {"error_U": error, "status": status}

        converged = error < self.tolerance
        return converged, {"error_U": error, "status": "OK"}


# =============================================================================
# FP CONVERGENCE CHECKER
# =============================================================================


class FPConvergenceChecker:
    """
    Convergence checker for standalone FP (Fokker-Planck) solvers.

    Checks convergence based on density M:
        ||M_new - M_old|| < tolerance

    Optionally tracks mass conservation error.

    Args:
        config: ConvergenceConfig with tolerance settings
        check_mass: Whether to compute mass conservation error
    """

    def __init__(self, config: ConvergenceConfig, check_mass: bool = True) -> None:
        self.config = config
        self.tolerance = config.tolerance_M if config.tolerance_M is not None else config.tolerance
        self.check_mass = check_mass

    def check(
        self,
        old_state: dict[str, NDArray[np.floating]],
        new_state: dict[str, NDArray[np.floating]],
    ) -> tuple[bool, dict[str, float]]:
        """
        Check FP convergence.

        Args:
            old_state: Must contain 'M' key
            new_state: Must contain 'M' key

        Returns:
            (converged, {'error_M': float, 'mass_error': float, 'status': str})
        """
        M_old = old_state["M"]
        M_new = new_state["M"]

        diff = M_new - M_old
        error = compute_norm(diff, self.config.norm)

        # Relative error if requested
        if self.config.relative:
            norm_new = compute_norm(M_new, self.config.norm)
            if norm_new > 1e-15:
                error = error / norm_new

        # Check for divergence
        is_diverged, status = _check_divergence(error)
        if is_diverged:
            metrics: dict[str, float] = {"error_M": error, "status": status}
            if self.check_mass:
                metrics["mass_error"] = float("nan")
            return False, metrics

        # Mass conservation check
        metrics = {"error_M": error, "status": "OK"}
        if self.check_mass:
            mass_old = float(np.sum(M_old))
            mass_new = float(np.sum(M_new))
            metrics["mass_error"] = abs(mass_new - mass_old)

        converged = error < self.tolerance
        return converged, metrics


# =============================================================================
# MFG CONVERGENCE CHECKER (COUPLED SYSTEM)
# =============================================================================


class MFGConvergenceChecker:
    """
    Convergence checker for coupled MFG (Mean Field Game) solvers.

    Orchestrates convergence checking for both HJB (U) and FP (M) components.
    Supports different tolerances for each component and configurable
    convergence criteria (both must converge vs either converges).

    Args:
        config: ConvergenceConfig with tolerance settings
        check_mass: Whether to compute mass conservation error for FP

    Example:
        config = ConvergenceConfig(
            tolerance_U=1e-6,
            tolerance_M=1e-4,  # FP often harder to converge
            require_both=True,
        )
        checker = MFGConvergenceChecker(config)

        # In solver loop
        old_state = {'U': U_old, 'M': M_old}
        new_state = {'U': U_new, 'M': M_new}
        converged, metrics = checker.check(old_state, new_state)
    """

    def __init__(self, config: ConvergenceConfig, check_mass: bool = True) -> None:
        self.config = config
        self.hjb_checker = HJBConvergenceChecker(config)
        self.fp_checker = FPConvergenceChecker(config, check_mass=check_mass)

    def check(
        self,
        old_state: dict[str, NDArray[np.floating]],
        new_state: dict[str, NDArray[np.floating]],
    ) -> tuple[bool, dict[str, float]]:
        """
        Check coupled MFG convergence.

        Args:
            old_state: Must contain 'U' and 'M' keys
            new_state: Must contain 'U' and 'M' keys

        Returns:
            (converged, {
                'error_U': float,
                'error_M': float,
                'mass_error': float,
                'converged_U': bool,
                'converged_M': bool,
                'status': str
            })
        """
        # Check each component
        u_converged, u_metrics = self.hjb_checker.check(old_state, new_state)
        m_converged, m_metrics = self.fp_checker.check(old_state, new_state)

        # Combine metrics
        metrics: dict[str, float] = {
            **u_metrics,
            **m_metrics,
            "converged_U": float(u_converged),
            "converged_M": float(m_converged),
        }

        # Check for any divergence
        if u_metrics.get("status", "OK") != "OK":
            metrics["status"] = u_metrics["status"]
            return False, metrics
        if m_metrics.get("status", "OK") != "OK":
            metrics["status"] = m_metrics["status"]
            return False, metrics

        # Determine overall convergence
        if self.config.require_both:
            converged = u_converged and m_converged
        else:
            converged = u_converged or m_converged

        metrics["status"] = "OK"
        return converged, metrics


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_convergence_checker(
    solver_type: Literal["hjb", "fp", "mfg"],
    config: ConvergenceConfig | None = None,
    **kwargs,
) -> ConvergenceChecker:
    """
    Factory function to create appropriate convergence checker.

    Args:
        solver_type: Type of solver ('hjb', 'fp', 'mfg')
        config: ConvergenceConfig (uses defaults if None)
        **kwargs: Additional arguments passed to checker constructor

    Returns:
        Appropriate ConvergenceChecker instance

    Example:
        checker = create_convergence_checker('mfg', ConvergenceConfig(tolerance=1e-6))
    """
    if config is None:
        config = ConvergenceConfig()

    if solver_type == "hjb":
        return HJBConvergenceChecker(config)
    elif solver_type == "fp":
        return FPConvergenceChecker(config, **kwargs)
    elif solver_type == "mfg":
        return MFGConvergenceChecker(config, **kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Expected 'hjb', 'fp', or 'mfg'.")
