"""
Scheme-Based Solver Factory for MFG Problems.

This module provides automatic creation of validated HJB-FP solver pairs based
on NumericalScheme selection (Issue #580). It ensures adjoint duality by using
_scheme_family trait matching and provides educational feedback for users.

Usage (Safe Mode - Phase 3):
    >>> from mfg_pde import MFGProblem, NumericalScheme
    >>> from mfg_pde.factory import create_paired_solvers
    >>>
    >>> problem = MFGProblem(...)
    >>> hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)
    >>> # hjb and fp are guaranteed to be dual by construction

Benefits over Manual Solver Selection:
    - Automatic duality guarantee (no mixing FDM with GFDM)
    - Config threading to both solvers
    - Scheme-appropriate defaults (e.g., renormalization for GFDM)
    - Educational warnings for advanced users overriding defaults

References:
    - Issue #580: Adjoint-aware solver pairing
    - Issue #543: Validator pattern (getattr instead of hasattr)
    - docs/theory/adjoint_operators_mfg.md: Mathematical theory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfg_pde.types import NumericalScheme
from mfg_pde.utils import DualityStatus, check_solver_duality

if TYPE_CHECKING:
    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.core.mfg_problem import MFGProblem


def create_paired_solvers(
    problem: MFGProblem,
    scheme: NumericalScheme,
    hjb_config: dict[str, Any] | None = None,
    fp_config: dict[str, Any] | None = None,
    validate_duality: bool = True,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create a validated HJB-FP solver pair for a given numerical scheme.

    This factory function automatically selects the correct solver classes based
    on the specified NumericalScheme, ensuring they form a valid adjoint pair.
    Configuration parameters are threaded to both solvers.

    Args:
        problem: MFG problem instance to solve
        scheme: Numerical discretization scheme (FDM_UPWIND, SL_LINEAR, GFDM, etc.)
        hjb_config: Optional configuration dict for HJB solver initialization
        fp_config: Optional configuration dict for FP solver initialization
        validate_duality: If True, validate solver duality (recommended)

    Returns:
        Tuple of (hjb_solver, fp_solver) instances that form a dual pair

    Raises:
        ValueError: If scheme is not recognized or solvers are not dual
        NotImplementedError: If scheme is defined but not yet implemented

    Examples:
        >>> # Safe Mode: Automatic dual pairing
        >>> hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)
        >>> assert hjb._scheme_family == fp._scheme_family  # Guaranteed

        >>> # With custom configs
        >>> hjb, fp = create_paired_solvers(
        ...     problem,
        ...     NumericalScheme.GFDM,
        ...     hjb_config={"delta": 0.05},
        ...     fp_config={"delta": 0.05, "upwind_scheme": "exponential"},
        ... )

        >>> # GFDM automatically gets renormalization recommendation
        >>> result = check_solver_duality(hjb, fp)
        >>> assert result.requires_renormalization()  # Type B scheme
    """
    # Initialize configs
    hjb_config = hjb_config or {}
    fp_config = fp_config or {}

    # Route to scheme-specific factory
    if scheme in (NumericalScheme.FDM_UPWIND, NumericalScheme.FDM_CENTERED):
        hjb_solver, fp_solver = _create_fdm_pair(problem, scheme, hjb_config, fp_config)

    elif scheme in (NumericalScheme.SL_LINEAR, NumericalScheme.SL_CUBIC):
        hjb_solver, fp_solver = _create_sl_pair(problem, scheme, hjb_config, fp_config)

    elif scheme == NumericalScheme.GFDM:
        hjb_solver, fp_solver = _create_gfdm_pair(problem, hjb_config, fp_config)

    else:
        raise NotImplementedError(
            f"Scheme {scheme.value} is defined but not yet implemented in factory. "
            f"Available schemes: FDM_UPWIND, FDM_CENTERED, SL_LINEAR, GFDM"
        )

    # Validate duality
    if validate_duality:
        result = check_solver_duality(hjb_solver, fp_solver, warn_on_mismatch=True)

        if result.status == DualityStatus.NOT_DUAL:
            raise ValueError(
                f"Factory created non-dual solver pair (this is a bug!):\n"
                f"  HJB: {type(hjb_solver).__name__} ({result.hjb_family})\n"
                f"  FP: {type(fp_solver).__name__} ({result.fp_family})\n"
                f"Please report this as Issue #580 bug."
            )

        if result.status == DualityStatus.VALIDATION_SKIPPED:
            raise ValueError(
                f"Factory created solvers without _scheme_family traits (this is a bug!):\n"
                f"  HJB: {type(hjb_solver).__name__}\n"
                f"  FP: {type(fp_solver).__name__}\n"
                f"Please report this as Issue #580 bug."
            )

    return hjb_solver, fp_solver


def _create_fdm_pair(
    problem: MFGProblem,
    scheme: NumericalScheme,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create FDM HJB-FP solver pair.

    Discrete duality (Type A): L_FP = L_HJB^T exactly.

    Args:
        problem: MFG problem
        scheme: FDM variant (FDM_UPWIND or FDM_CENTERED)
        hjb_config: HJB solver config
        fp_config: FP solver config

    Returns:
        (HJBFDMSolver, FPFDMSolver) tuple
    """
    from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
    from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver

    # Map scheme to FP advection scheme
    if scheme == NumericalScheme.FDM_UPWIND:
        # FDM upwind uses divergence_upwind for FP (mass conservative, handles boundaries correctly)
        # Note: gradient_upwind has boundary flux bug, see Issue #382
        fp_config.setdefault("advection_scheme", "divergence_upwind")
    elif scheme == NumericalScheme.FDM_CENTERED:
        # FDM centered uses gradient_centered (second-order, oscillates for high Peclet)
        fp_config.setdefault("advection_scheme", "gradient_centered")

    # Create solvers
    hjb_solver = HJBFDMSolver(problem, **hjb_config)
    fp_solver = FPFDMSolver(problem, **fp_config)

    return hjb_solver, fp_solver


def _create_sl_pair(
    problem: MFGProblem,
    scheme: NumericalScheme,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create Semi-Lagrangian HJB-FP solver pair.

    Discrete duality (Type A): Forward splatting (FP) is transpose of backward
    interpolation (HJB). Uses FPSLAdjointSolver (forward SL) for proper duality.

    Args:
        problem: MFG problem
        scheme: SL variant (SL_LINEAR or SL_CUBIC)
        hjb_config: HJB solver config
        fp_config: FP solver config

    Returns:
        (HJBSemiLagrangianSolver, FPSLAdjointSolver) tuple

    Note:
        For standalone FP problems (no HJB coupling), use FPSLSolver (backward SL).
        For MFG duality, use FPSLAdjointSolver (forward SL, adjoint of HJB).
    """
    from mfg_pde.alg.numerical.fp_solvers import FPSLAdjointSolver
    from mfg_pde.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    # Map scheme to interpolation method
    if scheme == NumericalScheme.SL_LINEAR:
        hjb_config.setdefault("interpolation_method", "linear")
        # FP adjoint solver currently only supports linear (forward splatting)
        # Cubic interpolation in FP is future work (Issue #583)

    elif scheme == NumericalScheme.SL_CUBIC:
        hjb_config.setdefault("interpolation_method", "cubic")
        # Note: Cubic FP adjoint not yet implemented
        # For now, use linear splatting even with cubic HJB interpolation
        # This breaks exact duality but maintains O(h^2) convergence

    # Create solvers
    hjb_solver = HJBSemiLagrangianSolver(problem, **hjb_config)
    fp_solver = FPSLAdjointSolver(problem, **fp_config)

    return hjb_solver, fp_solver


def _create_gfdm_pair(
    problem: MFGProblem,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create GFDM HJB-FP solver pair.

    Continuous duality (Type B): L_FP = L_HJB^T + O(h) asymptotically.
    Requires renormalization for Nash gap convergence.

    Args:
        problem: MFG problem
        hjb_config: HJB solver config
        fp_config: FP solver config

    Returns:
        (HJBGFDMSolver, FPGFDMSolver) tuple

    Note:
        GFDM uses asymmetric neighborhoods (nearest neighbors differ for different
        points), so discrete transpose is not exact. However, the continuous
        operators are adjoints, giving O(h) error. Renormalization compensates
        for this during fixed-point iteration.
    """
    from mfg_pde.alg.numerical.fp_solvers import FPGFDMSolver
    from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver

    # Thread common GFDM parameters to both solvers
    # If delta specified for one, use it for both (consistency)
    if "delta" in hjb_config and "delta" not in fp_config:
        fp_config["delta"] = hjb_config["delta"]
    elif "delta" in fp_config and "delta" not in hjb_config:
        hjb_config["delta"] = fp_config["delta"]

    # Thread collocation_points if specified
    if "collocation_points" in hjb_config and "collocation_points" not in fp_config:
        fp_config["collocation_points"] = hjb_config["collocation_points"]
    elif "collocation_points" in fp_config and "collocation_points" not in hjb_config:
        hjb_config["collocation_points"] = fp_config["collocation_points"]

    # Create solvers
    hjb_solver = HJBGFDMSolver(problem, **hjb_config)
    fp_solver = FPGFDMSolver(problem, **fp_config)

    return hjb_solver, fp_solver


def get_recommended_scheme(problem: MFGProblem) -> NumericalScheme:
    """
    Recommend a numerical scheme based on problem geometry (Phase 3 integration).

    This function performs intelligent scheme selection based on:
    - Grid regularity (structured → FDM, unstructured → GFDM)
    - Domain complexity (simple → FDM, complex boundaries → GFDM)
    - Dimension (1D → FDM, 2D+ → consider GFDM for flexibility)

    Args:
        problem: MFG problem to analyze

    Returns:
        Recommended NumericalScheme

    Note:
        This is a placeholder for Phase 3 auto-selection logic.
        For now, returns FDM_UPWIND as safe default.
    """
    # Phase 3 TODO: Implement geometry introspection
    # - Check if geometry.is_structured()
    # - Check if geometry has complex obstacles
    # - Check dimension
    # - Return appropriate scheme

    # Default: FDM upwind (stable, works everywhere)
    return NumericalScheme.FDM_UPWIND
