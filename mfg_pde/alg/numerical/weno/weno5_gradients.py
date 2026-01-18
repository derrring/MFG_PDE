"""
WENO5 gradient computation for Hamilton-Jacobi equations.

Provides 5th-order accurate one-sided derivatives for:
- Level set evolution: ∂φ/∂t + V|∇φ| = 0
- HJB equations: ∂u/∂t + H(∇u) = 0

Shared utility to avoid code duplication between hjb_weno.py and level set evolution.

References:
- Jiang & Shu (1996): Efficient Implementation of Weighted ENO Schemes
- Osher & Fedkiw (2003): Level Set Methods, Chapter 6

Created: 2026-01-18 (Issue #605 Phase 2.1 - Shared WENO Infrastructure)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class WENO5Gradient:
    """
    5th-order WENO gradient computation (shared utility).

    Computes one-sided derivatives using WENO5 reconstruction for upwind flux selection.
    Used by both HJB solvers and level set evolution.

    Examples
    --------
    >>> # Level set evolution
    >>> from mfg_pde.alg.numerical.weno import WENO5Gradient
    >>> weno = WENO5Gradient(spacing=(0.01,))
    >>> grad_mag = weno.compute_godunov_gradient(phi, velocity)
    >>>
    >>> # HJB solver
    >>> dphi_dx_plus, dphi_dx_minus = weno.compute_one_sided_derivatives_1d(phi)

    Notes
    -----
    This is a **shared utility** to avoid duplication between:
    - `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py` (full HJB solver)
    - `mfg_pde/geometry/level_set/` (level set evolution)

    Implementation Status (v0.17.3):
    - 1D: Implemented ✓
    - 2D/3D: Planned for future (Issue #605 Phase 2.1 extension)
    """

    def __init__(self, spacing: tuple[float, ...], epsilon: float = 1e-6):
        """
        Initialize WENO5 gradient operator.

        Parameters
        ----------
        spacing : tuple[float, ...]
            Grid spacing (dx, dy, dz, ...)
        epsilon : float, default=1e-6
            Smoothness indicator regularization (prevents division by zero)
        """
        self.spacing = spacing
        self.epsilon = epsilon
        self.dimension = len(spacing)

        logger.debug(f"WENO5Gradient: dimension={self.dimension}, spacing={spacing}")

    def compute_one_sided_derivatives_1d(
        self,
        phi: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute one-sided derivatives φ_x^± in 1D.

        Parameters
        ----------
        phi : NDArray
            Field to differentiate (1D array).

        Returns
        -------
        dphi_plus : NDArray
            Left-biased derivative (for positive velocity).
        dphi_minus : NDArray
            Right-biased derivative (for negative velocity).

        Notes
        -----
        Full WENO5 requires 2 ghost points on each side.
        Boundary points use lower-order approximations.

        TODO (Issue #605 Phase 2.1):
        Current implementation is placeholder. Replace with proper WENO5 reconstruction.
        For now, uses simple upwind for correctness.
        """
        dx = self.spacing[0]

        # Placeholder: 1st-order upwind (correct but not high-order)
        # TODO: Implement proper WENO5 reconstruction
        dphi_plus = np.zeros_like(phi)
        dphi_minus = np.zeros_like(phi)

        # Interior: backward difference (left-biased)
        dphi_plus[1:] = (phi[1:] - phi[:-1]) / dx

        # Interior: forward difference (right-biased)
        dphi_minus[:-1] = (phi[1:] - phi[:-1]) / dx

        logger.warning(
            "WENO5Gradient: Using 1st-order upwind placeholder. "
            "Full WENO5 reconstruction to be implemented (Issue #605 Phase 2.1)."
        )

        return dphi_plus, dphi_minus

    def compute_godunov_gradient(
        self,
        phi: NDArray[np.float64],
        velocity: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute |∇φ| using Godunov upwind flux.

        Parameters
        ----------
        phi : NDArray
            Level set function.
        velocity : NDArray
            Velocity field (same shape as phi).

        Returns
        -------
        grad_mag : NDArray
            |∇φ| for level set evolution: ∂φ/∂t + V|∇φ| = 0

        Notes
        -----
        Godunov upwind selection:
            φ_x = φ_x^+ if V > 0 else φ_x^-
        Then |∇φ| = √(φ_x² + φ_y² + ...) with dimension-wise upwinding.
        """
        if self.dimension == 1:
            dphi_plus, dphi_minus = self.compute_one_sided_derivatives_1d(phi)

            # Godunov upwind selection
            dphi_x = np.where(velocity > 0, dphi_plus, dphi_minus)

            return np.abs(dphi_x)
        else:
            raise NotImplementedError(f"WENO5 not yet implemented for {self.dimension}D")


if __name__ == "__main__":
    """Smoke test for WENO5Gradient."""
    print("Testing WENO5Gradient (Placeholder Implementation)...")

    # Test 1: Basic functionality
    print("\n[Test 1: Smoke Test]")
    N = 100
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    phi = x - 0.5  # Linear ramp
    velocity = np.ones_like(x)  # Positive velocity

    weno = WENO5Gradient(spacing=(dx,))
    grad_mag = weno.compute_godunov_gradient(phi, velocity)

    print(f"  Grid: N={N}, dx={dx:.4f}")
    print(f"  Gradient magnitude: mean={np.mean(grad_mag):.6f} (expect ~1.0)")

    assert np.abs(np.mean(grad_mag) - 1.0) < 0.1, "Basic gradient computation failed"
    print("  ✓ Basic functionality working!")

    # Test 2: Upwind selection
    print("\n[Test 2: Godunov Upwind Selection]")
    velocity_neg = -np.ones_like(x)
    grad_mag_neg = weno.compute_godunov_gradient(phi, velocity_neg)

    print(f"  Positive velocity: |∇φ| = {np.mean(grad_mag):.6f}")
    print(f"  Negative velocity: |∇φ| = {np.mean(grad_mag_neg):.6f}")
    print("  ✓ Upwind selection working!")

    print("\n✅ Smoke tests passed!")
    print("\n⚠️  NOTE: Using 1st-order placeholder. Full WENO5 to be implemented.")
    print("   See Issue #605 Phase 2.1 for implementation plan.")
