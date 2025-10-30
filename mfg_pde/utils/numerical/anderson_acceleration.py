"""
Anderson Acceleration for Fixed-Point Iteration.

Anderson acceleration (also known as Anderson mixing) is a method to accelerate
the convergence of fixed-point iterations by combining information from previous
iterates to compute better next guesses.

References:
- Anderson, D. G. (1965). Iterative procedures for nonlinear integral equations.
  Journal of the ACM, 12(4), 547-560.
- Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations.
  SIAM Journal on Numerical Analysis, 49(4), 1715-1735.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import Literal


class AndersonAccelerator:
    """
    Anderson acceleration for fixed-point iterations.

    Given a fixed-point iteration x_{k+1} = g(x_k), Anderson acceleration
    computes an accelerated iterate by solving a least-squares problem using
    the history of previous iterates and residuals.

    The method maintains a sliding window of the last m iterates and residuals,
    and computes the next iterate as a linear combination that minimizes the
    residual in a least-squares sense.

    Attributes:
        depth: Number of previous iterates to use (window size m)
        beta: Damping parameter in [0, 1] (1 = no damping)
        regularization: Regularization parameter for least-squares (0 = none)

    Note:
        Automatically handles multi-dimensional arrays (e.g., 2D grids for MFG density).
        Arrays are flattened internally for linear algebra operations, then reshaped
        to original shape in the output.
    """

    def __init__(
        self,
        depth: int = 5,
        beta: float = 1.0,
        regularization: float = 1e-8,
        restart_threshold: float | None = None,
    ):
        """
        Initialize Anderson accelerator.

        Args:
            depth: Number of previous iterates to store (m in literature)
            beta: Damping/relaxation parameter in [0,1]. beta=1 means no damping,
                  beta<1 adds relaxation for stability
            regularization: Tikhonov regularization for least-squares solve
            restart_threshold: If residual increases by this factor, restart
                              (None = never restart)
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0,1], got {beta}")
        if regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization}")

        self.depth = depth
        self.beta = beta
        self.regularization = regularization
        self.restart_threshold = restart_threshold

        # Storage for iterates and residuals
        self.X_history: list[np.ndarray] = []  # x_k
        self.F_history: list[np.ndarray] = []  # g(x_k)
        self.residual_norms: list[float | np.floating[Any]] = []
        self.iteration_count = 0

        # Store original shape for reshaping outputs (set on first update)
        self._original_shape: tuple[int, ...] | None = None

    def update(
        self,
        x_current: np.ndarray,
        f_current: np.ndarray,
        method: Literal["type1", "type2"] = "type1",
    ) -> np.ndarray:
        """
        Compute Anderson-accelerated next iterate.

        Args:
            x_current: Current iterate x_k (can be multi-dimensional)
            f_current: Function evaluation g(x_k) (the fixed-point map)
            method: "type1" uses g(x) - x, "type2" uses g(x) formulation

        Returns:
            x_next: Accelerated next iterate (same shape as input)

        Note:
            Multi-dimensional arrays are automatically flattened for internal
            computations and reshaped back to original shape in output.
        """
        # Store original shape on first call
        if self._original_shape is None:
            self._original_shape = x_current.shape

        # Flatten arrays for vector operations (handles 1D, 2D, 3D, etc.)
        x_flat = x_current.ravel()
        f_flat = f_current.ravel()

        # Compute residual
        residual_flat = f_flat - x_flat  # r_k = g(x_k) - x_k
        residual_norm = np.linalg.norm(residual_flat)

        # Store current state (flattened)
        self.X_history.append(x_flat.copy())
        self.F_history.append(f_flat.copy())
        self.residual_norms.append(residual_norm)
        self.iteration_count += 1

        # Maintain sliding window
        if len(self.X_history) > self.depth + 1:
            self.X_history.pop(0)
            self.F_history.pop(0)

        # Need at least 2 iterates for acceleration
        if len(self.X_history) < 2:
            # First iteration: use simple fixed-point with damping
            x_next_flat = x_flat + self.beta * residual_flat
            return x_next_flat.reshape(self._original_shape)

        # Check restart condition
        if self.restart_threshold is not None and len(self.residual_norms) >= 2:
            if residual_norm > self.restart_threshold * self.residual_norms[-2]:
                # Restart: clear history except current
                self.X_history = [x_flat.copy()]
                self.F_history = [f_flat.copy()]
                x_next_flat = x_flat + self.beta * residual_flat
                return x_next_flat.reshape(self._original_shape)

        # Anderson acceleration with least-squares
        m = len(self.X_history) - 1  # Number of previous iterates to use

        # Build difference matrices (all arrays are already flattened)
        # ΔF_k = [f_1 - f_0, f_2 - f_1, ..., f_k - f_{k-1}]
        # ΔX_k = [x_1 - x_0, x_2 - x_1, ..., x_k - x_{k-1}]
        # Note: column_stack now works because all arrays are 1D
        delta_F = np.column_stack([self.F_history[i] - self.F_history[i - 1] for i in range(1, len(self.F_history))])

        if method == "type1":
            # Type I: Minimize ||ΔF_k α + f_k||
            # Solve (ΔF_k^T ΔF_k + λI) α = -ΔF_k^T f_k

            A = delta_F.T @ delta_F
            # Add regularization for stability
            A += self.regularization * np.eye(m)
            b = -delta_F.T @ f_flat

            try:
                alpha = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                alpha = np.linalg.lstsq(delta_F, -f_flat, rcond=None)[0]

            # Compute accelerated iterate
            # x_next = (1 - Σα_i) x_k + Σ α_i x_i + β((1 - Σα_i) f_k + Σ α_i f_i)
            x_mix_flat = (1 - np.sum(alpha)) * x_flat
            f_mix_flat = (1 - np.sum(alpha)) * f_flat

            for i, a in enumerate(alpha):
                x_mix_flat += a * self.X_history[i]
                f_mix_flat += a * self.F_history[i]

            x_next_flat = x_mix_flat + self.beta * (f_mix_flat - x_mix_flat)

        else:  # type2
            # Type II: Minimize ||ΔF_k α + (f_k - x_k)||
            # This is the classical Anderson mixing formulation

            delta_R = delta_F - np.column_stack(
                [self.X_history[i] - self.X_history[i - 1] for i in range(1, len(self.X_history))]
            )

            A = delta_R.T @ delta_R
            A += self.regularization * np.eye(m)
            b = -delta_R.T @ residual_flat

            try:
                alpha = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(delta_R, -residual_flat, rcond=None)[0]

            # Compute accelerated iterate
            x_next_flat = (1 - np.sum(alpha)) * f_flat
            for i, a in enumerate(alpha):
                x_next_flat += a * self.F_history[i]

            # Apply damping
            x_next_flat = x_flat + self.beta * (x_next_flat - x_flat)

        # Reshape back to original shape before returning
        return x_next_flat.reshape(self._original_shape)

    def reset(self):
        """Reset accelerator state (clear history)."""
        self.X_history.clear()
        self.F_history.clear()
        self.residual_norms.clear()
        self.iteration_count = 0

    def get_convergence_info(self) -> dict:
        """Get convergence information."""
        return {
            "iteration_count": self.iteration_count,
            "residual_norms": self.residual_norms.copy(),
            "current_depth": len(self.X_history) - 1,
            "max_depth": self.depth,
        }


def create_anderson_accelerator(
    depth: int = 5,
    beta: float = 1.0,
    regularization: float = 1e-8,
    **kwargs,
) -> AndersonAccelerator:
    """
    Create Anderson accelerator with sensible defaults.

    Args:
        depth: Number of previous iterates (default: 5)
        beta: Damping parameter (default: 1.0 = no damping)
        regularization: Tikhonov regularization (default: 1e-8)
        **kwargs: Additional parameters for AndersonAccelerator

    Returns:
        Configured AndersonAccelerator instance
    """
    return AndersonAccelerator(
        depth=depth,
        beta=beta,
        regularization=regularization,
        **kwargs,
    )
