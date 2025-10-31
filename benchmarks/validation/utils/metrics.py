"""
Validation metrics for MFG solver comparison.

Provides functions to compute:
- L² error between solutions
- Mass conservation error
- Convergence rate estimation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_l2_error(
    solution1: NDArray[np.floating],
    solution2: NDArray[np.floating],
    dx: float | tuple[float, ...] | None = None,
) -> float:
    """
    Compute L² error between two solutions.

    L² error = sqrt(∫ |u1 - u2|² dx) ≈ sqrt(∑ |u1 - u2|² * ΔV)

    Parameters
    ----------
    solution1 : NDArray
        First solution (reference)
    solution2 : NDArray
        Second solution (to compare)
    dx : float or tuple of float, optional
        Grid spacing. If None, uses normalized error (no spatial weighting).
        For nD: tuple of (dx, dy, dz, ...)

    Returns
    -------
    float
        L² error

    Examples
    --------
    >>> u_ref = np.array([1.0, 2.0, 3.0])
    >>> u_test = np.array([1.1, 2.1, 2.9])
    >>> compute_l2_error(u_ref, u_test, dx=0.1)
    0.1732...

    Notes
    -----
    For normalized comparison (no dx), returns sqrt(mean((u1-u2)²)).
    """
    if solution1.shape != solution2.shape:
        raise ValueError(f"Solution shapes must match: {solution1.shape} vs {solution2.shape}")

    diff = solution1 - solution2
    squared_diff = diff**2

    if dx is None:
        # Normalized L² error (no spatial weighting)
        return float(np.sqrt(np.mean(squared_diff)))

    # Compute volume element
    if isinstance(dx, (int, float)):
        # 1D or uniform grid
        dV = float(dx) ** solution1.ndim
    else:
        # nD with different spacings
        dV = float(np.prod(dx))

    # L² error with spatial integration
    l2_error = np.sqrt(np.sum(squared_diff) * dV)
    return float(l2_error)


def compute_relative_l2_error(
    solution1: NDArray[np.floating],
    solution2: NDArray[np.floating],
    dx: float | tuple[float, ...] | None = None,
) -> float:
    """
    Compute relative L² error: ||u1 - u2||_L² / ||u1||_L²

    Parameters
    ----------
    solution1 : NDArray
        Reference solution
    solution2 : NDArray
        Test solution
    dx : float or tuple of float, optional
        Grid spacing

    Returns
    -------
    float
        Relative L² error

    Notes
    -----
    Useful when solutions have different magnitudes.
    """
    error = compute_l2_error(solution1, solution2, dx=dx)

    if dx is None:
        norm1 = float(np.sqrt(np.mean(solution1**2)))
    else:
        if isinstance(dx, (int, float)):
            dV = float(dx) ** solution1.ndim
        else:
            dV = float(np.prod(dx))
        norm1 = float(np.sqrt(np.sum(solution1**2) * dV))

    if norm1 < 1e-14:
        # Avoid division by zero
        return float(error)

    return float(error / norm1)


def compute_mass_conservation(
    density: NDArray[np.floating], dx: float | tuple[float, ...], target_mass: float = 1.0
) -> float:
    """
    Compute mass conservation error.

    Error = |∫ m(x) dx - target_mass| / target_mass

    Parameters
    ----------
    density : NDArray
        Density distribution m(x,t)
    dx : float or tuple of float
        Grid spacing
    target_mass : float, default=1.0
        Expected total mass (usually 1.0 for probability distribution)

    Returns
    -------
    float
        Relative mass conservation error (0.0 = perfect conservation)

    Examples
    --------
    >>> m = np.ones((10, 10)) * 0.01  # Uniform distribution
    >>> compute_mass_conservation(m, dx=0.1, target_mass=1.0)
    0.0  # Perfect mass conservation

    Notes
    -----
    For FDM with dimensional splitting, expect ~1-2% error per solve.
    For GFDM/particle methods, expect better conservation (<0.1%).
    """
    # Compute volume element
    if isinstance(dx, (int, float)):
        dV = float(dx) ** density.ndim
    else:
        dV = float(np.prod(dx))

    # Integrate density
    computed_mass = float(np.sum(density) * dV)

    # Relative error
    error = abs(computed_mass - target_mass) / target_mass
    return float(error)


def compute_convergence_rate(errors: NDArray[np.floating], grid_sizes: NDArray[np.floating]) -> float:
    """
    Estimate convergence rate from error vs grid size.

    Assumes power law: error ≈ C * h^p
    Fits log(error) vs log(h) to estimate p.

    Parameters
    ----------
    errors : NDArray
        Array of errors for different grid sizes
    grid_sizes : NDArray
        Array of grid spacings (h values)

    Returns
    -------
    float
        Estimated convergence rate p

    Examples
    --------
    >>> h = np.array([0.1, 0.05, 0.025])
    >>> errors = np.array([0.01, 0.0025, 0.000625])  # O(h²) convergence
    >>> compute_convergence_rate(errors, h)
    2.0

    Notes
    -----
    - FDM (centered differences): p ≈ 2
    - FDM (Strang splitting): p ≈ 2 (temporal) + 2 (spatial)
    - GFDM (2nd order Taylor): p ≈ 2-3
    """
    if len(errors) != len(grid_sizes):
        raise ValueError(f"Arrays must have same length: {len(errors)} vs {len(grid_sizes)}")

    if len(errors) < 2:
        raise ValueError("Need at least 2 data points to estimate convergence rate")

    # Filter out zero/negative errors (can't take log)
    mask = (errors > 0) & (grid_sizes > 0)
    if not np.any(mask):
        raise ValueError("All errors or grid sizes are non-positive")

    log_errors = np.log(errors[mask])
    log_h = np.log(grid_sizes[mask])

    # Linear fit: log(error) = log(C) + p * log(h)
    # Using numpy's polyfit for robustness
    coeffs = np.polyfit(log_h, log_errors, deg=1)
    p = float(coeffs[0])

    return p


def compute_iteration_convergence_rate(residuals: NDArray[np.floating]) -> float:
    """
    Estimate Picard iteration convergence rate.

    Assumes exponential decay: r_k ≈ r_0 * ρ^k
    Fits log(r_k) vs k to estimate ρ.

    Parameters
    ----------
    residuals : NDArray
        Array of residuals at each iteration

    Returns
    -------
    float
        Estimated convergence factor ρ (< 1 for convergence)

    Notes
    -----
    - ρ < 0.5: Fast convergence
    - 0.5 < ρ < 0.9: Moderate convergence
    - ρ > 0.9: Slow convergence
    - ρ ≥ 1: Non-convergent
    """
    if len(residuals) < 2:
        raise ValueError("Need at least 2 iterations to estimate convergence rate")

    # Filter out zero/negative residuals
    mask = residuals > 0
    if not np.any(mask):
        raise ValueError("All residuals are non-positive")

    # Use last N iterations for more stable estimate
    N = min(10, len(residuals))
    residuals_recent = residuals[-N:][mask[-N:]]

    if len(residuals_recent) < 2:
        # Not enough data
        return 1.0

    log_residuals = np.log(residuals_recent)
    iterations = np.arange(len(log_residuals))

    # Linear fit: log(r_k) = log(r_0) + k * log(ρ)
    coeffs = np.polyfit(iterations, log_residuals, deg=1)
    log_rho = coeffs[0]
    rho = float(np.exp(log_rho))

    return rho


def compute_solution_statistics(solution: NDArray[np.floating]) -> dict[str, float]:
    """
    Compute statistical properties of solution.

    Parameters
    ----------
    solution : NDArray
        Solution array (value function or density)

    Returns
    -------
    dict
        Dictionary with keys: min, max, mean, std, l2_norm

    Examples
    --------
    >>> u = np.random.randn(100)
    >>> stats = compute_solution_statistics(u)
    >>> 'min' in stats and 'max' in stats
    True
    """
    return {
        "min": float(np.min(solution)),
        "max": float(np.max(solution)),
        "mean": float(np.mean(solution)),
        "std": float(np.std(solution)),
        "l2_norm": float(np.linalg.norm(solution.ravel())),
    }
