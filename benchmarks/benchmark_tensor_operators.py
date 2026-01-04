#!/usr/bin/env python3
"""
Benchmark tensor diffusion operators with and without Numba JIT.

Measures performance of tensor diffusion via tensor_calculus.diffusion():
- Full tensor diffusion
- Diagonal diffusion (optimized path)

Compares:
- Pure NumPy baseline
- Numba JIT compiled versions
- Different grid sizes (50x50, 100x100, 200x200)
- Constant vs spatially-varying tensors

Note: Migrated from tensor_operators.py to tensor_calculus as of v0.17.0.
"""

import time
from collections.abc import Callable

import numpy as np

from mfg_pde.geometry.boundary import no_flux_bc

# Import unified diffusion API and Numba flags
from mfg_pde.utils.numerical.tensor_calculus import (
    NUMBA_AVAILABLE,
    USE_NUMBA,
    diffusion,
)

# ============================================================================
# Benchmark Configuration
# ============================================================================

GRID_SIZES = [
    (50, 50),
    (100, 100),
    (200, 200),
]

N_WARMUP = 3  # Warmup iterations (for JIT compilation)
N_ITERATIONS = 20  # Measurement iterations


# ============================================================================
# Benchmark Utilities
# ============================================================================


def create_test_density(Ny: int, Nx: int) -> np.ndarray:
    """Create test density field with Gaussian."""
    y = np.linspace(0, 1, Ny)
    x = np.linspace(0, 1, Nx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    m = np.exp(-10 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    m /= np.sum(m)
    return m.T  # Shape: (Ny, Nx)


def create_constant_tensor_2d() -> np.ndarray:
    """Create constant anisotropic tensor."""
    # Fast horizontal, slow vertical
    return np.array([[0.15**2, 0.02], [0.02, 0.05**2]])


def create_spatially_varying_tensor_2d(Ny: int, Nx: int) -> np.ndarray:
    """Create spatially-varying diagonal tensor."""
    y = np.linspace(0, 1, Ny)
    x = np.linspace(0, 1, Nx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Varying diffusion: higher in center
    sigma_x = 0.1 + 0.1 * np.exp(-5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    sigma_y = 0.05 + 0.05 * np.exp(-5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    Sigma = np.zeros((Ny, Nx, 2, 2))
    Sigma[:, :, 0, 0] = (sigma_x**2).T
    Sigma[:, :, 1, 1] = (sigma_y**2).T

    return Sigma


def create_diagonal_tensor_constant() -> np.ndarray:
    """Create constant diagonal tensor."""
    return np.array([0.15**2, 0.05**2])


def create_diagonal_tensor_varying(Ny: int, Nx: int) -> np.ndarray:
    """Create spatially-varying diagonal tensor."""
    y = np.linspace(0, 1, Ny)
    x = np.linspace(0, 1, Nx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    sigma_x = 0.1 + 0.1 * np.exp(-5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    sigma_y = 0.05 + 0.05 * np.exp(-5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    sigma_diag = np.zeros((Ny, Nx, 2))
    sigma_diag[:, :, 0] = (sigma_x**2).T
    sigma_diag[:, :, 1] = (sigma_y**2).T

    return sigma_diag


def benchmark_function(
    func: Callable,
    *args,
    n_warmup: int = N_WARMUP,
    n_iterations: int = N_ITERATIONS,
    **kwargs,
) -> dict[str, float]:
    """
    Benchmark a function with warmup and multiple iterations.

    Returns:
        Dictionary with timing statistics (mean, std, min, max)
    """
    # Warmup (important for JIT compilation)
    for _ in range(n_warmup):
        func(*args, **kwargs)

    # Measure
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


# ============================================================================
# Benchmark Scenarios
# ============================================================================


def benchmark_full_tensor_constant(Ny: int, Nx: int) -> dict:
    """Benchmark full tensor diffusion with constant tensor."""
    m = create_test_density(Ny, Nx)
    Sigma = create_constant_tensor_2d()
    dx = dy = 1.0 / Nx
    bc = no_flux_bc(dimension=2)

    stats = benchmark_function(diffusion, m, Sigma, [dx, dy], bc=bc)

    return {
        "scenario": "Full Tensor (Constant)",
        "grid_size": (Ny, Nx),
        **stats,
    }


def benchmark_full_tensor_varying(Ny: int, Nx: int) -> dict:
    """Benchmark full tensor diffusion with spatially-varying tensor."""
    m = create_test_density(Ny, Nx)
    Sigma = create_spatially_varying_tensor_2d(Ny, Nx)
    dx = dy = 1.0 / Nx
    bc = no_flux_bc(dimension=2)

    stats = benchmark_function(diffusion, m, Sigma, [dx, dy], bc=bc)

    return {
        "scenario": "Full Tensor (Spatially Varying)",
        "grid_size": (Ny, Nx),
        **stats,
    }


def benchmark_diagonal_constant(Ny: int, Nx: int) -> dict:
    """Benchmark diagonal diffusion with constant tensor."""
    m = create_test_density(Ny, Nx)
    sigma_diag = create_diagonal_tensor_constant()
    dx = dy = 1.0 / Nx
    bc = no_flux_bc(dimension=2)

    # Convert to full tensor for unified API
    sigma_tensor = np.diag(sigma_diag)
    stats = benchmark_function(diffusion, m, sigma_tensor, [dx, dy], bc=bc)

    return {
        "scenario": "Diagonal (Constant)",
        "grid_size": (Ny, Nx),
        **stats,
    }


def benchmark_diagonal_varying(Ny: int, Nx: int) -> dict:
    """Benchmark diagonal diffusion with spatially-varying tensor."""
    m = create_test_density(Ny, Nx)
    sigma_diag = create_diagonal_tensor_varying(Ny, Nx)
    dx = dy = 1.0 / Nx
    bc = no_flux_bc(dimension=2)

    # Convert diagonal arrays to full tensor format (Ny, Nx, 2, 2)
    Sigma = np.zeros((Ny, Nx, 2, 2))
    Sigma[:, :, 0, 0] = sigma_diag[:, :, 0]
    Sigma[:, :, 1, 1] = sigma_diag[:, :, 1]
    stats = benchmark_function(diffusion, m, Sigma, [dx, dy], bc=bc)

    return {
        "scenario": "Diagonal (Spatially Varying)",
        "grid_size": (Ny, Nx),
        **stats,
    }


# ============================================================================
# Main Benchmark Suite
# ============================================================================


def run_all_benchmarks():
    """Run all tensor operator benchmarks."""
    print("=" * 80)
    print("Tensor Diffusion Operator Benchmarks")
    print("=" * 80)
    print()
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print(f"Numba Enabled:   {USE_NUMBA}")
    print()

    if USE_NUMBA and NUMBA_AVAILABLE:
        mode_str = "Numba JIT"
    else:
        mode_str = "NumPy Baseline"

    print(f"Running in: {mode_str} mode")
    print("=" * 80)
    print()

    all_results = []

    for Ny, Nx in GRID_SIZES:
        print(f"Grid Size: {Ny} × {Nx}")
        print("-" * 80)

        # Full tensor benchmarks
        result = benchmark_full_tensor_constant(Ny, Nx)
        all_results.append(result)
        print(f"  {result['scenario']:35s}: {result['mean'] * 1000:8.3f} ± {result['std'] * 1000:6.3f} ms")

        result = benchmark_full_tensor_varying(Ny, Nx)
        all_results.append(result)
        print(f"  {result['scenario']:35s}: {result['mean'] * 1000:8.3f} ± {result['std'] * 1000:6.3f} ms")

        # Diagonal benchmarks
        result = benchmark_diagonal_constant(Ny, Nx)
        all_results.append(result)
        print(f"  {result['scenario']:35s}: {result['mean'] * 1000:8.3f} ± {result['std'] * 1000:6.3f} ms")

        result = benchmark_diagonal_varying(Ny, Nx)
        all_results.append(result)
        print(f"  {result['scenario']:35s}: {result['mean'] * 1000:8.3f} ± {result['std'] * 1000:6.3f} ms")

        print()

    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Mode: {mode_str}")
    print(f"  - Warmup iterations: {N_WARMUP}")
    print(f"  - Measurement iterations: {N_ITERATIONS}")
    print(f"  - Grid sizes tested: {len(GRID_SIZES)}")
    print(f"  - Total benchmarks: {len(all_results)}")
    print()

    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks()
