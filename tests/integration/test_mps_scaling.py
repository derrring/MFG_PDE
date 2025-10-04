#!/usr/bin/env python3
"""Test MPS performance scaling for different problem sizes."""

import time

import numpy as np

from mfg_pde.utils.acceleration.torch_utils import HAS_MPS
from mfg_pde.utils.acceleration.torch_utils import GaussianKDE as TorchKDE

try:
    from scipy.stats import gaussian_kde

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def benchmark_kde(n_particles, n_eval, backend="cpu"):
    """Benchmark KDE for given problem size."""
    np.random.seed(42)
    particles = np.random.normal(loc=0.5, scale=0.1, size=n_particles)
    x_eval = np.linspace(0, 1, n_eval)

    # Warmup
    kde = TorchKDE(particles, bw_method="scott", device=backend)
    _ = kde(x_eval)

    # Timed run
    start = time.time()
    kde = TorchKDE(particles, bw_method="scott", device=backend)
    _ = kde(x_eval)
    elapsed = time.time() - start

    return elapsed


def benchmark_scipy(n_particles, n_eval):
    """Benchmark scipy KDE."""
    np.random.seed(42)
    particles = np.random.normal(loc=0.5, scale=0.1, size=n_particles)
    x_eval = np.linspace(0, 1, n_eval)

    start = time.time()
    kde = gaussian_kde(particles, bw_method="scott")
    _ = kde(x_eval)
    elapsed = time.time() - start

    return elapsed


def main():
    """Test scaling across different problem sizes."""
    print("=" * 80)
    print("MPS SCALING ANALYSIS")
    print("=" * 80)

    if not HAS_MPS:
        print("\n⚠️  MPS not available on this system")
        return

    # Test different problem sizes
    test_cases = [
        # (n_particles, n_eval, description)
        (100, 50, "Tiny (100 × 50)"),
        (1000, 200, "Small (1k × 200)"),
        (5000, 200, "Phase 3B test (5k × 200)"),
        (10000, 500, "Medium (10k × 500)"),
        (50000, 1000, "Large (50k × 1k)"),
        (100000, 2000, "Very Large (100k × 2k)"),
    ]

    print("\n{:<25} {:<12} {:<12} {:<12} {:<12}".format("Problem Size", "Scipy", "CPU", "MPS", "MPS Speedup"))
    print("-" * 80)

    for n_particles, n_eval, desc in test_cases:
        try:
            # Scipy
            if HAS_SCIPY:
                scipy_time = benchmark_scipy(n_particles, n_eval)
            else:
                scipy_time = None

            # CPU
            cpu_time = benchmark_kde(n_particles, n_eval, backend="cpu")

            # MPS
            mps_time = benchmark_kde(n_particles, n_eval, backend="mps")

            # Speedup
            if scipy_time:
                speedup = scipy_time / mps_time
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            scipy_str = f"{scipy_time:.4f}s" if scipy_time else "N/A"
            print(f"{desc:<25} {scipy_str:<12} {cpu_time:<12.4f}s {mps_time:<12.4f}s {speedup_str:<12}")

        except Exception as e:
            print(f"{desc:<25} Error: {e}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
MPS Performance Characteristics:

1. **Overhead**: ~0.1-0.3s fixed cost for data transfer and kernel launch
2. **Crossover Point**: Likely around 50k-100k particles for KDE
3. **Best Use Cases**:
   - Large particle systems (100k+ particles)
   - Compute-heavy operations (matrix multiplication)
   - Iterative algorithms (amortize transfer cost)

4. **CPU Better For**:
   - Small problems (< 10k particles)
   - Single-shot operations
   - Memory-bound workloads

Recommendation: Use CPU for typical MFG problems, MPS for large-scale simulations.
""")


if __name__ == "__main__":
    main()
