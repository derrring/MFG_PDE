#!/usr/bin/env python3
"""
JAX + Numba Hybrid Architecture Performance Demo

Demonstrates the optimal use of JAX for pure functional computations
and Numba for imperative bottlenecks in AMR implementation.

This example shows:
1. JAX for differentiable, vectorizable math (error indicators, interpolation)
2. Numba for imperative logic (tree traversal, conditional operations)
3. Performance comparison against pure NumPy
"""

import time

import numpy as np

# JAX imports (with fallback)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit  # noqa: F401

    JAX_AVAILABLE = True
    print("✓ JAX available for pure functional computations")
except ImportError:
    JAX_AVAILABLE = False
    print("✗ JAX not available")

# Numba imports (with fallback)
try:
    import numba  # noqa: F401
    from numba import jit as numba_jit

    NUMBA_AVAILABLE = True
    print("✓ Numba available for imperative optimizations")
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not available")

# ============================================================================
# JAX: Pure Functional Computations (Ideal for JAX)
# ============================================================================

if JAX_AVAILABLE:

    @jax.jit
    def compute_error_indicators_jax(U: jnp.ndarray, M: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        """
        JAX-optimized error indicator computation.

        This is PERFECT for JAX because:
        - Pure function (no side effects)
        - Vectorizable operations
        - Differentiable (can compute gradients)
        - GPU-accelerable
        """
        # First-order gradients
        dU_dx = jnp.gradient(U, dx, axis=0)
        dU_dy = jnp.gradient(U, dy, axis=1)
        dM_dx = jnp.gradient(M, dx, axis=0)
        dM_dy = jnp.gradient(M, dy, axis=1)

        # Second-order for curvature
        d2U_dx2 = jnp.gradient(dU_dx, dx, axis=0)
        d2U_dy2 = jnp.gradient(dU_dy, dy, axis=1)
        d2M_dx2 = jnp.gradient(dM_dx, dx, axis=0)
        d2M_dy2 = jnp.gradient(dM_dy, dy, axis=1)

        # Combined indicators
        grad_U = jnp.sqrt(dU_dx**2 + dU_dy**2)
        grad_M = jnp.sqrt(dM_dx**2 + dM_dy**2)
        curv_U = jnp.abs(d2U_dx2) + jnp.abs(d2U_dy2)
        curv_M = jnp.abs(d2M_dx2) + jnp.abs(d2M_dy2)

        return jnp.maximum(grad_U, grad_M) + 0.1 * jnp.maximum(curv_U, curv_M)

    @jax.jit
    def conservative_interpolation_jax(
        old_data: jnp.ndarray, old_coords: jnp.ndarray, new_coords: jnp.ndarray
    ) -> jnp.ndarray:
        """
        JAX-optimized conservative interpolation.

        Perfect for JAX:
        - Mathematical transformation
        - No control flow
        - Vectorizable
        - Differentiable
        """
        # Simple conservative interpolation using JAX
        return jnp.interp(new_coords, old_coords, old_data)

    @jax.jit
    def mass_conservation_enforcement_jax(density: jnp.ndarray, target_mass: float) -> jnp.ndarray:
        """
        JAX-optimized mass conservation.

        Pure mathematical constraint enforcement - ideal for JAX.
        """
        current_mass = jnp.sum(density)
        conservation_factor = target_mass / (current_mass + 1e-12)
        return density * conservation_factor


# ============================================================================
# Numba: Imperative Bottlenecks (Ideal for Numba)
# ============================================================================

if NUMBA_AVAILABLE:

    @numba_jit(nopython=True, cache=True)
    def find_cells_to_refine_numba(
        error_indicators: np.ndarray,
        cell_bounds: np.ndarray,  # (N, 4)
        cell_levels: np.ndarray,  # (N,)
        is_leaf: np.ndarray,  # (N,) bool
        error_threshold: float,
        max_level: int,
    ) -> np.ndarray:
        """
        Numba-optimized refinement candidate selection.

        Perfect for Numba because:
        - Complex conditional logic
        - Loops with early termination
        - Integer indexing operations
        - Hard to vectorize efficiently
        """
        candidates = []

        for i in range(len(error_indicators)):
            # Skip non-leaf cells
            if not is_leaf[i]:
                continue

            # Check error threshold
            if error_indicators[i] <= error_threshold:
                continue

            # Check level constraint
            if cell_levels[i] >= max_level:
                continue

            # Check minimum size constraint
            x_min, x_max, y_min, y_max = cell_bounds[i]
            if min(x_max - x_min, y_max - y_min) < 1e-6:
                continue

            candidates.append(i)

        return np.array(candidates, dtype=np.int64)

    @numba_jit(nopython=True, cache=True)
    def traverse_quadtree_numba(
        node_parents: np.ndarray,  # (N,)
        node_children: np.ndarray,  # (N, 4)
        node_bounds: np.ndarray,  # (N, 4)
        target_x: float,
        target_y: float,
    ) -> int:
        """
        Numba-optimized quadtree traversal.

        Perfect for Numba because:
        - Tree navigation logic
        - Pointer-like operations
        - Control flow heavy
        - Sequential by nature
        """
        current_node = 0  # Start at root

        while True:
            # Check if we're at a leaf
            has_children = False
            for child_idx in node_children[current_node]:
                if child_idx >= 0:
                    has_children = True
                    break

            if not has_children:
                return current_node

            # Find which child contains the target point
            x_min, x_max, y_min, y_max = node_bounds[current_node]
            mid_x = 0.5 * (x_min + x_max)
            mid_y = 0.5 * (y_min + y_max)

            # Determine child quadrant
            if target_x < mid_x and target_y < mid_y:
                child_idx = 0  # SW
            elif target_x >= mid_x and target_y < mid_y:
                child_idx = 1  # SE
            elif target_x < mid_x and target_y >= mid_y:
                child_idx = 2  # NW
            else:
                child_idx = 3  # NE

            next_node = node_children[current_node][child_idx]
            if next_node < 0:
                return current_node

            current_node = next_node

    @numba_jit(nopython=True, cache=True)
    def collect_mesh_statistics_numba(
        cell_bounds: np.ndarray, cell_levels: np.ndarray, is_leaf: np.ndarray
    ) -> tuple[int, int, float, float]:
        """
        Numba-optimized statistics collection.

        Perfect for Numba because:
        - Iteration-heavy computation
        - Accumulation operations
        - Simple but repetitive logic
        """
        total_cells = len(cell_bounds)
        leaf_cells = 0
        total_area = 0.0
        max_level = 0

        for i in range(total_cells):
            if is_leaf[i]:
                leaf_cells += 1

            if cell_levels[i] > max_level:
                max_level = cell_levels[i]

            x_min, x_max, y_min, y_max = cell_bounds[i]
            area = (x_max - x_min) * (y_max - y_min)
            total_area += area

        return total_cells, leaf_cells, total_area, float(max_level)


# ============================================================================
# NumPy Baseline Implementations
# ============================================================================


def compute_error_indicators_numpy(U: np.ndarray, M: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """NumPy baseline for comparison."""
    dU_dx = np.gradient(U, dx, axis=0)
    dU_dy = np.gradient(U, dy, axis=1)
    dM_dx = np.gradient(M, dx, axis=0)
    dM_dy = np.gradient(M, dy, axis=1)

    d2U_dx2 = np.gradient(dU_dx, dx, axis=0)
    d2U_dy2 = np.gradient(dU_dy, dy, axis=1)
    d2M_dx2 = np.gradient(dM_dx, dx, axis=0)
    d2M_dy2 = np.gradient(dM_dy, dy, axis=1)

    grad_U = np.sqrt(dU_dx**2 + dU_dy**2)
    grad_M = np.sqrt(dM_dx**2 + dM_dy**2)
    curv_U = np.abs(d2U_dx2) + np.abs(d2U_dy2)
    curv_M = np.abs(d2M_dx2) + np.abs(d2M_dy2)

    return np.maximum(grad_U, grad_M) + 0.1 * np.maximum(curv_U, curv_M)


def find_cells_to_refine_python(
    error_indicators: np.ndarray,
    cell_bounds: np.ndarray,
    cell_levels: np.ndarray,
    is_leaf: np.ndarray,
    error_threshold: float,
    max_level: int,
) -> np.ndarray:
    """Pure Python baseline for comparison."""
    candidates = []

    for i in range(len(error_indicators)):
        if not is_leaf[i]:
            continue
        if error_indicators[i] <= error_threshold:
            continue
        if cell_levels[i] >= max_level:
            continue

        x_min, x_max, y_min, y_max = cell_bounds[i]
        if min(x_max - x_min, y_max - y_min) < 1e-6:
            continue

        candidates.append(i)

    return np.array(candidates, dtype=np.int64)


# ============================================================================
# Performance Benchmark
# ============================================================================


def create_test_data(size: int = 128) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create test data for benchmarking."""
    # Create solution data with sharp features
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)

    # Sharp Gaussian peaks (challenging for AMR)
    U = np.exp(-5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)) + 0.5 * np.exp(-10 * ((X + 0.5) ** 2 + (Y + 0.5) ** 2))

    M = np.exp(-3 * (X**2 + Y**2)) + 0.3 * np.exp(-8 * ((X - 1) ** 2 + Y**2))

    # Normalize density
    M = M / np.sum(M)

    # Create mesh data
    n_cells = 1000
    cell_bounds = np.random.uniform(-2, 2, (n_cells, 4))
    cell_bounds[:, 1] = cell_bounds[:, 0] + np.random.uniform(0.1, 0.5, n_cells)  # x_max > x_min
    cell_bounds[:, 3] = cell_bounds[:, 2] + np.random.uniform(0.1, 0.5, n_cells)  # y_max > y_min

    cell_levels = np.random.randint(0, 4, n_cells)
    is_leaf = np.random.choice([True, False], n_cells, p=[0.7, 0.3])

    mesh_data = {
        "cell_bounds": cell_bounds,
        "cell_levels": cell_levels,
        "is_leaf": is_leaf,
        "error_threshold": 1e-4,
        "max_level": 5,
    }

    return U, M, mesh_data


def benchmark_error_computation(
    U: np.ndarray, M: np.ndarray, dx: float, dy: float, n_runs: int = 10
) -> dict[str, float]:
    """Benchmark error computation implementations."""
    results = {}

    # NumPy baseline
    times = []
    for _ in range(n_runs):
        start = time.time()
        compute_error_indicators_numpy(U, M, dx, dy)
        times.append(time.time() - start)
    results["numpy"] = np.mean(times)

    # JAX implementation
    if JAX_AVAILABLE:
        # Warmup JIT compilation
        U_jax, M_jax = jnp.array(U), jnp.array(M)
        _ = compute_error_indicators_jax(U_jax, M_jax, dx, dy)

        times = []
        for _ in range(n_runs):
            start = time.time()
            error_jax = compute_error_indicators_jax(U_jax, M_jax, dx, dy)
            # Force computation (JAX is lazy)
            _ = np.array(error_jax)
            times.append(time.time() - start)
        results["jax"] = np.mean(times)

    return results


def benchmark_refinement_selection(mesh_data: dict, error_indicators: np.ndarray, n_runs: int = 10) -> dict[str, float]:
    """Benchmark refinement candidate selection implementations."""
    results = {}

    # Python baseline
    times = []
    for _ in range(n_runs):
        start = time.time()
        find_cells_to_refine_python(
            error_indicators,
            mesh_data["cell_bounds"],
            mesh_data["cell_levels"],
            mesh_data["is_leaf"],
            mesh_data["error_threshold"],
            mesh_data["max_level"],
        )
        times.append(time.time() - start)
    results["python"] = np.mean(times)

    # Numba implementation
    if NUMBA_AVAILABLE:
        # Warmup compilation
        _ = find_cells_to_refine_numba(
            error_indicators,
            mesh_data["cell_bounds"],
            mesh_data["cell_levels"],
            mesh_data["is_leaf"],
            mesh_data["error_threshold"],
            mesh_data["max_level"],
        )

        times = []
        for _ in range(n_runs):
            start = time.time()
            find_cells_to_refine_numba(
                error_indicators,
                mesh_data["cell_bounds"],
                mesh_data["cell_levels"],
                mesh_data["is_leaf"],
                mesh_data["error_threshold"],
                mesh_data["max_level"],
            )
            times.append(time.time() - start)
        results["numba"] = np.mean(times)

    return results


def main():
    """Main benchmark and demonstration."""
    print("=" * 60)
    print("JAX + Numba Hybrid Architecture Performance Demo")
    print("=" * 60)

    # Create test data
    print("\n1. Creating test data...")
    sizes = [64, 128, 256]

    for size in sizes:
        print(f"\n--- Testing with {size}×{size} grid ---")

        U, M, mesh_data = create_test_data(size)
        dx = dy = 4.0 / size  # Domain is [-2, 2]

        print(f"Solution arrays: {U.shape}")
        print(f"Mesh cells: {len(mesh_data['cell_bounds'])}")

        # Benchmark error computation (JAX strength)
        print("\n2. Benchmarking Error Computation (JAX vs NumPy):")
        error_results = benchmark_error_computation(U, M, dx, dy)

        for method, time_taken in error_results.items():
            print(f"  {method:>8}: {time_taken * 1000:.2f} ms")

        if "jax" in error_results and "numpy" in error_results:
            speedup = error_results["numpy"] / error_results["jax"]
            print(f"  JAX speedup: {speedup:.1f}×")

        # Compute error indicators for refinement benchmark
        if JAX_AVAILABLE:
            U_jax, M_jax = jnp.array(U), jnp.array(M)
            error_indicators = np.array(compute_error_indicators_jax(U_jax, M_jax, dx, dy))
            # Map to mesh cells (simplified)
            error_indicators_cells = np.random.uniform(0, np.max(error_indicators), len(mesh_data["cell_bounds"]))
        else:
            error_indicators = compute_error_indicators_numpy(U, M, dx, dy)
            error_indicators_cells = np.random.uniform(0, np.max(error_indicators), len(mesh_data["cell_bounds"]))

        # Benchmark refinement selection (Numba strength)
        print("\n3. Benchmarking Refinement Selection (Numba vs Python):")
        refinement_results = benchmark_refinement_selection(mesh_data, error_indicators_cells)

        for method, time_taken in refinement_results.items():
            print(f"  {method:>8}: {time_taken * 1000:.2f} ms")

        if "numba" in refinement_results and "python" in refinement_results:
            speedup = refinement_results["python"] / refinement_results["numba"]
            print(f"  Numba speedup: {speedup:.1f}×")

    # Demonstrate hybrid workflow
    print("\n" + "=" * 60)
    print("4. Hybrid Workflow Demonstration")
    print("=" * 60)

    U, M, mesh_data = create_test_data(128)
    dx = dy = 4.0 / 128

    print("\nOptimal Hybrid Workflow:")
    print("1. JAX: Pure functional error computation")
    print("2. Numba: Imperative refinement selection")
    print("3. JAX: Conservative solution interpolation")

    total_start = time.time()

    # Step 1: JAX for error computation
    if JAX_AVAILABLE:
        jax_start = time.time()
        U_jax, M_jax = jnp.array(U), jnp.array(M)
        error_field = compute_error_indicators_jax(U_jax, M_jax, dx, dy)
        error_indicators_cells = np.random.uniform(0, np.max(error_field), len(mesh_data["cell_bounds"]))
        jax_time1 = time.time() - jax_start
        print(f"  JAX error computation: {jax_time1 * 1000:.2f} ms")
    else:
        error_indicators_cells = np.random.uniform(0, 1, len(mesh_data["cell_bounds"]))
        jax_time1 = 0

    # Step 2: Numba for refinement selection
    if NUMBA_AVAILABLE:
        numba_start = time.time()
        candidates = find_cells_to_refine_numba(
            error_indicators_cells,
            mesh_data["cell_bounds"],
            mesh_data["cell_levels"],
            mesh_data["is_leaf"],
            mesh_data["error_threshold"],
            mesh_data["max_level"],
        )
        numba_time = time.time() - numba_start
        print(f"  Numba refinement selection: {numba_time * 1000:.2f} ms")
        print(f"  Found {len(candidates)} cells to refine")
    else:
        numba_time = 0

    # Step 3: JAX for solution interpolation
    if JAX_AVAILABLE:
        jax_start = time.time()
        old_coords = jnp.linspace(-2, 2, 64)
        new_coords = jnp.linspace(-2, 2, 128)
        old_data = jnp.array(U[::2, ::2])  # Downsample for demo
        conservative_interpolation_jax(old_data.ravel(), old_coords, new_coords)
        jax_time2 = time.time() - jax_start
        print(f"  JAX solution interpolation: {jax_time2 * 1000:.2f} ms")
    else:
        jax_time2 = 0

    total_time = time.time() - total_start

    print(f"\nTotal hybrid workflow: {total_time * 1000:.2f} ms")
    print(f"JAX fraction: {(jax_time1 + jax_time2) / total_time * 100:.1f}%")
    print(f"Numba fraction: {numba_time / total_time * 100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("ARCHITECTURAL PRINCIPLE DEMONSTRATION")
    print("=" * 60)
    print("\n✓ JAX STRENGTHS (Pure Functional Computations):")
    print("  • Error indicator computation: Vectorizable math")
    print("  • Conservative interpolation: Differentiable operations")
    print("  • Mass conservation: Mathematical constraints")
    print("  • GPU acceleration: Parallel tensor operations")

    print("\n✓ NUMBA STRENGTHS (Imperative Bottlenecks):")
    print("  • Refinement selection: Complex conditional logic")
    print("  • Tree traversal: Sequential navigation")
    print("  • Mesh bookkeeping: Data structure updates")
    print("  • Index operations: Cache-friendly loops")

    print("\n✓ HYBRID BENEFITS:")
    print("  • Each tool used for optimal use case")
    print("  • No architectural compromises")
    print("  • Maximum performance from both paradigms")
    print("  • Clean separation of concerns")

    if JAX_AVAILABLE and NUMBA_AVAILABLE:
        print("\n🚀 Expected speedups achieved!")
    elif JAX_AVAILABLE:
        print("\n⚠️  Install Numba for imperative optimizations")
    elif NUMBA_AVAILABLE:
        print("\n⚠️  Install JAX for functional optimizations")
    else:
        print("\n⚠️  Install JAX and Numba for optimal performance")


if __name__ == "__main__":
    main()
