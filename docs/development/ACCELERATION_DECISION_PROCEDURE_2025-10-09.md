# Acceleration Decision Procedure: PyO3 or Alternatives?

**Date**: 2025-10-09
**Purpose**: Systematic procedure to determine optimal acceleration strategy
**Question**: Do we need PyO3 (Rust) or will simpler alternatives suffice?

## Decision Framework

```
START
  ↓
[1. Profile & Measure Baseline] ← YOU ARE HERE
  ↓
[2. Try Numba JIT (2 hours)]
  ↓
  Speedup ≥ 5×? → YES → DONE (Use Numba)
  ↓ NO
[3. Try NumPy Optimization (1 hour)]
  ↓
  Speedup ≥ 5×? → YES → DONE (Use optimized NumPy)
  ↓ NO
[4. Implement PyO3 Rust (1-2 weeks)]
  ↓
DONE
```

## Step 1: Profile & Establish Baseline ⏱️ **1 hour**

### Task 1.1: Identify Actual Bottleneck
```bash
cd /Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE

# Profile WENO5 solver
python -m cProfile -o weno_profile.prof -s cumtime << 'EOF'
from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBWenoSolver
from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator

problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)
hjb_solver = HJBWenoSolver(problem, weno_variant="weno5")
fp_solver = FPFDMSolver(problem)
mfg_solver = FixedPointIterator(problem, hjb_solver, fp_solver)

result = mfg_solver.solve(max_iterations=10)
EOF

# Analyze profile
python -c "
import pstats
p = pstats.Stats('weno_profile.prof')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

**Expected Output**:
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001   45.230   45.230 fixed_point_iterator.py:150(solve)
      500    2.340    0.005   35.120    0.070 hjb_weno.py:320(solve_hjb_system)
    50000   15.230    0.000   25.450    0.001 hjb_weno.py:286(_compute_smoothness_indicators)  ← BOTTLENECK!
    50000    8.120    0.000    8.120    0.000 hjb_weno.py:340(_compute_weno_weights)
```

### Task 1.2: Microbenchmark Hot Function
```python
# benchmarks/baseline/weno5_smoothness_baseline.py
import numpy as np
import timeit

from mfg_pde.alg.numerical.hjb_solvers.hjb_weno import HJBWenoSolver

def benchmark_smoothness_indicators():
    # Create dummy solver to access method
    from mfg_pde import ExampleMFGProblem
    problem = ExampleMFGProblem(Nx=100, Nt=50)
    solver = HJBWenoSolver(problem)

    # Test data
    u = np.random.randn(5)

    # Warmup
    for _ in range(100):
        solver._compute_smoothness_indicators(u)

    # Benchmark
    times = timeit.repeat(
        lambda: solver._compute_smoothness_indicators(u),
        number=10000,
        repeat=10
    )

    median_time_us = np.median(times) / 10000 * 1e6

    print(f"BASELINE: WENO5 smoothness indicators")
    print(f"  Median time: {median_time_us:.2f} μs per call")
    print(f"  Min time:    {min(times)/10000*1e6:.2f} μs")
    print(f"  Max time:    {max(times)/10000*1e6:.2f} μs")

    return median_time_us

if __name__ == "__main__":
    baseline = benchmark_smoothness_indicators()
```

**Run it**:
```bash
python benchmarks/baseline/weno5_smoothness_baseline.py
```

**Decision Criteria**:
- If baseline > 100 μs → Optimization worth it
- If baseline < 10 μs → Not worth optimizing
- If baseline 10-100 μs → Borderline, consider

---

## Step 2: Try Numba JIT ⚡ **2 hours**

### Why Numba First?
- ✅ Pure Python (no Rust toolchain)
- ✅ 2 hours to implement vs 1-2 weeks for Rust
- ✅ Often achieves 70-80% of Rust performance
- ✅ No distribution complexity
- ✅ Falls back to Python if unavailable

### Task 2.1: Implement Numba Version
```python
# mfg_pde/alg/numerical/hjb_solvers/hjb_weno_numba.py
"""Numba-accelerated WENO5 kernels."""

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(cache=True, fastmath=True)
def compute_smoothness_indicators_numba(u):
    """
    Numba-accelerated WENO5 smoothness indicators.

    Args:
        u: 5-point stencil [u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}]

    Returns:
        beta: [β₀, β₁, β₂] smoothness indicators
    """
    # β₀: sub-stencil S₀ (u_{i-2}, u_{i-1}, u_i)
    beta_0 = (13.0/12.0) * (u[0] - 2.0*u[1] + u[2])**2 + \
             (1.0/4.0) * (u[0] - 4.0*u[1] + 3.0*u[2])**2

    # β₁: sub-stencil S₁ (u_{i-1}, u_i, u_{i+1})
    beta_1 = (13.0/12.0) * (u[1] - 2.0*u[2] + u[3])**2 + \
             (1.0/4.0) * (u[1] - u[3])**2

    # β₂: sub-stencil S₂ (u_i, u_{i+1}, u_{i+2})
    beta_2 = (13.0/12.0) * (u[2] - 2.0*u[3] + u[4])**2 + \
             (1.0/4.0) * (3.0*u[2] - 4.0*u[3] + u[4])**2

    return np.array([beta_0, beta_1, beta_2])

@njit(cache=True, fastmath=True)
def compute_weno_weights_numba(beta, d, epsilon=1e-6):
    """
    Numba-accelerated WENO5 weight computation.

    Args:
        beta: Smoothness indicators [β₀, β₁, β₂]
        d: Optimal linear weights [d₀, d₁, d₂]
        epsilon: Regularization parameter

    Returns:
        w: Nonlinear weights [w₀, w₁, w₂]
    """
    # Compute alpha (unnormalized weights)
    alpha_0 = d[0] / (epsilon + beta[0])**2
    alpha_1 = d[1] / (epsilon + beta[1])**2
    alpha_2 = d[2] / (epsilon + beta[2])**2

    # Normalize
    alpha_sum = alpha_0 + alpha_1 + alpha_2
    w_0 = alpha_0 / alpha_sum
    w_1 = alpha_1 / alpha_sum
    w_2 = alpha_2 / alpha_sum

    return np.array([w_0, w_1, w_2])
```

### Task 2.2: Integrate with HJBWenoSolver
```python
# mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py

# Add at top of file
try:
    from .hjb_weno_numba import (
        compute_smoothness_indicators_numba,
        compute_weno_weights_numba,
        NUMBA_AVAILABLE
    )
except ImportError:
    NUMBA_AVAILABLE = False

class HJBWenoSolver(BaseHJBSolver):
    def __init__(
        self,
        problem: MFGProblem,
        weno_variant: str = "weno5",
        use_numba: bool = True,  # NEW: Enable Numba acceleration
        **kwargs
    ):
        super().__init__(problem)
        self.weno_variant = weno_variant
        self.use_numba = use_numba and NUMBA_AVAILABLE

        if self.use_numba:
            print("✅ Numba acceleration enabled for WENO5")
        else:
            print("⚠️ Numba not available, using pure Python")

        # ... rest of __init__

    def _compute_smoothness_indicators(self, u: np.ndarray) -> np.ndarray:
        """Compute WENO smoothness indicators (Numba-accelerated if available)."""
        if self.use_numba:
            return compute_smoothness_indicators_numba(u)
        else:
            # Original pure Python implementation
            beta_0 = (13/12) * (u[0] - 2*u[1] + u[2])**2 + \
                     (1/4) * (u[0] - 4*u[1] + 3*u[2])**2
            # ... rest of original code
```

### Task 2.3: Benchmark Numba vs Python
```python
# benchmarks/numba/weno5_numba_benchmark.py
import numpy as np
import timeit
from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.numerical.hjb_solvers.hjb_weno import HJBWenoSolver

def benchmark_numba_speedup():
    problem = ExampleMFGProblem(Nx=100, Nt=50)

    # Python version
    solver_py = HJBWenoSolver(problem, use_numba=False)
    u = np.random.randn(5)

    time_py = timeit.timeit(
        lambda: solver_py._compute_smoothness_indicators(u),
        number=10000
    )

    # Numba version
    solver_numba = HJBWenoSolver(problem, use_numba=True)

    # Warmup Numba JIT
    for _ in range(100):
        solver_numba._compute_smoothness_indicators(u)

    time_numba = timeit.timeit(
        lambda: solver_numba._compute_smoothness_indicators(u),
        number=10000
    )

    speedup = time_py / time_numba

    print(f"Python:  {time_py/10000*1e6:.2f} μs per call")
    print(f"Numba:   {time_numba/10000*1e6:.2f} μs per call")
    print(f"Speedup: {speedup:.1f}×")

    return speedup

if __name__ == "__main__":
    speedup = benchmark_numba_speedup()

    if speedup >= 5.0:
        print("\n✅ SUCCESS: Numba achieves target speedup")
        print("   DECISION: Use Numba, no need for Rust")
    elif speedup >= 3.0:
        print("\n⚠️ BORDERLINE: Numba gives 3-5× speedup")
        print("   DECISION: Consider Rust for marginal gain")
    else:
        print("\n❌ INSUFFICIENT: Numba < 3× speedup")
        print("   DECISION: Proceed to Rust implementation")
```

**Decision Point**:
```bash
python benchmarks/numba/weno5_numba_benchmark.py
```

- **If speedup ≥ 5×** → ✅ DONE, use Numba
- **If speedup 3-5×** → Evaluate if marginal Rust gain worth it
- **If speedup < 3×** → Proceed to Step 3 (NumPy optimization)

---

## Step 3: Try NumPy Optimization 🔧 **1 hour**

### Why Try NumPy Optimization?
Sometimes pure NumPy can be faster than expected with:
- Vectorization tricks
- Broadcasting
- Avoiding Python loops

### Task 3.1: Vectorized NumPy Implementation
```python
# mfg_pde/alg/numerical/hjb_solvers/hjb_weno_vectorized.py
"""Vectorized NumPy WENO5 kernels."""

def compute_smoothness_indicators_vectorized(U: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Vectorized WENO5 smoothness indicators for entire array.

    Args:
        U: Array of values (any shape)
        axis: Axis along which to compute stencils

    Returns:
        beta: Smoothness indicators with shape (*U.shape[:-1], 3)
    """
    # Extract stencil points using array slicing (vectorized)
    u0 = np.take(U, 0, axis=axis)
    u1 = np.take(U, 1, axis=axis)
    u2 = np.take(U, 2, axis=axis)
    u3 = np.take(U, 3, axis=axis)
    u4 = np.take(U, 4, axis=axis)

    # Compute all beta values using broadcasting (vectorized)
    beta_0 = (13/12) * (u0 - 2*u1 + u2)**2 + (1/4) * (u0 - 4*u1 + 3*u2)**2
    beta_1 = (13/12) * (u1 - 2*u2 + u3)**2 + (1/4) * (u1 - u3)**2
    beta_2 = (13/12) * (u2 - 2*u3 + u4)**2 + (1/4) * (3*u2 - 4*u3 + u4)**2

    # Stack along new axis
    return np.stack([beta_0, beta_1, beta_2], axis=-1)

def compute_weno_reconstruction_vectorized(U: np.ndarray) -> np.ndarray:
    """
    Vectorized WENO5 reconstruction for entire grid.

    Instead of calling _compute_smoothness_indicators at each point,
    compute for entire array at once.
    """
    Nx = U.shape[-1]

    # Build stencil array (Nx-4, 5) - all stencils at once
    stencils = np.lib.stride_tricks.sliding_window_view(U, 5, axis=-1)

    # Compute all smoothness indicators at once (Nx-4, 3)
    beta_all = compute_smoothness_indicators_vectorized(stencils)

    # Compute all weights at once
    # ... vectorized weight computation

    return reconstructed_values
```

### Task 3.2: Benchmark Vectorized NumPy
```python
# benchmarks/numpy/weno5_vectorized_benchmark.py
import numpy as np
import timeit

def benchmark_vectorized_numpy():
    from mfg_pde.alg.numerical.hjb_solvers.hjb_weno_vectorized import (
        compute_smoothness_indicators_vectorized
    )

    # Test with typical problem size
    Nx = 100
    stencils = np.random.randn(Nx-4, 5)

    # Original: loop over stencils
    def original_loop():
        results = []
        for u in stencils:
            beta_0 = (13/12) * (u[0] - 2*u[1] + u[2])**2 + (1/4) * (u[0] - 4*u[1] + 3*u[2])**2
            beta_1 = (13/12) * (u[1] - 2*u[2] + u[3])**2 + (1/4) * (u[1] - u[3])**2
            beta_2 = (13/12) * (u[2] - 2*u[3] + u[4])**2 + (1/4) * (3*u[2] - 4*u[3] + u[4])**2
            results.append([beta_0, beta_1, beta_2])
        return np.array(results)

    # Vectorized: process all at once
    def vectorized():
        return compute_smoothness_indicators_vectorized(stencils)

    time_loop = timeit.timeit(original_loop, number=1000)
    time_vec = timeit.timeit(vectorized, number=1000)

    speedup = time_loop / time_vec

    print(f"Loop:        {time_loop/1000*1e3:.2f} ms per iteration")
    print(f"Vectorized:  {time_vec/1000*1e3:.2f} ms per iteration")
    print(f"Speedup:     {speedup:.1f}×")

    return speedup

if __name__ == "__main__":
    speedup = benchmark_vectorized_numpy()

    if speedup >= 5.0:
        print("\n✅ SUCCESS: Vectorized NumPy achieves target")
        print("   DECISION: Use vectorized NumPy, no need for Rust")
    else:
        print(f"\n⚠️ INSUFFICIENT: NumPy only {speedup:.1f}× speedup")
        print("   DECISION: Proceed to PyO3 Rust implementation")
```

**Decision Point**:
```bash
python benchmarks/numpy/weno5_vectorized_benchmark.py
```

- **If speedup ≥ 5×** → ✅ DONE, use vectorized NumPy
- **If speedup < 5×** → Proceed to Step 4 (PyO3 Rust)

---

## Step 4: Implement PyO3 Rust 🦀 **1-2 weeks**

### Only if Steps 1-3 Failed to Achieve Target

### Task 4.1: Create Rust Extension Skeleton
```bash
# Create Rust crate
cd /Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE
cargo new --lib mfg_pde_rust
cd mfg_pde_rust

# Configure Cargo.toml
cat > Cargo.toml << 'EOF'
[package]
name = "mfg_pde_rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "mfg_pde_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"

[build-dependencies]
maturin = "1.4"
EOF
```

### Task 4.2: Implement WENO5 Kernels in Rust
```rust
// mfg_pde_rust/src/lib.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn compute_smoothness_indicators(u: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    let u = u.as_slice().unwrap();

    // β₀
    let beta_0 = (13.0/12.0) * (u[0] - 2.0*u[1] + u[2]).powi(2)
                + (1.0/4.0) * (u[0] - 4.0*u[1] + 3.0*u[2]).powi(2);

    // β₁
    let beta_1 = (13.0/12.0) * (u[1] - 2.0*u[2] + u[3]).powi(2)
                + (1.0/4.0) * (u[1] - u[3]).powi(2);

    // β₂
    let beta_2 = (13.0/12.0) * (u[2] - 2.0*u[3] + u[4]).powi(2)
                + (1.0/4.0) * (3.0*u[2] - 4.0*u[3] + u[4]).powi(2);

    Python::with_gil(|py| {
        PyArray1::from_vec(py, vec![beta_0, beta_1, beta_2]).to_owned()
    })
}

#[pymodule]
fn mfg_pde_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_smoothness_indicators, m)?)?;
    Ok(())
}
```

### Task 4.3: Build and Test
```bash
# Install maturin
pip install maturin

# Build in development mode
maturin develop --release

# Test from Python
python -c "
import mfg_pde_rust
import numpy as np

u = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
beta = mfg_pde_rust.compute_smoothness_indicators(u)
print(f'Rust result: {beta}')
"
```

### Task 4.4: Benchmark Rust vs Alternatives
```python
# benchmarks/rust/weno5_rust_benchmark.py
import numpy as np
import timeit

def comprehensive_benchmark():
    try:
        import mfg_pde_rust
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

    from mfg_pde.alg.numerical.hjb_solvers.hjb_weno import HJBWenoSolver

    problem = ExampleMFGProblem(Nx=100, Nt=50)
    u = np.random.randn(5)

    results = {}

    # 1. Pure Python
    solver_py = HJBWenoSolver(problem, use_numba=False)
    time_py = timeit.timeit(
        lambda: solver_py._compute_smoothness_indicators(u),
        number=10000
    )
    results['Python'] = time_py / 10000 * 1e6  # μs

    # 2. Numba
    try:
        solver_numba = HJBWenoSolver(problem, use_numba=True)
        # Warmup
        for _ in range(100):
            solver_numba._compute_smoothness_indicators(u)
        time_numba = timeit.timeit(
            lambda: solver_numba._compute_smoothness_indicators(u),
            number=10000
        )
        results['Numba'] = time_numba / 10000 * 1e6
    except:
        results['Numba'] = None

    # 3. Rust
    if RUST_AVAILABLE:
        time_rust = timeit.timeit(
            lambda: mfg_pde_rust.compute_smoothness_indicators(u),
            number=10000
        )
        results['Rust'] = time_rust / 10000 * 1e6
    else:
        results['Rust'] = None

    # Print comparison
    baseline = results['Python']
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("="*50)
    for name, time_us in results.items():
        if time_us is not None:
            speedup = baseline / time_us
            print(f"{name:12} {time_us:8.2f} μs  ({speedup:5.1f}× speedup)")
        else:
            print(f"{name:12} {'N/A':>8}")

    print("="*50)

    # Make recommendation
    if results['Rust'] and results['Rust'] < results.get('Numba', float('inf')):
        print("\n✅ RECOMMENDATION: Use Rust (fastest)")
    elif results['Numba'] and results['Numba'] / baseline >= 5.0:
        print("\n✅ RECOMMENDATION: Use Numba (good enough, simpler)")
    elif results['Rust']:
        print("\n⚠️ RECOMMENDATION: Use Rust (Numba insufficient)")
    else:
        print("\n❌ WARNING: No acceleration method achieved target")

if __name__ == "__main__":
    comprehensive_benchmark()
```

---

## Decision Tree Summary

```
┌─────────────────────────────────────┐
│  1. Profile & Measure Baseline     │
│     Time: 1 hour                    │
└─────────────┬───────────────────────┘
              │
              ▼
        Bottleneck < 100μs?
              │
        Yes ──┴── No → Continue
              │
         ✅ STOP (Not worth optimizing)

              ▼
┌─────────────────────────────────────┐
│  2. Try Numba JIT                   │
│     Time: 2 hours                   │
└─────────────┬───────────────────────┘
              │
              ▼
        Speedup ≥ 5×?
              │
        Yes ──┴── No → Continue
              │
         ✅ USE NUMBA

              ▼
┌─────────────────────────────────────┐
│  3. Try Vectorized NumPy            │
│     Time: 1 hour                    │
└─────────────┬───────────────────────┘
              │
              ▼
        Speedup ≥ 5×?
              │
        Yes ──┴── No → Continue
              │
         ✅ USE VECTORIZED NUMPY

              ▼
┌─────────────────────────────────────┐
│  4. Implement PyO3 Rust             │
│     Time: 1-2 weeks                 │
└─────────────┬───────────────────────┘
              │
              ▼
         ✅ USE RUST
```

---

## What We Need: Equipment Check

### Step 1 (Profiling) - Already Have ✅
- Python 3.10+
- cProfile (built-in)
- timeit (built-in)

### Step 2 (Numba) - Install if Needed
```bash
pip install numba
```

### Step 3 (NumPy) - Already Have ✅
- NumPy 2.0+ (already installed)

### Step 4 (Rust) - Install if Needed
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Verify installation
rustc --version
cargo --version
maturin --version
```

---

## Execution Plan: Next 4 Hours

### Hour 1: Profile & Baseline ⏱️
```bash
# 1. Run profiler
python -m cProfile -o weno_profile.prof [script]

# 2. Analyze
python -m pstats weno_profile.prof

# 3. Microbenchmark
python benchmarks/baseline/weno5_smoothness_baseline.py
```

**Deliverable**: `baseline_performance.txt` with measurements

### Hour 2-3: Numba Implementation ⚡
```bash
# 1. Write Numba kernels (30 min)
# 2. Integrate with HJBWenoSolver (30 min)
# 3. Test correctness (30 min)
# 4. Benchmark (30 min)
```

**Deliverable**: Numba speedup measurement

### Hour 4: Decision & Documentation 📝
- If Numba ≥ 5× speedup → Document and close
- If Numba < 5× speedup → Plan Rust implementation

---

## Final Answer: Do We Need PyO3?

**TL;DR**: We don't know yet. Follow the procedure:

1. ✅ **Always do**: Profile & baseline (1 hour)
2. ✅ **Try first**: Numba (2 hours, 70% chance of success)
3. ✅ **Try second**: Vectorized NumPy (1 hour, 30% chance)
4. ⚠️ **Only if needed**: PyO3 Rust (1-2 weeks, 95% chance)

**Probability we need Rust**: ~20-30%
- Most numerical Python bottlenecks are solved by Numba
- Rust is overkill unless Numba fails

**Recommendation**: Start with Step 1 (profiling) now. We'll know in 1 hour if optimization is even necessary.

---

## Status

- [ ] Step 1: Profile & Baseline (1 hour)
- [ ] Step 2: Numba Implementation (2 hours)
- [ ] Step 3: NumPy Vectorization (1 hour)
- [ ] Step 4: PyO3 Rust (1-2 weeks) - only if Steps 2-3 fail

**Current Action**: Ready to start Step 1 profiling
