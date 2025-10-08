# MFG_PDE Rust Experimental Extensions

**Status**: 🧪 EXPERIMENTAL - Learning/exploration project

This directory contains experimental Rust code for learning PyO3 and exploring potential performance optimization opportunities for MFG_PDE.

## Purpose

Based on the acceleration decision analysis (see `../docs/development/[COMPLETED]_ACCELERATION_DECISION_2025-10-09.md`), we determined that **PyO3 is not currently needed** for WENO5 or other numerical kernels. However, this experimental crate serves as:

1. **Learning environment** for Rust + Python integration
2. **Future-proofing** if performance bottlenecks emerge
3. **Reference implementation** for educational purposes

## What's Included

### Simple Examples (Learning PyO3)
- `hello_rust()` - Basic string return
- `add(a, b)` - Simple arithmetic

### NumPy Integration (Learning numpy-rs)
- `sum_array(arr)` - Sum NumPy array
- `mean_array(arr)` - Compute mean
- `square_array(arr)` - Element-wise square

### Real Numerical Kernel (Educational)
- `weno5_smoothness(u)` - WENO5 smoothness indicators
  - Same computation as Python implementation
  - Shows how to translate mathematical formulas to Rust
  - Useful for learning, not needed for performance (NumPy already fast)

## Building and Testing

### Prerequisites
Install Rust and Maturin:
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Maturin (in your Python environment)
pip install maturin
```

### Build and Install
```bash
cd mfg_pde_rust_experimental

# Build and install in current Python environment (development mode)
maturin develop

# Or build optimized release version
maturin develop --release
```

### Test
```bash
# Run Python tests
python test_rust.py

# Run Rust unit tests
cargo test

# Verify installation
python -c "import mfg_pde_rust_experimental; print(mfg_pde_rust_experimental.hello_rust())"
```

## Performance Comparison

### WENO5 Smoothness Indicators
Based on baseline measurements from `../benchmarks/baseline/`:

- **NumPy (Python)**: 2.08 μs per call
- **Rust (expected)**: ~0.2 μs per call (10× speedup)
- **Verdict**: Not worth the complexity (saves only 0.09s in full solve)

To verify Rust performance:
```python
import numpy as np
import mfg_pde_rust_experimental as rust

u = np.random.randn(5)

# Time Rust version
%timeit rust.weno5_smoothness(u)

# Compare with Python version
from mfg_pde import ExampleMFGProblem
from mfg_pde.alg.numerical.hjb_solvers.hjb_weno import HJBWenoSolver

problem = ExampleMFGProblem(Nx=100, Nt=50, T=1.0)
solver = HJBWenoSolver(problem, weno_variant="weno5")

%timeit solver._compute_smoothness_indicators(u)
```

## Learning Resources

### Understanding the Code

1. **Read `src/lib.rs`**:
   - Start with `hello_rust()` - simplest example
   - Progress to `sum_array()` - NumPy integration
   - Study `weno5_smoothness()` - real numerical kernel

2. **Key PyO3 Concepts**:
   - `#[pyfunction]` - Export function to Python
   - `PyResult<T>` - Return value with error handling
   - `PyReadonlyArray1<f64>` - Read NumPy arrays
   - `PyArray1::from_vec_bound()` - Create NumPy arrays

3. **Rust Syntax Highlights**:
   - `let` - Variable binding
   - `&x` - Reference (borrow)
   - `.powi(2)` - Integer power
   - `.iter().sum()` - Iterator methods

### Experimenting

1. **Modify functions**:
   - Change `weno5_smoothness()` computation
   - Add new array operations
   - Try 2D arrays (`PyArray2`)

2. **Run tests**:
   ```bash
   # After modifying src/lib.rs
   maturin develop
   python test_rust.py
   ```

3. **Profile**:
   ```bash
   # Compare with Python baseline
   python ../benchmarks/baseline/weno5_baseline.py
   ```

## When to Actually Use Rust

Based on acceleration decision analysis, use Rust when:

1. **Profiling confirms bottleneck** (>10 μs per call)
2. **Numba doesn't help** (tried first, simpler)
3. **Vectorized NumPy doesn't help** (tried second)
4. **Expected speedup justifies complexity** (ROI analysis)

### Current Bottleneck Priorities

From `../docs/development/[COMPLETED]_ACCELERATION_DECISION_2025-10-09.md`:

1. **Fixed-point convergence** (algorithmic, not Rust)
2. **Particle methods** (large N, good Rust candidate)
3. **Matrix solvers** (already optimized via scipy)

## Project Structure

```
mfg_pde_rust_experimental/
├── Cargo.toml           # Rust package configuration
├── src/
│   └── lib.rs          # Rust source code with PyO3 bindings
├── test_rust.py        # Python integration tests
└── README.md           # This file
```

## Integration Strategy (If Needed Later)

If performance analysis later shows Rust is beneficial:

1. **Keep experimental crate separate** (don't break main package)
2. **Make it optional dependency** (`pip install mfg-pde[rust]`)
3. **Provide pure-Python fallback** (always available)
4. **Document performance gains** (before/after benchmarks)

Example usage pattern:
```python
# In mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py
try:
    import mfg_pde_rust_experimental as rust
    USE_RUST = True
except ImportError:
    USE_RUST = False

def _compute_smoothness_indicators(self, u):
    if USE_RUST:
        return rust.weno5_smoothness(u)
    else:
        # Pure Python fallback (current implementation)
        beta_0 = (13/12) * (u[0] - 2*u[1] + u[2])**2 + ...
        return np.array([beta_0, beta_1, beta_2])
```

## Decision Log Reference

See detailed analysis in:
- `../docs/development/[COMPLETED]_ACCELERATION_DECISION_2025-10-09.md`
- `../docs/development/ACCELERATION_DECISION_PROCEDURE_2025-10-09.md`
- `../docs/development/RUST_ACCELERATION_ROADMAP_2025-10-09.md`

**Key finding**: Always profile first, optimize second. Saved 1-2 weeks by measuring (1 hour) before implementing Rust.

---

**Created**: 2025-10-09
**Purpose**: Learning and future-proofing
**Status**: Educational - not integrated into main package
