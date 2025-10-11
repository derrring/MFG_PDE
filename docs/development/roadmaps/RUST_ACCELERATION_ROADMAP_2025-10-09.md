# Rust Acceleration Roadmap for MFG_PDE

**Date**: 2025-10-09
**Status**: Planning Phase
**Target**: High-performance compute kernels using PyO3

## Executive Summary

Strategic plan for integrating Rust extensions into MFG_PDE to accelerate compute-intensive numerical operations. Primary focus: WENO5 stencil computations with expected 5-10× speedup.

## Strategic Analysis: Where to Use Rust

Based on codebase profiling and architectural analysis:

### 1. WENO5 Stencil Computations ⭐ **TOP PRIORITY**
**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`

**Why Rust?**
- Tight loops with complex arithmetic (smoothness indicators, weight calculations)
- Per-grid-point operations (Nx × Nt iterations)
- Already isolated, pure numerical code
- No Python object overhead

**Current Bottleneck** (lines 286-315):
```python
def _compute_smoothness_indicators(self, u: np.ndarray) -> np.ndarray:
    # β₀ computation
    beta_0 = (13/12) * (u[0] - 2*u[1] + u[2])**2 + (1/4) * (u[0] - 4*u[1] + 3*u[2])**2

    # β₁ computation
    beta_1 = (13/12) * (u[1] - 2*u[2] + u[3])**2 + (1/4) * (u[1] - u[3])**2

    # β₂ computation
    beta_2 = (13/12) * (u[2] - 2*u[3] + u[4])**2 + (1/4) * (3*u[2] - 4*u[3] + u[4])**2

    return np.array([beta_0, beta_1, beta_2])
```

**Rust Implementation Strategy**:
```rust
// PyO3 extension for WENO smoothness indicators
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn compute_smoothness_indicators(u: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    let u = u.as_slice().unwrap();

    // β₀ computation (vectorized in Rust)
    let beta_0 = (13.0/12.0) * (u[0] - 2.0*u[1] + u[2]).powi(2)
                + (1.0/4.0) * (u[0] - 4.0*u[1] + 3.0*u[2]).powi(2);

    // β₁ computation
    let beta_1 = (13.0/12.0) * (u[1] - 2.0*u[2] + u[3]).powi(2)
                + (1.0/4.0) * (u[1] - u[3]).powi(2);

    // β₂ computation
    let beta_2 = (13.0/12.0) * (u[2] - 2.0*u[3] + u[4]).powi(2)
                + (1.0/4.0) * (3.0*u[2] - 4.0*u[3] + u[4]).powi(2);

    // Return as NumPy array (zero-copy)
    Python::with_gil(|py| {
        PyArray1::from_vec(py, vec![beta_0, beta_1, beta_2]).to_owned()
    })
}
```

**Expected Speedup**: 5-10× for stencil operations

---

### 2. Particle Evolution and Monte Carlo ⭐ **HIGH PRIORITY**
**File**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`

**Why Rust?**
- Particle advection loop (5000-10000 particles per time step)
- Embarrassingly parallel (each particle independent)
- Heavy random number generation

**Current Bottleneck** (lines 316-318):
```python
for n_time_idx in range(Nt - 1):
    # ... gradient computation

    # Particle update
    current_M_particles_t[n_time_idx + 1, :] = (
        current_M_particles_t[n_time_idx, :] + alpha_optimal_at_particles * Dt + sigma_sde * dW
    )
```

**Rust Implementation Strategy**:
```rust
use rayon::prelude::*;  // Data parallelism

#[pyfunction]
fn evolve_particles_parallel(
    positions: Vec<f64>,
    drift: Vec<f64>,
    noise: Vec<f64>,
    dt: f64
) -> Vec<f64> {
    positions.par_iter()
        .zip(drift.par_iter())
        .zip(noise.par_iter())
        .map(|((x, v), n)| {
            x + v * dt + n
        })
        .collect()
}
```

**Expected Speedup**: 10-20× with Rayon parallelism

---

### 3. Kernel Density Estimation (KDE) ⭐ **MEDIUM PRIORITY**
**File**: `mfg_pde/alg/numerical/density_estimation.py`

**Why Rust?**
- Currently uses `scipy.stats.gaussian_kde` (Python overhead)
- Double loop over particles and grid points
- Already has GPU version, Rust could be optimized CPU alternative

**Rust Implementation Strategy**:
```rust
#[pyfunction]
fn gaussian_kde_rust(
    particles: PyReadonlyArray1<f64>,
    grid: PyReadonlyArray1<f64>,
    bandwidth: f64
) -> Py<PyArray1<f64>> {
    let particles_slice = particles.as_slice().unwrap();
    let grid_slice = grid.as_slice().unwrap();

    // SIMD-optimized KDE
    let density: Vec<f64> = grid_slice.par_iter()
        .map(|&x_grid| {
            particles_slice.iter()
                .map(|&x_particle| {
                    let z = (x_grid - x_particle) / bandwidth;
                    (-0.5 * z * z).exp() / (bandwidth * SQRT_2PI)
                })
                .sum()
        })
        .collect();

    Python::with_gil(|py| {
        PyArray1::from_vec(py, density).to_owned()
    })
}
```

**Expected Speedup**: 3-5× over pure Python

---

### 4. Anderson Acceleration ⭐ **LOW-MEDIUM PRIORITY**
**File**: `mfg_pde/alg/numerical/mfg_solvers/fixed_point_utils.py`

**Why Consider?**
- Dense linear algebra (QR decomposition, least squares)
- Memory management (history buffers, circular arrays)

**Why NOT Priority?**
- Currently uses NumPy/SciPy (already uses BLAS/LAPACK)
- Rust won't provide significant benefit over existing implementation
- Complex to implement correctly

**Expected Speedup**: 2-3× (marginal improvement)

---

### 5. Sparse Matrix Operations ❌ **LOW PRIORITY**
**Why NOT Rust?**
- Currently uses `scipy.sparse` (highly optimized in C/Fortran)
- Rust won't provide benefit over existing implementation

---

## Rust Integration Approaches

### Option 1: PyO3 (Recommended) ✅
**Best for**: Individual hot functions, drop-in replacements

**Pros**:
- Zero-copy NumPy array sharing
- Easy integration with existing Python code
- Fine-grained performance optimization
- Uses `maturin` for easy building
- Modern, well-maintained

**Cons**:
- Requires Rust toolchain
- Build complexity for distribution

**Example Workflow**:
```bash
# Create Rust extension
cargo new --lib mfg_pde_rust
cd mfg_pde_rust

# Add to Cargo.toml
[lib]
name = "mfg_pde_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
rayon = "1.8"  # Parallelism

# Build with maturin
maturin develop --release

# Import in Python
from mfg_pde_rust import compute_smoothness_indicators
```

---

### Option 2: RustPython ❌ **NOT RECOMMENDED**
**Why not?**
- RustPython is a full Python interpreter in Rust
- Designed for embedding Python in Rust applications
- **Wrong direction** for our use case (we want Rust in Python, not Python in Rust)
- Adds complexity without performance benefit

---

### Option 3: Hybrid Backend Strategy ✅ **FUTURE DIRECTION**
Add Rust as a 4th backend alongside NumPy, PyTorch, JAX:

```python
# mfg_pde/backends/rust_backend.py
class RustBackend(BackendProtocol):
    def __init__(self):
        import mfg_pde_rust
        self.rust = mfg_pde_rust

    def weno_smoothness(self, u):
        return self.rust.compute_smoothness_indicators(u)

    def evolve_particles(self, positions, velocities, dt):
        return self.rust.evolve_particles_parallel(positions, velocities, dt)
```

---

## Implementation Roadmap

### Phase 1: WENO5 Optimization (Highest ROI) 🎯
**Timeline**: 1-2 weeks
**Priority**: Critical for high-order solver performance

**Tasks**:
1. ✅ Create `mfg_pde_rust` crate with PyO3
2. ✅ Implement `compute_smoothness_indicators` in Rust
3. ✅ Add `compute_weno_weights` with SIMD
4. ✅ Benchmark against NumPy version
5. ✅ Add as optional backend in `HJBWenoSolver`
6. ✅ Update documentation

**Acceptance Criteria**:
- 5× speedup minimum
- Zero-copy NumPy integration
- Fallback to Python if Rust not available
- Pass all existing WENO tests

---

### Phase 2: Particle Methods 🚀
**Timeline**: 2-3 weeks
**Priority**: High for large-scale particle simulations

**Tasks**:
1. Implement `evolve_particles_parallel` with Rayon
2. Add Rust-based KDE computation
3. Benchmark against existing particle solver
4. Integrate with backend selection system
5. Add particle-specific optimizations (boundary conditions)

**Acceptance Criteria**:
- 10× speedup for particle evolution
- Handles 100k+ particles efficiently
- Maintains mass conservation properties

---

### Phase 3: Advanced Optimizations 📈
**Timeline**: 1-2 weeks
**Priority**: Medium (only if Phase 1-2 show good results)

**Tasks**:
1. Evaluate Anderson acceleration in Rust
2. Custom sparse matrix operations (if needed)
3. Profile and optimize critical paths
4. SIMD optimizations where applicable

---

## Project Structure

```
MFG_PDE/
├── mfg_pde/                  # Python package
│   ├── backends/
│   │   └── rust_backend.py   # Rust backend interface
│   └── ...
├── mfg_pde_rust/             # Rust extension (NEW)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs            # PyO3 bindings
│   │   ├── weno.rs           # WENO kernels
│   │   ├── particles.rs      # Particle evolution
│   │   └── kde.rs            # KDE kernels
│   └── benches/              # Rust benchmarks
├── pyproject.toml            # Python build config
└── setup.py                  # Fallback build (optional)
```

---

## Dependencies

### Rust Dependencies (Cargo.toml)
```toml
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
rayon = "1.8"     # Data parallelism
ndarray = "0.15"  # N-dimensional arrays

[build-dependencies]
maturin = "1.4"
```

### Python Dependencies (pyproject.toml)
```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project.optional-dependencies]
rust = [
    "mfg-pde-rust>=0.1.0",  # Rust extension (optional)
]
```

---

## Testing Strategy

### Unit Tests (Rust)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothness_indicators() {
        let u = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let beta = compute_smoothness_indicators_internal(&u);

        // Verify mathematical properties
        assert!(beta[0] >= 0.0);
        assert!(beta[1] >= 0.0);
        assert!(beta[2] >= 0.0);
    }
}
```

### Integration Tests (Python)
```python
def test_rust_weno_accuracy():
    """Verify Rust WENO matches Python implementation."""
    import mfg_pde_rust
    import numpy as np

    u = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

    # Python reference
    beta_py = compute_smoothness_indicators_python(u)

    # Rust implementation
    beta_rust = mfg_pde_rust.compute_smoothness_indicators(u)

    np.testing.assert_allclose(beta_rust, beta_py, rtol=1e-14)
```

### Performance Benchmarks
```python
def benchmark_weno_rust_vs_python():
    """Benchmark Rust vs Python WENO."""
    import timeit
    import mfg_pde_rust

    setup = "import numpy as np; u = np.random.randn(5)"

    time_py = timeit.timeit("compute_smoothness_indicators_python(u)", setup=setup, number=10000)
    time_rust = timeit.timeit("mfg_pde_rust.compute_smoothness_indicators(u)", setup=setup, number=10000)

    speedup = time_py / time_rust
    print(f"Rust speedup: {speedup:.1f}×")
    assert speedup > 5.0  # Minimum 5× speedup
```

---

## Distribution Strategy

### Development Build
```bash
# Install in development mode
cd mfg_pde_rust
maturin develop --release

# Python can now import Rust extension
python -c "import mfg_pde_rust; print('Success!')"
```

### PyPI Distribution
```bash
# Build wheels for multiple platforms
maturin build --release --manylinux 2014

# Upload to PyPI
maturin publish
```

### Optional Dependency
Make Rust extension optional:
```python
# mfg_pde/backends/rust_backend.py
try:
    import mfg_pde_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

if not RUST_AVAILABLE:
    # Fallback to Python implementation
    warnings.warn("Rust extension not available, using Python fallback")
```

---

## Performance Targets

| Component | Current (Python) | Target (Rust) | Speedup |
|:----------|:-----------------|:--------------|:--------|
| WENO5 Smoothness | 100 μs | 10-20 μs | 5-10× |
| WENO5 Weights | 150 μs | 20-30 μs | 5-7× |
| Particle Evolution (10k) | 50 ms | 2-5 ms | 10-25× |
| KDE (10k particles) | 200 ms | 40-60 ms | 3-5× |

---

## Risk Analysis

### Technical Risks
1. **Build complexity**: Rust toolchain required for installation
   - **Mitigation**: Provide pre-built wheels for common platforms
   - **Fallback**: Pure Python implementation always available

2. **Numerical precision**: Rust vs Python floating-point differences
   - **Mitigation**: Extensive testing with tight tolerances (rtol=1e-14)
   - **Validation**: Compare against analytical solutions

3. **Maintenance burden**: Two implementations to maintain
   - **Mitigation**: Keep Rust extensions isolated, well-documented
   - **Strategy**: Only Rustify proven bottlenecks

### Distribution Risks
1. **Platform compatibility**: Rust might not compile on all platforms
   - **Mitigation**: Provide wheels for Windows, macOS, Linux
   - **Fallback**: Pure Python always works

2. **Dependency management**: Rust toolchain adds complexity
   - **Mitigation**: Optional dependency, not required for core functionality

---

## Success Metrics

1. **Performance**: 5-10× speedup for WENO5 operations
2. **Accuracy**: Numerical results match Python to machine precision
3. **Usability**: Transparent fallback when Rust not available
4. **Maintainability**: Clear separation between Python and Rust code
5. **Adoption**: Used in production benchmarks and publications

---

## References

- PyO3 documentation: https://pyo3.rs
- Maturin build tool: https://www.maturin.rs
- Rayon parallelism: https://github.com/rayon-rs/rayon
- NumPy Rust bindings: https://github.com/PyO3/rust-numpy

---

## Next Steps

1. **Immediate**: Create `mfg_pde_rust` crate skeleton
2. **Week 1**: Implement WENO5 smoothness indicators
3. **Week 2**: Benchmark and integrate with `HJBWenoSolver`
4. **Week 3**: Document and publish Phase 1 results

**Status**: Ready to proceed with Phase 1 implementation
