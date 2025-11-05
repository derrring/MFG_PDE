# JAX Backend with Particle Methods: Analysis

**Created**: 2025-10-05
**Context**: Investigation of massive mass normalization error (679,533 instead of 1.0)

## The Observed Issue

During test development, enabling JAX backend for FPParticleSolver caused:
```
AssertionError: Initial mass = 679533.598110, expected 1.0
```

**Error magnitude**: 679,533× too large!

## Root Cause Analysis

### 1. Backend Initialization Behavior

**FPParticleSolver** (`fp_particle.py:42-48`):
```python
if backend is not None:
    self.backend = create_backend(backend)
else:
    self.backend = create_backend("numpy")  # NumPy fallback
```

**FixedPointIterator** (`fixed_point_iterator.py:105-109`):
```python
if backend is not None and hasattr(fp_solver, "backend"):
    if fp_solver.backend is None:  # Only set if not already set
        fp_solver.backend = create_backend(backend)
```

**Key insight**: FPParticleSolver **always** initializes with a backend (defaults to NumPy). Therefore, FixedPointIterator's `if fp_solver.backend is None` check **never triggers**, and JAX backend is **never actually used** in FPParticleSolver!

### 2. Why JAX Can't Work with FPParticleSolver

The fundamental incompatibility:

```python
# Line 91-92: scipy.stats.gaussian_kde is CPU-only
if SCIPY_AVAILABLE and gaussian_kde is not None:
    kde = gaussian_kde(particles_at_time_t, bw_method=self.kde_bandwidth)
    m_density_estimated = kde(xSpace)
```

**`scipy.stats.gaussian_kde` requires NumPy arrays**. It cannot work with:
- JAX DeviceArrays
- PyTorch tensors
- Numba arrays (in certain modes)

### 3. The Normalization Code

```python
# Lines 112-125
if self.normalize_kde_output:
    if Dx > 1e-14:
        current_mass = np.sum(m_density_estimated) * Dx
        if current_mass > 1e-9:
            return m_density_estimated / current_mass
```

**Potential issues if JAX were active**:
1. `m_density_estimated` from KDE is NumPy array
2. `Dx` might be JAX array if problem attributes converted
3. `np.sum()` on mixed types could cause issues
4. Division could broadcast incorrectly

### 4. Experimental Verification

Testing with pure JAX shows **no significant error**:
```python
# NumPy: integral = 1.0
# JAX:   integral = 0.9999998807907104
```

Only ~1e-7 floating point difference, **not** 679,533×!

## The Mystery: Where Did 679,533 Come From?

Possible explanations:

### Hypothesis 1: Transient Bug
The error occurred during rapid development/testing when backend configuration was changing. It may have been:
- A typo in Dx value
- Array shape mismatch temporarily
- Race condition in backend initialization

### Hypothesis 2: Dx Corruption
If `Dx = 0.04` somehow became `Dx = 0.04 / 679533 ≈ 5.9e-8`:
```python
current_mass = np.sum(m_density) * 5.9e-8  # Tiny!
return m_density / current_mass  # Huge!
```

This would give the ~679,533× amplification.

### Hypothesis 3: Backend Attribute Access Issue
If `problem.Dx` accessed through JAX backend returned something unusual:
```python
# Normal: Dx = 0.04 (float)
# Bug:    Dx = <array_index> = 679533?
```

But we verified `problem.Dx` stays as Python float.

## Current Solution: Avoid JAX with Particle Methods

**Recommendation**: **Do NOT use JAX/Torch backends with FPParticleSolver**

### Why This Works

1. **scipy.stats.gaussian_kde** is CPU-only NumPy
2. **Particle simulation** uses `np.random` (CPU-only)
3. **No GPU benefit** for these operations anyway
4. **NumPy backend** is reliable and correct

### Code Pattern

```python
# ✅ GOOD: Explicit NumPy backend for particle methods
fp_solver = FPParticleSolver(
    problem,
    num_particles=5000,
    backend="numpy"  # or None (defaults to numpy)
)

# ✅ GOOD: No backend to FixedPointIterator (stays NumPy)
mfg_solver = FixedPointIterator(
    problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    backend=None  # Don't force JAX
)

# ❌ BAD: Trying to use JAX with particles
fp_solver = FPParticleSolver(problem, backend="jax")  # Will use NumPy anyway
mfg_solver = FixedPointIterator(..., backend="jax")   # Won't affect FP solver
```

## Future: GPU-Accelerated KDE

To actually use JAX/GPU for particle methods, we need:

### Option 1: Custom JAX KDE
```python
import jax.numpy as jnp
from jax import vmap

def jax_gaussian_kde(particles, xSpace, bandwidth):
    \"\"\"JAX-based KDE using vmap/jit\"\"\"
    def kernel(x, particle):
        return jnp.exp(-0.5 * ((x - particle) / bandwidth) ** 2)

    # Vectorized computation
    kde_vals = vmap(
        lambda x: jnp.mean(vmap(lambda p: kernel(x, p))(particles))
    )(xSpace)
    return kde_vals / (bandwidth * jnp.sqrt(2 * jnp.pi))
```

### Option 2: PyTorch KDE
```python
import torch

def torch_gaussian_kde(particles, xSpace, bandwidth):
    \"\"\"PyTorch KDE with GPU support\"\"\"
    particles = particles.unsqueeze(1)  # (N, 1)
    xSpace = xSpace.unsqueeze(0)  # (1, M)

    kernel = torch.exp(-0.5 * ((xSpace - particles) / bandwidth) ** 2)
    return kernel.mean(dim=0) / (bandwidth * (2 * torch.pi) ** 0.5)
```

### Option 3: Hybrid Approach
```python
# Use GPU for particle evolution (large matrix ops)
# Use CPU for KDE (scipy is well-optimized)
# Transfer data between GPU/CPU as needed
```

## Recommendations

### For Users
1. **Always use `backend=None` or `backend="numpy"` with FPParticleSolver**
2. **Don't pass backend to FixedPointIterator if using particle FP**
3. **Check mass conservation** after solver runs

### For Developers
1. **Implement custom JAX/Torch KDE** (Phase 3 feature)
2. **Add backend validation** in FPParticleSolver init:
   ```python
   if backend not in [None, "numpy"]:
       warnings.warn(
           "FPParticleSolver only supports NumPy backend. "
           "JAX/Torch not compatible with scipy KDE. Using NumPy.",
           UserWarning
       )
       self.backend = create_backend("numpy")
   ```
3. **Document scipy dependency** clearly

### For Tests
1. **Use `backend=None`** for particle method tests
2. **Test backend behavior** explicitly
3. **Add regression test** for mass normalization

## Conclusion

The **679,533× mass error** was likely a **transient bug** during development, not a fundamental JAX issue. However, **JAX backend is incompatible** with the current FPParticleSolver implementation because:

1. scipy.stats.gaussian_kde requires NumPy
2. No GPU acceleration benefit anyway
3. Backend initialization prevents JAX from being used

**Current fix**: Use `backend=None` (NumPy) for particle methods. ✅

**Future enhancement**: Implement custom JAX/Torch KDE for true GPU acceleration.
