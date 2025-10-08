# Track B Phase 2: Full GPU Particle Pipeline - Design Document

**Date**: 2025-10-08
**Status**: Design Phase
**Branch**: `feature/track-b-phase2-gpu-pipeline`
**Prerequisites**: ✅ Phase 1 complete (GPU KDE infrastructure)

---

## Objective

Implement full GPU particle evolution pipeline to achieve **5-10x overall speedup** for particle-based MFG solvers by eliminating CPU↔GPU transfer overhead identified in Phase 1 analysis.

---

## Phase 1 Learnings Applied

**Key Insight from Phase 1**: Transfer overhead (30-40%) prevents isolated GPU operations from achieving target speedup.

**Solution**: Keep **all particle data on GPU** throughout evolution:
```python
# Phase 1 (transfer overhead per iteration)
for t in range(Nt):
    particles_np = evolve_numpy(...)
    particles_gpu = backend.from_numpy(...)  # ← overhead
    density = gpu_kde(...)
    density_np = backend.to_numpy(...)       # ← overhead

# Phase 2 (transfer once at start/end)
X_gpu = initialize_gpu(...)
for t in range(Nt):
    # ALL operations on GPU (no transfers!)
    X_gpu[t+1] = evolve_gpu(X_gpu[t], ...)
    M_gpu[t+1] = gpu_kde_internal(X_gpu[t+1], ...)
result = backend.to_numpy(X_gpu, M_gpu)
```

---

## Architecture Design

### Component Breakdown

#### 1. GPU Particle State Management

**Purpose**: Keep particle trajectories on GPU throughout solve

**Implementation**:
```python
class FPParticleSolver(BaseFPSolver):
    def solve_fp_system_gpu(self, m_initial, U_drift):
        """Full GPU particle evolution."""
        Nt, Nx = U_drift.shape
        N = self.num_particles

        # Allocate on GPU ONCE
        X_particles = self.backend.zeros((Nt, N), device=self.backend.device)
        M_density = self.backend.zeros((Nt, Nx), device=self.backend.device)

        # Initialize particles on GPU
        X_particles[0, :] = self._sample_initial_particles_gpu(m_initial)

        # Evolution loop (all GPU)
        for t in range(Nt - 1):
            X_particles[t+1, :], M_density[t+1, :] = self._evolve_timestep_gpu(
                X_particles[t, :], U_drift[t, :], t
            )

        # Transfer to CPU ONCE at end
        return self.backend.to_numpy(M_density)
```

**Key Design Decisions**:
- Store full trajectory `(Nt, N)` on GPU (memory permitting)
- Alternative: Keep only current/next timestep, checkpoint periodically
- Use backend tensors natively (not convert per operation)

#### 2. GPU Interpolation

**Purpose**: Interpolate gradient ∇U from grid to particle positions (5-10x speedup target)

**Current (CPU)**:
```python
from scipy.interpolate import interp1d
interp_func = interp1d(x_grid, dUdx_grid, kind="linear")
dUdx_particles = interp_func(particles)  # Sequential
```

**GPU Implementation**:
```python
def interpolate_1d_gpu(x_query, x_grid, y_grid, backend):
    """
    GPU-accelerated 1D linear interpolation.

    Interpolates y_grid defined on x_grid to x_query points.
    All arrays are backend tensors (GPU).

    Implementation:
    - Find bracketing indices via searchsorted
    - Compute linear weights
    - Gather values and interpolate (all parallel)
    """
    xp = backend.array_module

    # Ensure sorted grid
    # Find bracketing indices for each query point
    # Handles boundary: clamp to [0, len(x_grid)-2]
    indices = xp.searchsorted(x_grid, x_query)
    indices = xp.clip(indices, 1, len(x_grid) - 1)

    # Lower and upper bracket indices
    idx_lo = indices - 1
    idx_up = indices

    # Gather grid values (parallel)
    x_lo = x_grid[idx_lo]
    x_up = x_grid[idx_up]
    y_lo = y_grid[idx_lo]
    y_up = y_grid[idx_up]

    # Linear interpolation weights
    weight = (x_query - x_lo) / (x_up - x_lo + 1e-10)

    # Interpolated values (all parallel)
    y_interp = y_lo + weight * (y_up - y_lo)

    return y_interp
```

**Complexity**: O(N) with parallel gather operations
**Expected Speedup**: 5-10x (scipy is sequential gather, GPU is parallel)

#### 3. GPU Particle Evolution

**Purpose**: Update all N particles in parallel

**Implementation**:
```python
def _evolve_timestep_gpu(self, X_current, U_current, timestep_idx):
    """
    Evolve particles for one timestep (all on GPU).

    Returns:
        X_next: Updated particle positions
        M_next: Density on grid
    """
    xp = self.backend.array_module

    # Compute gradient on grid (small array, can be CPU or GPU)
    if hasattr(U_current, 'cpu'):
        U_np = self.backend.to_numpy(U_current)
    else:
        U_np = U_current

    dUdx_grid = self._compute_gradient_numpy(U_np)  # FD on grid
    dUdx_grid_gpu = self.backend.from_numpy(dUdx_grid)
    x_grid_gpu = self.backend.from_numpy(self.problem.xSpace)

    # Interpolate gradient to particle positions (GPU parallel)
    dUdx_particles = interpolate_1d_gpu(
        X_current, x_grid_gpu, dUdx_grid_gpu, self.backend
    )

    # Compute drift (GPU parallel)
    coefCT = self.problem.running_cost_coef
    drift = -coefCT * dUdx_particles

    # Random noise (GPU native RNG)
    sigma = self.problem.sigma
    dt = self.problem.Dt
    noise = xp.randn(self.num_particles) * sigma * xp.sqrt(dt)

    # Euler-Maruyama update (GPU parallel)
    X_next = X_current + drift * dt + noise

    # Apply boundary conditions (GPU parallel)
    X_next = self._apply_boundary_conditions_gpu(X_next)

    # Estimate density (GPU KDE from Phase 1)
    from mfg_pde.alg.numerical.density_estimation import gaussian_kde_gpu

    # Convert bandwidth if needed
    if isinstance(self.kde_bandwidth, str):
        from mfg_pde.alg.numerical.density_estimation import adaptive_bandwidth_selection
        X_np = self.backend.to_numpy(X_next)
        bandwidth_value = adaptive_bandwidth_selection(X_np, method=self.kde_bandwidth)
    else:
        bandwidth_value = float(self.kde_bandwidth)

    M_next = gaussian_kde_gpu(
        self.backend.to_numpy(X_next),  # Still needs numpy for KDE
        self.problem.xSpace,
        bandwidth_value,
        self.backend
    )
    M_next = self.backend.from_numpy(M_next)

    return X_next, M_next
```

#### 4. GPU Random Number Generation

**Purpose**: Generate noise on GPU (2-3x speedup)

**Current (CPU)**:
```python
dW = np.random.normal(0.0, np.sqrt(Dt), N)
```

**GPU Implementation**:
```python
# PyTorch
xp = backend.array_module
dW = xp.randn((N,), device=backend.device) * sigma * xp.sqrt(dt)

# JAX
import jax.random as jrandom
key = jrandom.PRNGKey(seed)
dW = jrandom.normal(key, shape=(N,)) * sigma * jnp.sqrt(dt)
```

**Key**: Use backend's native GPU RNG, not numpy

#### 5. GPU Boundary Conditions

**Purpose**: Apply periodic/reflecting boundaries on GPU

**Implementation**:
```python
def _apply_boundary_conditions_gpu(self, particles):
    """Apply boundary conditions (all GPU operations)."""
    xp = self.backend.array_module
    xmin = self.problem.xmin
    xmax = self.problem.xmax
    Lx = xmax - xmin

    if self.boundary_conditions.type == "periodic":
        # Modulo operation (GPU parallel)
        particles = xmin + ((particles - xmin) % Lx)

    elif self.boundary_conditions.type == "no_flux":
        # Reflection (GPU parallel with masking)
        left_violations = particles < xmin
        right_violations = particles > xmax

        # Reflect left
        particles = xp.where(
            left_violations,
            2 * xmin - particles,
            particles
        )

        # Reflect right
        particles = xp.where(
            right_violations,
            2 * xmax - particles,
            particles
        )

    return particles
```

**Key**: Use `where` (ternary operator) instead of boolean indexing for GPU

---

## Implementation Plan

### Step 1: GPU Utilities Module (Week 1)

**Create**: `mfg_pde/alg/numerical/particle_utils.py`

**Functions**:
1. `interpolate_1d_gpu(x_query, x_grid, y_grid, backend)` - Linear interpolation
2. `apply_boundary_conditions_gpu(particles, xmin, xmax, bc_type, backend)` - Boundaries
3. `sample_from_density_gpu(density, grid, N, backend)` - Initial sampling

**Testing**: Unit tests for each utility function

### Step 2: GPU Particle Evolution (Week 2)

**Modify**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`

**Add Method**: `solve_fp_system_gpu()`

**Features**:
- Full GPU pipeline
- Fallback to CPU if backend is None
- Memory-efficient trajectory storage

**Testing**: End-to-end numerical validation vs CPU version

### Step 3: Optimization (Week 3)

**Enhancements**:
1. Memory management for large N (streaming)
2. Mixed precision (float32 everywhere, float16 for KDE if beneficial)
3. Minimize numpy conversions (keep KDE internal on GPU if possible)

**Testing**: Large-scale benchmarks (N=100k, Nt=200)

### Step 4: Integration and Benchmarks (Week 4)

**Create**: `examples/benchmarks/particle_solver_gpu_benchmark.py`

**Measure**:
- Overall solver speedup (CPU vs GPU)
- Component breakdown (interpolation, RNG, KDE, etc.)
- Scaling with N and Nt

**Validation**:
- Numerical accuracy vs CPU baseline
- Mass conservation
- Convergence behavior

---

## Expected Performance

### Component Speedups (Projected)

| Component | CPU Time % | GPU Speedup | Overall Impact |
|:----------|:-----------|:------------|:---------------|
| KDE | 70% | 3.74x | 2.6x |
| Interpolation | 15% | 5-10x | 1.2-1.5x |
| Particle update | 10% | 3-5x | 1.03-1.05x |
| Random numbers | 3% | 2-3x | 1.006x |
| Boundary conditions | 2% | 2x | Negligible |
| **Transfer overhead** | **0%** | **Eliminated** | **Critical** |

**Combined Effect** (Amdahl's Law):
- Without transfer overhead elimination: 1.3-1.5x (Phase 1)
- With transfer overhead eliminated: **5-8x** (Phase 2)

### Conservative Estimate

**For N=10k particles, Nt=100 timesteps**:
- Phase 1: 1.3x overall (KDE only, with transfers)
- Phase 2: **5-7x overall** (full pipeline, no per-iteration transfers)

### Optimistic Estimate

**For N=100k particles, Nt=200 timesteps**:
- Phase 2: **8-10x overall** (larger N, better GPU utilization)

---

## Memory Management

### Storage Requirements

**For N=100k particles, Nt=200 timesteps**:
- Particle trajectories: `200 × 100k × 4 bytes = 80 MB`
- Density fields: `200 × 100 grid × 4 bytes = 80 KB`
- Temporary arrays: ~20 MB
- **Total**: ~100 MB (easily fits on MPS)

### Streaming for Very Large N

**For N > 500k or Nt > 1000**:
```python
# Don't store full trajectory
X_current = initialize_gpu(...)
for t in range(Nt):
    X_next = evolve_gpu(X_current, ...)
    M[t] = density_gpu(X_next, ...)

    # Checkpoint periodically
    if t % checkpoint_interval == 0:
        save_checkpoint(X_next)

    X_current = X_next  # Reuse memory
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| GPU memory limits | Low | Medium | Streaming, checkpointing |
| Numerical accuracy issues | Low | High | Extensive validation tests |
| Performance lower than projected | Medium | Medium | Profile, optimize iteratively |
| Backend compatibility (JAX) | Medium | Low | Focus on PyTorch first |

---

## Testing Strategy

### Unit Tests

1. **`test_interpolate_1d_gpu`**: Accuracy vs scipy.interpolate
2. **`test_boundary_conditions_gpu`**: Periodic, no-flux, Dirichlet
3. **`test_sample_from_density_gpu`**: Distribution matches target

### Integration Tests

1. **`test_particle_evolution_gpu_vs_cpu`**: End-to-end numerical match
2. **`test_mass_conservation_gpu`**: ∫m(t,x)dx ≈ 1 for all t
3. **`test_convergence_gpu`**: MFG fixed point iteration converges

### Performance Tests

1. **`benchmark_component_speedups`**: Measure each component
2. **`benchmark_overall_solver`**: End-to-end CPU vs GPU
3. **`benchmark_scaling`**: Vary N and Nt

---

## Success Criteria

### Phase 2 Completion

| Criterion | Target | Measurement |
|:----------|:-------|:------------|
| Overall speedup | 5-10x | End-to-end benchmark |
| Numerical accuracy | <1% error vs CPU | Integration tests |
| Memory efficiency | N=100k feasible | Memory profiling |
| Code quality | Type hints, docs | Review |
| Backward compatible | Optional GPU | API unchanged |

### Stretch Goals

- 10x+ speedup for N=100k
- Mixed precision support
- Streaming for N=1M+
- JAX backend support

---

## Timeline

**Week 1**: GPU utilities (interpolation, boundaries, sampling)
**Week 2**: Full GPU particle evolution integration
**Week 3**: Optimization and large-scale testing
**Week 4**: Benchmarks, validation, documentation

**Total**: 4 weeks to production-ready Phase 2

---

## Next Actions

1. ✅ Create design document (this file)
2. ⏳ Implement `particle_utils.py` with GPU interpolation
3. ⏳ Add unit tests for utilities
4. ⏳ Integrate into `fp_particle.py`
5. ⏳ Benchmark and validate

---

**Phase 2 Status**: Design complete, ready for implementation
**Expected Outcome**: 5-10x overall speedup for particle-based MFG solvers
