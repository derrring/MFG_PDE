# Dual-Mode FP Solver: Grid vs General Geometry Support

**Date**: 2025-11-02
**Author**: Claude Code
**Purpose**: Technical discussion on integrating dual-mode FP solver capability into production MFG_PDE

---

## Executive Summary

**Current Situation**:
- Production MFG_PDE has `FPParticleSolver` that outputs to **grid only** (hybrid mode)
- Research repo has `DualModeFPParticleSolver` supporting both **grid** and **particle** outputs

**Key Innovation**:
The research implementation supports **collocation mode** where particles remain particles throughout, enabling:
- Pure particle-based MFG (no grid needed)
- Particle-collocation methods with GFDM
- High-dimensional problems (d ≥ 3) without curse of dimensionality

**Recommendation**: ✅ Migrate dual-mode capability to production as enhancement to `FPParticleSolver`.

---

## Problem Statement

### Current Production Limitation

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py` (385 lines)

```python
class FPParticleSolver(BaseFPSolver):
    """Particle-based FP solver - GRID OUTPUT ONLY"""

    def solve_fp_system(...) -> np.ndarray:
        # 1. Sample particles
        particles = self._sample_particles(M_initial)

        # 2. Advect particles
        for t in range(Nt):
            particles = self._advect(particles, U_drift[t])

        # 3. Always convert to grid via KDE
        M_grid[t] = self._particles_to_grid_kde(particles)

        return M_grid  # (Nt, Nx) - ALWAYS GRID
```

**Limitation**: Cannot output particle densities for use with `HJBGFDMSolver` (GFDM).

### Research Solution

**File**: `algorithms/particle_collocation/fp_particle_dual_mode.py` (563 lines)

```python
class DualModeFPParticleSolver:
    """
    Enhanced FP solver with two modes:

    HYBRID mode: Particles → Grid (for grid-based HJB)
    COLLOCATION mode: Particles → Particles (for GFDM HJB)
    """

    def solve_fp_system(...) -> np.ndarray:
        if self.mode == FPSolverMode.HYBRID:
            # Original behavior: KDE to grid
            return M_grid  # (Nt, Nx)
        else:  # COLLOCATION
            # NEW: Keep particles as particles
            return M_particles  # (Nt, N_particles)
```

---

## Two Operating Modes

### Mode 1: Hybrid (Grid-Based MFG)

**Geometry**: Regular grid
**Use With**: HJB-FDM, HJB-WENO, HJB-Semi-Lagrangian

**Workflow**:
```
Grid → Sample particles → Advect → KDE → Grid
                     ↓
              Grid-based HJB solver
```

**Example**:
```python
problem = ExampleMFGProblem(Nx=100, Nt=50)

# FP: Particles → Grid
solver_fp = FPParticleSolver(problem, mode="hybrid", num_particles=5000)

# HJB: Grid → Grid
solver_hjb = HJBFDMSolver(problem)

# Coupled solve
M_grid = solver_fp.solve_fp_system(U_hjb, M0)  # (Nt, Nx)
U_grid = solver_hjb.solve_hjb_system(M_grid, UT, U_prev)  # (Nt, Nx)
```

**Characteristics**:
- ✅ Works with all existing grid-based solvers
- ✅ Natural for 1D, 2D problems
- ✅ Easy visualization
- ❌ Curse of dimensionality (d ≥ 3)

### Mode 2: Collocation (Particle-Based MFG)

**Geometry**: Particle collocation points
**Use With**: HJB-GFDM

**Workflow**:
```
Particles → Advect particles → Particles
                          ↓
                   GFDM HJB solver
```

**Example**:
```python
# Sample particles once (shared between FP and HJB)
domain = ImplicitDomain(...)
particles = domain.sample_uniform(500)  # (N, d)

# FP: Particles → Particles
solver_fp = FPParticleSolver(
    problem,
    mode="collocation",
    external_particles=particles
)

# HJB: Particles → Particles
solver_hjb = HJBGFDMSolver(
    problem,
    collocation_points=particles
)

# Coupled solve (both on same particles)
M_particles = solver_fp.solve_fp_system(U_hjb, M0)  # (Nt, N)
U_particles = solver_hjb.solve_hjb_system(M_particles, UT, U_prev)  # (Nt, N)
```

**Characteristics**:
- ✅ No grid needed (pure particle method)
- ✅ Scales to high dimensions (d ≥ 3, d ≥ 5)
- ✅ Natural for complex geometries (obstacles, mazes)
- ✅ Consistent geometry (particles throughout)
- ❌ Requires interpolation for visualization

---

## Key Differences

### Input/Output Geometry

| Mode | Input | Particles | Output | Compatible HJB |
|:-----|:------|:----------|:-------|:---------------|
| Hybrid | Grid | Internal (sampled) | Grid | FDM, WENO, Semi-Lagrangian |
| Collocation | Particles | External (shared) | Particles | GFDM |

### Particle Management

**Hybrid Mode**:
```python
# FP creates its own particles
particles = fp_solver._sample_particles(M_initial)  # Independent
hjb_solver  # Uses grid (no particles)
```

**Collocation Mode**:
```python
# Both solvers share same particles
particles = domain.sample_uniform(N)  # Shared
fp_solver.external_particles = particles
hjb_solver.collocation_points = particles
```

---

## Implementation Details

### Research Code Analysis

**File**: `fp_particle_dual_mode.py` (563 lines - reasonable size!)

**Key Enhancements Over Production**:

1. **Mode Selection**:
```python
class FPSolverMode(Enum):
    HYBRID = "hybrid"      # Grid output (existing)
    COLLOCATION = "collocation"  # Particle output (NEW)
```

2. **External Particle Support**:
```python
def __init__(self, ..., external_particles=None):
    if mode == COLLOCATION:
        self.particles = external_particles  # Use provided
    else:  # HYBRID
        self.particles = self._sample_own()  # Create internally
```

3. **Conditional Output**:
```python
def solve_fp_system(self, ...):
    # Advection (same for both modes)
    particles = self._advect_particles(...)

    if self.mode == FPSolverMode.HYBRID:
        # Convert to grid via KDE
        return self._particles_to_grid_kde(particles)
    else:  # COLLOCATION
        # Keep as particle densities
        return self._compute_particle_densities(particles)
```

4. **GFDM-Based Derivatives** (for collocation mode):
```python
def _compute_gradient_gfdm(self, values, particles):
    """
    Compute gradient at particles using GFDM.

    Same methodology as HJB-GFDM but without QP constraints.
    FP is forward diffusion (naturally stable), no monotonicity needed.
    """
    # Build neighborhoods
    neighborhoods = self._build_neighborhoods(particles)

    # Taylor expansion (least-squares)
    gradients = self._taylor_approximation(values, neighborhoods)

    return gradients
```

---

## Migration Strategy

### Option 1: Extend Existing `FPParticleSolver` ✅ (Recommended)

**Approach**: Add mode parameter to production `FPParticleSolver`

**Advantages**:
- ✅ Preserves existing API (backward compatible)
- ✅ Minimal code duplication
- ✅ Consistent with MFG_PDE patterns

**Changes**:
```python
# mfg_pde/alg/numerical/fp_solvers/fp_particle.py

from enum import Enum

class FPSolverMode(Enum):
    HYBRID = "hybrid"
    COLLOCATION = "collocation"

class FPParticleSolver(BaseFPSolver):
    def __init__(
        self,
        problem,
        num_particles=5000,
        mode: FPSolverMode | str = FPSolverMode.HYBRID,  # NEW
        external_particles=None,  # NEW
        **kwargs
    ):
        # Mode-specific initialization
        if mode == FPSolverMode.COLLOCATION:
            self._init_collocation(external_particles)
        else:
            self._init_hybrid(num_particles)

    def solve_fp_system(self, ...):
        # Shared: advection logic
        particles = self._advect_particles(...)

        # Mode-specific output
        if self.mode == FPSolverMode.HYBRID:
            return self._output_to_grid(particles)
        else:  # COLLOCATION
            return self._output_to_particles(particles)
```

**Estimated Effort**: 2-3 days

### Option 2: Create Separate `FPCollocationSolver` ❌ (Not Recommended)

**Approach**: New solver class for collocation mode

**Advantages**:
- ✅ Cleaner separation

**Disadvantages**:
- ❌ Code duplication (advection logic is identical)
- ❌ Two classes to maintain
- ❌ User confusion (when to use which?)

**Verdict**: Not recommended - duplication outweighs benefits

---

## Backward Compatibility

### Existing Code (Unchanged)

```python
# All existing code continues to work
solver = FPParticleSolver(problem, num_particles=5000)
M = solver.solve_fp_system(U, M0)  # Still returns grid
```

### New Collocation Usage

```python
# New capability for particle-collocation
particles = domain.sample_uniform(500)
solver = FPParticleSolver(
    problem,
    mode="collocation",
    external_particles=particles
)
M = solver.solve_fp_system(U, M0)  # Returns particles
```

**Backward Compatibility**: 100% - default mode is `HYBRID`

---

## Testing Requirements

### Unit Tests

1. **Mode Selection**:
```python
def test_hybrid_mode_default():
    solver = FPParticleSolver(problem)
    assert solver.mode == FPSolverMode.HYBRID

def test_collocation_mode_explicit():
    solver = FPParticleSolver(problem, mode="collocation",
                              external_particles=particles)
    assert solver.mode == FPSolverMode.COLLOCATION
```

2. **Output Shape**:
```python
def test_hybrid_output_grid():
    solver = FPParticleSolver(problem, mode="hybrid")
    M = solver.solve_fp_system(U, M0)
    assert M.shape == (Nt, Nx)  # Grid

def test_collocation_output_particles():
    particles = np.random.rand(500, 2)
    solver = FPParticleSolver(problem, mode="collocation",
                              external_particles=particles)
    M = solver.solve_fp_system(U, M0)
    assert M.shape == (Nt, 500)  # Particles
```

3. **Particle Sharing**:
```python
def test_external_particles_preserved():
    particles = domain.sample_uniform(500)
    solver_fp = FPParticleSolver(problem, mode="collocation",
                                  external_particles=particles)
    solver_hjb = HJBGFDMSolver(problem, collocation_points=particles)

    # Verify same particles used
    assert np.array_equal(solver_fp.particles, particles)
    assert np.array_equal(solver_hjb.collocation_points, particles)
```

### Integration Tests

1. **Hybrid MFG** (existing functionality):
```python
def test_hybrid_mfg_picard_iteration():
    problem = ExampleMFGProblem(Nx=100)
    solver_fp = FPParticleSolver(problem, mode="hybrid")
    solver_hjb = HJBFDMSolver(problem)

    # Picard iteration
    for iteration in range(10):
        M = solver_fp.solve_fp_system(U, M0)
        U = solver_hjb.solve_hjb_system(M, UT, U)

    # Check convergence
    assert np.max(np.abs(M - M_prev)) < 1e-4
```

2. **Collocation MFG** (new functionality):
```python
def test_collocation_mfg_picard_iteration():
    particles = domain.sample_uniform(500)
    problem = GridBasedMFGProblem(...)  # Can still use this

    solver_fp = FPParticleSolver(problem, mode="collocation",
                                  external_particles=particles)
    solver_hjb = HJBGFDMSolver(problem, collocation_points=particles)

    # Picard iteration (both on particles)
    for iteration in range(10):
        M = solver_fp.solve_fp_system(U, M0)  # (Nt, 500)
        U = solver_hjb.solve_hjb_system(M, UT, U)  # (Nt, 500)

    # Check convergence
    assert np.max(np.abs(M - M_prev)) < 1e-4
```

---

## Documentation Updates

### User Guide

**File**: `docs/user/guides/solvers.md`

Add section:
```markdown
### FP Particle Solver Modes

The FP particle solver supports two operating modes:

#### Hybrid Mode (Grid Output)

Use for traditional grid-based MFG:

```python
solver = FPParticleSolver(problem, mode="hybrid", num_particles=5000)
M_grid = solver.solve_fp_system(U, M0)  # (Nt, Nx)
```

Compatible with: HJB-FDM, HJB-WENO, HJB-Semi-Lagrangian

#### Collocation Mode (Particle Output)

Use for particle-collocation MFG:

```python
particles = domain.sample_uniform(500)
solver = FPParticleSolver(problem, mode="collocation",
                          external_particles=particles)
M_particles = solver.solve_fp_system(U, M0)  # (Nt, 500)
```

Compatible with: HJB-GFDM
```

### Examples

**File**: `examples/advanced/particle_collocation_mfg_demo.py`

Create comprehensive example showing collocation mode usage.

---

## Performance Considerations

### Hybrid Mode

**Computational Cost**:
- Particle advection: O(N_particles × Nt)
- KDE conversion: O(N_particles × Nx) per time step

**Typical**: N_particles = 5000, Nx = 100 → Fast

### Collocation Mode

**Computational Cost**:
- Particle advection: O(N_particles × Nt)
- GFDM derivatives: O(N_particles × k) where k = neighbors
- No KDE conversion needed

**Typical**: N_particles = 500, k = 30 → Similar or faster

**Key Advantage**: No KDE overhead, fewer particles needed (shared with HJB)

---

## Dependency Analysis

### New Dependencies

**None** - All required functionality already in MFG_PDE:
- ✅ Particle advection (existing)
- ✅ GFDM neighborhoods (`HJBGFDMSolver`)
- ✅ Taylor expansion (`HJBGFDMSolver`)
- ✅ KDE (`scipy` - optional, already used)

### Code Reuse

**From Research Implementation**:
- `FPSolverMode` enum
- Collocation mode initialization
- Particle density computation
- GFDM derivative methods (simplified from HJB-GFDM)

**From Production**:
- All existing hybrid mode logic
- Base class infrastructure
- Backend support (NumPy, JAX, PyTorch)

---

## Migration Roadmap

### Week 1: Investigation & Design ✅ (Current)
- [x] Analyze research implementation
- [x] Create discussion document
- [ ] Review with stakeholders

### Week 2: Implementation
- [ ] Add `FPSolverMode` enum
- [ ] Extend `FPParticleSolver.__init__` with mode parameter
- [ ] Implement collocation mode initialization
- [ ] Add particle output method
- [ ] Integrate GFDM derivatives

### Week 3: Testing
- [ ] Unit tests (mode selection, output shapes)
- [ ] Integration tests (collocation MFG)
- [ ] Regression tests (hybrid mode unchanged)
- [ ] Performance benchmarks

### Week 4: Documentation & Examples
- [ ] Update API documentation
- [ ] Create user guide section
- [ ] Write particle-collocation example
- [ ] Update quickstart if needed

---

## Open Questions

### 1. Should we create a dedicated `ParticleBasedMFGProblem` class?

**Current**: `GridBasedMFGProblem` works but assumes grid

**Options**:
- A) Keep using `GridBasedMFGProblem` (works, but name misleading)
- B) Create `ParticleBasedMFGProblem` (cleaner, more work)
- C) Create `GeneralMFGProblem` base class (most general, most refactoring)

**Recommendation**: Start with (A), consider (B) later if demand exists

### 2. How to handle initial density in collocation mode?

**Options**:
- A) Require M0 as particle array directly
- B) Sample from grid-based M0 if provided
- C) Support both (check input type)

**Recommendation**: (C) - flexible, user-friendly

### 3. Visualization for particle densities?

**Options**:
- A) Leave to user (scatter plots)
- B) Provide helper: `particles_to_grid()` for visualization
- C) Integrate with `create_visualization_manager()`

**Recommendation**: (B) + (C) - helper function + integration

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| API breaks existing code | Low | High | Default to hybrid mode |
| Performance regression | Low | Medium | Benchmark both modes |
| GFDM integration issues | Medium | Medium | Reuse existing HJB-GFDM code |
| Test coverage gaps | Medium | High | Comprehensive test suite |

### Mitigation Strategies

1. **Backward Compatibility**: Default mode="hybrid" preserves all existing behavior
2. **Code Reuse**: Leverage existing GFDM infrastructure from `HJBGFDMSolver`
3. **Incremental Migration**: Add collocation mode gradually, hybrid mode unchanged
4. **Comprehensive Testing**: Unit + integration + regression tests

---

## Recommendations

### High Priority: Migrate Dual-Mode Capability ✅

**Why**:
1. ✅ **Enables particle-collocation MFG** (key feature for high-d problems)
2. ✅ **Modest complexity** (563 lines, well-structured)
3. ✅ **100% backward compatible** (default mode unchanged)
4. ✅ **Complements existing GFDM** (uses same infrastructure)
5. ✅ **No new dependencies** (everything already in MFG_PDE)

**Estimated Effort**: 2-3 weeks (implementation + testing + docs)

**Risk Level**: Low (isolated change, backward compatible)

### Implementation Approach

**Recommended**: Extend existing `FPParticleSolver` with mode parameter

**Steps**:
1. Add `FPSolverMode` enum
2. Add `mode` and `external_particles` parameters
3. Conditional logic for output method
4. Comprehensive tests
5. Documentation and examples

---

## Conclusion

**Dual-mode FP solver is a valuable addition** that:
- Enables pure particle-based MFG methods
- Scales to high dimensions (d ≥ 3)
- Maintains full backward compatibility
- Requires minimal new code (reuses existing infrastructure)

**Key Innovation**: Geometry flexibility - output matches input (grid→grid or particles→particles)

**Recommendation**: ✅ **Proceed with migration** - High value, low risk, good architectural fit.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Status**: Ready for review and implementation
