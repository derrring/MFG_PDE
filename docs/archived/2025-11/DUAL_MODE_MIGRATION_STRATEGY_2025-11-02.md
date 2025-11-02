# Dual-Mode FP Solver Migration Strategy

**Date**: 2025-11-02
**Author**: Claude Code
**Purpose**: Detailed strategy for migrating dual-mode capability to production

---

## Executive Summary

**Question**: Extend existing `fp_particle.py` or create new file?

**Answer**: ✅ **EXTEND EXISTING** `fp_particle.py` (strongly recommended)

**Rationale**:
- Production file is 547 lines (manageable size)
- Dual-mode changes are minimal (add mode parameter, conditional logic)
- Perfect backward compatibility (default mode = existing behavior)
- Avoids code duplication (particle evolution logic shared)
- Follows SOLID principles (Single Responsibility: particle-based FP solving)

---

## Migration Strategy Comparison

### Option A: Extend Existing fp_particle.py ✅ RECOMMENDED

**Advantages**:
- ✅ **No code duplication** - particle evolution logic shared
- ✅ **Single source of truth** - one particle FP solver
- ✅ **Backward compatible** - default mode preserves existing behavior
- ✅ **Minimal code changes** - add mode parameter, conditional output
- ✅ **Easier maintenance** - single file to update
- ✅ **Clear API** - `FPParticleSolver(mode="hybrid"|"collocation")`
- ✅ **Follows existing patterns** - like `KDENormalization` enum

**Disadvantages**:
- ⚠️ File size grows (547 → ~650 lines, still manageable)
- ⚠️ More complex `__init__` (adds 2 parameters)

**Code Changes Required**:
```python
# Add mode enum
class ParticleMode(str, Enum):
    HYBRID = "hybrid"        # Sample own particles, output to grid (existing)
    COLLOCATION = "collocation"  # Use external particles, output on particles (new)

# Extend __init__
def __init__(
    self,
    problem,
    mode: ParticleMode | str = ParticleMode.HYBRID,  # NEW
    external_particles: np.ndarray | None = None,    # NEW
    num_particles: int = 5000,  # Only for hybrid mode
    ...
):
    # Mode handling
    if isinstance(mode, str):
        mode = ParticleMode(mode)
    self.mode = mode

    if mode == ParticleMode.COLLOCATION:
        if external_particles is None:
            raise ValueError("collocation mode requires external_particles")
        self.collocation_points = external_particles
        self.num_particles = len(external_particles)
    else:  # HYBRID (default)
        self.collocation_points = None
        self.num_particles = num_particles

# Modify solve_fp_system
def solve_fp_system(self, m_initial_condition, U_solution_for_drift):
    if self.mode == ParticleMode.COLLOCATION:
        return self._solve_fp_system_collocation(...)
    else:  # HYBRID
        return self._solve_fp_system_cpu(...) or self._solve_fp_system_gpu(...)

# Add new method
def _solve_fp_system_collocation(self, m_initial_condition, U_solution_for_drift):
    # Use external particles, output on same particles (no KDE)
    ...
```

**Estimated Changes**: ~100-150 lines added, 547 → ~650-700 lines

---

### Option B: Create New fp_particle_dual_mode.py ❌ NOT RECOMMENDED

**Advantages**:
- ✅ Existing code unchanged (lower risk for existing users)
- ✅ Clear separation (old vs new)

**Disadvantages**:
- ❌ **Code duplication** - particle evolution logic duplicated
- ❌ **Two sources of truth** - `FPParticleSolver` vs `FPParticleDualModeSolver`
- ❌ **Maintenance burden** - bug fixes need to be applied to both files
- ❌ **API confusion** - users must choose between two similar classes
- ❌ **No backward compatibility benefit** - new class anyway
- ❌ **Against DRY principle** - Don't Repeat Yourself

**Code Organization**:
```
fp_particle.py (547 lines) - Original solver
fp_particle_dual_mode.py (650 lines) - New solver with dual-mode

Problem: 80% code overlap (particle evolution logic identical)
```

**Verdict**: Creates more problems than it solves

---

## Detailed Migration Plan (Option A - Extend Existing)

### Step 1: Add ParticleMode Enum

**Location**: After `KDENormalization` enum (line 32)

```python
class ParticleMode(str, Enum):
    """Particle solver operating mode."""

    HYBRID = "hybrid"  # Sample own particles, output to grid via KDE (default)
    COLLOCATION = "collocation"  # Use external particles, output on particles (no KDE)
```

**Rationale**: Follows existing pattern (`KDENormalization` enum on line 26)

---

### Step 2: Extend __init__ Parameters

**Location**: Line 35 (`def __init__`)

**Add Parameters**:
```python
def __init__(
    self,
    problem: MFGProblem,
    # NEW: Mode selection
    mode: ParticleMode | str = ParticleMode.HYBRID,
    external_particles: np.ndarray | None = None,
    # Existing parameters
    num_particles: int = 5000,  # Only used in hybrid mode
    kde_bandwidth: Any = "scott",
    kde_normalization: KDENormalization | str = KDENormalization.ALL,
    boundary_conditions: BoundaryConditions | None = None,
    backend: str | None = None,
    # Deprecated parameters
    normalize_kde_output: bool | None = None,
    normalize_only_initial: bool | None = None,
) -> None:
```

**Add Validation** (after line 75):
```python
# Convert string to enum if needed
if isinstance(mode, str):
    mode = ParticleMode(mode)
self.mode = mode

# Validate mode-specific parameters
if mode == ParticleMode.COLLOCATION:
    if external_particles is None:
        raise ValueError(
            "Collocation mode requires external_particles. "
            "Pass the collocation points used by your HJB solver."
        )
    self.collocation_points = external_particles
    self.num_particles = len(external_particles)
    self.fp_method_name = "Particle-Collocation"
else:  # HYBRID mode (default)
    self.collocation_points = None
    self.num_particles = num_particles
    self.fp_method_name = "Particle"
```

**Backward Compatibility**: ✅ PERFECT
- Default `mode=ParticleMode.HYBRID` preserves existing behavior
- All existing code continues to work without changes

---

### Step 3: Modify solve_fp_system Dispatcher

**Location**: Line 262 (`def solve_fp_system`)

**Current Code** (lines 295-300):
```python
# Execute using selected strategy's pipeline
if self.current_strategy.name == "cpu":
    return self._solve_fp_system_cpu(m_initial_condition, U_solution_for_drift)
else:
    # GPU or Hybrid strategy (both use GPU pipeline)
    return self._solve_fp_system_gpu(m_initial_condition, U_solution_for_drift)
```

**New Code**:
```python
# Execute using mode-appropriate method
if self.mode == ParticleMode.COLLOCATION:
    # Collocation mode: particles → particles (no KDE)
    return self._solve_fp_system_collocation(m_initial_condition, U_solution_for_drift)
else:
    # Hybrid mode: particles → grid (existing behavior)
    if self.current_strategy.name == "cpu":
        return self._solve_fp_system_cpu(m_initial_condition, U_solution_for_drift)
    else:
        return self._solve_fp_system_gpu(m_initial_condition, U_solution_for_drift)
```

---

### Step 4: Add _solve_fp_system_collocation Method

**Location**: After `_solve_fp_system_gpu` method (around line 547)

**Implementation**:
```python
def _solve_fp_system_collocation(
    self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray
) -> np.ndarray:
    """
    Solve FP system in collocation mode.

    Uses external particles (from HJB solver) and outputs density on same particles.
    No KDE interpolation to grid - true meshfree representation.

    Args:
        m_initial_condition: Initial density (on collocation points)
        U_solution_for_drift: Value function for drift computation

    Returns:
        M_solution: Density evolution on collocation points (Nt, N_particles)
    """
    Nt = self.problem.Nt + 1
    N_particles = len(self.collocation_points)

    # Initialize particles at collocation points
    # In collocation mode, particles ARE the collocation points (Eulerian)
    particles_current = self.collocation_points.copy()

    # Storage for density evolution on particles
    M_solution = np.zeros((Nt, N_particles))
    M_solution[0, :] = m_initial_condition  # Initial density on particles

    # Time evolution (same particle dynamics as hybrid mode)
    dt = self.problem.T / self.problem.Nt
    sigma = getattr(self.problem, "sigma", 0.1)

    for t_idx in range(Nt - 1):
        # Compute drift from value function gradient
        # NOTE: In collocation mode, U_solution_for_drift is on collocation points
        U_current = U_solution_for_drift[t_idx, :]

        # Interpolate U to particle locations for gradient computation
        # (In collocation mode, particles = collocation points, so direct lookup)
        drift = -self._compute_particle_drift(particles_current, U_current)

        # Diffusion (Brownian motion)
        diffusion = sigma * np.sqrt(dt) * np.random.randn(N_particles, self.problem.dimension)

        # Update particle positions
        particles_current = particles_current + drift * dt + diffusion

        # Handle boundary conditions (periodic, reflecting, etc.)
        particles_current = self._apply_boundary_conditions(particles_current)

        # CRITICAL DIFFERENCE: No KDE!
        # Density is carried by particles directly (Lagrangian mass)
        # Each particle carries mass = initial_density[i] / N_particles
        M_solution[t_idx + 1, :] = M_solution[0, :]  # Constant mass per particle

    return M_solution  # Shape: (Nt, N_particles) - particle output

def _compute_particle_drift(self, particles: np.ndarray, U_values: np.ndarray) -> np.ndarray:
    """
    Compute drift term for particles from value function.

    In collocation mode: U_values are on collocation points, interpolate gradient.

    Args:
        particles: Particle positions (N, d)
        U_values: Value function on collocation points (N,)

    Returns:
        drift: Drift vectors for each particle (N, d)
    """
    # Use GFDM-style gradient approximation
    # (Since we're in collocation mode, can use neighborhood structure)

    # Simplified version: finite difference approximation
    # Full version would use GFDM weights from HJB solver

    N, d = particles.shape
    drift = np.zeros((N, d))

    for i in range(N):
        # Find neighbors
        distances = np.linalg.norm(particles - particles[i], axis=1)
        neighbors = np.argsort(distances)[1:d+2]  # Nearest neighbors

        # Finite difference gradient estimate
        for dim in range(d):
            # Simple gradient in each dimension
            if len(neighbors) > 0:
                h = distances[neighbors[0]]
                if h > 1e-10:
                    drift[i, dim] = (U_values[neighbors[0]] - U_values[i]) / h

    return drift
```

**Size Estimate**: ~100-150 lines

**Key Differences from Hybrid Mode**:
1. **No particle sampling** - use external collocation points
2. **No KDE** - density output directly on particles
3. **Lagrangian mass** - each particle carries constant mass
4. **Output shape** - `(Nt, N_particles)` instead of `(Nt, Nx)`

---

### Step 5: Update Exports and Documentation

**File**: `mfg_pde/alg/numerical/fp_solvers/__init__.py`

**Add Export**:
```python
from .fp_particle import FPParticleSolver, KDENormalization, ParticleMode  # Add ParticleMode

__all__ = [
    # ... existing exports ...
    "ParticleMode",  # NEW
]
```

**Update Docstrings**:
```python
class FPParticleSolver(BaseFPSolver):
    """
    Particle-based Fokker-Planck solver with dual-mode support.

    Modes:
    - HYBRID (default): Sample own particles, output to grid via KDE
      Use with grid-based HJB solvers (FDM, WENO, Semi-Lagrangian)

    - COLLOCATION: Use external particles, output on particles
      Use with particle-collocation HJB solvers (GFDM)
      Enables true Eulerian meshfree MFG

    Examples:
        # Hybrid mode (default, existing behavior)
        >>> solver = FPParticleSolver(problem, num_particles=5000)
        >>> M = solver.solve_fp_system(m0, U)  # Shape: (Nt, Nx)

        # Collocation mode (new capability)
        >>> points = domain.sample_uniform(5000)
        >>> solver = FPParticleSolver(
        ...     problem,
        ...     mode="collocation",
        ...     external_particles=points
        ... )
        >>> M = solver.solve_fp_system(m0, U)  # Shape: (Nt, 5000)
    """
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_fp_particle.py`

**Add Tests**:
```python
class TestParticleMode:
    """Test dual-mode functionality."""

    def test_hybrid_mode_default(self):
        """Test that hybrid mode is default (backward compatibility)."""
        solver = FPParticleSolver(problem, num_particles=1000)
        assert solver.mode == ParticleMode.HYBRID

    def test_collocation_mode_requires_external_particles(self):
        """Test that collocation mode validates external_particles."""
        with pytest.raises(ValueError, match="requires external_particles"):
            FPParticleSolver(problem, mode="collocation")

    def test_collocation_mode_output_shape(self):
        """Test that collocation mode outputs on particles."""
        points = np.random.uniform(0, 1, (100, 2))
        solver = FPParticleSolver(
            problem,
            mode="collocation",
            external_particles=points
        )

        m0 = np.ones(100) / 100
        U = np.zeros((problem.Nt + 1, 100))
        M = solver.solve_fp_system(m0, U)

        assert M.shape == (problem.Nt + 1, 100)  # Particle output

    def test_hybrid_mode_output_shape(self):
        """Test that hybrid mode outputs on grid (existing behavior)."""
        solver = FPParticleSolver(problem, num_particles=1000)

        m0 = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U = np.zeros((problem.Nt + 1, problem.Nx + 1))
        M = solver.solve_fp_system(m0, U)

        assert M.shape == (problem.Nt + 1, problem.Nx + 1)  # Grid output
```

**Regression Tests**:
- ✅ All existing tests should pass (hybrid mode is default)
- ✅ Add new tests for collocation mode

---

### Integration Tests

**File**: `tests/integration/test_particle_collocation_mfg.py` (new)

**Test Full MFG Workflow**:
```python
def test_particle_collocation_mfg_workflow():
    """Test complete MFG solve using particle collocation."""
    from mfg_pde.geometry.implicit import Hyperrectangle
    from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
    from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver

    # Sample collocation points
    domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
    points = domain.sample_uniform(1000, seed=42)

    # Create HJB solver (GFDM on particles)
    hjb_solver = HJBGFDMSolver(
        problem,
        collocation_points=points,
        delta=0.1,
    )

    # Create FP solver (collocation mode on same particles)
    fp_solver = FPParticleSolver(
        problem,
        mode="collocation",
        external_particles=points,
    )

    # Picard iteration
    M_current = np.ones(len(points)) / len(points)
    for _ in range(5):
        U = hjb_solver.solve_hjb_system(M_current, ...)
        M_new = fp_solver.solve_fp_system(M_current[0], U)
        M_current = M_new

    # Verify shapes match
    assert U.shape == M_new.shape
    assert U.shape == (problem.Nt + 1, len(points))
```

---

## Migration Timeline

### Week 1: Design and Planning
- [x] Analyze existing code structure
- [x] Design mode parameter API
- [x] Plan backward compatibility strategy
- [ ] Review with stakeholders

### Week 2: Implementation
- [ ] Add `ParticleMode` enum
- [ ] Extend `__init__` parameters
- [ ] Implement `_solve_fp_system_collocation`
- [ ] Add `_compute_particle_drift` helper
- [ ] Update dispatcher logic

### Week 3: Testing
- [ ] Unit tests for mode selection
- [ ] Unit tests for collocation mode
- [ ] Integration tests (MFG workflow)
- [ ] Regression tests (hybrid mode)
- [ ] Performance benchmarks

### Week 4: Documentation and Release
- [ ] Update docstrings
- [ ] Create example (particle collocation demo)
- [ ] Update user guide
- [ ] Create PR and review
- [ ] Merge to main

**Total Effort**: 3-4 weeks (part-time, ~10-15 hours/week)

---

## Backward Compatibility Analysis

### API Changes

**Existing Code** (no changes needed):
```python
# All existing usage continues to work
solver = FPParticleSolver(problem, num_particles=5000)
M = solver.solve_fp_system(m0, U)
```

**New Usage** (opt-in):
```python
# Users must explicitly request collocation mode
solver = FPParticleSolver(
    problem,
    mode="collocation",
    external_particles=points,
)
M = solver.solve_fp_system(m0, U)
```

**Compatibility**: ✅ 100% backward compatible (default mode = existing behavior)

---

### Output Shape Compatibility

**Hybrid Mode** (existing):
- Input: `m0` shape `(Nx,)`, `U` shape `(Nt, Nx)`
- Output: `M` shape `(Nt, Nx)` - grid-based

**Collocation Mode** (new):
- Input: `m0` shape `(N_particles,)`, `U` shape `(Nt, N_particles)`
- Output: `M` shape `(Nt, N_particles)` - particle-based

**User Responsibility**: Choose appropriate mode for HJB solver type

---

## Risk Assessment

### Low Risk Migration

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Breaking changes | Very Low | High | Default mode = existing behavior |
| API confusion | Low | Medium | Clear docstrings, mode naming |
| Test failures | Low | Medium | Comprehensive test suite |
| Performance regression | Very Low | Medium | Mode-specific methods, no overhead |

**Overall Risk**: ✅ **MINIMAL** (well-isolated changes, perfect backward compatibility)

---

## Alternative Considered: Separate Class

**Option**: Create `FPParticleCollocationSolver` (separate class)

**Advantages**:
- Clear separation
- No risk to existing code

**Disadvantages**:
- ❌ Code duplication (~400 lines of particle evolution logic)
- ❌ Maintenance burden (bug fixes in two places)
- ❌ API confusion (two similar classes)
- ❌ Violates DRY principle

**Verdict**: ❌ Not recommended (creates more problems than it solves)

---

## Recommendation

### ✅ STRONGLY RECOMMEND: Extend Existing fp_particle.py

**Rationale**:
1. **Single Responsibility**: Particle-based FP solving (both modes use particles)
2. **Code Reuse**: Particle evolution logic shared between modes
3. **Backward Compatibility**: Default mode preserves existing behavior
4. **Clean API**: Single class with mode parameter (like `KDENormalization`)
5. **Maintainability**: Single source of truth for particle FP
6. **Follows Patterns**: Consistent with existing enum-based configuration

**Implementation Path**:
1. Add `ParticleMode` enum (10 lines)
2. Extend `__init__` validation (20 lines)
3. Add `_solve_fp_system_collocation` method (100 lines)
4. Update dispatcher (5 lines)
5. Add tests (50 lines)

**Total Addition**: ~200 lines (547 → ~750 lines, still manageable)

**Timeline**: 3-4 weeks (design, implementation, testing, documentation)

---

## Next Steps

### Immediate
1. Review this migration strategy with team/stakeholders
2. Confirm decision: Extend existing vs separate file
3. Create GitHub issue with migration plan

### Week 1
4. Create feature branch: `feature/dual-mode-fp-solver`
5. Implement `ParticleMode` enum
6. Extend `__init__` parameters with validation

### Week 2
7. Implement `_solve_fp_system_collocation` method
8. Add particle drift computation helper
9. Update dispatcher logic

### Week 3
10. Comprehensive testing (unit, integration, regression)
11. Create example demonstration
12. Update documentation

### Week 4
13. Create PR with all changes
14. Code review
15. Merge to main
16. Update research repo to use production version

---

## Conclusion

**Migration Strategy**: ✅ **Extend existing `fp_particle.py`**

**Key Benefits**:
- Minimal code changes (~200 lines added)
- Perfect backward compatibility
- No code duplication
- Clean, extensible API
- Follows existing patterns

**Risk Level**: Very Low (well-isolated changes, default mode unchanged)

**Effort Estimate**: 3-4 weeks (design + implementation + testing + docs)

**Value Proposition**: Enables true Eulerian meshfree MFG workflows not possible today

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Status**: Migration strategy ready ✅
