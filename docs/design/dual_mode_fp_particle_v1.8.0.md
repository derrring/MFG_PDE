# Dual-Mode FP-Particle Solver Design (v1.8.0)

**Status**: Prototype validated in mfg-research ✅
**Target Release**: MFG_PDE v1.8.0
**Implementation**: ~95 lines of new code
**Backward Compatibility**: 100% ✅

---

## Summary

Enhance `FPParticleSolver` to support **two operating modes**:

1. **Hybrid mode** (existing): Sample own particles → KDE → Grid output
2. **Collocation mode** (NEW): Use external particles → Particle output

This enables particle-collocation MFG methods for high-dimensional problems (d ≥ 3).

---

## Motivation

### Current Limitation

`FPParticleSolver` uses particles internally but always converts to grid via KDE:

```python
# Current (v1.7.3)
solver = FPParticleSolver(problem, num_particles=5000)
M = solver.solve_fp_system(m_initial, U_drift)
# Returns: (Nt, Nx) on grid (always!)
```

**Problem**: Cannot be used for particle-collocation where:
- HJB-GFDM operates on particles (not grid)
- FP must use **SAME particles** as HJB-GFDM
- No grid exists in particle-native problems

### Proposed Solution

Add `mode` parameter:

```python
# Hybrid mode (existing, default)
solver = FPParticleSolver(problem, num_particles=5000, mode="hybrid")
M = solver.solve_fp_system(...)  # Returns (Nt, Nx) on grid

# Collocation mode (NEW)
particles = domain.sample(500)
solver = FPParticleSolver(
    problem,
    mode="collocation",
    external_particles=particles
)
M = solver.solve_fp_system(...)  # Returns (Nt, N=500) on particles!
```

---

## Design

### API Changes

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`

```python
class FPParticleSolver(BaseFPSolver):
    def __init__(
        self,
        problem: MFGProblem,
        num_particles: int = 5000,
        mode: str = "hybrid",  # NEW parameter
        external_particles: np.ndarray | None = None,  # NEW parameter
        ...
    ):
        """
        Initialize FP-Particle solver.

        Args:
            mode: "hybrid" (default) or "collocation"
            external_particles: Required for collocation mode
        """
        self.mode = mode

        if mode == "hybrid":
            # Existing behavior
            self.num_particles = num_particles

        elif mode == "collocation":
            # NEW behavior
            if external_particles is None:
                raise ValueError("Collocation mode requires external_particles")
            self.external_particles = external_particles
            self.num_particles = len(external_particles)
```

### Implementation

**Dispatch solve based on mode**:

```python
def solve_fp_system(self, m_initial, U_drift):
    if self.mode == "hybrid":
        return self._solve_hybrid_mode(m_initial, U_drift)
    elif self.mode == "collocation":
        return self._solve_collocation_mode(m_initial, U_drift)
```

**Hybrid mode**: Existing implementation (unchanged)

```python
def _solve_hybrid_mode(self, m_initial, U_drift):
    """Sample particles → Advect → KDE to grid."""
    # Lines 297-380 in current v1.7.3 (UNCHANGED)
    ...
```

**Collocation mode**: NEW implementation

```python
def _solve_collocation_mode(self, m_initial, U_drift):
    """Use external particles → Advect → Stay on particles."""
    N = len(self.external_particles)
    Nt = U_drift.shape[0]

    M = np.zeros((Nt, N))
    M[0, :] = m_initial  # On particles

    particles = self.external_particles.copy()

    for t in range(Nt - 1):
        # Gradient at particles (no interpolation!)
        gradient_U = self._compute_gradient_at_particles(
            U_drift[t, :], particles
        )

        # Advect particles
        drift = -gradient_U
        particles_new = particles + drift * dt + noise

        # Density follows particles (Lagrangian)
        M[t+1, :] = M[t, :]

        particles = particles_new

    return M  # (Nt, N) on particles, NO KDE!
```

---

## Code Changes

**Lines of code**:
- `__init__` modifications: ~10 lines
- `_solve_collocation_mode()`: ~50 lines
- `_compute_gradient_at_particles()`: ~30 lines
- Dispatch logic: ~5 lines
- **Total**: ~95 lines

**Existing code**: UNCHANGED (hybrid mode uses current implementation)

---

## Backward Compatibility

**100% backward compatible**:

```python
# All existing code works unchanged
solver = FPParticleSolver(problem, num_particles=5000)
# mode defaults to "hybrid"
M = solver.solve_fp_system(...)
# Works exactly as before!
```

**New capability (opt-in)**:

```python
# New usage
particles = np.random.rand(500, 2)
solver = FPParticleSolver(
    problem,
    mode="collocation",
    external_particles=particles
)
M = solver.solve_fp_system(...)
```

---

## Prototype Validation

**Location**: `mfg-research/algorithms/particle_collocation/`

**Files**:
- `fp_particle_dual_mode.py` (427 lines implementation)
- `test_fp_dual_mode.py` (377 lines tests)

**Test Results**: 13/13 tests passing ✅
- 5 hybrid mode tests ✅
- 6 collocation mode tests ✅
- 2 integration tests ✅

**Integration Test**: ✅ Works with particle-collocation MFG solver

---

## Use Case

**Particle-Collocation MFG**:

```python
# Setup
particles = domain.sample_uniform(500)
problem = ParticleCollocationProblem(N=500, particles=particles, ...)

# HJB on particles
hjb_solver = HJBGFDMSolver(problem, collocation_points=particles)

# FP on SAME particles (collocation mode)
fp_solver = FPParticleSolver(
    problem,
    mode="collocation",
    external_particles=particles
)

# Full particle-collocation MFG
solver = FixedPointIterator(problem, hjb_solver, fp_solver)
U, M, info = solver.solve()  # Returns (Nt, N) on particles!
```

---

## Benefits

**For Users**:
- ✅ Enables high-dimensional MFG (d ≥ 3)
- ✅ Complex geometries (mazes, obstacles)
- ✅ Meshfree methods as first-class citizens
- ✅ 100% backward compatible

**For MFG_PDE**:
- ✅ First framework with meshfree MFG support
- ✅ Expands to new problem classes
- ✅ Low risk (~95 lines)
- ✅ No breaking changes

---

## Relationship to Pure Particle Interface

This complements the Pure Particle Interface design:

**Pure Particle Interface**:
- `MFGProblem` supports particle discretization
- Unified interface (`num_spatial_points`, `spatial_points`)

**Dual-Mode FPParticleSolver**:
- FP solver works natively with particles
- Completes particle-collocation capability

**Together**: Full particle-based MFG in MFG_PDE v1.8.0

---

## Implementation Roadmap

**Phase 1**: Port prototype to MFG_PDE (Week 1-2)
- [ ] Copy `_solve_collocation_mode()` to `FPParticleSolver`
- [ ] Add mode parameter to `__init__`
- [ ] Add dispatch logic to `solve_fp_system()`
- [ ] Add GPU support for collocation mode
- [ ] Code review

**Phase 2**: Testing (Week 3)
- [ ] Add collocation mode tests
- [ ] Verify all existing tests pass (backward compat)
- [ ] Performance benchmarks
- [ ] Type checking, linting

**Phase 3**: Documentation (Week 4)
- [ ] Update docstrings
- [ ] Tutorial: "Particle-Collocation MFG"
- [ ] Example: 3D crowd dynamics
- [ ] API reference

**Phase 4**: Release (Week 5-6)
- [ ] Final review
- [ ] Version bump to v1.8.0
- [ ] Release notes
- [ ] Publish to PyPI

**Total**: 6 weeks to v1.8.0

---

## Success Metrics

**Must-Have**:
- ✅ All existing tests pass
- ✅ All new collocation tests pass
- ✅ No performance regression
- ✅ Type checking passes
- ✅ Documentation complete

**Should-Have**:
- ✅ Code coverage >90%
- ✅ 2+ examples
- ✅ Tutorial reviewed

---

## Risk Assessment

**Risk Level**: LOW

**Mitigations**:
- Prototype validated (13/13 tests)
- 100% backward compatible
- Minimal code changes (~95 lines)
- Opt-in feature (mode parameter)

---

## References

**Prototype**: `mfg-research/algorithms/particle_collocation/`
- Implementation: `fp_particle_dual_mode.py`
- Tests: `test_fp_dual_mode.py`
- Proposal: `docs/DUAL_MODE_FP_PROPOSAL_v1.8.0.md`

**Design Docs** (mfg-research):
- `FP_PARTICLE_HYBRID_VS_COLLOCATION.md`
- `HYBRID_SOLVER_INTERFACE_ANALYSIS.md`
- `HYBRID_VS_PARTICLE_COLLOCATION.md`

---

**Last Updated**: 2025-10-17
**Status**: Prototype complete, ready for MFG_PDE integration
**Maintainer**: MFG Research Group
