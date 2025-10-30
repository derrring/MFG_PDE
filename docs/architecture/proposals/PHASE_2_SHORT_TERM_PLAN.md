# Phase 2: Short-Term Improvements Plan

**Status**: READY TO BEGIN
**Timeline**: 3 months (12 weeks)
**Priority**: HIGH
**Dependencies**: Phase 1 complete ✅ (PR #201, #202 merged)

---

## Executive Summary

Phase 2 focuses on high-impact improvements that address remaining research pain points identified in the architecture audit. All changes maintain backward compatibility and follow established patterns from Phase 1.

**Goals**:
1. Extend FDM solvers to 2D/3D (unblock baseline comparisons)
2. Add missing utilities (reduce code duplication in research)
3. Quick wins (improve user experience)

**Expected Impact**:
- Unblock 1 additional research feature (2D FDM baselines)
- Save ~100-200 hours/year in research code duplication
- Improve overall user experience

---

## Timeline Overview

```
Week 1-6:   2D/3D FDM Solvers (HIGH priority)
Week 7-10:  Missing Utilities (MEDIUM priority)
Week 11:    Quick Wins (LOW priority, high ROI)
Week 12:    Documentation, Testing, Release
```

**Parallelization Opportunities**:
- Weeks 7-11: Utilities and Quick Wins can proceed in parallel with testing
- Continuous integration testing throughout all phases

---

## Priority 1: 2D/3D FDM Solvers (Weeks 1-6)

### Motivation

**Current Limitation**: FDM solvers only work in 1D
- `HJBFDMSolver`: 1D only
- `FPFDMSolver`: 1D only

**Impact on Research**:
- Cannot provide FDM baseline for 2D research
- Forces use of GFDM (more complex, harder to validate)
- Missing classical comparison point for papers
- Blocks 1 of 5 initially blocked features

**User Need**: "I want to compare my novel 2D method against classical FDM"

### Implementation Plan

#### Week 1-2: HJB FDM 2D Extension

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`

**Approach**: Dimensional splitting
```python
# 2D HJB: -∂u/∂t + H(∇u, m) = 0
# Split into x and y sweeps:
# Step 1: -∂u/∂t + H_x(∂u/∂x, m) = 0  (sweep in x)
# Step 2: -∂u/∂t + H_y(∂u/∂y, m) = 0  (sweep in y)
```

**Changes Required**:
1. **Detect problem dimension** from `problem.dim` or grid shape
2. **2D grid indexing**: Convert 1D code to handle `(Nx, Ny)` grids
3. **Dimensional splitting loop**:
   ```python
   for dim in range(problem.dim):
       # Sweep in dimension `dim`
       u = self._solve_1d_sweep(u, m, dim=dim, dt=dt)
   ```
4. **Boundary conditions**: Extend to 2D (already have infrastructure)
5. **Monotone scheme check**: Extend QP monotonicity to 2D

**Backward Compatibility**: Detect 1D problems and use existing path

**Tests**:
- 2D convergence test (compare with known solution)
- Backward compatibility (1D problems unchanged)
- Mass conservation check
- Boundary condition handling

**Estimated Time**: 2 weeks

---

#### Week 3-4: FP FDM 2D Extension

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

**Approach**: Dimensional splitting for advection-diffusion
```python
# 2D FP: ∂m/∂t + ∇·(m v) = σ² Δm
# Split into:
# Step 1: ∂m/∂t + ∂(m v_x)/∂x = σ² ∂²m/∂x²  (x-direction)
# Step 2: ∂m/∂t + ∂(m v_y)/∂y = σ² ∂²m/∂y²  (y-direction)
```

**Changes Required**:
1. **2D grid support**: Handle `(Nx, Ny)` density arrays
2. **Dimensional splitting**:
   ```python
   for dim in range(problem.dim):
       m = self._fp_1d_sweep(m, velocity[dim], dim=dim, dt=dt)
   ```
3. **Velocity field**: Extract `(v_x, v_y)` from HJB gradient
4. **Mass conservation**: Verify total mass preserved in 2D
5. **Positivity**: Ensure m ≥ 0 throughout

**Backward Compatibility**: Detect 1D and use existing path

**Tests**:
- 2D mass conservation (∫m dx dy = 1)
- Positivity (m ≥ 0 everywhere)
- Convergence to known solution
- Coupling with 2D HJB FDM

**Estimated Time**: 2 weeks

---

#### Week 5: 3D Extension (If Time Permits)

**Goal**: Extend both solvers to 3D using same dimensional splitting

**Feasibility**: If 2D implementation is clean, 3D is straightforward
- Same splitting approach
- Just loop over 3 dimensions instead of 2
- May need sparse storage for 3D grids

**Tests**:
- 3D mass conservation
- 3D boundary conditions
- Performance benchmarks

**Fallback**: Skip 3D if 2D takes longer than expected. Can add in future release.

**Estimated Time**: 1 week (optional)

---

#### Week 6: Integration and Validation

**Factory Integration**:
```python
from mfg_pde.factory import create_fast_solver
from mfg_pde import HighDimMFGProblem

# 2D problem
problem = HighDimMFGProblem(dim=2, bounds=[(0, 1), (0, 1)], N=[50, 50])
solver = create_fast_solver(problem, solver_type="fdm")  # Auto-detects 2D
result = solver.solve()
```

**Documentation**:
- Update solver documentation
- Add 2D FDM example to `examples/basic/`
- Migration guide for 1D → 2D users

**Validation**:
- Compare 2D FDM vs 2D GFDM on standard problems
- Benchmark performance
- Verify accuracy matches expected convergence rates

**Estimated Time**: 1 week

---

### Deliverables (Priority 1)

**Code**:
- ✅ `HJBFDMSolver` supports 2D (and optionally 3D)
- ✅ `FPFDMSolver` supports 2D (and optionally 3D)
- ✅ Backward compatible with 1D problems
- ✅ Dimensional splitting implementation
- ✅ Comprehensive tests (8+ new tests)

**Documentation**:
- ✅ Updated solver docs
- ✅ 2D FDM example
- ✅ Migration guide

**Tests**:
- 2D HJB convergence
- 2D FP mass conservation
- 2D coupled MFG system
- Backward compatibility (1D)
- Boundary conditions (2D)
- Performance benchmarks

**Success Metrics**:
- Can solve 2D MFG with pure FDM (no GFDM)
- Mass conservation ≤ 10⁻¹⁰ error
- Convergence rate O(h²) for smooth problems
- No regressions in 1D problems

---

## Priority 2: Missing Utilities (Weeks 7-10)

### Motivation

**Current Pain Points**:
- Research code duplicates interpolation logic
- SDF computations done manually each time
- Convergence monitoring requires custom code
- QP results not cached (recompute every iteration)

**Impact**: ~100-200 hours/year spent on repetitive utility code

### Implementation Plan

#### Week 7: Particle Interpolation Utilities

**File**: `mfg_pde/utils/interpolation.py` (new)

**Functionality**:
1. **RBF Interpolation**:
   ```python
   from mfg_pde.utils.interpolation import RBFInterpolator

   interp = RBFInterpolator(particles, values, kernel='gaussian')
   values_at_x = interp(query_points)
   ```

2. **K-Nearest Neighbors**:
   ```python
   from mfg_pde.utils.interpolation import KNNInterpolator

   interp = KNNInterpolator(particles, values, k=10)
   values_at_x = interp(query_points)
   ```

3. **Adaptive K (already in research code)**:
   ```python
   interp = KNNInterpolator(particles, values, adaptive_k=True)
   ```

**Tests**:
- Interpolation accuracy tests
- Performance benchmarks (vs scipy)
- Edge cases (duplicate points, boundary points)

**Estimated Time**: 1 week

---

#### Week 8: SDF and Geometry Helpers

**File**: `mfg_pde/utils/geometry.py` (enhance existing)

**Functionality**:
1. **SDF computation**:
   ```python
   from mfg_pde.utils.geometry import compute_sdf

   # For obstacles
   obstacle_sdf = compute_sdf(obstacle_points, query_points)

   # For boundaries
   domain = Domain2D(bounds=[(0, 1), (0, 1)])
   boundary_sdf = domain.compute_sdf(query_points)
   ```

2. **Inside/Outside Tests**:
   ```python
   is_inside = domain.contains(points)
   distance_to_boundary = domain.distance(points)
   ```

3. **Projection to Boundary**:
   ```python
   projected_points = domain.project_to_boundary(points)
   ```

**Tests**:
- SDF accuracy for simple shapes
- Inside/outside correctness
- Projection accuracy

**Estimated Time**: 1 week

---

#### Week 9: Convergence Monitoring

**File**: `mfg_pde/utils/monitoring.py` (new)

**Functionality**:
1. **Convergence Tracker**:
   ```python
   from mfg_pde.utils.monitoring import ConvergenceMonitor

   monitor = ConvergenceMonitor(
       metrics=['residual', 'value_change', 'density_change'],
       tolerance=1e-4,
       window=5  # Check last 5 iterations
   )

   for iteration in range(max_iter):
       U_new, M_new = solve_step(U, M)

       converged = monitor.update({
           'residual': compute_residual(U_new, M_new),
           'value_change': np.linalg.norm(U_new - U),
           'density_change': np.linalg.norm(M_new - M)
       })

       if converged:
           break

   monitor.plot()  # Convergence curves
   monitor.summary()  # Statistics
   ```

2. **Progress Display**:
   ```python
   from mfg_pde.utils.monitoring import ProgressDisplay

   with ProgressDisplay(max_iter=100) as progress:
       for i in range(max_iter):
           # ... solve ...
           progress.update(metrics={'residual': res, 'time': elapsed})
   ```

**Tests**:
- Convergence detection accuracy
- Visualization correctness
- Performance overhead (should be minimal)

**Estimated Time**: 1 week

---

#### Week 10: QP Result Caching

**File**: `mfg_pde/utils/caching.py` (new)

**Functionality**:
1. **QP Cache**:
   ```python
   from mfg_pde.utils.caching import QPCache

   cache = QPCache(max_size=1000)

   # Check cache before solving QP
   key = cache.make_key(Q, c, constraints)
   if key in cache:
       solution = cache[key]
   else:
       solution = solve_qp(Q, c, constraints)
       cache[key] = solution
   ```

2. **Particle Neighborhood Cache**:
   ```python
   from mfg_pde.utils.caching import NeighborhoodCache

   cache = NeighborhoodCache(particles)
   neighbors = cache.get_neighbors(particle_idx, radius=0.1)
   ```

**Tests**:
- Cache hit/miss correctness
- Performance improvement measurements
- Memory usage tests

**Estimated Time**: 1 week

---

### Deliverables (Priority 2)

**Code**:
- ✅ `mfg_pde.utils.interpolation` module
- ✅ `mfg_pde.utils.geometry` enhancements
- ✅ `mfg_pde.utils.monitoring` module
- ✅ `mfg_pde.utils.caching` module

**Documentation**:
- API documentation for each utility
- Usage examples
- Performance characteristics

**Tests**:
- 10+ new utility tests
- Performance benchmarks
- Integration tests with solvers

**Success Metrics**:
- Research code can import and use utilities
- Eliminates ~50-100 lines duplicate code per research project
- No performance regressions

---

## Priority 3: Quick Wins (Week 11)

### Motivation

**Low effort, high impact improvements** based on research feedback.

### Implementation Plan

#### Standardize Solver Return Format

**Current Issue**: Different solvers return different formats
- Some return `(U, M, info)`
- Some return `SolverResult` object
- Inconsistent `info` dictionaries

**Fix**:
```python
@dataclass
class MFGSolution:
    """Standardized MFG solution container."""
    U: np.ndarray  # Value function
    M: np.ndarray  # Density
    converged: bool
    iterations: int
    residual: float
    computation_time: float
    solver_info: dict  # Solver-specific details

    def save(self, filename): ...
    def plot(self): ...
```

**Impact**: Consistent interface, easier to write generic code

**Estimated Time**: 2 days

---

#### Add Convergence Monitoring to All Solvers

**Current Issue**: Some solvers don't track convergence metrics

**Fix**: Integrate `ConvergenceMonitor` from utilities
```python
class BaseMFGSolver:
    def solve(self, monitor=True):
        if monitor:
            tracker = ConvergenceMonitor(...)

        for iter in range(max_iter):
            # ... solve ...
            if monitor and tracker.update(...):
                break

        return MFGSolution(converged=tracker.converged, ...)
```

**Impact**: Better visibility into solver behavior

**Estimated Time**: 2 days

---

#### Improve Error Messages

**Current Issue**: Generic errors like "Solver failed to converge"

**Fix**: Actionable error messages
```python
# Before
raise ValueError("Solver failed to converge")

# After
raise ConvergenceError(
    "Solver failed to converge after 100 iterations. "
    "Residual: 1.2e-3 (tolerance: 1e-4). "
    "Try: (1) Increase max_iter, (2) Decrease tolerance, "
    "(3) Improve initial guess, or (4) Use damping (α=0.5)"
)
```

**Impact**: Users spend less time debugging

**Estimated Time**: 1 day

---

### Deliverables (Priority 3)

**Code**:
- ✅ Standardized `MFGSolution` return type
- ✅ Convergence monitoring in all solvers
- ✅ Actionable error messages

**Documentation**:
- Migration guide for return format changes
- Error message catalog

**Tests**:
- Test standard return format
- Test error message content

**Success Metrics**:
- All solvers return `MFGSolution`
- All solvers support `monitor=True`
- User feedback: Error messages are helpful

---

## Week 12: Documentation, Testing, and Release

### Documentation

**Updates Required**:
1. **API Documentation**:
   - Document 2D/3D FDM solvers
   - Document new utilities
   - Update solver return format docs

2. **Examples**:
   - `examples/basic/2d_fdm_crowd_motion.py`
   - `examples/basic/using_utilities.py`
   - `examples/advanced/custom_convergence_monitoring.py`

3. **Migration Guides**:
   - 1D → 2D FDM migration
   - Adopting new utilities
   - Updating to `MFGSolution` return format

4. **CHANGELOG**:
   - Document all Phase 2 changes
   - Highlight backward compatibility

**Estimated Time**: 3 days

---

### Testing

**Comprehensive Test Suite**:
1. **Regression Tests**:
   - All Phase 1 tests still passing
   - No performance regressions
   - Backward compatibility verified

2. **New Feature Tests**:
   - 2D/3D FDM comprehensive tests (10+ tests)
   - Utility tests (10+ tests)
   - Quick wins tests (5+ tests)

3. **Integration Tests**:
   - 2D FDM + utilities
   - Full MFG system with monitoring
   - Example scripts execution

4. **Performance Tests**:
   - 2D FDM vs GFDM benchmarks
   - Utility performance measurements
   - Memory usage profiling

**Estimated Time**: 2 days

---

### Release Preparation

**Version**: v1.8.0 (minor version bump for new features)

**Release Checklist**:
- [ ] All tests passing (>97.8% pass rate maintained)
- [ ] Documentation complete and reviewed
- [ ] CHANGELOG updated
- [ ] Examples working
- [ ] Performance benchmarks documented
- [ ] Migration guides complete
- [ ] GitHub release notes drafted

**Release Notes Structure**:
```markdown
# MFG_PDE v1.8.0 - Phase 2 Short-Term Improvements

## New Features

### 2D/3D FDM Solvers
- HJBFDMSolver now supports 2D/3D via dimensional splitting
- FPFDMSolver now supports 2D/3D
- Example: `examples/basic/2d_fdm_crowd_motion.py`

### New Utilities
- `mfg_pde.utils.interpolation`: RBF and KNN interpolators
- `mfg_pde.utils.geometry`: SDF and geometry helpers
- `mfg_pde.utils.monitoring`: Convergence tracking
- `mfg_pde.utils.caching`: QP and neighborhood caching

### Quick Wins
- Standardized `MFGSolution` return type
- Convergence monitoring in all solvers
- Improved error messages with actionable guidance

## Backward Compatibility

All changes are backward compatible. Existing code continues to work.

## Migration Guides

See `docs/migration_guides/phase2_improvements.md`

## Performance

- 2D FDM comparable to GFDM for regular domains
- Utilities reduce research code duplication by ~50-100 lines/project
- No performance regressions in existing solvers
```

**Estimated Time**: 0.5 days

---

## Risk Assessment and Mitigation

### Risk 1: 2D FDM Performance Issues

**Risk**: Dimensional splitting may be slower than expected
**Probability**: MEDIUM
**Impact**: MEDIUM
**Mitigation**:
- Benchmark early (Week 2)
- If too slow, document and recommend GFDM for large problems
- Consider sparse matrix optimization
**Fallback**: Release 2D only if 3D is too slow

---

### Risk 2: Utility API Design

**Risk**: Utilities API may not match all research use cases
**Probability**: LOW
**Impact**: LOW
**Mitigation**:
- Design APIs based on actual research code patterns
- Make utilities extensible (base classes)
- Gather feedback during Weeks 7-10
**Fallback**: Release minimal utilities, extend in v1.8.1

---

### Risk 3: Timeline Overrun

**Risk**: Tasks take longer than estimated
**Probability**: MEDIUM
**Impact**: MEDIUM
**Mitigation**:
- Prioritize: FDM solvers (HIGH) > Utilities (MEDIUM) > Quick Wins (LOW)
- If behind schedule, cut scope:
  - Skip 3D extension (add in v1.8.1)
  - Reduce utilities (ship interpolation only)
  - Defer some quick wins
**Fallback**: Release v1.8.0 with FDM 2D only, utilities in v1.9.0

---

### Risk 4: Breaking Changes Discovered

**Risk**: Changes require breaking API
**Probability**: LOW
**Impact**: HIGH
**Mitigation**:
- Design Phase 2 to be backward compatible
- Use deprecation warnings if API changes needed
- Extensive backward compatibility testing
**Fallback**: Delay breaking changes to Phase 3 or v2.0

---

## Success Criteria

**Technical**:
- ✅ 2D FDM solvers working with dimensional splitting
- ✅ Utilities reduce research code duplication
- ✅ All quick wins implemented
- ✅ >97.8% test pass rate maintained
- ✅ No performance regressions

**Documentation**:
- ✅ API docs complete
- ✅ 3+ new examples
- ✅ Migration guides written
- ✅ CHANGELOG updated

**Research Impact**:
- ✅ Can solve 2D MFG with pure FDM
- ✅ Research code imports utilities instead of duplicating
- ✅ Positive user feedback on improvements

**Timeline**:
- ✅ Complete within 12 weeks
- ✅ Release v1.8.0 by end of Week 12

---

## Post-Phase 2

**After v1.8.0 Release**:
1. **Gather Feedback** (2 weeks): Monitor GitHub issues, user reports
2. **Bug Fixes** (ongoing): Address issues in v1.8.1, v1.8.2
3. **Plan Phase 3** (4 weeks): Long-term refactoring planning
4. **Begin Phase 3** (Month 4): Unified problem class, configuration simplification

**Phase 3 Timeline**: 6-9 months (Weeks 13-52)

---

## Resources Required

**Development Time**:
- **Weeks 1-6**: 1 developer full-time (2D/3D FDM)
- **Weeks 7-10**: 1 developer full-time (Utilities)
- **Week 11**: 0.5 developer (Quick Wins)
- **Week 12**: 0.5 developer (Release)
- **Total**: ~11 developer-weeks

**Testing Time**:
- Continuous testing throughout (included in estimates)
- Week 12: 2 days intensive testing

**Documentation Time**:
- Inline docs: Included in development estimates
- Examples and guides: Week 12 (3 days)

**Total Estimated Effort**: ~12 weeks (1 developer, full-time)

---

## Conclusion

**Phase 2 is well-scoped, achievable, and high-impact.**

**Key Benefits**:
1. Unblocks 2D FDM baseline comparisons
2. Reduces research code duplication
3. Improves overall user experience
4. Maintains backward compatibility
5. Clear path from Phase 1 → Phase 2 → Phase 3

**Ready to Begin**: Phase 1 complete, requirements clear, plan detailed.

---

**Document Version**: 1.0
**Created**: 2025-10-30
**Status**: APPROVED FOR IMPLEMENTATION
**Next Review**: End of Week 6 (checkpoint after FDM completion)
**Contact**: See GitHub Issue #200 for discussions
