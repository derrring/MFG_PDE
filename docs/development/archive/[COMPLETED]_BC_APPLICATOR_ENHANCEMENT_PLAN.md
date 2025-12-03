# BC Applicator Enhancement Plan

**Date**: 2025-11-27
**Status**: COMPLETE - Integrated with tensor_operators

---

## Current Defects

### Critical (Phase 1)

| Issue | Current State | Impact |
|-------|---------------|--------|
| Dirichlet ghost formula | `2*g - interior` assumes boundary at cell face | Incorrect BC for cell-centered grids |
| Neumann sign convention | Sign based on boundary side name | May be inverted for some stencils |
| No stencil awareness | Assumes 3-point stencil | Wrong for WENO, high-order FD |

### Functional (Phase 2-3)

| Issue | Current State | Impact |
|-------|---------------|--------|
| Robin BC | Approximated as Neumann | Cannot solve Robin problems |
| Time-dependent values | `value(x,t)` not evaluated | Static BC only |
| Periodic in mixed BC | Returns interior value | Doesn't wrap correctly |

### Performance (Phase 4)

| Issue | Current State | Impact |
|-------|---------------|--------|
| Per-point iteration | O(N_boundary * N_segments) | Slow for fine grids |
| No caching | Recomputes segment match each call | Redundant computation |

### Robustness (Phase 5)

| Issue | Current State | Impact |
|-------|---------------|--------|
| No shape validation | Assumes correct input | Silent failures |
| No NaN/Inf check | Propagates bad values | Solver divergence |

---

## Enhancement Plan

### Phase 1: Critical Bug Fixes

**Goal**: Correct ghost cell formulas for standard FD stencils.

**Changes**:
1. Add `stencil_type` parameter: `"cell_centered"` vs `"vertex_centered"`
2. Fix Dirichlet formula:
   - Cell-centered: `ghost = 2*g - interior` (boundary at face)
   - Vertex-centered: `ghost = g` (boundary at vertex)
3. Fix Neumann formula with explicit normal direction
4. Add unit tests with known analytical solutions

### Phase 2: Proper Robin BC

**Goal**: Implement Robin BC: `alpha*u + beta*du/dn = g`

**Formula** (cell-centered, outward normal):
```
alpha * (ghost + interior)/2 + beta * (ghost - interior)/(2*dx) = g
ghost = (g - alpha*interior/2 + beta*interior/(2*dx)) / (alpha/2 + beta/(2*dx))
```

**Changes**:
1. Add Robin ghost cell computation
2. Handle edge cases (alpha=0 -> Neumann, beta=0 -> Dirichlet)
3. Add tests comparing to analytical Robin solutions

### Phase 3: Time-Dependent BC

**Goal**: Support `value(x, t)` callable for time-varying BCs.

**Changes**:
1. Add `time` parameter to `apply_boundary_conditions_2d()`
2. Evaluate callable values with `(point, time)`
3. Handle both scalar and callable values uniformly

### Phase 4: Vectorization

**Goal**: O(N_boundary) instead of O(N_boundary * N_segments).

**Strategy**:
1. Pre-compute boundary masks once per grid configuration
2. Use boolean indexing for ghost cell assignment
3. Cache masks in `MixedBoundaryConditions` object

**Expected speedup**: 5-10x for typical mixed BC configurations.

### Phase 5: Input Validation

**Goal**: Fail fast with clear error messages.

**Checks**:
1. Field shape matches domain_bounds dimension
2. Grid spacing is positive
3. No NaN/Inf in field
4. BC values are finite
5. Domain bounds are valid (min < max)

---

## Implementation Order

```
Phase 1 (Critical)     [DONE] - Correct ghost cell formulas
    |
    v
Phase 2 (Robin)        [DONE] - alpha*u + beta*du/dn = g
    |
    v
Phase 3 (Time-dep)     [DONE] - value(point, time) support
    |
    v
Phase 4 (Performance)  [DEFERRED] - Vectorization
    |
    v
Phase 5 (Validation)   [DONE] - NaN/Inf/bounds checks
    |
    v
Integration with tensor_operators  <- NEXT
```

---

## Test Plan

Each phase adds corresponding tests:

- **Phase 1**: Convergence tests with manufactured solutions
- **Phase 2**: Robin BC analytical comparison
- **Phase 3**: Time-varying BC tests
- **Phase 4**: Performance benchmarks
- **Phase 5**: Error message tests

---

## Success Criteria

- [x] All existing tests pass (68 tests)
- [x] Phase 1: Dirichlet/Neumann ghost cell formulas corrected
- [x] Phase 2: Robin BC implemented with alpha/beta coefficients
- [x] Phase 3: Time-dependent BC via callable values
- [ ] Phase 4: Vectorization (deferred - current perf acceptable)
- [x] Phase 5: Clear errors for invalid input (NaN, Inf, bounds)
- [x] Integration: tensor_operators uses new BC applicator

---

## Completion Summary

**Completed**: 2025-11-27

### Files Modified

1. `mfg_pde/geometry/boundary/mixed_bc.py`
   - Added `alpha`, `beta` attributes to `BCSegment` for Robin BC

2. `mfg_pde/geometry/boundary/bc_applicator.py`
   - `GhostCellConfig`: Grid type configuration (cell/vertex-centered)
   - `apply_boundary_conditions_2d()`: Main BC application function
   - `_compute_ghost_value_enhanced()`: Correct ghost cell formulas
   - `_evaluate_bc_value()`: Time-dependent BC evaluation
   - `_validate_field_2d()`, `_validate_domain_bounds_2d()`: Input validation

3. `mfg_pde/utils/numerical/tensor_operators.py`
   - Updated `_apply_bc_2d()` to use new BC applicator
   - Added `domain_bounds` and `time` parameters to public API

### Test Coverage

- `test_mixed_bc.py`: 30 tests (SDF, normal matching, regions)
- `test_bc_applicator.py`: 24 tests (Robin, time-dep, validation)
- `test_tensor_operators.py`: 14 tests (integration verified)

### Usage Example

```python
from mfg_pde.geometry.boundary import (
    BCSegment, BCType, MixedBoundaryConditions,
    apply_boundary_conditions_2d,
)

# Define mixed BCs
exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=lambda pt, t: np.sin(t),  # Time-dependent
    boundary="x_max",
    priority=1,
)
robin_bc = BCSegment(
    name="wall",
    bc_type=BCType.ROBIN,
    value=0.0,
    alpha=1.0, beta=0.5,  # Robin coefficients
    priority=0,
)
mixed_bc = MixedBoundaryConditions(
    dimension=2,
    segments=[exit_bc, robin_bc],
    domain_bounds=np.array([[0, 1], [0, 1]]),
)

# Apply to field
padded = apply_boundary_conditions_2d(field, mixed_bc, time=0.5)
```
