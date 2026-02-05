# Boundary Condition Handling: Comprehensive Architecture Analysis

**Last Updated:** 2026-01-22
**Status:** Living document - updated as architecture evolves

---

## 1. Executive Summary

The boundary condition (BC) handling in MFG_PDE is **architecturally ambitious but inconsistently executed**. The core data model (`BoundaryConditions`, `BCSegment`) is well-designed, but the application layer suffers from:

- **Parallel implementations** that were never unified
- **Deprecated code** still in active use
- **Incomplete migrations** (GhostBuffer, provider pattern)
- **Dimension-specific implementations** that don't generalize

The recent Issue #625 (BCValueProvider) addresses one antipattern but exposes others. This document provides an honest assessment to guide cleanup and completion.

---

## 2. Architecture Overview

### 2.1 Layered Design (Intended)

```
┌─────────────────────────────────────────────────────────────┐
│  User API: BoundaryConditions, BCSegment, factory functions │
├─────────────────────────────────────────────────────────────┤
│  Provider Layer: BCValueProvider, AdjointConsistentProvider │  ← NEW (v0.18.0)
├─────────────────────────────────────────────────────────────┤
│  Application Layer: Topology + Calculator composition       │  ← INCOMPLETE
├─────────────────────────────────────────────────────────────┤
│  Legacy Layer: apply_boundary_conditions_*() functions      │  ← DEPRECATED but used
├─────────────────────────────────────────────────────────────┤
│  Solver Integration: HJB/FP solvers consume BCs            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Files (Verified 2026-01-22)

| File | Lines | Purpose | Health |
|------|-------|---------|--------|
| `types.py` | 679 | BCType, BCSegment | Good (hasattr violations) |
| `conditions.py` | 1063 | BoundaryConditions | Bloated god class |
| `providers.py` | 384 | BCValueProvider | Good (new, clean) |
| `applicator_fdm.py` | 3457 | FDM BC application | **Chaotic** |
| `applicator_base.py` | 1869 | Topology/Calculator | Well-designed, unused |
| `bc_coupling.py` | 329 | Adjoint-consistent BC | 1D only |
| `dispatch.py` | 292 | Applicator routing | FEM NotImplemented |
| `constraints.py` | 664 | VI constraints | Good |
| `fem_bc_*.py` | ~1900 | FEM BC handlers | hasattr violations |

**Total BC module**: 24 files, 15,163 lines

---

## 3. Strengths

### 3.1 Data Model Design

The `BCSegment` dataclass is well-designed with:
- **Multiple matching modes**: boundary ID, coordinate ranges, SDF, normal direction, region names
- **Priority-based resolution**: Overlapping segments resolved deterministically
- **Lazy dimension binding**: BC can be created before dimension is known
- **Callable value support**: Time-dependent and space-dependent BCs

### 3.2 Provider Pattern (Issue #625) ✅ IMPLEMENTED

Clean separation of concerns:
```python
# Intent stored in BC object, not solver
segment = BCSegment(
    bc_type=BCType.ROBIN,
    value=AdjointConsistentProvider(side="left", sigma=0.2),  # Provider, not value
)

# Iterator resolves providers, solver stays generic
with problem.using_resolved_bc(state):
    U = hjb_solver.solve(...)  # Receives concrete values
```

### 3.3 Adjoint-Consistent BC Theory

Correct mathematical treatment of reflecting boundaries:
```
∂U/∂n = -σ²/2 · ∂ln(m)/∂n
```
This couples HJB boundary condition to FP density gradient, essential for equilibrium consistency at boundary stall points.

---

## 4. Critical Analysis: Problems and Antipatterns

### 4.1 ANTIPATTERN: Parallel Implementations Never Unified

**Problem:** Three separate BC application paths exist:

| Path | Location | Used By | Status |
|------|----------|---------|--------|
| Legacy functions | `applicator_fdm.py` | Most solvers | Deprecated but active |
| GhostBuffer | `applicator_base.py` | Nothing yet | Incomplete |
| Direct manipulation | Various solvers | Some solvers | Ad-hoc |

**Evidence:**
```python
# applicator_fdm.py has 3000+ lines with:
def apply_boundary_conditions_2d(...)  # Deprecated v0.19.0
def apply_boundary_conditions_nd(...)  # Deprecated v0.19.0
def _apply_bc_uniform_2d(...)          # Internal, duplicates logic
def _apply_bc_mixed_2d(...)            # Internal, duplicates logic
# ... 20+ more functions with overlapping responsibilities
```

**Impact:** Bug fixes must be applied in multiple places. Inconsistent behavior between paths.

---

### 4.2 ANTIPATTERN: Incomplete Migration (GhostBuffer)

**Problem:** Issue #516 introduced `GhostBuffer` architecture for zero-allocation BC padding, but migration was never completed.

**Current state:**
- `Topology` classes: Implemented (PeriodicTopology, BoundedTopology) - `applicator_base.py:295-375`
- `Calculator` classes: Implemented (Dirichlet, Neumann, Robin, ZeroFlux, etc.) - `applicator_base.py:382-749`
- `LinearConstraint` for matrix assembly: Implemented - `applicator_base.py:790-894`
- **PreallocatedGhostBuffer**: Never materialized
- **Solver integration**: Zero solvers use the new pattern
- **FEM BC**: `NotImplementedError` at `dispatch.py:218`

**Code smell:**
```python
# dispatch.py:216-219 - FEM BC application not implemented
elif discretization == DiscretizationType.FEM:
    raise NotImplementedError(
        "FEM BC application requires matrix/rhs modification. Use FEMApplicator.apply(matrix, rhs, mesh) instead."
    )
```

**Impact:**
- The well-designed Topology/Calculator layer in `applicator_base.py` (1869 lines) sits unused
- Solvers still call deprecated `apply_boundary_conditions_*()` functions
- Performance benefits of zero-allocation never realized

---

### 4.3 ANTIPATTERN: Dimension-Specific Implementations

**Problem:** Many features are implemented only for 1D, with `NotImplementedError` for higher dimensions.

| Feature | 1D | 2D | 3D | nD |
|---------|----|----|----|----|
| Adjoint-consistent BC | ✅ | ❌ | ❌ | ❌ |
| Robin corner handling | ✅ | ⚠️ | ⚠️ | ❌ |
| GhostBuffer | ✅ | ⚠️ | ❌ | ❌ |
| SDF region matching | ✅ | ⚠️ | ❌ | ❌ |

**Evidence:**
```python
# bc_coupling.py:250
def create_adjoint_consistent_bc_nd(m_current, geometry, sigma, ...):
    dimension = geometry.dimension
    if dimension == 1:
        return create_adjoint_consistent_bc_1d(...)
    else:
        raise NotImplementedError(
            f"Adjoint-consistent BC for {dimension}D not yet implemented"
        )
```

**Impact:** Research problems requiring 2D/3D adjoint-consistent BC cannot use this feature.

---

### 4.4 ANTIPATTERN: God Class (`conditions.py`)

**Problem:** `BoundaryConditions` class has grown to 1000+ lines with too many responsibilities:

- BC segment storage and querying
- Dimension binding
- Provider detection and resolution
- Domain bounds management
- SDF handling
- Validation
- String representation
- Factory method delegation

**Smell indicators:**
- 30+ methods on single class
- Multiple `# --- Section ---` comments to organize
- Methods that belong on `BCSegment` put on `BoundaryConditions`

**Recommended refactor:**
```python
# Split into:
class BCSegmentCollection:  # Storage and querying
class BCDimensionBinder:    # Dimension binding logic
class BCProviderResolver:   # Provider resolution
class BCValidator:          # Validation logic
```

---

### 4.5 ANTIPATTERN: Deprecated Code Still Primary Path

**Problem:** Deprecated functions are still the main code path.

```python
# applicator_fdm.py - marked deprecated but called by all FDM solvers
@deprecated(since="v0.19.0", replacement="PreallocatedGhostBuffer")
def apply_boundary_conditions_2d(field, bc, ...):
    ...

# hjb_fdm.py - still uses deprecated path
def _apply_bc_to_solution(self, U, ...):
    return apply_boundary_conditions_2d(U, self.bc, ...)  # Deprecated!
```

**Impact:** Technical debt accumulates. New features must support both paths.

---

### 4.6 WRONG DESIGN: bc_mode on Solver (Removed in v0.18.0) ✅ COMPLETED

**Problem (removed):** MFG coupling logic was embedded in HJB solver via `bc_mode` parameter.

**Fix (Issue #625, #703):** Provider pattern moves coupling to iterator. The `bc_mode` parameter was removed in v0.18.0.

```python
# Current (v0.18.0+) - Solver is generic, iterator handles coupling
class HJBFDMSolver:
    def solve(self, ...):
        bc = self.problem.boundary_conditions  # Just uses what it's given

class FixedPointIterator:
    def step(self, m_current, ...):
        with self.problem.using_resolved_bc({"m_current": m_current}):
            U = self.hjb_solver.solve(...)  # Provider resolved here
```

**Status:** ✅ Removed. Callers passing `bc_mode=` get `TypeError`.

---

### 4.7 CHAOTIC: applicator_fdm.py Structure

**Problem:** This file is 3000+ lines of accumulated implementations with:

- Functions at module level mixed with classes
- Multiple naming conventions (`apply_*`, `_apply_*`, `*_bc_*`)
- Duplicate logic across uniform/mixed/1D/2D/nD variants
- Dead code paths
- Inconsistent parameter ordering

**Example of chaos:**
```python
# These all do similar things with slightly different signatures:
apply_boundary_conditions_2d(field, bc, domain_bounds, time, config, geometry)
apply_boundary_conditions_nd(field, bc, domain_bounds, time, config, geometry)
_apply_bc_uniform_2d(field, bc_type, value, domain_bounds, time, config)
_apply_bc_mixed_2d(field, bc, domain_bounds, time, config)
_apply_ghost_cells_1d(field, bc, dx, ...)
_apply_ghost_cells_2d(field, bc, dx, dy, ...)
# ... and more
```

**Recommended action:** Delete file after GhostBuffer migration complete.

---

### 4.8 INCOMPLETE: Corner Handling Strategy

**Problem:** Corners where different BC types meet are handled by simple averaging.

```python
# applicator_fdm.py - corner handling
def _apply_corner_values_nd(field, ...):
    # Average ghost values from adjacent faces
    corner_value = (ghost_from_face_1 + ghost_from_face_2) / 2
```

**Mathematical issue:** At a corner where Dirichlet (u=g₁) meets Neumann (∂u/∂n=g₂), averaging ghost values doesn't satisfy either condition exactly.

**Impact:** Local O(h) errors at corners, acceptable for most applications but limits high-accuracy work.

---

### 4.9 POLICY VIOLATION: hasattr() Usage (CLAUDE.md)

**Problem:** Multiple files violate CLAUDE.md's explicit prohibition on `hasattr()` for duck typing:

| File:Line | Usage | Violation Type |
|-----------|-------|----------------|
| `bc_coupling.py:239` | `hasattr(geometry, "domain_bounds")` | Duck typing |
| `types.py:474` | `hasattr(geometry, "point_to_indices")` | Duck typing |
| `types.py:489` | `hasattr(geometry, "bounds") and hasattr(geometry, "Nx_points")` | Duck typing |
| `conditions.py:1045` | `hasattr(geometry, "bounds")` | Duck typing |

**CLAUDE.md Rule Violated:**
```markdown
### hasattr() Usage Rules ⚠️ **CRITICAL**
**Prohibited**: Duck typing with hasattr()
**Goal**: Move from "guessing" object capabilities to explicit contracts (ABC/Protocol).
```

**Correct Pattern:**
```python
# BAD - Duck typing
if hasattr(geometry, "domain_bounds"):
    bounds = geometry.domain_bounds

# GOOD - Protocol-based
from mfg_pde.geometry.protocols import SupportsStructuredGrid

if isinstance(geometry, SupportsStructuredGrid):
    bounds = geometry.domain_bounds

# GOOD - Optional attribute with getattr
bounds = getattr(geometry, "domain_bounds", None)
```

**Impact:** Type safety compromised. Static analyzers cannot verify code correctness. JIT compilers cannot optimize.

---

### 4.10 ISSUE: Redundant try/except Instead of Protocol

**Problem:** FEM BC files comment "Use try/except instead of hasattr()" but this is still duck typing:

```python
# fem_bc_2d.py:107 - Still duck typing, just different syntax
# Issue #543: Use try/except instead of hasattr() for FEM mesh attributes
try:
    mesh.boundary_markers
except AttributeError:
    ...
```

**Better approach:** Define `FEMMesh` Protocol with required attributes, then use `isinstance()` check.

---

## 5. Technical Debt Inventory

### 5.1 NotImplementedError Locations (Verified)

| File:Line | Feature | Blocking | Priority |
|-----------|---------|----------|----------|
| `bc_coupling.py:250` | Adjoint BC for nD | Issue #624 | **High** |
| `dispatch.py:218` | FEM BC application | FEM solvers | Medium |
| `applicator_fdm.py:2341` | Robin corner nD | High-accuracy | Low |
| `applicator_fdm.py:2986` | High-order periodic/Robin | WENO schemes | Low |
| `types.py:515` | Region name without geometry | Issue #596 | Medium |
| `applicator_meshfree.py:260` | SDF region matching | GFDM | Low |

### 5.2 Deprecated APIs Still in Use (Verified)

| API | Deprecated | Replacement | Call Sites | Files |
|-----|------------|-------------|------------|-------|
| `apply_boundary_conditions_2d()` | v0.19.0 | `GhostBuffer` | 8 calls | dispatch.py, hjb_sl, fp_sl, fp_particle |
| `apply_boundary_conditions_nd()` | v0.19.0 | `GhostBuffer` | 2 calls | dispatch.py |
| `apply_boundary_conditions_1d()` | v0.19.0 | `GhostBuffer` | 2 calls | dispatch.py |
| `apply_boundary_conditions_3d()` | v0.19.0 | `GhostBuffer` | 1 call | dispatch.py |
| `mixed_bc()` | v0.18.0 | `BoundaryConditions(segments=)` | ~5 calls | Various |
| `bc_mode` parameter | v0.18.0 | `BCValueProvider` | **Removed in v0.18.0** | — |

**Deprecation warning locations in applicator_fdm.py:** Lines 155, 1100, 1592, 1634, 1707

### 5.3 CLAUDE.md Policy Violations

| File:Line | Violation | Correct Approach |
|-----------|-----------|------------------|
| `bc_coupling.py:239` | `hasattr(geometry, "domain_bounds")` | `getattr(geometry, "domain_bounds", None)` |
| `types.py:474` | `hasattr(geometry, "point_to_indices")` | Use Protocol |
| `types.py:489` | Multiple hasattr checks | Use Protocol |
| `conditions.py:1045` | `hasattr(geometry, "bounds")` | Use Protocol or getattr |

### 5.4 Inconsistent Naming

| Pattern | Examples | Recommendation |
|---------|----------|----------------|
| BC type enum | `NO_FLUX` vs `REFLECTING` | Document equivalence |
| Boundary names | `"left"` vs `"x_min"` vs `0` | Standardize on `"x_min"` |
| Method names | `get_bc_at_point` vs `get_bc_type_at_boundary` | Unify to `get_*` |
| Provider side names | `"left"/"right"` vs `"x_min"/"x_max"` | Normalize in provider |

### 5.5 TODO/FIXME in Codebase

| File:Line | Comment | Status |
|-----------|---------|--------|
| `applicator_fdm.py:1644` | TODO: Add optimized 3D implementation | Open |
| `bc_coupling.py:249` | TODO: Implement using geometry.get_gradient_operator() | Issue #624 |

---

## 6. Recommended Roadmap

### Phase 1: Complete Issue #625 (Current)
- [x] BCValueProvider protocol
- [x] AdjointConsistentProvider for 1D
- [x] Iterator integration
- [x] bc_mode deprecation
- [ ] Unit tests for providers
- [ ] Towel-on-beach validation

### Phase 2: Extend to nD (Issue #624)
- [ ] `compute_boundary_log_density_gradient_nd()` using geometry operators
- [ ] `AdjointConsistentProvider` dimension-aware compute
- [ ] 2D validation with known solution
- [ ] 3D smoke test

### Phase 3: GhostBuffer Completion (Issue #516)
- [ ] `MixedCalculator` for heterogeneous boundaries
- [ ] Pre-computed boundary masks
- [ ] Solver integration (one solver first)
- [ ] Performance benchmarks

### Phase 4: Code Quality (CLAUDE.md Compliance)
- [ ] Fix `hasattr()` violations in `types.py`, `bc_coupling.py`, `conditions.py`
- [ ] Define `SupportsStructuredGrid` Protocol for geometry duck typing
- [ ] Replace try/except duck typing in FEM BC files with Protocol checks
- [ ] Add type annotations to all public APIs

### Phase 5: Cleanup
- [ ] Delete `applicator_fdm.py` legacy functions (after Phase 3)
- [ ] Split `conditions.py` god class (BCSegmentCollection, BCValidator, etc.)
- [ ] Unify naming conventions (`x_min` standard)
- [ ] Remove deprecated parameter handling (`bc_mode`, etc.)

---

## 7. Implementation Reference

### 7.1 Provider Pattern (v0.18.0)

**Files:**
- `providers.py:65` - `BCValueProvider` protocol
- `providers.py:137` - `AdjointConsistentProvider`
- `types.py:523` - `BCSegment.get_value()` with state
- `conditions.py:892` - `has_providers()`, `with_resolved_providers()`
- `mfg_components.py:921` - `using_resolved_bc()` context manager
- `fixed_point_iterator.py:361` - BC resolution in coupling loop

**Usage:**
```python
from mfg_pde.geometry.boundary import (
    BoundaryConditions, BCSegment, BCType, AdjointConsistentProvider
)

bc = BoundaryConditions(
    segments=[
        BCSegment(
            name="left_ac",
            bc_type=BCType.ROBIN,
            alpha=0.0, beta=1.0,
            value=AdjointConsistentProvider(side="left", sigma=0.2),
            boundary="x_min",
        ),
        BCSegment(
            name="right_ac",
            bc_type=BCType.ROBIN,
            alpha=0.0, beta=1.0,
            value=AdjointConsistentProvider(side="right", sigma=0.2),
            boundary="x_max",
        ),
    ],
    dimension=1,
)
```

### 7.2 Ghost Cell Formulas

For cell-centered grids where boundary lies at cell face:

| BC Type | Ghost Value Formula |
|---------|---------------------|
| Dirichlet (u=g) | `u_g = 2g - u_i` |
| Neumann (∂u/∂n=g) | `u_g = u_i ± 2dx·g` |
| Robin (αu + β∂u/∂n=g) | `u_g = (g - u_i(α/2 - β/(2dx))) / (α/2 + β/(2dx))` |

### 7.3 Adjoint-Consistent BC Mathematics

At reflecting boundaries with stall point:
```
Standard Neumann: ∂U/∂n = 0  (wrong at stall points)
Adjoint-consistent: ∂U/∂n = -σ²/2 · ∂ln(m)/∂n  (correct for boundary stall)
```

Implemented as Robin BC with α=0, β=1:
```
0·U + 1·∂U/∂n = g  where  g = -σ²/2 · ∂ln(m)/∂n
```

**SCOPE LIMITATION (Issue #625 Validation):**

The adjoint-consistent BC formula is derived from the **zero-flux equilibrium condition** at reflecting boundaries:
```
J·n = 0  where  J = -σ²/2·∇m + m·α*  and  α* = -∇U
Rearranging:  ∂U/∂n = -σ²/2 · ∂ln(m)/∂n
```

**This derivation assumes the boundary IS the equilibrium point (stall).** When the stall point is in the domain interior:

- At stall point: optimal drift α* = 0, so zero-flux condition holds naturally
- At non-stall boundaries: optimal drift α* ≠ 0, agents have a preferred direction

**Applicability:**

| Configuration | AC BC | Neumann BC | Strict Adjoint |
|---------------|-------|------------|----------------|
| Boundary stall (x=0 or x=1) | **Best** | Moderate | Good |
| Interior stall (x=0.5) | **Wrong** | Moderate | **Best** |

**Validation results:**
- Boundary stall: AC BC error 1.36 vs Neumann 2.09 (1.54x better)
- Interior stall: AC BC error 5.88 vs Neumann 1.55 (3.8x worse!)
- Strict Adjoint Mode (L_FP = L_HJB^T): Best for all cases

**Recommendation:** Use `AdjointConsistentProvider` only for boundary stall problems. For interior stall or uncertain configurations, prefer Strict Adjoint Mode (Issue #622).

---

## 8. Metrics Summary

### 8.1 Code Volume
- **Total BC module**: 24 files, 15,163 lines
- **Largest file**: `applicator_fdm.py` (3,457 lines) - candidate for deletion
- **Best designed**: `applicator_base.py` (1,869 lines) - unused Topology/Calculator pattern
- **Newest**: `providers.py` (384 lines) - clean BCValueProvider implementation

### 8.2 Error Handling Complexity
- **103 `raise ValueError/TypeError`** across 16 files
- Heavy defensive programming - may mask deeper issues

### 8.3 Deprecation Status
- **6 deprecated APIs** still in active use
- **9 DeprecationWarning** calls in boundary module
- **Estimated migration effort**: Medium (well-designed replacement exists but unused)

---

## 9. Conclusion

The BC handling architecture has a **solid foundation** (data model, provider pattern, Topology/Calculator) but suffers from **incomplete migrations** and **accumulated technical debt**. Key observations:

### Positive
- ✅ `BCSegment` and `BoundaryConditions` data model is well-designed
- ✅ `BCValueProvider` pattern (Issue #625) demonstrates correct architectural direction
- ✅ `applicator_base.py` Topology/Calculator pattern is clean and ready for use
- ✅ Ghost cell formulas are mathematically correct and documented

### Negative
- ❌ 3,457 lines of deprecated code (`applicator_fdm.py`) still primary path
- ❌ Zero solvers use the new Topology/Calculator pattern
- ❌ hasattr() violations compromise type safety
- ❌ 1D-only implementations for adjoint-consistent BC

### Path Forward
1. **Complete what's started** - Wire up Topology/Calculator to one solver as proof-of-concept
2. **Fix policy violations** - hasattr() → Protocol pattern
3. **Extend to nD** - Implement `AdjointConsistentProvider` for 2D/3D
4. **Delete deprecated code** - Once replacements are validated
5. **Refactor bloated classes** - Split `conditions.py` god class

The Issue #625 provider pattern is a positive step that demonstrates the right architectural direction: **explicit intent, clean separation, testable components**.
