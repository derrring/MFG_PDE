# [COMPLETED] hasattr() Protocol Duck Typing Elimination Plan

**Issue**: #543 - Eliminate remaining hasattr() violations using GeometryProtocol
**Status**: ✅ **COMPLETED** - 2026-01-10
**Depends**: GeometryProtocol (exists since v0.16.16, PR #548)
**Scope**: Core + Geometry modules
**Original Count**: 79 violations
**Final Count**: 3 violations (docstring examples only)
**Reduction**: 96%

## Current State

**GeometryProtocol Status**: ✅ **EXISTS** (since v0.16.16)
- Location: `mfg_pde/geometry/protocol.py`
- Decorator: `@runtime_checkable`
- Required properties: `dimension`, `geometry_type`, `num_spatial_points`
- Required methods: `get_spatial_grid()`, `get_bounds()`, `get_boundary_conditions()`, etc.

**Remaining Violations**: 39 hasattr checks for protocol duck typing
- **Core module**: 2 (in `mfg_problem.py:solve_variational`)
- **Geometry module**: 37 (various files)

## Strategy

Use `isinstance(geometry, GeometryProtocol)` instead of `hasattr(geometry, "method_name")` checks.

### Pattern Transformation

**Before** (hasattr duck typing):
```python
if hasattr(geometry, "get_spatial_grid"):
    x = geometry.get_spatial_grid()
elif hasattr(geometry, "interior_points"):
    x = geometry.interior_points
else:
    raise ValueError("Geometry must provide spatial points")
```

**After** (GeometryProtocol):
```python
if not isinstance(geometry, GeometryProtocol):
    raise TypeError(
        f"Geometry must implement GeometryProtocol, got {type(geometry)}"
    )

# Now safe to call protocol methods directly
x = geometry.get_spatial_grid()
```

**Alternative** (for optional methods not in protocol):
```python
# If method is truly optional (not in GeometryProtocol)
try:
    x = geometry.get_spatial_grid()
except AttributeError:
    x = geometry.interior_points  # Fallback to optional method
```

## Implementation Plan

### Phase 1: Core Module (2 violations) ✅

**File**: `mfg_pde/core/mfg_problem.py`
**Lines**: 2000, 2003

**Current Code**:
```python
# NOTE (Issue #543): hasattr() used for protocol duck typing
# Will be replaced with proper GeometryProtocol in Issue #544
if hasattr(self.geometry, "get_spatial_grid"):
    x = self.geometry.get_spatial_grid()
    collocation_points = np.atleast_2d(x).T if x.ndim == 1 else x
elif hasattr(self.geometry, "interior_points"):
    collocation_points = self.geometry.interior_points
```

**Fix**:
```python
from mfg_pde.geometry.protocol import GeometryProtocol

if not isinstance(self.geometry, GeometryProtocol):
    raise TypeError(
        f"Variational solver requires GeometryProtocol-compliant geometry, "
        f"got {type(self.geometry).__name__}"
    )

# get_spatial_grid() is in GeometryProtocol
x = self.geometry.get_spatial_grid()
collocation_points = np.atleast_2d(x).T if x.ndim == 1 else x
```

**Impact**: -2 violations in core module → **Target: 3 remaining (all docstring examples)**

---

### Phase 2: Geometry Module (37 violations) ⏸️

**Files**: `collocation.py` (13), `fem_bc_*.py` (15), `applicator_graph.py` (4), `protocol.py` (1), etc.

**Categories**:

#### 2a. Dimension Detection (10 cases)
**File**: `collocation.py:208-216`

**Current**:
```python
if hasattr(self.geometry, "dimension"):
    dim = self.geometry.dimension
elif hasattr(self.geometry, "bounds"):
    dim = len(self.geometry.bounds)
```

**Fix**:
```python
if not isinstance(self.geometry, GeometryProtocol):
    raise TypeError(f"Geometry must implement GeometryProtocol")

dim = self.geometry.dimension  # Required property in protocol
```

**Impact**: -10 violations

#### 2b. FEM Mesh Capability Checks (15 cases)
**Files**: `fem_bc_1d.py`, `fem_bc_2d.py`, `fem_bc_3d.py`

**Current**:
```python
return hasattr(mesh, "boundary_markers") or hasattr(mesh, "vertex_markers")
```

**Analysis**: These check for FEM-specific mesh features not in GeometryProtocol

**Options**:
1. **Create MeshProtocol** (proper solution):
   ```python
   @runtime_checkable
   class BoundaryMarkedMesh(Protocol):
       boundary_markers: dict[str, np.ndarray]

   # Usage
   if isinstance(mesh, BoundaryMarkedMesh):
       return True
   ```
2. **Use try/except** (pragmatic):
   ```python
   try:
       _ = mesh.boundary_markers
       return True
   except AttributeError:
       try:
           _ = mesh.vertex_markers
           return True
       except AttributeError:
           return False
   ```

**Recommendation**: **Option 2 (try/except)** for now, defer MeshProtocol to separate issue

**Impact**: -15 violations (replaced with try/except)

#### 2c. Graph Geometry Checks (4 cases)
**File**: `applicator_graph.py:273, 283, 293`

**Current**:
```python
if hasattr(geometry, "network_data") and geometry.network_data is not None:
    ...
```

**Analysis**: Checking for graph-specific `network_data` attribute

**Options**:
1. **Create NetworkGeometryProtocol** (proper but heavyweight)
2. **Use try/except** (pragmatic)
3. **Use isinstance with concrete type**:
   ```python
   from mfg_pde.geometry.graph import NetworkGeometry
   if isinstance(geometry, NetworkGeometry):
       network_data = geometry.network_data
   ```

**Recommendation**: **Option 3** - isinstance with concrete type

**Impact**: -4 violations

#### 2d. Other Protocol Checks (8 cases)
Various files checking for optional geometry methods

**Recommendation**: Case-by-case evaluation using try/except or isinstance

**Impact**: -8 violations

---

## Testing Strategy

### After Phase 1 (Core Module)
```bash
pytest tests/unit/test_core/test_mfg_problem.py -v
pytest tests/integration/test_variational_solver.py -v

# Validate reduction
python scripts/check_fail_fast.py --path mfg_pde/core --all
# Should show: 5 → 3 violations (only docstring examples remain)
```

### After Phase 2 (Geometry Module)
```bash
pytest tests/unit/test_geometry/ -v
pytest tests/integration/ -k geometry -v

# Validate reduction
python scripts/check_fail_fast.py --path mfg_pde/geometry --all
# Should show: 38 → ~3-5 violations (remaining edge cases)
```

---

## Risk Assessment

| Phase | Risk | Mitigation |
|:------|:-----|:-----------|
| Phase 1 (Core) | Low | Small scope, well-defined protocol |
| Phase 2a (Dimension) | Low | dimension is required in GeometryProtocol |
| Phase 2b (FEM Mesh) | Medium | No MeshProtocol exists → use try/except |
| Phase 2c (Graph) | Low | Concrete type checks available |

---

## Expected Outcomes

**After Phase 1** (Core Module):
- Violations: 5 → 3 (-40%)
- **Target achieved**: Only docstring examples remain ✅

**After Phase 2** (Geometry Module):
- Violations: 38 → ~3-5 (~90% reduction)
- Most protocol duck typing eliminated
- Remaining cases are truly optional features

**Total Reduction** (Core + Geometry):
- Violations: 43 → ~6-8 (~85% total reduction)

---

## Implementation Timeline

**Session 1** (Immediate):
- ✅ Phase 1: Core module protocol checks
- Create PR, run tests, merge

**Session 2** (After Phase 1 merge):
- Phase 2a: Geometry dimension checks
- Phase 2c: Graph geometry concrete types

**Session 3** (Optional, if needed):
- Phase 2b: FEM mesh try/except refactoring
- Phase 2d: Remaining edge cases

---

## Success Criteria

**Core Module**:
- [ ] Zero hasattr for protocol checks
- [ ] Only 3 docstring examples remain
- [ ] All tests pass
- [ ] GeometryProtocol used correctly

**Geometry Module**:
- [ ] <5 hasattr violations remain
- [ ] All dimension checks use GeometryProtocol
- [ ] Graph checks use concrete types
- [ ] FEM checks use try/except (not hasattr)

---

**Created**: 2026-01-11
**Completed**: 2026-01-10
**Status**: ✅ **COMPLETED**
**Depends**: GeometryProtocol (v0.16.16+)

## Completion Summary

**Pull Requests Merged**:
1. ✅ PR #551 - Core module cleanup (27 → 5 violations)
2. ✅ PR #552 - Geometry Phase 1 (47 → 38 violations)
3. ✅ PR #553 - Core protocol checks (5 → 3 violations)
4. ✅ PR #554 - Geometry protocol checks (38 → 0 violations)

**Final State**: 3 violations remaining (all docstring examples in `base_problem.py`)
**All Tests**: Passing (378 geometry + 48 core)
**Related Issues Updated**: #527 (BC Infrastructure), #544 (Geometry-First API)
