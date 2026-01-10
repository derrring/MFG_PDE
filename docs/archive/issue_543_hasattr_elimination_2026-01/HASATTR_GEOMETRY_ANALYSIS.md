# hasattr() Analysis: mfg_pde/geometry Module

**Issue**: #543 - Eliminate hasattr() duck typing
**Scope**: Geometry module
**Count**: 47 violations
**Priority**: High (after core module completion)
**Status**: Analysis phase

## Executive Summary

**Total Violations**: 47 hasattr checks in geometry module
**Assessment**: **80% require GeometryProtocol** (blocked by #544), **20% can be fixed immediately**

**Key Finding**: Most geometry hasattr violations are **legitimate protocol duck typing** for checking geometry capabilities. These should be deferred to Issue #544 (Geometry-First API) when proper Protocol definitions are created.

**Recommended Approach**:
1. Fix immediate wins (lazy init, legacy fallback) → **~10 violations**
2. Defer protocol checks to Issue #544 → **~37 violations**

---

## Violation Categories

### 1. Geometry Protocol Duck Typing (37 cases) ⏸️ **DEFER TO #544**

**Locations**: `collocation.py`, `protocol.py`, `dispatch.py`, `operators/projection.py`

```python
# collocation.py:208-218 (10 cases)
if hasattr(self.geometry, "dimension"):
    dim = self.geometry.dimension
elif hasattr(self.geometry, "bounds"):
    dim = len(self.geometry.bounds)

if hasattr(self.geometry, "bounds"):
    bounds = self.geometry.bounds
elif hasattr(self.geometry, "get_bounds"):
    bounds = self.geometry.get_bounds()
```

**Problem**: Checking which methods/attributes different geometry types provide

**Root Cause**: No formal GeometryProtocol defining required vs optional methods

**Why Defer**:
- Requires architectural decision on protocol design
- Affects broader codebase (not just geometry module)
- Issue #544 specifically addresses geometry protocol definition
- Premature to fix without protocol design

**Proper Solution** (after #544):
```python
# Define in mfg_pde/geometry/protocol.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class DimensionalGeometry(Protocol):
    @property
    def dimension(self) -> int: ...

@runtime_checkable
class BoundedGeometry(Protocol):
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

# Usage
if isinstance(self.geometry, DimensionalGeometry):
    dim = self.geometry.dimension
elif isinstance(self.geometry, BoundedGeometry):
    bounds = self.geometry.get_bounds()
    dim = len(bounds[0])
```

**Temporary Fix** (optional, for consistency with core module):
```python
# Use contextlib.suppress() like in core module
dim = None
with contextlib.suppress(AttributeError):
    dim = self.geometry.dimension
if dim is None:
    with contextlib.suppress(AttributeError):
        bounds = self.geometry.get_bounds()
        dim = len(bounds[0])
```

**Decision**: ⏸️ **DEFER TO #544** - Protocol design is prerequisite

---

### 2. FEM Mesh Capability Checks (15 cases) ⏸️ **DEFER TO #544**

**Locations**: `fem_bc_1d.py`, `fem_bc_2d.py`, `fem_bc_3d.py`

```python
# fem_bc_2d.py:107
return hasattr(mesh, "boundary_markers") or hasattr(mesh, "vertex_markers")

# fem_bc_2d.py:156
return hasattr(mesh, "boundary_normals") or hasattr(mesh, "edge_normals")

# fem_bc_2d.py:419
if hasattr(mesh, "boundary_edges") and mesh.boundary_edges is not None:
    ...
```

**Problem**: Checking if FEM mesh has specific boundary support capabilities

**Root Cause**: Different mesh types (FEM 1D/2D/3D) have different features

**Why Defer**:
- Requires MeshProtocol definition (part of #544)
- Multiple mesh types with different capabilities
- Architectural decision on mesh interface

**Proper Solution** (after #544):
```python
@runtime_checkable
class BoundaryMarkedMesh(Protocol):
    boundary_markers: dict[str, np.ndarray]

@runtime_checkable
class EdgeMesh(Protocol):
    boundary_edges: np.ndarray | None

# Usage
def has_boundary_markers(mesh) -> bool:
    return isinstance(mesh, BoundaryMarkedMesh)
```

**Decision**: ⏸️ **DEFER TO #544** - Mesh protocol design needed

---

### 3. Lazy Initialization (2 cases) ✅ **FIX IMMEDIATELY**

**Location**: `mesh_3d.py:637, 662`

```python
# mesh_3d.py:637
if not hasattr(self, "_mesh_data") or self._mesh_data is None:
    self._compute_mesh_data()
```

**Problem**: Defensive check assumes attribute might not exist

**Solution**: Initialize explicitly in `__init__`
```python
# In Mesh3D.__init__
self._mesh_data = None

# Later (lines 637, 662)
if self._mesh_data is None:
    self._compute_mesh_data()
```

**Risk**: Low - straightforward explicit initialization

**Impact**: -2 violations

---

### 4. Legacy Attribute Fallback (5 cases) ✅ **FIX IMMEDIATELY**

**Locations**: `applicator_fdm.py`, `applicator_graph.py`, `dispatch.py`

```python
# applicator_fdm.py:1859-1860
alpha = seg.alpha if hasattr(seg, "alpha") else 1.0
beta = seg.beta if hasattr(seg, "beta") else 0.0

# applicator_graph.py:796-798
if hasattr(maze_geometry, "num_nodes"):
    num_nodes = maze_geometry.num_nodes
elif hasattr(maze_geometry, "n_nodes"):
    num_nodes = maze_geometry.n_nodes

# dispatch.py:95-97
if hasattr(geometry, "num_nodes"):
    ...
elif hasattr(geometry, "num_spatial_points"):
    ...
```

**Problem**: Supporting multiple attribute names for same concept

**Solution**: Use `getattr()` with default
```python
# applicator_fdm.py (if alpha/beta are truly optional)
alpha = getattr(seg, "alpha", 1.0)
beta = getattr(seg, "beta", 0.0)

# For attribute naming inconsistency (num_nodes vs n_nodes)
num_nodes = getattr(geometry, "num_nodes", None) or getattr(geometry, "n_nodes", None)
if num_nodes is None:
    raise AttributeError("Geometry must have 'num_nodes' or 'n_nodes' attribute")
```

**Alternative** (for mandatory attributes):
```python
# If alpha/beta are required, use try-except
try:
    alpha, beta = seg.alpha, seg.beta
except AttributeError as e:
    raise AttributeError(
        f"BCSegment must have 'alpha' and 'beta' attributes for Robin BC. "
        f"Missing: {e}"
    )
```

**Risk**: Low - simple attribute access normalization

**Impact**: -5 violations

---

### 5. Optional Attribute Access (3 cases) ⚠️ **EVALUATE**

**Location**: `base.py:1804`, `applicator_fdm.py:2425`, `dispatch.py:270`

```python
# base.py:1804
if hasattr(self, "node_positions") and self.node_positions is not None:
    ...

# applicator_fdm.py:2425
v = bc.left_value if hasattr(bc, "left_value") and bc.left_value is not None else 0.0

# dispatch.py:270
bc_type = boundary_conditions.default_bc if hasattr(boundary_conditions, "default_bc") else None
```

**Problem**: Checking for optional attributes that may not exist

**Option 1: contextlib.suppress()** (CLAUDE.md recommended):
```python
# base.py:1804
node_pos = None
with contextlib.suppress(AttributeError):
    node_pos = self.node_positions

if node_pos is not None:
    ...
```

**Option 2: getattr() with default**:
```python
v = getattr(bc, "left_value", 0.0) or 0.0
bc_type = getattr(boundary_conditions, "default_bc", None)
```

**Recommendation**: Use getattr() for cleaner one-liners

**Risk**: Low - straightforward replacement

**Impact**: -3 violations

---

## Summary by Category

| Category | Count | Action | Blocker |
|:---------|:------|:-------|:--------|
| Geometry protocol duck typing | 37 | **DEFER** | Issue #544 |
| FEM mesh capability checks | 15 | **DEFER** | Issue #544 |
| Lazy initialization | 2 | **FIX** | None |
| Legacy attribute fallback | 5 | **FIX** | None |
| Optional attribute access | 3 | **FIX** | None |
| **Total** | **62*** | | |

*Note: Some violations counted in multiple categories (e.g., FEM checks are also protocol typing)

**Net Total**: 47 unique violations

---

## Implementation Plan

### Phase 1: Quick Wins (Immediate) ✅

**Target**: -10 violations (21% reduction)
**Risk**: Low
**Blocks**: None

**Steps**:
1. Fix lazy init in `mesh_3d.py` (-2)
2. Fix legacy fallback in `applicator_fdm.py`, `applicator_graph.py`, `dispatch.py` (-5)
3. Fix optional attribute access with `getattr()` (-3)

**Estimated Time**: 30-45 minutes

---

### Phase 2: Protocol Duck Typing (After #544) ⏸️

**Target**: -37 violations (79% reduction of remaining)
**Blocks**: Issue #544 (Geometry-First API)

**Dependencies**:
1. Define GeometryProtocol with required/optional methods
2. Define MeshProtocol hierarchy (FEM 1D/2D/3D)
3. Update geometry classes to implement protocols
4. Replace hasattr with isinstance() checks

**Estimated Time**: After #544 completion

---

## Detailed File Analysis

### `collocation.py` (13 violations)

**Lines 208-216**: Geometry dimension detection
- **Type**: Protocol duck typing
- **Action**: **DEFER to #544**
- **Reason**: Needs DimensionalGeometry protocol

**Lines 456-468**: Bounds/bounding box detection
- **Type**: Protocol duck typing
- **Action**: **DEFER to #544**
- **Reason**: Needs BoundedGeometry protocol

**Line 901**: Signed distance gradient check
- **Type**: Protocol duck typing
- **Action**: **DEFER to #544**
- **Reason**: Needs ImplicitGeometry protocol

**Lines 961-978**: Geometry type detection (grid vs mesh vs implicit)
- **Type**: Protocol duck typing
- **Action**: **DEFER to #544**
- **Reason**: Fundamental protocol design needed

---

### `applicator_fdm.py` (3 violations)

**Lines 1859-1860**: Robin BC coefficients (alpha, beta)
- **Type**: Legacy attribute fallback
- **Action**: ✅ **FIX** with `getattr()`
- **Fix**:
  ```python
  alpha = getattr(seg, "alpha", 1.0)
  beta = getattr(seg, "beta", 0.0)
  ```

**Line 2425**: Legacy BC left_value
- **Type**: Optional attribute access
- **Action**: ✅ **FIX** with `getattr()`
- **Fix**:
  ```python
  v = getattr(bc, "left_value", None)
  v = v if v is not None else 0.0
  ```

---

### `mesh_3d.py` (2 violations)

**Lines 637, 662**: Lazy _mesh_data initialization
- **Type**: Lazy initialization
- **Action**: ✅ **FIX** - initialize in `__init__`
- **Fix**:
  ```python
  # In __init__
  self._mesh_data = None

  # Lines 637, 662
  if self._mesh_data is None:
      self._compute_mesh_data()
  ```

---

### `applicator_graph.py` (4 violations)

**Lines 273, 283, 293**: network_data checks
- **Type**: Protocol duck typing
- **Action**: **DEFER to #544**
- **Reason**: Needs NetworkGeometry protocol

**Lines 796-798**: num_nodes vs n_nodes fallback
- **Type**: Legacy attribute fallback
- **Action**: ✅ **FIX** with getattr normalization
- **Fix**:
  ```python
  num_nodes = getattr(maze_geometry, "num_nodes", None) or getattr(maze_geometry, "n_nodes", None)
  if num_nodes is None:
      raise AttributeError("Geometry must have 'num_nodes' or 'n_nodes'")
  ```

---

### FEM BC Files (15 violations combined)

**fem_bc_1d.py**, **fem_bc_2d.py**, **fem_bc_3d.py**:
- All checking for mesh boundary support capabilities
- **Action**: **DEFER to #544**
- **Reason**: Requires MeshProtocol hierarchy design

---

## Testing Strategy

### Phase 1 Testing (Quick Wins)
```bash
# After fixing lazy init and legacy fallback
pytest tests/unit/test_geometry/ -v
pytest tests/integration/test_geometry_integration.py -v

# Validate reduction
python scripts/check_fail_fast.py --path mfg_pde/geometry --all
# Should show: 47 → 37 violations (-10)
```

### Phase 2 Testing (After #544)
- Full geometry test suite
- Protocol isinstance() checks
- Mesh capability verification

---

## Risk Assessment

| Phase | Risk | Mitigation |
|:------|:-----|:-----------|
| Phase 1 (Quick Wins) | Low | Straightforward getattr() and explicit init |
| Phase 2 (Protocol) | High | Requires #544 design, broader impact |

---

## Recommendations

### Immediate (This Session) ✅
1. Execute Phase 1: Fix 10 quick wins
   - mesh_3d.py lazy init (-2)
   - applicator_fdm.py legacy fallback (-3)
   - applicator_graph.py, dispatch.py, base.py (-5)
2. Create branch: `refactor/eliminate-hasattr-geometry`
3. Run tests, commit, create PR

### Short Term (This Week)
- None - wait for Issue #544 completion

### Medium Term (After #544) ⏸️
1. Define GeometryProtocol hierarchy
2. Define MeshProtocol hierarchy
3. Replace hasattr with isinstance() checks (Phase 2)
4. Achieve target: <5 hasattr in geometry module

---

## Expected Outcomes

**After Phase 1**:
- Violations: 47 → 37 (-21%)
- All easy wins captured
- Remaining violations documented as deferred to #544

**After Phase 2** (requires #544):
- Violations: 37 → <5 (~90% total reduction)
- Proper protocol-based architecture
- Type-safe geometry handling

---

**Created**: 2026-01-11
**Author**: Issue #543 Continuation
**Status**: Analysis Complete, Phase 1 Ready for Implementation
