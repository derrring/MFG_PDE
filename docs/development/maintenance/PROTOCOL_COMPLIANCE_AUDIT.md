# MFGProblem Protocol Compliance Audit

**Date**: 2025-11-09
**Purpose**: Document current state vs MFGProblemProtocol requirements
**Status**: Phase 1 - Gap Analysis

---

## Protocol Requirements vs Current Implementation

### ✅ **Already Compliant Attributes**

| Protocol Attribute | Current Implementation | Notes |
|:-------------------|:-----------------------|:------|
| `dimension` | ✅ `self.dimension` | Set in all modes (1D, nD, geometry, network*) |
| `spatial_bounds` | ✅ `self.spatial_bounds` | Set in nD mode, needs unification |
| `spatial_discretization` | ✅ `self.spatial_discretization` | Set in nD mode, needs unification |
| `T` | ✅ `self.T` | Universally set |
| `Nt` | ✅ `self.Nt` | Universally set |
| `sigma` | ✅ `self.sigma` | Universally set |
| Methods | ✅ All present | `hamiltonian()`, `terminal_cost()`, etc. |

### ⚠️ **Naming Inconsistencies**

| Protocol Name | Current Name | Mode | Fix Needed |
|:--------------|:-------------|:-----|:-----------|
| `dt` | `Dt` (uppercase) | All | Add `dt` property aliasing `Dt` |
| `time_grid` | `tSpace` | All | Add `time_grid` property aliasing `tSpace` |
| `spatial_grid` | `xSpace` (1D), `_grid` (nD) | Mixed | Add unified `spatial_grid` property |

### ❌ **Missing Properties**

| Protocol Attribute | Status | Implementation Strategy |
|:-------------------|:-------|:------------------------|
| `grid_shape` | Missing | Add property: `(N₀, N₁, ...)` tuple |
| `grid_spacing` | Missing | Add property: `[Δx₀, Δx₁, ...]` list |

---

## Detailed Gaps by Problem Mode

### 1. **1D Mode** (Legacy: `xmin`, `xmax`, `Nx`)

**Current State:**
```python
problem = MFGProblem(xmin=0, xmax=1, Nx=100, T=1, Nt=50, sigma=0.1)
# Sets:
# - self.dimension = 1 ✅
# - self.Nx = 100
# - self.Dx = 0.01
# - self.xSpace = np.linspace(...)
# - self.Dt = 0.02
# - self.tSpace = np.linspace(...)
```

**Missing for Protocol:**
- `spatial_bounds` → should be `[(0, 1)]`
- `spatial_discretization` → should be `[100]`
- `grid_shape` → should be `(100,)`
- `grid_spacing` → should be `[0.01]`
- `dt` property → alias for `Dt`
- `time_grid` property → alias for `tSpace`
- `spatial_grid` property → alias for `xSpace`

### 2. **nD Mode** (Modern: `spatial_bounds`, `spatial_discretization`)

**Current State:**
```python
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    T=1, Nt=100, sigma=0.1
)
# Sets:
# - self.dimension = 2 ✅
# - self.spatial_bounds = [(0, 1), (0, 1)] ✅
# - self.spatial_discretization = [50, 50] ✅
# - self.spatial_shape = (50, 50) ⚠️ (should be grid_shape)
# - self._grid = TensorProductGrid(...)
# - self.Dt = 0.01
# - self.tSpace = np.linspace(...)
```

**Missing for Protocol:**
- `grid_shape` → currently `spatial_shape`, need property
- `grid_spacing` → need to compute from bounds/discretization
- `dt` property → alias for `Dt`
- `time_grid` property → alias for `tSpace`
- `spatial_grid` property → delegate to `_grid.get_spatial_grid()`

### 3. **Geometry Mode** (GeometryProtocol)

**Current State:**
```python
geometry = Domain1D(...)
problem = MFGProblem(geometry=geometry)
# Sets:
# - self.dimension = geometry.dimension ✅
# - Most properties delegated to geometry
```

**Missing for Protocol:**
- All properties should delegate to geometry where applicable
- Need consistent interface

### 4. **Network Mode**

**Current State:**
```python
network = NetworkGraph(...)
problem = NetworkMFGProblem(network=network)
# Sets:
# - self.dimension = "network" ⚠️ (should be int or special handling)
```

**Issue**: `dimension = "network"` violates Protocol (expects `int`)
**Solution**: Either use `dimension = 0` or make network inherit different protocol

---

## Implementation Strategy: Add Properties

### Property 1: `grid_shape`

```python
@property
def grid_shape(self) -> tuple[int, ...]:
    """
    Shape of spatial grid (N₀, N₁, ...).

    Returns tuple for dimension-agnostic code.
    """
    if hasattr(self, "spatial_shape"):  # nD mode
        return self.spatial_shape
    elif hasattr(self, "Nx") and self.Nx is not None:  # 1D mode
        return (self.Nx,)
    elif hasattr(self, "geometry") and self.geometry is not None:
        # Geometry mode - extract from geometry
        config = self.geometry.get_problem_config()
        return tuple(config.get("spatial_discretization", []))
    else:
        raise AttributeError("Cannot determine grid_shape")
```

### Property 2: `grid_spacing`

```python
@property
def grid_spacing(self) -> list[float]:
    """
    Grid spacing [Δx₀, Δx₁, ...] for each dimension.

    Returns list for dimension-agnostic code.
    """
    if hasattr(self, "Dx") and self.Dx is not None:  # 1D mode
        return [self.Dx]
    elif hasattr(self, "_grid"):  # nD mode with TensorProductGrid
        # Compute from bounds and discretization
        spacing = []
        for (xmin, xmax), N in zip(self.spatial_bounds, self.spatial_discretization):
            dx = (xmax - xmin) / N if N > 0 else 0.0
            spacing.append(dx)
        return spacing
    elif hasattr(self, "geometry") and self.geometry is not None:
        # Geometry mode - delegate to geometry
        # TensorProductGrid doesn't have grid_spacing yet, need to add
        # For now, compute manually
        config = self.geometry.get_problem_config()
        bounds = config.get("spatial_bounds", [])
        disc = config.get("spatial_discretization", [])
        return [(xmax - xmin) / N for (xmin, xmax), N in zip(bounds, disc)]
    else:
        raise AttributeError("Cannot determine grid_spacing")
```

### Property 3: `dt` (lowercase alias)

```python
@property
def dt(self) -> float:
    """Time step size Δt (alias for Dt)."""
    return self.Dt
```

### Property 4: `time_grid` (alias)

```python
@property
def time_grid(self) -> np.ndarray:
    """Array of time points [t₀, t₁, ..., t_Nt] (alias for tSpace)."""
    return self.tSpace
```

### Property 5: `spatial_grid` (unified)

```python
@property
def spatial_grid(self) -> np.ndarray | list[np.ndarray]:
    """
    Spatial coordinate arrays.

    Returns:
        - 1D: np.ndarray of shape (Nx+1,)
        - nD: List of 1D coordinate arrays or flattened grid points
    """
    if hasattr(self, "xSpace") and self.xSpace is not None:  # 1D mode
        return self.xSpace
    elif hasattr(self, "_grid") and self._grid is not None:  # nD mode
        return self._grid.get_spatial_grid()
    elif hasattr(self, "geometry") and self.geometry is not None:
        return self.geometry.get_spatial_grid()
    else:
        raise AttributeError("Cannot determine spatial_grid")
```

---

## Priority Order for Phase 1 Completion

1. **HIGH: Add simple property aliases** (5 min)
   - `dt` → `Dt`
   - `time_grid` → `tSpace`

2. **MEDIUM: Add `grid_shape` property** (10 min)
   - Handle 1D, nD, geometry modes

3. **MEDIUM: Add `grid_spacing` property** (15 min)
   - Compute from bounds/discretization

4. **LOW: Add `spatial_grid` property** (10 min)
   - Unify xSpace/_grid/geometry

5. **TEST: Create Protocol compliance test** (15 min)
   - Test 1D, 2D, 3D problems
   - Verify `isinstance(problem, MFGProblemProtocol)`

**Total Estimated Time**: ~1 hour

---

## Network Mode Special Case

**Issue**: `NetworkMFGProblem` has `dimension = "network"` (string, not int)

**Options**:
1. Create separate `NetworkMFGProblemProtocol`
2. Use `dimension = 0` for networks
3. Use `dimension = num_nodes` for networks
4. Skip Protocol for networks (not grid-based anyway)

**Recommendation**: Option 4 - Networks are fundamentally different, don't force Protocol compliance

---

## Next Concrete Steps

1. Add properties to `mfg_pde/core/mfg_problem.py`
2. Create `tests/unit/test_core/test_protocol_compliance.py`
3. Run tests
4. Update plan with actual completion
5. Commit as Phase 1 completion

---

## Version Strategy

**Agreement**: Not rushing to v1.0, can use v0.11.x

**Proposed Versioning**:
- **v0.11.0**: Phase 1 complete (Protocol + properties)
- **v0.12.0**: Phase 2 complete (Full dimension-agnostic properties)
- **v0.13.0**: Phase 3 complete (Solver updates)
- **v0.14.0**: Phase 4 complete (Deprecations)
- **v0.15.0**: Phase 5 complete (Documentation)
- **v1.0.0**: Phase 6 complete (Performance validation + release)

This gives us ~6 minor versions to incrementally evolve, very safe!
