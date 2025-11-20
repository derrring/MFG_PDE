# WENO Dimensional Dispatch Consolidation Plan

## Current Structure (45 methods)

### Problem: Method Explosion
**Pattern**: Separate methods for each dimension and axis combination

**Current dimensional dispatch methods** (22 methods):
```python
# Top-level solve dispatch (4 methods)
solve_hjb_system()                    # Public API
_solve_hjb_system_1d()               # 1D-specific
_solve_hjb_system_2d()               # 2D-specific  
_solve_hjb_system_3d()               # 3D-specific
_solve_hjb_system_nd()               # nD-generic

# Per-axis stepping (9 methods)
_solve_hjb_step_direction_nd()       # nD generic axis step
_solve_hjb_step_1d_direction_adapted()
_solve_hjb_step_2d_x_direction()
_solve_hjb_step_2d_y_direction()
_solve_hjb_step_1d_y_adapted()
_solve_hjb_step_3d_x_direction()
_solve_hjb_step_3d_y_direction()
_solve_hjb_step_3d_z_direction()
_solve_hjb_step_1d_z_adapted()

# Per-axis dt computation (4 methods)
_compute_dt_stable_1d()
_compute_dt_stable_2d()
_compute_dt_stable_3d()
_compute_dt_stable_nd()

# Per-axis WENO reconstruction (3 methods)
_weno_reconstruction()               # x-axis (1D)
_weno_reconstruction_y_adapted()     # y-axis (2D)
_weno_reconstruction_z_adapted()     # z-axis (3D)

# Per-axis time integration (2 methods each for y and z)
_solve_hjb_tvd_rk3_y_adapted()
_solve_hjb_explicit_euler_y_adapted()
_solve_hjb_tvd_rk3_z_adapted()
_solve_hjb_explicit_euler_z_adapted()
```

---

## Proposed Consolidation

### Strategy: Axis-Parametric Methods

**Key insight**: Most "dimension-specific" methods differ only in:
1. Which axis to process (0=x, 1=y, 2=z)
2. How to slice/reshape arrays for 1D operations

**Consolidation targets** (reduce 22 → 8 methods):

```python
# Top-level solve (1 method instead of 5)
def solve_hjb_system(self, ...) -> np.ndarray:
    """Unified solver with dimension dispatch."""
    if self.dimension == 1:
        return self._solve_1d_loop(...)
    else:
        return self._solve_nd_dimensional_splitting(...)

# Dimensional splitting (1 method instead of 9)
def _solve_hjb_step_along_axis(
    self, 
    u: np.ndarray, 
    m: np.ndarray, 
    dt: float, 
    axis: int
) -> np.ndarray:
    """Apply WENO step along specified axis (0=x, 1=y, 2=z)."""
    # Generic implementation works for any axis
    # Uses np.moveaxis() to treat axis=0 uniformly
    
# Timestep computation (1 method instead of 4)
def _compute_dt_stable(self, u: np.ndarray, m: np.ndarray) -> float:
    """Compute stable dt for current dimension."""
    # CFL condition applies uniformly across dimensions
    # Just needs correct grid spacing (dx, dy, dz)

# WENO reconstruction (1 method instead of 3)
def _weno_reconstruct_along_axis(
    self, 
    u: np.ndarray, 
    axis: int = 0
) -> np.ndarray:
    """WENO reconstruction along specified axis."""
    # Use np.moveaxis() to make target axis first dimension
    # Apply standard WENO stencil
    # Move axis back
```

---

## Implementation Plan

### Phase 1: Core Consolidation (High Impact)

**Target**: Consolidate dimensional solve methods

**Before** (5 methods, ~400 lines):
```python
_solve_hjb_system_1d()     # 30 lines
_solve_hjb_system_2d()     # 40 lines
_solve_hjb_system_3d()     # 50 lines
_solve_hjb_system_nd()     # 60 lines
solve_hjb_system()         # 30 lines (dispatch)
```

**After** (1 method, ~100 lines):
```python
def solve_hjb_system(self, ...) -> np.ndarray:
    """Unified HJB system solver with dimensional splitting."""
    # Initialize solution array
    U_solution = np.zeros((self.Nt + 1, *self.shape))
    U_solution[-1, :] = U_final_condition_at_T
    
    # Backward time loop
    for n in range(self.Nt - 1, -1, -1):
        dt = self.dt
        u_current = U_solution[n + 1, :]
        m_current = M_density_evolution_from_FP[n + 1, :]
        
        # Dimensional splitting
        if self.splitting_method == "strang":
            u_half = u_current
            # Forward sweep
            for axis in range(self.dimension):
                dt_axis = dt / (2 * self.dimension)
                u_half = self._solve_hjb_step_along_axis(u_half, m_current, dt_axis, axis)
            # Backward sweep
            for axis in range(self.dimension - 1, -1, -1):
                dt_axis = dt / (2 * self.dimension)
                u_half = self._solve_hjb_step_along_axis(u_half, m_current, dt_axis, axis)
            U_solution[n, :] = u_half
        else:
            # Sequential splitting
            ...
    
    return U_solution
```

**Savings**: 5 methods → 1 method, ~300 lines saved

---

### Phase 2: Axis-Step Consolidation (Medium Impact)

**Target**: Consolidate per-axis stepping methods

**Before** (9 methods, ~200 lines):
```python
_solve_hjb_step_direction_nd()
_solve_hjb_step_1d_direction_adapted()
_solve_hjb_step_2d_x_direction()
_solve_hjb_step_2d_y_direction()
_solve_hjb_step_1d_y_adapted()
_solve_hjb_step_3d_x_direction()
_solve_hjb_step_3d_y_direction()
_solve_hjb_step_3d_z_direction()
_solve_hjb_step_1d_z_adapted()
```

**After** (1 method, ~40 lines):
```python
def _solve_hjb_step_along_axis(
    self, 
    u: np.ndarray, 
    m: np.ndarray, 
    dt: float, 
    axis: int
) -> np.ndarray:
    """Apply HJB step along specified axis using WENO."""
    # Move target axis to front for uniform processing
    u_moved = np.moveaxis(u, axis, 0)
    m_moved = np.moveaxis(m, axis, 0)
    
    # Apply 1D WENO operator along axis=0
    u_stepped = self._apply_1d_weno_operator(u_moved, m_moved, dt, axis)
    
    # Move axis back
    return np.moveaxis(u_stepped, 0, axis)
```

**Savings**: 9 methods → 1 method, ~160 lines saved

---

### Phase 3: DT and Reconstruction (Low Impact but Cleaner)

**Before** (7 methods, ~150 lines):
```python
# DT computation (4 methods)
_compute_dt_stable_1d()
_compute_dt_stable_2d()
_compute_dt_stable_3d()
_compute_dt_stable_nd()

# WENO reconstruction (3 methods)
_weno_reconstruction()
_weno_reconstruction_y_adapted()
_weno_reconstruction_z_adapted()
```

**After** (2 methods, ~60 lines):
```python
def _compute_dt_stable(self, u: np.ndarray, m: np.ndarray) -> float:
    """Compute stable timestep for current dimension."""
    # Generic CFL computation
    grid_spacings = [self.dx, self.dy, self.dz][:self.dimension]
    min_spacing = min(grid_spacings)
    return self.cfl_number * min_spacing

def _weno_reconstruct_along_axis(
    self, 
    u: np.ndarray, 
    axis: int = 0
) -> np.ndarray:
    """WENO reconstruction along specified axis."""
    u_moved = np.moveaxis(u, axis, 0)
    reconstructed = self._apply_weno_stencil(u_moved)
    return np.moveaxis(reconstructed, 0, axis)
```

**Savings**: 7 methods → 2 methods, ~90 lines saved

---

## Total Impact

| Category | Before | After | Methods Saved | Lines Saved |
|----------|--------|-------|---------------|-------------|
| Solve dispatch | 5 methods | 1 method | -4 | ~300 |
| Axis stepping | 9 methods | 1 method | -8 | ~160 |
| DT + reconstruction | 7 methods | 2 methods | -5 | ~90 |
| **Total** | **21 methods** | **4 methods** | **-17** | **~550 lines** |

**Overall reduction**: 45 methods → 28 methods (37% reduction)
**File size**: 1176 lines → ~650 lines (45% reduction)

---

## Benefits

1. **Maintainability**: Single implementation for all dimensions
2. **Testability**: Test axis-parametric methods once instead of per-dimension
3. **Extensibility**: Adding 4D+ support requires no new methods
4. **Readability**: Clearer algorithmic structure

---

## Risks & Mitigation

### Risk 1: Performance Regression
**Concern**: `np.moveaxis()` overhead in tight loops

**Mitigation**:
- Profile before/after with benchmark suite
- WENO is already compute-bound (reconstruction dominates)
- Array movement is O(1) view operation, not copy

### Risk 2: Breaking Existing Code
**Concern**: Public API change

**Mitigation**:
- Keep `solve_hjb_system()` signature identical (public API)
- Only refactor private `_*` methods (internal)
- Add deprecation warnings if needed

### Risk 3: Dimension-Specific Edge Cases
**Concern**: 2D/3D may have special logic

**Mitigation**:
- Carefully review existing dimension-specific methods
- Preserve any legitimate special cases
- Add comprehensive tests for all dimensions

---

## Implementation Steps

1. Create feature branch: `refactor/weno-consolidate-dimensional-dispatch`
2. Implement unified `_solve_hjb_step_along_axis()`
3. Refactor `solve_hjb_system()` to use unified method
4. Run existing tests to verify no regression
5. Remove old dimensional methods one-by-one
6. Update smoke test to test all dimensions
7. Performance benchmark comparison
8. Create PR with before/after metrics

---

## Success Criteria

- [ ] All existing unit tests pass
- [ ] Smoke tests pass for 1D, 2D, 3D
- [ ] Performance within 5% of original
- [ ] Method count: 45 → ~28 (≥35% reduction)
- [ ] File size: 1176 → ~650 lines (≥40% reduction)
- [ ] Code coverage maintained or improved
