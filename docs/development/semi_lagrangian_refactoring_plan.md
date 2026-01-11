# Semi-Lagrangian Solver Refactoring Plan

**Issue**: #545 - Refactor solver BC handling: Mixin Hell → Composition + Common Interface
**Date**: 2026-01-11
**Status**: Planning Phase

## Overview

Eliminate all 16 hasattr checks from `hjb_semi_lagrangian.py` and apply the BoundaryHandler protocol, following the same pattern successfully used for FDM solver.

## Current State Analysis

### hasattr Usage (16 occurrences)

| Location | Count | Purpose | Pattern to Apply |
|:---------|:------|:--------|:-----------------|
| Lines 246-247 | 2 | Dimension detection | try/except (same as FDM) |
| Line 250 | 1 | Dimension fallback | try/except |
| Lines 827, 860 | 2 | Coupling coefficient | try/except with getattr fallback |
| Line 868 | 1 | Hamiltonian check | try/except cascade |
| Lines 935-940 | 4 | BC retrieval (characteristics) | Centralized method |
| Lines 1066-1069 | 4 | BC retrieval (diffusion) | Centralized method |
| Lines 1181, 1185, 1195 | 3 | Hamiltonian interface | try/except cascade (same as FDM) |
| Line 1240 | 1 | Geometry check | try/except |
| Line 1469 | 1 | Test assertion | Direct access (test code) |

### BC Handling Patterns

**Current BC Architecture:**
```
┌──────────────────────────────────────────┐
│   HJBSemiLagrangianSolver                │
│   • No centralized BC retrieval          │
│   • Duplicate hasattr checks             │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│   hjb_sl_characteristics module          │
│   • apply_boundary_conditions_1d()       │
│   • apply_boundary_conditions_nd()       │
│   • BC enforcement by clamping           │
└──────────────────────────────────────────┘
```

**BC Usage Points:**
1. **Characteristic tracing** (lines 933-949, 962-966): Clamp departure points to domain
2. **Diffusion term** (lines 1072-1082): Handle periodic BC in Laplacian computation

### Key Differences from FDM Solver

| Aspect | FDM Solver | Semi-Lagrangian Solver |
|:-------|:-----------|:-----------------------|
| **BC Enforcement** | Ghost cells | Clamping characteristic departure points |
| **BC Delegation** | Built-in methods | hjb_sl_characteristics module |
| **BC Complexity** | High (17 hasattr) | Moderate (16 hasattr) |
| **Protocol Fit** | Direct implementation | Needs adapter pattern |

## Refactoring Tasks

### Task 1: Dimension Detection (Lines 246-251) ✅ STRAIGHTFORWARD

**Current Code:**
```python
def _detect_dimension(self, problem) -> int:
    # Try geometry.dimension first (unified interface)
    if hasattr(problem, "geometry") and hasattr(problem.geometry, "dimension"):
        return problem.geometry.dimension

    # Fall back to problem.dimension
    if hasattr(problem, "dimension"):
        return problem.dimension

    # Legacy 1D detection
    if getattr(problem, "Nx", None) is not None and getattr(problem, "Ny", None) is None:
        return 1

    raise ValueError("Cannot determine problem dimension...")
```

**Refactored Code:**
```python
def _detect_dimension(self, problem) -> int:
    """Detect spatial dimension from geometry (unified interface)."""
    # Issue #545: Use try/except instead of hasattr
    # Priority 1: geometry.dimension
    try:
        return problem.geometry.dimension
    except AttributeError:
        pass

    # Priority 2: problem.dimension
    try:
        return problem.dimension
    except AttributeError:
        pass

    # Legacy 1D detection
    if getattr(problem, "Nx", None) is not None and getattr(problem, "Ny", None) is None:
        return 1

    raise ValueError(
        "Cannot determine problem dimension. "
        "Ensure problem has 'geometry' with 'dimension' attribute or 'dimension' property."
    )
```

**Rationale**: Identical to FDM solver pattern (already validated in tests).

---

### Task 2: BC Retrieval (Lines 935-940, 1066-1069) ✅ HIGH PRIORITY

**Current Code (Duplicate Pattern):**
```python
# Location 1: _trace_characteristic_backward (lines 935-940)
bc_type = None
if hasattr(self.problem, "get_boundary_conditions"):
    bc = self.problem.get_boundary_conditions()
    if bc is not None and hasattr(bc, "default_bc"):
        bc_type_enum = bc.default_bc
        if bc_type_enum is not None:
            bc_type = bc_type_enum.value if hasattr(bc_type_enum, "value") else str(bc_type_enum)

# Location 2: _compute_diffusion_term (lines 1066-1069)
bc_type = None
if hasattr(self.problem, "get_boundary_conditions"):
    bc = self.problem.get_boundary_conditions()
    if bc is not None and hasattr(bc, "default_bc") and bc.default_bc is not None:
        bc_type = bc.default_bc.value if hasattr(bc.default_bc, "value") else str(bc.default_bc)
```

**Refactored Code:**

**Step 1: Add centralized BC retrieval method**
```python
def _get_boundary_conditions(self):
    """
    Retrieve BC from geometry or problem.

    Priority order:
    1. geometry.boundary_conditions (attribute)
    2. geometry.get_boundary_conditions() (method)
    3. problem.boundary_conditions (attribute)
    4. problem.get_boundary_conditions() (method)

    Returns:
        BoundaryConditions object or None if not available
    """
    # Issue #545: Centralized BC retrieval (NO hasattr)

    # Priority 1: geometry.boundary_conditions
    try:
        bc = self.problem.geometry.boundary_conditions
        if bc is not None:
            return bc
    except AttributeError:
        pass

    # Priority 2: geometry.get_boundary_conditions()
    try:
        bc = self.problem.geometry.get_boundary_conditions()
        if bc is not None:
            return bc
    except AttributeError:
        pass

    # Priority 3: problem.boundary_conditions
    try:
        bc = self.problem.boundary_conditions
        if bc is not None:
            return bc
    except AttributeError:
        pass

    # Priority 4: problem.get_boundary_conditions()
    try:
        bc = self.problem.get_boundary_conditions()
        if bc is not None:
            return bc
    except AttributeError:
        pass

    return None

def _get_bc_type_string(self, bc) -> str | None:
    """
    Extract BC type string from BoundaryConditions object.

    Args:
        bc: BoundaryConditions object or None

    Returns:
        BC type string ("periodic", "dirichlet", "neumann") or None
    """
    if bc is None:
        return None

    # Try to get default_bc attribute
    try:
        bc_type_enum = bc.default_bc
        if bc_type_enum is None:
            return None

        # Try to get .value attribute (BCType enum)
        try:
            return bc_type_enum.value
        except AttributeError:
            # Fall back to string conversion
            return str(bc_type_enum)
    except AttributeError:
        return None
```

**Step 2: Replace duplicate code**
```python
# Location 1: _trace_characteristic_backward (lines 933-949)
bc = self._get_boundary_conditions()
bc_type = self._get_bc_type_string(bc)

bounds = self.problem.geometry.get_bounds()
xmin, xmax = bounds[0][0], bounds[1][0]
return apply_boundary_conditions_1d(
    x_departure,
    xmin=xmin,
    xmax=xmax,
    bc_type=bc_type,
)

# Location 2: _compute_diffusion_term (lines 1064-1070)
bc = self._get_boundary_conditions()
bc_type = self._get_bc_type_string(bc)
```

**Benefits:**
- Eliminates 8 hasattr checks (4 + 4)
- Consolidates duplicate BC retrieval logic
- Consistent with FDM solver pattern
- Easier to maintain and test

---

### Task 3: Hamiltonian Detection (Lines 1181, 1185, 1195, 868) ✅ STRAIGHTFORWARD

**Current Code:**
```python
# Line 1181
if hasattr(self.problem, "H"):
    return self.problem.H(x_idx, m, derivs=derivs, t_idx=time_idx)

# Line 1185
elif hasattr(self.problem, "hamiltonian"):
    return self.problem.hamiltonian(x_idx, m, derivs=derivs, t=t_value)

# Lines 1193-1201 (fallback with legacy signature)
try:
    if hasattr(self.problem, "hamiltonian"):
        x_vec = np.atleast_1d(x)
        p_vec = np.atleast_1d(p)
        return self.problem.hamiltonian(x_vec, m, p_vec, t_value)
except Exception as legacy_err:
    logger.debug(f"Legacy Hamiltonian signature failed: {legacy_err}")
```

**Refactored Code:**
```python
def _evaluate_hamiltonian(...) -> float:
    """Evaluate Hamiltonian H(x, p, m) at given point."""
    # Build DerivativeTensors from gradient p
    derivs = self._build_derivative_tensors(p)
    x_idx = self._position_to_index(x)
    t_value = time_idx * self.problem.T / self.problem.Nt if time_idx is not None else 0.0

    # Issue #545: Use try/except instead of hasattr
    try:
        # Priority 1: problem.H() with DerivativeTensors (preferred)
        return self.problem.H(x_idx, m, derivs=derivs, t_idx=time_idx)
    except AttributeError:
        pass

    try:
        # Priority 2: problem.hamiltonian() with derivs kwarg
        return self.problem.hamiltonian(x_idx, m, derivs=derivs, t=t_value)
    except (AttributeError, TypeError):
        pass

    # Priority 3: Legacy signature hamiltonian(x, m, p, t)
    try:
        x_vec = np.atleast_1d(x)
        p_vec = np.atleast_1d(p)
        return self.problem.hamiltonian(x_vec, m, p_vec, t_value)
    except (AttributeError, TypeError) as e:
        logger.debug(f"Legacy Hamiltonian signature failed: {e}")

    # Fallback: Default quadratic Hamiltonian
    return self._default_hamiltonian(derivs, m)
```

**Rationale**: Identical to FDM solver pattern, consolidates 4 hasattr checks into try/except cascade.

---

### Task 4: Coupling Coefficient (Lines 827, 860) ⚠️ SIMPLE

**Current Code:**
```python
# Line 827
if hasattr(self.problem, "coupling_coefficient"):
    return 0.0

# Line 860
if hasattr(self.problem, "coupling_coefficient"):
    return np.zeros(self.dimension)
```

**Refactored Code:**
```python
# Line 827
# Issue #545: Use try/except or getattr for optional attribute
coupling_coef = getattr(self.problem, "coupling_coefficient", None)
if coupling_coef is not None:
    return 0.0

# OR use try/except for consistency:
try:
    _ = self.problem.coupling_coefficient
    return 0.0  # Standard quadratic Hamiltonian
except AttributeError:
    pass  # Continue to numerical optimization

# Line 860 (similar pattern)
coupling_coef = getattr(self.problem, "coupling_coefficient", None)
if coupling_coef is not None:
    return np.zeros(self.dimension)
```

**Rationale**: Simple attribute check, use getattr or try/except for consistency.

---

### Task 5: Geometry Check (Line 1240) ✅ TRIVIAL

**Current Code:**
```python
bounds = (
    self.problem.geometry.bounds
    if hasattr(self.problem, "geometry") and self.problem.geometry is not None
    else None
)
```

**Refactored Code:**
```python
# Issue #545: Use try/except instead of hasattr
try:
    bounds = self.problem.geometry.bounds if self.problem.geometry is not None else None
except AttributeError:
    bounds = None
```

**Rationale**: Standard try/except pattern.

---

### Task 6: Test Code (Line 1469) ℹ️ TEST ONLY

**Current Code:**
```python
assert hasattr(solver_2d, "_adi_compatible")
```

**Refactored Code:**
```python
# Direct attribute access - will raise AttributeError if missing
assert solver_2d._adi_compatible is not None
# OR test the actual value
assert solver_2d._adi_compatible  # Should be True for scalar sigma
```

**Rationale**: Test code in `if __name__ == "__main__"` block, safe to use direct access.

---

## BoundaryHandler Protocol Implementation

### Current BC Workflow

Semi-Lagrangian solver delegates BC enforcement to `hjb_sl_characteristics` module:
- `apply_boundary_conditions_1d()` - Clamps 1D departure points
- `apply_boundary_conditions_nd()` - Clamps nD departure points

This is **different** from FDM/GFDM ghost cell approach.

### Implementation Strategy

**Option 1: Minimal Adapter (Recommended)**

Implement protocol methods that delegate to existing infrastructure:

```python
class HJBSemiLagrangianSolver(BaseHJBSolver):
    """Semi-Lagrangian HJB solver with BoundaryHandler protocol support."""

    # ... existing code ...

    # ═══════════════════════════════════════════════════════════════════
    # BoundaryHandler Protocol (Issue #545)
    # ═══════════════════════════════════════════════════════════════════

    def get_boundary_indices(self) -> np.ndarray:
        """
        Identify boundary points in solver's discretization.

        Returns:
            Array of integer indices identifying boundary grid points.

        Note:
            For Semi-Lagrangian method, boundary points are those on the
            domain boundary where characteristic tracing requires clamping.
        """
        if self.dimension == 1:
            # 1D: First and last grid points
            Nx = len(self.x_grid)
            return np.array([0, Nx - 1], dtype=np.int64)
        else:
            # nD: All grid points on boundary faces
            # For tensor grid: point is on boundary if any coordinate is at min/max
            boundary_mask = np.zeros(self._grid_shape, dtype=bool)

            # Mark boundary faces in each dimension
            for d in range(self.dimension):
                # Lower boundary face (index 0 along dimension d)
                slices_lower = [slice(None)] * self.dimension
                slices_lower[d] = 0
                boundary_mask[tuple(slices_lower)] = True

                # Upper boundary face (index -1 along dimension d)
                slices_upper = [slice(None)] * self.dimension
                slices_upper[d] = -1
                boundary_mask[tuple(slices_upper)] = True

            # Return flat indices of boundary points
            return np.flatnonzero(boundary_mask.ravel())

    def apply_boundary_conditions(
        self,
        values: np.ndarray,
        bc: BoundaryConditions,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Apply boundary conditions to solution values.

        For Semi-Lagrangian method, BC enforcement is handled during
        characteristic tracing (clamping departure points to domain).
        This method is a no-op adapter for protocol compliance.

        Args:
            values: Solution values at all grid points
            bc: Boundary conditions object
            time: Current time (unused for Semi-Lagrangian)

        Returns:
            Solution values (unchanged, BC enforced during characteristic tracing)
        """
        # Semi-Lagrangian enforces BCs during characteristic tracing,
        # not as a post-processing step. Return values unchanged.
        return values

    def get_bc_type_for_point(self, point_idx: int) -> str:
        """
        Determine BC type for a specific grid point.

        Args:
            point_idx: Index of grid point

        Returns:
            BC type string: "periodic", "dirichlet", "neumann", or "none"
        """
        bc = self._get_boundary_conditions()
        if bc is None:
            return "none"

        # For uniform BC (most common case)
        bc_type_str = self._get_bc_type_string(bc)
        if bc_type_str is not None:
            return bc_type_str

        # For mixed BC, would need to query BC segments
        # (Semi-Lagrangian typically uses uniform BC)
        return "none"
```

**Option 2: Full Implementation**

Implement BC enforcement as post-processing step (not recommended, changes solver behavior).

---

## Implementation Checklist

### Phase 1: hasattr Elimination ✅ HIGH PRIORITY
- [ ] Replace dimension detection (lines 246-251)
- [ ] Create centralized `_get_boundary_conditions()` method
- [ ] Create `_get_bc_type_string()` helper
- [ ] Replace BC retrieval in `_trace_characteristic_backward` (lines 935-940)
- [ ] Replace BC retrieval in `_compute_diffusion_term` (lines 1066-1069)
- [ ] Replace Hamiltonian detection (lines 1181, 1185, 1195, 868)
- [ ] Replace coupling coefficient checks (lines 827, 860)
- [ ] Replace geometry check (line 1240)
- [ ] Update test assertion (line 1469)

### Phase 2: BoundaryHandler Protocol ✅ MEDIUM PRIORITY
- [ ] Implement `get_boundary_indices()` method
- [ ] Implement `apply_boundary_conditions()` adapter
- [ ] Implement `get_bc_type_for_point()` method
- [ ] Add protocol documentation to docstring

### Phase 3: Testing ✅ CRITICAL
- [ ] Run all Semi-Lagrangian unit tests
- [ ] Verify convergence behavior unchanged
- [ ] Test with periodic, Dirichlet, Neumann BCs
- [ ] Test 1D and nD cases
- [ ] Validate protocol compliance with `validate_boundary_handler()`

### Phase 4: Documentation
- [ ] Update `issue_545_solver_refactoring_progress.md`
- [ ] Add implementation notes to `BOUNDARY_HANDLING.md`
- [ ] Update solver docstring with protocol reference

## Expected Results

### Before Refactoring
```
Total hasattr:       16
BC Interface:        Inconsistent (duplicate retrieval logic)
Protocol Support:    None
```

### After Refactoring
```
Total hasattr:       0 ✅
BC Interface:        Centralized (_get_boundary_conditions)
Protocol Support:    BoundaryHandler protocol ✅
Test Coverage:       Unchanged (all tests pass)
```

## Validation Criteria

1. ✅ **All hasattr eliminated** - 0 occurrences in main code
2. ✅ **BC retrieval centralized** - Single `_get_boundary_conditions()` method
3. ✅ **Protocol implemented** - 3 core BoundaryHandler methods
4. ✅ **Tests pass** - All existing Semi-Lagrangian tests pass
5. ✅ **Behavior unchanged** - Convergence results identical to before

## Notes

**Semi-Lagrangian BC Philosophy:**
- BC enforcement during **characteristic tracing** (clamp departure points)
- NOT post-processing like FDM ghost cells
- Protocol `apply_boundary_conditions()` is adapter/no-op
- True BC enforcement in `hjb_sl_characteristics.apply_boundary_conditions_*`

**Advantages of this approach:**
- Maintains existing, tested BC enforcement logic
- Adds protocol compliance without changing solver behavior
- Centralized BC retrieval improves maintainability
- Consistent with Issue #545 goals

---

**Created**: 2026-01-11
**Author**: Claude Code (Opus 4.5)
**Related**: Issue #545, FDM refactoring (eecd7ce), BoundaryHandler protocol (265ae91)
