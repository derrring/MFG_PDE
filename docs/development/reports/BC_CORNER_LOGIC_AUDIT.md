# BC Corner Logic Audit Report

**Issue**: #486 Phase 0a, 0e
**Date**: December 2025
**Status**: Audit Complete, Stateless Verified

## Executive Summary

The `applicator_fdm.py` module handles corner values using a simple **averaging strategy**. This is numerically stable but may not be optimal for all BC combinations. This audit documents the current implementation and identifies potential improvements.

## Current Implementation

### 1. Uniform BC Corners (2D)

**Location**: `_apply_uniform_bc_2d()` lines 198-201, 222-225, 270-274, 292-295

```python
# Dirichlet example:
padded[0, 0] = 2 * g - field[0, 0]
padded[0, -1] = 2 * g - field[0, -1]
padded[-1, 0] = 2 * g - field[-1, 0]
padded[-1, -1] = 2 * g - field[-1, -1]
```

**Analysis**: For uniform BCs, corners use the same ghost cell formula as edges. This is mathematically consistent since all boundaries have the same BC type.

**Status**: Correct for uniform BCs.

### 2. Mixed BC Corners (2D)

**Location**: `_apply_mixed_bc_2d()` lines 405-409

```python
# Handle corners (average of adjacent boundary values)
padded[0, 0] = 0.5 * (padded[0, 1] + padded[1, 0])
padded[0, -1] = 0.5 * (padded[0, -2] + padded[1, -1])
padded[-1, 0] = 0.5 * (padded[-1, 1] + padded[-2, 0])
padded[-1, -1] = 0.5 * (padded[-1, -2] + padded[-2, -1])
```

**Analysis**: After setting ghost values on all four edges, corners are computed as the average of the two adjacent edge ghost values.

**Pros**:
- Simple and numerically stable
- Provides smooth transition between different BC types
- Works for all BC combinations

**Cons**:
- May not preserve exact BC semantics at corners
- No BC type precedence (e.g., Dirichlet corner should enforce value, not average)

### 3. nD Corner Handling

**Location**: `_apply_corner_values_nd()` lines 829-887

```python
def _apply_corner_values_nd(padded: NDArray[np.floating], d: int) -> None:
    """Apply corner/edge values in nD using averaging."""
    # Iterate over all corner positions
    for corner_idx in np.ndindex(*([3] * d)):
        # Count boundary dimensions
        boundary_count = sum(1 for c in corner_idx if c != 1)
        if boundary_count < 2:
            continue  # Not a corner

        # Average neighbors that differ in exactly one boundary dimension
        neighbors = [...]
        avg_val = np.mean([padded[n] for n in neighbors])
        padded[actual_idx] = avg_val
```

**Analysis**: Generalizes the 2D averaging strategy to arbitrary dimensions. For d dimensions:
- 2D: 4 corners (0D intersections)
- 3D: 12 edges (1D) + 8 corners (0D)
- nD: 3^d - 2d - 1 intersections of various codimension

## Corner Strategy Options

### Option 1: Averaging (Current)

```
Corner value = mean(adjacent edge ghost values)
```

**Use case**: Default strategy, smooth solution required

### Option 2: Priority-Based

```
BC Precedence: Periodic > Dirichlet > Robin > Neumann > No-flux
Corner inherits BC from highest-priority adjacent edge
```

**Use case**: When exact BC enforcement is critical

### Option 3: Mollified

```
Corner region uses weighted blend based on distance to edge midpoints
```

**Use case**: High-order methods, smooth corner transition needed

### Option 4: Extrapolation

```
Corner value extrapolated from adjacent interior points
```

**Use case**: When corner is sufficiently interior that BCs don't dominate

## Recommendations

### Short-term (No code changes needed)

1. The current averaging strategy is acceptable for most MFG applications
2. Document the corner behavior in user-facing documentation
3. Add a warning if mixed BCs with incompatible types meet at corners

### Medium-term (Phase 0b deliverable)

1. Add `corner_strategy` parameter to `GhostCellConfig`:
   ```python
   @dataclass
   class GhostCellConfig:
       grid_type: Literal["cell_centered", "vertex_centered"] = "cell_centered"
       corner_strategy: Literal["average", "priority", "extrapolate"] = "average"
   ```

2. Implement priority-based corner handling for Dirichlet-Neumann interfaces

### Long-term (Phase 1+)

1. Support high-order ghost extrapolation for WENO (needs 3+ ghost cells)
2. Add corner convergence tests in validation suite
3. Consider mollified corners for smooth solutions

## Test Coverage

Current corner-related tests: **Minimal**

Recommended additions:
- `test_corner_dirichlet_neumann_interface`: Mixed BC corner accuracy
- `test_corner_convergence_2d`: Convergence rate at corners
- `test_corner_3d_edges`: 3D edge ghost values
- `test_corner_periodic_consistency`: Periodic BC corner wrapping

## Stateless Verification (Phase 0e)

**Status**: Verified ✅

The ghost cell functions in `applicator_fdm.py` are confirmed to be **stateless**:

### Properties Verified

1. **No Global State**
   - No `global` keyword usage
   - No module-level mutable variables
   - No class-level shared state

2. **Input Immutability**
   - `apply_boundary_conditions_1d()`: Input array unchanged ✅
   - `apply_boundary_conditions_2d()`: Input array unchanged ✅
   - `apply_boundary_conditions_3d()`: Input array unchanged ✅
   - `apply_boundary_conditions_nd()`: Input array unchanged ✅

3. **Determinism**
   - Same input → same output (verified)
   - No random state or time-dependent behavior

4. **Side Effect Documentation**
   - Internal `_apply_*` functions modify `padded` array in-place (documented)
   - Public `apply_boundary_conditions_*` functions return new arrays

### Implications for Solver Integration

- Functions can be called multiple times safely
- Thread-safe for parallel solver implementations
- Compatible with JIT compilation (Numba/JAX)
- No hidden state that could cause bugs in time-stepping loops

## Conclusion

The current averaging strategy is **adequate** for typical MFG problems. The main limitation is lack of BC type precedence, which could matter for problems with sharp Dirichlet-Neumann interfaces.

**Recommendation**: Proceed to Phase 0b (document corner strategies) without code changes. The averaging strategy handles the common cases well.

## References

- `applicator_fdm.py`: Main implementation
- Issue #486: BC Unification tracking
- `BC_UNIFICATION_TECHNICAL_REPORT.md`: Overall technical design
