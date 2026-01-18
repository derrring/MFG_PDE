# Issue #598: BCApplicatorProtocol → ABC Template Method Pattern

**Status**: Design Phase
**Priority**: MEDIUM
**Estimated Effort**: 2-3 days
**Created**: 2026-01-18

---

## Executive Summary

Refactor `BCApplicatorProtocol` from Protocol to ABC with Template Method Pattern to eliminate ghost cell logic duplication across FDM applicators (1D/2D/3D/nD).

**Current State**:
- ~300 lines of duplicated ghost cell formulas across dimension-specific functions
- Ghost cell logic at lines 232, 1099, 1255, 1647, 1696, 2664 in `applicator_fdm.py`
- `BaseBCApplicator(ABC)` exists but only defines properties (no template method)

**Goal**: Extract shared logic to ABC base class, define hook methods for dimension-specific code.

---

## Analysis: Duplication Patterns

### 1. Ghost Cell Formula Duplication

**Dirichlet BC** - Cell-centered grid formula: `u_ghost = 2*g - u_interior`

Found in:
- Line 232: 2D uniform Dirichlet
  ```python
  padded[0, 1:-1] = 2 * g - field[0, :]
  padded[-1, 1:-1] = 2 * g - field[-1, :]
  ```
- Line 1255: 1D legacy Dirichlet
  ```python
  padded[0] = 2.0 * left_val - field[0]
  padded[-1] = 2.0 * right_val - field[-1]
  ```
- Line 1725: `_compute_ghost_pair()` helper
  ```python
  if config.is_vertex_centered:
      return g, g
  else:
      return 2 * g - u_int_left, 2 * g - u_int_right
  ```

**Neumann BC** - Cell-centered grid formula: `u_ghost = u_interior ± 2*dx*g`

Found in:
- Line 1274: 1D reflection (zero-flux case)
  ```python
  padded[0] = field[1]  # Reflection: ghost = u[next_interior]
  padded[-1] = field[-2]  # Reflection: ghost = u[prev_interior]
  ```
- Line 1099: nD reflection logic
  ```python
  slices_next_low[axis] = slice(2, 3)
  padded[tuple(slices_low)] = padded[tuple(slices_next_low)]
  ```
- Line 1748: `_compute_ghost_pair()` Neumann case
  ```python
  return u_next_left - 2 * dx * g, u_prev_right + 2 * dx * g
  ```

### 2. Slicing Logic Duplication

Building nD array slices for boundary extraction repeated in:
- Lines 1637-1676: Mixed BC ghost computation (40 lines of slicing)
  ```python
  slices_left = [slice(None)] * d
  slices_left[axis] = 0
  u_int_left = field[tuple(slices_left)]

  slices_next_left = [slice(None)] * d
  slices_next_left[axis] = 1 if shape_axis > 1 else 0
  u_next_left = field[tuple(slices_next_left)]
  ```

### 3. Validation Logic Duplication

NaN/Inf checks and shape validation:
- Line 1247: 1D validation
  ```python
  if not np.isfinite(field).all():
      raise ValueError("Field contains NaN or Inf values")
  ```
- Similar validation repeated in 2D, 3D, nD functions

### 4. Grid Spacing Computation Duplication

Computing spacing from domain bounds:
- Line 2664: `PreallocatedGhostBuffer`
  ```python
  extent = domain_bounds[d, 1] - domain_bounds[d, 0]
  n_points = interior_shape[d]
  spacing.append(extent / (n_points - 1) if n_points > 1 else extent)
  ```
- Similar logic in multiple BC application functions

---

## Proposed Architecture: Template Method Pattern

### Current Class Hierarchy

```
BCApplicatorProtocol (Protocol)
  ↑
BaseBCApplicator (ABC) ← Currently minimal (only properties)
  ↑
BaseStructuredApplicator
  ↑
FDMApplicator ← Delegates to dimension-specific functions
```

### Proposed Refactored Hierarchy

```
BaseBCApplicator (ABC with Template Method Pattern)
  ├─ apply()              # Template method (orchestrates workflow)
  ├─ _validate_inputs()   # Shared validation
  ├─ _create_padded_buffer()  # Shared buffer creation
  ├─ _compute_ghost_dirichlet()  # Shared Dirichlet formula
  ├─ _compute_ghost_neumann()    # Shared Neumann formula
  ├─ _compute_ghost_robin()      # Shared Robin formula
  └─ _fill_ghost_cells()  # Abstract hook (dimension-specific)
      ↑
BaseStructuredApplicator
  ↑
FDMApplicator
  └─ _fill_ghost_cells()  # Overrides for FDM-specific logic
```

---

## Template Method Design

### Template Method: `apply()`

```python
class BaseBCApplicator(ABC):
    """Base class for boundary condition applicators (Template Method Pattern)."""

    def apply(
        self,
        field: NDArray,
        boundary_conditions: BoundaryConditions,
        domain_bounds: NDArray | None = None,
        time: float = 0.0,
        geometry = None,  # SupportsRegionMarking | None
        **kwargs,
    ) -> NDArray:
        """
        Apply boundary conditions to field (template method).

        Orchestrates the BC application workflow:
        1. Validate inputs
        2. Create padded buffer
        3. Fill ghost cells (dimension-specific)
        4. Finalize and return

        This method is final - subclasses should not override.
        Override _fill_ghost_cells() instead.
        """
        # 1. Validate inputs (shared logic)
        self._validate_inputs(field, boundary_conditions, domain_bounds, geometry)

        # 2. Create padded buffer (shared logic)
        padded = self._create_padded_buffer(field)

        # 3. Fill ghost cells (hook - dimension-specific)
        #    Subclasses override this method
        self._fill_ghost_cells(
            padded, field, boundary_conditions,
            domain_bounds, time, geometry, **kwargs
        )

        # 4. Finalize (shared post-processing)
        return self._finalize(padded)
```

### Abstract Hook Method: `_fill_ghost_cells()`

```python
    @abstractmethod
    def _fill_ghost_cells(
        self,
        padded: NDArray,
        field: NDArray,
        boundary_conditions: BoundaryConditions,
        domain_bounds: NDArray | None,
        time: float,
        geometry,
        **kwargs,
    ) -> None:
        """
        Fill ghost cells in padded array (dimension-specific logic).

        Subclasses implement dimension-specific slicing and iteration.
        Use shared ghost cell formula methods:
        - self._compute_ghost_dirichlet()
        - self._compute_ghost_neumann()
        - self._compute_ghost_robin()

        Args:
            padded: Padded array to fill (modified in-place)
            field: Interior field values (read-only)
            boundary_conditions: BC specification
            domain_bounds: Domain bounds for spacing computation
            time: Current time for time-dependent BCs
            geometry: Geometry with region markings
        """
        pass
```

### Shared Validation: `_validate_inputs()`

```python
    def _validate_inputs(
        self,
        field: NDArray,
        boundary_conditions: BoundaryConditions,
        domain_bounds: NDArray | None,
        geometry,
    ) -> None:
        """
        Validate inputs before BC application (shared logic).

        Checks:
        - Field contains finite values (no NaN/Inf)
        - BC type is supported
        - Domain bounds provided if needed (mixed BC)
        - Geometry provided if BC uses regions

        Raises:
            ValueError: If validation fails
            TypeError: If BC type unsupported
        """
        # NaN/Inf check
        if not np.isfinite(field).all():
            raise ValueError(
                f"Field contains NaN or Inf values. "
                f"Check solver convergence and boundary conditions."
            )

        # BC type check
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"Expected BoundaryConditions, got {type(boundary_conditions).__name__}. "
                f"Legacy BC types should be converted first."
            )

        # Mixed BC validation
        if not boundary_conditions.is_uniform and domain_bounds is None:
            raise ValueError(
                "Mixed boundary conditions require domain_bounds parameter. "
                "Provide domain bounds as shape (ndim, 2) array."
            )

        # Region-based BC validation (Issue #596 Phase 2.5)
        if self._uses_regions(boundary_conditions) and geometry is None:
            raise ValueError(
                "Boundary conditions use region_name but geometry parameter not provided. "
                "Pass geometry with marked regions to apply() method."
            )
```

### Shared Ghost Cell Formulas

```python
    def _compute_ghost_dirichlet(
        self,
        u_interior: NDArray,
        g: float | Callable,
        time: float = 0.0,
    ) -> NDArray:
        """
        Compute Dirichlet ghost cell value (shared formula).

        Cell-centered grid: u_ghost = 2*g - u_interior
        Vertex-centered grid: u_ghost = g

        Args:
            u_interior: Interior values adjacent to boundary
            g: Boundary value (constant or callable)
            time: Current time (for callable g)

        Returns:
            Ghost cell values
        """
        # Evaluate callable BC values
        if callable(g):
            g = g(time)

        # Apply formula based on grid type
        if self._grid_type == GridType.CELL_CENTERED:
            return 2.0 * g - u_interior
        else:  # VERTEX_CENTERED
            return np.full_like(u_interior, g)

    def _compute_ghost_neumann(
        self,
        u_interior: NDArray,
        u_next_interior: NDArray,
        g: float | Callable,
        dx: float,
        side: str,  # "left" or "right"
        time: float = 0.0,
    ) -> NDArray:
        """
        Compute Neumann ghost cell value (shared formula).

        For central difference gradient stencil:
        - Left boundary: u_ghost = u_next_interior - 2*dx*g
        - Right boundary: u_ghost = u_prev_interior + 2*dx*g

        Special case g=0 (zero-flux): u_ghost = u_next_interior (reflection)

        Args:
            u_interior: Interior values adjacent to boundary
            u_next_interior: Interior values one step away
            g: Flux value (constant or callable)
            dx: Grid spacing
            side: Boundary side ("left" or "right")
            time: Current time (for callable g)

        Returns:
            Ghost cell values

        Note:
            Issue #542 fix: Uses reflection formula for central difference,
            not edge extension (ghost = u_interior).
        """
        # Evaluate callable BC values
        if callable(g):
            g = g(time)

        # Zero-flux case: reflection
        if np.isclose(g, 0.0):
            return u_next_interior.copy()

        # General Neumann case
        if side == "left":
            # Left boundary: outward normal points left (negative direction)
            # du/dn = (u_interior - u_ghost) / (2*dx) = g
            # => u_ghost = u_interior - 2*dx*g
            # But for reflection, use u_next_interior instead
            return u_next_interior - 2.0 * dx * g
        else:  # "right"
            # Right boundary: outward normal points right (positive direction)
            # du/dn = (u_ghost - u_interior) / (2*dx) = g
            # => u_ghost = u_interior + 2*dx*g
            # But for reflection, use u_prev_interior instead
            return u_next_interior + 2.0 * dx * g

    def _compute_ghost_robin(
        self,
        u_interior: NDArray,
        alpha: float,
        beta: float,
        g: float | Callable,
        dx: float,
        side: str,
        time: float = 0.0,
    ) -> NDArray:
        """
        Compute Robin ghost cell value (shared formula).

        Robin BC: alpha*u + beta*du/dn = g at boundary

        For cell-centered grid:
        alpha * (u_ghost + u_interior)/2 + beta * (u_ghost - u_interior)/(2*dx) = g
        => u_ghost = (g - u_interior * (alpha/2 - beta/(2*dx))) / (alpha/2 + beta/(2*dx))

        Args:
            u_interior: Interior values adjacent to boundary
            alpha: Robin coefficient for value term
            beta: Robin coefficient for flux term
            g: Robin boundary value
            dx: Grid spacing
            side: Boundary side ("left" or "right")
            time: Current time (for callable g)

        Returns:
            Ghost cell values
        """
        # Evaluate callable BC values
        if callable(g):
            g = g(time)

        # Robin formula
        coeff_ghost = alpha / 2.0 + beta / (2.0 * dx)
        coeff_interior = alpha / 2.0 - beta / (2.0 * dx)

        return (g - u_interior * coeff_interior) / coeff_ghost
```

### Shared Buffer Creation: `_create_padded_buffer()`

```python
    def _create_padded_buffer(
        self,
        field: NDArray,
        ghost_depth: int = 1,
    ) -> NDArray:
        """
        Create zero-initialized padded buffer (shared logic).

        Args:
            field: Interior field
            ghost_depth: Number of ghost cells per side (default: 1)

        Returns:
            Zero-initialized padded array with shape (N1+2*depth, N2+2*depth, ...)
        """
        # Compute padded shape
        padded_shape = tuple(n + 2 * ghost_depth for n in field.shape)

        # Create zero buffer
        padded = np.zeros(padded_shape, dtype=field.dtype)

        # Copy interior values
        interior_slices = tuple(
            slice(ghost_depth, -ghost_depth) for _ in range(field.ndim)
        )
        padded[interior_slices] = field

        return padded
```

### Shared Grid Spacing Computation

```python
    def _compute_grid_spacing(
        self,
        field: NDArray,
        domain_bounds: NDArray,
    ) -> tuple[float, ...]:
        """
        Compute grid spacing from domain bounds (shared logic).

        Args:
            field: Interior field (to get shape)
            domain_bounds: Domain bounds shape (ndim, 2)

        Returns:
            Grid spacing for each dimension
        """
        domain_bounds = np.atleast_2d(domain_bounds)
        spacing = []

        for d in range(field.ndim):
            extent = domain_bounds[d, 1] - domain_bounds[d, 0]
            n_points = field.shape[d]
            dx = extent / (n_points - 1) if n_points > 1 else extent
            spacing.append(dx)

        return tuple(spacing)
```

---

## Migration Strategy

### Phase 1: Add Template Method to `BaseBCApplicator`

**File**: `mfg_pde/geometry/boundary/applicator_base.py`

**Changes**:
1. Add template method `apply()` to `BaseBCApplicator`
2. Add abstract hook `_fill_ghost_cells()`
3. Add shared validation `_validate_inputs()`
4. Add shared ghost cell formula methods
5. Add shared buffer creation method

**Backward Compatibility**: Existing subclasses continue to work (no breaking changes).

### Phase 2: Refactor `FDMApplicator` to Use Template Method

**File**: `mfg_pde/geometry/boundary/applicator_fdm.py`

**Changes**:
1. Override `_fill_ghost_cells()` in `FDMApplicator`
2. Use shared ghost cell formula methods
3. Remove duplicated validation logic
4. Keep dimension-specific slicing in `_fill_ghost_cells()`

**Backward Compatibility**: Keep existing `apply_boundary_conditions_*d()` functions as thin wrappers calling new implementation.

### Phase 3: Deprecate Function-Based API

**Timeline**: After v0.19.0 (Issue #577)

Mark function-based API as deprecated:
```python
@deprecated(
    "apply_boundary_conditions_2d() will be removed in v0.19.0. "
    "Use FDMApplicator.apply() or pad_array_with_ghosts() instead.",
    category=DeprecationWarning,
)
def apply_boundary_conditions_2d(...):
    ...
```

---

## Benefits

### 1. DRY Principle (Don't Repeat Yourself)

- Ghost cell formulas defined once, reused across 1D/2D/3D/nD
- Validation logic centralized
- Grid spacing computation shared

**Metrics**:
- **Before**: ~300 lines of duplicated ghost cell logic
- **After**: ~50 lines of shared ghost cell formulas
- **Reduction**: 83% fewer lines for ghost cell computation

### 2. Consistency

- All applicators follow same workflow
- Bug fixes propagate to all dimensions automatically
- Same validation everywhere

**Example**: Issue #542 fix (Neumann reflection) required changes in 4 places. With ABC pattern, would require 1 change.

### 3. Maintainability

- Clear separation: template method (workflow) vs hook method (dimension logic)
- Easier to understand code structure
- Self-documenting through template method pattern

### 4. Extensibility

- New BC types added once in shared utilities
- Subclasses inherit new functionality automatically
- Easier to add new applicators (FEM, GFDM, etc.)

---

## Risks and Mitigation

### Risk 1: Performance Regression

**Concern**: Method calls might add overhead vs inlined code.

**Mitigation**:
- Profile before/after refactoring
- Ensure shared methods are simple (NumPy operations, not loops)
- JIT compilation possible if needed (Numba)

**Acceptable Overhead**: <5% (per Issue #596 performance tests)

### Risk 2: Breaking Changes

**Concern**: Existing code might break during migration.

**Mitigation**:
- Keep function-based API as thin wrappers (no breaking changes)
- Add deprecation warnings (Issue #577 timeline)
- Comprehensive test coverage during migration

**Test Coverage**: All 134 BC tests must pass after refactoring.

### Risk 3: Complexity for Subclasses

**Concern**: Template method pattern might be harder to understand.

**Mitigation**:
- Comprehensive docstrings with examples
- Document template method pattern in architecture guide
- Provide subclass implementation examples

---

## Success Criteria

- [x] Analysis complete (duplication patterns identified)
- [ ] Design document approved
- [ ] Template method added to `BaseBCApplicator`
- [ ] Shared ghost cell formulas implemented
- [ ] `FDMApplicator` refactored to use template method
- [ ] All 134 BC tests passing (no regressions)
- [ ] Performance overhead <5%
- [ ] Documentation updated
- [ ] Commit with clear message

---

## Implementation Timeline

**Total**: 2-3 days (12-18 hours)

### Day 1: Template Method Infrastructure (6 hours)
- Add template method to `BaseBCApplicator` (2 hours)
- Add shared ghost cell formulas (2 hours)
- Add validation and buffer creation (1 hour)
- Unit tests for shared methods (1 hour)

### Day 2: FDMApplicator Migration (6 hours)
- Refactor `FDMApplicator._fill_ghost_cells()` (3 hours)
- Update dimension-specific functions to use shared logic (2 hours)
- Run full test suite, fix regressions (1 hour)

### Day 3: Documentation and Finalization (2-4 hours)
- Update architecture documentation (1 hour)
- Add template method examples (1 hour)
- Performance profiling (1 hour)
- Final review and commit (1 hour)

---

## References

- **Issue #598**: BCApplicatorProtocol → ABC refactoring
- **Issue #596**: Trait system (geometry protocol integration)
- **Issue #577**: Function API deprecation (timeline alignment)
- **Issue #542**: Neumann BC fix (example of formula duplication)

---

## Notes

**Why Template Method Pattern?**

Template Method Pattern is ideal for this refactoring because:
1. **Stable workflow**: BC application always follows same steps (validate → create buffer → fill ghosts → finalize)
2. **Dimension-specific variation**: Only slicing logic varies by dimension
3. **Shared formulas**: Ghost cell math is identical across dimensions
4. **Extensibility**: New BC types extend base class without touching dimension logic

**Alternative Considered: Strategy Pattern**

Strategy Pattern (encapsulate ghost cell computation as strategies) was considered but rejected:
- **Pro**: More flexible, each BC type is separate strategy
- **Con**: More complex, requires strategy factory, harder to share validation
- **Verdict**: Template Method simpler for this use case (workflow is stable)

---

**Created**: 2026-01-18
**Author**: Claude Code (Sonnet 4.5)
**Status**: Design Phase (awaiting implementation)
