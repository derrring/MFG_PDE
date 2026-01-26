# IC/BC Geometry Compatibility Validation

**Issue**: #679
**Status**: Proposed
**Related**: Fail Fast Initiative (Issue #674)

## Problem Statement

When users pass initial/terminal conditions (`m_initial`, `u_final`) or boundary conditions to `MFGComponents`, there's an implicit contract that these must be compatible with the geometry. Currently, this compatibility is only partially validated.

## Current Gaps

### 1. Callable vs NDArray Handling Incomplete

```python
# MFGComponents allows both types:
m_initial: Callable | NDArray | None = None

# But _setup_custom_initial_density() only handles Callable:
initial_func = self.components.m_initial
self.m_initial[i] = initial_func(x_i)  # Crashes if NDArray!
```

**Location**: `mfg_pde/core/mfg_components.py:841-854`

### 2. No Dimension Compatibility Validation

```python
# User may pass wrong dimension function:
m_initial = lambda x: x[0] + x[1]  # 2D function
geometry = TensorProductGrid(bounds=[(0, 1)], ...)  # 1D geometry
# Runtime IndexError, no early detection
```

### 3. Array Shape Validation Missing

```python
# If array passed, no shape check:
m_initial = np.ones(100)  # 100 points
geometry = TensorProductGrid(Nx_points=[51])  # 51 points
# Silent error or later crash
```

## Proposed Solution

### Validation Function

```python
def _validate_condition_compatibility(
    condition: Callable | NDArray,
    name: str,  # "m_initial" or "u_final"
    geometry: GeometryProtocol,
) -> Literal["callable", "array"]:
    """Validate IC/BC compatibility with geometry. Raises ValueError on mismatch."""

    if isinstance(condition, np.ndarray):
        # Array case: validate shape
        expected_shape = geometry.get_grid_shape()
        if condition.shape != expected_shape:
            raise ValueError(
                f"{name} array shape {condition.shape} doesn't match "
                f"geometry shape {expected_shape}. "
                f"Hint: Use geometry.get_grid_shape() to get expected shape."
            )
        return "array"

    elif callable(condition):
        # Callable case: probe signature/dimension
        spatial_grid = geometry.get_spatial_grid()
        sample_point = spatial_grid[0]  # First grid point

        try:
            result = condition(sample_point)
        except (IndexError, TypeError) as e:
            raise ValueError(
                f"{name} function failed on sample point {sample_point}. "
                f"Expected function signature: f(x) where x is {geometry.dimension}D. "
                f"Error: {e}"
            ) from e

        # Check result is scalar
        if not np.isscalar(result):
            raise ValueError(
                f"{name} function must return scalar, got {type(result).__name__}"
            )

        return "callable"

    else:
        raise TypeError(
            f"{name} must be Callable or NDArray, got {type(condition).__name__}"
        )
```

### Integration Points

1. **Validation**: In `_initialize_functions()` before setup
2. **Array handling**: Modify `_setup_custom_initial_density()` and `_setup_custom_final_value()`

```python
def _initialize_functions(self, **kwargs: Any) -> None:
    # === Validate IC/BC compatibility with geometry (Issue #679) ===
    if self.components is not None:
        if self.components.m_initial is not None:
            m_type = _validate_condition_compatibility(
                self.components.m_initial, "m_initial", self.geometry
            )
            if m_type == "array":
                self.m_initial = self.components.m_initial.copy()
            else:
                self._setup_custom_initial_density()
```

## Boundary Condition Compatibility

| Geometry Type | BC Type | Compatibility Check |
|:--------------|:--------|:--------------------|
| TensorProductGrid | Dirichlet/Neumann/Robin | Check dimension match |
| UnstructuredMesh | Boundary segment BC | Check boundary point count |
| PointCloud | N/A (no boundary) | Warn if BC specified |
| Graph/Network | Node BC | Check node IDs exist |

## Acceptance Criteria

- [ ] Array m_initial/u_final works correctly
- [ ] Dimension mismatch raises clear error
- [ ] Shape mismatch raises clear error
- [ ] Callable signature validation with helpful message
- [ ] Tests for all validation cases

## References

- Issue #672: m_initial validation (negativity, zero mass) - **COMPLETED**
- Issue #674: Fail Fast initiative
- `mfg_pde/core/mfg_components.py`: ConditionsMixin
- `mfg_pde/core/mfg_problem.py`: _initialize_functions()
