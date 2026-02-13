# Phase 2.5: Region-Based Mixed Boundary Conditions - Design Document

**Issue**: #596 (Phase 2.5)
**Dependencies**: #590 Phase 1.3 (Region Registry) ✅ Complete
**Status**: Design Phase
**Author**: Claude Code
**Date**: 2026-01-18

---

## 1. Objective

Enable mixed boundary conditions using named regions marked via `SupportsRegionMarking` protocol, allowing flexible BC specification beyond standard rectangular boundaries.

### Current Limitations

**Existing Mixed BC Support**:
```python
# Works: Standard rectangular boundaries
bc = BCSegment(name="left_wall", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min")
bc = BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", region={"y": (4.25, 5.75)})
```

**Gap**: Cannot reference regions marked dynamically via `geometry.mark_region()`:
```python
# NOT CURRENTLY SUPPORTED
geometry.mark_region("inlet", predicate=lambda x: (x[:, 0] < 0.1) & (x[:, 1] > 0.4) & (x[:, 1] < 0.6))
geometry.mark_region("outlet", boundary="x_max")  # Uses standard boundary

bc_inlet = BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0, region_name="inlet")  # ❌ No region_name field
```

### Target API

```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BCSegment, BCType, mixed_bc_from_regions

# 1. Mark regions using SupportsRegionMarking
geometry = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[50, 50])

geometry.mark_region("inlet", predicate=lambda x: (x[:, 0] < 0.1) & (x[:, 1] > 0.4) & (x[:, 1] < 0.6))
geometry.mark_region("outlet", boundary="x_max")
geometry.mark_region("walls", predicate=lambda x: (x[:, 1] < 0.1) | (x[:, 1] > 0.9))

# 2. Define BC segments referencing marked regions
bc_config = {
    "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
    "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
    "walls": BCSegment(name="wall_bc", bc_type=BCType.DIRICHLET, value=0.0),
    "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC, value=0.0)
}

# 3. Create BoundaryConditions object with region mapping
bc = mixed_bc_from_regions(geometry, bc_config)

# 4. Use in solver
problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
problem.boundary_conditions = bc
```

---

## 2. Implementation Strategy

### Option A: Extend BCSegment (Recommended)

**Advantages**:
- Natural extension of existing API
- Minimal changes to applicator logic
- Backward compatible

**Implementation**:
```python
@dataclass
class BCSegment:
    """Boundary condition segment specification."""
    name: str
    bc_type: BCType
    value: float | Callable = 0.0

    # Existing region specifications
    boundary: str | None = None
    region: dict[str, tuple[float, float]] | None = None
    sdf_region: Callable[[np.ndarray], float] | None = None
    normal_direction: np.ndarray | None = None

    # NEW: Reference to marked region (Issue #596 Phase 2.5)
    region_name: str | None = None  # Name from geometry.mark_region()

    # ... rest of fields

    def matches_point(self, x: np.ndarray, geometry: SupportsRegionMarking | None = None) -> bool:
        """Check if point x is in this segment's region."""
        # NEW: Check region_name first
        if self.region_name is not None:
            if geometry is None or not isinstance(geometry, SupportsRegionMarking):
                raise ValueError(f"Segment uses region_name='{self.region_name}' but geometry doesn't support region marking")
            region_mask = geometry.get_region_mask(self.region_name)
            # Find point index and check mask
            # ... implementation details

        # Existing logic for boundary, region, sdf_region, etc.
        # ...
```

**Changes Required**:
1. `mfg_pde/geometry/boundary/types.py`: Add `region_name` field to `BCSegment`
2. `mfg_pde/geometry/boundary/conditions.py`: Pass geometry to `matches_point()` calls
3. `mfg_pde/geometry/boundary/applicator_fdm.py`: Update BC application logic
4. Add helper function `mixed_bc_from_regions()` for convenient creation

---

### Option B: New RegionBC Class

**Advantages**:
- Clean separation of region-based vs standard BCs
- No risk of breaking existing code

**Disadvantages**:
- Duplicates existing BC specification logic
- Parallel type hierarchy complexity

**Not Recommended**: Option A is cleaner and more maintainable.

---

## 3. Detailed Design (Option A)

### 3.1 BCSegment Extension

```python
# mfg_pde/geometry/boundary/types.py

@dataclass
class BCSegment:
    """
    Boundary condition specification for a geometric segment.

    Supports multiple region specification methods:
    - **Standard boundaries**: `boundary="x_min"` for rectangular domains
    - **Coordinate ranges**: `region={"y": (0.4, 0.6)}` for partial boundaries
    - **SDF regions**: `sdf_region=lambda x: ...` for arbitrary shapes
    - **Normal directions**: `normal_direction=np.array([0, 1])` for orientation-based
    - **Marked regions** (NEW): `region_name="inlet"` for geometry-registered regions

    Examples:
        Standard boundary:
        >>> bc = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min")

        Marked region (NEW):
        >>> geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
        >>> bc = BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0, region_name="inlet")
    """
    name: str
    bc_type: BCType
    value: float | Callable = 0.0
    boundary: str | None = None
    region: dict[str, tuple[float, float]] | None = None
    sdf_region: Callable[[np.ndarray], float] | None = None
    normal_direction: np.ndarray | None = None
    normal_tolerance: float = 0.7
    priority: int = 0

    # NEW (Issue #596 Phase 2.5)
    region_name: str | None = None  # References geometry.mark_region(name, ...)

    def __post_init__(self):
        """Validate segment specification."""
        # Count non-None region specs
        specs = [
            self.boundary is not None,
            self.region is not None,
            self.sdf_region is not None,
            self.normal_direction is not None,
            self.region_name is not None,  # NEW
        ]

        if sum(specs) == 0:
            # Default: apply to all boundaries
            pass
        elif sum(specs) > 1:
            raise ValueError(
                f"BCSegment '{self.name}': Only one of boundary, region, sdf_region, "
                f"normal_direction, region_name may be specified"
            )
```

### 3.2 Geometry Region Lookup

```python
# mfg_pde/geometry/boundary/applicator_fdm.py

class FDMBCApplicator1D(BaseStructuredApplicator):
    """Boundary condition applicator for 1D FDM."""

    def apply(
        self,
        field: np.ndarray,
        boundary_conditions: BoundaryConditions,
        time: float = 0.0,
        geometry: SupportsRegionMarking | None = None,  # NEW
    ) -> np.ndarray:
        """
        Apply boundary conditions to 1D field.

        Args:
            field: Interior field values (Nx,)
            boundary_conditions: BC specification
            time: Current time for time-dependent BCs
            geometry: Geometry object supporting region marking (NEW)

        Returns:
            Padded field with ghost cells (Nx+2,)
        """
        # For each BC segment, check if it uses region_name
        for segment in boundary_conditions.segments:
            if segment.region_name is not None:
                # Validate geometry supports region marking
                if geometry is None:
                    raise ValueError(
                        f"Segment '{segment.name}' uses region_name='{segment.region_name}' "
                        f"but no geometry provided to apply()"
                    )
                if not isinstance(geometry, SupportsRegionMarking):
                    raise TypeError(
                        f"Segment '{segment.name}' uses region_name but geometry "
                        f"{type(geometry).__name__} doesn't implement SupportsRegionMarking"
                    )

                # Get region mask from geometry
                region_mask = geometry.get_region_mask(segment.region_name)

                # Apply BC only to points in region
                # ... implementation
```

### 3.3 Helper Function

```python
# mfg_pde/geometry/boundary/conditions.py

def mixed_bc_from_regions(
    geometry: SupportsRegionMarking,
    bc_config: dict[str, BCSegment],
    dimension: int | None = None,
) -> BoundaryConditions:
    """
    Create mixed boundary conditions from marked regions.

    Convenient factory for region-based BCs without manual region_name assignment.

    Args:
        geometry: Geometry with marked regions
        bc_config: Mapping from region name to BC segment
            - Keys are region names from geometry.mark_region()
            - "default" key specifies fallback BC
        dimension: Spatial dimension (inferred from geometry if None)

    Returns:
        BoundaryConditions object with region-based segments

    Example:
        >>> geometry = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[50, 50])
        >>> geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
        >>> geometry.mark_region("outlet", boundary="x_max")
        >>>
        >>> bc_config = {
        ...     "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
        ...     "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
        ...     "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC)
        ... }
        >>>
        >>> bc = mixed_bc_from_regions(geometry, bc_config)
    """
    # Validate geometry supports region marking
    if not isinstance(geometry, SupportsRegionMarking):
        raise TypeError(
            f"mixed_bc_from_regions requires geometry implementing SupportsRegionMarking, "
            f"got {type(geometry).__name__}"
        )

    # Infer dimension if not provided
    if dimension is None:
        dimension = geometry.dimension

    # Separate default BC from region-specific BCs
    default_segment = bc_config.pop("default", None)

    # Create segments with region_name field populated
    segments = []
    for region_name, segment_template in bc_config.items():
        # Verify region exists
        if region_name not in geometry.get_region_names():
            raise ValueError(
                f"Region '{region_name}' not found in geometry. "
                f"Available regions: {geometry.get_region_names()}"
            )

        # Clone segment and set region_name
        segment = replace(segment_template, region_name=region_name)
        segments.append(segment)

    # Create BoundaryConditions object
    return BoundaryConditions(
        dimension=dimension,
        segments=segments,
        default_bc=default_segment.bc_type if default_segment else BCType.PERIODIC,
        default_value=default_segment.value if default_segment else 0.0,
    )
```

---

## 4. Implementation Plan

### Phase 2.5A: Core Infrastructure (1-2 days)

**Tasks**:
1. Add `region_name` field to `BCSegment` dataclass
2. Update `BCSegment.__post_init__()` validation
3. Implement `mixed_bc_from_regions()` helper function
4. Add unit tests for `BCSegment` with `region_name`

**Files Modified**:
- `mfg_pde/geometry/boundary/types.py` (+10 lines)
- `mfg_pde/geometry/boundary/conditions.py` (+60 lines)
- `tests/unit/test_bc_types.py` (+50 lines)

**Success Criteria**:
- ✅ `BCSegment` accepts `region_name` parameter
- ✅ `mixed_bc_from_regions()` creates valid BoundaryConditions
- ✅ All existing BC tests still pass

---

### Phase 2.5B: Applicator Integration (2-3 days)

**Tasks**:
1. Update `FDMBCApplicator1D.apply()` to accept geometry parameter
2. Implement region mask lookup and BC application
3. Extend to 2D and nD applicators
4. Update all solver callsites to pass geometry

**Files Modified**:
- `mfg_pde/geometry/boundary/applicator_fdm.py` (+80 lines)
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` (pass geometry to applicator)
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` (pass geometry to applicator)

**Success Criteria**:
- ✅ Applicators can look up region masks from geometry
- ✅ BCs applied correctly to marked regions
- ✅ Backward compatible (geometry parameter optional)

---

### Phase 2.5C: Integration Tests (1-2 days)

**Tasks**:
1. Create test with inlet/outlet/walls regions
2. Verify BC values at region boundaries
3. Test region intersection and priority handling
4. Performance test (region lookup overhead)

**Files Created**:
- `tests/integration/test_region_based_bc.py` (+200 lines)

**Test Cases**:
1. **Mixed inlet/outlet BC**: Dirichlet inlet, Neumann outlet, periodic walls
2. **Region intersection**: Overlapping regions with priority handling
3. **Dynamic regions**: Regions defined via predicates vs boundaries
4. **Performance**: Region-based vs standard BC application (<5% overhead)

---

### Phase 2.5D: Documentation & Examples (1 day)

**Tasks**:
1. Update BC user guide with region-based BC section
2. Create example: 2D corridor with inlet/outlet/obstacle
3. Update API reference docstrings
4. Add to deprecation/modernization guide

**Files Modified**:
- `docs/user/boundary_conditions_guide.md` (+100 lines)
- `examples/advanced/mixed_bc_regions.py` (new file, ~150 lines)
- `docs/user/DEPRECATION_MODERNIZATION_GUIDE.md` (+30 lines)

**Examples**:
1. **Corridor flow**: Inlet at left, outlet at right, no-flux walls
2. **Obstacle problem**: Dirichlet BC on obstacle interior
3. **Multi-region**: Complex geometry with >3 distinct BC regions

---

## 5. Testing Strategy

### Unit Tests

```python
def test_bc_segment_with_region_name():
    """Test BCSegment accepts region_name parameter."""
    segment = BCSegment(
        name="inlet_bc",
        bc_type=BCType.DIRICHLET,
        value=1.0,
        region_name="inlet"
    )
    assert segment.region_name == "inlet"
    assert segment.boundary is None

def test_mixed_bc_from_regions():
    """Test factory function for region-based BCs."""
    geometry = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[50, 50])
    geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
    geometry.mark_region("outlet", boundary="x_max")

    bc_config = {
        "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
        "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
        "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC)
    }

    bc = mixed_bc_from_regions(geometry, bc_config)
    assert len(bc.segments) == 2
    assert bc.segments[0].region_name == "inlet"
    assert bc.segments[1].region_name == "outlet"
```

### Integration Tests

```python
def test_region_based_bc_application_1d():
    """Test applying BCs to marked regions in 1D."""
    # Setup: 1D grid with inlet/outlet regions
    geometry = TensorProductGrid(dimension=1, bounds=[(0, 10)], Nx_points=[101])
    geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 1.0)
    geometry.mark_region("outlet", predicate=lambda x: x[:, 0] > 9.0)

    # BCs: Dirichlet inlet (u=1), Neumann outlet (du/dx=0), periodic elsewhere
    bc_config = {
        "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
        "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
        "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC)
    }

    bc = mixed_bc_from_regions(geometry, bc_config)

    # Apply to field
    field = np.zeros(101)
    applicator = FDMBCApplicator1D(dimension=1)
    field_with_bc = applicator.apply(field, bc, geometry=geometry)

    # Verify: Inlet has Dirichlet value, outlet has Neumann gradient
    assert np.isclose(field_with_bc[0], 1.0, atol=1e-6)  # Inlet
    assert np.isclose(field_with_bc[-1], field_with_bc[-2], atol=1e-6)  # Outlet (zero gradient)
```

---

## 6. Backward Compatibility

### Existing Code Unaffected

**No breaking changes**:
- `region_name` field defaults to `None`
- Geometry parameter in `apply()` is optional
- Existing BCSegment creation still works:
  ```python
  # Still works (no region_name)
  bc = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min")
  ```

### Migration Path

**Old**: Standard rectangular BC
```python
bc_left = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min")
bc_right = BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max")
bc = mixed_bc([bc_left, bc_right], dimension=1)
```

**New** (optional): Region-based BC
```python
geometry.mark_region("inlet", boundary="x_min")
geometry.mark_region("outlet", boundary="x_max")
bc = mixed_bc_from_regions(geometry, {
    "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
    "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0)
})
```

**Benefits of migration**:
- More flexible region definitions (predicates, unions, intersections)
- Semantic names ("inlet" vs "x_min")
- Works with complex geometries beyond rectangles

---

## 7. Performance Considerations

### Region Mask Lookup

**Concern**: Extra lookup overhead for each BC application
**Mitigation**: Cache region masks in applicator

```python
class FDMBCApplicator1D:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._region_mask_cache: dict[str, np.ndarray] = {}  # Cache masks

    def _get_region_mask(self, geometry: SupportsRegionMarking, region_name: str) -> np.ndarray:
        """Get region mask with caching."""
        cache_key = f"{id(geometry)}:{region_name}"
        if cache_key not in self._region_mask_cache:
            self._region_mask_cache[cache_key] = geometry.get_region_mask(region_name)
        return self._region_mask_cache[cache_key]
```

**Expected Overhead**: <1% (single dict lookup per segment)

---

## 8. Future Extensions

### 8.1 Region Operations in BCSegment

```python
# Union of regions
bc = BCSegment(
    name="walls",
    bc_type=BCType.DIRICHLET,
    value=0.0,
    region_name="top_wall | bottom_wall"  # Union syntax
)

# Intersection
bc = BCSegment(
    name="corner",
    bc_type=BCType.NEUMANN,
    value=0.0,
    region_name="boundary & obstacle"  # Intersection syntax
)
```

**Implementation**: Parse `region_name` for operators, call `geometry.union_regions()` / `geometry.intersect_regions()`

### 8.2 Time-Dependent Regions

```python
# Region that changes over time
geometry.mark_region("moving_obstacle", predicate=lambda x, t: ...)
bc = BCSegment(name="obstacle_bc", bc_type=BCType.DIRICHLET, value=0.0, region_name="moving_obstacle")
```

**Requires**: Time parameter in region predicates (Issue #590 extension)

---

## 9. Success Criteria

### Phase 2.5 Complete When:

- ✅ `BCSegment` supports `region_name` field
- ✅ `mixed_bc_from_regions()` factory function implemented
- ✅ FDM applicators (1D, 2D, nD) support region-based BCs
- ✅ Integration tests pass (inlet/outlet/walls example)
- ✅ Documentation and examples complete
- ✅ No performance regression (<5% overhead)
- ✅ 100% backward compatibility (all existing tests pass)

---

## 10. Timeline

**Total Estimate**: 5-8 days

| Phase | Tasks | Days |
|:------|:------|:-----|
| 2.5A | Core infrastructure | 1-2 |
| 2.5B | Applicator integration | 2-3 |
| 2.5C | Integration tests | 1-2 |
| 2.5D | Documentation | 1 |

**Completion Target**: 2026-01-24 (±2 days)

---

**Status**: Design complete, ready for implementation
**Next Step**: Begin Phase 2.5A (Core Infrastructure)
