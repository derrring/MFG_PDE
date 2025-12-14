# Mixed BC Extension: General Domains and Lipschitz Boundaries

**Date**: 2025-11-27
**Status**: Proposal
**Depends On**: `MIXED_BC_DESIGN.md` (Phase 1 complete)

---

## 1. Problem Statement

The current `MixedBoundaryConditions` implementation only supports **rectangular/box domains** with axis-aligned boundaries. This limits applicability to:

- Curved boundaries (circles, ellipses, arbitrary smooth curves)
- Complex polygons
- Domains with obstacles (holes)
- Lipschitz domains (non-smooth boundaries with corners/cusps)
- SDF-defined implicit domains

### Current Implementation Limitations

**`BCSegment.matches_point()`** (`mixed_bc.py:98-149`):
```python
# Only checks axis-aligned coordinate ranges
for axis_key, (range_min, range_max) in self.region.items():
    coord = point[axis_idx]
    if coord < range_min - tolerance or coord > range_max + tolerance:
        return False
```

**`MixedBoundaryConditions.identify_boundary_id()`** (`mixed_bc.py:276-303`):
```python
# Only detects axis min/max boundaries
if abs(point[axis_idx] - self.domain_bounds[axis_idx, 0]) < tolerance:
    return f"{axis_name}_min"
```

---

## 2. Proposed Extension: SDF-Based BC Segments

### 2.1 Core Idea

Use **Signed Distance Functions (SDFs)** to define boundary segments on arbitrary domains:

1. **Boundary detection**: Point is on boundary if `|phi(x)| < tolerance`
2. **Segment matching**: Use SDF gradient `nabla phi` (outward normal) or SDF regions
3. **Corner handling**: Mollify SDF or use priority-based fallback

### 2.2 New `BCSegment` Attributes

```python
@dataclass
class BCSegment:
    """Extended BC segment supporting general domains."""

    name: str
    bc_type: BCType
    value: float | Callable = 0.0

    # === Existing (rectangular domains) ===
    boundary: str | None = None  # "left", "right", "x_min", etc.
    region: dict[str | int, tuple[float, float]] | None = None

    # === NEW (general domains) ===

    # SDF-based region (negative = inside segment region)
    sdf_region: Callable[[np.ndarray], float] | None = None

    # Normal-based matching: segment applies where normal is in this direction
    # E.g., normal_direction=[1, 0] matches right-facing boundaries
    normal_direction: np.ndarray | None = None
    normal_tolerance: float = 0.5  # cos(angle) threshold

    # Arc-length parameterization for curved boundaries
    arc_param_range: tuple[float, float] | None = None  # (s_min, s_max)

    priority: int = 0
```

### 2.3 Extended Matching Logic

```python
def matches_point(
    self,
    point: np.ndarray,
    boundary_id: str | None,  # Can be None for SDF domains
    domain_bounds: np.ndarray | None,
    domain_sdf: Callable | None = None,  # NEW: domain SDF
    tolerance: float = 1e-8,
    axis_names: dict[int, str] | None = None,
) -> bool:
    """Check if this BC segment applies to a given boundary point."""

    # Method 1: Axis-aligned boundary ID (existing)
    if self.boundary is not None and boundary_id is not None:
        if self.boundary != boundary_id and self.boundary != "all":
            return False

    # Method 2: Coordinate range check (existing)
    if self.region is not None:
        # ... existing range check logic ...

    # Method 3: SDF region check (NEW)
    if self.sdf_region is not None:
        if self.sdf_region(point) > tolerance:
            return False

    # Method 4: Normal direction check (NEW)
    if self.normal_direction is not None and domain_sdf is not None:
        # Compute outward normal via SDF gradient
        normal = sdf_gradient(point.reshape(1, -1), domain_sdf).ravel()
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        # Check if normal aligns with specified direction
        target_dir = self.normal_direction / np.linalg.norm(self.normal_direction)
        cos_angle = np.dot(normal, target_dir)
        if cos_angle < self.normal_tolerance:
            return False

    return True
```

---

## 3. Extended `MixedBoundaryConditions`

### 3.1 New Attributes

```python
@dataclass
class MixedBoundaryConditions:
    """Mixed BC supporting both rectangular and general domains."""

    dimension: int
    segments: list[BCSegment] = field(default_factory=list)
    default_bc: BCType = BCType.PERIODIC
    default_value: float = 0.0

    # === Existing ===
    domain_bounds: np.ndarray | None = None  # For rectangular domains

    # === NEW ===
    domain_sdf: Callable[[np.ndarray], float] | None = None  # For general domains
    domain_geometry: ImplicitDomain | None = None  # Full geometry object

    # Corner handling strategy
    corner_strategy: Literal["priority", "average", "mollify"] = "priority"
    corner_mollification_radius: float = 0.1  # For "mollify" strategy
```

### 3.2 SDF-Based Boundary Identification

```python
def identify_boundary_id(
    self,
    point: np.ndarray,
    tolerance: float = 1e-8
) -> str | None:
    """
    Identify which boundary a point lies on.

    For SDF domains, returns a synthetic ID based on normal direction.
    """
    # Rectangular domain: use axis-aligned detection (existing)
    if self.domain_bounds is not None and self.domain_sdf is None:
        # ... existing logic ...

    # SDF domain: detect boundary via |phi| < tolerance
    if self.domain_sdf is not None:
        phi = self.domain_sdf(point)
        if abs(phi) > tolerance:
            return None  # Not on boundary

        # Compute normal direction
        normal = sdf_gradient(point.reshape(1, -1), self.domain_sdf).ravel()

        # Map normal to boundary ID
        return self._normal_to_boundary_id(normal)

    return None

def _normal_to_boundary_id(self, normal: np.ndarray) -> str:
    """Map outward normal to boundary identifier."""
    # Find dominant axis
    abs_normal = np.abs(normal)
    dominant_axis = np.argmax(abs_normal)
    axis_name = ["x", "y", "z", "w"][dominant_axis] if dominant_axis < 4 else f"axis{dominant_axis}"

    # Determine direction
    if normal[dominant_axis] > 0:
        return f"{axis_name}_max"
    else:
        return f"{axis_name}_min"
```

---

## 4. Corner Handling Strategies

### 4.1 Problem: Non-Smooth Boundaries

At corners of Lipschitz domains:
- SDF gradient is **undefined** or **discontinuous**
- Multiple BC segments may "claim" the corner
- Numerical instability in normal computation

### 4.2 Strategy Options

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Priority** | First matching segment wins | Simple, predictable | May cause discontinuities |
| **Average** | Blend BC values from adjacent segments | Smooth transition | May not be physically meaningful |
| **Mollify** | Smooth the SDF near corners | Well-defined normals | Modifies domain geometry |

### 4.3 Mollification Implementation

```python
def get_mollified_sdf(
    self,
    point: np.ndarray,
    mollification_radius: float,
) -> float:
    """
    Compute mollified (smoothed) SDF near corners.

    Uses convolution with smooth kernel to regularize SDF gradient.
    """
    # Sample nearby points
    n_samples = 10
    offsets = np.random.randn(n_samples, self.dimension) * mollification_radius
    nearby_points = point + offsets

    # Average SDF values
    sdf_values = np.array([self.domain_sdf(p) for p in nearby_points])
    return np.mean(sdf_values)
```

---

## 5. Usage Examples

### 5.1 Circular Domain with Exit

```python
from mfg_pde.geometry import BCSegment, BCType, MixedBoundaryConditions
from mfg_pde.utils.numerical import sdf_sphere

# Define circular domain
def circle_sdf(x):
    return sdf_sphere(x, center=[0, 0], radius=5.0)

# Exit segment: top of circle (normal pointing up)
exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=0.0,
    normal_direction=np.array([0, 1]),  # Upward normal
    normal_tolerance=0.7,  # ~45 degree cone
    priority=1,
)

# Walls: everywhere else
wall_bc = BCSegment(
    name="walls",
    bc_type=BCType.NEUMANN,
    value=0.0,
    priority=0,
)

mixed_bc = MixedBoundaryConditions(
    dimension=2,
    segments=[exit_bc, wall_bc],
    domain_sdf=circle_sdf,
)
```

### 5.2 L-Shaped Domain (Lipschitz)

```python
from mfg_pde.utils.numerical import sdf_box, sdf_difference

# L-shaped domain: large box minus small box at corner
def l_shape_sdf(x):
    large_box = sdf_box(x, bounds=[[0, 2], [0, 2]])
    cutout = sdf_box(x, bounds=[[1, 2], [1, 2]])
    return sdf_difference(large_box, cutout)

# Different BC on each face
left_bc = BCSegment(
    name="left",
    bc_type=BCType.DIRICHLET,
    value=1.0,
    normal_direction=np.array([-1, 0]),
    priority=1,
)

bottom_bc = BCSegment(
    name="bottom",
    bc_type=BCType.NEUMANN,
    value=0.0,
    normal_direction=np.array([0, -1]),
    priority=1,
)

# Handle re-entrant corner specially
corner_bc = BCSegment(
    name="reentrant_corner",
    bc_type=BCType.NEUMANN,
    value=0.0,
    sdf_region=lambda x: sdf_sphere(x, center=[1, 1], radius=0.2),  # Small ball at corner
    priority=2,  # Highest priority
)

mixed_bc = MixedBoundaryConditions(
    dimension=2,
    segments=[corner_bc, left_bc, bottom_bc],
    domain_sdf=l_shape_sdf,
    corner_strategy="mollify",
    corner_mollification_radius=0.1,
)
```

### 5.3 Domain with Circular Obstacle

```python
# Room with circular obstacle
def room_with_obstacle_sdf(x):
    room = sdf_box(x, bounds=[[0, 10], [0, 10]])
    obstacle = sdf_sphere(x, center=[5, 5], radius=2.0)
    return sdf_difference(room, obstacle)

# Exit on right wall
exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=0.0,
    normal_direction=np.array([1, 0]),
    sdf_region=lambda x: sdf_box(x, bounds=[[9.9, 10.1], [4, 6]]),  # Right wall, y in [4,6]
    priority=2,
)

# Obstacle boundary: reflecting
obstacle_bc = BCSegment(
    name="obstacle",
    bc_type=BCType.NEUMANN,
    value=0.0,
    sdf_region=lambda x: sdf_sphere(x, center=[5, 5], radius=2.1) - 0.1,  # Near obstacle
    priority=1,
)

# Outer walls
wall_bc = BCSegment(
    name="walls",
    bc_type=BCType.NEUMANN,
    value=0.0,
    priority=0,
)
```

---

## 6. Integration with Existing Code

### 6.1 Backward Compatibility

- All existing rectangular domain usage continues to work unchanged
- `domain_sdf=None` (default) triggers existing axis-aligned logic
- New attributes are optional

### 6.2 Required Changes

| File | Changes |
|------|---------|
| `mixed_bc.py` | Add new attributes, extend `matches_point()`, add corner handling |
| `geometry/__init__.py` | Export new utilities |
| `tests/unit/test_geometry/test_mixed_bc.py` | Add tests for SDF-based segments |

### 6.3 Dependencies

- `sdf_gradient` from `mfg_pde.utils.numerical.sdf_utils`
- `ImplicitDomain` from `mfg_pde.geometry.implicit`

---

## 7. Implementation Plan

### Phase 2a: SDF Segment Matching (Priority: High)

1. Add `sdf_region` attribute to `BCSegment`
2. Extend `matches_point()` to use SDF regions
3. Add tests for circular domain with segments

### Phase 2b: Normal-Based Matching (Priority: Medium)

1. Add `normal_direction`, `normal_tolerance` to `BCSegment`
2. Implement `_normal_to_boundary_id()`
3. Test on domains where normal direction identifies faces

### Phase 2c: Corner Handling (Priority: Medium)

1. Add `corner_strategy` to `MixedBoundaryConditions`
2. Implement `get_mollified_sdf()`
3. Test on L-shaped domain

### Phase 2d: Solver Integration (Priority: High)

1. Modify HJB solver to use new `MixedBoundaryConditions`
2. Modify FP solver
3. Full integration tests

---

## 8. Mathematical Considerations

### 8.1 Lipschitz Domains

A domain Omega has **Lipschitz boundary** if:
- Locally, the boundary can be represented as the graph of a Lipschitz function
- Equivalent: The boundary normal exists almost everywhere and is bounded

**Implications for BCs**:
- Well-posedness of elliptic/parabolic PDEs is guaranteed
- Corners have undefined normal but finite "cone" of directions
- Trace theorems hold (boundary values are well-defined)

### 8.2 SDF Properties

For an exact SDF phi:
- `|nabla phi| = 1` almost everywhere
- `nabla phi` points in the direction of steepest ascent (outward normal)
- Near corners, `nabla phi` is discontinuous but bounded

### 8.3 Numerical Considerations

- **Tolerance selection**: Should be O(h) where h is mesh spacing
- **Normal computation**: Use central differences with small epsilon
- **Corner detection**: `|nabla phi|` is small or varies rapidly

---

## 9. Success Criteria

**MVP (Phase 2a)**:
- SDF-based segment matching works for simple domains
- Circular domain with exit/wall BC validates correctly

**Full Success (Phase 2d)**:
- HJB and FP solvers handle general domains
- L-shaped domain (Lipschitz) works with corner handling
- Integration tests pass

---

## 10. References

- `mfg_pde/utils/numerical/sdf_utils.py` - SDF primitives and gradient
- `mfg_pde/geometry/implicit/implicit_domain.py` - ImplicitDomain base class
- `mfg_pde/geometry/boundary/mixed_bc.py` - Current implementation
- `docs/development/MIXED_BC_DESIGN.md` - Phase 1 design
