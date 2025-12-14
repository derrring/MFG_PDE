# Lipschitz Boundary Condition Implementation Plan

**Date**: 2025-12-02
**Status**: Planning
**GitHub Issue**: #352
**Supersedes**: `MIXED_BC_LIPSCHITZ_EXTENSION.md` (consolidated)

---

## 1. Executive Summary

This document provides the complete implementation roadmap for Lipschitz boundary support in MFG_PDE. Lipschitz boundaries include curves, surfaces, and manifolds with bounded gradients - essential for realistic crowd dynamics, evacuation, and obstacle avoidance problems.

### Current State

| Component | Status | Location |
|:----------|:-------|:---------|
| SDF-based domains | Implemented | `geometry/implicit/` |
| Meshfree BC applicator | Implemented | `boundary/applicator_meshfree.py` |
| Mixed BC for rectangular | Implemented | `boundary/conditions.py`, `types.py` |
| Mask-based region spec | Implemented | `geometry/masks.py`, `types.py:BCSegment` |
| SDF gradient/normal | Partial | Only "simple" projection in `ImplicitDomain` |
| FDM on curved domains | Not implemented | Limited to rectangular grids |

### Target Capabilities

1. **Accurate boundary normals** from SDF gradients
2. **Corner/edge handling** at non-smooth boundary points
3. **Gradient-based projection** to nearest boundary point
4. **Mixed BC on curved domains** via SDF-based region matching

---

## 2. Problem Statement

The current `MixedBoundaryConditions` implementation only supports **rectangular/box domains** with axis-aligned boundaries. This limits applicability to:

- Curved boundaries (circles, ellipses, arbitrary smooth curves)
- Complex polygons
- Domains with obstacles (holes)
- Lipschitz domains (non-smooth boundaries with corners/cusps)
- SDF-defined implicit domains

### Current Implementation Limitations

**`BCSegment.matches_point()`** only checks axis-aligned coordinate ranges:
```python
for axis_key, (range_min, range_max) in self.region.items():
    coord = point[axis_idx]
    if coord < range_min - tolerance or coord > range_max + tolerance:
        return False
```

---

## 3. Design: SDF-Based BC Segments

### 3.1 Core Idea

Use **Signed Distance Functions (SDFs)** to define boundary segments on arbitrary domains:

1. **Boundary detection**: Point is on boundary if `|phi(x)| < tolerance`
2. **Segment matching**: Use SDF gradient `nabla phi` (outward normal) or SDF regions
3. **Corner handling**: Mollify SDF or use priority-based fallback

### 3.2 New `BCSegment` Attributes

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
    normal_direction: np.ndarray | None = None  # E.g., [1, 0] for right-facing
    normal_tolerance: float = 0.5  # cos(angle) threshold

    # Arc-length parameterization for curved boundaries
    arc_param_range: tuple[float, float] | None = None  # (s_min, s_max)

    priority: int = 0
```

### 3.3 Extended `MixedBoundaryConditions`

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

---

## 4. Implementation Phases

### Phase 1: SDF Boundary Normals (Priority: HIGH)

**Goal**: Expose outward normal computation from SDF gradient

**Files to modify**:
- `geometry/implicit/implicit_domain.py`

**Implementation**:

```python
# Add to ImplicitDomain class

def get_boundary_normal(
    self,
    points: NDArray[np.floating],
    epsilon: float = 1e-6,
) -> NDArray[np.floating]:
    """
    Compute outward normal vectors at boundary points using SDF gradient.

    The outward normal is n = nabla(phi) / |nabla(phi)|

    Args:
        points: Array of shape (n, d) - points on or near boundary
        epsilon: Finite difference step for gradient computation

    Returns:
        Array of shape (n, d) - unit outward normal vectors
    """
    points = np.atleast_2d(points)
    n_points, dim = points.shape

    # Compute gradient via central differences
    normals = np.zeros_like(points)

    for d in range(dim):
        offset = np.zeros(dim)
        offset[d] = epsilon

        phi_plus = np.array([self.signed_distance(p + offset) for p in points])
        phi_minus = np.array([self.signed_distance(p - offset) for p in points])

        normals[:, d] = (phi_plus - phi_minus) / (2 * epsilon)

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero at corners

    return normals / norms
```

**Tests**:
- Unit test: Verify normals are outward-pointing for sphere, rectangle
- Unit test: Verify normals have unit length
- Unit test: Check behavior at corners (L-shaped domain)

**Effort**: 1 day

---

### Phase 2: Gradient-Based Projection (Priority: HIGH)

**Goal**: Implement Newton iteration for nearest boundary point projection

**Files to modify**:
- `geometry/implicit/implicit_domain.py`

**Implementation**:

```python
def project_to_boundary(
    self,
    points: NDArray[np.floating],
    method: Literal["simple", "gradient"] = "gradient",
    max_iterations: int = 10,
    tolerance: float = 1e-8,
) -> NDArray[np.floating]:
    """
    Project points onto the domain boundary.

    Methods:
        - "simple": Clip to bounding box (fast but inaccurate for curved)
        - "gradient": Newton iteration along SDF gradient (accurate)

    For "gradient" method:
        x_{k+1} = x_k - phi(x_k) * n(x_k)

    where n(x_k) is the outward normal at x_k.
    """
    if method == "simple":
        return self._project_simple(points)

    points = np.atleast_2d(points).copy()

    for _ in range(max_iterations):
        phi = np.array([self.signed_distance(p) for p in points])

        # Check convergence
        if np.all(np.abs(phi) < tolerance):
            break

        # Gradient step: move along negative gradient by phi distance
        normals = self.get_boundary_normal(points)
        points = points - phi[:, np.newaxis] * normals

    return points
```

**Tests**:
- Unit test: Project point inside sphere to surface
- Unit test: Project point outside sphere to surface
- Unit test: Verify projection error < tolerance
- Unit test: Compare simple vs gradient accuracy on curved domain

**Effort**: 1 day

---

### Phase 3: Corner Normal Handling (Priority: MEDIUM)

**Goal**: Handle non-smooth boundary points (corners, edges)

**Files to modify**:
- `geometry/implicit/implicit_domain.py`
- `geometry/implicit/csg_operations.py`

**Implementation strategies**:

#### Strategy A: Mollification (smooth the SDF)

```python
def get_mollified_normal(
    self,
    point: NDArray[np.floating],
    mollification_radius: float = 0.1,
    n_samples: int = 16,
) -> NDArray[np.floating]:
    """
    Compute mollified normal near corners using local averaging.

    At corners, the SDF gradient is discontinuous. Mollification
    averages gradients over a small neighborhood to get a smoothed normal.
    """
    # Sample points in a ball around the query point
    offsets = np.random.randn(n_samples, self.dimension) * mollification_radius
    nearby_points = point + offsets

    # Compute gradients at nearby points
    normals = self.get_boundary_normal(nearby_points)

    # Average and normalize
    avg_normal = np.mean(normals, axis=0)
    return avg_normal / (np.linalg.norm(avg_normal) + 1e-12)
```

#### Strategy B: Priority-based fallback

For CSG domains (union, intersection, difference), track which primitive
the point is closest to and use that primitive's normal:

```python
def get_boundary_normal(self, points):
    """For CSG domains, use the normal from the nearest primitive."""
    # Compute distance to each primitive
    dist_A = np.abs(self.domain_A.signed_distance(points))
    dist_B = np.abs(self.domain_B.signed_distance(points))

    # Use normal from closest primitive
    use_A = dist_A < dist_B

    normals = np.zeros((len(points), self.dimension))
    if np.any(use_A):
        normals[use_A] = self.domain_A.get_boundary_normal(points[use_A])
    if np.any(~use_A):
        normals[~use_A] = self.domain_B.get_boundary_normal(points[~use_A])

    return normals
```

**Tests**:
- Unit test: L-shaped domain (re-entrant corner)
- Unit test: Room with circular obstacle (smooth + corner)
- Validation: Compare mollified vs priority-based at corners

**Effort**: 2 days

---

### Phase 4: Normal-Based BC Segment Matching (Priority: MEDIUM)

**Goal**: Match BC segments based on outward normal direction

**Files to modify**:
- `geometry/boundary/types.py` (BCSegment)
- `geometry/boundary/conditions.py` (MixedBoundaryConditions)

**Implementation**:

BCSegment already has `sdf_region` support. Add normal-based matching:

```python
def matches_point(
    self,
    point: np.ndarray,
    boundary_id: str | None,
    domain_bounds: np.ndarray | None,
    domain_sdf: Callable | None = None,  # NEW
    domain_geometry: GeometryProtocol | None = None,  # NEW
    tolerance: float = 1e-8,
    axis_names: dict[int, str] | None = None,
) -> bool:
    # ... existing methods 0-3 ...

    # Method 4: Normal direction matching (NEW)
    if self.normal_direction is not None and domain_geometry is not None:
        normal = domain_geometry.get_boundary_normal(point.reshape(1, -1))[0]
        target = self.normal_direction / np.linalg.norm(self.normal_direction)
        cos_angle = np.dot(normal, target)
        if cos_angle < self.normal_tolerance:
            return False

    return True
```

**Effort**: 2 days

---

### Phase 5: FDM on Curved Domains (Priority: LOW)

**Goal**: Enable FDM solvers on curved domains via cut-cell or immersed boundary

**Approach options**:

| Method | Complexity | Accuracy | Best For |
|:-------|:-----------|:---------|:---------|
| Cut-cell FDM | High | High | Interior problems |
| Immersed boundary | Medium | Medium | Moving boundaries |
| Embedded SDF | Low | Low | Approximate |
| Use GFDM/particle | None (existing) | High | Complex geometry |

**Recommendation**: For curved domains, use existing GFDM or particle solvers rather than extending FDM. FDM extension is low priority.

**Effort**: 5+ days (if implemented)

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
    sdf_region=lambda x: sdf_sphere(x, center=[1, 1], radius=0.2),
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
    sdf_region=lambda x: sdf_box(x, bounds=[[9.9, 10.1], [4, 6]]),
    priority=2,
)

# Obstacle boundary: reflecting
obstacle_bc = BCSegment(
    name="obstacle",
    bc_type=BCType.NEUMANN,
    value=0.0,
    sdf_region=lambda x: sdf_sphere(x, center=[5, 5], radius=2.1) - 0.1,
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

## 6. Corner Handling Strategies

### 6.1 Problem: Non-Smooth Boundaries

At corners of Lipschitz domains:
- SDF gradient is **undefined** or **discontinuous**
- Multiple BC segments may "claim" the corner
- Numerical instability in normal computation

### 6.2 Strategy Comparison

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Priority** | First matching segment wins | Simple, predictable | May cause discontinuities |
| **Average** | Blend BC values from adjacent segments | Smooth transition | May not be physically meaningful |
| **Mollify** | Smooth the SDF near corners | Well-defined normals | Modifies domain geometry |

---

## 7. Integration with Existing Solvers

### Recommended Solver Combinations

| Domain Type | HJB Solver | FP Solver | BC Applicator |
|:------------|:-----------|:----------|:--------------|
| Rectangular | FDM | FDM | `FDMApplicator` |
| Curved/Lipschitz | GFDM | Particle | `MeshfreeApplicator` |
| Mixed (rect + obstacle) | GFDM | Particle | `MeshfreeApplicator` |
| Network/graph | Network | Network | `GraphApplicator` |

### HybridMFGSolver Integration

```python
from mfg_pde.alg.coupling.hybrid_solver import HybridMFGSolver
from mfg_pde.geometry.boundary import MixedBoundaryConditions, BCSegment

# Create domain with circular obstacle
domain = DifferenceDomain(
    Hyperrectangle([[0, 10], [0, 10]]),
    Hypersphere(center=[5, 5], radius=2)
)

# Mixed BC: reflecting walls, absorbing exit
bc = MixedBoundaryConditions(
    dimension=2,
    segments=[
        BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
                  normal_direction=[1, 0], normal_tolerance=0.7),
        BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0),
    ],
    domain_geometry=domain,
)

solver = HybridMFGSolver(
    problem=problem,
    hjb_solver_type="gfdm",
    fp_solver_type="particle",
    boundary_conditions=bc,
)
```

---

## 8. Testing Strategy

### Unit Tests

| Test | File | Description |
|:-----|:-----|:------------|
| `test_sdf_normal_sphere` | `test_implicit_domain.py` | Normal computation on sphere |
| `test_sdf_normal_rectangle` | `test_implicit_domain.py` | Normal computation on rectangle |
| `test_gradient_projection` | `test_implicit_domain.py` | Projection accuracy |
| `test_corner_normal` | `test_implicit_domain.py` | Corner handling |
| `test_normal_bc_matching` | `test_mixed_bc.py` | Normal-based segment matching |

### Integration Tests

| Test | File | Description |
|:-----|:-----|:------------|
| `test_circle_evacuation` | `test_curved_bc.py` | Crowd evacuation from circular room |
| `test_l_shaped_domain` | `test_curved_bc.py` | HJB on L-shaped domain |
| `test_room_with_obstacle` | `test_curved_bc.py` | Navigation around obstacle |

### Validation Tests

| Test | Description | Success Criterion |
|:-----|:------------|:------------------|
| Mass conservation | FP on curved domain | Relative error < 5% |
| Value function monotonicity | HJB backward solve | Monotonic toward exit |
| BC satisfaction | Check BC at boundary points | Exact for Dirichlet |

---

## 9. Mathematical Background

### 9.1 Lipschitz Domains

A domain Omega has **Lipschitz boundary** if:
- Locally, the boundary can be represented as the graph of a Lipschitz function
- Equivalent: The boundary normal exists almost everywhere and is bounded

**Implications for BCs**:
- Well-posedness of elliptic/parabolic PDEs is guaranteed
- Corners have undefined normal but finite "cone" of directions
- Trace theorems hold (boundary values are well-defined)

### 9.2 SDF Properties

For an exact SDF phi:
- `|nabla phi| = 1` almost everywhere
- `nabla phi` points in the direction of steepest ascent (outward normal)
- Near corners, `nabla phi` is discontinuous but bounded

### 9.3 Numerical Considerations

- **Tolerance selection**: Should be O(h) where h is mesh spacing
- **Normal computation**: Use central differences with small epsilon
- **Corner detection**: `|nabla phi|` is small or varies rapidly

---

## 10. Timeline and Priorities

### Immediate (Week 1)

1. **Phase 1**: SDF boundary normals - HIGH priority, foundation for all else
2. **Phase 2**: Gradient-based projection - HIGH priority, enables accurate BC

### Short-term (Week 2-3)

3. **Phase 3**: Corner normal handling - MEDIUM priority, needed for L-shaped domains
4. **Phase 4**: Normal-based BC matching - MEDIUM priority, enables mixed BC on curves

### Deferred

5. **Phase 5**: FDM on curved domains - LOW priority, use GFDM/particle instead

---

## 11. Success Criteria

**MVP (Phases 1-2 complete)**:
- SDF-based domains expose accurate boundary normals
- Gradient-based projection has error < 1e-6
- Particle reflector uses accurate normals

**Full Success (Phases 1-4 complete)**:
- Corner handling works for L-shaped domain
- Normal-based BC segment matching works for circular domain
- Integration test passes: crowd evacuation with curved walls

---

## 12. References

### Code Locations

- `mfg_pde/geometry/implicit/implicit_domain.py` - Base ImplicitDomain class
- `mfg_pde/geometry/implicit/csg_operations.py` - CSG operations (union, difference)
- `mfg_pde/geometry/boundary/applicator_meshfree.py` - Particle BC applicator
- `mfg_pde/geometry/boundary/types.py` - BCSegment dataclass
- `mfg_pde/geometry/boundary/conditions.py` - MixedBoundaryConditions

### Documentation

- `docs/development/BC_GEOMETRY_ROADMAP_GAP_ANALYSIS.md` - Status tracking
- `docs/development/GEOMETRY_MODULE_ARCHITECTURE.md` - Module architecture
- `docs/user/dual_geometry_usage.md` - User guide for geometry
