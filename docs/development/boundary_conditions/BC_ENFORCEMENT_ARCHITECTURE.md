# Boundary Condition Enforcement Architecture

**Status**: Architectural design document
**Version**: v0.17.0+ roadmap
**Related**: Issue #542 (fixed), Issue #549 (planned)

---

## Current State (v0.16.16)

### What Works

**Fixed boundaries on tensor-product grids**:
- ✅ Dirichlet BC: `u(x_min) = g`, `u(x_max) = g`
- ✅ Neumann BC: `∂u/∂n|_{boundary} = g`
- ✅ Mixed BC: Combination of Dirichlet and Neumann
- ✅ Periodic BC: Fallback when `bc=None`

**Implementation**: `base_hjb.py:1109-1153`

### What Doesn't Work

**Limitations**:
- ❌ Free boundaries (moving frontiers)
- ❌ Curved boundaries (manifolds)
- ❌ Non-tensor-product geometries
- ❌ Mesh-based discretizations (FEM/DGM)
- ❌ Implicit surface boundaries

---

## Architectural Problems

### 1. Hardcoded Boundary Indices

**Current**:
```python
U[0] = dirichlet_value  # Assumes left boundary at index 0
U[-1] = dirichlet_value  # Assumes right boundary at index -1
```

**Problem**: Boundary location is hardcoded, not computed from geometry.

**Impact**: Cannot handle:
- Free boundaries where location changes with time
- Irregular domains where boundaries aren't at grid endpoints
- Curved boundaries requiring interpolation

### 2. Coordinate-Based Boundary Identification

**Current**:
```python
BCSegment(boundary="x_min", ...)  # Assumes tensor-product grid
BCSegment(boundary="y_max", ...)  # Coupled to coordinate axes
```

**Problem**: Terminology assumes rectangular domains with axis-aligned boundaries.

**Impact**: Cannot handle:
- Rotated domains
- Curved boundaries
- Triangulated meshes (need face markers, not coordinate names)
- General manifolds (no natural "x_min")

### 3. Monolithic Enforcement Logic

**Current**: BC enforcement is embedded in Newton solver (`solve_hjb_timestep_newton()`).

**Problem**: Tight coupling between solver and BC enforcement.

**Impact**: Cannot:
- Swap enforcement strategies (fixed vs free boundary)
- Reuse enforcement logic across solvers
- Test BC enforcement independently
- Extend to new BC types without modifying core solver

---

## Proposed Architecture (v0.17.0+)

### Design Principles

1. **Separation of Concerns**:
   - `BCSegment`: **What** BC to apply (type, value)
   - `Geometry`: **Where** to apply it (indices/faces/nodes)
   - `BoundaryEnforcer`: **How** to apply it (enforcement algorithm)

2. **Protocol-Based Extension**:
   - Define protocols for each component
   - Allow custom implementations via dependency injection
   - Preserve backward compatibility with default implementations

3. **Geometry-Agnostic BC Specification**:
   - Use abstract boundary identifiers, not coordinate names
   - Geometry provides mapping from IDs to discrete locations
   - Support both legacy strings ("x_min") and new IDs ("inlet")

---

## Component Redesign

### 1. Boundary Enforcer Protocol

**Purpose**: Strategy pattern for BC enforcement.

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class BoundaryEnforcer(Protocol):
    """Protocol for boundary condition enforcement strategies."""

    def get_boundary_indices(
        self,
        bc_segment: BCSegment,
        geometry: Geometry,
        time: float
    ) -> np.ndarray:
        """
        Return indices where BC should be enforced at given time.

        For fixed boundaries: Returns constant indices (e.g., [0, -1])
        For free boundaries: Returns frontier location at time t

        Returns:
            1D array of indices for 1D problems
            Tuple of arrays for nD problems (e.g., (i_indices, j_indices))
        """
        ...

    def apply(
        self,
        U: np.ndarray,
        bc_segment: BCSegment,
        geometry: Geometry,
        time: float,
        **solver_state
    ) -> None:
        """
        Apply boundary condition to solution array.

        Modifies U in-place at boundary locations.

        Args:
            U: Solution array to modify
            bc_segment: BC specification (type, value)
            geometry: Geometry providing spatial information
            time: Current time for time-dependent BC
            **solver_state: Additional solver info (dx, gradient arrays, etc.)
        """
        ...
```

**Implementations**:

```python
class FixedBoundaryEnforcer:
    """Current behavior - boundaries at fixed grid locations."""

    def get_boundary_indices(self, bc_segment, geometry, time):
        # Map boundary ID to grid indices (constant in time)
        if bc_segment.boundary == "x_min":
            return np.array([0])
        elif bc_segment.boundary == "x_max":
            return np.array([-1])
        else:
            return geometry.map_boundary_to_indices(bc_segment.boundary)

    def apply(self, U, bc_segment, geometry, time, **solver_state):
        indices = self.get_boundary_indices(bc_segment, geometry, time)

        if bc_segment.bc_type == BCType.DIRICHLET:
            U[indices] = bc_segment.value

        elif bc_segment.bc_type == BCType.NEUMANN:
            dx = solver_state.get("dx", 1.0)
            # Neumann enforcement logic (current implementation)
            for idx in indices:
                if idx == 0:  # Left
                    U[0] = U[1] - bc_segment.value * dx
                elif idx == len(U) - 1:  # Right
                    U[-1] = U[-2] + bc_segment.value * dx


class FreeBoundaryEnforcer:
    """Free boundary - frontier location evolves with solution."""

    def __init__(self, frontier_tracker):
        """
        Args:
            frontier_tracker: Object tracking frontier location
                             (e.g., level set, moving mesh)
        """
        self.frontier_tracker = frontier_tracker

    def get_boundary_indices(self, bc_segment, geometry, time):
        # Query frontier tracker for current boundary location
        return self.frontier_tracker.locate_boundary(bc_segment.boundary, time)

    def apply(self, U, bc_segment, geometry, time, **solver_state):
        indices = self.get_boundary_indices(bc_segment, geometry, time)

        # Apply BC at current frontier location
        if bc_segment.bc_type == BCType.DIRICHLET:
            U[indices] = bc_segment.value

        # Update frontier location for next time step
        self.frontier_tracker.update(U, time)


class MeshBoundaryEnforcer:
    """Mesh-based enforcement for FEM/DGM on triangular/tetrahedral grids."""

    def get_boundary_indices(self, bc_segment, geometry, time):
        # Get boundary face/node markers from mesh
        return geometry.get_boundary_nodes(bc_segment.boundary_id)

    def apply(self, U, bc_segment, geometry, time, **solver_state):
        # FEM-specific enforcement (e.g., strong vs weak BC)
        boundary_nodes = self.get_boundary_indices(bc_segment, geometry, time)

        if bc_segment.bc_type == BCType.DIRICHLET:
            # Strong enforcement: set DOF values directly
            U[boundary_nodes] = bc_segment.value
        elif bc_segment.bc_type == BCType.NEUMANN:
            # Weak enforcement: modify RHS (done during assembly)
            pass
```

### 2. Geometry Boundary Mapping

**Purpose**: Geometry provides mapping from abstract boundary IDs to discrete locations.

```python
class TensorProductGrid:
    """Current implementation - add boundary mapping method."""

    def map_boundary_to_indices(self, boundary_spec: str | int) -> np.ndarray:
        """
        Map boundary specification to grid indices.

        Args:
            boundary_spec: Either legacy string ("x_min") or boundary ID

        Returns:
            Array of indices where BC should be enforced
        """
        # Legacy string matching (backward compat)
        if boundary_spec == "x_min":
            return np.array([0])
        elif boundary_spec == "x_max":
            return np.array([self.Nx[0] - 1])
        elif boundary_spec == "y_min":
            return self._get_y_min_indices()  # 2D case
        elif boundary_spec == "y_max":
            return self._get_y_max_indices()

        # New boundary ID system
        elif isinstance(boundary_spec, int):
            return self._boundary_id_map[boundary_spec]

        else:
            raise ValueError(f"Unknown boundary: {boundary_spec}")


class TriangularMesh:
    """Future implementation for FEM/DGM."""

    def __init__(self, vertices, faces, boundary_markers):
        self.vertices = vertices
        self.faces = faces
        self.boundary_markers = boundary_markers  # Dict[str, array of face IDs]

    def map_boundary_to_indices(self, boundary_spec):
        """Return node indices on specified boundary."""
        if isinstance(boundary_spec, str):
            # String → boundary marker → face IDs → node IDs
            boundary_faces = self.boundary_markers[boundary_spec]
            return self._faces_to_nodes(boundary_faces)
        else:
            return self.boundary_markers[boundary_spec]
```

### 3. Enhanced BCSegment

**Purpose**: Support both legacy coordinate strings and abstract boundary IDs.

```python
@dataclass
class BCSegment:
    """Boundary condition specification (enhanced for v0.17.0+)."""

    name: str
    bc_type: BCType
    value: float | Callable

    # Legacy field (backward compat)
    boundary: str | None = None  # "x_min", "x_max", etc.

    # New field (v0.17.0+)
    boundary_id: str | int | None = None  # "inlet", "wall", 42, etc.

    # Mesh-based specification (FEM/DGM)
    boundary_faces: list[int] | None = None
    boundary_nodes: list[int] | None = None

    # Implicit boundary specification
    boundary_geometry: dict | None = None  # {"type": "plane", "normal": [...]}

    def get_boundary_spec(self):
        """Return boundary specification (prioritize new over legacy)."""
        if self.boundary_id is not None:
            return self.boundary_id
        elif self.boundary_faces is not None:
            return self.boundary_faces
        elif self.boundary_nodes is not None:
            return self.boundary_nodes
        elif self.boundary is not None:
            return self.boundary  # Legacy
        else:
            raise ValueError("No boundary specification provided")
```

---

## Integration with Solvers

### Current (v0.16.16)

BC enforcement is **embedded** in `solve_hjb_timestep_newton()`:

```python
def solve_hjb_timestep_newton(..., bc: BoundaryConditions | None = None):
    # Newton iteration
    for newton_iter in range(max_newton_iterations):
        # ... solve ...
        pass

    # Hardcoded enforcement
    if bc is not None:
        left_type, left_value = _get_bc_type_and_value_1d(bc, "left", current_time)
        if left_type == BCType.DIRICHLET:
            U[0] = left_value
        # ...
```

### Proposed (v0.17.0+)

BC enforcement is **injected** as strategy:

```python
def solve_hjb_timestep_newton(
    ...,
    bc: BoundaryConditions | None = None,
    bc_enforcer: BoundaryEnforcer | None = None  # ← New parameter
):
    # Default to fixed boundary enforcer
    if bc_enforcer is None:
        bc_enforcer = FixedBoundaryEnforcer()

    # Newton iteration
    for newton_iter in range(max_newton_iterations):
        # ... solve ...
        pass

    # Strategy-based enforcement
    if bc is not None:
        for segment in bc.segments:
            bc_enforcer.apply(
                U=U_n_current_newton_iterate,
                bc_segment=segment,
                geometry=geometry,
                time=current_time,
                dx=dx  # Pass solver state
            )
```

**Benefits**:
- ✅ Decoupled: Can swap enforcement strategy without modifying solver
- ✅ Testable: Can test enforcement independently
- ✅ Extensible: Add new enforcer types without touching core solver
- ✅ Backward compatible: Default enforcer provides current behavior

---

## Migration Path

### Phase 1: Add Protocol and Default Implementation (v0.17.0)

**Changes**:
1. Define `BoundaryEnforcer` protocol in `mfg_pde/geometry/boundary/enforcement.py`
2. Extract current logic into `FixedBoundaryEnforcer`
3. Add `bc_enforcer` parameter to solvers (default = `FixedBoundaryEnforcer()`)
4. Add `map_boundary_to_indices()` to `TensorProductGrid`
5. Add `boundary_id` field to `BCSegment` (optional)

**Backward compatibility**: 100% - existing code uses default enforcer

### Phase 2: Mesh-Based Enforcement (v0.18.0)

**Changes**:
1. Implement `MeshBoundaryEnforcer` for FEM/DGM
2. Add `TriangularMesh` geometry class with boundary markers
3. Add `boundary_faces`/`boundary_nodes` to `BCSegment`

**Use case**: Enable FEM/DGM solvers on triangulated domains

### Phase 3: Free Boundary Support (v0.19.0)

**Changes**:
1. Implement `FreeBoundaryEnforcer`
2. Add frontier tracking utilities (level set, moving mesh)
3. Add examples: Stefan problem, American options

**Use case**: Enable free boundary MFG problems

### Phase 4: Deprecate Legacy Strings (v1.0.0)

**Changes**:
1. Deprecation warnings for `boundary="x_min"` syntax
2. Require `boundary_id` for new code
3. Keep legacy support until v2.0.0

---

## Examples

### Current API (v0.16.16)

```python
from mfg_pde.geometry.boundary import BCSegment, BoundaryConditions
from mfg_pde.geometry.boundary.types import BCType

bc = BoundaryConditions(segments=[
    BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
    BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max"),
])

grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
problem = MFGProblem(geometry=grid, T=1.0, Nt=10)
solver = HJBFDMSolver(problem)
U = solver.solve()  # Uses default FixedBoundaryEnforcer
```

### Future API - Abstract Boundary IDs (v0.17.0+)

```python
bc = BoundaryConditions(segments=[
    BCSegment(name="inlet", bc_type=BCType.DIRICHLET, value=1.0, boundary_id="west"),
    BCSegment(name="outlet", bc_type=BCType.NEUMANN, value=0.0, boundary_id="east"),
])

# Geometry maps IDs to indices
grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
grid.register_boundary_id("west", "x_min")  # Map "west" → x_min
grid.register_boundary_id("east", "x_max")  # Map "east" → x_max

problem = MFGProblem(geometry=grid, T=1.0, Nt=10)
solver = HJBFDMSolver(problem)
U = solver.solve()  # Still uses FixedBoundaryEnforcer
```

### Future API - Free Boundary (v0.19.0+)

```python
from mfg_pde.geometry.boundary.enforcement import FreeBoundaryEnforcer
from mfg_pde.geometry.boundary.frontier import LevelSetTracker

# Define BC on free boundary
bc = BoundaryConditions(segments=[
    BCSegment(name="free_surface", bc_type=BCType.DIRICHLET, value=0.0, boundary_id="frontier"),
])

# Free boundary enforcer with level set tracking
frontier_tracker = LevelSetTracker(initial_location=0.5)
bc_enforcer = FreeBoundaryEnforcer(frontier_tracker)

grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=bc)
problem = MFGProblem(geometry=grid, T=1.0, Nt=10)
solver = HJBFDMSolver(problem)

# Inject free boundary enforcer
U = solver.solve(bc_enforcer=bc_enforcer)

# Query final frontier location
final_frontier = frontier_tracker.get_location(t=1.0)
```

### Future API - Mesh-Based (v0.18.0+)

```python
from mfg_pde.geometry import TriangularMesh
from mfg_pde.geometry.boundary.enforcement import MeshBoundaryEnforcer

# Load mesh with boundary markers
mesh = TriangularMesh.from_gmsh("domain.msh")
# Mesh has boundary markers: {"inlet": [1,2,3], "wall": [4,5,6,7], "outlet": [8,9]}

bc = BoundaryConditions(segments=[
    BCSegment(name="inlet", bc_type=BCType.DIRICHLET, value=1.0, boundary_id="inlet"),
    BCSegment(name="wall", bc_type=BCType.NEUMANN, value=0.0, boundary_id="wall"),
])

bc_enforcer = MeshBoundaryEnforcer()

problem = MFGProblem(geometry=mesh, T=1.0, Nt=10)
solver = HJBDGMSolver(problem)  # DGM solver for triangular mesh
U = solver.solve(bc_enforcer=bc_enforcer)
```

---

## Testing Strategy

### Unit Tests

```python
def test_fixed_boundary_enforcer_dirichlet():
    """Test Dirichlet BC enforcement on fixed boundaries."""
    enforcer = FixedBoundaryEnforcer()

    U = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    bc = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=5.0, boundary="x_min")
    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[5])

    enforcer.apply(U, bc, grid, time=0.0, dx=0.25)

    assert U[0] == 5.0, "Left boundary should be set to 5.0"
    assert U[1:].sum() == 6.0, "Interior should be unchanged"


def test_free_boundary_enforcer():
    """Test free boundary enforcement with moving frontier."""
    frontier_tracker = MockFrontierTracker(initial_location=2)
    enforcer = FreeBoundaryEnforcer(frontier_tracker)

    U = np.zeros(10)
    bc = BCSegment(name="frontier", bc_type=BCType.DIRICHLET, value=1.0, boundary_id="free")
    grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[10])

    enforcer.apply(U, bc, grid, time=0.0)
    assert U[2] == 1.0, "BC should be enforced at frontier location (index 2)"

    # Frontier moves to index 5
    frontier_tracker.set_location(5)
    enforcer.apply(U, bc, grid, time=0.5)
    assert U[5] == 1.0, "BC should be enforced at new frontier location"
```

### Integration Tests

Test scripts in `scripts/`:
- `validate_issue_542_fix.py`: Fixed boundary validation (current)
- `validate_free_boundary_enforcement.py`: Free boundary validation (v0.19.0+)
- `validate_mesh_boundary_enforcement.py`: Mesh-based validation (v0.18.0+)

---

## Performance Considerations

### Overhead

**Additional cost per time step**:
1. Boundary index lookup: O(1) for fixed, O(N) for free boundary
2. BC enforcement: O(B) where B = number of boundary points

**Total**: Negligible compared to Newton iteration (O(N²) for FDM, O(N³) for direct solve)

### Optimization

For performance-critical applications:
- Cache boundary indices for fixed boundaries
- Use JIT compilation for enforcement loops
- Vectorize multi-segment enforcement

---

## Open Questions

1. **Should BC enforcement be before or after Newton iteration?**
   - Current: After (allows Newton to converge freely, then enforce)
   - Alternative: During (enforce at each Newton step)
   - **Decision**: After is correct - BC is part of the problem, not the solver

2. **How to handle time-dependent BC values?**
   - Current: `value` can be `float` or `Callable[[float], float]`
   - **Decision**: Keep callable support, evaluate at current_time

3. **Should Neumann BC modify boundary value or ghost cell?**
   - Current: Modifies boundary value to satisfy gradient
   - Alternative: Use ghost cell (already done in Laplacian)
   - **Decision**: Current approach is correct - ghost cells are for derivatives, boundary values must be set explicitly

4. **How to handle mixed Dirichlet-Neumann at same boundary?**
   - Example: Dirichlet on part of x_min, Neumann on another part
   - **Solution**: Use multiple BCSegments with refined boundary_id specification

---

## References

- Issue #542: FDM BC handling bug (fixed in v0.16.16)
- Issue #549: BC generalization for non-tensor-product geometries
- PR #548: BC-aware Laplacian implementation

---

**Last Updated**: 2026-01-10
**Status**: Design document for v0.17.0+ implementation
**Authors**: Claude Sonnet 4.5 (architectural review for Issue #542)
