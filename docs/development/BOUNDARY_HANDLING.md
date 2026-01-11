# Unified Boundary Condition Handling Workflow

**Issue**: #545 (Mixin Refactoring - Phase 1)
**Created**: 2026-01-11
**Status**: Active Reference Document

## Purpose

This document defines the **unified workflow** for boundary condition handling across all MFG_PDE solvers. It eliminates duplicate BC detection logic and mixin hell by standardizing how solvers use the existing geometry and BC infrastructure.

## Problem: Mixin Hell

**Before (GFDM Solver - 5 class hierarchy)**:
```python
class HJBGFDMSolver(
    GFDMInterpolationMixin,   # Grid mapping
    GFDMStencilMixin,          # Stencil computation
    GFDMBoundaryMixin,         # BC handling ← IMPLICIT STATE SHARING
    MonotonicityMixin,         # Monotonicity enforcement
    BaseHJBSolver              # Core logic
):
    pass  # Logic scattered across 5 files

# GFDMBoundaryMixin expects 8+ attributes from host class:
# - collocation_points, dimension, domain_bounds, boundary_indices,
# - boundary_conditions, neighborhoods, _use_ghost_nodes, problem
```

**Issues**:
1. **Implicit Contracts**: Mixins expect attributes from host class
2. **Duplicate Detection**: Each solver implements own boundary detection
3. **Custom Normals**: Duplicate `geometry.get_boundary_normal()` logic
4. **Scattered Logic**: BC handling across 5 files
5. **Hard to Trace**: Data flow unclear

## Solution: Unified Workflow

**Use existing infrastructure**:
1. `GeometryProtocol` - Boundary detection and normals
2. `BCApplicator` - BC application for each discretization type
3. **Composition** over inheritance

## Infrastructure Overview

### 1. GeometryProtocol (`mfg_pde/geometry/protocol.py`)

**Core Boundary Methods** (mandatory for all geometries):
```python
class GeometryProtocol(Protocol):
    def is_on_boundary(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> NDArray[np.bool_]:
        """Check if points are on boundary. Returns boolean array."""
        ...

    def get_boundary_normal(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Get outward unit normals at boundary points."""
        ...

    def project_to_boundary(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Project points onto boundary."""
        ...

    def get_boundary_regions(self) -> dict[str, dict]:
        """Get named boundary regions for mixed BCs."""
        ...

    def get_boundary_conditions(self):
        """Get BoundaryConditions object for this geometry."""
        ...
```

**Helper Methods** (added in v0.16.17 for Issue #545):
```python
    def get_boundary_indices(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> NDArray[np.intp]:
        """
        Get indices of boundary points (convenience wrapper).

        Returns:
            Array of indices (not boolean mask)
        """
        on_boundary = self.is_on_boundary(points, tolerance)
        return np.where(on_boundary)[0]

    def get_boundary_info(
        self,
        points: NDArray[np.floating],
        tolerance: float = 1e-10,
    ) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
        """
        Get boundary indices AND normals in one call.

        Returns:
            (boundary_indices, normals) tuple
        """
        boundary_indices = self.get_boundary_indices(points, tolerance)
        if len(boundary_indices) == 0:
            return boundary_indices, np.array([]).reshape(0, self.dimension)

        boundary_points = points[boundary_indices]
        normals = self.get_boundary_normal(boundary_points)
        return boundary_indices, normals
```

**Default Implementation**: Available in `Geometry` base class (`mfg_pde/geometry/base.py`) - all geometries inherit these for free.

### 2. BC Applicator Infrastructure (`mfg_pde/geometry/boundary/`)

**BCApplicatorProtocol**:
```python
@runtime_checkable
class BCApplicatorProtocol(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def discretization_type(self) -> DiscretizationType: ...

    def supports_bc_type(self, bc: BoundaryConditions) -> bool: ...
```

**Applicator Implementations**:
- `applicator_fdm.py` - FDM ghost cells
- `applicator_fem.py` - FEM matrix modification
- `applicator_particle.py` - Particle reflection
- `applicator_meshfree.py` - GFDM/collocation
- `applicator_graph.py` - Network/graph BCs

### 3. BoundaryCapable Protocol (for Solvers)

```python
@runtime_checkable
class BoundaryCapable(Protocol):
    @property
    def supported_bc_types(self) -> frozenset: ...

    @property
    def boundary_conditions(self) -> BoundaryConditions | None: ...

    @property
    def discretization_type(self) -> DiscretizationType: ...
```

## Unified Workflow for Solvers

### Pattern 1: Particle Solver (Simple Reflection)

**Use Case**: Particle-based FP solver with reflection BC

```python
class FPParticleSolver:
    """
    Particle solver using composition (not mixins).
    """

    def __init__(self, problem: MFGProblem, **kwargs):
        self.problem = problem
        self.geometry = problem.geometry
        self.bc = problem.geometry.get_boundary_conditions()

        # Initialize particles
        self.particles = self._initialize_particles()
        self.velocities = np.zeros_like(self.particles)

    def _handle_boundary_reflection(self, dt: float):
        """
        Apply boundary reflection using geometry methods.

        OLD WAY (custom detection):
            for i, particle in enumerate(self.particles):
                if self._is_near_boundary(particle):  # Custom logic
                    normal = self._compute_normal(particle)  # Duplicate
                    self.velocities[i] = self._reflect(...)

        NEW WAY (use geometry):
        """
        # Step 1: Detect boundary particles using geometry
        boundary_indices, normals = self.geometry.get_boundary_info(
            self.particles,
            tolerance=1e-10
        )

        # Step 2: Apply reflection
        for idx, normal in zip(boundary_indices, normals):
            # Reflect velocity: v_new = v - 2(v·n)n
            velocity = self.velocities[idx]
            v_normal = np.dot(velocity, normal)
            self.velocities[idx] = velocity - 2.0 * v_normal * normal

        # Step 3: Project particles back inside domain
        if len(boundary_indices) > 0:
            outside = self.particles[boundary_indices]
            self.particles[boundary_indices] = self.geometry.project_to_interior(outside)
```

### Pattern 2: GFDM Solver (Collocation + Stencils)

**Use Case**: GFDM solver with Neumann/Dirichlet BCs

```python
class HJBGFDMSolver:
    """
    GFDM solver using composition (not mixins).
    """

    def __init__(self, problem: MFGProblem, collocation_points: np.ndarray, **kwargs):
        self.problem = problem
        self.geometry = problem.geometry
        self.bc = problem.geometry.get_boundary_conditions()
        self.collocation_points = collocation_points

        # Detect boundary points using geometry (not custom logic)
        self._detect_boundary_points()

        # Create BC applicator (composition, not mixin)
        from mfg_pde.geometry.boundary import create_applicator
        self.bc_applicator = create_applicator(
            bc=self.bc,
            geometry=self.geometry,
            discretization_type=DiscretizationType.GFDM
        )

    def _detect_boundary_points(self):
        """
        Detect boundary points using geometry methods.

        OLD WAY (GFDMBoundaryMixin):
            self.boundary_indices = []
            for i, point in enumerate(self.collocation_points):
                for d in range(self.dimension):
                    if abs(point[d] - bounds[d][0]) < tol:
                        self.boundary_indices.append(i)
                        break

        NEW WAY (use geometry):
        """
        # Get boundary info from geometry
        self.boundary_indices, self.boundary_normals = \\
            self.geometry.get_boundary_info(self.collocation_points)

        # Cache for performance
        self._boundary_points = self.collocation_points[self.boundary_indices]

    def _apply_boundary_conditions(self, U: np.ndarray) -> np.ndarray:
        """
        Apply BCs using geometry + applicator (not custom logic).

        OLD WAY (scattered across mixins):
            if self._use_ghost_nodes:
                self._apply_ghost_node_bc(...)
            elif self._use_wind_dependent_bc:
                self._apply_wind_bc(...)
            else:
                self._compute_outward_normal(...)  # Duplicate geometry.get_boundary_normal

        NEW WAY (use applicator):
        """
        if self.bc is None:
            return U

        # Use BC applicator (delegates to appropriate method)
        U_with_bc = self.bc_applicator.apply(
            values=U,
            geometry=self.geometry,
            solver_data={'collocation_points': self.collocation_points}
        )

        return U_with_bc
```

### Pattern 3: FDM Solver (Ghost Cells)

**Use Case**: FDM solver with ghost cell BC enforcement

```python
class HJBFDMSolver:
    """
    FDM solver - already uses good pattern (centralized get_ghost_values_nd).
    """

    def __init__(self, problem: MFGProblem, **kwargs):
        self.problem = problem
        self.geometry = problem.geometry
        self.bc = problem.geometry.get_boundary_conditions()

        # FDM-specific grid info
        self.grid_shape = self.geometry.get_grid_shape()
        self.spacing = self.geometry.spacing  # For CartesianGrid

    def _apply_boundary_conditions(self, U: np.ndarray) -> np.ndarray:
        """
        FDM uses centralized ghost cell method (GOOD PATTERN - keep it).
        """
        from mfg_pde.geometry.boundary import get_ghost_values_nd

        # Apply ghost cells
        U_with_ghost = get_ghost_values_nd(
            U=U,
            bc=self.bc,
            geometry=self.geometry,
            spacing=self.spacing
        )

        return U_with_ghost
```

## Key Principles

### 1. Use Geometry for Boundary Detection

**❌ DON'T**: Implement custom boundary detection in each solver
```python
# BAD - duplicates geometry logic
def _is_on_boundary_custom(self, point):
    tol = 1e-10
    for d in range(self.dimension):
        if abs(point[d] - self.domain_bounds[d][0]) < tol:
            return True
    return False
```

**✅ DO**: Use geometry's boundary methods
```python
# GOOD - uses centralized geometry method
boundary_mask = self.geometry.is_on_boundary(points)
boundary_indices = self.geometry.get_boundary_indices(points)
```

### 2. Use Geometry for Normals

**❌ DON'T**: Compute normals manually
```python
# BAD - duplicates SDF gradient logic
def _compute_normal_custom(self, point):
    normal = np.zeros(self.dimension)
    for d in range(self.dimension):
        if abs(point[d] - low) < tol:
            normal[d] = -1.0
    return normal / np.linalg.norm(normal)
```

**✅ DO**: Use geometry's normal method
```python
# GOOD - geometry knows its own boundary structure
normals = self.geometry.get_boundary_normal(boundary_points)
```

### 3. Use Composition Over Mixins

**❌ DON'T**: Deep mixin hierarchies with implicit state
```python
# BAD - implicit contracts, scattered logic
class MySolver(MixinA, MixinB, MixinC, BaseSolver):
    pass  # Where is boundary_indices defined? Who knows!
```

**✅ DO**: Explicit composition with clear data flow
```python
# GOOD - explicit composition, clear ownership
class MySolver:
    def __init__(self, problem):
        self.geometry = problem.geometry  # Explicit
        self.bc = problem.geometry.get_boundary_conditions()  # Clear source
        self.bc_applicator = create_applicator(...)  # Composition
```

### 4. One Method, One Purpose

**❌ DON'T**: Methods that do everything
```python
# BAD - mixin method does detection + application + modification
def _handle_boundary(self):
    self._detect()  # Duplicates geometry
    self._compute_normals()  # Duplicates geometry
    self._apply_bc()  # Mixed with above
```

**✅ DO**: Separate concerns, delegate to experts
```python
# GOOD - geometry does detection, applicator does application
def _setup_boundary(self):
    self.boundary_indices, self.boundary_normals = \\
        self.geometry.get_boundary_info(self.points)  # Geometry's job

def _apply_boundary_conditions(self, U):
    return self.bc_applicator.apply(U, ...)  # Applicator's job
```

## Migration Guide

### Refactoring Checklist

**For each solver with custom BC handling**:

1. ✅ **Identify current BC detection logic**
   - Search for: `boundary_indices`, `_is_on_boundary`, boundary loops
   - Note: What does it detect? How?

2. ✅ **Replace with geometry methods**
   - Before: Custom detection loops
   - After: `boundary_indices = geometry.get_boundary_indices(points)`

3. ✅ **Replace normal computation**
   - Before: Custom `_compute_outward_normal()`
   - After: `normals = geometry.get_boundary_normal(boundary_points)`

4. ✅ **Use helper method for common case**
   - Before: Two separate calls
   - After: `boundary_indices, normals = geometry.get_boundary_info(points)`

5. ✅ **Remove mixin if only BC logic**
   - If mixin only provides BC methods → delete, use geometry
   - If mixin has other logic → extract non-BC logic to separate class

6. ✅ **Add BC applicator (if needed)**
   - For structured methods (FDM, FEM): Use existing applicators
   - For custom methods: Either use applicator or document why custom

7. ✅ **Document in solver docstring**
   - Note which geometry methods are used
   - Explain BC application strategy

### Example Refactoring: GFDMBoundaryMixin → Composition

**Step 1: Identify what mixin provides**
```python
# GFDMBoundaryMixin provides:
# - _compute_outward_normal() → Use geometry.get_boundary_normal()
# - _apply_local_coordinate_rotation() → Keep (GFDM-specific math)
# - _create_ghost_nodes() → Keep or use applicator
```

**Step 2: Extract GFDM-specific logic**
```python
# Create GFDMStencilHandler (composition, not mixin)
class GFDMStencilHandler:
    """Handles GFDM-specific stencil operations (not boundary detection)."""

    def apply_local_coordinate_rotation(self, point_idx, stencil):
        """Rotate stencil to align with boundary normal."""
        # Use geometry for normal
        point = self.collocation_points[point_idx]
        normal = self.geometry.get_boundary_normal(point.reshape(1, -1))[0]
        # Apply rotation (GFDM-specific math)
        ...
```

**Step 3: Update solver to use composition**
```python
class HJBGFDMSolver:
    def __init__(self, problem, collocation_points, **kwargs):
        # Explicit dependencies
        self.geometry = problem.geometry
        self.collocation_points = collocation_points

        # Composition: specialized handlers
        self.stencil_handler = GFDMStencilHandler(
            geometry=self.geometry,
            collocation_points=collocation_points
        )

        # Use geometry for boundary detection
        self.boundary_indices, self.boundary_normals = \\
            self.geometry.get_boundary_info(collocation_points)
```

## Testing Boundary Helpers

### Unit Tests

```python
def test_get_boundary_indices():
    """Test boundary index detection."""
    geometry = TensorGrid(
        dimension=2,
        bounds=[(0, 1), (0, 1)],
        Nx_points=[11, 11]
    )

    points = geometry.get_collocation_points()
    boundary_indices = geometry.get_boundary_indices(points)

    # Boundary points: 4 edges of 11x11 grid
    # Total: 11 + 11 + 9 + 9 = 40 (corners counted once)
    assert len(boundary_indices) == 40

    # Verify they're actually on boundary
    boundary_points = points[boundary_indices]
    on_boundary = geometry.is_on_boundary(boundary_points)
    assert np.all(on_boundary)

def test_get_boundary_info():
    """Test combined boundary info retrieval."""
    geometry = TensorGrid(
        dimension=2,
        bounds=[(0, 1), (0, 1)],
        Nx_points=[11, 11]
    )

    points = geometry.get_collocation_points()
    boundary_indices, normals = geometry.get_boundary_info(points)

    # Check shapes
    assert len(boundary_indices) == len(normals)
    assert normals.shape[1] == 2  # 2D normals

    # Verify normals are unit vectors
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0)

    # Verify normals point outward (dot product with radial > 0)
    boundary_points = points[boundary_indices]
    center = np.array([0.5, 0.5])
    radial = boundary_points - center
    dot_products = np.sum(normals * radial, axis=1)
    # For rectangular domain, most should point outward
    assert np.sum(dot_products > 0) > len(boundary_indices) * 0.8
```

## Summary

### What Changed (v0.16.17)

**Added to GeometryProtocol**:
- `get_boundary_indices(points, tolerance)` - Returns indices, not boolean mask
- `get_boundary_info(points, tolerance)` - Returns (indices, normals) in one call

**Default Implementation**:
- Available in `Geometry` base class
- All geometries inherit for free
- No need to override unless specialized behavior needed

**Workflow Standardized**:
1. Boundary detection → Use `geometry.get_boundary_indices()`
2. Normal computation → Use `geometry.get_boundary_normal()`
3. Combined → Use `geometry.get_boundary_info()`
4. BC application → Use `BCApplicator` or document custom approach

### Benefits

1. **No Duplicate Detection**: One implementation in geometry
2. **Clear Data Flow**: Explicit method calls, not mixin magic
3. **Easier Testing**: Test geometry methods once
4. **Better Maintainability**: Logic in one place
5. **Composition Over Inheritance**: Explicit dependencies

### Next Steps (Issue #545)

- **Phase 1 (This)**: ✅ Define unified workflow, add helper methods
- **Phase 2**: Refactor FPParticleSolver as composition template
- **Phase 3**: Apply pattern to GFDM, FDM, FEM, DGM solvers

---

**Created**: 2026-01-11
**Issue**: #545 (Mixin Refactoring - Phase 1)
**Related**: #542 (FDM BC Fix), #543 (hasattr Elimination)
**Status**: Active Reference Document
## Update: BoundaryHandler Protocol (2026-01-11)

**Phase 2 Complete**: Issue #545 now 71% complete with BoundaryHandler protocol defined.

### BoundaryHandler Protocol

Common interface for solver boundary condition handling, defined in `mfg_pde/geometry/boundary/handler_protocol.py`.

```python
from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from mfg_pde.geometry.boundary import BoundaryConditions

class BoundaryHandler(Protocol):
    """Common interface for solver boundary condition handling."""

    def get_boundary_indices(self) -> NDArray[np.integer]:
        """Identify boundary points in solver's discretization."""
        ...

    def apply_boundary_conditions(
        self,
        values: NDArray,
        bc: BoundaryConditions,
        time: float = 0.0,
    ) -> NDArray:
        """Apply boundary conditions to solution values."""
        ...

    def get_bc_type_for_point(self, point_idx: int) -> str:
        """Determine BC type (dirichlet/neumann/periodic) for a point."""
        ...
```

### Design Principles

1. **Geometry provides context**: Boundary points, normals, segments
2. **Solver applies enforcement**: Ghost cells, penalties, reflections (method-specific)
3. **Clear separation**: Geometry knows domain, solver knows discretization

### Architecture

```
┌──────────────────────────────────────────┐
│          Geometry Layer                  │
│  • boundary_conditions                   │
│  • get_boundary_indices()                │
│  • get_boundary_normal()                 │
│  • BCSegment.matches_point()             │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│      BoundaryHandler Protocol            │
│  • get_boundary_indices()                │
│  • apply_boundary_conditions()           │
│  • get_bc_type_for_point()               │
└──────────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│          Solver Layer                    │
│  FDM: Ghost cells, one-sided stencils    │
│  GFDM: Ghost nodes, rotation, penalties  │
│  Particle: Reflection, resampling        │
│  FEM: Essential/Natural BCs              │
└──────────────────────────────────────────┘
```

### Unified BC Retrieval Pattern

**Centralized Method** (all solvers should use):

```python
class BaseSolver:
    """Base class for all solvers."""

    def _get_boundary_conditions(self):
        """
        Retrieve BC from geometry or problem.

        Priority order:
        1. geometry.boundary_conditions (attribute)
        2. geometry.get_boundary_conditions() (method)
        3. problem.boundary_conditions (attribute)
        4. problem.get_boundary_conditions() (method)

        Returns:
            BoundaryConditions or None
        """
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
```

**Key Principles**:
- **NO hasattr**: Use try/except instead (Issue #543, #545)
- **Centralized**: One method, used everywhere
- **Clear priority**: Geometry first, then problem
- **Explicit None**: Return None instead of silent fallback

### Example Implementation (FDM Solver)

```python
class HJBFDMSolver(BaseHJBSolver):
    """FDM solver implementing BoundaryHandler protocol."""

    def __init__(self, problem, ...):
        super().__init__(problem)
        # Initialize BC warning flag (NO hasattr)
        self._bc_warning_emitted: bool = False

    def _get_boundary_conditions(self):
        """Centralized BC retrieval (see pattern above)."""
        # Implementation follows priority order
        pass

    def get_boundary_indices(self) -> NDArray[np.integer]:
        """Identify grid boundary points."""
        if self.dimension == 1:
            return np.array([0, self.shape[0] - 1])
        else:
            # Use geometry method for nD
            try:
                return self.problem.geometry.get_boundary_indices()
            except AttributeError:
                return self._detect_boundary_manual()

    def apply_boundary_conditions(self, values, bc, time=0.0):
        """Apply BC using ghost cells (FDM method)."""
        from mfg_pde.geometry.boundary import get_ghost_values_nd

        ghost_values = get_ghost_values_nd(
            values.reshape(self.shape),
            bc,
            self.spacing,
            time=time
        )
        return self._apply_ghost_cells(values, ghost_values)

    def get_bc_type_for_point(self, point_idx: int) -> str:
        """Determine BC type using BCSegment.matches_point()."""
        bc = self._get_boundary_conditions()
        if bc is None:
            return "none"

        point = self._get_point_coordinates(point_idx)
        for segment in bc.segments:
            if segment.matches_point(
                point,
                self.domain_bounds,
                self.dimension
            ):
                return segment.bc_type.value

        return "none"
```

### Migration Checklist

For migrating existing solvers to BoundaryHandler protocol:

- [x] **FDM Solver** (HJBFDMSolver) - hasattr eliminated (Issue #545)
  - [x] Add `_get_boundary_conditions()` method
  - [x] Consolidate 11 hasattr checks into try/except pattern
  - [x] All 40 unit tests passing

- [x] **GFDM Solver** (HJBGFDMSolver) - composition refactored (Issue #545)
  - [x] Extract mixins to 4 components (GridCollocationMapper, MonotonicityEnforcer, BoundaryHandler, NeighborhoodBuilder)
  - [x] Eliminate all hasattr checks
  - [x] 47/49 tests passing (96%)

- [x] **Particle Solver** (FPParticleSolver) - hasattr eliminated (Issue #543)
  - [x] Use try/except instead of hasattr
  - [x] All tests passing

- [ ] **Semi-Lagrangian Solver** - To be audited
- [ ] **FEM Solver** - To be audited  
- [ ] **DGM Solver** - Different paradigm (neural network, low priority)

### Runtime Validation

```python
from mfg_pde.geometry.boundary import validate_boundary_handler

# Check if solver implements protocol
if validate_boundary_handler(solver):
    # Safe to use protocol methods
    boundary_idx = solver.get_boundary_indices()
    u_bc = solver.apply_boundary_conditions(u, bc, time=t)
else:
    # Fallback to solver-specific BC handling
    warnings.warn(f"{solver.__class__.__name__} does not implement BoundaryHandler protocol")
```

### Best Practices

1. **Use try/except instead of hasattr**:
   ```python
   # ❌ BAD
   if hasattr(self.problem, "geometry"):
       bc = self.problem.geometry.boundary_conditions

   # ✅ GOOD
   try:
       bc = self.problem.geometry.boundary_conditions
       if bc is not None:
           return bc
   except AttributeError:
       pass
   ```

2. **Centralize BC retrieval**:
   - Don't duplicate BC retrieval code across methods
   - Use single `_get_boundary_conditions()` method

3. **Leverage geometry methods**:
   - Use `geometry.get_boundary_indices()` instead of custom detection
   - Use `geometry.get_boundary_normal()` instead of manual computation

4. **Clear error messages**:
   - Inform user when BC not found
   - Suggest remediation (attach BC to geometry)

### Testing

```python
def test_solver_implements_protocol(solver):
    """Test that solver implements BoundaryHandler protocol."""
    from mfg_pde.geometry.boundary import validate_boundary_handler

    assert validate_boundary_handler(solver)

def test_get_boundary_indices(solver):
    """Test boundary detection."""
    boundary_idx = solver.get_boundary_indices()
    assert len(boundary_idx) > 0
    assert all(0 <= idx < solver.n_points for idx in boundary_idx)

def test_apply_boundary_conditions(solver, bc):
    """Test BC application."""
    u = np.zeros(solver.n_points)
    u_bc = solver.apply_boundary_conditions(u, bc, time=0.0)
    assert u_bc.shape == u.shape
```

### Status Update

**Issue #545 Progress**: 5/7 criteria complete (71%)

**Completed**:
1. ✅ GFDM refactored using composition pattern
2. ✅ BoundaryHandler protocol defined
3. ✅ Geometry interface provides get_boundary_indices(), get_normals()
4. ✅ FDM hasattr eliminated (17 → 0)
5. ✅ Duplicate BC detection removed

**Remaining**:
1. Apply protocol to remaining solvers (Semi-Lagrangian, etc.)
2. Document solver-specific BC enforcement methods

### References

- **Protocol Definition**: `mfg_pde/geometry/boundary/handler_protocol.py`
- **Issue #545**: Refactor solver BC handling (Mixin Hell → Composition)
- **Issue #542**: GFDM BC fixes (motivation)
- **Issue #543**: hasattr elimination (Particle solver)
- **Commits**:
  - `eecd7ce` - FDM hasattr elimination
  - `265ae91` - BoundaryHandler protocol definition
  - `fb0d355` - Progress update

---

**Last Updated**: 2026-01-11 (Phase 2 Complete)
**Status**: Protocol Defined, FDM/GFDM/Particle Migrated
**Next**: Apply to Semi-Lagrangian, document enforcement methods
