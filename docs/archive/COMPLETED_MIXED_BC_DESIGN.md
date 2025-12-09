# Mixed Boundary Conditions - Design Document

**Status**: Planning Phase
**Target**: v0.13.0
**Author**: MFG_PDE Development Team
**Created**: 2025-11-23

## Executive Summary

MFG_PDE currently only supports **uniform** boundary conditions (periodic, Dirichlet, Neumann) across the entire domain boundary. This document proposes a design for **mixed boundary conditions**, where different BC types apply to different boundary segments.

**Critical Use Case**: 2D crowd motion with exit (Dirichlet `u=0`) and reflective walls (Neumann `∂u/∂n=0`).

---

## 1. Current Architecture

### 1.1 Existing BC Structure

**File**: `mfg_pde/geometry/boundary/bc_1d.py`

```python
@dataclass
class BoundaryConditions:
    type: str  # 'periodic', 'dirichlet', 'neumann', 'no_flux', 'robin'
    left_value: float | None = None
    right_value: float | None = None
    # Robin parameters...
```

**Limitations**:
- ✗ Single `type` for entire boundary
- ✗ Only supports 1D (left/right)
- ✗ No segment-wise specification
- ✓ Robin BC exists but not mixed with other types

### 1.2 Solver BC Enforcement

**HJB FDM Solver** (`hjb_fdm.py`):
- BC enforcement in `_solve_single_timestep()` via linear solve
- Assumes uniform BC type
- No mixed BC handling

**FP FDM Solver** (`fp_fdm.py`):
- BC enforcement in diffusion operator
- Uses `self.boundary_conditions.type` uniformly

---

## 2. Proposed Design

### 2.1 Core Principle

**Segment-Based BC Specification**: Divide boundary into named segments, each with its own BC type and parameters.

### 2.2 New BC Structure

**File**: `mfg_pde/geometry/boundary/mixed_bc.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable

class BCType(Enum):
    """Boundary condition types."""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    NO_FLUX = "no_flux"

@dataclass
class BCSegment:
    """
    Boundary condition for a single segment.

    Attributes:
        name: Segment identifier (e.g., "exit", "left_wall", "top_wall")
        bc_type: Type of boundary condition
        value: Boundary value (Dirichlet: u value, Neumann: ∂u/∂n value)
        region: Spatial region specification
    """
    name: str
    bc_type: BCType
    value: float | Callable = 0.0  # Can be function for inhomogeneous BC

    # Region specification (dimension-dependent)
    region: dict | None = None  # e.g., {"x": (9.9, 10.0), "y": (4.25, 5.75)}

    # Robin parameters
    alpha: float | None = None
    beta: float | None = None

@dataclass
class MixedBoundaryConditions:
    """
    Mixed boundary conditions with multiple segments.

    Examples:
        # 2D room with exit
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            region={"boundary": "right", "y_range": (4.25, 5.75)}
        )
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            region={"boundary": "all_except", "exclude": ["exit"]}
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            default_type=BCType.NEUMANN,
        )
    """
    dimension: int
    segments: list[BCSegment]
    default_type: BCType = BCType.PERIODIC  # Fallback for unspecified regions

    def get_bc_at_point(self, point: tuple, boundary_id: str) -> BCSegment:
        """
        Get BC for a specific boundary point.

        Args:
            point: Spatial coordinates (x, y) or (x, y, z)
            boundary_id: Which boundary ("left", "right", "top", "bottom", etc.)

        Returns:
            BCSegment applicable to this point
        """
        # Check each segment in priority order
        for segment in self.segments:
            if self._point_in_region(point, boundary_id, segment.region):
                return segment

        # Return default if no match
        return BCSegment(
            name="default",
            bc_type=self.default_type,
            value=0.0,
        )

    def _point_in_region(self, point: tuple, boundary_id: str, region: dict) -> bool:
        """Check if point belongs to region."""
        if region is None:
            return False

        # Check boundary match
        if "boundary" in region:
            if region["boundary"] == "all_except":
                if boundary_id in region.get("exclude", []):
                    return False
            elif region["boundary"] != boundary_id and region["boundary"] != "all":
                return False

        # Check spatial ranges
        if self.dimension == 2:
            x, y = point
            if "x_range" in region:
                x_min, x_max = region["x_range"]
                if not (x_min <= x <= x_max):
                    return False
            if "y_range" in region:
                y_min, y_max = region["y_range"]
                if not (y_min <= y <= y_max):
                    return False

        return True

    def is_mixed(self) -> bool:
        """Check if BC is truly mixed (multiple types)."""
        types = {seg.bc_type for seg in self.segments}
        return len(types) > 1
```

### 2.3 Integration with Existing `BoundaryConditions`

**Backward Compatibility Wrapper**:

```python
# In mfg_pde/geometry/__init__.py
from .boundary.bc_1d import BoundaryConditions  # Legacy 1D
from .boundary.mixed_bc import MixedBoundaryConditions, BCSegment, BCType

# Automatic detection
def create_boundary_conditions(
    dimension: int = 1,
    bc_type: str | None = None,
    segments: list[BCSegment] | None = None,
    **kwargs
) -> BoundaryConditions | MixedBoundaryConditions:
    """
    Factory function for creating boundary conditions.

    Returns:
        - BoundaryConditions if dimension=1 and uniform
        - MixedBoundaryConditions if segments specified or dimension>1
    """
    if segments is not None:
        # Mixed BC
        return MixedBoundaryConditions(
            dimension=dimension,
            segments=segments,
            default_type=BCType(bc_type) if bc_type else BCType.PERIODIC,
        )
    elif dimension == 1:
        # Legacy 1D uniform BC
        return BoundaryConditions(type=bc_type or "periodic", **kwargs)
    else:
        # nD uniform BC - wrap as single segment
        segment = BCSegment(
            name="all_boundaries",
            bc_type=BCType(bc_type) if bc_type else BCType.PERIODIC,
            value=kwargs.get("value", 0.0),
            region={"boundary": "all"},
        )
        return MixedBoundaryConditions(
            dimension=dimension,
            segments=[segment],
            default_type=BCType(bc_type) if bc_type else BCType.PERIODIC,
        )
```

---

## 3. Solver Implementation

### 3.1 HJB FDM Solver Modifications

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`

**Required Changes**:

1. **Detect Mixed BC** in `__init__`:
   ```python
   def __init__(self, problem: MFGProblem):
       super().__init__(problem)
       self.bc = problem.get_boundary_conditions()
       self.is_mixed_bc = isinstance(self.bc, MixedBoundaryConditions)
   ```

2. **BC Enforcement Hook** in `_solve_single_timestep()`:
   ```python
   def _enforce_boundary_conditions(self, U: np.ndarray):
       """Apply BC to solution array."""
       if self.is_mixed_bc:
           return self._enforce_mixed_bc(U)
       else:
           return self._enforce_uniform_bc(U)

   def _enforce_mixed_bc(self, U: np.ndarray):
       """Apply mixed BC segment by segment."""
       if self.dimension == 2:
           Nx, Ny = U.shape

           # Iterate over boundary points
           for i in [0, Nx-1]:  # Left/right walls
               for j in range(Ny):
                   point = (i * self.dx, j * self.dy)
                   boundary_id = "left" if i == 0 else "right"
                   bc_seg = self.bc.get_bc_at_point(point, boundary_id)

                   if bc_seg.bc_type == BCType.DIRICHLET:
                       U[i, j] = bc_seg.value
                   elif bc_seg.bc_type == BCType.NEUMANN:
                       # Ghost cell reflection
                       if i == 0:
                           U[i, j] = U[i+1, j]  # ∂u/∂x = 0
                       else:
                           U[i, j] = U[i-1, j]

           # Similar for top/bottom walls...

       return U
   ```

### 3.2 FP FDM Solver Modifications

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

**Similar pattern**: BC enforcement in diffusion operator construction.

---

## 4. Usage Examples

### 4.1 Protocol v1.4: 2D Crowd Motion

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import MixedBoundaryConditions, BCSegment, BCType

# Define boundary segments
exit_bc = BCSegment(
    name="exit",
    bc_type=BCType.DIRICHLET,
    value=0.0,
    region={
        "boundary": "right",
        "y_range": (4.25, 5.75),  # Exit segment
    }
)

wall_bc = BCSegment(
    name="walls",
    bc_type=BCType.NEUMANN,
    value=0.0,  # ∂u/∂n = 0
    region={
        "boundary": "all_except",
        "exclude": ["exit"],  # All boundaries except exit
    }
)

# Create mixed BC
mixed_bc = MixedBoundaryConditions(
    dimension=2,
    segments=[exit_bc, wall_bc],
    default_type=BCType.NEUMANN,  # Fallback
)

# Create problem with mixed BC
problem = MFGProblem(
    spatial_bounds=[(0.0, 10.0), (0.0, 10.0)],
    spatial_discretization=[59, 59],
    T=2.0,
    Nt=40,
    sigma=0.5,
    # Mixed BC specification
    components=MFGComponents(
        boundary_conditions=mixed_bc,
        # ... other components
    )
)
```

### 4.2 Robin BC on Part of Boundary

```python
# Heat flux on part of boundary, insulated elsewhere
flux_bc = BCSegment(
    name="heat_source",
    bc_type=BCType.ROBIN,
    value=100.0,  # g in αu + β∂u/∂n = g
    alpha=1.0,
    beta=0.1,
    region={"boundary": "right", "y_range": (0.4, 0.6)}
)

insulated_bc = BCSegment(
    name="insulated",
    bc_type=BCType.NEUMANN,
    value=0.0,
    region={"boundary": "all_except", "exclude": ["heat_source"]}
)
```

---

## 5. Implementation Plan

### 5.1 Phase 1: Core Infrastructure (Week 1)

**Tasks**:
1. ✅ Create `mfg_pde/geometry/boundary/mixed_bc.py`
2. ✅ Implement `BCSegment`, `BCType`, `MixedBoundaryConditions`
3. ✅ Add factory function `create_boundary_conditions()`
4. ✅ Write unit tests for region matching logic

**Deliverable**: Mixed BC data structures ready

### 5.2 Phase 2: HJB Solver Integration (Week 2)

**Tasks**:
1. Modify `HJBFDMSolver._enforce_boundary_conditions()`
2. Implement `_enforce_mixed_bc()` for 2D
3. Add ghost cell reflection for Neumann segments
4. Test with Protocol v1.4 2D crowd problem

**Deliverable**: HJB FDM solver supports mixed BC in 2D

### 5.3 Phase 3: FP Solver Integration (Week 3)

**Tasks**:
1. Modify `FPFDMSolver` diffusion operator
2. Handle mixed BC in mass conservation
3. Test Fokker-Planck with mixed BC

**Deliverable**: FP FDM solver supports mixed BC

### 5.4 Phase 4: Extended Support (Week 4)

**Tasks**:
1. Extend to 3D mixed BC
2. Add time-dependent BC: `value=lambda t: ...`
3. Implement for Semi-Lagrangian solver
4. Comprehensive documentation and examples

**Deliverable**: Full mixed BC support across solvers

---

## 6. Testing Strategy

### 6.1 Unit Tests

**File**: `tests/unit/geometry/test_mixed_bc.py`

```python
def test_mixed_bc_region_matching_2d():
    """Test that points are correctly assigned to BC segments."""
    exit_bc = BCSegment(
        name="exit",
        bc_type=BCType.DIRICHLET,
        value=0.0,
        region={"boundary": "right", "y_range": (4.25, 5.75)}
    )
    wall_bc = BCSegment(
        name="walls",
        bc_type=BCType.NEUMANN,
        value=0.0,
        region={"boundary": "all_except", "exclude": ["exit"]}
    )

    mixed_bc = MixedBoundaryConditions(dimension=2, segments=[exit_bc, wall_bc])

    # Point on exit
    bc1 = mixed_bc.get_bc_at_point((10.0, 5.0), "right")
    assert bc1.bc_type == BCType.DIRICHLET
    assert bc1.name == "exit"

    # Point on wall (not exit)
    bc2 = mixed_bc.get_bc_at_point((10.0, 2.0), "right")
    assert bc2.bc_type == BCType.NEUMANN
    assert bc2.name == "walls"
```

### 6.2 Integration Tests

**File**: `tests/integration/test_mixed_bc_hjb.py`

Test Protocol v1.4 2D crowd motion with actual solver run.

### 6.3 Validation Tests

Compare mixed BC solution against known analytical solutions (if available) or benchmark results.

---

## 7. Documentation

### 7.1 User Guide

**File**: `docs/user/mixed_boundary_conditions.md`

- Concept explanation
- API reference
- Examples for common scenarios
- Migration guide from uniform BC

### 7.2 Developer Guide

**File**: `docs/development/MIXED_BC_IMPLEMENTATION.md`

- Architecture details
- Solver integration pattern
- How to extend to new solvers

---

## 8. Open Questions

1. **Corner Treatment**: How to handle points at corners where two different BC types meet?
   - **Proposal**: Priority order in `segments` list (first match wins)
   - **Alternative**: Average BC values at corners

2. **Performance**: Segment lookup per boundary point might be slow
   - **Optimization**: Pre-compute BC masks during initialization

3. **3D Complexity**: Face/edge/corner BC in 3D requires careful design
   - **Defer to Phase 4** after 2D is stable

---

## 9. Success Criteria

✅ **Minimum Viable Product (MVP)**:
- 2D mixed Dirichlet + Neumann BC works in HJB FDM solver
- Protocol v1.4 crowd motion problem solves correctly
- BC validation shows exit BC: `|u| < 1e-8`, wall BC: `|∂u/∂n| < 1e-4`

✅ **Full Success**:
- Mixed BC works in HJB and FP solvers
- Supports 2D and 3D
- Comprehensive test coverage
- User documentation complete

---

## 10. References

- Protocol v1.4: `PROTOCOL_2D_CROWD_HJB.md`
- Current BC implementation: `mfg_pde/geometry/boundary/bc_1d.py`
- HJB FDM solver: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`

---

**Next Action**: Create GitHub issue for Phase 1 implementation and assign milestone.
