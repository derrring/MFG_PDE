# FPParticleSolver: Composition Pattern Template

**Created**: 2026-01-11
**Issue**: #545 (Mixin Refactoring - Phase 2)
**Purpose**: Document FPParticleSolver as template for composition-based solver design

---

## Overview

FPParticleSolver demonstrates the **composition pattern** for MFG solvers, serving as a clean alternative to deep mixin hierarchies. This document explains the pattern and how to adapt it for other solvers.

## Why Composition Over Mixins?

### Problem: Mixin Hell

Deep mixin hierarchies create:
- **Implicit dependencies**: Methods from one mixin rely on state from another
- **Unclear data flow**: Hard to track where attributes come from
- **Testing complexity**: Mocking requires understanding the entire hierarchy
- **Tight coupling**: Changes ripple across multiple classes

**Example**: GFDM solver has 5-class hierarchy with implicit state sharing.

### Solution: Composition

Explicit dependencies via composition:
- **Clear interfaces**: Each component has a defined protocol
- **Testable**: Components can be tested and mocked independently
- **Flexible**: Easy to swap implementations (e.g., different BC applicators)
- **Maintainable**: Changes localized to specific components

## FPParticleSolver Architecture

### Core Pattern

```python
class FPParticleSolver(BaseFPSolver):
    """Particle-based FP solver using composition."""

    def __init__(self, problem: MFGProblem, ...):
        super().__init__(problem)

        # Explicit composition - inject dependencies
        self._applicator = ParticleApplicator()  # BC application
        self.geometry = problem.geometry          # Domain information
        self.boundary_conditions = ...            # Explicit BC resolution
```

### Key Components

1. **BC Applicator** (`self._applicator: ParticleApplicator`)
   - **Purpose**: Apply boundary conditions to particles
   - **Interface**: Implements BCApplicatorProtocol
   - **Methods**: `apply_boundary_conditions_to_particles()`
   - **Why composition**: Different discretizations need different BC logic

2. **Geometry** (`self.geometry: GeometryProtocol`)
   - **Purpose**: Provide domain information and boundary detection
   - **Interface**: Implements GeometryProtocol
   - **Key methods**:
     - `is_on_boundary(points)` - Detect boundary points
     - `get_boundary_normal(points)` - Get outward normals
     - `get_boundary_info(points)` - Combined detection + normals
     - `project_to_interior(points)` - Clip particles to domain
   - **Why composition**: Solver doesn't own the geometry, just uses it

3. **Boundary Conditions** (`self.boundary_conditions: BoundaryConditions`)
   - **Purpose**: Specify what BCs to apply
   - **Resolution**: Fail-fast hierarchy (see below)
   - **Why explicit**: No silent defaults, clear error messages

## Boundary Condition Resolution (Issue #545)

FPParticleSolver implements a **fail-fast** BC resolution hierarchy:

```python
# Location: mfg_pde/alg/numerical/fp_solvers/fp_particle.py:163-188

# 1. Explicit parameter (highest priority)
if boundary_conditions is not None:
    self.boundary_conditions = boundary_conditions
else:
    # 2. Try geometry BC (use try/except, NOT hasattr - Issue #543)
    try:
        self.boundary_conditions = problem.geometry.get_boundary_conditions()
    except AttributeError as e:
        # 3. FAIL FAST - no silent fallback (CLAUDE.md principle)
        raise ValueError(
            "FPParticleSolver requires explicit boundary conditions. "
            "Boundary conditions not provided via:\n"
            "  1. boundary_conditions=... parameter, OR\n"
            "  2. problem.geometry.get_boundary_conditions()\n\n"
            f"Original error: {e}"
        ) from e

    # Validate we got BCs (geometry method might return None)
    if self.boundary_conditions is None:
        raise ValueError(
            "FPParticleSolver requires boundary conditions. "
            "problem.geometry.get_boundary_conditions() returned None."
        )
```

### Design Principles

1. **No `hasattr()` checks** (Issue #543)
   - Use try/except AttributeError instead
   - Preserves protocol typing
   - Better error messages

2. **No silent fallbacks** (CLAUDE.md: Fail Fast)
   - Don't hide missing configuration
   - Surface problems during development
   - Explicit > Implicit

3. **Clear error messages**
   - Tell user exactly what's missing
   - Show how to fix it
   - Chain original error for debugging

## Using Geometry Methods

FPParticleSolver delegates boundary operations to geometry:

### Boundary Detection

```python
# OLD WAY (custom detection in solver)
on_boundary_mask = (
    (particles[:, 0] < bounds[0][0] + tol) |
    (particles[:, 0] > bounds[0][1] - tol) |
    (particles[:, 1] < bounds[1][0] + tol) |
    (particles[:, 1] > bounds[1][1] - tol)
)
boundary_indices = np.where(on_boundary_mask)[0]

# NEW WAY (use geometry)
boundary_indices = self.geometry.get_boundary_indices(particles)
```

### Boundary Normals

```python
# OLD WAY (custom normal computation)
def _compute_outward_normal(self, point):
    # Custom logic based on solver knowledge
    if abs(point[0] - self.xmin) < tol:
        return np.array([-1.0, 0.0])
    elif abs(point[0] - self.xmax) < tol:
        return np.array([1.0, 0.0])
    # ... more cases

# NEW WAY (use geometry)
normals = self.geometry.get_boundary_normal(boundary_points)
```

### Combined Operations (Efficient)

```python
# Get indices AND normals in one call (avoids duplicate detection)
boundary_indices, normals = self.geometry.get_boundary_info(particles)
```

See `docs/development/BOUNDARY_HANDLING.md` for complete workflow documentation.

## When to Use Geometry vs Custom Logic

### Use Geometry Methods When:

✅ **Boundary detection** - Geometry knows its boundary
✅ **Normals** - Geometry computes accurate normals
✅ **Projection** - Clipping to domain, boundary snap
✅ **Bounds** - Domain extent, bounding boxes

### Use Custom Solver Logic When:

✅ **BC application** - Solver-specific (reflection for particles, ghost values for FDM)
✅ **Time stepping** - Solver's responsibility
✅ **Numerical schemes** - Solver implementation details
✅ **Particle-specific operations** - Resampling, KDE, SDE integration

**Principle**: Geometry provides **geometric information**, solver implements **numerical algorithms**.

## Adapting Pattern for Other Solvers

### Step 1: Identify Components

Extract mixin functionality into composable components:

```python
# BEFORE (FDM with mixins)
class FPFDMSolver(BaseFPSolver, BoundaryMixin, GhostValueMixin):
    pass

# AFTER (FDM with composition)
class FPFDMSolver(BaseFPSolver):
    def __init__(self, problem, ...):
        super().__init__(problem)
        self._applicator = FDMApplicator()     # BC via ghost values
        self.geometry = problem.geometry       # Domain info
        self.boundary_conditions = ...         # Explicit resolution
```

### Step 2: Define Interfaces

Create protocols for each component:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class BCApplicatorProtocol(Protocol):
    """Interface for BC application strategies."""

    def apply_boundary_conditions(
        self,
        values: NDArray,
        boundary_conditions: BoundaryConditions,
        geometry: GeometryProtocol,
    ) -> NDArray:
        """Apply BCs to solution values."""
        ...
```

### Step 3: Implement BC Resolution

Use the fail-fast pattern:

```python
if boundary_conditions is not None:
    self.boundary_conditions = boundary_conditions
else:
    try:
        self.boundary_conditions = problem.geometry.get_boundary_conditions()
    except AttributeError as e:
        raise ValueError(
            f"{self.__class__.__name__} requires explicit boundary conditions. "
            "Boundary conditions not provided via:\n"
            "  1. boundary_conditions=... parameter, OR\n"
            "  2. problem.geometry.get_boundary_conditions()\n\n"
            f"Original error: {e}"
        ) from e

    if self.boundary_conditions is None:
        raise ValueError(
            f"{self.__class__.__name__} requires boundary conditions. "
            "problem.geometry.get_boundary_conditions() returned None."
        )
```

### Step 4: Delegate to Components

Replace mixin methods with component calls:

```python
# BEFORE (mixin method)
def _apply_boundary(self, u):
    # Complex logic in mixin
    return self._boundary_mixin_apply(u)

# AFTER (delegation)
def _apply_boundary(self, u):
    # Clear delegation
    return self._applicator.apply_boundary_conditions(
        u, self.boundary_conditions, self.geometry
    )
```

### Step 5: Document Dependencies

Update class docstring to document composition:

```python
class MySolver(BaseSolver):
    """
    My solver using composition pattern.

    Composition Pattern:
        This solver uses composition instead of mixins:
        - self._applicator: BCApplicator for BC application
        - self.geometry: GeometryProtocol for domain information
        - Explicit dependencies, no implicit state sharing

    Boundary Conditions:
        Requires explicit boundary conditions via:
        1. boundary_conditions parameter, OR
        2. problem.geometry.get_boundary_conditions()

        No default fallback - fails fast with clear error.
    """
```

## Testing Composition

Composition enables better testing through dependency injection:

### Component Testing

```python
def test_bc_applicator_independent():
    """Test BC applicator without full solver."""
    applicator = ParticleApplicator()
    geometry = TensorProductGrid(...)
    bc = dirichlet_bc(dimension=2, value=0.0)

    result = applicator.apply_boundary_conditions(particles, bc, geometry)
    assert ...  # Test just the BC logic
```

### Solver Testing with Mocks

```python
def test_solver_with_mock_applicator():
    """Test solver logic with mocked BC applicator."""
    from unittest.mock import Mock

    problem = create_test_problem()
    solver = MySolver(problem)

    # Inject mock applicator
    mock_applicator = Mock()
    solver._applicator = mock_applicator

    solver.solve()

    # Verify applicator was called correctly
    mock_applicator.apply_boundary_conditions.assert_called_once()
```

### BC Resolution Testing

```python
def test_solver_requires_boundary_conditions():
    """Test fail-fast BC requirement."""
    from unittest.mock import Mock

    # Mock problem without geometry.get_boundary_conditions()
    mock_geometry = Mock(spec=[])  # No methods
    mock_problem = Mock()
    mock_problem.geometry = mock_geometry

    # Should fail fast with clear error
    with pytest.raises(ValueError, match="requires explicit boundary conditions"):
        MySolver(mock_problem)
```

See `tests/unit/test_fp_particle.py::TestBoundaryConditionRequirements` for complete examples.

## Migration Checklist

When refactoring a mixin-based solver to composition:

- [ ] Identify mixin responsibilities (BC application, geometry queries, etc.)
- [ ] Create protocol interfaces for each component
- [ ] Implement concrete components (e.g., FDMApplicator, ParticleApplicator)
- [ ] Replace mixin inheritance with composition
- [ ] Implement fail-fast BC resolution
- [ ] Replace `hasattr()` with try/except AttributeError
- [ ] Delegate geometry queries to `self.geometry`
- [ ] Update class docstring documenting composition
- [ ] Write component tests
- [ ] Write integration tests
- [ ] Verify all existing tests still pass
- [ ] Document pattern in solver docstring

## Benefits Achieved

After refactoring to composition:

1. **Testability**: Components can be tested and mocked independently
2. **Clarity**: Explicit dependencies, clear data flow
3. **Flexibility**: Easy to swap implementations (e.g., different BC strategies)
4. **Maintainability**: Changes localized to specific components
5. **Type Safety**: Protocols enable static type checking
6. **Fail-Fast**: Missing configuration surfaces immediately with clear errors

## Related Documentation

- **Boundary Handling Workflow**: `docs/development/BOUNDARY_HANDLING.md`
- **Issue #545**: Mixin refactoring initiative
- **Issue #543**: hasattr elimination
- **CLAUDE.md**: Fail-fast principle

## Example: Complete Solver Structure

```python
from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfg_pde.geometry.boundary.applicator_particle import ParticleApplicator

class FPParticleSolver(BaseFPSolver):
    """
    Particle-based Fokker-Planck solver using composition.

    Composition Pattern (Issue #545):
        - self._applicator = ParticleApplicator() for BC application
        - self.geometry = problem.geometry for domain information
        - Explicit dependencies, no implicit state sharing
    """

    def __init__(self, problem, num_particles=5000, boundary_conditions=None):
        super().__init__(problem)

        # Component 1: BC Applicator (composition)
        self._applicator = ParticleApplicator()

        # Component 2: Geometry (delegation)
        self.geometry = problem.geometry

        # Component 3: Boundary Conditions (fail-fast resolution)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            try:
                self.boundary_conditions = problem.geometry.get_boundary_conditions()
            except AttributeError as e:
                raise ValueError(
                    "FPParticleSolver requires explicit boundary conditions. "
                    "Provide via boundary_conditions parameter or geometry."
                ) from e

            if self.boundary_conditions is None:
                raise ValueError(
                    "FPParticleSolver requires boundary conditions. "
                    "geometry.get_boundary_conditions() returned None."
                )

        # Solver-specific state
        self.num_particles = num_particles
        self.particles = None  # Initialize in solve()

    def solve_fp_system(self, m0, U, show_progress=True):
        """Solve FP using particle method."""
        # Use geometry for boundary detection
        boundary_indices, normals = self.geometry.get_boundary_info(self.particles)

        # Use applicator for BC application
        self.particles = self._applicator.apply_boundary_conditions_to_particles(
            self.particles,
            self.boundary_conditions,
            self.geometry,
        )

        # Solver-specific logic
        # ...
```

---

**Template Status**: ✅ Production-ready composition pattern
**Tested**: See `tests/unit/test_fp_particle.py`
**Reference Implementation**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`
